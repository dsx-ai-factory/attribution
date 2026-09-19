# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

from attribution_blueprint.connectors.attribution import (
    AttributionConnector,
    AttributorInfoV2,
    ServiceObservation,
)
from attribution_blueprint.connectors.execution_contexts import ExecutionContextAdapter
from attribution_blueprint.connectors.logs import LogConnector
from attribution_blueprint.connectors.logs.service_observation_adapter import adapt_log_observation_for_service
from attribution_blueprint.connectors.metrics import MetricsConnector
from attribution_blueprint.connectors.outputs import OutputConnector
from attribution_blueprint.result import ConnectorReport, enrich_attribution_result, service_outcome
from attribution_blueprint.taxonomy import taxonomy_for_service_value


def _service_timezone_name(start: datetime, end: datetime, configured: str | None = None) -> str:
    """Return the IANA timezone key required by Attribution Service v2.

    Args:
        start: Timezone-aware workload start timestamp.
        end: Timezone-aware workload end timestamp.
        configured: Optional explicit IANA timezone key.

    Returns:
        A validated IANA timezone key.

    Raises:
        ValueError: If fixed offsets lose regional/DST identity or cannot be
            represented by an IANA timezone key.
    """
    if configured is not None and configured != "":
        try:
            return ZoneInfo(configured).key
        except (TypeError, ZoneInfoNotFoundError) as exc:
            raise ValueError(f"ctx['timezone'] must be a valid IANA timezone key, got {configured!r}") from exc

    tzinfo = start.tzinfo
    if isinstance(tzinfo, ZoneInfo):
        return tzinfo.key

    start_offset = start.utcoffset()
    end_offset = end.utcoffset()
    if start_offset is None or end_offset is None:
        raise ValueError("execution context timestamps must have a defined UTC offset")
    if start_offset != end_offset:
        raise ValueError(
            "execution context spans a UTC offset change without an IANA timezone; "
            "set [eca.slurm] cluster_tz or ctx['timezone']"
        )

    total_seconds = int(start_offset.total_seconds())
    if total_seconds == 0:
        return "UTC"
    hours, remainder = divmod(abs(total_seconds), 3600)
    if remainder != 0:
        raise ValueError(
            f"UTC offset {start_offset} cannot be represented by an IANA fixed-offset zone; "
            "set ctx['timezone'] to the workload's IANA timezone key"
        )
    # POSIX Etc/GMT signs are inverted: UTC-7 is Etc/GMT+7.
    sign = "-" if total_seconds > 0 else "+"
    candidate = f"Etc/GMT{sign}{hours}"
    try:
        ZoneInfo(candidate)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"UTC offset {start_offset} has no IANA fixed-offset zone; "
            "set ctx['timezone'] to the workload's IANA timezone key"
        ) from exc
    return candidate


class AttributionOrchestrator:
    """Orchestrates discovery -> evidence collection -> service submission -> output."""

    def __init__(
        self,
        execution_context_adapters: list[ExecutionContextAdapter],
        log_connectors: list[LogConnector],
        metrics_connectors: list[MetricsConnector],
        attribution: AttributionConnector,
        outputs: list[OutputConnector],
        dump_observations_dir: Path | None = None,
        gpus_per_node: int | None = None,
    ):
        self.execution_context_adapters = execution_context_adapters
        self.log_connectors = log_connectors
        self.metrics_connectors = metrics_connectors
        self.attribution = attribution
        self.outputs = outputs
        self.dump_observations_dir = dump_observations_dir
        self.gpus_per_node = gpus_per_node
        # Per-run counters surfaced via `last_run_stats` so callers can decide
        # whether to exit non-zero. `succeeded` counts contexts whose verdict
        # reached at least one output sink.
        self.last_run_stats: dict[str, int] = {"attempted": 0, "succeeded": 0}
        self.last_run_summaries: list[dict[str, str]] = []

    def analyze(
        self,
        *,
        ids: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        nodelist: list[str] | None = None,
    ) -> list[dict]:
        contexts = []
        for eca in self.execution_context_adapters:
            if ids is not None:
                contexts.extend(eca.discover_by_ids(ids))
            elif since is None and until is None and nodelist is None:
                contexts.extend(eca.discover())
            else:
                contexts.extend(eca.discover_by_time_range(since, until, nodelist))

        collectors = [*self.log_connectors, *self.metrics_connectors]
        results: list[dict] = []

        # Each context is processed in a single try/except so any failure —
        # including in attributor creation or info building — keeps the batch
        # moving.
        self.last_run_stats = {"attempted": len(contexts), "succeeded": 0}
        self.last_run_summaries = []
        try:
            for ctx in contexts:
                try:
                    processed = self._process_one(ctx, collectors)
                except Exception as e:
                    logger.warning(f"processing failed for ctx {ctx.get('id', '?')}: {e}")
                    continue
                if processed is None:
                    continue
                result, output_succeeded = processed
                results.append(result)
                self.last_run_summaries.append(_summarize_result(ctx, result))
                if output_succeeded:
                    self.last_run_stats["succeeded"] += 1
        finally:
            self._close_outputs()

        return results

    def _process_one(self, ctx: dict, collectors: list) -> tuple[dict, bool] | None:
        observations: list[dict] = []
        connector_reports: list[ConnectorReport] = []
        for collector in collectors:
            try:
                collected = collector.collect(ctx)
            except Exception as e:
                logger.warning(f"collector {getattr(collector, 'source', '?')} failed for ctx {ctx['id']}: {e}")
                connector_reports.append(_connector_report(collector, "error", str(e), 0))
                continue
            observations.extend(collected)
            connector_reports.append(_connector_report(collector, "ok", "", len(collected)))

        if self.dump_observations_dir is not None:
            self._dump_observations(ctx, observations)

        info = self._build_attributor_info(ctx)
        attributor_id = self.attribution.create_or_replace_attributor(info)

        for obs in observations:
            try:
                service_obs = ServiceObservation.model_validate(
                    adapt_log_observation_for_service(obs, ctx.get("timezone"))
                )
                self.attribution.ingest_observation(attributor_id, service_obs)
            except Exception as e:
                obs_source = obs.get("context", {}).get("source", "?")
                obs_source_value = obs_source.value if hasattr(obs_source, "value") else obs_source
                logger.warning(f"ingest_observation failed for ctx {ctx['id']} source={obs_source_value}: {e}")
                continue

        # Get the verdict, then ALWAYS clean up via a finally so we don't
        # leave attributors behind on the service even when get_attribution
        # raises. (Real-world trigger: attribution_lib's checkpoint path
        # raises 500 when the cached payload contains a naive datetime.)
        result: dict | None = None
        try:
            result = self.attribution.get_attribution(attributor_id)
        except Exception as e:
            logger.warning(f"get_attribution failed for ctx {ctx['id']}: {e}")
        finally:
            try:
                self.attribution.delete_attributor(attributor_id)
            except Exception as e:
                logger.warning(f"delete_attributor failed for {attributor_id}: {e}")

        if result is None:
            return None

        result = enrich_attribution_result(result, connector_reports)

        # Per-sink try/except: a broken sqlite write must not skip the
        # other sinks for the same ctx.
        output_succeeded = False
        output_ctx = _context_with_workload_gpu_defaults(ctx, self.gpus_per_node)
        for sink in self.outputs:
            try:
                sink.write(output_ctx, result)
            except Exception as e:
                logger.warning(f"{type(sink).__name__}.write failed for ctx {ctx['id']}: {e}")
                continue
            output_succeeded = True
        return result, output_succeeded

    def _close_outputs(self) -> None:
        for sink in self.outputs:
            try:
                sink.close()
            except Exception as e:
                logger.warning(f"{type(sink).__name__}.close failed: {e}")

    def _dump_observations(self, ctx: dict, observations: list[dict]) -> None:
        """Write each observation body to disk as `<ctx_id>.<source>.json` for review."""
        self.dump_observations_dir.mkdir(parents=True, exist_ok=True)
        for obs in observations:
            source = obs["context"]["source"]
            source_value = source.value if hasattr(source, "value") else source
            path = self.dump_observations_dir / f"{ctx['id']}.{source_value}.json"
            # Most connector bodies are JSON-encoded (ENTRY_TIMESERIES,
            # LOKI_DICT_LIST, PROMETHEUS_METRICS). local_file syslog emits
            # `Format.RAW_LOG` whose body is the raw file text — not JSON.
            # Try to decode; if it isn't JSON, preserve the raw string under a
            # `body_raw` key so the dump round-trips for any (Source, Format)
            # pair without crashing the per-context loop.
            raw_body = obs["body"]
            body_field: dict[str, object]
            try:
                body_field = {"body": json.loads(raw_body)}
            except (TypeError, ValueError):
                body_field = {"body_raw": raw_body if isinstance(raw_body, str) else str(raw_body)}
            payload = {
                "context": {
                    "source": source_value,
                    "format": (
                        obs["context"]["format"].value
                        if hasattr(obs["context"]["format"], "value")
                        else obs["context"]["format"]
                    ),
                    "time_interval": {
                        "start": _isoformat_or_none(obs["context"]["time_interval"]["start"]),
                        "end": _isoformat_or_none(obs["context"]["time_interval"]["end"]),
                    },
                    "resources": obs["context"]["resources"],
                },
                **body_field,
            }
            path.write_text(json.dumps(payload, indent=2))

    def _build_attributor_info(self, ctx: dict) -> AttributorInfoV2:
        # Service-side AttributorWorkloadInfo.job_start_time is AwareDatetime;
        # a naive datetime here would 422 the request, so fail fast at the BP
        # boundary rather than turning a programming error into an HTTP error.
        for field in ("start_time", "end_time"):
            value = ctx.get(field)
            if isinstance(value, datetime) and value.tzinfo is None:
                raise ValueError(
                    f"ctx[{field!r}] must be tz-aware (got naive datetime); "
                    f"the service requires AwareDatetime for job_{field}"
                )
        # Service-side verify_node() lowercases the input but compares against
        # workload.nodes verbatim, so any node we submit must already be
        # lowercase or every syslog entry gets dropped as "unknown node".
        nodes = [n.lower() for n in (ctx.get("nodes") or [])]
        # Blueprint config calls this GPUs per node. In the supported training
        # model, that is the same as ranks per node because each rank maps to
        # one GPU. Attribution-service v2 transports this as explicit
        # node-to-ranks entries.
        ranks_per_node = 4 if self.gpus_per_node is None else self.gpus_per_node
        if ranks_per_node <= 0:
            raise ValueError(f"gpus_per_node must be positive, got {ranks_per_node}")
        node_to_ranks = {
            node: list(range(index * ranks_per_node, (index + 1) * ranks_per_node)) for index, node in enumerate(nodes)
        }
        # srun_cmd is intentionally omitted so the service's default ("dummy")
        # applies. Sending "" makes JobInfo.srun_cmd None and trips the
        # "no symptoms, no srun" fallback even when symptoms exist.
        workload: dict = {
            "id": ctx["id"],
            "type": ctx.get("workload_type", ""),
            "status": ctx.get("state", "UNKNOWN"),
            "framework": ctx.get("workload_framework", ""),
            "name": ctx.get("name", ""),
            "username": ctx.get("user_name", ""),
            "job_start_time": ctx["start_time"].isoformat(),
            "timezone": _service_timezone_name(ctx["start_time"], ctx["end_time"], ctx.get("timezone")),
            "job_end_time": ctx["end_time"].isoformat(),
            "interest_start_time": ctx["start_time"].isoformat(),
            "interest_end_time": ctx["end_time"].isoformat(),
            "nodes": node_to_ranks,
        }
        if "failed_node" in ctx:
            workload["failed_node"] = ctx["failed_node"].lower()
        if "cancelled_by" in ctx:
            workload["cancelled_by"] = ctx["cancelled_by"]
        if "time_limit" in ctx:
            workload["time_limit"] = ctx["time_limit"]

        return AttributorInfoV2.model_validate(
            {
                "workload": workload,
                "metadata": {
                    # The wire format requires `cluster`; the blueprint doesn't
                    # track it as a concept (per the data-model design), so we
                    # send an empty string just to satisfy the schema. Any
                    # cluster-aware analytics live downstream of attribution.
                    "cluster": "",
                    "agent": "attribution-blueprint",
                    "tenant": "",
                    "ruleset": "",
                },
                # `parameters` is intentionally omitted. The service defaults it,
                # and its AttributionParameters model forbids unknown fields, so
                # the legacy straggler-parameter keys would 422 every request.
            }
        )


def _isoformat_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _context_with_workload_gpu_defaults(ctx: dict, gpus_per_node: int | None) -> dict:
    """Add workload GPU metadata to the output context when available.

    Args:
        ctx: Execution context discovered by a scheduler adapter.
        gpus_per_node: Effective workload GPUs-per-node value from config.

    Returns:
        A shallow context copy for output sinks. Existing scheduler-provided
        values are preserved.
    """
    output_ctx = dict(ctx)
    if gpus_per_node is None:
        return output_ctx
    if "gpus_per_node" not in output_ctx:
        output_ctx["gpus_per_node"] = gpus_per_node
    if "gpu_count" not in output_ctx:
        nodes = output_ctx.get("nodes")
        node_count = len(nodes) if isinstance(nodes, list) else 0
        billed_nodes = node_count if node_count > 0 else 1
        output_ctx["gpu_count"] = billed_nodes * gpus_per_node
    return output_ctx


def _connector_report(
    collector: object,
    status: str,
    reason: str,
    observation_count: int,
) -> ConnectorReport:
    source = getattr(collector, "source", "")
    if hasattr(source, "value"):
        source = source.value
    return {
        "connector": type(collector).__name__,
        "source": str(source),
        "status": status,
        "reason": reason,
        "observation_count": observation_count,
    }


def _summarize_result(ctx: dict, result: dict) -> dict[str, str]:
    """Return the stable fields the CLI prints after a run completes."""
    verdict = result.get("verdict")
    if not isinstance(verdict, dict):
        verdict = {}
    root = verdict.get("root_symptom")
    root_symptom = root.get("symptom", "") if isinstance(root, dict) else ""
    symptoms = result.get("symptoms")
    symptom_names = ", ".join(sorted(str(name) for name in symptoms)) if isinstance(symptoms, dict) else ""
    context_id = ctx.get("id")
    job_id = ctx.get("job_id")
    if job_id is None:
        job_id = context_id
    attempt_id = ctx.get("attempt_id")
    taxonomy = taxonomy_for_service_value(service_outcome(result))
    return {
        "context_id": str(context_id if context_id is not None else "?"),
        "job_id": str(job_id if job_id is not None else "?"),
        "attempt_id": str(attempt_id if attempt_id is not None else ""),
        "terminal_state": taxonomy.terminal_state,
        "sub_domain": taxonomy.sub_domain,
        "root_symptom": root_symptom if root_symptom != "" else "none",
        "symptoms": symptom_names if symptom_names != "" else "none",
    }
