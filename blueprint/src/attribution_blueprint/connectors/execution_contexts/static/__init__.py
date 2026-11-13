# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-fed ExecutionContextAdapter.

Used when there is no scheduler to discover from, the operator supplies the
ExecutionContext fields directly via CLI flags or TOML. `discover` returns
that configured context. `discover_by_ids` preserves explicit ID validation
for callers that provide one. `discover_by_time_range` is unsupported; this
ECA does not scan.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.types import ExecutionContext


def _iana_timezone_key(value: object, field: str) -> str:
    """Validate and canonicalize an IANA timezone key.

    Args:
        value: Candidate timezone key.
        field: Configuration field name for validation errors.

    Returns:
        The canonical ZoneInfo key.

    Raises:
        ValueError: If the value is not a valid IANA timezone key.
    """
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a valid IANA timezone key, got {value!r}")
    try:
        return ZoneInfo(value).key
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"{field} must be a valid IANA timezone key, got {value!r}") from exc


class StaticExecutionContextAdapter:
    def __init__(
        self,
        *,
        id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        nodes: list[str] | None = None,
        state: str = "UNKNOWN",
        name: str | None = None,
        user_name: str | None = None,
        std_out: str | None = None,
        std_err: str | None = None,
        workload_type: str = "",
        workload_framework: str | None = None,
        attempt_id: str = "0",
        gpu_count: int | None = None,
        timezone: str | None = None,
        contexts: list[ExecutionContext] | None = None,
    ):
        if gpu_count is not None and gpu_count < 0:
            raise ValueError("static ECA: gpu_count must be non-negative")
        if contexts is not None:
            if contexts == []:
                raise ValueError("static ECA: contexts must be non-empty")
            self._contexts = []
            for ctx in contexts:
                normalized = dict(ctx)
                context_timezone = normalized.get("timezone")
                if context_timezone is not None and context_timezone != "":
                    normalized["timezone"] = _iana_timezone_key(
                        context_timezone, f"static ECA context {ctx.get('id', '?')!r} timezone"
                    )
                self._contexts.append(normalized)
            self._id = ""
            self._start_time = None
            self._end_time = None
            self._nodes = []
            self._state = state
            self._name = name
            self._user_name = user_name
            self._std_out = std_out
            self._std_err = std_err
            self._workload_type = workload_type
            self._workload_framework = workload_framework
            self._attempt_id = attempt_id
            self._gpu_count = gpu_count
            self._timezone = None
            return

        if id is None or id == "":
            raise ValueError("static ECA: id is required")
        if start_time is None:
            raise ValueError("static ECA: start_time is required")
        if end_time is None:
            raise ValueError("static ECA: end_time is required")
        if nodes is None or nodes == []:
            raise ValueError("static ECA: nodes must be non-empty")
        # tz-aware datetimes are required so the adapter never silently
        # anchors to the blueprint host's local tz (the same anti-pattern the
        # Slurm ECA's `cluster_tz` parameter avoids). Callers must pass either
        # UTC or the cluster's own zone explicitly; the hash-derived
        # `static-<sha256[:16]>` id then stays deterministic across hosts.
        if start_time.tzinfo is None:
            raise ValueError("static ECA: start_time must be tz-aware (got naive datetime)")
        if end_time.tzinfo is None:
            raise ValueError("static ECA: end_time must be tz-aware (got naive datetime)")
        self._contexts = None
        self._id = id
        self._start_time = start_time
        self._end_time = end_time
        self._nodes = list(nodes)
        self._state = state
        self._name = name
        self._user_name = user_name
        self._std_out = std_out
        self._std_err = std_err
        self._workload_type = workload_type
        self._workload_framework = workload_framework
        self._attempt_id = attempt_id
        self._gpu_count = gpu_count
        self._timezone = (
            _iana_timezone_key(timezone, "static ECA: timezone") if timezone is not None and timezone != "" else None
        )

    def discover(self) -> list[ExecutionContext]:
        if self._contexts is not None:
            return [dict(ctx) for ctx in self._contexts]

        ctx: ExecutionContext = {
            "id": self._id,
            "attempt_id": self._attempt_id,
            "workload_type": self._workload_type,
            # BP-internal connector identity. Hardcoded "static", connector
            # identity is structural, not user-configurable. Operators can
            # still set `workload_type` to spoof the wire (e.g. "slurm"),
            # but SQLite's `scheduler_type` column will always show "static"
            # for contexts discovered via this adapter.
            "scheduler": "static",
            "nodes": self._nodes,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "job_id": self._id,
            "state": self._state,
        }
        if self._name is not None:
            ctx["name"] = self._name
        if self._user_name is not None:
            ctx["user_name"] = self._user_name
        if self._std_out is not None:
            ctx["std_out"] = self._std_out
        if self._std_err is not None:
            ctx["std_err"] = self._std_err
        if self._workload_framework is not None:
            ctx["workload_framework"] = self._workload_framework
        if self._gpu_count is not None:
            ctx["gpu_count"] = self._gpu_count
        if self._timezone is not None:
            ctx["timezone"] = self._timezone
        return [ctx]

    def discover_by_ids(self, ids: list[str]) -> list[ExecutionContext]:
        if self._contexts is not None:
            if ids == []:
                raise ValueError("static ECA requires at least one --ids value")
            by_id = {ctx["id"]: ctx for ctx in self._contexts}
            missing = [id_ for id_ in ids if id_ not in by_id]
            if missing != []:
                raise ValueError(f"static ECA id mismatch: unknown --ids values {missing!r}")
            return [dict(by_id[id_]) for id_ in ids]

        if len(ids) != 1:
            raise ValueError(f"static ECA supports exactly one --ids value, got {ids!r}")
        if ids[0] != self._id:
            raise ValueError(f"static ECA id mismatch: --ids={ids[0]!r} but configured id={self._id!r}")
        return self.discover()

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[ExecutionContext]:
        raise NotImplementedError("static ECA does not scan time ranges; use --ids with --execution-context static")


def _derive_static_id(start_time: datetime, end_time: datetime, nodes: list[str]) -> str:
    payload = f"{start_time.isoformat()}|{end_time.isoformat()}|{','.join(sorted(nodes))}"
    return f"static-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _load_contexts(path: Path) -> list[ExecutionContext]:
    """Load static execution contexts from a JSON file.

    Args:
        path: JSON file containing either a list of contexts or a mapping with
            a ``contexts`` list.

    Returns:
        Normalized execution contexts ready for the static adapter.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MCAError(f"eca.static.contexts_path could not be read: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MCAError(f"eca.static.contexts_path is not valid JSON: {path}") from exc

    if isinstance(raw, dict):
        raw_contexts = raw.get("contexts")
    else:
        raw_contexts = raw
    if not isinstance(raw_contexts, list) or raw_contexts == []:
        raise MCAError("eca.static.contexts_path must contain a non-empty contexts list")

    contexts = [_normalize_context(raw_context, index, path) for index, raw_context in enumerate(raw_contexts, start=1)]
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for ctx in contexts:
        ctx_id = ctx["id"]
        if ctx_id in seen:
            duplicate_ids.append(ctx_id)
        seen.add(ctx_id)
    if duplicate_ids != []:
        raise MCAError(f"eca.static.contexts_path contains duplicate context ids: {sorted(set(duplicate_ids))!r}")
    return contexts


def _normalize_context(raw_context: object, index: int, path: Path) -> ExecutionContext:
    """Coerce one JSON object into the ExecutionContext contract.

    Args:
        raw_context: Raw JSON value from the context list.
        index: One-based row number used in validation messages.
        path: Source JSON path used in validation messages.

    Returns:
        A normalized execution context.
    """
    from attribution_blueprint.mca import _coerce_datetime, _coerce_int, _coerce_str_list

    if not isinstance(raw_context, dict):
        raise MCAError(f"{path}: context #{index} must be an object")
    missing = [
        key
        for key in ("id", "start_time", "end_time", "nodes")
        if key not in raw_context or raw_context[key] is None or raw_context[key] == "" or raw_context[key] == []
    ]
    if missing != []:
        raise MCAError(f"{path}: context #{index} missing required keys: {missing}")

    start_time = _coerce_datetime(raw_context["start_time"], f"{path}: context #{index} start_time")
    end_time = _coerce_datetime(raw_context["end_time"], f"{path}: context #{index} end_time")
    if start_time.tzinfo is None:
        raise MCAError(f"{path}: context #{index} start_time must be tz-aware")
    if end_time.tzinfo is None:
        raise MCAError(f"{path}: context #{index} end_time must be tz-aware")

    ctx: ExecutionContext = dict(raw_context)
    ctx["id"] = str(raw_context["id"])
    ctx["start_time"] = start_time
    ctx["end_time"] = end_time
    ctx["nodes"] = _coerce_str_list(raw_context["nodes"], f"{path}: context #{index} nodes")
    ctx["attempt_id"] = str(raw_context.get("attempt_id", "0"))
    ctx["workload_type"] = str(raw_context.get("workload_type", ""))
    ctx["scheduler"] = str(raw_context.get("scheduler", "static"))
    ctx["job_id"] = str(raw_context.get("job_id", ctx["id"]))
    ctx["state"] = str(raw_context.get("state", "UNKNOWN"))
    context_timezone = raw_context.get("timezone")
    if context_timezone is not None and context_timezone != "":
        try:
            ctx["timezone"] = _iana_timezone_key(context_timezone, f"{path}: context #{index} timezone")
        except ValueError as exc:
            raise MCAError(str(exc)) from exc
    else:
        ctx.pop("timezone", None)
    if "time_limit" in raw_context and raw_context["time_limit"] is not None:
        ctx["time_limit"] = _coerce_int(raw_context["time_limit"], f"{path}: context #{index} time_limit")
    if "gpu_count" in raw_context and raw_context["gpu_count"] is not None:
        gpu_count = _coerce_int(raw_context["gpu_count"], f"{path}: context #{index} gpu_count")
        if gpu_count < 0:
            raise MCAError(f"{path}: context #{index} gpu_count must be non-negative")
        ctx["gpu_count"] = gpu_count
    return ctx


def _build(spec: dict, ctx: dict) -> StaticExecutionContextAdapter:
    from attribution_blueprint.mca import _coerce_datetime, _coerce_int, _coerce_str_list

    contexts_path = spec.get("contexts_path")
    if contexts_path is not None and contexts_path != "":
        return StaticExecutionContextAdapter(contexts=_load_contexts(Path(str(contexts_path))))

    missing = [k for k in ("start_time", "end_time", "nodes") if not spec.get(k)]
    if missing:
        raise MCAError(f"eca.static missing required keys: {missing}")
    start_time = _coerce_datetime(spec["start_time"], "eca.static.start_time")
    end_time = _coerce_datetime(spec["end_time"], "eca.static.end_time")
    nodes = _coerce_str_list(spec["nodes"], "eca.static.nodes")
    id_ = spec.get("id")
    id_str = str(id_) if id_ not in (None, "") else _derive_static_id(start_time, end_time, nodes)
    gpu_count_raw = spec.get("gpu_count")
    gpu_count = None
    if gpu_count_raw is not None:
        gpu_count = _coerce_int(gpu_count_raw, "eca.static.gpu_count")
        if gpu_count < 0:
            raise MCAError("eca.static.gpu_count must be non-negative")
    try:
        return StaticExecutionContextAdapter(
            id=id_str,
            start_time=start_time,
            end_time=end_time,
            nodes=nodes,
            state=spec.get("state", "UNKNOWN"),
            name=spec.get("name"),
            user_name=spec.get("user_name"),
            std_out=spec.get("std_out"),
            std_err=spec.get("std_err"),
            workload_type=str(spec.get("workload_type", "")),
            workload_framework=spec.get("workload_framework"),
            attempt_id=str(spec.get("attempt_id", "0")),
            gpu_count=gpu_count,
            timezone=spec.get("timezone"),
        )
    except ValueError as exc:
        raise MCAError(str(exc)) from exc


_registry.register_component(
    framework="eca",
    name="static",
    schema={
        "id": _PARAM,
        "start_time": _PARAM,
        "end_time": _PARAM,
        "nodes": _PARAM,
        "state": _PARAM,
        "name": _PARAM,
        "user_name": _PARAM,
        "std_out": _PARAM,
        "std_err": _PARAM,
        "workload_type": _PARAM,
        "attempt_id": _PARAM,
        "gpu_count": _PARAM,
        "timezone": _PARAM,
        "contexts_path": _PARAM,
        # Optional workload-framework label (e.g. "megatron", "pytorch").
        # Omitted by default; populated only when known. The key name is
        # `workload_framework` (not `framework`) to avoid colliding with the
        # MCA connector-framework concept that pervades the codebase.
        "workload_framework": _PARAM,
    },
    build=_build,
)


__all__ = ["StaticExecutionContextAdapter"]
