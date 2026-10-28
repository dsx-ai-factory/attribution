# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kueue (Kubernetes batch admission queue) ExecutionContextAdapter.

Mirrors the Slurm adapter's shape: read the raw Kubernetes API payloads,
normalize them into the wire-format `ExecutionContext` TypedDict, and never
let any Kubernetes/Kueue-shaped data structure escape this file.

Wire-format mapping vs. the Slurm adapter:

    Slurm field        Kueue analog
    -----------------  -----------------------------------------------------
    sluid              Workload.metadata.uid (carried as `workload_uid`)
    attempt_id         Canonical Workload UID + admission boundary + JobSet
                       restart generation identity
    job_id             "<namespace>/<name>" of the Workload object
    state              Service-facing scheduler state. Terminal Kueue states
                       are normalized to COMPLETED, FAILED, TIMEOUT,
                       NODE_FAIL, PREEMPTED, REQUEUED, or EVICTED.
    kueue_state        Derived from Workload status conditions + Finished
                       reason
    start_time         Earliest lastTransitionTime of any Admitted=True
                       condition (workloads that were preempted and
                       re-admitted keep their original admission as
                       start_time so a time-window query over the first
                       interval still matches)
    end_time           lastTransitionTime of the Finished=True condition
                       (falls back to now() while the workload is running,
                       same convention as the Slurm adapter for RUNNING ExecutionContexts)
    nodes              Union of .spec.nodeName across the owner Job's pods
    failed_node        First pod observed with a non-zero terminated.exitCode
    time_limit         Underlying batch/v1 Job .spec.activeDeadlineSeconds,
                       if present, rounded up to service-facing minutes
    name               Workload .metadata.name (the user-visible workload)
    user_name          Resolved from common batch-tooling labels on the
                       Workload's own metadata (`user`, `username`,
                       `kueue.x-k8s.io/user`, `app.kubernetes.io/created-by`)

`std_out` / `std_err` are intentionally absent. Slurm carries a filesystem
path because the local_file log connector needs one; in Kubernetes, pod
stdout is fetched via `kubectl logs` and belongs to a future log connector,
not the ECA.

Pod ownership is resolved via the standard batch/v1 Job pod labels, not the
Kueue label — see `pod_selector_for` in helpers.py for the selector logic
and `_jobs_for_workload` for how owner Jobs are verified by uid, not name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple, NotRequired, TypedDict

from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kueue.attempts import (
    WorkloadAttempt,
    attempts_from_live_resources,
    attempts_from_workload_history,
    merge_attempts,
)
from attribution_blueprint.connectors.execution_contexts.kueue.helpers import (
    POD_JOBSET_LABEL as _POD_JOBSET_LABEL,
    POD_OWNER_LABEL as _POD_OWNER_LABEL,
    controller_owner as _controller_owner,
)
from attribution_blueprint.connectors.execution_contexts.kueue.prometheus_placement import (
    PlacementQuery,
    PlacementResolver,
)
from attribution_blueprint.kubernetes.client import (
    KubernetesClient,
    KubernetesClientError,
)
from attribution_blueprint.kubernetes.daemon.models import PlacementRecord, PodPlacement
from attribution_blueprint.kubernetes.daemon.store import PlacementReader
from attribution_blueprint.kubernetes.helpers import (
    NODE_LOSS_DISRUPTION_REASONS as _NODE_LOSS_DISRUPTION_REASONS,
    earliest_job_failure_reason as _earliest_job_failure_reason,
    job_time_limit_minutes as _job_time_limit_minutes,
    pod_node_failure_reason as _pod_node_failure_reason,
)
from attribution_blueprint.types import assert_aware

_KUEUE_GROUP = "kueue.x-k8s.io"
# Kueue versions its Workload CRD; this is a config default, not a
# runtime-discovered "preferred version" — see `KueueExecutionContextAdapter`.
_DEFAULT_WORKLOAD_API_VERSION = f"{_KUEUE_GROUP}/v1beta1"


class KueueExecutionContext(TypedDict):
    """Adapter-specific ExecutionContext shape.

    Same spine as `attribution_blueprint.types.ExecutionContext`; the
    `workload_uid`, `namespace`, `kueue_state`, `queue_name`, and
    `priority_class` keys are Kueue-specific extensions stored on the dict for
    the per-context raw_scheduler_json column (see output_schema.md).
    """

    id: str
    workload_type: str
    # BP-internal connector identity (always "kueue"); consumed by
    # SQLiteOutput as the `scheduler_type` column independently of the
    # service-facing `workload_type`.
    scheduler: str
    nodes: list[str]
    start_time: datetime
    end_time: datetime

    job_id: str
    workload_uid: str
    attempt_id: str
    namespace: str
    state: str
    kueue_state: str

    name: NotRequired[str]
    user_name: NotRequired[str]
    failed_node: NotRequired[str]
    time_limit: NotRequired[int]
    queue_name: NotRequired[str]
    priority_class: NotRequired[int]
    gpu_count: NotRequired[int]
    admission_key: NotRequired[str | None]
    restart_attempt: NotRequired[str]
    owner_job_uids: NotRequired[list[str]]
    attempt_identity_resolved: NotRequired[bool]
    # Pod names belonging to this Workload at the time of discovery. Empty
    # list when the owner type isn't supported or all pods have been
    # garbage-collected.
    pod_names: NotRequired[list[str]]
    pod_placements: NotRequired[list[PodPlacement]]
    history_complete: NotRequired[bool]
    # Pod name -> `.spec.nodeName`, for connectors that need per-pod node
    # identity. Unscheduled pods are absent.
    pod_nodes: NotRequired[dict[str, str]]


@dataclass
class _AttemptEvidence:
    """Inputs for one admitted execution, keyed by canonical attempt ID."""

    attempts: list[WorkloadAttempt] = field(default_factory=list)
    workload: dict | None = None
    pods: dict[str, dict] = field(default_factory=dict)
    jobs: dict[str, dict] = field(default_factory=dict)
    placements: dict[str, PlacementRecord] = field(default_factory=dict)
    retained_coverage: list[dict] = field(default_factory=list)

    @property
    def attempt(self) -> WorkloadAttempt:
        return merge_attempts(self.attempts)

    def merge(self, other: _AttemptEvidence) -> None:
        self.attempts.extend(other.attempts)
        if other.workload is not None:
            self.workload = other.workload
        self.pods.update(other.pods)
        self.jobs.update(other.jobs)
        self.placements.update(other.placements)
        self.retained_coverage.extend(other.retained_coverage)


# Kueue's Workload condition vocabulary. Stable since Kueue v0.6 and described
# in kueue.x-k8s.io/v1beta1; values are documented in the Kueue API reference.
_COND_QUOTA_RESERVED = "QuotaReserved"
_COND_ADMITTED = "Admitted"
_COND_PODS_READY = "PodsReady"
_COND_FINISHED = "Finished"
_COND_EVICTED = "Evicted"
_COND_PREEMPTED = "Preempted"
_COND_REQUEUED = "Requeued"

# Native Workload Evicted reasons from the Kueue v1beta1 API.
_EVICTED_PREEMPTED = "EVICTED_PREEMPTED"
_EVICTED_NODE_FAILURES = "EVICTED_NODEFAILURES"


def _pod_selector_for(workload: dict) -> tuple[str, str, str] | None:
    """Return the namespace-scoped Pod selector for a supported owner."""
    owner = _controller_owner(workload)
    if owner is None:
        return None
    namespace = (workload.get("metadata") or {}).get("namespace") or "default"
    if owner.get("kind") == "Job" and owner.get("uid"):
        return namespace, _POD_OWNER_LABEL, owner["uid"]
    if owner.get("kind") == "JobSet" and owner.get("name"):
        return namespace, _POD_JOBSET_LABEL, owner["name"]
    return None


def _pod_matches_selector(pod: dict, selector: tuple[str, str, str]) -> bool:
    """Return whether a Pod matches a namespace-scoped owner selector."""
    namespace, label_key, label_value = selector
    metadata = pod.get("metadata") or {}
    return metadata.get("namespace") == namespace and (metadata.get("labels") or {}).get(label_key) == label_value


def _jobset_owner_uid(job: dict) -> str | None:
    """Return the UID of the JobSet that controls a Job."""
    owner = _controller_owner(job, "JobSet")
    return owner.get("uid") if owner else None


def _jobs_for_workload(workload: dict, owner_jobs: dict[str, dict]) -> list[dict]:
    """Return UID-verified direct or JobSet child Jobs for one Workload."""
    owner = _controller_owner(workload)
    if owner is None:
        return []
    namespace = (workload.get("metadata") or {}).get("namespace") or ""
    owner_name = owner.get("name")
    owner_uid = owner.get("uid")
    if not owner_name:
        return []
    if owner.get("kind") == "Job":
        job = owner_jobs.get(f"{namespace}/{owner_name}")
        if job is None or (job.get("metadata") or {}).get("uid") != owner_uid:
            return []
        return [job]
    if owner.get("kind") == "JobSet":
        return [
            job
            for job in owner_jobs.values()
            if (job.get("metadata") or {}).get("namespace") == namespace
            and ((job.get("metadata") or {}).get("labels") or {}).get(_POD_JOBSET_LABEL) == owner_name
            and _jobset_owner_uid(job) == owner_uid
        ]
    return []


def _owner_job_uids(workload: dict, owner_jobs: dict[str, dict]) -> set[str]:
    """Return exact direct or child Job UIDs belonging to one Workload."""
    owner = _controller_owner(workload)
    if owner is None:
        return set()
    if owner.get("kind") == "Job" and owner.get("uid"):
        return {owner["uid"]}
    if owner.get("kind") == "JobSet":
        return {
            uid for job in _jobs_for_workload(workload, owner_jobs) if (uid := (job.get("metadata") or {}).get("uid"))
        }
    return set()


def _parse_dt(value: str | None) -> datetime | None:
    """Parse a Kubernetes RFC3339 timestamp into a tz-aware datetime.

    The k8s API server always serializes timestamps with a trailing `Z`.
    Python's `datetime.fromisoformat` accepts `Z` since 3.11, which matches
    the project's `requires-python` floor.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _condition(conditions: list[dict], cond_type: str) -> dict | None:
    """Return the most recent (by lastTransitionTime) condition of `cond_type`."""
    matches = [c for c in conditions if c.get("type") == cond_type]
    if not matches:
        return None
    return max(matches, key=lambda c: c.get("lastTransitionTime", ""))


def _earliest_condition_time(conditions: list[dict], cond_type: str) -> datetime | None:
    """Return the EARLIEST lastTransitionTime of any True `cond_type` condition.

    Used for `start_time` derivation: a workload that was admitted, preempted,
    and re-admitted has two Admitted=True conditions; the original admission
    is what callers expect from `start_time`, so a time-window query covering
    the first admission interval still matches.
    """
    times: list[datetime] = []
    for c in conditions:
        if c.get("type") != cond_type or c.get("status") != "True":
            continue
        dt = _parse_dt(c.get("lastTransitionTime"))
        if dt is not None:
            times.append(dt)
    return min(times) if times else None


def _derive_state(conditions: list[dict]) -> str:
    """Map a Workload's condition set to a single coarse state string.

    Order of precedence reflects Kueue's lifecycle: Finished beats Evicted
    beats Admitted beats QuotaReserved beats anything else. A finished
    workload that was also evicted earlier in its lifetime still reports
    FINISHED, mirroring how Slurm's terminal state wins over earlier
    transitions.
    """
    finished = _condition(conditions, _COND_FINISHED)
    if finished and finished.get("status") == "True":
        reason = (finished.get("reason") or "").strip()
        return f"FINISHED_{reason.upper()}" if reason else "FINISHED"
    evicted = _condition(conditions, _COND_EVICTED)
    if evicted and evicted.get("status") == "True":
        reason = (evicted.get("reason") or "").strip()
        return f"EVICTED_{reason.upper()}" if reason else "EVICTED"
    preempted = _condition(conditions, _COND_PREEMPTED)
    if preempted and preempted.get("status") == "True":
        return "PREEMPTED"
    admitted = _condition(conditions, _COND_ADMITTED)
    if admitted and admitted.get("status") == "True":
        ready = _condition(conditions, _COND_PODS_READY)
        if ready and ready.get("status") == "True":
            return "RUNNING"
        return "ADMITTED"
    quota = _condition(conditions, _COND_QUOTA_RESERVED)
    if quota and quota.get("status") == "True":
        return "QUOTA_RESERVED"
    return "PENDING"


def _condition_time(conditions: list[dict], cond_type: str) -> datetime | None:
    """Most-recent lastTransitionTime of a True condition. Used for end_time."""
    c = _condition(conditions, cond_type)
    if c is None or c.get("status") != "True":
        return None
    return _parse_dt(c.get("lastTransitionTime"))


def _pod_reports_node_failure(pod: dict) -> bool:
    """Return whether a Pod carries a node-loss signal.

    Delegates to the shared Kubernetes helper so live discovery and retained
    daemon evidence cannot drift apart.
    """
    return _pod_node_failure_reason(pod) is not None


class _PodEvidence(NamedTuple):
    nodes: list[str]
    failed_node: str | None
    pod_names: list[str]
    pod_nodes: dict[str, str]


def _collect_pod_evidence(
    pods: list[dict],
    selector: tuple[str, str, str] | None,
) -> _PodEvidence:
    """Return sorted unique nodes, the lex-smallest failed_node, sorted pod
    names, and the pod -> node mapping for pods matching `selector`.

    Re-checks each pod against `_pod_matches_selector` as a defensive filter,
    since cluster-wide discovery can pool pods from other namespaces. Returns
    empty evidence when `selector` is None (no recognised owner).
    """
    if not selector:
        return _PodEvidence([], None, [], {})
    nodes: set[str] = set()
    node_loss_nodes: set[str] = set()
    exit_code_nodes: set[str] = set()
    pod_names: set[str] = set()
    pod_nodes: dict[str, str] = {}
    for pod in pods:
        if not _pod_matches_selector(pod, selector):
            continue
        meta = pod.get("metadata") or {}
        pod_name = meta.get("name")
        if pod_name:
            pod_names.add(pod_name)
        spec = pod.get("spec") or {}
        node_name = spec.get("nodeName")
        if node_name:
            nodes.add(node_name)
            if pod_name:
                pod_nodes[pod_name] = node_name
        status = pod.get("status") or {}
        if _pod_reports_node_failure(pod) and node_name:
            node_loss_nodes.add(node_name)
        for cs in status.get("containerStatuses") or []:
            terminated = (cs.get("state") or {}).get("terminated") or {}
            exit_code = terminated.get("exitCode")
            if isinstance(exit_code, int) and exit_code != 0 and node_name:
                exit_code_nodes.add(node_name)
                break
    # `failed_node` is the lexicographically smallest node within the
    # strongest evidence tier available, so the result is stable across
    # API response orderings. Node-loss evidence outranks a generic
    # non-zero exit code.
    if node_loss_nodes:
        failed_node = min(node_loss_nodes)
    elif exit_code_nodes:
        failed_node = min(exit_code_nodes)
    else:
        failed_node = None
    return _PodEvidence(sorted(nodes), failed_node, sorted(pod_names), pod_nodes)


def _live_pod_placements(pods: list[dict], selector: tuple[str, str, str] | None) -> list[PodPlacement]:
    """Return bound Pod placements matching one Workload owner selector."""
    if selector is None:
        return []
    placements: list[PodPlacement] = []
    for pod in pods:
        if not _pod_matches_selector(pod, selector):
            continue
        metadata = pod.get("metadata") or {}
        node_name = (pod.get("spec") or {}).get("nodeName")
        if metadata.get("uid") and metadata.get("name") and node_name:
            placements.append(
                {
                    "pod_uid": metadata["uid"],
                    "pod_name": metadata["name"],
                    "node_name": node_name,
                    "source": "live_pod",
                }
            )
    return placements


def _event_is_in_context(event: dict, context: KueueExecutionContext) -> bool:
    """Reject stale Events outside this Workload's execution interval."""
    timestamp = _parse_dt(
        event.get("lastTimestamp") or event.get("eventTime") or (event.get("metadata") or {}).get("creationTimestamp")
    )
    return timestamp is not None and context["start_time"] <= timestamp <= context["end_time"]


def _event_message(event: dict) -> str:
    """Return an Event message without Kubernetes' aggregation prefix."""
    message = (event.get("message") or "").strip()
    return message.removeprefix("(combined from similar events):").strip()


def _pod_names_from_creation_events(
    creation_events: list[dict],
    context: KueueExecutionContext,
    job_uids: set[str],
) -> set[str]:
    """Extract pod names from a Job's `SuccessfulCreate` Events.

    `job_uids` re-checks the exact owner Job UID. Events are also constrained
    to the Workload interval to avoid stale same-name data.
    """
    pod_names: set[str] = set()
    for event in creation_events:
        obj = event.get("involvedObject") or {}
        if obj.get("kind") != "Job" or obj.get("uid") not in job_uids or event.get("reason") != "SuccessfulCreate":
            continue
        if not _event_is_in_context(event, context):
            continue
        message = _event_message(event)
        if message.startswith("Created pod:"):
            pod_name = message.removeprefix("Created pod:").strip()
            if pod_name:
                pod_names.add(pod_name)
    return pod_names


def _node_from_scheduled_events(
    scheduled_events: list[dict],
    context: KueueExecutionContext,
    namespace: str,
    pod_name: str,
) -> str | None:
    """Extract the assigned node from one Pod's `Scheduled` Events."""
    for event in scheduled_events:
        obj = event.get("involvedObject") or {}
        if obj.get("kind") != "Pod" or obj.get("name") != pod_name or event.get("reason") != "Scheduled":
            continue
        if not _event_is_in_context(event, context):
            continue
        message = _event_message(event)
        prefix = f"Successfully assigned {namespace}/{pod_name} to "
        if message.startswith(prefix):
            node = message.removeprefix(prefix).strip()
            if node:
                return node
    return None


def _has_node_failure_pod(pods: list[dict], selector: tuple[str, str, str] | None) -> bool:
    """Return whether a selected Pod reports a Kubernetes node-loss signal."""
    if selector is None:
        return False
    for pod in pods:
        if not _pod_matches_selector(pod, selector):
            continue
        if _pod_reports_node_failure(pod):
            return True
    return False


def _workload_condition_is_true(workload: dict, condition_type: str) -> bool:
    conditions = (workload.get("status") or {}).get("conditions") or []
    condition = _condition(conditions, condition_type)
    return bool(condition and condition.get("status") == "True")


# kueue_state values where the Workload currently holds admission/quota.
# These outrank stale Requeued=True history in `_service_state` below.
_CURRENT_ADMISSION_OR_RESERVATION_STATES = frozenset({"ADMITTED", "RUNNING", "QUOTA_RESERVED"})


def _service_state(
    kueue_state: str,
    workload: dict,
    pods: list[dict],
    owner_jobs: dict[str, dict],
) -> str:
    """Normalize Kueue states for service refiners."""
    if kueue_state == "FINISHED_SUCCEEDED":
        return "COMPLETED"
    if kueue_state in {_EVICTED_PREEMPTED, "PREEMPTED"}:
        return "PREEMPTED"
    if kueue_state == _EVICTED_NODE_FAILURES:
        return "NODE_FAIL"
    if _has_node_failure_pod(pods, _pod_selector_for(workload)):
        return "NODE_FAIL"
    # Requeued=True can stay True after re-admission, so gate on kueue_state instead
    currently_admitted_or_reserved = kueue_state in _CURRENT_ADMISSION_OR_RESERVATION_STATES
    currently_finished = kueue_state.startswith("FINISHED")
    if (
        not currently_admitted_or_reserved
        and not currently_finished
        and _workload_condition_is_true(workload, _COND_REQUEUED)
    ):
        return "REQUEUED"
    if kueue_state.startswith("EVICTED"):
        # Any other Evicted reasons (e.g. queue stopped,
        # admission check failed, pods-ready timeout, deactivated) are
        # collapsed to the generic EVICTED.
        return "EVICTED"
    if kueue_state != "FINISHED_FAILED":
        return kueue_state
    if _earliest_job_failure_reason(_jobs_for_workload(workload, owner_jobs)) == "DeadlineExceeded":
        return "TIMEOUT"
    return "FAILED"


def _extract_time_limit(workload: dict, owner_jobs: dict[str, dict]) -> int | None:
    """Pull the verified owner Job deadline as service-facing minutes."""
    return _job_time_limit_minutes(_jobs_for_workload(workload, owner_jobs))


def _non_negative_int(value: object) -> int | None:
    """Coerce a raw Kubernetes resource value to a non-negative int, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _resource_gpu_count(resources: object) -> int:
    """Return the sum of GPU-like extended resources in one `requests`/`limits` map."""
    if not isinstance(resources, dict):
        return 0
    total = 0
    for key, raw_value in resources.items():
        key_text = str(key)
        if key_text != "gpu" and key_text.endswith("/gpu") is False:
            continue
        value = _non_negative_int(raw_value)
        if value is not None:
            total += value
    return total


def _container_gpu_count(container: object) -> int:
    """Return GPU count requested by one container, falling back to limits when requests are absent."""
    if not isinstance(container, dict):
        return 0
    resources = container.get("resources")
    if not isinstance(resources, dict):
        return 0
    requests_count = _resource_gpu_count(resources.get("requests"))
    if requests_count > 0:
        return requests_count
    return _resource_gpu_count(resources.get("limits"))


def _workload_gpu_count(workload: dict) -> int | None:
    """Return total GPU count across all pod sets.

    Returns 0 when pod sets are parseable but no GPU resource is declared,
    or None when the Workload shape itself is missing/malformed.
    """
    total = 0
    spec = workload.get("spec")
    if not isinstance(spec, dict):
        return None
    pod_sets = spec.get("podSets")
    if not isinstance(pod_sets, list):
        return None
    saw_container = False
    for pod_set in pod_sets:
        if not isinstance(pod_set, dict):
            continue
        count = _non_negative_int(pod_set.get("count"))
        if count is None:
            count = 1
        template = pod_set.get("template")
        if not isinstance(template, dict):
            continue
        pod_spec = template.get("spec")
        if not isinstance(pod_spec, dict):
            continue
        containers = pod_spec.get("containers")
        if not isinstance(containers, list):
            continue
        if containers != []:
            saw_container = True
        per_pod = sum(_container_gpu_count(container) for container in containers)
        total += count * per_pod
    if saw_container is False:
        return None
    return total


def _workload_window(workload: dict, now: datetime) -> tuple[datetime, datetime] | None:
    """Return `(start_time, end_time)` for a Workload using only data carried
    on the Workload object — no pod or owner-Job fetches.

    Cheap pre-filter for `discover_by_time_range`, before the full
    `_parse_workload` pass. Returns `None` if no start time can be derived.
    """
    meta = workload.get("metadata") or {}
    if not meta.get("uid") or not meta.get("name"):
        return None
    conditions = (workload.get("status") or {}).get("conditions") or []
    start_time = _earliest_condition_time(conditions, _COND_ADMITTED)
    if start_time is None:
        start_time = _parse_dt(meta.get("creationTimestamp"))
    if start_time is None:
        return None
    end_time = _condition_time(conditions, _COND_FINISHED)
    if end_time is None:
        evicted = _condition_time(conditions, _COND_EVICTED)
        if evicted is not None:
            end_time = evicted
    if end_time is None:
        end_time = now
    return start_time, end_time


def _context_from_attempt_evidence(
    evidence: _AttemptEvidence,
) -> KueueExecutionContext:
    """Build one admitted context after live and retained evidence is united."""
    attempt = evidence.attempt
    workload = evidence.workload
    member_pods = list(evidence.pods.values())
    selector = _pod_selector_for(workload) if workload is not None else None
    pod_evidence = _collect_pod_evidence(member_pods, selector)
    state, kueue_state = _history_attempt_state(attempt)
    jobs_by_name = {
        (f"{(job.get('metadata') or {}).get('namespace') or ''}/{(job.get('metadata') or {}).get('name') or ''}"): job
        for job in evidence.jobs.values()
    }
    if workload is not None:
        live_kueue_state = _derive_state((workload.get("status") or {}).get("conditions") or [])
        live_state = _service_state(
            live_kueue_state,
            workload,
            member_pods,
            jobs_by_name,
        )
        if attempt.terminal is None:
            state = live_state
            kueue_state = live_kueue_state

    owner_job_uids = {uid for job in evidence.jobs.values() if (uid := (job.get("metadata") or {}).get("uid"))}
    if workload is not None:
        owner = _controller_owner(workload)
        if owner and owner.get("kind") == "Job" and owner.get("uid"):
            owner_job_uids.add(owner["uid"])

    placements: dict[str, PodPlacement] = {
        placement["pod_uid"]: placement for placement in _live_pod_placements(member_pods, selector)
    }
    retained_node_loss_nodes: set[str] = set()
    retained_exit_nodes: set[str] = set()
    for record in evidence.placements.values():
        if record.pod_uid not in attempt.pod_uids:
            resource_uids = {
                resource.uid
                for resource in (
                    *record.owner_references,
                    *record.associated_resources,
                )
            }
            if record.owner is not None:
                resource_uids.add(record.owner.uid)
            scheduled_at = _parse_dt(record.scheduled_at)
            if (
                not owner_job_uids.intersection(resource_uids)
                or scheduled_at is None
                or not (attempt.start_time <= scheduled_at <= attempt.end_time)
            ):
                continue
        placements.setdefault(
            record.pod_uid,
            {
                "pod_uid": record.pod_uid,
                "pod_name": record.pod_name,
                "node_name": record.node_name,
                "source": "kubernetes_daemon",
            },
        )
        if record.node_failure_reason == "NodeLost" or record.node_failure_reason in _NODE_LOSS_DISRUPTION_REASONS:
            retained_node_loss_nodes.add(record.node_name)
        elif record.node_failure_reason:
            retained_exit_nodes.add(record.node_name)

    live_node_loss_nodes = {
        node_name
        for pod in member_pods
        if _pod_reports_node_failure(pod) and (node_name := (pod.get("spec") or {}).get("nodeName"))
    }
    node_loss_nodes = retained_node_loss_nodes | live_node_loss_nodes
    if node_loss_nodes and state not in {
        "COMPLETED",
        "PREEMPTED",
        "NODE_FAIL",
    }:
        state = "NODE_FAIL"

    nodes = {placement["node_name"] for placement in placements.values()}
    nodes.update(pod_evidence.nodes)
    pod_names = {placement["pod_name"] for placement in placements.values()}
    pod_names.update(pod_evidence.pod_names)
    context: KueueExecutionContext = {
        "id": attempt.attempt_id,
        "attempt_id": attempt.attempt_id,
        "workload_type": "kueue",
        "scheduler": "kueue",
        "job_id": f"{attempt.namespace}/{attempt.workload_name}",
        "workload_uid": attempt.workload_uid,
        "admission_key": attempt.admission_key,
        "restart_attempt": attempt.restart_attempt,
        "namespace": attempt.namespace,
        "name": attempt.workload_name,
        "nodes": sorted(nodes),
        "start_time": attempt.start_time,
        "end_time": attempt.end_time,
        "state": state,
        "kueue_state": kueue_state,
        "owner_job_uids": sorted(owner_job_uids),
        "attempt_identity_resolved": attempt.boundary_resolved,
    }
    if placements:
        context["pod_placements"] = [placements[pod_uid] for pod_uid in sorted(placements)]
    if pod_names:
        context["pod_names"] = sorted(pod_names)
    if pod_evidence.pod_nodes:
        context["pod_nodes"] = pod_evidence.pod_nodes
    if node_loss_nodes:
        context["failed_node"] = min(node_loss_nodes)
    elif pod_evidence.failed_node:
        context["failed_node"] = pod_evidence.failed_node
    elif retained_exit_nodes:
        context["failed_node"] = min(retained_exit_nodes)

    if evidence.retained_coverage:
        context["history_complete"] = any(
            _history_complete(attempt, coverage) for coverage in evidence.retained_coverage
        )
    if workload is None:
        return context

    spec = workload.get("spec") or {}
    if queue_name := spec.get("queueName"):
        context["queue_name"] = queue_name
    if isinstance(spec.get("priority"), int):
        context["priority_class"] = spec["priority"]
    gpu_count = _workload_gpu_count(workload)
    if gpu_count is not None:
        context["gpu_count"] = gpu_count
    if time_limit := _extract_time_limit(workload, jobs_by_name):
        context["time_limit"] = time_limit
    labels = (workload.get("metadata") or {}).get("labels") or {}
    for key in (
        "user",
        "username",
        "kueue.x-k8s.io/user",
        "app.kubernetes.io/created-by",
    ):
        if value := labels.get(key):
            context["user_name"] = value
            break
    return context


def _parse_never_admitted_workload(
    workload: dict,
    now: datetime,
) -> KueueExecutionContext | None:
    """Build the explicit live-only context for a never-admitted Workload."""
    metadata = workload.get("metadata") or {}
    uid = metadata.get("uid")
    name = metadata.get("name")
    created = _parse_dt(metadata.get("creationTimestamp"))
    if not uid or not name or created is None:
        return None
    namespace = metadata.get("namespace") or "default"
    conditions = (workload.get("status") or {}).get("conditions") or []
    context: KueueExecutionContext = {
        "id": f"kueue:pre:v1:{uid}",
        "attempt_id": f"kueue:pre:v1:{uid}",
        "workload_type": "kueue",
        "scheduler": "kueue",
        "job_id": f"{namespace}/{name}",
        "workload_uid": uid,
        "namespace": namespace,
        "name": name,
        "nodes": [],
        "start_time": created,
        "end_time": now,
        "state": _service_state(
            _derive_state(conditions),
            workload,
            [],
            {},
        ),
        "kueue_state": _derive_state(conditions),
        "attempt_identity_resolved": False,
        "owner_job_uids": [],
    }
    return context


def _parse_workload(
    workload: dict,
    pods: list[dict],
    owner_jobs: dict[str, dict],
    now: datetime,
) -> KueueExecutionContext | None:
    """Compatibility wrapper around the shared attempt/evidence builder."""
    jobs = _jobs_for_workload(workload, owner_jobs)
    attempts = attempts_from_live_resources(
        workload,
        pods,
        jobs,
        now=now,
    )
    if not attempts:
        if _workload_condition_is_true(workload, _COND_ADMITTED):
            return None
        return _parse_never_admitted_workload(workload, now)
    attempt = attempts[-1]
    evidence = _AttemptEvidence(
        attempts=[attempt],
        workload=workload,
        pods={
            uid: pod
            for pod in pods
            if (uid := (pod.get("metadata") or {}).get("uid"))
            and uid in attempt.pod_uids
        },
        jobs={
            uid: job
            for job in jobs
            if (uid := (job.get("metadata") or {}).get("uid"))
            and uid in attempt.job_uids
        },
    )
    return _context_from_attempt_evidence(evidence)


def _split_csv(value: str) -> list[str]:
    """CSV-split a string, stripping whitespace and dropping empty entries."""
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _history_attempt_state(
    attempt: WorkloadAttempt,
) -> tuple[str, str]:
    terminal = attempt.terminal
    if terminal is None:
        return ("RUNNING", "RUNNING") if attempt.open else ("UNKNOWN", "UNKNOWN")
    terminal_type = terminal["type"]
    reason = str(terminal.get("reason") or "")
    if terminal["kind"] == "JobSet":
        if terminal_type == "Completed":
            return "COMPLETED", "FINISHED_SUCCEEDED"
        return "FAILED", "FINISHED_FAILED"
    if terminal["kind"] == "Job":
        if terminal_type == "Complete":
            return "COMPLETED", "FINISHED_SUCCEEDED"
        if reason == "DeadlineExceeded":
            return "TIMEOUT", "FINISHED_FAILED"
        return "FAILED", "FINISHED_FAILED"
    if terminal_type == "Finished":
        return ("FAILED", "FINISHED_FAILED") if "fail" in reason.lower() else ("COMPLETED", "FINISHED_SUCCEEDED")
    if reason.lower() == "preempted":
        return "PREEMPTED", "EVICTED_PREEMPTED"
    if reason.lower() == "nodefailures":
        return "NODE_FAIL", "EVICTED_NODEFAILURES"
    return "EVICTED", (f"EVICTED_{reason.upper()}" if reason else "EVICTED")


def _history_complete(
    attempt: WorkloadAttempt,
    coverage: dict,
) -> bool:
    if not attempt.boundary_resolved:
        return False
    required_sources = {"kueue-workloads", "jobs", "pods"}
    if attempt.owner_kind == "JobSet":
        required_sources.add("jobsets")
    disabled = coverage.get("disabledWatches") or {}
    sources = {item.get("sourceName"): item for item in coverage.get("sources") or []}
    for source_name in required_sources:
        if source_name in disabled:
            return False
        source = sources.get(source_name)
        if source is None:
            return False
        coverage_start = _parse_dt(source.get("coverageStartedAt"))
        checkpoint = _parse_dt(source.get("checkpointedAt"))
        if (
            coverage_start is None
            or checkpoint is None
            or coverage_start > attempt.start_time
            or checkpoint < attempt.end_time
        ):
            return False
    for gap in coverage.get("gaps") or []:
        if gap.get("sourceName") not in required_sources:
            continue
        gap_start = _parse_dt(gap.get("startedAt"))
        gap_end = _parse_dt(gap.get("recoveredAt"))
        if gap_start is None:
            return False
        if gap_start <= attempt.end_time and (gap_end is None or gap_end >= attempt.start_time):
            return False
    return True


class KueueExecutionContextAdapter:
    """ExecutionContextAdapter that talks to a Kubernetes cluster over the
    Kubernetes API.

    `kubeconfig` and `context` select a kubeconfig file and context; both are
    optional and when unset this falls back to the default kubeconfig
    (KUBECONFIG env var, ~/.kube/config) or, if neither resolves, in-cluster
    service-account credentials. `namespace` selects a single namespace; the
    empty string queries every namespace. `workload_api_version` pins the
    Kueue Workload CRD's `<group>/<version>` the adapter reads (see
    `_DEFAULT_WORKLOAD_API_VERSION` above).
    """

    def __init__(
        self,
        kubeconfig: str | None = None,
        context: str | None = None,
        namespace: str | None = None,
        timeout_seconds: int = 60,
        placement_records_enabled: bool = False,
        placement_reader: PlacementReader | None = None,
        event_max_age_seconds: int = 3600,
        prometheus_placement_resolver: PlacementResolver | None = None,
        workload_api_version: str = _DEFAULT_WORKLOAD_API_VERSION,
    ):
        self.kubeconfig = kubeconfig
        self.context = context
        self.namespace = namespace or ""
        self.timeout_seconds = timeout_seconds
        self.workload_api_version = workload_api_version
        self._kubernetes = KubernetesClient(
            kubeconfig=kubeconfig,
            context=context,
            timeout_seconds=timeout_seconds,
        )
        if not isinstance(placement_records_enabled, bool):
            raise ValueError("placement_records_enabled must be a bool")
        self.placement_records_enabled = placement_records_enabled
        self.placement_reader = placement_reader
        self.history_reader = placement_reader if callable(getattr(placement_reader, "find_histories", None)) else None
        self.event_max_age_seconds = event_max_age_seconds
        self.prometheus_placement_resolver = prometheus_placement_resolver

    # --- public protocol ---------------------------------------------------

    def discover(self) -> list[KueueExecutionContext]:
        """No-runtime-selector entrypoint.

        Like the Slurm adapter, there is no defensible "all workloads" default
        scope — querying every workload across every namespace produces an
        unbounded result. Callers use `discover_by_ids` or
        `discover_by_time_range` to pick a window.
        """
        return []

    def discover_by_ids(self, ids: list[str]) -> list[KueueExecutionContext]:
        """Look up workloads by `<namespace>/<name>` (or bare `<name>` when the
        adapter is namespace-scoped).

        Returns the same list shape `discover_by_time_range` does. Missing
        workloads are silently skipped — the Kubernetes API returns 404 for
        unknown names.
        """
        workloads: list[dict] = []
        retained_evidence: list[_AttemptEvidence] = []
        for ref in ids:
            ns, _, name = ref.partition("/")
            if not name:
                # `<name>` without a namespace is ambiguous against multi-tenant clusters.
                if not self.namespace:
                    raise ValueError(
                        f"kueue ECA --ids: {ref!r} is missing a namespace; "
                        f"use `<namespace>/<name>` or configure eca.kueue.namespace"
                    )
                ns, name = self.namespace, ref
            histories = []
            if self.history_reader is not None:
                histories = self.history_reader.find_histories(
                    ns,
                    "Workload",
                    name=name,
                )
                if histories:
                    retained_evidence.extend(
                        self._evidence_from_histories(
                            histories,
                            workload_name=name,
                        )
                    )
            try:
                obj = self._kubernetes.get_one(
                    "Workload",
                    name,
                    namespace=ns,
                    api_version=self.workload_api_version,
                )
            except KubernetesClientError as error:
                if histories:
                    logger.warning(f"live Workload lookup failed for {ns}/{name}; using retained history")
                    continue
                if error.status == 404:
                    continue
                raise
            if obj is not None:
                workloads.append(obj)
        return self._finalize(workloads, retained_evidence)

    @staticmethod
    def _resource_from_observation(observation: dict) -> dict:
        metadata = dict(observation.get("metadata") or {})
        selected = observation.get("selectedMetadata") or {}
        for metadata_field in ("labels", "annotations"):
            if selected.get(metadata_field):
                metadata[metadata_field] = selected[metadata_field]
        return {
            "apiVersion": observation.get("apiVersion") or "",
            "kind": observation.get("kind") or "",
            "metadata": metadata,
            "spec": observation.get("spec") or {},
            "status": observation.get("status") or {},
        }

    def _evidence_from_histories(
        self,
        histories: list[dict],
        *,
        workload_name: str | None,
    ) -> list[_AttemptEvidence]:
        assert self.placement_reader is not None
        now = datetime.now(tz=timezone.utc)
        evidence_items: list[_AttemptEvidence] = []
        for history in histories:
            workloads = {
                metadata["uid"]: metadata["name"]
                for observation in history.get("observations") or []
                if observation.get("kind") == "Workload"
                and (metadata := observation.get("metadata") or {}).get("uid")
                and metadata.get("name")
                and (workload_name is None or metadata["name"] == workload_name)
            }
            history_placements = [
                PlacementRecord.from_resource(resource) for resource in history.get("placements") or []
            ]
            resources_by_uid = {
                metadata["uid"]: self._resource_from_observation(observation)
                for observation in history.get("observations") or []
                if (metadata := observation.get("metadata") or {}).get("uid")
            }
            for retained_name in sorted(set(workloads.values())):
                for attempt in attempts_from_workload_history(
                    history,
                    workload_name=retained_name,
                    now=now,
                ):
                    evidence_items.append(
                        _AttemptEvidence(
                            attempts=[attempt],
                            pods={uid: resources_by_uid[uid] for uid in attempt.pod_uids if uid in resources_by_uid},
                            jobs={uid: resources_by_uid[uid] for uid in attempt.job_uids if uid in resources_by_uid},
                            placements={placement.pod_uid: placement for placement in history_placements},
                            retained_coverage=[history.get("coverage") or {}],
                        )
                    )
        return evidence_items

    @staticmethod
    def _combine_attempt_evidence(
        evidence_items: list[_AttemptEvidence],
    ) -> list[_AttemptEvidence]:
        by_attempt_id: dict[str, _AttemptEvidence] = {}
        for evidence in evidence_items:
            attempt_id = evidence.attempt.attempt_id
            if attempt_id in by_attempt_id:
                by_attempt_id[attempt_id].merge(evidence)
            else:
                by_attempt_id[attempt_id] = evidence
        return [
            by_attempt_id[attempt_id]
            for attempt_id in sorted(
                by_attempt_id,
                key=lambda key: (
                    by_attempt_id[key].attempt.start_time,
                    key,
                ),
            )
        ]

    @staticmethod
    def _context_from_history_attempt(
        attempt: WorkloadAttempt,
        placements: list[PlacementRecord],
        coverage: dict,
    ) -> KueueExecutionContext:
        return _context_from_attempt_evidence(
            _AttemptEvidence(
                attempts=[attempt],
                placements={
                    placement.pod_uid: placement for placement in placements if placement.pod_uid in attempt.pod_uids
                },
                retained_coverage=[coverage],
            )
        )

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[KueueExecutionContext]:
        """List workloads, then filter by (start_time, end_time) intersection.

        Kueue's API does NOT support server-side time filtering on Workloads,
        so we list and filter client-side. `nodelist` entries may be CSVs of
        plain node names; a workload survives only if its nodes intersect.
        `start`/`end` must be tz-aware to compare against parsed condition
        times.
        """
        assert_aware(start, "discover_by_time_range.start")
        assert_aware(end, "discover_by_time_range.end")
        histories = []
        retained_evidence: list[_AttemptEvidence] = []
        if self.history_reader is not None:
            if not self.namespace:
                raise ValueError("kueue retained-history time-range discovery requires eca.kueue.namespace")
            histories = self.history_reader.find_histories(
                self.namespace,
                "Workload",
            )
            retained_evidence = self._evidence_from_histories(
                histories,
                workload_name=None,
            )
        try:
            listing = self._kubernetes.list(
                "Workload",
                namespace=self.namespace or None,
                api_version=self.workload_api_version,
            )
        except KubernetesClientError:
            if not histories:
                raise
            logger.warning("live Workload list failed; using retained history")
            listing = []
        # parse only the timestamps each workload object
        # already carries (conditions + creationTimestamp).
        now = datetime.now(tz=timezone.utc)
        survivors: list[dict] = []
        for wl in listing:
            window = _workload_window(wl, now=now)
            if window is None:
                continue
            wl_start, wl_end = window
            if wl_start <= end and wl_end >= start:
                survivors.append(wl)
        # Full parse (which fetches pods + owner Jobs) only on survivors.
        contexts = self._finalize(survivors, retained_evidence)
        filtered = [context for context in contexts if context["start_time"] <= end and context["end_time"] >= start]
        if not nodelist:
            return filtered
        wanted: set[str] = set()
        for entry in nodelist:
            wanted.update(_split_csv(str(entry)))
        if not wanted:
            return filtered
        return [c for c in filtered if wanted.intersection(c["nodes"])]

    def _fetch_pods_for(self, workloads: list[dict]) -> list[dict]:
        """Fetch all pods that might belong to any workload in `workloads`.

        The pod label selector depends on the Workload's owner kind (see
        `_pod_selector_for`):

        - batch/v1 Job owner →
          `batch.kubernetes.io/controller-uid=<job.uid>`
        - JobSet owner →
          `jobset.sigs.k8s.io/jobset-name=<jobset.name>` (matches every
          pod across every child Job of every replicatedJob)
        """
        pods: list[dict] = []
        for wl in workloads:
            selector = _pod_selector_for(wl)
            if selector is None:
                continue
            namespace, label_key, label_value = selector
            pods.extend(
                self._kubernetes.list(
                    "Pod",
                    namespace=namespace or None,
                    label_selector=f"{label_key}={label_value}",
                    api_version="v1",
                )
            )
        return pods

    def _fetch_owner_jobs(self, workloads: list[dict]) -> dict[str, dict]:
        """Fetch direct batch/v1 Jobs and JobSet child Jobs, keyed `<ns>/<name>`.

        Direct Jobs are fetched by name. JobSet child Jobs are selected by
        `jobset.sigs.k8s.io/jobset-name` so their failure reasons and deadlines
        are available for state normalization.
        """
        out: dict[str, dict] = {}
        seen: set[tuple[str, str, str]] = set()
        for wl in workloads:
            meta = wl.get("metadata") or {}
            ns = meta.get("namespace") or ""
            for owner in meta.get("ownerReferences") or []:
                kind = owner.get("kind")
                name = owner.get("name")
                if kind not in {"Job", "JobSet"} or not name:
                    continue
                key = (kind, ns, name)
                if key in seen:
                    continue
                seen.add(key)
                if kind == "Job":
                    try:
                        out[f"{ns}/{name}"] = self._kubernetes.get_one(
                            "Job",
                            name,
                            namespace=ns,
                            api_version="batch/v1",
                        )
                    except KubernetesClientError as error:
                        if error.status != 404:
                            raise
                    continue

                child_jobs = self._kubernetes.list(
                    "Job",
                    namespace=ns or None,
                    label_selector=f"{_POD_JOBSET_LABEL}={name}",
                    api_version="batch/v1",
                )
                for job in child_jobs:
                    job_meta = job.get("metadata") or {}
                    job_name = job_meta.get("name")
                    job_namespace = job_meta.get("namespace") or ns
                    if job_name:
                        out[f"{job_namespace}/{job_name}"] = job
        return out

    def _recover_pod_evidence_from_events(
        self,
        workload: dict,
        context: KueueExecutionContext,
    ) -> tuple[list[str], list[str]]:
        """Recover `(nodes, pod_names)` from short-lived Kubernetes Events.

        Queries only the exact owner Job UID(s)' `SuccessfulCreate` Events,
        then the resulting pod names' `Scheduled` Events, via
        `--field-selector` — never a namespace-wide list. Skips entirely once
        `context["end_time"]` is older than `event_max_age_seconds`.
        """
        age = datetime.now(tz=timezone.utc) - context["end_time"]
        if age.total_seconds() > self.event_max_age_seconds:
            return [], []

        namespace = context["namespace"]
        job_uids = set(context.get("owner_job_uids") or [])
        if not job_uids:
            return [], []

        pod_names: set[str] = set()
        # Events are optional evidence, so a failed query is skipped. Every other
        # read decides whether a workload exists and raises on failure.
        for job_uid in job_uids:
            try:
                creation_events = self._kubernetes.list(
                    "Event",
                    namespace=namespace,
                    field_selector=f"involvedObject.uid={job_uid},involvedObject.kind=Job,reason=SuccessfulCreate",
                    api_version="v1",
                    # Pin to the core API (not events.k8s.io/v1, also kind=Event):
                    # both are marked preferred on clusters that serve both, so
                    # omitting this makes discovery ambiguous.
                    prefix="api",
                )
            except KubernetesClientError as error:
                logger.warning(f"event lookup for Job {job_uid} failed; skipping: {error}")
                continue
            pod_names |= _pod_names_from_creation_events(creation_events, context, job_uids)

        nodes: set[str] = set()
        for pod_name in pod_names:
            try:
                scheduled_events = self._kubernetes.list(
                    "Event",
                    namespace=namespace,
                    field_selector=f"involvedObject.name={pod_name},involvedObject.kind=Pod,reason=Scheduled",
                    api_version="v1",
                    prefix="api",
                )
            except KubernetesClientError as error:
                logger.warning(f"event lookup for Pod {pod_name} failed; skipping: {error}")
                continue
            node = _node_from_scheduled_events(scheduled_events, context, namespace, pod_name)
            if node:
                nodes.add(node)

        return sorted(nodes), sorted(pod_names)

    def _fetch_placement_records(self, context: KueueExecutionContext) -> list[PlacementRecord]:
        if not self.placement_records_enabled:
            return []
        if self.placement_reader is None:
            raise RuntimeError("placement records are enabled but no KubernetesDaemon client is configured")
        try:
            return self.placement_reader.list_for_resource(
                context["namespace"],
                context["workload_uid"],
            )
        except Exception as error:
            raise RuntimeError(f"KubernetesDaemon placement lookup failed: {error}") from error

    @staticmethod
    def _merge_placements(context: KueueExecutionContext, additions: list[PodPlacement]) -> None:
        by_uid = {
            placement["pod_uid"]: placement
            for placement in context.get("pod_placements") or []
            if placement.get("pod_uid")
        }
        for placement in additions:
            by_uid.setdefault(placement["pod_uid"], placement)
        if not by_uid:
            return
        placements = sorted(by_uid.values(), key=lambda placement: placement["pod_uid"])
        context["pod_placements"] = placements
        context["nodes"] = sorted(set(context["nodes"]) | {item["node_name"] for item in placements})
        context["pod_names"] = sorted(set(context.get("pod_names") or []) | {item["pod_name"] for item in placements})

    def _merge_prometheus_placements_batched(
        self,
        parsed: list[tuple[dict, KueueExecutionContext]],
        owner_jobs: dict[str, dict] | None = None,
    ) -> None:
        if self.prometheus_placement_resolver is None:
            return

        queries: list[PlacementQuery] = []
        contexts_by_workload_uid: dict[str, KueueExecutionContext] = {}
        latest_resolved: dict[
            str,
            tuple[dict, KueueExecutionContext],
        ] = {}
        for workload, context in parsed:
            if context.get("attempt_identity_resolved") is False:
                continue
            workload_uid = context["workload_uid"]
            previous = latest_resolved.get(workload_uid)
            if previous is None or context["start_time"] > previous[1]["start_time"]:
                latest_resolved[workload_uid] = (workload, context)
        for workload, context in latest_resolved.values():
            workload_uid = context["workload_uid"]
            owner_job_uids = set(context.get("owner_job_uids") or [])
            if not owner_job_uids and owner_jobs is not None:
                owner_job_uids = _owner_job_uids(
                    workload,
                    owner_jobs,
                )
            if not owner_job_uids or not workload_uid:
                continue
            queries.append(
                PlacementQuery(
                    workload_uid=workload_uid,
                    namespace=context["namespace"],
                    owner_job_uids=frozenset(owner_job_uids),
                    start_time=context["start_time"],
                    end_time=context["end_time"],
                )
            )
            contexts_by_workload_uid[workload_uid] = context

        if not queries:
            return
        try:
            placements_by_workload = self.prometheus_placement_resolver.placements_for_many(queries)
        except Exception as error:
            logger.warning(f"batched Prometheus placement query failed: {error}")
            return
        for workload_uid, placements in placements_by_workload.items():
            context = contexts_by_workload_uid.get(workload_uid)
            if context is not None:
                if context.get("attempt_identity_resolved"):
                    verified_pod_uids = {
                        placement["pod_uid"]
                        for placement in context.get("pod_placements") or []
                        if placement.get("pod_uid")
                    }
                    placements = [
                        placement
                        for placement in placements
                        if placement.get("pod_uid")
                        in verified_pod_uids
                    ]
                self._merge_placements(context, placements)

    def _finalize(
        self,
        workloads: list[dict],
        retained_evidence: list[_AttemptEvidence] | None = None,
    ) -> list[KueueExecutionContext]:
        if not workloads and not retained_evidence:
            return []
        pods = self._fetch_pods_for(workloads) if workloads else []
        owner_jobs = self._fetch_owner_jobs(workloads) if workloads else {}
        now = datetime.now(tz=timezone.utc)
        evidence_items = list(retained_evidence or [])
        pending_contexts: list[tuple[dict, KueueExecutionContext]] = []
        workloads_by_uid: dict[str, dict] = {}
        for workload in workloads:
            metadata = workload.get("metadata") or {}
            workload_uid = metadata.get("uid")
            if workload_uid:
                workloads_by_uid[workload_uid] = workload
            selector = _pod_selector_for(workload)
            member_pods = [pod for pod in pods if selector is not None and _pod_matches_selector(pod, selector)]
            jobs = _jobs_for_workload(workload, owner_jobs)
            attempts = attempts_from_live_resources(
                workload,
                member_pods,
                jobs,
                now=now,
            )
            if not attempts:
                if _workload_condition_is_true(
                    workload,
                    _COND_ADMITTED,
                ):
                    continue
                pending = _parse_never_admitted_workload(workload, now)
                if pending is not None:
                    pending_contexts.append((workload, pending))
                continue
            records = self._fetch_placement_records(
                {
                    "namespace": metadata.get("namespace") or "default",
                    "workload_uid": workload_uid,
                }
            )
            pods_by_uid = {
                uid: pod
                for pod in member_pods
                if (uid := (pod.get("metadata") or {}).get("uid"))
            }
            jobs_by_uid = {
                uid: job
                for job in jobs
                if (uid := (job.get("metadata") or {}).get("uid"))
            }
            for attempt in attempts:
                evidence_items.append(
                    _AttemptEvidence(
                        attempts=[attempt],
                        workload=workload,
                        pods={uid: pods_by_uid[uid] for uid in attempt.pod_uids if uid in pods_by_uid},
                        jobs={uid: jobs_by_uid[uid] for uid in attempt.job_uids if uid in jobs_by_uid},
                        placements={record.pod_uid: record for record in records},
                    )
                )

        parsed: list[tuple[dict, KueueExecutionContext]] = []
        retained_only: list[KueueExecutionContext] = []
        for evidence in self._combine_attempt_evidence(evidence_items):
            context = _context_from_attempt_evidence(evidence)
            workload = workloads_by_uid.get(context["workload_uid"])
            if workload is None:
                retained_only.append(context)
            else:
                parsed.append((workload, context))
        parsed.extend(pending_contexts)
        self._merge_prometheus_placements_batched(parsed)

        missing = [
            (workload, ctx)
            for workload, ctx in parsed
            if ctx["kueue_state"] not in {"PENDING", "QUOTA_RESERVED"}
            and (not ctx["nodes"] or not ctx.get("pod_names"))
        ]
        if not missing:
            return [*retained_only, *(ctx for _, ctx in parsed)]

        for workload, ctx in missing:
            metadata = workload.get("metadata") or {}
            namespace = ctx["namespace"]
            recovered_nodes, recovered_pods = self._recover_pod_evidence_from_events(
                workload,
                ctx,
            )
            if recovered_nodes:
                ctx["nodes"] = sorted(set(ctx["nodes"]) | set(recovered_nodes))
            if recovered_pods:
                ctx["pod_names"] = sorted(set(ctx.get("pod_names") or []) | set(recovered_pods))

            unavailable: list[str] = []
            if not ctx.get("pod_names"):
                unavailable.append("pod list")
            if not ctx["nodes"]:
                unavailable.append("node list")
            if unavailable:
                logger.warning(
                    f"workload {namespace}/{metadata.get('name')}: "
                    f"{' and '.join(unavailable)} unavailable from live pods and Kubernetes Events"
                )
        return [*retained_only, *(ctx for _, ctx in parsed)]
