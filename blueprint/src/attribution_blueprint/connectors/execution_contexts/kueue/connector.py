# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Kueue (Kubernetes batch admission queue) ExecutionContextAdapter.

Mirrors the Slurm adapter's shape: shell out to `kubectl` for the raw API
payloads, normalize them into the wire-format `ExecutionContext` TypedDict,
and never let any Kubernetes/Kueue-shaped data structure escape this file.

Wire-format mapping vs. the Slurm adapter:

    Slurm field        Kueue analog
    -----------------  -----------------------------------------------------
    sluid              Workload.metadata.uid (carried as `workload_uid`)
    attempt_id         Workload.metadata.uid (per-attempt: Kueue creates a
                       new Workload per requeue, so its uid is the natural
                       attempt key — analogous to Slurm's SLUID)
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

Pod ownership is resolved via the **standard batch/v1 Job pod labels**,
not the Kueue label. `kueue.x-k8s.io/job-uid` is set by Kueue on the
**Workload** object (carrying the owner Job's UID so callers can link a
Workload back to its Job. The pods are created by the Kubernetes Job 
controller, which stamps the `batch.kubernetes.io/controller-uid=<job.uid>` 
and `batch.kubernetes.io/job-name=<job.name>` labels. 
This adapter pulls the owner Job's UID out of the Workload's 
`ownerReferences` and selects pods with 
`batch.kubernetes.io/controller-uid=<owner_uid>`. 

The JobSet pod label carries only a name, which is unique per namespace
but not cluster-wide, so `_pod_selector_for`/`_pod_matches_selector` 
always scope matching to the Workload's own namespace.

Owner Jobs/JobSet children are looked up by namespace/name, but names are
reusable after deletion. `_jobs_for_workload` confirms the fetched
object's uid against the ownerReference's uid before trusting it as the
real owner.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import NamedTuple, NotRequired, TypedDict

from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kueue.helpers import (
    NODE_LOSS_DISRUPTION_REASONS as _NODE_LOSS_DISRUPTION_REASONS,
    POD_JOBSET_LABEL as _POD_JOBSET_LABEL,
    jobs_for_workload as _jobs_for_workload,
    owner_job_uids as _owner_job_uids,
    pod_matches_selector as _pod_matches_selector,
    pod_node_failure_reason as _pod_node_failure_reason,
    pod_selector_for as _pod_selector_for,
)
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import (
    PlacementQuery,
    PlacementRecord,
    PodPlacement,
)
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.store import (
    PlacementReader,
    PlacementResolver,
)
from attribution_blueprint.types import assert_aware


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
    # Pod names belonging to this Workload at the time of discovery. Empty 
    # list when the owner type isn't supported or all pods have been 
    # garbage-collected.
    pod_names: NotRequired[list[str]]
    pod_placements: NotRequired[list[PodPlacement]]
    # Pod name -> `.spec.nodeName`, for connectors that need per-pod node
    # identity. Unscheduled pods are absent.
    pod_nodes: NotRequired[dict[str, str]]


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

    Delegates to the shared `helpers.pod_node_failure_reason` so this check
    and the placement recorder's copy (which captures the same evidence
    before Pod GC removes it) can never drift apart.
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

    The caller is responsible for narrowing the input list to pods that
    might belong to this workload — typically via the kubectl label
    selector in `_fetch_pods_for`. This function still re-checks each pod
    against `_pod_matches_selector` as a defensive filter (namespace +
    label), since cluster-wide discovery can pool pods across namespaces.

    Pod names are returned alongside the node evidence so downstream
    connectors (e.g. `kubectl_logs`) don't have to re-derive the
    per-owner pod selector and re-fetch the pod list.

    Returns empty evidence when `selector` is None — Workloads without
    a recognised owner have no pods to attribute.
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
    # kubectl response orderings. Node-loss evidence outranks a generic
    # non-zero exit code.
    if node_loss_nodes:
        failed_node = min(node_loss_nodes)
    elif exit_code_nodes:
        failed_node = min(exit_code_nodes)
    else:
        failed_node = None
    return _PodEvidence(sorted(nodes), failed_node, sorted(pod_names), pod_nodes)


def _live_node_loss_nodes(
    pods: list[dict],
    selector: tuple[str, str, str] | None,
) -> set[str]:
    if selector is None:
        return set()
    return {
        node_name
        for pod in pods
        if _pod_matches_selector(pod, selector)
        and (node_name := (pod.get("spec") or {}).get("nodeName"))
        and _pod_reports_node_failure(pod)
    }


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


def _owner_job_failure_reason(workload: dict, owner_jobs: dict[str, dict]) -> str | None:
    """Return the failure reason of the earliest-failing direct or JobSet child Job.

    Mirrors JobSet's own controller convention for resolving multiple child
    failures (kubernetes-sigs/jobset's `pkg/controllers/failure_policy.go`): the
    earliest `Failed`/`FailureTarget` condition by `lastTransitionTime` is
    authoritative.
    """
    earliest_time: datetime | None = None
    earliest_reason: str | None = None
    for job in _jobs_for_workload(workload, owner_jobs):
        conditions = (job.get("status") or {}).get("conditions") or []
        for condition_type in ("Failed", "FailureTarget"):
            condition = _condition(conditions, condition_type)
            if not condition or condition.get("status") != "True":
                continue
            reason = (condition.get("reason") or "").strip()
            if not reason:
                continue
            transition_time = _parse_dt(condition.get("lastTransitionTime"))
            if transition_time is None:
                continue
            if earliest_time is None or transition_time < earliest_time:
                earliest_time = transition_time
                earliest_reason = reason
    return earliest_reason


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
    if _owner_job_failure_reason(workload, owner_jobs) == "DeadlineExceeded":
        return "TIMEOUT"
    return "FAILED"


def _extract_time_limit(workload: dict, owner_jobs: dict[str, dict]) -> int | None:
    """Pull activeDeadlineSeconds as service-facing minutes.

    `owner_jobs` maps `"<namespace>/<name>"` of any batch/v1 Job objects the
    caller fetched alongside the workloads. For a JobSet, the shortest child
    deadline is the effective limit. Partial minutes are rounded up because the
    service's `time_limit` contract uses positive integer minutes. Returns None
    when no matching Job has a deadline.
    """
    deadlines_seconds: list[int] = []
    for job in _jobs_for_workload(workload, owner_jobs):
        deadline = (job.get("spec") or {}).get("activeDeadlineSeconds")
        if isinstance(deadline, int) and not isinstance(deadline, bool) and deadline > 0:
            deadlines_seconds.append(deadline)
    if not deadlines_seconds:
        return None
    shortest_deadline_seconds = min(deadlines_seconds)
    return (shortest_deadline_seconds + 59) // 60


def _non_negative_int(value: object) -> int | None:
    """Coerce Kubernetes resource quantities that should be plain integers.

    Args:
        value: Raw Kubernetes resource value.

    Returns:
        A non-negative integer, or None when no integer is present.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _resource_gpu_count(resources: object) -> int:
    """Return GPU count from one Kubernetes resources map.

    Args:
        resources: Container ``resources.requests`` or ``resources.limits``.

    Returns:
        Sum of GPU-like extended resources in the map.
    """
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
    """Return GPU count requested by one Kubernetes container.

    Args:
        container: Pod template container object.

    Returns:
        Requested GPU count, falling back to limits when requests are absent.
    """
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
    """Return total GPU count requested by a Kueue Workload.

    Args:
        workload: Kueue Workload API object.

    Returns:
        Total GPUs across all pod sets. Returns 0 when the Workload pod sets
        are parseable but no GPU resource is declared, or None when the
        Workload shape is missing.
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
    """Return `(start_time, end_time)` for a Workload using ONLY data carried
    on the Workload object itself — no pod or owner-Job fetches.

    Used by `discover_by_time_range` to filter the cluster-wide listing by
    time window BEFORE issuing per-workload `kubectl get pods` /
    `kubectl get job` round-trips. The full `_parse_workload` computes the
    same window plus pod/node/time-limit evidence; this helper short-circuits
    when the caller only needs to know "does this workload overlap [start,
    end]?". Returns `None` if no start time can be derived (the workload
    would be skipped by the full parse anyway).
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


def _parse_workload(
    workload: dict,
    pods: list[dict],
    owner_jobs: dict[str, dict],
    now: datetime,
) -> KueueExecutionContext | None:
    """Normalize one Workload object into our wire format.

    Returns None when the workload has no UID/name (malformed object) or
    has no parseable start time — without a start the orchestrator can't
    build a valid attributor window, so the workload is skipped with a
    warning, matching the Slurm adapter's behavior for rows with
    unparseable Start/End.
    """
    meta = workload.get("metadata") or {}
    uid = meta.get("uid")
    name = meta.get("name")
    namespace = meta.get("namespace") or "default"
    if not uid or not name:
        logger.warning(f"workload missing uid/name; skipping: {meta!r}")
        return None

    status = workload.get("status") or {}
    conditions = status.get("conditions") or []
    kueue_state = _derive_state(conditions)
    state = _service_state(kueue_state, workload, pods, owner_jobs)

    start_time = _earliest_condition_time(conditions, _COND_ADMITTED)
    end_time = _condition_time(conditions, _COND_FINISHED)
    if end_time is None:
        evicted = _condition_time(conditions, _COND_EVICTED)
        if evicted is not None:
            end_time = evicted
    if start_time is None:
        # Fall back to the workload's creationTimestamp — for never-admitted
        # workloads it's the only timestamp Kubernetes guarantees to populate.
        start_time = _parse_dt(meta.get("creationTimestamp"))

    if start_time is None:
        logger.warning(f"workload {namespace}/{name} has no start time; skipping")
        return None

    if end_time is None:
        # Active workload — substitute now() as the evidence-collection end,
        # mirroring the Slurm adapter's behavior for RUNNING ExecutionContexts. Re-running
        # attribution on a still-active workload will produce a sliding window.
        end_time = now
        logger.debug(f"workload {namespace}/{name} not finished; substituting now() for end_time")

    # `pods` is pooled across every workload in the batch. The JobSet label is
    # only unique within a namespace, so a same-named JobSet elsewhere would
    # otherwise leak its pods (and overwrite `pod_nodes`) into this workload.
    own_pods = [pod for pod in pods if ((pod.get("metadata") or {}).get("namespace") or "default") == namespace]
    evidence = _collect_pod_evidence(own_pods, _pod_selector_for(workload))

    ctx: KueueExecutionContext = {
        "id": uid,
        # V2 accepts scheduler identity as a string, allowing service rules and
        # provenance to distinguish Kueue from other execution-context sources.
        "workload_type": "kueue",
        # BP-internal connector identity. SQLite's `scheduler_type` column
        # reads this in preference to the service-facing `workload_type`.
        "scheduler": "kueue",
        "nodes": evidence.nodes,
        "start_time": start_time,
        "end_time": end_time,
        "job_id": f"{namespace}/{name}",
        "workload_uid": uid,
        # `attempt_id` is the scheduler-agnostic per-attempt key used by
        # the sqlite output's unique-row constraint. Kueue creates a new
        # Workload per requeue, so its uid is the natural attempt key.
        "attempt_id": uid,
        "namespace": namespace,
        "state": state,
        "kueue_state": kueue_state,
        "name": name,
    }

    spec = workload.get("spec") or {}
    queue_name = spec.get("queueName")
    if queue_name:
        ctx["queue_name"] = queue_name
    priority = spec.get("priority")
    if isinstance(priority, int):
        ctx["priority_class"] = priority
    gpu_count = _workload_gpu_count(workload)
    if gpu_count is not None:
        ctx["gpu_count"] = gpu_count

    if evidence.failed_node:
        ctx["failed_node"] = evidence.failed_node

    if len(evidence.pod_names) > 0:
        ctx["pod_names"] = evidence.pod_names
    if len(evidence.pod_nodes) > 0:
        ctx["pod_nodes"] = evidence.pod_nodes
    pod_placements = _live_pod_placements(own_pods, _pod_selector_for(workload))
    if pod_placements:
        ctx["pod_placements"] = pod_placements

    time_limit = _extract_time_limit(workload, owner_jobs)
    if time_limit is not None:
        ctx["time_limit"] = time_limit

    labels = meta.get("labels") or {}
    for key in ("user", "username", "kueue.x-k8s.io/user", "app.kubernetes.io/created-by"):
        value = labels.get(key)
        if value:
            ctx["user_name"] = value
            break

    return ctx


def _split_csv(value: str) -> list[str]:
    """CSV-split a string, stripping whitespace and dropping empty entries."""
    return [piece.strip() for piece in value.split(",") if piece.strip()]


class KueueExecutionContextAdapter:
    """ExecutionContextAdapter that talks to a Kubernetes cluster via kubectl.

    `kubectl_path` is the binary to invoke (default "kubectl"). `kubeconfig`
    and `context` map to `--kubeconfig`/`--context`; both are optional and
    when unset kubectl falls back to its default discovery (KUBECONFIG env
    var, ~/.kube/config, in-cluster service account). `namespace` selects a
    single namespace; the empty string queries every namespace via `-A`.
    """

    def __init__(
        self,
        kubectl_path: str = "kubectl",
        kubeconfig: str | None = None,
        context: str | None = None,
        namespace: str | None = None,
        timeout_seconds: int = 60,
        placement_records_enabled: bool = False,
        placement_reader: PlacementReader | None = None,
        event_max_age_seconds: int = 3600,
        prometheus_placement_resolver: PlacementResolver | None = None,
    ):
        self.kubectl_path = kubectl_path
        self.kubeconfig = kubeconfig
        self.context = context
        self.namespace = namespace or ""
        self.timeout_seconds = timeout_seconds
        if not isinstance(placement_records_enabled, bool):
            raise ValueError("placement_records_enabled must be a bool")
        self.placement_records_enabled = placement_records_enabled
        self.placement_reader = placement_reader
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
        workloads are silently skipped — `kubectl get workload <name>` exits
        non-zero for unknown names.
        """
        workloads: list[dict] = []
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
            obj = self._kubectl_get_one("workload", name, namespace=ns)
            if obj is not None:
                workloads.append(obj)
        return self._finalize(workloads)

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[KueueExecutionContext]:
        """List workloads, then filter by (start_time, end_time) intersection.

        Kueue's API does NOT support server-side time filtering on Workloads,
        so we list and filter client-side — same pattern as a sacct call
        without `--jobs`. The `nodelist` argument intersects against the
        observed pod nodes; empty intersection drops the workload. Each list
        entry may be a CSV of plain k8s node names — Kueue parses comma lists
        here rather than at the CLI layer so the orchestrator's contract stays
        scheduler-agnostic.

        `start` and `end` must be tz-aware. Naive datetimes would crash the
        per-workload comparison against (tz-aware) parsed condition times.
        """
        assert_aware(start, "discover_by_time_range.start")
        assert_aware(end, "discover_by_time_range.end")
        listing = self._kubectl_list("workloads", namespace=self.namespace or None)
        # parse only the timestamps each workload object
        # already carries (conditions + creationTimestamp).
        from datetime import timezone

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
        filtered = self._finalize(survivors)
        if not nodelist:
            return filtered
        wanted: set[str] = set()
        for entry in nodelist:
            wanted.update(_split_csv(str(entry)))
        if not wanted:
            return filtered
        return [c for c in filtered if wanted.intersection(c["nodes"])]

    def _kubectl_base(self) -> list[str]:
        cmd = [self.kubectl_path]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        if self.context:
            cmd += ["--context", self.context]
        return cmd

    def _run_json(self, args: list[str], *, required: bool = False) -> dict | None:
        cmd = self._kubectl_base() + args + ["-o", "json"]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                timeout=self.timeout_seconds,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            if required:
                raise RuntimeError(
                    f"required kubectl query failed with exit {e.returncode}: {(e.stderr or '').strip()}"
                ) from e
            logger.warning(f"kubectl exited {e.returncode}: {(e.stderr or '').strip() or '<no stderr>'}")
            return None
        except subprocess.TimeoutExpired:
            if required:
                raise RuntimeError(f"required kubectl query timed out after {self.timeout_seconds}s")
            logger.warning(f"kubectl timed out after {self.timeout_seconds}s: {cmd!r}")
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            if required:
                raise RuntimeError(f"required kubectl query returned invalid JSON: {e}") from e
            logger.warning(f"kubectl returned invalid JSON: {e}")
            return None

    def _kubectl_get_one(self, kind: str, name: str, *, namespace: str) -> dict | None:
        args = ["get", kind, name]
        if namespace:
            args += ["-n", namespace]
        return self._run_json(args)

    def _kubectl_list(self, kind: str, *, namespace: str | None, field_selector: str | None = None) -> list[dict]:
        args = ["get", kind]
        if field_selector:
            args += ["--field-selector", field_selector]
        if namespace:
            args += ["-n", namespace]
        else:
            args += ["-A"]
        payload = self._run_json(args)
        if not payload:
            return []
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        return items

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
            args = ["get", "pods", "-l", f"{label_key}={label_value}"]
            if namespace:
                args += ["-n", namespace]
            else:
                args += ["-A"]
            payload = self._run_json(args)
            if not payload:
                continue
            items = payload.get("items")
            if isinstance(items, list):
                pods.extend(items)
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
                    job = self._kubectl_get_one("job", name, namespace=ns)
                    if job is not None:
                        out[f"{ns}/{name}"] = job
                    continue

                args = ["get", "jobs", "-l", f"{_POD_JOBSET_LABEL}={name}"]
                if ns:
                    args += ["-n", ns]
                else:
                    args += ["-A"]
                payload = self._run_json(args)
                for job in (payload or {}).get("items") or []:
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
        owner_jobs: dict[str, dict],
    ) -> tuple[list[str], list[str]]:
        """Recover `(nodes, pod_names)` from short-lived Kubernetes Events.

        Queries only the exact owner Job UID(s)' `SuccessfulCreate` Events,
        then only the resulting pod names' `Scheduled` Events, via
        `--field-selector` — never a namespace-wide Event list. This keeps
        the RBAC/scale/retention footprint bounded to this one workload
        one `kubectl get events` call per owner Job UID and per pod name.
        Skips entirely (no calls) once `context["end_time"]` is older than
        `event_max_age_seconds`.
        """
        age = datetime.now(tz=timezone.utc) - context["end_time"]
        if age.total_seconds() > self.event_max_age_seconds:
            return [], []

        namespace = context["namespace"]
        job_uids = _owner_job_uids(workload, owner_jobs)
        if not job_uids:
            return [], []

        pod_names: set[str] = set()
        for job_uid in job_uids:
            creation_events = self._kubectl_list(
                "events",
                namespace=namespace,
                field_selector=f"involvedObject.uid={job_uid},involvedObject.kind=Job,reason=SuccessfulCreate",
            )
            pod_names |= _pod_names_from_creation_events(creation_events, context, job_uids)

        nodes: set[str] = set()
        for pod_name in pod_names:
            scheduled_events = self._kubectl_list(
                "events",
                namespace=namespace,
                field_selector=f"involvedObject.name={pod_name},involvedObject.kind=Pod,reason=Scheduled",
            )
            node = _node_from_scheduled_events(scheduled_events, context, namespace, pod_name)
            if node:
                nodes.add(node)

        return sorted(nodes), sorted(pod_names)

    def _fetch_placement_records(self, context: KueueExecutionContext) -> list[PlacementRecord]:
        if not self.placement_records_enabled:
            return []
        if self.placement_reader is None:
            raise RuntimeError("placement records are enabled but no recorder client is configured")
        try:
            return self.placement_reader.list_for_workload(
                context["namespace"],
                context["workload_uid"],
            )
        except Exception as error:
            raise RuntimeError(f"placement recorder lookup failed: {error}") from error

    def _merge_placement_records(
        self,
        context: KueueExecutionContext,
        live_node_loss_nodes: set[str] | None = None,
    ) -> None:
        placements: list[PodPlacement] = []
        retained_node_loss_nodes: set[str] = set()
        exit_code_nodes: set[str] = set()
        for record in self._fetch_placement_records(context):
            placements.append(
                {
                    "pod_uid": record.pod_uid,
                    "pod_name": record.pod_name,
                    "node_name": record.node_name,
                    "source": "placement_record",
                }
            )
            reason = record.node_failure_reason
            if reason == "NodeLost" or reason in _NODE_LOSS_DISRUPTION_REASONS:
                retained_node_loss_nodes.add(record.node_name)
            elif reason:
                exit_code_nodes.add(record.node_name)
        self._merge_placements(context, placements)
        # PlacementRecords are the only post-GC source for Pod status evidence.
        # Preserve the same precedence as `_service_state`.
        if retained_node_loss_nodes and context.get("state") not in {"COMPLETED", "PREEMPTED", "NODE_FAIL"}:
            context["state"] = "NODE_FAIL"
        # Explicit node-loss evidence outranks generic exit-code evidence,
        # regardless of whether it came from a live Pod or a retained record.
        node_loss_nodes = set(live_node_loss_nodes or ()) | retained_node_loss_nodes
        if node_loss_nodes:
            context["failed_node"] = min(node_loss_nodes)
        elif exit_code_nodes and not context.get("failed_node"):
            context["failed_node"] = min(exit_code_nodes)

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
        owner_jobs: dict[str, dict],
    ) -> None:
        if self.prometheus_placement_resolver is None:
            return

        queries: list[PlacementQuery] = []
        contexts_by_workload_uid: dict[str, KueueExecutionContext] = {}
        for workload, context in parsed:
            owner_job_uids = _owner_job_uids(workload, owner_jobs)
            workload_uid = context["workload_uid"]
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
                self._merge_placements(context, placements)

    def _finalize(self, workloads: list[dict]) -> list[KueueExecutionContext]:
        if not workloads:
            return []
        pods = self._fetch_pods_for(workloads)
        owner_jobs = self._fetch_owner_jobs(workloads)
        now = datetime.now(tz=timezone.utc)
        parsed: list[tuple[dict, KueueExecutionContext]] = []
        for wl in workloads:
            ctx = _parse_workload(wl, pods, owner_jobs, now=now)
            if ctx is not None:
                live_node_loss_nodes = _live_node_loss_nodes(pods, _pod_selector_for(wl))
                self._merge_placement_records(ctx, live_node_loss_nodes)
                parsed.append((wl, ctx))
        self._merge_prometheus_placements_batched(parsed, owner_jobs)

        missing = [
            (workload, ctx)
            for workload, ctx in parsed
            if ctx["kueue_state"] not in {"PENDING", "QUOTA_RESERVED"}
            and (not ctx["nodes"] or not ctx.get("pod_names"))
        ]
        if not missing:
            return [ctx for _, ctx in parsed]

        for workload, ctx in missing:
            metadata = workload.get("metadata") or {}
            namespace = ctx["namespace"]
            recovered_nodes, recovered_pods = self._recover_pod_evidence_from_events(workload, ctx, owner_jobs)
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
        return [ctx for _, ctx in parsed]
