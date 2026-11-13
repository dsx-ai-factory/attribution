# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KAI Scheduler ExecutionContextAdapter.

Emits one workload-level context per PodGroup, with these KAI specifics:

    ExecutionContext   KAI source
    -----------------  -----------------------------------------------------
    id / attempt_id    PodGroup.metadata.uid. Identifies one PodGroup
                       incarnation (a recreated PodGroup gets a new uid);
                       it is not a scheduling attempt.
    job_id             "<namespace>/<name>" of the PodGroup
    start_time         Earliest creation time of the PodGroup or a member Pod
    end_time           Terminal time of the owning workload; now() while
                       the owner is active or has no terminal timestamp.
                       Pods never determine it.
    nodes              Union of .spec.nodeName across the member Pods
    state              From the owning workload: COMPLETED, FAILED,
                       TIMEOUT (failed with DeadlineExceeded or
                       PreRunningDeadlineExceeded, or a PyTorchJob whose
                       Failed condition reports an exceeded deadline),
                       CANCELLED (a stopped RayJob), or RUNNING while the
                       owner has no terminal signal, including while
                       suspended or when a Deployment rollout hits
                       ProgressDeadlineExceeded. `owner_state` keeps the
                       owner's own wording. A member Pod that reports node
                       loss sets state to NODE_FAIL and sets `failed_node`;
                       a completed one stays COMPLETED without one.

Member Pods are joined on the `pod-group-name` annotation plus namespace,
then validated against the owning workload UID; refer to `_member_pods` for
why membership intentionally does not use timestamps as a discriminator.

The owner comes from the `kai.scheduler/top-owner-metadata` annotation, or
else the PodGroup's controller ownerReference. Supported owner kinds from
the Karta catalog are `batch` Job, `jobset.x-k8s.io` JobSet,
`leaderworkerset.x-k8s.io` LeaderWorkerSet, `kubeflow.org` PyTorchJob,
`ray.io` RayJob, `apps` Deployment, and `nvidia.com`
DynamoGraphDeployment; the owner must still exist with the same UID.

Discovery reads only the owner resource kind referenced by each PodGroup
(`get` on that owner resource, plus `list` on `batch/v1` Jobs when the
owner is a JobSet). A failed Kubernetes read raises; only NotFound is
treated as absence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NotRequired, TypedDict

from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kai.helpers import (
    POD_GROUP_ANNOTATION as _POD_GROUP_ANNOTATION,
    controller_owner as _controller_owner,
    resolve_owner as _resolve_owner,
    top_owner as _top_owner,
)
from attribution_blueprint.kubernetes.client import (
    KubernetesClient,
    KubernetesClientError,
)
from attribution_blueprint.kubernetes.daemon.models import PlacementRecord, PodPlacement
from attribution_blueprint.kubernetes.daemon.store import PlacementReader
from attribution_blueprint.kubernetes.helpers import (
    NODE_LOSS_DISRUPTION_REASONS,
    earliest_job_failure_reason,
    job_time_limit_minutes,
    pod_node_failure_reason,
)
from attribution_blueprint.types import assert_aware

_PODGROUP_RESOURCE = "PodGroup"
_PODGROUP_GROUP = "scheduling.run.ai"
_JOB_RESOURCE = "Job"
_JOB_API_VERSION = "batch/v1"
_JOBSET_RESOURCE = "JobSet"
_JOBSET_GROUP = "jobset.x-k8s.io"
_JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"
_LWS_RESOURCE = "LeaderWorkerSet"
_LWS_GROUP = "leaderworkerset.x-k8s.io"
_LWS_API_VERSION = "v1"
_PYTORCHJOB_RESOURCE = "PyTorchJob"
_PYTORCHJOB_GROUP = "kubeflow.org"
_PYTORCHJOB_API_VERSION = "v1"
_RAYJOB_RESOURCE = "RayJob"
_RAYJOB_GROUP = "ray.io"
_RAYJOB_API_VERSION = "v1"
_DEPLOYMENT_RESOURCE = "Deployment"
_DEPLOYMENT_API_VERSION = "apps/v1"
_DYNAMO_GRAPH_DEPLOYMENT_RESOURCE = "DynamoGraphDeployment"
_DYNAMO_GROUP = "nvidia.com"
_DYNAMO_API_VERSION = "v1alpha1"

_OWNER_LOOKUPS: dict[str, tuple[str, dict[str, str]]] = {
    _JOB_RESOURCE: ("batch", {"api_version": _JOB_API_VERSION}),
    _JOBSET_RESOURCE: (_JOBSET_GROUP, {"group": _JOBSET_GROUP}),
    _LWS_RESOURCE: (_LWS_GROUP, {"group": _LWS_GROUP, "api_version": _LWS_API_VERSION}),
    _PYTORCHJOB_RESOURCE: (
        _PYTORCHJOB_GROUP,
        {"group": _PYTORCHJOB_GROUP, "api_version": _PYTORCHJOB_API_VERSION},
    ),
    _RAYJOB_RESOURCE: (_RAYJOB_GROUP, {"group": _RAYJOB_GROUP, "api_version": _RAYJOB_API_VERSION}),
    _DEPLOYMENT_RESOURCE: ("apps", {"api_version": _DEPLOYMENT_API_VERSION}),
    _DYNAMO_GRAPH_DEPLOYMENT_RESOURCE: (
        _DYNAMO_GROUP,
        {"group": _DYNAMO_GROUP, "api_version": _DYNAMO_API_VERSION},
    ),
}

_COMPLETED_OWNER_STATE: dict[str, str] = {
    _JOB_RESOURCE: "Complete",
    _JOBSET_RESOURCE: "Completed",
    _PYTORCHJOB_RESOURCE: "Succeeded",
    _RAYJOB_RESOURCE: "Succeeded",
}

_DEADLINE_FAILURE_REASONS = frozenset({"DeadlineExceeded", "PreRunningDeadlineExceeded"})
_PYTORCHJOB_DEADLINE_MESSAGES = (
    "active longer than specified deadline",
    "deadline exceeded",
)
_RAYJOB_ACTIVE_JOB_STATUSES = frozenset({"PENDING", "RUNNING"})
_RAYJOB_TERMINAL_JOB_STATUSES = frozenset({"SUCCEEDED", "FAILED", "STOPPED"})
_RAYJOB_RETRY_OR_SUSPENDED_STATUSES = frozenset({"Retrying", "Suspended", "Suspending"})
_RAYJOB_TERMINAL_DEPLOYMENT_STATUSES = frozenset({"Complete", "Failed"})
_RAYJOB_TERMINAL_CONDITION_TYPES = frozenset({"Failed", "Succeeded", "Completed"})

_SHAPE_OWNED = "owned"
_SHAPE_HIERARCHICAL = "hierarchical"

_CHILD_CONTROLLER_KINDS: dict[str, frozenset[str]] = {
    _DEPLOYMENT_RESOURCE: frozenset({"ReplicaSet"}),
    _LWS_RESOURCE: frozenset({"StatefulSet"}),
    _RAYJOB_RESOURCE: frozenset({"RayCluster", _JOB_RESOURCE}),
    _DYNAMO_GRAPH_DEPLOYMENT_RESOURCE: frozenset(
        {
            "DynamoComponentDeployment",
            _LWS_RESOURCE,
            "PodGang",
            "PodClique",
            "PodCliqueSet",
            "PodCliqueScalingGroup",
            "ReplicaSet",
            "StatefulSet",
        }
    ),
}


class KaiExecutionContext(TypedDict):
    """Adapter-specific ExecutionContext shape.

    Same spine as `attribution_blueprint.types.ExecutionContext`; the keys
    below the spine are KAI-specific extensions kept on the dict for the
    per-context raw_scheduler_json column.
    """

    id: str
    workload_type: str
    scheduler: str
    nodes: list[str]
    start_time: datetime
    end_time: datetime
    # Service-facing state: COMPLETED, FAILED, TIMEOUT, CANCELLED, NODE_FAIL, or RUNNING.
    state: str

    job_id: str
    pod_group_uid: str
    attempt_id: str
    namespace: str
    name: str
    podgroup_shape: str
    owner_kind: str
    owner_name: str
    owner_uid: str
    owner_state: str  # native owner wording; see `_owner_state`
    failed_node: NotRequired[str]
    time_limit: NotRequired[int]

    queue_name: NotRequired[str]
    min_member: NotRequired[int]
    # Raw PodGroup status, preserved unmodified.
    scheduling_conditions: NotRequired[list]
    resources_status: NotRequired[object]
    pod_names: NotRequired[list[str]]
    # Pod name -> `.spec.nodeName`.
    pod_nodes: NotRequired[dict[str, str]]
    pod_placements: NotRequired[list[PodPlacement]]


def _parse_dt(value: str | None) -> datetime | None:
    """Parse a Kubernetes RFC3339 timestamp into a tz-aware datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _split_csv(value: str) -> list[str]:
    """CSV-split a string, stripping whitespace and dropping empty entries."""
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _index_pods_by_pod_group(pods: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group Pods by `(namespace, pod_group_name)`, skipping Pods that have no
    `pod-group-name` annotation or no namespace."""
    index: dict[tuple[str, str], list[dict]] = {}
    for pod in pods:
        meta = pod.get("metadata") or {}
        namespace = meta.get("namespace")
        pod_group_name = (meta.get("annotations") or {}).get(_POD_GROUP_ANNOTATION)
        if namespace and pod_group_name:
            index.setdefault((namespace, pod_group_name), []).append(pod)
    return index


def _podgroup_name_embeds_owner_uid(pod_group_name: str | None, owner: dict) -> bool:
    """Return whether `pod_group_name` embeds `owner['uid']` per KAI pod-grouper naming."""
    owner_uid = owner.get("uid")
    owner_name = owner.get("name")
    if not pod_group_name or not owner_uid:
        return False
    prefixes = [f"pg-{owner_uid}"]
    if owner_name:
        prefixes.append(f"pg-{owner_name}-{owner_uid}")
    return any(pod_group_name == prefix or pod_group_name.startswith(f"{prefix}-") for prefix in prefixes)


def _pod_matches_non_job_owner(pod: dict, owner: dict) -> bool:
    """Return whether a PodGroup-annotated Pod has positive ownership evidence for `owner`."""
    meta = pod.get("metadata") or {}
    pod_top = _top_owner(meta)
    controller = _controller_owner(meta)
    if pod_top is not None:
        if pod_top["kind"] != owner["kind"] or pod_top["uid"] != owner["uid"]:
            return False
        return controller is None or controller["kind"] != owner["kind"] or controller["uid"] == owner["uid"]
    if controller is None:
        return False
    if controller["kind"] == owner["kind"]:
        return controller["uid"] == owner["uid"]
    if controller["kind"] not in _CHILD_CONTROLLER_KINDS.get(owner["kind"], frozenset()):
        return False
    pod_group_name = (meta.get("annotations") or {}).get(_POD_GROUP_ANNOTATION)
    return _podgroup_name_embeds_owner_uid(pod_group_name, owner)


def _member_pods(owner: dict, owner_job_uids: set[str], pods: list[dict]) -> list[dict]:
    """From Pods already matched by `pod-group-name` and namespace, keep only
    those belonging to the verified workload owner.

    For Job and JobSet owners, Pods must be controlled by the owning Job or one
    of the JobSet's current child Jobs. For non-Job workload owners, Pods must
    carry positive ownership evidence matching the verified owner UID via
    `kai.scheduler/top-owner-metadata`, a direct controller ownerReference of
    the same kind, or an expected child-controller kind on a KAI PodGroup whose
    name embeds the verified owner UID.

    No additional time-based filter: neither `managedFields` time (reflects
    any owned field, not `pod-group-name` specifically) nor a Pod's own
    `creationTimestamp` (can legitimately predate its PodGroup) reliably
    proves which PodGroup incarnation a Pod belongs to. A PodGroup recreated
    under the same name with the same still-live owning workload is therefore a
    known, accepted limitation.
    """
    if owner["kind"] in {_JOB_RESOURCE, _JOBSET_RESOURCE}:
        return [
            pod
            for pod in pods
            if (controller := _controller_owner(pod.get("metadata") or {})) is not None
            and controller["kind"] == _JOB_RESOURCE
            and controller["uid"] in owner_job_uids
        ]
    return [pod for pod in pods if _pod_matches_non_job_owner(pod, owner)]


def _jobset_child_jobs(owner: dict, jobs: list[dict]) -> list[dict]:
    """Return Jobs controlled by this exact JobSet incarnation."""
    return [
        job
        for job in jobs
        if (controller := _controller_owner(job.get("metadata") or {})) is not None
        and controller["kind"] == "JobSet"
        and controller["uid"] == owner["uid"]
        and (job.get("metadata") or {}).get("uid")
    ]


def _podgroup_shape(spec: dict) -> str:
    """Return whether a PodGroup spec is hierarchical or flat owned."""
    return _SHAPE_HIERARCHICAL if spec.get("subGroups") else _SHAPE_OWNED


def _non_negative_int(value: object) -> int | None:
    """Return `value` when it is a non-boolean non-negative int, else None."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _job_end_time(job: dict) -> datetime | None:
    """Completion time of a finished Job, or None while it is still active."""
    status = job.get("status") or {}
    completed = _parse_dt(status.get("completionTime"))
    if completed is not None:
        return completed
    for condition in status.get("conditions") or []:
        if (condition or {}).get("type") == "Failed" and (condition or {}).get("status") == "True":
            return _parse_dt(condition.get("lastTransitionTime"))
    return None


def _owner_group(owner: dict) -> str | None:
    """Extract the API group from a resolved owner mapping."""
    group = owner.get("group")
    api_version = owner.get("apiVersion")
    if group is None and isinstance(api_version, str) and "/" in api_version:
        group = api_version.split("/", 1)[0]
    return group


def _owner_version(owner: dict) -> str | None:
    """Extract the API version token from a resolved owner mapping."""
    version = owner.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    api_version = owner.get("apiVersion")
    if isinstance(api_version, str):
        _, _, suffix = api_version.partition("/")
        token = suffix if suffix else api_version
        if token.strip():
            return token.strip()
    return None


def _owner_lookup_options(owner: dict, owner_resource: str) -> dict[str, str]:
    """Return `KubernetesClient.get_one` lookup options for `owner`."""
    options = dict(_OWNER_LOOKUPS[owner_resource][1])
    if "group" in options and "api_version" in options and (version := _owner_version(owner)):
        options["api_version"] = version
    return options


def _is_resource_discovery_error(error: KubernetesClientError) -> bool:
    """Return whether `error` came from dynamic API resource resolution."""
    return error.status is None and "resource discovery failed" in str(error)


def _owner_resource(owner: dict) -> str | None:
    """Return the Kubernetes resource kind when `owner` is a supported workload."""
    kind = owner.get("kind")
    lookup = _OWNER_LOOKUPS.get(kind) if isinstance(kind, str) else None
    if lookup is None:
        return None
    expected_group, _ = lookup
    if _owner_group(owner) not in (None, expected_group):
        return None
    return kind


def _jobset_terminal(jobset: dict) -> tuple[datetime, str] | None:
    """Terminal transition time and condition type of a JobSet, or None while active."""
    status = jobset.get("status") or {}
    terminal_state = status.get("terminalState")
    condition_types = {terminal_state} if terminal_state in {"Completed", "Failed"} else {"Completed", "Failed"}
    terminals = [
        (transition, condition["type"])
        for condition in status.get("conditions") or []
        if (condition or {}).get("type") in condition_types
        and condition.get("status") == "True"
        and (transition := _parse_dt(condition.get("lastTransitionTime"))) is not None
    ]
    return max(terminals) if terminals else None


def _jobset_end_time(jobset: dict) -> datetime | None:
    """Terminal condition time of a finished JobSet, or None while active."""
    terminal = _jobset_terminal(jobset)
    return terminal[0] if terminal else None


def _pytorchjob_end_time(pytorchjob: dict) -> datetime | None:
    """Completion or failure time of a finished PyTorchJob, or None while active."""
    status = pytorchjob.get("status") or {}
    completed = _parse_dt(status.get("completionTime"))
    if completed is not None:
        return completed
    transitions = [
        transition
        for condition in status.get("conditions") or []
        if (condition or {}).get("type") in {"Succeeded", "Failed"}
        and condition.get("status") == "True"
        and (transition := _parse_dt(condition.get("lastTransitionTime"))) is not None
    ]
    return max(transitions) if transitions else None


def _pytorchjob_is_terminal(pytorchjob: dict) -> bool:
    """Return whether a PyTorchJob has a completion timestamp or terminal condition."""
    status = pytorchjob.get("status") or {}
    if _parse_dt(status.get("completionTime")) is not None:
        return True
    return any(_condition(pytorchjob, kind) is not None for kind in ("Succeeded", "Failed"))


def _pytorchjob_is_deadline_exceeded(pytorchjob: dict) -> bool:
    """Return whether a failed PyTorchJob exceeded its active deadline."""
    cond = _condition(pytorchjob, "Failed")
    if cond is None:
        return False
    reason = (cond.get("reason") or "").strip()
    if reason in _DEADLINE_FAILURE_REASONS:
        return True
    message = (cond.get("message") or "").strip().lower()
    return any(marker in message for marker in _PYTORCHJOB_DEADLINE_MESSAGES)


def _rayjob_is_failed(rayjob: dict) -> bool:
    """Return whether a RayJob reports a failed job or deployment status."""
    status = rayjob.get("status") or {}
    if status.get("jobDeploymentStatus") in _RAYJOB_RETRY_OR_SUSPENDED_STATUSES:
        return False
    if status.get("jobStatus") == "FAILED" or status.get("jobDeploymentStatus") == "Failed":
        return True
    return _condition(rayjob, "Failed") is not None


def _rayjob_is_stopped(rayjob: dict) -> bool:
    """Return whether a RayJob finished in the STOPPED state without failing."""
    status = rayjob.get("status") or {}
    if status.get("jobDeploymentStatus") in _RAYJOB_RETRY_OR_SUSPENDED_STATUSES:
        return False
    return status.get("jobStatus") == "STOPPED" and not _rayjob_is_failed(rayjob)


def _rayjob_is_terminal(rayjob: dict) -> bool:
    """Return whether a RayJob has reached a terminal job, deployment, or condition state."""
    status = rayjob.get("status") or {}
    job_status = status.get("jobStatus")
    deployment_status = status.get("jobDeploymentStatus")
    if deployment_status in _RAYJOB_RETRY_OR_SUSPENDED_STATUSES or (
        job_status in _RAYJOB_ACTIVE_JOB_STATUSES and deployment_status not in _RAYJOB_TERMINAL_DEPLOYMENT_STATUSES
    ):
        return False
    return (
        job_status in _RAYJOB_TERMINAL_JOB_STATUSES
        or deployment_status in _RAYJOB_TERMINAL_DEPLOYMENT_STATUSES
        or any(_condition(rayjob, kind) is not None for kind in _RAYJOB_TERMINAL_CONDITION_TYPES)
    )


def _rayjob_end_time(rayjob: dict) -> datetime | None:
    """Terminal time of a finished RayJob, or None while it is still active or untimed."""
    if not _rayjob_is_terminal(rayjob):
        return None
    status = rayjob.get("status") or {}
    ray_job_info = status.get("rayJobInfo") or {}
    end_time = _parse_dt(status.get("endTime") or ray_job_info.get("endTime"))
    if end_time is not None:
        return end_time
    transitions = [
        transition
        for condition in status.get("conditions") or []
        if (condition or {}).get("type") in _RAYJOB_TERMINAL_CONDITION_TYPES
        and condition.get("status") == "True"
        and (transition := _parse_dt(condition.get("lastTransitionTime"))) is not None
    ]
    return max(transitions) if transitions else None


def _deployment_stalled_reason(deployment: dict) -> str | None:
    """Return `ProgressDeadlineExceeded` when a Deployment rollout is stalled, or None."""
    for condition in (deployment.get("status") or {}).get("conditions") or []:
        if (
            (condition or {}).get("type") == "Progressing"
            and condition.get("status") == "False"
            and condition.get("reason") == "ProgressDeadlineExceeded"
        ):
            return "ProgressDeadlineExceeded"
    return None


def _lws_failed(lws: dict) -> tuple[datetime | None, str] | None:
    """Return `(transition_time, reason)` for a failed LeaderWorkerSet, or None."""
    failed = _condition(lws, "Failed")
    if failed is None:
        return None
    return _parse_dt(failed.get("lastTransitionTime")), (failed.get("reason") or "").strip()


def _dynamo_reason(raw_reason: str | None) -> str:
    """Normalize a DynamoGraphDeployment failure reason."""
    reason = (raw_reason or "").strip()
    return "" if reason.lower() == "failed" else reason


def _dynamo_failure_condition(dgd: dict) -> dict | None:
    """Return the Failed or failed-Ready condition of a DynamoGraphDeployment, or None."""
    status = dgd.get("status") or {}
    state = (status.get("state") or "").strip().lower()
    failed_cond = _condition(dgd, "Failed")
    if failed_cond is not None:
        return failed_cond
    return next(
        (
            cond
            for cond in status.get("conditions") or []
            if isinstance(cond, dict)
            and cond.get("type") == "Ready"
            and cond.get("status") == "False"
            and (state == "failed" or cond.get("reason") == "Failed")
        ),
        None,
    )


def _dynamo_is_failed(dgd: dict) -> bool:
    """Return whether a DynamoGraphDeployment is in a terminal failed state."""
    status = dgd.get("status") or {}
    state = (status.get("state") or "").strip().lower()
    return state == "failed" or _dynamo_failure_condition(dgd) is not None


def _lws_end_time(lws: dict) -> datetime | None:
    """Failure transition time of a LeaderWorkerSet, or None while active."""
    failed = _lws_failed(lws)
    return failed[0] if failed is not None else None


def _dynamo_end_time(dgd: dict) -> datetime | None:
    """Failure transition time of a DynamoGraphDeployment, or None while active or untimed."""
    if not _dynamo_is_failed(dgd):
        return None
    cond = _dynamo_failure_condition(dgd)
    if cond is not None and (transition := _parse_dt(cond.get("lastTransitionTime"))) is not None:
        return transition
    status = dgd.get("status") or {}
    for key in ("endTime", "completionTime", "lastTransitionTime"):
        if (transition := _parse_dt(status.get(key))) is not None:
            return transition
    return None


_END_TIME_RESOLVERS = {
    _JOB_RESOURCE: _job_end_time,
    _JOBSET_RESOURCE: _jobset_end_time,
    _PYTORCHJOB_RESOURCE: _pytorchjob_end_time,
    _RAYJOB_RESOURCE: _rayjob_end_time,
    _LWS_RESOURCE: _lws_end_time,
    _DYNAMO_GRAPH_DEPLOYMENT_RESOURCE: _dynamo_end_time,
}


def _owner_end_time(owner: dict, owner_object: dict) -> datetime | None:
    """Return the terminal timestamp of `owner_object`, or None while active."""
    resolver = _END_TIME_RESOLVERS.get(owner["kind"])
    return resolver(owner_object) if resolver is not None else None


def _condition(owner_object: dict, condition_type: str) -> dict | None:
    """Return the first True condition of `condition_type` on `owner_object`."""
    for condition in (owner_object.get("status") or {}).get("conditions") or []:
        if (condition or {}).get("type") == condition_type and condition.get("status") == "True":
            return condition
    return None


def _condition_reason(owner_object: dict, condition_type: str) -> str | None:
    """Return the stripped reason of a True `condition_type`, or None."""
    cond = _condition(owner_object, condition_type)
    return (cond.get("reason") or "").strip() if cond is not None else None


def _is_suspended(owner_object: dict) -> bool:
    """Return whether `owner_object` is currently suspended."""
    if _condition(owner_object, "Suspended") is not None:
        return True
    status = owner_object.get("status") or {}
    return status.get("jobDeploymentStatus") == "Suspended"


def _job_failure_reason(job: dict) -> str | None:
    """Return the failure reason of a Job, or None when it completed."""
    if _parse_dt((job.get("status") or {}).get("completionTime")) is not None:
        return None
    return _condition_reason(job, "Failed")


def _jobset_failure_reason(jobset: dict) -> str | None:
    """Return the failure reason of a JobSet, or None when it completed."""
    terminal = _jobset_terminal(jobset)
    if terminal is None or terminal[1] != "Failed":
        return None
    return _condition_reason(jobset, "Failed") or ""


def _pytorchjob_failure_reason(pytorchjob: dict) -> str | None:
    """Return the failure reason of a PyTorchJob, or None when it succeeded."""
    cond = _condition(pytorchjob, "Failed")
    if cond is None:
        return None
    reason = (cond.get("reason") or "").strip()
    if not reason and _pytorchjob_is_deadline_exceeded(pytorchjob):
        return "DeadlineExceeded"
    return reason


def _rayjob_failure_reason(rayjob: dict) -> str | None:
    """Return the failure reason of a RayJob, or None when it succeeded."""
    if not _rayjob_is_failed(rayjob):
        return None
    status = rayjob.get("status") or {}
    return (status.get("reason") or _condition_reason(rayjob, "Failed") or "").strip()


def _lws_failure_reason(lws: dict) -> str | None:
    """Return the failure reason of a LeaderWorkerSet, or None while active."""
    failed = _lws_failed(lws)
    return failed[1] if failed is not None else None


def _dynamo_failure_reason(dgd: dict) -> str | None:
    """Return the failure reason of a DynamoGraphDeployment, or None while active."""
    if not _dynamo_is_failed(dgd):
        return None
    status = dgd.get("status") or {}
    cond = _dynamo_failure_condition(dgd)
    raw = (cond.get("reason") if cond is not None else None) or status.get("reason")
    return _dynamo_reason(raw)


_FAILURE_REASON_RESOLVERS = {
    _JOB_RESOURCE: _job_failure_reason,
    _JOBSET_RESOURCE: _jobset_failure_reason,
    _PYTORCHJOB_RESOURCE: _pytorchjob_failure_reason,
    _RAYJOB_RESOURCE: _rayjob_failure_reason,
    _LWS_RESOURCE: _lws_failure_reason,
    _DYNAMO_GRAPH_DEPLOYMENT_RESOURCE: _dynamo_failure_reason,
}


def _owner_failure_reason(owner: dict, owner_object: dict) -> str | None:
    """Return the failure reason string when `owner_object` failed, or None when it completed."""
    resolver = _FAILURE_REASON_RESOLVERS.get(owner["kind"])
    return resolver(owner_object) if resolver is not None else None


_TERMINAL_PREDICATES = {
    _PYTORCHJOB_RESOURCE: _pytorchjob_is_terminal,
    _RAYJOB_RESOURCE: _rayjob_is_terminal,
    _DYNAMO_GRAPH_DEPLOYMENT_RESOURCE: _dynamo_is_failed,
    _LWS_RESOURCE: lambda lws: _lws_failed(lws) is not None,
}


def _owner_is_terminal(owner: dict, owner_object: dict) -> bool:
    """Return whether `owner_object` has reached a terminal state."""
    if _owner_end_time(owner, owner_object) is not None:
        return True
    predicate = _TERMINAL_PREDICATES.get(owner["kind"])
    return predicate(owner_object) if predicate is not None else False


def _active_owner_state(owner: dict, owner_object: dict) -> str:
    """Return the native owner_state for a non-terminal workload."""
    if _is_suspended(owner_object):
        return "Suspended"
    if owner["kind"] == _DEPLOYMENT_RESOURCE and (stalled := _deployment_stalled_reason(owner_object)):
        return stalled
    return "Active"


def _rayjob_stopped_state(rayjob: dict) -> str:
    """Return the native owner_state for a stopped RayJob."""
    reason = ((rayjob.get("status") or {}).get("reason") or "").strip()
    return f"Stopped:{reason}" if reason else "Stopped"


def _owner_state(owner: dict, owner_object: dict, owner_jobs: list[dict]) -> tuple[str, str]:
    """Return (service-facing state, native owner state).

    Service state is COMPLETED, FAILED, TIMEOUT, CANCELLED, or RUNNING
    (including while suspended or rollout-stalled). Native state keeps the
    owner's own wording, e.g. "Failed:BackoffLimitExceeded".
    """
    if not _owner_is_terminal(owner, owner_object):
        return "RUNNING", _active_owner_state(owner, owner_object)

    if owner["kind"] == _RAYJOB_RESOURCE and _rayjob_is_stopped(owner_object):
        return "CANCELLED", _rayjob_stopped_state(owner_object)

    owner_reason = _owner_failure_reason(owner, owner_object)
    if owner_reason is None:
        return "COMPLETED", _COMPLETED_OWNER_STATE.get(owner["kind"], "Completed")

    native = f"Failed:{owner_reason}" if owner_reason else "Failed"
    failure_reason = earliest_job_failure_reason(owner_jobs) or owner_reason
    is_timeout = failure_reason in _DEADLINE_FAILURE_REASONS or (
        owner["kind"] == _PYTORCHJOB_RESOURCE and _pytorchjob_is_deadline_exceeded(owner_object)
    )
    return ("TIMEOUT" if is_timeout else "FAILED"), native


def _skip_reason(owner: dict, owner_object: dict | None) -> str | None:
    """Why the owner cannot supply a window, or None when it can."""
    if owner_object is None:
        return f"owner {owner['kind']} not found"
    if (owner_object.get("metadata") or {}).get("uid") != owner["uid"]:
        return f"owner {owner['kind']} uid mismatch"
    return None


def _podgroup_identity(podgroup: dict) -> tuple[str, str, str, datetime] | None:
    """Extract `(uid, name, namespace, start_time)` from `podgroup`, or None when invalid."""
    meta = podgroup.get("metadata") or {}
    uid = meta.get("uid")
    name = meta.get("name")
    namespace = meta.get("namespace")
    if not uid or not name or not namespace:
        logger.warning(f"podgroup missing uid/name/namespace; skipping: {meta!r}")
        return None
    start_time = _parse_dt(meta.get("creationTimestamp"))
    if start_time is None:
        logger.warning(f"podgroup {namespace}/{name} has no creationTimestamp; skipping")
        return None
    return uid, name, namespace, start_time


def _member_pod_summary(
    members: list[dict],
    podgroup_start: datetime,
) -> tuple[datetime, dict[str, str], list[str], list[str]]:
    """Summarize earliest start time, node mappings, pod names, and node-loss nodes."""
    member_starts = [
        created
        for pod in members
        if (created := _parse_dt((pod.get("metadata") or {}).get("creationTimestamp"))) is not None
    ]
    start_time = min(podgroup_start, *member_starts) if member_starts else podgroup_start
    pod_nodes = {
        pod_name: node
        for pod in members
        if (pod_name := (pod.get("metadata") or {}).get("name")) and (node := (pod.get("spec") or {}).get("nodeName"))
    }
    pod_names = sorted(pod_name for pod in members if (pod_name := (pod.get("metadata") or {}).get("name")))
    node_loss_nodes = sorted(
        {node for pod in members if pod_node_failure_reason(pod) and (node := (pod.get("spec") or {}).get("nodeName"))}
    )
    return start_time, pod_nodes, pod_names, node_loss_nodes


def _attach_optional_fields(
    ctx: KaiExecutionContext,
    podgroup: dict,
    owner_jobs: list[dict],
    pod_names: list[str],
    pod_nodes: dict[str, str],
    node_loss_nodes: list[str],
) -> None:
    """Populate optional PodGroup, owner, and Pod fields on `ctx`."""
    spec = podgroup.get("spec") or {}
    if queue_name := spec.get("queue"):
        ctx["queue_name"] = queue_name
    if (min_member := _non_negative_int(spec.get("minMember"))) is not None:
        ctx["min_member"] = min_member
    if (time_limit := job_time_limit_minutes(owner_jobs)) is not None:
        ctx["time_limit"] = time_limit
    if ctx["state"] == "NODE_FAIL" and node_loss_nodes:
        ctx["failed_node"] = node_loss_nodes[0]
    status = podgroup.get("status") or {}
    if isinstance(conditions := status.get("schedulingConditions"), list):
        ctx["scheduling_conditions"] = conditions
    if "resourcesStatus" in status:
        ctx["resources_status"] = status["resourcesStatus"]
    if pod_names:
        ctx["pod_names"] = pod_names
    if pod_nodes:
        ctx["pod_nodes"] = pod_nodes


def _parse_pod_group(
    podgroup: dict,
    owner: dict,
    owner_jobs: list[dict],
    pods: list[dict],
    owner_object: dict | None,
    now: datetime,
) -> KaiExecutionContext | None:
    """Normalize one PodGroup and its member Pods into wire format.

    `owner` identifies the PodGroup's workload owner and `owner_object`
    is the live owner object when one was found.

    Returns None when the PodGroup lacks a uid, name, namespace, or a
    parseable creationTimestamp (without a start the orchestrator cannot build
    a valid attributor window), or when the owner cannot supply an end.
    """
    identity = _podgroup_identity(podgroup)
    if identity is None:
        return None
    uid, name, namespace, start_time = identity

    skip_reason = _skip_reason(owner, owner_object)
    if skip_reason is not None or owner_object is None:
        logger.info(f"podgroup {namespace}/{name}: {skip_reason}; skipping")
        return None

    owner_job_uids = {job_uid for job in owner_jobs if (job_uid := (job.get("metadata") or {}).get("uid")) is not None}
    members = _member_pods(owner, owner_job_uids, pods)
    start_time, pod_nodes, pod_names, node_loss_nodes = _member_pod_summary(members, start_time)

    shape = _podgroup_shape(podgroup.get("spec") or {})
    if shape != _SHAPE_OWNED:
        logger.info(f"podgroup {namespace}/{name}: {shape} PodGroup; recording its shape only")
    owner_end = _owner_end_time(owner, owner_object)
    if owner_end is None:
        logger.debug(
            f"podgroup {namespace}/{name}: owner {owner['kind']} has no terminal timestamp; using now() for end_time"
        )
    end_time = max(owner_end if owner_end is not None else now, start_time)
    state, owner_state = _owner_state(owner, owner_object, owner_jobs)
    if state != "COMPLETED" and any(pod_node_failure_reason(pod) for pod in members):
        state = "NODE_FAIL"
    if not pod_nodes:
        logger.warning(f"podgroup {namespace}/{name}: node list unavailable from member Pods")

    ctx: KaiExecutionContext = {
        "id": uid,
        "workload_type": "kai",
        "scheduler": "kai",
        "nodes": sorted(set(pod_nodes.values())),
        "start_time": start_time,
        "end_time": end_time,
        "state": state,
        "job_id": f"{namespace}/{name}",
        "pod_group_uid": uid,
        # PodGroup incarnation identity, not a scheduling attempt.
        "attempt_id": uid,
        "namespace": namespace,
        "name": name,
        "podgroup_shape": shape,
        "owner_kind": owner["kind"],
        "owner_name": owner["name"],
        "owner_uid": owner["uid"],
        "owner_state": owner_state,
    }
    _attach_optional_fields(ctx, podgroup, owner_jobs, pod_names, pod_nodes, node_loss_nodes)
    return ctx


class KaiExecutionContextAdapter:
    """ExecutionContextAdapter that discovers KAI PodGroups via Kubernetes.

    `kubeconfig` and `context` configure the shared persistent API client.
    `namespace` selects the namespace used for time-range discovery; ID
    discovery can instead carry a namespace in each ID.
    """

    def __init__(
        self,
        kubeconfig: str | None = None,
        context: str | None = None,
        namespace: str | None = None,
        timeout_seconds: int = 60,
        placement_records_enabled: bool = False,
        placement_reader: PlacementReader | None = None,
    ):
        """Initialize the KAI ExecutionContextAdapter and Kubernetes client."""
        self.namespace = namespace or ""
        self.timeout_seconds = timeout_seconds
        if not isinstance(placement_records_enabled, bool):
            raise ValueError("placement_records_enabled must be a bool")
        self.placement_records_enabled = placement_records_enabled
        self.placement_reader = placement_reader
        self._kubernetes = KubernetesClient(
            kubeconfig=kubeconfig,
            context=context,
            timeout_seconds=timeout_seconds,
        )

    # --- public protocol ---------------------------------------------------

    def discover(self) -> list[KaiExecutionContext]:
        """No-runtime-selector entrypoint.

        There is no defensible "all PodGroups" default scope, so callers use
        `discover_by_ids` or `discover_by_time_range`.
        """
        return []

    def discover_by_ids(self, ids: list[str]) -> list[KaiExecutionContext]:
        """Look up PodGroups by `<namespace>/<name>` (or bare `<name>` when the
        adapter is namespace-scoped). PodGroups that do not exist are skipped;
        any other failed lookup raises."""
        podgroups: list[dict] = []
        for ref in ids:
            ns, _, name = ref.partition("/")
            if not name:
                if not self.namespace:
                    raise ValueError(
                        f"kai ECA --ids: {ref!r} is missing a namespace; "
                        f"use `<namespace>/<name>` or configure eca.kai.namespace"
                    )
                ns, name = self.namespace, ref
            if self.placement_records_enabled and self.namespace and ns != self.namespace:
                raise ValueError(f"kai ECA --ids: {ref!r} is outside configured namespace {self.namespace!r}")
            try:
                podgroups.append(
                    self._kubernetes.get_one(
                        _PODGROUP_RESOURCE,
                        name,
                        namespace=ns,
                        group=_PODGROUP_GROUP,
                    )
                )
            except KubernetesClientError as error:
                if error.status != 404:
                    raise
        return self._finalize(podgroups)

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[KaiExecutionContext]:
        """List PodGroups, then keep those whose window intersects `[start, end]`.

        The window is checked after member Pods and owners are resolved:
        a member Pod may predate its PodGroup, and the owner supplies the end.
        A failed list raises.
        `nodelist` entries may be CSVs of plain node names; a PodGroup
        survives if any member Pod ran on one of them.
        """
        assert_aware(start, "discover_by_time_range.start")
        assert_aware(end, "discover_by_time_range.end")
        if not self.namespace:
            raise ValueError("kai ECA time-range discovery requires eca.kai.namespace")
        listing = self._kubernetes.list(
            _PODGROUP_RESOURCE,
            namespace=self.namespace,
            group=_PODGROUP_GROUP,
        )
        contexts = self._finalize(listing, time_range=(start, end))
        if not nodelist:
            return contexts
        wanted: set[str] = set()
        for entry in nodelist:
            wanted.update(_split_csv(str(entry)))
        if not wanted:
            return contexts
        return [c for c in contexts if wanted.intersection(c["nodes"])]

    # --- internals ---------------------------------------------------------

    def _fetch_owner(
        self,
        namespace: str,
        owner: dict,
        owner_resource: str,
    ) -> dict | None:
        """Fetch the live owner object, falling back to group lookup if a version is unserved."""
        if not namespace:
            return None
        owner_options = _owner_lookup_options(owner, owner_resource)
        candidates = (
            (owner_options, {"group": owner_options["group"]})
            if "group" in owner_options and "api_version" in owner_options
            else (owner_options,)
        )
        for options in candidates:
            try:
                return self._kubernetes.get_one(
                    owner_resource,
                    owner["name"],
                    namespace=namespace,
                    **options,
                )
            except KubernetesClientError as error:
                if _is_resource_discovery_error(error) and "api_version" in options and len(candidates) > 1:
                    continue
                if error.status != 404:
                    raise
                return None
        return None

    def _fetch_owner_and_jobs(
        self,
        namespace: str,
        owner: dict,
        owner_resource: str,
    ) -> tuple[dict | None, list[dict]]:
        """Fetch the live owner object and any Job/JobSet child Jobs."""
        owner_object = self._fetch_owner(namespace, owner, owner_resource)
        owner_jobs = [owner_object] if owner["kind"] == _JOB_RESOURCE and owner_object is not None else []
        if owner["kind"] == _JOBSET_RESOURCE and _skip_reason(owner, owner_object) is None:
            owner_jobs = _jobset_child_jobs(
                owner,
                self._kubernetes.list(
                    _JOB_RESOURCE,
                    namespace=namespace,
                    label_selector=f"{_JOBSET_NAME_LABEL}={owner['name']}",
                    api_version=_JOB_API_VERSION,
                ),
            )
        return owner_object, owner_jobs

    def _finalize(
        self,
        podgroups: list[dict],
        *,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[KaiExecutionContext]:
        """Resolve owners and member Pods for `podgroups` and build contexts."""
        # Unsupported owners cannot supply an end time, so drop them before
        # fetching Pods or owners.
        owned: list[tuple[dict, dict, str]] = []
        for podgroup in podgroups:
            meta = podgroup.get("metadata") or {}
            owner = _resolve_owner(meta)
            owner_resource = _owner_resource(owner) if owner is not None else None
            if owner is None or owner_resource is None:
                logger.info(
                    f"podgroup {meta.get('namespace')}/{meta.get('name')}: not owned by a supported workload; skipping"
                )
                continue
            owned.append((podgroup, owner, owner_resource))
        if not owned:
            return []
        pods_by_key = _index_pods_by_pod_group(self._fetch_pods([pg for pg, _, _ in owned]))
        now = datetime.now(tz=timezone.utc)
        contexts: list[KaiExecutionContext] = []
        for podgroup, owner, owner_resource in owned:
            meta = podgroup.get("metadata") or {}
            namespace = meta.get("namespace") or ""
            owner_object, owner_jobs = self._fetch_owner_and_jobs(namespace, owner, owner_resource)
            ctx = _parse_pod_group(
                podgroup,
                owner,
                owner_jobs,
                pods_by_key.get((namespace, meta.get("name") or ""), []),
                owner_object,
                now,
            )
            if ctx is None:
                continue
            if time_range is not None:
                start, end = time_range
                if ctx["start_time"] > end or ctx["end_time"] < start:
                    continue
            self._merge_placement_records(ctx)
            contexts.append(ctx)
        return contexts

    def _fetch_placement_records(self, context: KaiExecutionContext) -> list[PlacementRecord]:
        """Fetch retained placement records for `context` from the daemon reader."""
        if not self.placement_records_enabled:
            return []
        if self.placement_reader is None:
            raise RuntimeError("placement records are enabled but no KubernetesDaemon client is configured")
        try:
            return self.placement_reader.list_for_resource(
                context["namespace"],
                context["pod_group_uid"],
            )
        except Exception as error:
            raise RuntimeError(f"KubernetesDaemon placement lookup failed: {error}") from error

    def _merge_placement_records(self, context: KaiExecutionContext) -> None:
        """Merge retained daemon placement records and node-loss evidence into `context`."""
        records = self._fetch_placement_records(context)
        if not records:
            return
        placements: list[PodPlacement] = [
            {
                "pod_uid": record.pod_uid,
                "pod_name": record.pod_name,
                "node_name": record.node_name,
                "source": "kubernetes_daemon",
            }
            for record in records
        ]
        existing = {
            placement["pod_uid"]: placement
            for placement in context.get("pod_placements") or []
            if placement.get("pod_uid")
        }
        for placement in placements:
            existing.setdefault(placement["pod_uid"], placement)
        merged = sorted(existing.values(), key=lambda placement: placement["pod_uid"])
        context["pod_placements"] = merged
        context["nodes"] = sorted(set(context["nodes"]) | {placement["node_name"] for placement in merged})
        context["pod_names"] = sorted(
            set(context.get("pod_names") or []) | {placement["pod_name"] for placement in merged}
        )
        pod_nodes = dict(context.get("pod_nodes") or {})
        for placement in placements:
            pod_nodes.setdefault(placement["pod_name"], placement["node_name"])
        context["pod_nodes"] = pod_nodes

        node_loss_nodes = {
            record.node_name
            for record in records
            if record.node_failure_reason == "NodeLost" or record.node_failure_reason in NODE_LOSS_DISRUPTION_REASONS
        }
        if node_loss_nodes and context["state"] != "COMPLETED":
            context["state"] = "NODE_FAIL"
            context["failed_node"] = min(node_loss_nodes)

    def _fetch_pods(self, podgroups: list[dict]) -> list[dict]:
        """List Pods once per distinct namespace.

        Pods carry the PodGroup only as an annotation, which Kubernetes field
        and label selectors cannot select on, so the join is done client-side
        after a namespace-scoped list.
        """
        namespaces = sorted({ns for pg in podgroups if (ns := (pg.get("metadata") or {}).get("namespace"))})
        pods: list[dict] = []
        for ns in namespaces:
            pods.extend(self._kubernetes.list("Pod", namespace=ns, api_version="v1"))
        return pods
