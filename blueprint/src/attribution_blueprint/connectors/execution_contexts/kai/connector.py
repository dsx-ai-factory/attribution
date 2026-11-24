# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KAI Scheduler ExecutionContextAdapter.

Emits one workload-level context per PodGroup, with these KAI specifics:

    ExecutionContext   KAI source
    -----------------  -----------------------------------------------------
    id / attempt_id    For Job and JobSet owners, canonical PodGroup UID +
                       scheduling boundary + JobSet restart generation
                       identity. Unscheduled Job/JobSet PodGroups and
                       other supported owner kinds retain their UID-based
                       live identity.
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
                       suspended, when a Deployment rollout hits
                       ProgressDeadlineExceeded, or when a
                       DynamoGraphDeployment reports a reconciliation
                       failure. `owner_state` keeps the owner's own
                       wording. A member Pod that reports node loss sets
                       state to NODE_FAIL and sets `failed_node`; a
                       completed one stays COMPLETED without one.

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
owner is a JobSet). Owner lookups that return NotFound are skipped as
absent, and owner lookups denied with Forbidden (403) log a warning and
skip that PodGroup; any other failed Kubernetes read raises.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple, NotRequired, TypedDict

from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kai.attempts import (
    PodGroupAttempt,
    attempts_from_podgroup_history,
    snapshots_from_live_resources,
)
from attribution_blueprint.connectors.execution_contexts.kai.helpers import (
    POD_GROUP_ANNOTATION as _POD_GROUP_ANNOTATION,
    controller_owner as _controller_owner,
    resolve_owner as _resolve_owner,
    top_owner as _top_owner,
)
from attribution_blueprint.kubernetes.attempts import (
    kai_podgroup_start_timestamp,
    resource_from_snapshot,
    retained_history_covers_window,
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
_HTTP_FORBIDDEN = 403
_HTTP_NOT_FOUND = 404
_JOB_RESOURCE = "Job"
_JOB_API_VERSION = "batch/v1"
_JOBSET_RESOURCE = "JobSet"
_JOBSET_GROUP = "jobset.x-k8s.io"
_JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"
_LWS_RESOURCE = "LeaderWorkerSet"
_LWS_GROUP = "leaderworkerset.x-k8s.io"
_LWS_API_VERSION = "v1"
_LWS_NAME_LABEL = "leaderworkerset.sigs.k8s.io/name"
_PYTORCHJOB_RESOURCE = "PyTorchJob"
_PYTORCHJOB_GROUP = "kubeflow.org"
_PYTORCHJOB_API_VERSION = "v1"
_RAYJOB_RESOURCE = "RayJob"
_RAYJOB_GROUP = "ray.io"
_RAYJOB_API_VERSION = "v1"
_RAY_CLUSTER_LABEL = "ray.io/cluster"
_DEPLOYMENT_RESOURCE = "Deployment"
_DEPLOYMENT_API_VERSION = "apps/v1"
_DYNAMO_GRAPH_DEPLOYMENT_RESOURCE = "DynamoGraphDeployment"
_DYNAMO_GROUP = "nvidia.com"
_DYNAMO_API_VERSION = "v1alpha1"
_DYNAMO_DEPLOYMENT_NAME_LABEL = "nvidia.com/dynamo-graph-deployment-name"

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

_OWNER_WORKLOAD_LABELS: dict[str, str] = {
    _LWS_RESOURCE: _LWS_NAME_LABEL,
    _DYNAMO_GRAPH_DEPLOYMENT_RESOURCE: _DYNAMO_DEPLOYMENT_NAME_LABEL,
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
    owner_kind: str
    owner_name: str
    owner_uid: str
    podgroup_shape: NotRequired[str]
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
    history_complete: NotRequired[bool]


@dataclass
class _AttemptEvidence:
    """Inputs for one canonical reconstructed KAI attempt."""

    attempt: PodGroupAttempt
    podgroup: dict | None = None
    owner_object: dict | None = None
    pods: dict[str, dict] = field(default_factory=dict)
    jobs: dict[str, dict] = field(default_factory=dict)
    placements: dict[str, PlacementRecord] = field(default_factory=dict)
    retained_coverage: list[dict] = field(default_factory=list)


@dataclass
class _WorkloadEvidence:
    """Raw live or retained evidence collected before attempt reconstruction."""

    namespace: str
    pod_group_name: str
    observed_until: datetime
    podgroup: dict | None = None
    owner_object: dict | None = None
    placements: dict[str, PlacementRecord] = field(default_factory=dict)
    retained_coverage: list[dict] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)


class _PodEvidence(NamedTuple):
    """Summarize scheduler evidence from verified member Pods.

    Attributes:
        nodes: Distinct assigned node names.
        pod_names: Distinct member Pod names.
        pod_nodes: Current Pod-name-to-node mapping.
        placements: Live Pod placements with provenance.
        node_loss_nodes: Assigned nodes whose Pods report node loss.
        has_node_failure: Whether any member reports node failure, including a
            Pod with no assigned node.
    """

    nodes: list[str]
    pod_names: list[str]
    pod_nodes: dict[str, str]
    placements: list[PodPlacement]
    node_loss_nodes: frozenset[str]
    has_node_failure: bool


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


def _latest_resources_by_uid(
    snapshots: list[dict],
) -> dict[str, dict]:
    """Restore the latest resource version, preferring full live data on ties.

    Daemon snapshots omit spec. Preserve the latest available full spec for
    the same UID even when a newer retained status wins. This spec is the last
    available value, not a claim that it was fetched at the selected version.
    """
    selected: dict[str, tuple[tuple, dict]] = {}
    full: dict[str, tuple[tuple, dict]] = {}
    for snapshot in snapshots:
        metadata = snapshot.get("metadata", {})
        uid = metadata.get("uid")
        observed_at = _parse_dt(snapshot.get("observedAt"))
        if not uid or observed_at is None:
            continue
        resource_version = metadata.get("resourceVersion")
        if resource_version is None:
            resource_version = ""
        resource_version = str(resource_version)
        key = (
            bool(resource_version),
            (int(resource_version) if resource_version.isdigit() else -1),
            resource_version,
            "spec" in snapshot,
            observed_at,
            json.dumps(
                snapshot,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
        if "spec" in snapshot and (uid not in full or key > full[uid][0]):
            full[uid] = (key, snapshot)
        if uid not in selected or key > selected[uid][0]:
            selected[uid] = (key, snapshot)
    return {
        uid: resource_from_snapshot(
            {
                **snapshot,
                **({"spec": full[uid][1]["spec"]} if uid in full else {}),
            }
        )
        for uid, (_key, snapshot) in selected.items()
    }


def _split_csv(value: str) -> list[str]:
    """CSV-split a string, stripping whitespace and dropping empty entries."""
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _index_pods_by_pod_group(pods: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group Pods by `(namespace, pod_group_name)`, skipping Pods that have no
    `pod-group-name` annotation or no namespace."""
    index: dict[tuple[str, str], list[dict]] = {}
    for pod in pods:
        meta = pod.get("metadata", {})
        namespace = meta.get("namespace")
        pod_group_name = meta.get("annotations", {}).get(_POD_GROUP_ANNOTATION)
        if namespace and pod_group_name:
            index.setdefault((namespace, pod_group_name), []).append(pod)
    return index


def _collect_pod_evidence(pods: list[dict]) -> _PodEvidence:
    """Collect placement and node-loss evidence from verified member Pods."""
    nodes: set[str] = set()
    pod_names: set[str] = set()
    pod_nodes: dict[str, str] = {}
    placements: list[PodPlacement] = []
    node_loss_nodes: set[str] = set()
    has_node_failure = False
    for pod in pods:
        metadata = pod.get("metadata", {})
        pod_uid = metadata.get("uid")
        pod_name = metadata.get("name")
        node_name = pod.get("spec", {}).get("nodeName")
        if pod_name:
            pod_names.add(pod_name)
        if node_name:
            nodes.add(node_name)
            if pod_name:
                pod_nodes[pod_name] = node_name
        if pod_node_failure_reason(pod):
            has_node_failure = True
            if node_name:
                node_loss_nodes.add(node_name)
        if pod_uid and pod_name and node_name:
            placements.append(
                {
                    "pod_uid": pod_uid,
                    "pod_name": pod_name,
                    "node_name": node_name,
                    "source": "live_pod",
                }
            )
    return _PodEvidence(
        sorted(nodes),
        sorted(pod_names),
        pod_nodes,
        placements,
        frozenset(node_loss_nodes),
        has_node_failure,
    )


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


def _name_matches_owner(name: str, owner_name: str) -> bool:
    """Return whether `name` equals `owner_name` or has prefix `<owner_name>-`."""
    return name == owner_name or name.startswith(f"{owner_name}-")


def _rayjob_child_matches(
    labels: dict,
    controller_name: str,
    owner_name: str,
    owner_object: dict,
) -> bool:
    """Return whether a RayJob child Pod matches the RayJob or its RayCluster."""
    ray_cluster_name = ((owner_object.get("status") or {}).get("rayClusterName") or "").strip()
    if cluster_label := labels.get(_RAY_CLUSTER_LABEL):
        return cluster_label == ray_cluster_name if ray_cluster_name else _name_matches_owner(cluster_label, owner_name)
    if ray_cluster_name and controller_name == ray_cluster_name:
        return True
    return _name_matches_owner(controller_name, owner_name)


def _child_controller_matches_owner(
    pod: dict,
    controller: dict,
    owner: dict,
    owner_object: dict,
) -> bool:
    """Return whether a child-controlled Pod's controller or workload label matches `owner`."""
    owner_name = owner.get("name") or ""
    controller_name = controller.get("name") or ""
    owner_kind = owner["kind"]
    if owner_kind == _DEPLOYMENT_RESOURCE:
        return bool(owner_name) and controller_name.startswith(f"{owner_name}-")
    if owner_kind == _RAYJOB_RESOURCE:
        labels = (pod.get("metadata") or {}).get("labels") or {}
        return bool(owner_name) and _rayjob_child_matches(labels, controller_name, owner_name, owner_object)
    labels = (pod.get("metadata") or {}).get("labels") or {}
    if (label_key := _OWNER_WORKLOAD_LABELS.get(owner_kind)) and (label_value := labels.get(label_key)):
        return label_value == owner_name
    return bool(owner_name) and _name_matches_owner(controller_name, owner_name)


def _pod_matches_non_job_owner(pod: dict, owner: dict, owner_object: dict) -> bool:
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
    return _podgroup_name_embeds_owner_uid(pod_group_name, owner) and _child_controller_matches_owner(
        pod, controller, owner, owner_object
    )


def _member_pods(
    owner: dict,
    owner_job_uids: set[str],
    pods: list[dict],
    owner_object: dict,
) -> list[dict]:
    """From Pods already matched by `pod-group-name` and namespace, keep only
    those belonging to the verified workload owner.

    For Job and JobSet owners, Pods must be controlled by the owning Job or one
    of the JobSet's current child Jobs. For non-Job workload owners, Pods must
    carry positive ownership evidence matching the verified owner UID via
    `kai.scheduler/top-owner-metadata`, a direct controller ownerReference of
    the same kind, or an expected child-controller kind whose name or workload
    label matches the owner on a KAI PodGroup whose name embeds the verified
    owner UID.

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
    return [pod for pod in pods if _pod_matches_non_job_owner(pod, owner, owner_object)]


def _jobset_child_jobs(owner: dict, jobs: list[dict]) -> list[dict]:
    """Return Jobs controlled by this exact JobSet incarnation."""
    return [
        job
        for job in jobs
        if (controller := _controller_owner(job.get("metadata", {}))) is not None
        and controller["kind"] == "JobSet"
        and controller["uid"] == owner["uid"]
        and job.get("metadata", {}).get("uid")
    ]


def _podgroup_shape(spec: dict) -> str:
    """Return whether a PodGroup spec is hierarchical or flat owned."""
    return _SHAPE_HIERARCHICAL if spec.get("subGroups") else _SHAPE_OWNED


def _non_negative_int(value: object) -> int | None:
    """Return an integer only when a Kubernetes field is non-negative.

    Booleans are excluded even though `bool` subclasses `int`; accepting them
    would turn malformed JSON into a valid count.

    Args:
        value: Decoded Kubernetes field value.

    Returns:
        The non-negative integer, or None for every other value.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _job_end_time(job: dict) -> datetime | None:
    """Completion time of a finished Job, or None while it is still active."""
    status = job.get("status", {})
    completed = _parse_dt(status.get("completionTime"))
    if completed is not None:
        return completed
    for condition in status.get("conditions", []):
        if condition.get("type") == "Failed" and condition.get("status") == "True":
            return _parse_dt(condition.get("lastTransitionTime"))
    return None


def _owner_group(owner: dict) -> str | None:
    """Return the API group used to decide whether an owner is supported.

    KAI's annotation may provide `group` directly, while Kubernetes owner
    references provide only `apiVersion`.

    Args:
        owner: Resolved KAI owner or Kubernetes owner reference.

    Returns:
        The explicit or API-version-derived group, or None for a legacy Job
        reference with neither value.
    """
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
    status = jobset.get("status", {})
    terminal_state = status.get("terminalState")
    condition_types = {terminal_state} if terminal_state in {"Completed", "Failed"} else {"Completed", "Failed"}
    terminals = [
        (transition, condition["type"])
        for condition in status.get("conditions", [])
        if condition.get("type") in condition_types
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


def _dynamo_failed_state(dgd: dict) -> str | None:
    """Return the native Failed owner_state for a failed DynamoGraphDeployment, or None."""
    status = dgd.get("status") or {}
    state = (status.get("state") or "").strip().lower()
    cond = _dynamo_failure_condition(dgd)
    if state != "failed" and cond is None:
        return None
    raw = (cond.get("reason") if cond is not None else None) or status.get("reason")
    reason = _dynamo_reason(raw)
    return f"Failed:{reason}" if reason else "Failed"


def _lws_end_time(lws: dict) -> datetime | None:
    """Failure transition time of a LeaderWorkerSet, or None while active."""
    failed = _lws_failed(lws)
    return failed[0] if failed is not None else None


_END_TIME_RESOLVERS = {
    _JOB_RESOURCE: _job_end_time,
    _JOBSET_RESOURCE: _jobset_end_time,
    _PYTORCHJOB_RESOURCE: _pytorchjob_end_time,
    _RAYJOB_RESOURCE: _rayjob_end_time,
    _LWS_RESOURCE: _lws_end_time,
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


_FAILURE_REASON_RESOLVERS = {
    _JOB_RESOURCE: _job_failure_reason,
    _JOBSET_RESOURCE: _jobset_failure_reason,
    _PYTORCHJOB_RESOURCE: _pytorchjob_failure_reason,
    _RAYJOB_RESOURCE: _rayjob_failure_reason,
    _LWS_RESOURCE: _lws_failure_reason,
}


def _owner_failure_reason(owner: dict, owner_object: dict) -> str | None:
    """Return the failure reason string when `owner_object` failed, or None when it completed."""
    resolver = _FAILURE_REASON_RESOLVERS.get(owner["kind"])
    return resolver(owner_object) if resolver is not None else None


_TERMINAL_PREDICATES = {
    _PYTORCHJOB_RESOURCE: _pytorchjob_is_terminal,
    _RAYJOB_RESOURCE: _rayjob_is_terminal,
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
    if owner["kind"] == _DYNAMO_GRAPH_DEPLOYMENT_RESOURCE and (failed := _dynamo_failed_state(owner_object)):
        return failed
    return "Active"


def _rayjob_stopped_state(rayjob: dict) -> str:
    """Return the native owner_state for a stopped RayJob."""
    reason = ((rayjob.get("status") or {}).get("reason") or "").strip()
    return f"Stopped:{reason}" if reason else "Stopped"


def _owner_state(owner: dict, owner_object: dict, owner_jobs: list[dict]) -> tuple[str, str]:
    """Return (service-facing state, native owner state).

    Service state is COMPLETED, FAILED, TIMEOUT, CANCELLED, or RUNNING
    (including while suspended, rollout-stalled, or reconciliation-failed).
    Native state keeps the owner's own wording, e.g.
    "Failed:BackoffLimitExceeded".
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
    if owner_object.get("metadata", {}).get("uid") != owner["uid"]:
        return f"owner {owner['kind']} uid mismatch"
    return None


def _base_context(
    *,
    context_id: str,
    pod_group_uid: str,
    namespace: str,
    name: str,
    owner: dict,
    owner_state: str,
    start_time: datetime,
    end_time: datetime,
    state: str,
    podgroup: dict,
) -> KaiExecutionContext:
    """Build fields shared by scheduled and unscheduled KAI contexts."""
    spec = podgroup.get("spec") or {}
    context: KaiExecutionContext = {
        "id": context_id,
        "attempt_id": context_id,
        "workload_type": "kai",
        "scheduler": "kai",
        "job_id": f"{namespace}/{name}",
        "pod_group_uid": pod_group_uid,
        "namespace": namespace,
        "name": name,
        "owner_kind": owner["kind"],
        "owner_name": owner["name"],
        "owner_uid": owner["uid"],
        "owner_state": owner_state,
        "nodes": [],
        "start_time": start_time,
        "end_time": end_time,
        "state": state,
    }
    if "spec" in podgroup:
        context["podgroup_shape"] = _podgroup_shape(spec)
    if queue_name := spec.get("queue"):
        context["queue_name"] = queue_name
    if (min_member := _non_negative_int(spec.get("minMember"))) is not None:
        context["min_member"] = min_member
    status = podgroup.get("status") or {}
    conditions = status.get("schedulingConditions")
    if isinstance(conditions, list):
        context["scheduling_conditions"] = conditions
    if "resourcesStatus" in status:
        context["resources_status"] = status["resourcesStatus"]
    return context


def _merge_placement_evidence(
    context: KaiExecutionContext,
    records: list[PlacementRecord],
    *,
    allowed_pod_uids: frozenset[str] | None,
) -> set[str]:
    """Merge daemon placements and return accepted node-loss nodes."""
    placements = {
        placement["pod_uid"]: placement for placement in context.get("pod_placements", []) if placement.get("pod_uid")
    }
    node_loss_nodes: set[str] = set()
    for record in records:
        if allowed_pod_uids is not None and record.pod_uid not in allowed_pod_uids:
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
        if record.node_failure_reason == "NodeLost" or record.node_failure_reason in NODE_LOSS_DISRUPTION_REASONS:
            node_loss_nodes.add(record.node_name)
    if not placements:
        return node_loss_nodes

    context["pod_placements"] = [placements[pod_uid] for pod_uid in sorted(placements)]
    context["nodes"] = sorted(set(context["nodes"]) | {placement["node_name"] for placement in placements.values()})
    context["pod_names"] = sorted(
        set(context.get("pod_names", [])) | {placement["pod_name"] for placement in placements.values()}
    )
    pod_nodes = dict(context.get("pod_nodes", {}))
    for placement in placements.values():
        pod_nodes.setdefault(
            placement["pod_name"],
            placement["node_name"],
        )
    context["pod_nodes"] = pod_nodes
    return node_loss_nodes


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


def _earliest_podgroup_start(members: list[dict], podgroup_start: datetime) -> datetime:
    """Return the earliest creation timestamp across `podgroup_start` and `members`."""
    member_starts = [
        created
        for pod in members
        if (created := _parse_dt((pod.get("metadata") or {}).get("creationTimestamp"))) is not None
    ]
    return min(podgroup_start, *member_starts) if member_starts else podgroup_start


def _attach_pod_evidence_fields(
    ctx: KaiExecutionContext,
    pod_evidence: _PodEvidence,
    owner_jobs: list[dict],
) -> None:
    """Attach Pod evidence, failed-node attribution, and Job time limit to `ctx`."""
    ctx["nodes"] = pod_evidence.nodes
    if (time_limit := job_time_limit_minutes(owner_jobs)) is not None:
        ctx["time_limit"] = time_limit
    if ctx["state"] == "NODE_FAIL" and pod_evidence.node_loss_nodes:
        ctx["failed_node"] = min(pod_evidence.node_loss_nodes)
    if pod_evidence.pod_names:
        ctx["pod_names"] = pod_evidence.pod_names
    if pod_evidence.pod_nodes:
        ctx["pod_nodes"] = pod_evidence.pod_nodes


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
    members = _member_pods(owner, owner_job_uids, pods, owner_object)
    start_time = _earliest_podgroup_start(members, start_time)
    pod_evidence = _collect_pod_evidence(members)

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
    if state != "COMPLETED" and pod_evidence.has_node_failure:
        state = "NODE_FAIL"
    if not pod_evidence.pod_nodes:
        logger.warning(f"podgroup {namespace}/{name}: node list unavailable from member Pods")

    ctx = _base_context(
        context_id=uid,
        pod_group_uid=uid,
        namespace=namespace,
        name=name,
        owner=owner,
        owner_state=owner_state,
        start_time=start_time,
        end_time=end_time,
        state=state,
        podgroup=podgroup,
    )
    ctx["podgroup_shape"] = shape
    _attach_pod_evidence_fields(ctx, pod_evidence, owner_jobs)
    return ctx


def _history_attempt_state(
    attempt: PodGroupAttempt,
) -> tuple[str, str]:
    """Translate reconstructed outcomes into service and native owner states."""
    terminal = attempt.terminal
    if terminal is None:
        return ("RUNNING", "Active") if attempt.open else ("UNKNOWN", "Unknown")
    terminal_type = terminal["type"]
    reason = str(terminal.get("reason") or "")
    if attempt.owner_kind == "Job":
        if terminal_type == "Complete":
            return "COMPLETED", "Complete"
        native = f"Failed:{reason}" if reason else "Failed"
        return (
            "TIMEOUT" if reason == "DeadlineExceeded" else "FAILED",
            native,
        )
    if terminal_type == "Completed":
        return "COMPLETED", "Completed"
    native = f"Failed:{reason}" if reason else "Failed"
    return "FAILED", native


def _history_complete(
    attempt: PodGroupAttempt,
    coverage: dict,
) -> bool:
    """Require resolved boundaries, membership, and watch coverage for the full attempt."""
    if not attempt.boundary_resolved or not attempt.membership_resolved:
        return False
    required_sources = {"kai-podgroups", "jobs", "pods"}
    if attempt.owner_kind == "JobSet":
        required_sources.add("jobsets")
    return retained_history_covers_window(
        coverage,
        required_sources,
        attempt.start_time,
        attempt.end_time,
    )


def _context_from_attempt_evidence(
    evidence: _AttemptEvidence,
) -> KaiExecutionContext:
    """Construct one context after live and retained evidence is combined."""
    attempt = evidence.attempt
    pods = [pod for uid, pod in evidence.pods.items() if uid in attempt.pod_uids]
    pod_evidence = _collect_pod_evidence(pods)
    state, owner_state = _history_attempt_state(attempt)
    owner = {
        "kind": attempt.owner_kind,
        "name": attempt.owner_name,
        "uid": attempt.owner_uid,
    }
    allowed_job_uids = attempt.job_uids | ({attempt.owner_uid} if attempt.owner_kind == "Job" else set())
    jobs = [job for uid, job in evidence.jobs.items() if uid in allowed_job_uids]
    if evidence.owner_object is not None and attempt.terminal is None:
        state, owner_state = _owner_state(
            owner,
            evidence.owner_object,
            jobs,
        )
    elif (
        state == "FAILED" and attempt.owner_kind == "JobSet" and earliest_job_failure_reason(jobs) == "DeadlineExceeded"
    ):
        state = "TIMEOUT"

    podgroup = evidence.podgroup or {}
    context = _base_context(
        context_id=attempt.attempt_id,
        pod_group_uid=attempt.pod_group_uid,
        namespace=attempt.namespace,
        name=attempt.pod_group_name,
        owner=owner,
        owner_state=owner_state,
        start_time=attempt.start_time,
        end_time=attempt.end_time,
        state=state,
        podgroup=podgroup,
    )
    context["nodes"] = pod_evidence.nodes
    if pod_evidence.pod_names:
        context["pod_names"] = pod_evidence.pod_names
    if pod_evidence.pod_nodes:
        context["pod_nodes"] = pod_evidence.pod_nodes
    if pod_evidence.placements:
        context["pod_placements"] = pod_evidence.placements
    retained_node_loss_nodes = _merge_placement_evidence(
        context,
        list(evidence.placements.values()),
        allowed_pod_uids=attempt.pod_uids,
    )
    node_loss_nodes = retained_node_loss_nodes | set(pod_evidence.node_loss_nodes)
    if (retained_node_loss_nodes or pod_evidence.has_node_failure) and context["state"] != "COMPLETED":
        context["state"] = "NODE_FAIL"
    if context["state"] == "NODE_FAIL" and node_loss_nodes:
        context["failed_node"] = min(node_loss_nodes)
    if evidence.retained_coverage:
        context["history_complete"] = any(
            _history_complete(attempt, coverage) for coverage in evidence.retained_coverage
        )
    if (time_limit := job_time_limit_minutes(jobs)) is not None:
        context["time_limit"] = time_limit
    return context


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
        history_reader=None,
    ):
        """Configure live discovery and optional durable evidence readers.

        Args:
            kubeconfig: Path passed to the Kubernetes API client.
            context: Kubeconfig context passed to the Kubernetes API client.
            namespace: Default namespace for bare IDs and time-range scope.
            timeout_seconds: Timeout for each Kubernetes API request.
            placement_records_enabled: Whether to read durable placements.
            placement_reader: Reader for durable placement lookups.
            history_reader: Reader for retained Kubernetes snapshots,
                independent of placement enablement.

        Raises:
            ValueError: If `placement_records_enabled` is not a boolean.
        """
        self.namespace = namespace or ""
        self.timeout_seconds = timeout_seconds
        if not isinstance(placement_records_enabled, bool):
            raise ValueError("placement_records_enabled must be a bool")
        self.placement_records_enabled = placement_records_enabled
        self.placement_reader = placement_reader
        self.history_reader = history_reader
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

    def _resolve_id_ref(self, ref: str) -> tuple[str, str]:
        """Split a `<namespace>/<name>` or bare `<name>` ID reference and validate namespace scope."""
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
        return ns, name

    def discover_by_ids(self, ids: list[str]) -> list[KaiExecutionContext]:
        """Look up PodGroups by `<namespace>/<name>` (or bare `<name>` when the
        adapter is namespace-scoped). Failed lookups use retained history when
        available; otherwise missing PodGroups are skipped and other errors raise."""
        podgroups: list[dict] = []
        retained_evidence: list[_WorkloadEvidence] = []
        for ref in ids:
            ns, name = self._resolve_id_ref(ref)
            histories = (
                self.history_reader.find_histories(
                    ns,
                    _PODGROUP_RESOURCE,
                    name=name,
                )
                if self.history_reader is not None
                else []
            )
            if histories:
                retained_evidence.extend(
                    self._evidence_from_histories(
                        histories,
                        pod_group_name=name,
                    )
                )
            try:
                obj = self._kubernetes.get_one(
                    _PODGROUP_RESOURCE,
                    name,
                    namespace=ns,
                    group=_PODGROUP_GROUP,
                )
            except KubernetesClientError as error:
                if histories:
                    logger.warning(f"live PodGroup lookup failed for {ns}/{name}; using retained history")
                    continue
                if error.status == _HTTP_NOT_FOUND:
                    continue
                raise
            if obj is not None:
                podgroups.append(obj)
        return self._finalize(podgroups, retained_evidence)

    @staticmethod
    def _deduplicate_snapshots(items: list[_WorkloadEvidence]) -> list[dict]:
        """Return unique snapshots across `items` in first-seen order."""
        snapshots: list[dict] = []
        seen_snapshots: set[str] = set()
        for item in items:
            for snapshot in item.snapshots:
                key = json.dumps(
                    snapshot,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                if key not in seen_snapshots:
                    seen_snapshots.add(key)
                    snapshots.append(snapshot)
        return snapshots

    @staticmethod
    def _combine_attempt_evidence(
        evidence_items: list[_WorkloadEvidence],
    ) -> list[_AttemptEvidence]:
        """Combine snapshots by namespace/name before deriving canonical attempt IDs.

        Reconstruct once, filter resources by attempt membership, and attach live
        owner state only to the latest attempt for each PodGroup UID."""
        workload_groups: dict[
            tuple[str, str],
            list[_WorkloadEvidence],
        ] = {}
        for evidence in evidence_items:
            workload_groups.setdefault(
                (
                    evidence.namespace,
                    evidence.pod_group_name,
                ),
                [],
            ).append(evidence)

        combined: list[_AttemptEvidence] = []
        for (_namespace, pod_group_name), items in workload_groups.items():
            snapshots = KaiExecutionContextAdapter._deduplicate_snapshots(items)
            attempts = attempts_from_podgroup_history(
                {
                    "snapshots": snapshots,
                    "coverages": [coverage for item in items for coverage in item.retained_coverage],
                },
                pod_group_name=pod_group_name,
                now=max(item.observed_until for item in items),
            )
            if not attempts:
                continue

            resources_by_uid = _latest_resources_by_uid(snapshots)
            pods = {uid: resource for uid, resource in resources_by_uid.items() if resource.get("kind") == "Pod"}
            jobs = {uid: resource for uid, resource in resources_by_uid.items() if resource.get("kind") == "Job"}
            placements: dict[str, PlacementRecord] = {}
            retained_coverage: list[dict] = []
            live_podgroups: dict[str, dict] = {}
            live_owners: dict[str, dict] = {}
            latest_attempts = {attempt.pod_group_uid: attempt for attempt in attempts}
            for item in items:
                placements.update(item.placements)
                retained_coverage.extend(item.retained_coverage)
                pod_group_uid = item.podgroup.get("metadata", {}).get("uid") if item.podgroup is not None else None
                if pod_group_uid:
                    live_podgroups[pod_group_uid] = item.podgroup
                    if item.owner_object is not None:
                        live_owners[pod_group_uid] = item.owner_object

            for attempt in attempts:
                # A Job owner is itself the attempt resource. A JobSet owner
                # contributes only child Jobs already proven by UID.
                allowed_job_uids = attempt.job_uids | ({attempt.owner_uid} if attempt.owner_kind == "Job" else set())
                latest = latest_attempts.get(attempt.pod_group_uid)
                owner_object = (
                    live_owners.get(attempt.pod_group_uid)
                    if latest and latest.attempt_id == attempt.attempt_id
                    else None
                )
                combined.append(
                    _AttemptEvidence(
                        attempt=attempt,
                        podgroup=live_podgroups.get(attempt.pod_group_uid),
                        owner_object=owner_object,
                        pods={uid: pods[uid] for uid in attempt.pod_uids if uid in pods},
                        jobs={uid: jobs[uid] for uid in allowed_job_uids if uid in jobs},
                        placements=dict(placements),
                        retained_coverage=list(retained_coverage),
                    )
                )
        return sorted(
            combined,
            key=lambda evidence: (
                evidence.attempt.start_time,
                evidence.attempt.attempt_id,
            ),
        )

    @staticmethod
    def _evidence_from_histories(
        histories: list[dict],
        *,
        pod_group_name: str | None,
    ) -> list[_WorkloadEvidence]:
        """Group retained snapshots, placements, and coverage by PodGroup for reconstruction."""
        now = datetime.now(tz=timezone.utc)
        evidence_items = []
        for history in histories:
            podgroups = {
                (
                    (metadata.get("namespace") or history.get("namespace") or ""),
                    metadata["name"],
                )
                for snapshot in history.get("snapshots", [])
                if snapshot.get("kind") == "PodGroup"
                and (metadata := snapshot.get("metadata", {})).get("uid")
                and metadata.get("name")
                and (pod_group_name is None or metadata["name"] == pod_group_name)
            }
            placements: dict[str, PlacementRecord] = {}
            for resource in history.get("placements", []):
                placement = PlacementRecord.from_resource(resource)
                placements[placement.pod_uid] = placement
            for namespace, retained_name in sorted(podgroups):
                evidence_items.append(
                    _WorkloadEvidence(
                        namespace=namespace,
                        pod_group_name=retained_name,
                        observed_until=now,
                        placements=dict(placements),
                        retained_coverage=[history.get("coverage", {})],
                        snapshots=list(history.get("snapshots", [])),
                    )
                )
        return evidence_items

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[KaiExecutionContext]:
        """List PodGroups, then keep those whose window intersects `[start, end]`.

        The window is checked after member Pods and owners are resolved:
        a member Pod may predate its PodGroup, and the owner supplies the end.
        A failed list uses retained history when available; otherwise it raises.
        `nodelist` entries may be CSVs of plain node names; a PodGroup
        survives if any member Pod ran on one of them.
        """
        assert_aware(start, "discover_by_time_range.start")
        assert_aware(end, "discover_by_time_range.end")
        if not self.namespace:
            raise ValueError("kai ECA time-range discovery requires eca.kai.namespace")
        histories = []
        retained_evidence: list[_WorkloadEvidence] = []
        if self.history_reader is not None:
            histories = self.history_reader.find_histories(
                self.namespace,
                _PODGROUP_RESOURCE,
                start=start,
                end=end,
            )
            retained_evidence = self._evidence_from_histories(
                histories,
                pod_group_name=None,
            )
        try:
            listing = self._kubernetes.list(
                _PODGROUP_RESOURCE,
                namespace=self.namespace,
                group=_PODGROUP_GROUP,
            )
        except KubernetesClientError:
            if not histories:
                raise
            logger.warning("live PodGroup list failed; using retained history")
            listing = []
        contexts = self._finalize(
            listing,
            retained_evidence,
            time_range=(start, end),
        )
        contexts = [context for context in contexts if context["start_time"] <= end and context["end_time"] >= start]
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
        """Fetch the live owner object, falling back to group lookup if a version is unserved.

        Returns None when the owner is NotFound (404); other API errors (including
        403 Forbidden, handled per PodGroup by `_collect_live_podgroup`) raise.
        """
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
                if error.status != _HTTP_NOT_FOUND:
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

    def _collect_live_podgroup(
        self,
        podgroup: dict,
        owner: dict,
        owner_resource: str,
        pods: list[dict],
        *,
        now: datetime,
    ) -> tuple[_WorkloadEvidence | None, KaiExecutionContext | None]:
        """Resolve one live PodGroup into started-attempt evidence or a live context."""
        meta = podgroup.get("metadata") or {}
        namespace = meta.get("namespace") or ""
        try:
            owner_object, owner_jobs = self._fetch_owner_and_jobs(namespace, owner, owner_resource)
        except KubernetesClientError as error:
            if error.status != _HTTP_FORBIDDEN:
                raise
            logger.warning(
                f"podgroup {namespace}/{meta.get('name')}: "
                f"owner {owner['kind']} {owner['name']!r} lookup forbidden; skipping: {error}"
            )
            return None, None
        owner_job_uids = {uid for job in owner_jobs if (uid := (job.get("metadata") or {}).get("uid")) is not None}
        members = _member_pods(owner, owner_job_uids, pods, owner_object or {})
        live_snapshots = (
            snapshots_from_live_resources(
                podgroup,
                members,
                owner_jobs,
                owner_object,
                now=now,
            )
            if owner["kind"] in {_JOB_RESOURCE, _JOBSET_RESOURCE}
            and _skip_reason(owner, owner_object) is None
            and owner_object is not None
            else []
        )
        if live_snapshots and kai_podgroup_start_timestamp(podgroup):
            records = self._fetch_placement_records(
                {
                    "namespace": namespace,
                    "pod_group_uid": meta.get("uid") or "",
                }
            )
            return (
                _WorkloadEvidence(
                    namespace=namespace,
                    pod_group_name=meta.get("name") or "",
                    observed_until=now,
                    podgroup=podgroup,
                    owner_object=owner_object,
                    placements={record.pod_uid: record for record in records},
                    snapshots=live_snapshots,
                ),
                None,
            )
        return None, _parse_pod_group(
            podgroup,
            owner,
            owner_jobs,
            members,
            owner_object,
            now,
        )

    def _collect_live_evidence(
        self,
        podgroups: list[dict],
        *,
        now: datetime,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> tuple[list[_WorkloadEvidence], list[KaiExecutionContext]]:
        """Collect live started-attempt evidence and unscheduled contexts."""
        # Unsupported owners cannot supply an end time, so drop them before
        # fetching Pods or owners.
        owned: list[tuple[dict, dict, str]] = []
        for podgroup in podgroups:
            meta = podgroup.get("metadata", {})
            owner = _resolve_owner(meta)
            owner_resource = _owner_resource(owner) if owner is not None else None
            if owner is None or owner_resource is None:
                logger.info(
                    f"podgroup {meta.get('namespace')}/{meta.get('name')}: not owned by a supported workload; skipping"
                )
                continue
            owned.append((podgroup, owner, owner_resource))
        if not owned:
            return [], []
        pods_by_key = _index_pods_by_pod_group(self._fetch_pods([pg for pg, _, _ in owned]))
        evidence_items: list[_WorkloadEvidence] = []
        pending_contexts: list[KaiExecutionContext] = []
        for podgroup, owner, owner_resource in owned:
            meta = podgroup.get("metadata") or {}
            pods = pods_by_key.get(
                (meta.get("namespace") or "", meta.get("name") or ""),
                [],
            )
            evidence, context = self._collect_live_podgroup(
                podgroup,
                owner,
                owner_resource,
                pods,
                now=now,
            )
            if evidence is not None:
                evidence_items.append(evidence)
                continue
            if context is None:
                continue
            if time_range is not None:
                start, end = time_range
                if context["start_time"] > end or context["end_time"] < start:
                    continue
            self._merge_placement_records(context)
            pending_contexts.append(context)

        return evidence_items, pending_contexts

    def _finalize(
        self,
        podgroups: list[dict],
        retained_evidence: list[_WorkloadEvidence] | None = None,
        *,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[KaiExecutionContext]:
        """Combine live and retained contexts, retain pending groups, and filter by time."""
        if not podgroups and not retained_evidence:
            return []
        live_evidence, pending_contexts = self._collect_live_evidence(
            podgroups,
            now=datetime.now(tz=timezone.utc),
            time_range=time_range,
        )
        contexts = [
            _context_from_attempt_evidence(evidence)
            for evidence in self._combine_attempt_evidence([*(retained_evidence or []), *live_evidence])
        ]
        admitted_uids = {context["pod_group_uid"] for context in contexts}
        contexts.extend(context for context in pending_contexts if context["pod_group_uid"] not in admitted_uids)
        if time_range is None:
            return contexts
        start, end = time_range
        return [context for context in contexts if context["start_time"] <= end and context["end_time"] >= start]

    def _fetch_placement_records(self, context: KaiExecutionContext) -> list[PlacementRecord]:
        """Read durable placements associated with one PodGroup incarnation.

        Args:
            context: Context containing the namespace and PodGroup UID used by
                the daemon association index.

        Returns:
            Placement records when enabled, otherwise an empty list.

        Raises:
            RuntimeError: If placement reads lack a reader or lookup fails.
        """
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
        """Enrich an unscheduled live context with durable placements.

        Reconstructed attempts filter by proven Pod UIDs. This legacy live
        path instead relies on the daemon's PodGroup association because no
        attempt-specific UID set exists yet.

        Args:
            context: Context to update in place with placements, nodes, and a
                node-failure outcome when supported by retained evidence.
        """
        records = self._fetch_placement_records(context)
        node_loss_nodes = _merge_placement_evidence(
            context,
            records,
            allowed_pod_uids=None,
        )
        if node_loss_nodes and context["state"] != "COMPLETED":
            context["state"] = "NODE_FAIL"
            context["failed_node"] = min(node_loss_nodes)

    def _fetch_pods(self, podgroups: list[dict]) -> list[dict]:
        """List Pods once per distinct namespace.

        Pods carry the PodGroup only as an annotation, which Kubernetes field
        and label selectors cannot select on, so the join is done client-side
        after a namespace-scoped list.
        """
        namespaces = sorted(
            {namespace for podgroup in podgroups if (namespace := podgroup.get("metadata", {}).get("namespace"))}
        )
        pods: list[dict] = []
        for ns in namespaces:
            pods.extend(self._kubernetes.list("Pod", namespace=ns, api_version="v1"))
        return pods
