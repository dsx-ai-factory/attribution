# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KAI Scheduler ExecutionContextAdapter.

Emits one workload-level context per PodGroup, with these KAI specifics:

    ExecutionContext   KAI source
    -----------------  -----------------------------------------------------
    id / attempt_id    Canonical PodGroup UID + scheduling boundary +
                       JobSet restart generation identity. A PodGroup that
                       has not started retains its UID-based live identity.
    job_id             "<namespace>/<name>" of the PodGroup
    start_time         Earliest creation time of the PodGroup or a member Pod
    end_time           Terminal time of the owning Job or JobSet; now()
                       while the owner is active. Pods never determine it.
    nodes              Union of .spec.nodeName across the member Pods
    state              From the owning Job or JobSet: COMPLETED, FAILED,
                       TIMEOUT (a Job failed with DeadlineExceeded), or
                       RUNNING while the owner has no terminal signal,
                       including while suspended. `owner_state` keeps the
                       owner's own wording. A member Pod that reports node
                       loss sets state to NODE_FAIL and sets `failed_node`;
                       a completed one stays COMPLETED without one.

Member Pods are joined on the `pod-group-name` annotation plus namespace,
then validated against the exact owning Job UID — see `_member_pods` for
why membership intentionally does not use timestamps as a discriminator.

The owner comes from the `kai.scheduler/top-owner-metadata` annotation, or
else the PodGroup's controller ownerReference. Only PodGroups owned by a
`batch` Job or `jobset.x-k8s.io` JobSet that still exists with the same UID
are emitted; the owner is the only source of an end time.

A failed Kubernetes read raises; only NotFound is treated as absence.
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
    observations_from_live_resources,
)
from attribution_blueprint.connectors.execution_contexts.kai.helpers import (
    POD_GROUP_ANNOTATION as _POD_GROUP_ANNOTATION,
    controller_owner as _controller_owner,
    resolve_owner as _resolve_owner,
)
from attribution_blueprint.kubernetes.attempts import (
    kai_podgroup_start_timestamp,
    resource_from_observation,
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
_JOB_RESOURCE = "Job"
_JOB_API_VERSION = "batch/v1"
_JOBSET_RESOURCE = "JobSet"
_JOBSET_GROUP = "jobset.x-k8s.io"
_JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"

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
    # Service-facing state: COMPLETED, FAILED, TIMEOUT or RUNNING.
    state: str

    job_id: str
    pod_group_uid: str
    attempt_id: str
    namespace: str
    name: str
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
    observations: list[dict] = field(default_factory=list)


class _PodEvidence(NamedTuple):
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


def _collect_pod_evidence(pods: list[dict]) -> _PodEvidence:
    """Collect placement and node-loss evidence from verified member Pods."""
    nodes: set[str] = set()
    pod_names: set[str] = set()
    pod_nodes: dict[str, str] = {}
    placements: list[PodPlacement] = []
    node_loss_nodes: set[str] = set()
    has_node_failure = False
    for pod in pods:
        metadata = pod.get("metadata") or {}
        pod_uid = metadata.get("uid")
        pod_name = metadata.get("name")
        node_name = (pod.get("spec") or {}).get("nodeName")
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


def _member_pods(owner_job_uids: set[str], pods: list[dict]) -> list[dict]:
    """From Pods already matched by `pod-group-name` and namespace, keep only
    those controlled by the current Job or JobSet child Jobs.

    No additional time-based filter: neither `managedFields` time (reflects
    any owned field, not `pod-group-name` specifically) nor a Pod's own
    `creationTimestamp` (can legitimately predate its PodGroup) reliably
    proves which PodGroup incarnation a Pod belongs to. A PodGroup recreated
    under the same name with the same still-live owning Job is therefore a
    known, accepted limitation.
    """
    return [
        pod
        for pod in pods
        if (controller := _controller_owner(pod.get("metadata") or {})) is not None
        and controller["kind"] == "Job"
        and controller["uid"] in owner_job_uids
    ]


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


def _non_negative_int(value: object) -> int | None:
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
    group = owner.get("group")
    api_version = owner.get("apiVersion")
    if group is None and isinstance(api_version, str) and "/" in api_version:
        group = api_version.split("/", 1)[0]
    return group


def _owner_resource(owner: dict) -> str | None:
    group = _owner_group(owner)
    if owner["kind"] == "Job" and group in (None, "batch"):
        return _JOB_RESOURCE
    if owner["kind"] == "JobSet" and group == "jobset.x-k8s.io":
        return _JOBSET_RESOURCE
    return None


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


def _owner_end_time(owner: dict, owner_object: dict) -> datetime | None:
    if owner["kind"] == "Job":
        return _job_end_time(owner_object)
    return _jobset_end_time(owner_object)


def _condition(owner_object: dict, condition_type: str) -> dict | None:
    for condition in (owner_object.get("status") or {}).get("conditions") or []:
        if (condition or {}).get("type") == condition_type and condition.get("status") == "True":
            return condition
    return None


def _owner_state(owner: dict, owner_object: dict, owner_jobs: list[dict]) -> tuple[str, str]:
    """Return (service-facing state, native owner state).

    Service state is COMPLETED, FAILED, TIMEOUT, or RUNNING (including while
    suspended), derived from the same terminal signal as the end time. Native
    state keeps the owner's own wording, e.g. "Failed:BackoffLimitExceeded".
    """
    if _owner_end_time(owner, owner_object) is None:
        return "RUNNING", "Suspended" if _condition(owner_object, "Suspended") else "Active"

    if owner["kind"] == "Job":
        completed = _parse_dt((owner_object.get("status") or {}).get("completionTime")) is not None
        failed = None if completed else _condition(owner_object, "Failed")
    else:
        terminal = _jobset_terminal(owner_object)
        failed = _condition(owner_object, "Failed") if terminal and terminal[1] == "Failed" else None
    if failed is None:
        return "COMPLETED", "Complete" if owner["kind"] == "Job" else "Completed"

    owner_reason = (failed.get("reason") or "").strip()
    native = f"Failed:{owner_reason}" if owner_reason else "Failed"
    failure_reason = earliest_job_failure_reason(owner_jobs) or owner_reason
    return ("TIMEOUT" if failure_reason == "DeadlineExceeded" else "FAILED"), native


def _skip_reason(owner: dict, owner_object: dict | None) -> str | None:
    """Why the owner cannot supply a window, or None when it can."""
    if owner_object is None:
        return f"owner {owner['kind']} not found"
    if (owner_object.get("metadata") or {}).get("uid") != owner["uid"]:
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
        placement["pod_uid"]: placement
        for placement in context.get("pod_placements") or []
        if placement.get("pod_uid")
    }
    node_loss_nodes: set[str] = set()
    for record in records:
        if (
            allowed_pod_uids is not None
            and record.pod_uid not in allowed_pod_uids
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
        if (
            record.node_failure_reason == "NodeLost"
            or record.node_failure_reason
            in NODE_LOSS_DISRUPTION_REASONS
        ):
            node_loss_nodes.add(record.node_name)
    if not placements:
        return node_loss_nodes

    context["pod_placements"] = [
        placements[pod_uid] for pod_uid in sorted(placements)
    ]
    context["nodes"] = sorted(
        set(context["nodes"])
        | {
            placement["node_name"]
            for placement in placements.values()
        }
    )
    context["pod_names"] = sorted(
        set(context.get("pod_names") or [])
        | {
            placement["pod_name"]
            for placement in placements.values()
        }
    )
    pod_nodes = dict(context.get("pod_nodes") or {})
    for placement in placements.values():
        pod_nodes.setdefault(
            placement["pod_name"],
            placement["node_name"],
        )
    context["pod_nodes"] = pod_nodes
    return node_loss_nodes


def _parse_pod_group(
    podgroup: dict,
    owner: dict,
    owner_jobs: list[dict],
    pods: list[dict],
    owner_object: dict | None,
    now: datetime,
) -> KaiExecutionContext | None:
    """Normalize one PodGroup and its member Pods into wire format.

    `owner` identifies the PodGroup's Job or JobSet owner and `owner_object`
    is the live owner object when one was found.

    Returns None when the PodGroup lacks a uid, name, namespace, or a
    parseable creationTimestamp (without a start the orchestrator cannot build
    a valid attributor window), or when the owner cannot supply an end.
    """
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

    owner_job_uids = {job_uid for job in owner_jobs if (job_uid := (job.get("metadata") or {}).get("uid")) is not None}
    members = _member_pods(owner_job_uids, pods)
    member_starts = [
        created
        for pod in members
        if (created := _parse_dt((pod.get("metadata") or {}).get("creationTimestamp"))) is not None
    ]
    if member_starts:
        start_time = min(start_time, *member_starts)
    pod_evidence = _collect_pod_evidence(members)

    skip_reason = _skip_reason(owner, owner_object)
    if skip_reason is not None:
        logger.info(f"podgroup {namespace}/{name}: {skip_reason}; skipping")
        return None
    owner_end = _owner_end_time(owner, owner_object)
    if owner_end is None:
        logger.debug(f"podgroup {namespace}/{name}: owner {owner['kind']} still active; using now() for end_time")
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
    ctx["nodes"] = pod_evidence.nodes
    if (time_limit := job_time_limit_minutes(owner_jobs)) is not None:
        ctx["time_limit"] = time_limit
    if state == "NODE_FAIL" and pod_evidence.node_loss_nodes:
        ctx["failed_node"] = min(pod_evidence.node_loss_nodes)
    if pod_evidence.pod_names:
        ctx["pod_names"] = pod_evidence.pod_names
    if pod_evidence.pod_nodes:
        ctx["pod_nodes"] = pod_evidence.pod_nodes

    return ctx


def _history_attempt_state(
    attempt: PodGroupAttempt,
) -> tuple[str, str]:
    terminal = attempt.terminal
    if terminal is None:
        return (
            ("RUNNING", "Active")
            if attempt.open
            else ("UNKNOWN", "Unknown")
        )
    terminal_type = terminal["type"]
    reason = str(terminal.get("reason") or "")
    if attempt.owner_kind == "Job":
        if terminal_type == "Complete":
            return "COMPLETED", "Complete"
        native = f"Failed:{reason}" if reason else "Failed"
        return (
            "TIMEOUT"
            if reason == "DeadlineExceeded"
            else "FAILED",
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
    if (
        not attempt.boundary_resolved
        or not attempt.membership_resolved
    ):
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
    pods = [
        pod
        for uid, pod in evidence.pods.items()
        if uid in attempt.pod_uids
    ]
    pod_evidence = _collect_pod_evidence(pods)
    state, owner_state = _history_attempt_state(attempt)
    owner = {
        "kind": attempt.owner_kind,
        "name": attempt.owner_name,
        "uid": attempt.owner_uid,
    }
    allowed_job_uids = attempt.job_uids | (
        {attempt.owner_uid}
        if attempt.owner_kind == "Job"
        else set()
    )
    jobs = [
        job
        for uid, job in evidence.jobs.items()
        if uid in allowed_job_uids
    ]
    if evidence.owner_object is not None and attempt.terminal is None:
        state, owner_state = _owner_state(
            owner,
            evidence.owner_object,
            jobs,
        )
    elif (
        state == "FAILED"
        and attempt.owner_kind == "JobSet"
        and earliest_job_failure_reason(jobs)
        == "DeadlineExceeded"
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
    node_loss_nodes = (
        retained_node_loss_nodes | set(pod_evidence.node_loss_nodes)
    )
    if (
        retained_node_loss_nodes
        or pod_evidence.has_node_failure
    ) and context["state"] != "COMPLETED":
        context["state"] = "NODE_FAIL"
    if node_loss_nodes:
        context["failed_node"] = min(node_loss_nodes)
    if evidence.retained_coverage:
        context["history_complete"] = any(
            _history_complete(attempt, coverage)
            for coverage in evidence.retained_coverage
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

    def discover_by_ids(self, ids: list[str]) -> list[KaiExecutionContext]:
        """Look up PodGroups by `<namespace>/<name>` (or bare `<name>` when the
        adapter is namespace-scoped). PodGroups that do not exist are skipped;
        any other failed lookup raises."""
        podgroups: list[dict] = []
        retained_evidence: list[_WorkloadEvidence] = []
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
                raise ValueError(
                    f"kai ECA --ids: {ref!r} is outside configured namespace "
                    f"{self.namespace!r}"
                )
            histories = []
            if self.history_reader is not None:
                histories = self.history_reader.find_histories(
                    ns,
                    _PODGROUP_RESOURCE,
                    name=name,
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
                    logger.warning(
                        f"live PodGroup lookup failed for {ns}/{name}; "
                        "using retained history"
                    )
                    continue
                if error.status == 404:
                    continue
                raise
            if obj is not None:
                podgroups.append(obj)
        return self._finalize(podgroups, retained_evidence)

    @staticmethod
    def _combine_attempt_evidence(
        evidence_items: list[_WorkloadEvidence],
    ) -> list[_AttemptEvidence]:
        """Combine observations by namespace/name before deriving canonical attempt IDs.

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
        for (_namespace, pod_group_name), items in (
            workload_groups.items()
        ):
            observations: list[dict] = []
            seen_observations: set[str] = set()
            for item in items:
                for observation in item.observations:
                    key = json.dumps(
                        observation,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    if key not in seen_observations:
                        seen_observations.add(key)
                        observations.append(observation)

            attempts = attempts_from_podgroup_history(
                {
                    "observations": observations,
                    "coverages": [coverage for item in items for coverage in item.retained_coverage],
                },
                pod_group_name=pod_group_name,
                now=max(item.observed_until for item in items),
            )
            if not attempts:
                continue

            resources_by_uid = {
                metadata["uid"]: resource_from_observation(
                    observation
                )
                for observation in observations
                if (
                    metadata := observation.get("metadata") or {}
                ).get("uid")
            }
            pods = {
                uid: resource
                for uid, resource in resources_by_uid.items()
                if resource.get("kind") == "Pod"
            }
            jobs = {
                uid: resource
                for uid, resource in resources_by_uid.items()
                if resource.get("kind") == "Job"
            }
            placements: dict[str, PlacementRecord] = {}
            retained_coverage: list[dict] = []
            live_podgroups: dict[str, dict] = {}
            live_owners: dict[str, dict] = {}
            latest_attempts = {
                attempt.pod_group_uid: attempt
                for attempt in attempts
            }
            for item in items:
                placements.update(item.placements)
                retained_coverage.extend(item.retained_coverage)
                pod_group_uid = (
                    (item.podgroup.get("metadata") or {}).get(
                        "uid"
                    )
                    if item.podgroup is not None
                    else None
                )
                if pod_group_uid:
                    live_podgroups[pod_group_uid] = item.podgroup
                    if item.owner_object is not None:
                        live_owners[pod_group_uid] = item.owner_object

            for attempt in attempts:
                allowed_job_uids = attempt.job_uids | (
                    {attempt.owner_uid}
                    if attempt.owner_kind == "Job"
                    else set()
                )
                latest = latest_attempts.get(attempt.pod_group_uid)
                owner_object = (
                    live_owners.get(attempt.pod_group_uid)
                    if latest
                    and latest.attempt_id == attempt.attempt_id
                    else None
                )
                combined.append(
                    _AttemptEvidence(
                        attempt=attempt,
                        podgroup=live_podgroups.get(
                            attempt.pod_group_uid
                        ),
                        owner_object=owner_object,
                        pods={
                            uid: pods[uid]
                            for uid in attempt.pod_uids
                            if uid in pods
                        },
                        jobs={
                            uid: jobs[uid]
                            for uid in allowed_job_uids
                            if uid in jobs
                        },
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
        now = datetime.now(tz=timezone.utc)
        evidence_items = []
        for history in histories:
            podgroups = {
                (
                    (
                        metadata.get("namespace")
                        or history.get("namespace")
                        or ""
                    ),
                    metadata["name"],
                )
                for observation in history.get("observations") or []
                if observation.get("kind") == "PodGroup"
                and (metadata := observation.get("metadata") or {}).get(
                    "uid"
                )
                and metadata.get("name")
                and (
                    pod_group_name is None
                    or metadata["name"] == pod_group_name
                )
            }
            placements: dict[str, PlacementRecord] = {}
            for resource in history.get("placements") or []:
                placement = PlacementRecord.from_resource(resource)
                placements[placement.pod_uid] = placement
            for namespace, retained_name in sorted(podgroups):
                evidence_items.append(
                    _WorkloadEvidence(
                        namespace=namespace,
                        pod_group_name=retained_name,
                        observed_until=now,
                        placements=dict(placements),
                        retained_coverage=[
                            history.get("coverage") or {}
                        ],
                        observations=list(
                            history.get("observations") or []
                        ),
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
        A failed list raises.
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
            logger.warning(
                "live PodGroup list failed; using retained history"
            )
            listing = []
        contexts = self._finalize(
            listing,
            retained_evidence,
            time_range=(start, end),
        )
        contexts = [
            context
            for context in contexts
            if context["start_time"] <= end
            and context["end_time"] >= start
        ]
        if not nodelist:
            return contexts
        wanted: set[str] = set()
        for entry in nodelist:
            wanted.update(_split_csv(str(entry)))
        if not wanted:
            return contexts
        return [c for c in contexts if wanted.intersection(c["nodes"])]

    # --- internals ---------------------------------------------------------

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
            meta = podgroup.get("metadata") or {}
            owner = _resolve_owner(meta)
            owner_resource = _owner_resource(owner) if owner is not None else None
            if owner is None or owner_resource is None:
                logger.info(
                    f"podgroup {meta.get('namespace')}/{meta.get('name')}: "
                    "not owned by a supported Job or JobSet; skipping"
                )
                continue
            owned.append((podgroup, owner, owner_resource))
        if not owned:
            return [], []
        pods_by_key = _index_pods_by_pod_group(
            self._fetch_pods([pg for pg, _, _ in owned])
        )
        evidence_items: list[_WorkloadEvidence] = []
        pending_contexts: list[KaiExecutionContext] = []
        for podgroup, owner, owner_resource in owned:
            meta = podgroup.get("metadata") or {}
            namespace = meta.get("namespace") or ""
            owner_options = (
                {"api_version": _JOB_API_VERSION}
                if owner_resource == _JOB_RESOURCE
                else {"group": _JOBSET_GROUP}
            )
            owner_object = None
            if namespace:
                try:
                    owner_object = self._kubernetes.get_one(
                        owner_resource,
                        owner["name"],
                        namespace=namespace,
                        **owner_options,
                    )
                except KubernetesClientError as error:
                    if error.status != 404:
                        raise
            owner_jobs = [owner_object] if owner["kind"] == "Job" and owner_object is not None else []
            if owner["kind"] == "JobSet" and _skip_reason(owner, owner_object) is None:
                owner_jobs = _jobset_child_jobs(
                    owner,
                    self._kubernetes.list(
                        _JOB_RESOURCE,
                        namespace=namespace,
                        label_selector=f"{_JOBSET_NAME_LABEL}={owner['name']}",
                        api_version=_JOB_API_VERSION,
                    ),
                )
            members = _member_pods(
                {
                    uid
                    for job in owner_jobs
                    if (
                        uid := (job.get("metadata") or {}).get("uid")
                    )
                },
                pods_by_key.get(
                    (namespace, meta.get("name") or ""),
                    [],
                ),
            )
            live_observations = (
                observations_from_live_resources(
                    podgroup,
                    members,
                    owner_jobs,
                    owner_object,
                    now=now,
                )
                if _skip_reason(owner, owner_object) is None
                else []
            )
            if live_observations and kai_podgroup_start_timestamp(
                podgroup
            ):
                records = self._fetch_placement_records(
                    {
                        "namespace": namespace,
                        "pod_group_uid": meta.get("uid") or "",
                    }
                )
                evidence_items.append(
                    _WorkloadEvidence(
                        namespace=namespace,
                        pod_group_name=meta.get("name") or "",
                        observed_until=now,
                        podgroup=podgroup,
                        owner_object=owner_object,
                        placements={
                            record.pod_uid: record
                            for record in records
                        },
                        observations=live_observations,
                    )
                )
                continue

            context = _parse_pod_group(
                podgroup,
                owner,
                owner_jobs,
                members,
                owner_object,
                now,
            )
            if context is None:
                continue
            if time_range is not None:
                start, end = time_range
                if (
                    context["start_time"] > end
                    or context["end_time"] < start
                ):
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
        if not podgroups and not retained_evidence:
            return []
        live_evidence, pending_contexts = self._collect_live_evidence(
            podgroups,
            now=datetime.now(tz=timezone.utc),
            time_range=time_range,
        )
        contexts = [
            _context_from_attempt_evidence(evidence)
            for evidence in self._combine_attempt_evidence(
                [*(retained_evidence or []), *live_evidence]
            )
        ]
        admitted_uids = {
            context["pod_group_uid"] for context in contexts
        }
        contexts.extend(
            context
            for context in pending_contexts
            if context["pod_group_uid"] not in admitted_uids
        )
        if time_range is None:
            return contexts
        start, end = time_range
        return [
            context
            for context in contexts
            if context["start_time"] <= end
            and context["end_time"] >= start
        ]

    def _fetch_placement_records(self, context: KaiExecutionContext) -> list[PlacementRecord]:
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
        namespaces = sorted({ns for pg in podgroups if (ns := (pg.get("metadata") or {}).get("namespace"))})
        pods: list[dict] = []
        for ns in namespaces:
            pods.extend(self._kubernetes.list("Pod", namespace=ns, api_version="v1"))
        return pods
