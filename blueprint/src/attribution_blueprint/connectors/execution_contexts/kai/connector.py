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
    end_time           Terminal time of the owning Job or JobSet; now()
                       while the owner is active. Pods never determine it.
    nodes              Union of .spec.nodeName across the member Pods

Member Pods are joined on the `pod-group-name` annotation plus namespace,
then validated against the exact owning Job UID; JobSet members must belong
to current child Jobs. Kubernetes exposes no reliable per-annotation
assignment timestamp (`managedFields` records a field manager's last write to
*any* owned field, and a Pod's own `creationTimestamp` can predate its
PodGroup by design), so membership intentionally does not use time as a
discriminator; see `_member_pods` for the resulting known limitation.

The owner comes from the `kai.scheduler/top-owner-metadata` annotation, or
else the PodGroup's controller ownerReference. Only PodGroups owned by a
`batch` Job or `jobset.x-k8s.io` JobSet that still exists with the same UID
are emitted; the owner is the only source of an end time.

A failed Kubernetes read raises; only NotFound is treated as absence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NotRequired, TypedDict

import yaml
from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kubernetes_client import (
    KubernetesClient,
    KubernetesClientError,
)
from attribution_blueprint.types import assert_aware

_PODGROUP_RESOURCE = "PodGroup"
_PODGROUP_GROUP = "scheduling.run.ai"
_JOB_RESOURCE = "Job"
_JOB_API_VERSION = "batch/v1"
_JOBSET_RESOURCE = "JobSet"
_JOBSET_GROUP = "jobset.x-k8s.io"
_POD_GROUP_ANNOTATION = "pod-group-name"
_TOP_OWNER_ANNOTATION = "kai.scheduler/top-owner-metadata"
_JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"

_SHAPE_OWNED = "owned"
_SHAPE_HIERARCHICAL = "hierarchical"


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

    job_id: str
    pod_group_uid: str
    attempt_id: str
    namespace: str
    name: str
    podgroup_shape: str
    owner_kind: str
    owner_name: str
    owner_uid: str

    queue_name: NotRequired[str]
    min_member: NotRequired[int]
    # Raw PodGroup status, preserved unmodified.
    scheduling_conditions: NotRequired[list]
    resources_status: NotRequired[object]
    pod_names: NotRequired[list[str]]
    # Pod name -> `.spec.nodeName`.
    pod_nodes: NotRequired[dict[str, str]]


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


def _jobset_child_job_uids(owner: dict, jobs: list[dict]) -> set[str]:
    """Return UIDs of Jobs controlled by this exact JobSet incarnation."""
    child_uids: set[str] = set()
    for job in jobs:
        controller = _controller_owner(job.get("metadata") or {})
        uid = (job.get("metadata") or {}).get("uid")
        if controller and controller["kind"] == "JobSet" and controller["uid"] == owner["uid"] and uid:
            child_uids.add(uid)
    return child_uids


def _top_owner(meta: dict) -> dict | None:
    raw = (meta.get("annotations") or {}).get(_TOP_OWNER_ANNOTATION)
    if not raw:
        return None
    try:
        owner = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if not isinstance(owner, dict):
        return None
    if not all(owner.get(field) for field in ("kind", "name", "uid")):
        return None
    return owner


def _controller_owner(meta: dict) -> dict | None:
    refs = [ref for ref in meta.get("ownerReferences") or [] if isinstance(ref, dict)]
    for ref in refs:
        if ref.get("controller") and ref.get("kind") and ref.get("name") and ref.get("uid"):
            return ref
    return None


def _resolve_owner(meta: dict) -> dict | None:
    return _top_owner(meta) or _controller_owner(meta)


def _podgroup_shape(spec: dict) -> str:
    return _SHAPE_HIERARCHICAL if spec.get("subGroups") else _SHAPE_OWNED


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


def _jobset_end_time(jobset: dict) -> datetime | None:
    """Terminal condition time of a finished JobSet, or None while active."""
    status = jobset.get("status") or {}
    terminal_state = status.get("terminalState")
    condition_types = {terminal_state} if terminal_state in {"Completed", "Failed"} else {"Completed", "Failed"}
    times = [
        transition
        for condition in status.get("conditions") or []
        if (condition or {}).get("type") in condition_types
        and (condition or {}).get("status") == "True"
        and (transition := _parse_dt(condition.get("lastTransitionTime"))) is not None
    ]
    return max(times) if times else None


def _owner_end_time(owner: dict, owner_object: dict) -> datetime | None:
    if owner["kind"] == "Job":
        return _job_end_time(owner_object)
    return _jobset_end_time(owner_object)


def _skip_reason(owner: dict, owner_object: dict | None) -> str | None:
    """Why the owner cannot supply a window, or None when it can."""
    if owner_object is None:
        return f"owner {owner['kind']} not found"
    if (owner_object.get("metadata") or {}).get("uid") != owner["uid"]:
        return f"owner {owner['kind']} uid mismatch"
    return None


def _parse_pod_group(
    podgroup: dict,
    owner: dict,
    owner_job_uids: set[str],
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

    members = _member_pods(owner_job_uids, pods)
    member_starts = [
        created
        for pod in members
        if (created := _parse_dt((pod.get("metadata") or {}).get("creationTimestamp"))) is not None
    ]
    if member_starts:
        start_time = min(start_time, *member_starts)
    pod_nodes = {
        pod_name: node
        for pod in members
        if (pod_name := (pod.get("metadata") or {}).get("name")) and (node := (pod.get("spec") or {}).get("nodeName"))
    }
    pod_names = sorted(pod_name for pod in members if (pod_name := (pod.get("metadata") or {}).get("name")))

    spec = podgroup.get("spec") or {}
    shape = _podgroup_shape(spec)

    skip_reason = _skip_reason(owner, owner_object)
    if skip_reason is not None:
        logger.info(f"podgroup {namespace}/{name}: {skip_reason}; skipping")
        return None
    if shape != _SHAPE_OWNED:
        logger.info(f"podgroup {namespace}/{name}: {shape} PodGroup; recording its shape only")
    owner_end = _owner_end_time(owner, owner_object)
    if owner_end is None:
        logger.debug(f"podgroup {namespace}/{name}: owner {owner['kind']} still active; using now() for end_time")
    end_time = max(owner_end if owner_end is not None else now, start_time)
    if not pod_nodes:
        logger.warning(f"podgroup {namespace}/{name}: node list unavailable from member Pods")

    ctx: KaiExecutionContext = {
        "id": uid,
        "workload_type": "kai",
        "scheduler": "kai",
        "nodes": sorted(set(pod_nodes.values())),
        "start_time": start_time,
        "end_time": end_time,
        "job_id": f"{namespace}/{name}",
        "pod_group_uid": uid,
        # PodGroup incarnation identity, not a scheduling attempt.
        "attempt_id": uid,
        "namespace": namespace,
        "name": name,
        "podgroup_shape": shape,
    }

    queue_name = spec.get("queue")
    if queue_name:
        ctx["queue_name"] = queue_name
    min_member = _non_negative_int(spec.get("minMember"))
    if min_member is not None:
        ctx["min_member"] = min_member

    ctx["owner_kind"] = owner["kind"]
    ctx["owner_name"] = owner["name"]
    ctx["owner_uid"] = owner["uid"]

    status = podgroup.get("status") or {}
    conditions = status.get("schedulingConditions")
    if isinstance(conditions, list):
        ctx["scheduling_conditions"] = conditions
    if "resourcesStatus" in status:
        ctx["resources_status"] = status["resourcesStatus"]

    if pod_names:
        ctx["pod_names"] = pod_names
    if pod_nodes:
        ctx["pod_nodes"] = pod_nodes

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
    ):
        self.namespace = namespace or ""
        self.timeout_seconds = timeout_seconds
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
        contexts = [c for c in self._finalize(listing) if c["start_time"] <= end and c["end_time"] >= start]
        if not nodelist:
            return contexts
        wanted: set[str] = set()
        for entry in nodelist:
            wanted.update(_split_csv(str(entry)))
        if not wanted:
            return contexts
        return [c for c in contexts if wanted.intersection(c["nodes"])]

    # --- internals ---------------------------------------------------------

    def _finalize(self, podgroups: list[dict]) -> list[KaiExecutionContext]:
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
            return []
        pods_by_key = _index_pods_by_pod_group(self._fetch_pods([pg for pg, _, _ in owned]))
        now = datetime.now(tz=timezone.utc)
        contexts: list[KaiExecutionContext] = []
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
            owner_job_uids = {owner["uid"]}
            if owner["kind"] == "JobSet" and _skip_reason(owner, owner_object) is None:
                child_jobs = self._kubernetes.list(
                    _JOB_RESOURCE,
                    namespace=namespace,
                    label_selector=f"{_JOBSET_NAME_LABEL}={owner['name']}",
                    api_version=_JOB_API_VERSION,
                )
                owner_job_uids = _jobset_child_job_uids(owner, child_jobs)
            ctx = _parse_pod_group(
                podgroup,
                owner,
                owner_job_uids,
                pods_by_key.get((namespace, meta.get("name") or ""), []),
                owner_object,
                now,
            )
            if ctx is not None:
                contexts.append(ctx)
        return contexts

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
