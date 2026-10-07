# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""KAI Scheduler ExecutionContextAdapter.

Emits one workload-level context per PodGroup, with these KAI specifics:

    ExecutionContext   KAI source
    -----------------  -----------------------------------------------------
    id / attempt_id    PodGroup.metadata.uid. Identifies one PodGroup
                       incarnation (a recreated PodGroup gets a new uid);
                       it is not a scheduling attempt.
    job_id             "<namespace>/<name>" of the PodGroup
    start_time         Earliest creation time of the PodGroup or a member Pod
    end_time           Owning Job's completionTime (succeeded) or Failed
                       condition time (failed); now() while it is active.
                       Pods never determine it.
    nodes              Union of .spec.nodeName across the member Pods

Member Pods are joined on the `pod-group-name` annotation plus namespace.
KAI can create a PodGroup after its Pods, so creation times are not used. A
Pod is dropped only when its controller and the PodGroup's owner are the same
kind with different UIDs (an earlier incarnation of the same name).

The owner comes from the `kai.scheduler/top-owner-metadata` annotation, or
else the PodGroup's controller ownerReference. Only PodGroups owned by a
`batch` Job that still exists with the same UID are emitted; the Job is the
only source of an end time, so every other PodGroup is skipped.

A failed Kubernetes read raises; only NotFound is treated as absence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NotRequired, TypedDict

import yaml
from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kubectl_client import KubectlClient
from attribution_blueprint.types import assert_aware

_PODGROUP_RESOURCE = "podgroups.scheduling.run.ai"
_POD_GROUP_ANNOTATION = "pod-group-name"
_TOP_OWNER_ANNOTATION = "kai.scheduler/top-owner-metadata"

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
    # BP-internal connector identity (always "kai").
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


def _member_pods(owner: dict, pods: list[dict]) -> list[dict]:
    """Drop Pods that belong to an earlier incarnation of this PodGroup.

    A recreated PodGroup reuses its name, so name-and-namespace alone would
    also match Pods left over from the earlier incarnation.
    """
    members = []
    for pod in pods:
        controller = _controller_owner(pod.get("metadata") or {})
        if controller is not None and controller["kind"] == owner["kind"] and controller["uid"] != owner["uid"]:
            continue
        members.append(pod)
    return members


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


def _is_batch_job(owner: dict) -> bool:
    if owner["kind"] != "Job":
        return False
    group = owner.get("group")
    api_version = owner.get("apiVersion")
    if group is None and isinstance(api_version, str) and "/" in api_version:
        group = api_version.split("/", 1)[0]
    return group in (None, "batch")


def _skip_reason(owner: dict, job: dict | None) -> str | None:
    """Why the owning Job cannot supply a window, or None when it can."""
    if job is None:
        return "owner Job not found"
    if (job.get("metadata") or {}).get("uid") != owner["uid"]:
        return "owner Job uid mismatch"
    return None


def _parse_pod_group(
    podgroup: dict, owner: dict, pods: list[dict], owner_job: dict | None, now: datetime
) -> KaiExecutionContext | None:
    """Normalize one PodGroup and its member Pods into wire format.

    `owner` is the PodGroup's resolved Job owner and `owner_job` the live Job
    object when one was found.

    Returns None when the PodGroup lacks a uid, name, namespace, or a
    parseable creationTimestamp (without a start the orchestrator cannot build
    a valid attributor window), or when the owner Job cannot supply an end.
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

    members = _member_pods(owner, pods)
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
        if (pod_name := (pod.get("metadata") or {}).get("name"))
        and (node := (pod.get("spec") or {}).get("nodeName"))
    }
    pod_names = sorted(
        pod_name for pod in members if (pod_name := (pod.get("metadata") or {}).get("name"))
    )

    spec = podgroup.get("spec") or {}
    shape = _podgroup_shape(spec)
    if shape != _SHAPE_OWNED:
        logger.info(f"podgroup {namespace}/{name}: {shape} PodGroup; emitting raw context only")

    skip_reason = _skip_reason(owner, owner_job)
    if skip_reason is not None:
        logger.info(f"podgroup {namespace}/{name}: {skip_reason}; skipping")
        return None
    owner_end = _job_end_time(owner_job)
    if owner_end is None:
        logger.debug(f"podgroup {namespace}/{name}: owner Job still active; using now() for end_time")
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
    """ExecutionContextAdapter that discovers KAI PodGroups via kubectl.

    `kubectl_path`, `kubeconfig`, and `context` configure the shared kubectl
    runner. `namespace` selects a single namespace; the empty string queries
    every namespace via `-A`.
    """

    def __init__(
        self,
        kubectl_path: str = "kubectl",
        kubeconfig: str | None = None,
        context: str | None = None,
        namespace: str | None = None,
        timeout_seconds: int = 60,
    ):
        self.namespace = namespace or ""
        self.timeout_seconds = timeout_seconds
        self._kubectl = KubectlClient(
            kubectl_path=kubectl_path,
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
            obj = self._kubectl.get_one(_PODGROUP_RESOURCE, name, namespace=ns)
            if obj is not None:
                podgroups.append(obj)
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
        listing = self._kubectl.list(_PODGROUP_RESOURCE, namespace=self.namespace or None)
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
        # Only Job-owned PodGroups can have an end time, so everything else is
        # dropped before any Pod or Job is fetched.
        owned: list[tuple[dict, dict]] = []
        for podgroup in podgroups:
            meta = podgroup.get("metadata") or {}
            owner = _resolve_owner(meta)
            if owner is None or not _is_batch_job(owner):
                logger.info(
                    f"podgroup {meta.get('namespace')}/{meta.get('name')}: not owned by a batch Job; skipping"
                )
                continue
            owned.append((podgroup, owner))
        if not owned:
            return []
        pods_by_key = _index_pods_by_pod_group(self._fetch_pods([pg for pg, _ in owned]))
        now = datetime.now(tz=timezone.utc)
        contexts: list[KaiExecutionContext] = []
        for podgroup, owner in owned:
            meta = podgroup.get("metadata") or {}
            namespace = meta.get("namespace") or ""
            owner_job = (
                self._kubectl.get_one("job.batch", owner["name"], namespace=namespace) if namespace else None
            )
            ctx = _parse_pod_group(
                podgroup,
                owner,
                pods_by_key.get((namespace, meta.get("name") or ""), []),
                owner_job,
                now,
            )
            if ctx is not None:
                contexts.append(ctx)
        return contexts

    def _fetch_pods(self, podgroups: list[dict]) -> list[dict]:
        """List Pods once per distinct namespace.

        Pods carry the PodGroup only as an annotation, which kubectl cannot
        select on, so the join is done client-side after a namespace-scoped
        list.
        """
        namespaces = sorted(
            {ns for pg in podgroups if (ns := (pg.get("metadata") or {}).get("namespace"))}
        )
        pods: list[dict] = []
        for ns in namespaces:
            pods.extend(self._kubectl.list("pods", namespace=ns))
        return pods
