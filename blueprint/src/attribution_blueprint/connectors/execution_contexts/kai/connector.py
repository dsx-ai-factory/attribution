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

This adapter finds PodGroups and emits one workload-level
context for each.

Field mapping:

    ExecutionContext   KAI source
    -----------------  -----------------------------------------------------
    id / attempt_id    PodGroup.metadata.uid. This identifies one PodGroup
                       *incarnation* (a PodGroup deleted and recreated with
                       the same name gets a new uid). This field is NOT a
                       scheduling attempt.
    job_id             "<namespace>/<name>" of the PodGroup
    start_time         PodGroup.metadata.creationTimestamp
    end_time           Latest container finish time once every member Pod has
                       reached a terminal phase; otherwise now().
    nodes              Union of .spec.nodeName across the member Pods

Member Pods are joined on the `pod-group-name` annotation plus namespace.
Pods carry that annotation but no scheduler labels, and a PodGroup name is
only unique within a namespace, so the namespace is always part of the key. A
Pod created before the PodGroup's creationTimestamp belongs to an earlier
incarnation that had the same name and is not attributed to this one.

The workload's owner comes from the `kai.scheduler/top-owner-metadata`
annotation, falling back to the PodGroup's controller ownerReference.
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
_SHAPE_EXTERNAL = "external"
_SHAPE_HIERARCHICAL = "hierarchical"

_TERMINAL_POD_PHASES = frozenset({"Succeeded", "Failed"})


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

    queue_name: NotRequired[str]
    min_member: NotRequired[int]
    owner_kind: NotRequired[str]
    owner_name: NotRequired[str]
    owner_uid: NotRequired[str]
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


def _pod_group_key(pod: dict) -> tuple[str, str] | None:
    """Return the `(namespace, pod_group_name)` key for a Pod, or None when it
    has no `pod-group-name` annotation or no namespace."""
    meta = pod.get("metadata") or {}
    namespace = meta.get("namespace")
    pod_group_name = (meta.get("annotations") or {}).get(_POD_GROUP_ANNOTATION)
    if not namespace or not pod_group_name:
        return None
    return (namespace, pod_group_name)


def _index_pods_by_pod_group(pods: list[dict]) -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = {}
    for pod in pods:
        key = _pod_group_key(pod)
        if key is not None:
            index.setdefault(key, []).append(pod)
    return index


def _member_pods(podgroup_created: datetime | None, pods: list[dict]) -> list[dict]:
    """Drop Pods that predate this PodGroup incarnation.

    A recreated PodGroup reuses its name, so name-and-namespace alone would
    also match Pods left over from the earlier incarnation.
    """
    if podgroup_created is None:
        return pods
    members = []
    for pod in pods:
        pod_created = _parse_dt((pod.get("metadata") or {}).get("creationTimestamp"))
        if pod_created is not None and pod_created < podgroup_created:
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


def _podgroup_shape(spec: dict, owner: dict | None) -> str:
    if spec.get("subGroups"):
        return _SHAPE_HIERARCHICAL
    if owner is None:
        return _SHAPE_EXTERNAL
    return _SHAPE_OWNED


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _finished_at(pod: dict) -> datetime | None:
    finished = []
    for status in (pod.get("status") or {}).get("containerStatuses") or []:
        terminated = ((status or {}).get("state") or {}).get("terminated") or {}
        parsed = _parse_dt(terminated.get("finishedAt"))
        if parsed is not None:
            finished.append(parsed)
    return max(finished) if finished else None


def _end_time(pods: list[dict], now: datetime) -> datetime:
    """Latest container finish time once every member Pod is terminal.

    With no Pods, or any Pod still pending or running, the workload may still
    be active, so the evidence window ends now.
    """
    if not pods:
        return now
    if any(((pod.get("status") or {}).get("phase")) not in _TERMINAL_POD_PHASES for pod in pods):
        return now
    finishes = [ts for ts in (_finished_at(pod) for pod in pods) if ts is not None]
    return max(finishes) if finishes else now


def _parse_pod_group(podgroup: dict, pods: list[dict], now: datetime) -> KaiExecutionContext | None:
    """Normalize one PodGroup and its member Pods into wire format.

    Returns None when the PodGroup lacks a uid, name, namespace, or a
    parseable creationTimestamp; without a start the orchestrator cannot build
    a valid attributor window.
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

    members = _member_pods(start_time, pods)
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
    owner = _resolve_owner(meta)
    shape = _podgroup_shape(spec, owner)
    if shape != _SHAPE_OWNED:
        logger.info(f"podgroup {namespace}/{name}: {shape} PodGroup; emitting raw context only")

    end_time = max(_end_time(members, now), start_time)

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

    if owner is not None:
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
        self.kubectl_path = kubectl_path
        self.kubeconfig = kubeconfig
        self.context = context
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
        adapter is namespace-scoped). Missing PodGroups are skipped."""
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

        A PodGroup's end is only known from its Pods, so the listing is first
        narrowed to PodGroups created no later than `end`, and the window is
        checked after the member Pods are fetched. `nodelist` entries may be
        CSVs of plain node names; a PodGroup survives if any member Pod ran on
        one of them.
        """
        assert_aware(start, "discover_by_time_range.start")
        assert_aware(end, "discover_by_time_range.end")
        listing = self._kubectl.list(_PODGROUP_RESOURCE, namespace=self.namespace or None)
        candidates = []
        for podgroup in listing:
            created = _parse_dt((podgroup.get("metadata") or {}).get("creationTimestamp"))
            if created is not None and created <= end:
                candidates.append(podgroup)
        contexts = [c for c in self._finalize(candidates) if c["start_time"] <= end and c["end_time"] >= start]
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
        if not podgroups:
            return []
        pods_by_key = _index_pods_by_pod_group(self._fetch_pods(podgroups))
        now = datetime.now(tz=timezone.utc)
        contexts: list[KaiExecutionContext] = []
        for podgroup in podgroups:
            meta = podgroup.get("metadata") or {}
            key = (meta.get("namespace") or "", meta.get("name") or "")
            ctx = _parse_pod_group(podgroup, pods_by_key.get(key, []), now)
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
