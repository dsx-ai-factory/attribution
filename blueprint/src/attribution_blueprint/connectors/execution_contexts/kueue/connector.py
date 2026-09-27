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
    state              Derived from Workload status conditions + Finished
                       reason (e.g. ADMITTED, FINISHED_SUCCEEDED, EVICTED)
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
                       if present, in seconds
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
Workload back to its Job); it is NOT stamped on the pods. The pods are
created by the Kubernetes Job controller, which stamps the canonical
`batch.kubernetes.io/controller-uid=<job.uid>` and
`batch.kubernetes.io/job-name=<job.name>` labels. This adapter pulls the
owner Job's UID out of the Workload's `ownerReferences` and selects pods
with `batch.kubernetes.io/controller-uid=<owner_uid>`. (Both an earlier
bug — looking up by Workload UID — and a follow-on bug — looking up by
the Kueue label rather than the batch label — produced empty pod sets
on real clusters; the fixtures masked both by conflating UIDs.)
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import NamedTuple, NotRequired, TypedDict

from loguru import logger

from attribution_blueprint.types import assert_aware


class KueueExecutionContext(TypedDict):
    """Adapter-specific ExecutionContext shape.

    Same spine as `attribution_blueprint.types.ExecutionContext`; the
    `workload_uid`, `namespace`, `queue_name`, and `priority_class` keys are
    Kueue-specific extensions stored on the dict for the per-context
    raw_scheduler_json column (see output_schema.md).
    """

    id: str
    workload_type: str
    # BP-internal connector identity (always "kueue"); consumed by
    # SQLiteOutput as the `scheduler_type` column. Distinct from
    # `workload_type` because the service enum doesn't yet know "kueue",
    # so the wire value is "" while SQLite still labels the row honestly.
    scheduler: str
    nodes: list[str]
    start_time: datetime
    end_time: datetime

    job_id: str
    workload_uid: str
    attempt_id: str
    namespace: str
    state: str

    name: NotRequired[str]
    user_name: NotRequired[str]
    failed_node: NotRequired[str]
    time_limit: NotRequired[int]
    queue_name: NotRequired[str]
    priority_class: NotRequired[int]
    gpu_count: NotRequired[int]
    # Pod names belonging to this Workload at the time of discovery. Populated
    # in `_finalize` from the same pod listing used for node/failed_node
    # evidence. Downstream log/metric connectors consume this list so they
    # don't need to re-derive the per-owner pod selector. Empty list when the
    # owner type isn't supported or all pods have been garbage-collected.
    pod_names: NotRequired[list[str]]
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

# The standard batch/v1 Job pod label. The Kubernetes Job controller stamps
# `batch.kubernetes.io/controller-uid=<job.metadata.uid>` on every pod it
# creates from a Job; we pull that UID out of the Workload's ownerReferences.
# We deliberately do NOT use Kueue's `kueue.x-k8s.io/job-uid` label here —
# that label is set on the Workload (not on its pods), so a label-selector
# query against pods returns nothing.
_POD_OWNER_LABEL = "batch.kubernetes.io/controller-uid"

# JobSet (jobset.sigs.k8s.io) stamps this label on every pod it transitively
# owns, including pods of all child Jobs of all replicatedJobs. When a Kueue
# Workload's `ownerReferences[].kind` is `JobSet`, the Workload has no
# `controller-uid` to match against (each child Job has its own distinct
# uid), so we instead label-select pods by the JobSet's name. The label key
# is stable across JobSet versions since v0.5.
_POD_JOBSET_LABEL = "jobset.sigs.k8s.io/jobset-name"


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
    the first admission interval still matches. (max() would silently shift
    the window forward to the latest re-admission.)
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


def _pod_selector_for(workload: dict) -> tuple[str, str] | None:
    """Return the `(label-key, label-value)` pair identifying pods that
    belong to this Workload, or None if no selector can be derived.

    Two owner shapes are supported:

    - **batch/v1 Job** (`ownerReferences[].kind == "Job"`): the Kubernetes
      Job controller stamps `batch.kubernetes.io/controller-uid=<job.uid>`
      on every pod it creates from the Job. We pull that uid out of the
      Workload's ownerReferences.
    - **JobSet** (`ownerReferences[].kind == "JobSet"`): JobSet creates
      one batch/v1 Job per replicatedJob, each with its own distinct uid,
      so there is no single controller-uid to match. JobSet instead
      stamps `jobset.sigs.k8s.io/jobset-name=<jobset.name>` on every pod
      it transitively owns. We select by that label.

    Contract: Kueue guarantees at most one Job/JobSet ownerReference per
    Workload. We take the first match defensively; if multiple ever
    appear, the behavior is silently first-wins. `_extract_time_limit`
    makes the same assumption for its batch/v1 Job lookup.

    Returns None for Workloads with no recognised owner — pending
    Workloads whose Job hasn't been created yet, or workloads owned by
    custom resources like RayJob/MPIJob that this adapter doesn't yet
    know how to attribute.
    """
    for owner in (workload.get("metadata") or {}).get("ownerReferences") or []:
        kind = owner.get("kind")
        if kind == "Job" and owner.get("uid"):
            return (_POD_OWNER_LABEL, owner["uid"])
        if kind == "JobSet" and owner.get("name"):
            return (_POD_JOBSET_LABEL, owner["name"])
    return None


class _PodEvidence(NamedTuple):
    nodes: list[str]
    failed_node: str | None
    pod_names: list[str]
    pod_nodes: dict[str, str]


def _collect_pod_evidence(
    pods: list[dict],
    selector: tuple[str, str] | None,
) -> _PodEvidence:
    """Return sorted unique nodes, the lex-smallest failed_node, sorted pod
    names and the pod -> node mapping for pods whose
    `metadata.labels[selector.key] == selector.value`.

    The caller is responsible for narrowing the input list to pods that
    might belong to this workload — typically via the kubectl label
    selector in `_fetch_pods_for`. This function still re-checks each
    pod's label as a defensive filter: a pod missing the label, or
    carrying a different value, must NEVER be associated with this
    workload's evidence even if it leaked into the input list.

    Pod names are returned alongside the node evidence so downstream
    connectors (e.g. `kubectl_logs`) don't have to re-derive the
    per-owner pod selector and re-fetch the pod list.

    Returns empty evidence when `selector` is None — Workloads without
    a recognised owner have no pods to attribute.
    """
    if not selector:
        return _PodEvidence([], None, [], {})
    key, value = selector
    nodes: set[str] = set()
    failed_nodes: set[str] = set()
    pod_names: set[str] = set()
    pod_nodes: dict[str, str] = {}
    for pod in pods:
        meta = pod.get("metadata") or {}
        labels = meta.get("labels") or {}
        if labels.get(key) != value:
            continue
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
        for cs in status.get("containerStatuses") or []:
            terminated = (cs.get("state") or {}).get("terminated") or {}
            exit_code = terminated.get("exitCode")
            if isinstance(exit_code, int) and exit_code != 0 and node_name:
                failed_nodes.add(node_name)
                break
    # `failed_node` is the lexicographically smallest among nodes that
    # reported a non-zero exit, so the result is stable across kubectl
    # response orderings. Previously this picked the first iteration-order
    # match, which flakes when the API server returns pods in a different
    # order between runs.
    failed_node = min(failed_nodes) if failed_nodes else None
    return _PodEvidence(sorted(nodes), failed_node, sorted(pod_names), pod_nodes)


def _extract_time_limit(workload: dict, owner_jobs: dict[str, dict]) -> int | None:
    """Pull the underlying Job's activeDeadlineSeconds, if Kueue is driving a Job.

    `owner_jobs` maps `"<namespace>/<name>"` of any batch/v1 Job objects the
    caller fetched alongside the workloads. Returns None when no matching
    Job is in the map, or when the Job has no deadline.
    """
    owner_refs = (workload.get("metadata") or {}).get("ownerReferences") or []
    namespace = (workload.get("metadata") or {}).get("namespace") or ""
    for owner in owner_refs:
        if owner.get("kind") != "Job":
            continue
        key = f"{namespace}/{owner.get('name')}"
        job = owner_jobs.get(key)
        if not job:
            continue
        deadline = (job.get("spec") or {}).get("activeDeadlineSeconds")
        if isinstance(deadline, int):
            return deadline
    return None


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
    state = _derive_state(conditions)

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
        # Wire-format workload.type value. The service's WorkloadTypeEnum
        # currently knows only {'', 'slurm', 'runai'} — sending 'kueue' 422s.
        # UNDEFINED ('') is the documented "we don't know / unsupported source"
        # path and lets the service's last-resort rules fire. Switch to
        # 'kueue' once the service enum learns about it.
        "workload_type": "",
        # BP-internal connector identity. SQLite's `scheduler_type` column
        # reads this in preference to `workload_type`, so kueue rows are
        # distinguishable from other UNDEFINED sources without involving
        # the attribution service.
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

    time_limit = _extract_time_limit(workload, owner_jobs)
    if time_limit is not None:
        ctx["time_limit"] = time_limit

    # The owning Job's user is the most reliable username source — Workloads
    # themselves don't carry one. We accept `username` from any common label
    # spelling Kubernetes batch tooling stamps; absent the label, the field
    # is left off (NotRequired).
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
    ):
        self.kubectl_path = kubectl_path
        self.kubeconfig = kubeconfig
        self.context = context
        self.namespace = namespace or ""
        self.timeout_seconds = timeout_seconds

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
        non-zero for unknown names and the per-id loop tolerates that, the
        same way the Slurm adapter tolerates sacct's non-zero exit for
        unknown ExecutionContext ids.
        """
        workloads: list[dict] = []
        for ref in ids:
            ns, _, name = ref.partition("/")
            if not name:
                # Bare name: only meaningful when the adapter is pinned to a
                # namespace via config. Otherwise reject — `<name>` without a
                # namespace is ambiguous against multi-tenant clusters.
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
        # Cheap first pass: parse only the timestamps each workload object
        # already carries (conditions + creationTimestamp). Skip the
        # per-workload `kubectl get pods` / `kubectl get job` round-trips
        # that `_finalize` would otherwise issue for every entry in the
        # listing. On a long-history cluster this avoids O(N) wasted calls
        # for workloads we'd discard at the window filter anyway.
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

    # --- private kubectl plumbing -----------------------------------------

    def _kubectl_base(self) -> list[str]:
        cmd = [self.kubectl_path]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        if self.context:
            cmd += ["--context", self.context]
        return cmd

    def _run_json(self, args: list[str]) -> dict | None:
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
            logger.warning(f"kubectl exited {e.returncode}: {(e.stderr or '').strip() or '<no stderr>'}")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"kubectl timed out after {self.timeout_seconds}s: {cmd!r}")
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.warning(f"kubectl returned invalid JSON: {e}")
            return None

    def _kubectl_get_one(self, kind: str, name: str, *, namespace: str) -> dict | None:
        args = ["get", kind, name]
        if namespace:
            args += ["-n", namespace]
        return self._run_json(args)

    def _kubectl_list(self, kind: str, *, namespace: str | None) -> list[dict]:
        args = ["get", kind]
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

        Workloads with no recognised owner (pending Workloads whose Job
        hasn't been created yet, or custom-resource owners we don't
        attribute) are skipped — there are no pods to fetch. Kueue's own
        `kueue.x-k8s.io/job-uid` label is NOT used: it lives on the
        Workload, not on the pods.
        """
        pods: list[dict] = []
        for wl in workloads:
            selector = _pod_selector_for(wl)
            if selector is None:
                continue
            key, value = selector
            ns = (wl.get("metadata") or {}).get("namespace") or ""
            args = ["get", "pods", "-l", f"{key}={value}"]
            if ns:
                args += ["-n", ns]
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
        """Fetch the batch/v1 Job objects owning each workload, keyed `<ns>/<name>`.

        Workloads whose owner isn't a batch/v1 Job (e.g. RayJob, MPIJob,
        JobSet) are skipped — the connector only consults the owner for the
        `time_limit` field, which lives on batch/v1 Job's
        `.spec.activeDeadlineSeconds`. Custom-resource owners can populate
        that field via a future code path without changing the wire format.
        """
        out: dict[str, dict] = {}
        seen: set[tuple[str, str]] = set()
        for wl in workloads:
            meta = wl.get("metadata") or {}
            ns = meta.get("namespace") or ""
            for owner in meta.get("ownerReferences") or []:
                if owner.get("kind") != "Job":
                    continue
                name = owner.get("name")
                if not name:
                    continue
                key = (ns, name)
                if key in seen:
                    continue
                seen.add(key)
                job = self._kubectl_get_one("job", name, namespace=ns)
                if job is not None:
                    out[f"{ns}/{name}"] = job
        return out

    def _finalize(self, workloads: list[dict]) -> list[KueueExecutionContext]:
        if not workloads:
            return []
        pods = self._fetch_pods_for(workloads)
        owner_jobs = self._fetch_owner_jobs(workloads)
        now = datetime.now(tz=timezone.utc)
        results: list[KueueExecutionContext] = []
        for wl in workloads:
            ctx = _parse_workload(wl, pods, owner_jobs, now=now)
            if ctx is not None:
                results.append(ctx)
        return results
