# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import random
import threading
from dataclasses import replace
from datetime import datetime

from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kueue.helpers import (
    WORKLOAD_ANNOTATION,
    OwnerIdentity,
    controller_owner,
    identities_from,
    pod_container_exit_reason,
    pod_node_failure_reason,
)
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import PlacementRecord
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.health import HealthState
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.store import CreateResult, PlacementStore


class ResolutionPending(RuntimeError):
    pass


class UnsupportedOwner(RuntimeError):
    pass


class StaleOwnerReference(RuntimeError):
    """A namespace/name lookup returned an object whose uid doesn't match
    the uid already known from the referencing object's ownerReference.

    Kubernetes names are reusable after deletion. Treated the same as ResolutionPending: 
    retry, since the correct object may become visible."""


class KubernetesIdentityResolver:
    def __init__(
        self,
        batch_api,
        custom_api,
        *,
        workload_version: str = "v1beta1",
        jobset_version: str = "v1alpha2",
    ):
        self.batch_api = batch_api
        self.custom_api = custom_api
        self.workload_version = workload_version
        self.jobset_version = jobset_version

    def resolve(self, pod: dict) -> tuple:
        from kubernetes.client.exceptions import ApiException

        try:
            return self._resolve(pod)
        except ApiException as error:
            if error.status == 404:
                raise ResolutionPending("owner or Workload is not visible yet") from error
            raise

    def _resolve(self, pod: dict) -> tuple:
        metadata = pod.get("metadata") or {}
        namespace = metadata.get("namespace")
        pod_owner = controller_owner(pod, "Job")
        if not namespace or not pod_owner or not pod_owner.get("name"):
            raise UnsupportedOwner("Pod is not controlled by a batch/v1 Job")

        job = self.batch_api.read_namespaced_job(pod_owner["name"], namespace).to_dict()
        job = _normalize_resource(job, api_version="batch/v1", kind="Job")
        job_uid = (job.get("metadata") or {}).get("uid")
        pod_owner_uid = pod_owner.get("uid")
        if not pod_owner_uid or job_uid != pod_owner_uid:
            raise StaleOwnerReference(
                f"Job {namespace}/{pod_owner['name']} uid mismatch "
                f"(Pod controller ref uid={pod_owner_uid!r}, fetched uid={job_uid!r})"
            )
        jobset_ref = controller_owner(job, "JobSet")
        if jobset_ref and jobset_ref.get("name"):
            owner_resource = self.custom_api.get_namespaced_custom_object(
                "jobset.x-k8s.io",
                self.jobset_version,
                namespace,
                "jobsets",
                jobset_ref["name"],
            )
            owner_uid = (owner_resource.get("metadata") or {}).get("uid")
            if jobset_ref.get("uid") and owner_uid != jobset_ref["uid"]:
                raise StaleOwnerReference(
                    f"JobSet {namespace}/{jobset_ref['name']} uid mismatch "
                    f"(Job controller ref uid={jobset_ref['uid']!r}, fetched uid={owner_uid!r})"
                )
        else:
            owner_resource = job

        owner_metadata = owner_resource.get("metadata") or {}
        owner = OwnerIdentity(
            owner_resource.get("apiVersion") or ("jobset.x-k8s.io/" + self.jobset_version),
            owner_resource.get("kind") or ("JobSet" if jobset_ref else "Job"),
            owner_metadata.get("name") or "",
            owner_metadata.get("uid") or "",
        )
        if not owner.name or not owner.uid:
            raise ResolutionPending("owner object is not fully visible yet")

        annotations = metadata.get("annotations") or {}
        workload_name = annotations.get(WORKLOAD_ANNOTATION)
        source = "kueue-workload-annotation"
        if workload_name:
            candidates = [
                self.custom_api.get_namespaced_custom_object(
                    "kueue.x-k8s.io",
                    self.workload_version,
                    namespace,
                    "workloads",
                    workload_name,
                )
            ]
        else:
            source = "owner-uid"
            response = self.custom_api.list_namespaced_custom_object(
                "kueue.x-k8s.io",
                self.workload_version,
                namespace,
                "workloads",
                label_selector=f"kueue.x-k8s.io/job-uid={owner.uid}",
            )
            candidates = response.get("items") or []

        matches = []
        for workload in candidates:
            pair = identities_from(workload, owner_resource)
            if pair:
                matches.append(pair)
        if len(matches) != 1:
            if not matches:
                raise ResolutionPending("matching Kueue Workload is not visible yet")
            raise UnsupportedOwner("multiple Kueue Workloads match the Pod owner")
        workload, resolved_owner = matches[0]
        return workload, resolved_owner, source


class PodPlacementRecorder:
    def __init__(
        self,
        core_api,
        resolver: KubernetesIdentityResolver,
        store: PlacementStore,
        namespace: str,
        *,
        queue_size: int = 1000,
        workers: int = 2,
        watch_timeout_seconds: int = 60,
        max_resolution_attempts: int = 5,
    ):
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        self.core_api = core_api
        self.resolver = resolver
        self.store = store
        self.namespace = namespace
        self.watch_timeout_seconds = watch_timeout_seconds
        self.max_resolution_attempts = max_resolution_attempts
        self.health = HealthState()
        self.items: queue.Queue[dict] = queue.Queue(maxsize=queue_size)
        self.worker_count = workers
        self.stop_event = threading.Event()

    def run(self) -> None:
        from kubernetes.client.exceptions import ApiException
        from kubernetes.watch import Watch

        workers = [
            threading.Thread(target=self._worker, name=f"placement-worker-{index}", daemon=True)
            for index in range(self.worker_count)
        ]
        for worker in workers:
            worker.start()

        resource_version = None
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                if resource_version is None:
                    listing = self.core_api.list_namespaced_pod(self.namespace)
                    resource_version = listing.metadata.resource_version
                    for pod in listing.items:
                        self._enqueue(_as_dict(pod))
                    self.health.update(initial_list_complete=True)

                watcher = Watch()
                for event in watcher.stream(
                    self.core_api.list_namespaced_pod,
                    namespace=self.namespace,
                    resource_version=resource_version,
                    timeout_seconds=self.watch_timeout_seconds,
                    allow_watch_bookmarks=True,
                ):
                    pod = event.get("object")
                    pod_dict = _as_dict(pod)
                    resource_version = (pod_dict.get("metadata") or {}).get("resourceVersion") or resource_version
                    self._enqueue(pod_dict)
                    if self.stop_event.is_set():
                        watcher.stop()
                        break
                backoff = 1.0
            except ApiException as error:
                if error.status == 410:
                    # Retained placement records remain readable through a
                    # relist, so once the initial list has ever completed,
                    # readiness must not flip false just because the watch's
                    # resourceVersion expired.
                    logger.warning("Pod watch resourceVersion expired; relisting")
                    resource_version = None
                    continue
                logger.warning(f"Pod watch failed: {error}")
            except Exception:
                logger.exception("Pod watch failed")
            self.stop_event.wait(backoff + random.uniform(0, backoff / 4))
            backoff = min(backoff * 2, 30)

        self.items.join()

    def stop(self) -> None:
        self.stop_event.set()

    def _enqueue(self, pod: dict) -> None:
        if not (pod.get("spec") or {}).get("nodeName"):
            return
        while not self.stop_event.is_set():
            try:
                self.items.put(pod, timeout=0.5)
                self.health.update(queue_saturated=False)
                return
            except queue.Full:
                self.health.update(queue_saturated=True)
                logger.error("placement work queue is saturated; readiness is failing")

    def _worker(self) -> None:
        while not self.stop_event.is_set() or not self.items.empty():
            try:
                pod = self.items.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._record(pod)
            finally:
                self.items.task_done()

    def _record(self, pod: dict) -> None:
        # Checked on every watch event for this Pod (not just the first), so
        # evidence that only appears later in the Pod's lifecycle is captured
        # before the Pod is garbage-collected. 
        failure_reason = pod_node_failure_reason(pod) or pod_container_exit_reason(pod)
        metadata = pod.get("metadata") or {}
        pod_uid = metadata.get("uid")
        if pod_uid:
            try:
                existing = self.store.get_by_pod_uid(pod_uid)
            except Exception:
                logger.exception("failed to look up existing Pod placement")
                return
            if existing is not None:
                try:
                    result = self.store.create(replace(existing, node_failure_reason=failure_reason))
                except Exception:
                    logger.exception("failed to record Pod placement")
                    return
                detail = f" (node_failure_reason={failure_reason})" if result == CreateResult.UPDATED else ""
                logger.info(f"{result.value}{detail} placement for Pod {self.namespace}/{metadata.get('name')}")
                return

        for attempt in range(self.max_resolution_attempts):
            try:
                workload, owner, source = self.resolver.resolve(pod)
                result = self.store.create(
                    PlacementRecord.for_pod(pod, workload, owner, source, node_failure_reason=failure_reason)
                )
                detail = f" (node_failure_reason={failure_reason})" if result == CreateResult.UPDATED else ""
                logger.info(f"{result.value}{detail} placement for Pod {self.namespace}/{metadata.get('name')}")
                return
            except ResolutionPending as error:
                if attempt + 1 == self.max_resolution_attempts or self.stop_event.is_set():
                    logger.error(f"could not resolve Pod placement: {error}")
                    return
                self.stop_event.wait(min(2**attempt, 8))
            except UnsupportedOwner as error:
                logger.warning(f"skipping unsupported Pod: {error}")
                return
            except StaleOwnerReference as error:
                # The name-reuse collision is permanent for this Pod (its own
                # controller-uid reference points at an object that is gone
                # for good), so retrying will not help; skip immediately.
                logger.warning(f"skipping Pod with stale owner reference: {error}")
                return
            except Exception:
                logger.exception("failed to record Pod placement")
                return


def _as_dict(resource) -> dict:
    if isinstance(resource, dict):
        return resource
    return _normalize_resource(resource.to_dict())


def _normalize_resource(resource: dict, *, api_version: str | None = None, kind: str | None = None) -> dict:
    """Convert Kubernetes client's snake_case dictionaries to API JSON keys.

    Also makes the result recursively JSON-safe: the Kubernetes SDK's
    `.to_dict()` leaves timestamp fields (e.g. `last_transition_time`) as
    Python `datetime` objects, not strings. Renaming a key to
    `lastTransitionTime` doesn't change its value's type, and a raw
    `datetime` reaching `json.dumps` in the store raises `TypeError`. Rather
    than stringifying only the one known timestamp field, `_to_jsonable`
    below converts every `datetime` anywhere in the structure to RFC3339,
    so any other SDK-provided timestamp is safe too.
    """
    metadata = resource.get("metadata") or {}
    normalized_metadata = {
        "name": metadata.get("name"),
        "namespace": metadata.get("namespace"),
        "uid": metadata.get("uid"),
        "labels": metadata.get("labels") or {},
        "annotations": metadata.get("annotations") or {},
        "resourceVersion": metadata.get("resource_version") or metadata.get("resourceVersion"),
        "ownerReferences": [
            {
                "apiVersion": owner.get("api_version") or owner.get("apiVersion"),
                "kind": owner.get("kind"),
                "name": owner.get("name"),
                "uid": owner.get("uid"),
                "controller": owner.get("controller"),
            }
            for owner in metadata.get("owner_references") or metadata.get("ownerReferences") or []
        ],
    }
    spec = resource.get("spec") or {}
    status = resource.get("status") or {}
    container_statuses = status.get("container_statuses") or status.get("containerStatuses") or []
    normalized = {
        **resource,
        "apiVersion": resource.get("api_version") or resource.get("apiVersion") or api_version,
        "kind": resource.get("kind") or kind,
        "metadata": normalized_metadata,
        "spec": {**spec, "nodeName": spec.get("node_name") or spec.get("nodeName")},
        "status": {
            **status,
            "conditions": [
                {
                    **condition,
                    "lastTransitionTime": condition.get("last_transition_time") or condition.get("lastTransitionTime"),
                }
                for condition in status.get("conditions") or []
            ],
            "containerStatuses": [_normalize_container_status(item) for item in container_statuses],
        },
    }
    return _to_jsonable(normalized)


def _to_jsonable(value):
    """Recursively convert `datetime` values to RFC3339 strings.

    Applied to every Kubernetes resource this recorder normalizes, so a
    `datetime` the SDK left in a nested field (not just the known
    `lastTransitionTime` one) can never reach `json.dumps` downstream.
    """
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _normalize_container_status(container_status: dict) -> dict:
    state = container_status.get("state") or {}
    terminated = state.get("terminated")
    if not terminated:
        return {**container_status, "state": state}
    return {
        **container_status,
        "state": {
            **state,
            "terminated": {
                **terminated,
                "exitCode": terminated.get("exit_code")
                if "exit_code" in terminated
                else terminated.get("exitCode"),
            },
        },
    }
