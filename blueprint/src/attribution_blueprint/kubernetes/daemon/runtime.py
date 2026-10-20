# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import random
import threading

from loguru import logger

from attribution_blueprint.kubernetes.daemon.health import HealthState
from attribution_blueprint.kubernetes.daemon.models import PlacementRecord
from attribution_blueprint.kubernetes.daemon.resolvers import (
    AssociatedResourceResolver,
    OwnershipResolver,
    ResolutionPending,
    StaleOwnerReference,
    UnsupportedOwner,
    normalize_resource,
)
from attribution_blueprint.kubernetes.daemon.store import CreateResult, PlacementStore
from attribution_blueprint.kubernetes.helpers import pod_node_failure_reason


class KubernetesDaemon:
    """Record Pod placement before optional ownership/scheduler enrichment."""

    def __init__(
        self,
        core_api,
        store: PlacementStore,
        namespace: str,
        *,
        owner_resolver: OwnershipResolver | None = None,
        resource_resolvers: tuple[AssociatedResourceResolver, ...] = (),
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
        self.store = store
        self.namespace = namespace
        self.owner_resolver = owner_resolver
        self.resource_resolvers = resource_resolvers
        self.watch_timeout_seconds = watch_timeout_seconds
        self.max_resolution_attempts = max_resolution_attempts
        self.health = HealthState()
        self.items: queue.Queue[tuple[dict, bool]] = queue.Queue(maxsize=queue_size)
        self.worker_count = workers
        self.stop_event = threading.Event()

    def run(self) -> None:
        from kubernetes.client.exceptions import ApiException
        from kubernetes.watch import Watch

        workers = [
            threading.Thread(target=self._worker, name=f"kubernetes-daemon-worker-{index}", daemon=True)
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
                    resource_version = self._enqueue_listing(listing)
                    self.health.update(initial_list_complete=True)

                watcher = Watch()
                for event in watcher.stream(
                    self.core_api.list_namespaced_pod,
                    namespace=self.namespace,
                    resource_version=resource_version,
                    timeout_seconds=self.watch_timeout_seconds,
                    allow_watch_bookmarks=True,
                ):
                    pod_dict = _as_dict(event.get("object"))
                    resource_version = (pod_dict.get("metadata") or {}).get("resourceVersion") or resource_version
                    self._enqueue(pod_dict, deleted=event.get("type") == "DELETED")
                    if self.stop_event.is_set():
                        watcher.stop()
                        break
                backoff = 1.0
            except ApiException as error:
                if error.status == 410:
                    logger.warning("Pod watch resourceVersion expired; relisting")
                    resource_version = None
                    self.health.update(initial_list_complete=False)
                    continue
                logger.warning(f"Pod watch failed: {error}")
            except Exception:
                logger.exception("Pod watch failed")
            self.stop_event.wait(backoff + random.uniform(0, backoff / 4))
            backoff = min(backoff * 2, 30)

        self.items.join()

    def stop(self) -> None:
        self.stop_event.set()

    def _enqueue_listing(self, listing) -> str:
        pods = [_as_dict(pod) for pod in listing.items]
        for pod in pods:
            self._enqueue(pod)
        active_pod_uids = {
            uid
            for pod in pods
            if (uid := (pod.get("metadata") or {}).get("uid"))
        }
        self.store.reconcile_active_pods(self.namespace, active_pod_uids)
        return listing.metadata.resource_version

    def _enqueue(self, pod: dict, *, deleted: bool = False) -> None:
        # Unbound Pod ingestion belongs to the reusable watch interface in
        # step 3. The placement foundation remains bound-Pod-only for now.
        if not (pod.get("spec") or {}).get("nodeName"):
            return
        while not self.stop_event.is_set():
            try:
                self.items.put((pod, deleted), timeout=0.5)
                self.health.update(queue_saturated=False)
                return
            except queue.Full:
                self.health.update(queue_saturated=True)
                logger.error("KubernetesDaemon work queue is saturated; readiness is failing")

    def _worker(self) -> None:
        while not self.stop_event.is_set() or not self.items.empty():
            try:
                pod, deleted = self.items.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._record(pod, deleted=deleted)
            except Exception:
                # A malformed observation or an unexpected enrichment bug
                # must never terminate a recording worker.
                self.health.update(recording_failed=True)
                logger.exception("unexpected KubernetesDaemon recording failure")
            else:
                self.health.update(recording_failed=False)
            finally:
                self.items.task_done()

    def _record(self, pod: dict, *, deleted: bool = False) -> None:
        pod_uid = (pod.get("metadata") or {}).get("uid")
        try:
            self._record_placement(pod)
        finally:
            if deleted and pod_uid:
                self._mark_deleted(pod_uid)

    def _record_placement(self, pod: dict) -> None:
        failure_reason = pod_node_failure_reason(pod) or _pod_container_exit_reason(pod)
        base = PlacementRecord.for_pod(pod, node_failure_reason=failure_reason)
        result = self._persist(base, "scheduler-neutral Pod placement")
        if result is None:
            return

        metadata = pod.get("metadata") or {}
        logger.info(f"{result.value} placement for Pod {self.namespace}/{metadata.get('name')}")
        if self.owner_resolver is None:
            return

        owner = None
        for attempt in range(self.max_resolution_attempts):
            try:
                owner = self.owner_resolver.resolve(pod)
                break
            except ResolutionPending as error:
                if attempt + 1 == self.max_resolution_attempts or self.stop_event.is_set():
                    logger.warning(f"retained unresolved Pod placement: {error}")
                    return
                self.stop_event.wait(min(2**attempt, 8))
            except UnsupportedOwner as error:
                logger.info(f"retained placement without supported owner: {error}")
                return
            except StaleOwnerReference as error:
                logger.warning(f"retained placement with stale owner reference: {error}")
                return
            except Exception:
                logger.exception("failed to enrich Pod placement; scheduler-neutral evidence was retained")
                return

        assert owner is not None
        owned = base.enriched(owner=owner, node_failure_reason=failure_reason)
        owner_result = self._persist(owned, "Pod owner enrichment")
        if owner_result is None:
            return
        if owner_result != CreateResult.DUPLICATE:
            logger.info(f"{owner_result.value} owner metadata for Pod {self.namespace}/{metadata.get('name')}")

        enriched = owned
        for resolver in self.resource_resolvers:
            resolver_name = type(resolver).__name__
            for attempt in range(self.max_resolution_attempts):
                try:
                    resources = resolver.resolve(pod, owner)
                    enriched = enriched.enriched(associated_resources=resources)
                    resource_result = self._persist(enriched, f"resources from {resolver_name}")
                    if resource_result is None:
                        break
                    if resource_result != CreateResult.DUPLICATE:
                        logger.info(
                            f"{resource_result.value} resources from {resolver_name} for "
                            f"Pod {self.namespace}/{metadata.get('name')}"
                        )
                    break
                except ResolutionPending as error:
                    if attempt + 1 == self.max_resolution_attempts or self.stop_event.is_set():
                        logger.warning(f"retained Pod placement without resources from {resolver_name}: {error}")
                        break
                    self.stop_event.wait(min(2**attempt, 8))
                except UnsupportedOwner as error:
                    logger.info(f"retained Pod placement without resources from {resolver_name}: {error}")
                    break
                except Exception:
                    logger.exception(
                        f"failed to enrich resources from {resolver_name}; "
                        "scheduler-neutral evidence was retained"
                    )
                    break

    def _persist(self, record: PlacementRecord, operation: str) -> CreateResult | None:
        try:
            result = self.store.create(record)
        except Exception:
            self.health.update(persistence_failed=True)
            logger.exception(f"failed to persist {operation}")
            return None
        self.health.update(persistence_failed=False)
        return result

    def _mark_deleted(self, pod_uid: str) -> None:
        try:
            self.store.mark_deleted(pod_uid)
        except Exception:
            self.health.update(persistence_failed=True)
            logger.exception(f"failed to persist deletion for Pod UID {pod_uid}")


def _as_dict(resource) -> dict:
    if isinstance(resource, dict):
        return resource
    return normalize_resource(resource.to_dict())


def _pod_container_exit_reason(pod: dict) -> str | None:
    for container_status in (pod.get("status") or {}).get("containerStatuses") or []:
        terminated = (container_status.get("state") or {}).get("terminated") or {}
        exit_code = terminated.get("exitCode")
        if isinstance(exit_code, int) and exit_code != 0:
            return f"ExitCode:{exit_code}"
    return None
