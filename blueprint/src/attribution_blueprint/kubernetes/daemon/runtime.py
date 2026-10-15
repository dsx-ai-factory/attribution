# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import random
import threading
from datetime import datetime, timezone

from loguru import logger

from attribution_blueprint.kubernetes.daemon.health import HealthState
from attribution_blueprint.kubernetes.daemon.models import (
    CoverageGapReason,
    ObservationEventType,
    PlacementRecord,
    ResourceObservation,
)
from attribution_blueprint.kubernetes.daemon.resolvers import (
    AssociatedResourceResolver,
    OwnershipResolver,
    ResolutionPending,
    StaleOwnerReference,
    UnsupportedOwner,
)
from attribution_blueprint.kubernetes.daemon.store import CreateResult, PlacementStore
from attribution_blueprint.kubernetes.daemon.watch_sources import (
    ResourceWatchSource,
    pod_watch_source,
    selected_metadata,
    source_timestamp,
)
from attribution_blueprint.kubernetes.helpers import pod_container_exit_reason, pod_node_failure_reason


class _ObservationPersistenceFailed(RuntimeError):
    pass


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
        cluster_id: str = "default",
        watch_sources: tuple[ResourceWatchSource, ...] | None = None,
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
        self.cluster_id = cluster_id
        self.watch_sources = watch_sources
        if self.watch_sources is None:
            self.watch_sources = (pod_watch_source(core_api, namespace),) if core_api is not None else ()
        self.health = HealthState()
        self.items: queue.Queue[dict] = queue.Queue(maxsize=queue_size)
        self.worker_count = workers
        self.stop_event = threading.Event()
        self._initial_lists_complete: set[str] = set()
        self._initial_lists_lock = threading.Lock()

    def run(self) -> None:
        workers = [
            threading.Thread(target=self._worker, name=f"kubernetes-daemon-worker-{index}", daemon=True)
            for index in range(self.worker_count)
        ]
        for worker in workers:
            worker.start()

        watchers = [
            threading.Thread(
                target=self._watch_source,
                args=(source,),
                name=f"kubernetes-daemon-watch-{source.name}",
                daemon=True,
            )
            for source in self.watch_sources
        ]
        for watcher in watchers:
            watcher.start()

        while any(watcher.is_alive() for watcher in watchers):
            if self.stop_event.wait(0.5):
                break
        for watcher in watchers:
            watcher.join(timeout=self.watch_timeout_seconds + 1)
        self.items.join()

    def _watch_source(self, source: ResourceWatchSource) -> None:
        from kubernetes.client.exceptions import ApiException

        resource_version = None
        gap_id = None
        checkpoint_loaded = False
        pending_gap_reason: CoverageGapReason | None = None
        pending_gap_resource_version: str | None = None
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                if not checkpoint_loaded:
                    coverage = self.store.get_watch_coverage(
                        self.cluster_id,
                        self.namespace,
                        source.name,
                    )
                    if pending_gap_reason is not None:
                        assert pending_gap_resource_version is not None
                        gap_id = self.store.begin_coverage_gap(
                            cluster_id=self.cluster_id,
                            namespace=self.namespace,
                            source_name=source.name,
                            api_version=source.api_version,
                            kind=source.kind,
                            reason=pending_gap_reason,
                            start_resource_version=pending_gap_resource_version,
                            detected_at=_utc_now(),
                        )
                    elif coverage is not None:
                        if (
                            coverage.api_version != source.api_version
                            or coverage.kind != source.kind
                        ):
                            gap_id = self.store.begin_coverage_gap(
                                cluster_id=self.cluster_id,
                                namespace=self.namespace,
                                source_name=source.name,
                                api_version=source.api_version,
                                kind=source.kind,
                                reason="WATCH_SOURCE_CHANGED",
                                start_resource_version=coverage.resource_version,
                                detected_at=_utc_now(),
                            )
                        else:
                            resource_version = coverage.resource_version
                            self._mark_initial_list_complete(source.name)
                    checkpoint_loaded = True

                if resource_version is None:
                    resources, listed_resource_version = source.list_resources()
                    for resource in resources:
                        if not self._capture(source, resource, "SNAPSHOT"):
                            raise _ObservationPersistenceFailed
                    completed_at = _utc_now()
                    self.store.complete_watch_list(
                        cluster_id=self.cluster_id,
                        namespace=self.namespace,
                        source_name=source.name,
                        api_version=source.api_version,
                        kind=source.kind,
                        resource_version=listed_resource_version,
                        completed_at=completed_at,
                        gap_id=gap_id,
                    )
                    gap_id = None
                    pending_gap_reason = None
                    pending_gap_resource_version = None
                    resource_version = listed_resource_version
                    self._mark_initial_list_complete(source.name)

                for event in source.stream(resource_version, self.watch_timeout_seconds):
                    resource = event.get("object") or {}
                    metadata = resource.get("metadata") or {}
                    event_resource_version = metadata.get("resourceVersion")
                    event_type = event.get("type")
                    if event_type in {"ADDED", "MODIFIED", "DELETED"}:
                        if not self._capture(source, resource, event_type):
                            raise _ObservationPersistenceFailed
                        if event_resource_version:
                            resource_version = event_resource_version
                    elif event_type == "BOOKMARK" and event_resource_version:
                        self._mark_watch_checkpoint(source, event_resource_version)
                        resource_version = event_resource_version
                    if self.stop_event.is_set():
                        break
                backoff = 1.0
            except _ObservationPersistenceFailed:
                logger.warning(
                    f"{source.name} observation persistence failed; "
                    f"restarting from resourceVersion {resource_version or 'initial-list'}"
                )
            except ApiException as error:
                if error.status == 410:
                    logger.warning(f"{source.name} watch resourceVersion expired; relisting")
                    assert resource_version is not None
                    self._mark_initial_list_incomplete(source.name)
                    checkpoint_loaded = False
                    pending_gap_reason = "RESOURCE_VERSION_EXPIRED"
                    pending_gap_resource_version = resource_version
                    resource_version = None
                    continue
                if error.status == 404 and source.kind == "JobSet":
                    logger.warning(
                        f"{source.name} Kubernetes API is absent; disabling the JobSet watch"
                    )
                    self.health.disable_watch(source.name, "api-not-found")
                    self._mark_initial_list_complete(source.name)
                    return
                logger.warning(f"{source.name} watch failed: {error}")
            except Exception:
                logger.exception(f"{source.name} watch failed")
            self.stop_event.wait(backoff + random.uniform(0, backoff / 4))
            backoff = min(backoff * 2, 30)

    def _capture(
        self,
        source: ResourceWatchSource,
        resource: dict,
        event_type: ObservationEventType,
    ) -> bool:
        status = source.selected_status(resource)
        metadata = resource.get("metadata") or {}
        observation = ResourceObservation.from_resource(
            resource,
            cluster_id=self.cluster_id,
            event_type=event_type,
            status=status,
            selected_metadata=selected_metadata(resource),
            source_timestamp=source_timestamp(
                {
                    "status": status,
                    "metadata": {
                        "creationTimestamp": metadata.get("creationTimestamp"),
                        "deletionTimestamp": metadata.get("deletionTimestamp"),
                    },
                }
            ),
        )
        placement = None
        if observation.kind == "Pod" and (resource.get("spec") or {}).get("nodeName"):
            failure_reason = pod_node_failure_reason(resource) or pod_container_exit_reason(resource)
            placement = PlacementRecord.for_pod(resource, node_failure_reason=failure_reason)
        try:
            result = self.store.capture_observation(
                observation,
                source_name=source.name,
                placement=placement,
                advance_checkpoint=event_type != "SNAPSHOT",
            )
        except Exception:
            self.health.update(persistence_failed=True)
            logger.exception(
                f"failed to persist {source.name} capture "
                f"{observation.namespace}/{observation.name}@{observation.resource_version}"
            )
            return False
        self.health.update(persistence_failed=False)

        if placement is not None:
            assert result.placement is not None
            if result.placement != CreateResult.DUPLICATE:
                logger.info(
                    f"{result.placement.value} placement for Pod "
                    f"{placement.namespace}/{placement.pod_name}"
                )
            self._enqueue(resource)
        return True

    def _mark_initial_list_complete(self, source_name: str) -> None:
        with self._initial_lists_lock:
            self._initial_lists_complete.add(source_name)
            all_complete = len(self._initial_lists_complete) == len(self.watch_sources)
        self.health.update(initial_list_complete=all_complete)

    def _mark_initial_list_incomplete(self, source_name: str) -> None:
        with self._initial_lists_lock:
            self._initial_lists_complete.discard(source_name)
        self.health.update(initial_list_complete=False)

    def _mark_watch_checkpoint(
        self,
        source: ResourceWatchSource,
        resource_version: str,
    ) -> None:
        try:
            self.store.mark_watch_checkpoint(
                cluster_id=self.cluster_id,
                namespace=self.namespace,
                source_name=source.name,
                resource_version=resource_version,
                checkpointed_at=_utc_now(),
            )
        except Exception as error:
            self.health.update(persistence_failed=True)
            raise _ObservationPersistenceFailed from error

    def stop(self) -> None:
        self.stop_event.set()

    def _enqueue(self, pod: dict) -> None:
        if not (pod.get("spec") or {}).get("nodeName"):
            return
        try:
            self.items.put_nowait(pod)
            self.health.update(queue_saturated=False)
        except queue.Full:
            # Base placement is already durable by this point. Only optional
            # owner/scheduler association enrichment is deferred here, so
            # dropping this item cannot lose placement evidence. The
            # association may remain unresolved until another observation
            # queues this Pod again.
            self.health.update(queue_saturated=True)
            logger.error(
                "KubernetesDaemon enrichment queue is saturated; "
                "placement was persisted but owner/scheduler enrichment was not queued"
            )

    def _worker(self) -> None:
        while not self.stop_event.is_set() or not self.items.empty():
            try:
                pod = self.items.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._record(pod)
            except Exception:
                # A malformed observation or an unexpected enrichment bug
                # must never terminate a recording worker.
                self.health.update(recording_failed=True)
                logger.exception("unexpected KubernetesDaemon recording failure")
            else:
                self.health.update(recording_failed=False)
            finally:
                self.items.task_done()

    def _record(self, pod: dict) -> None:
        metadata = pod.get("metadata") or {}
        base = self.store.get_by_pod_uid(metadata.get("uid"))
        if base is None:
            logger.error(
                f"cannot enrich Pod {self.namespace}/{metadata.get('name')}: "
                "base placement is not persisted"
            )
            return
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
        owned = base.enriched(owner=owner)
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
