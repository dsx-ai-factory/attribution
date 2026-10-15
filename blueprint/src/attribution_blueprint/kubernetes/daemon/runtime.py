# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import threading
import time
from datetime import datetime, timezone
from itertools import count

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

_WORK_GENERATIONS = count()


class RequiredPersistenceError(RuntimeError):
    """A required placement write failed before its watch checkpoint."""


class _ObservationPersistenceFailed(RuntimeError):
    pass


class KubernetesDaemon:
    """Watch Pods, record their placement, and enrich it with ownership and
    scheduler-specific evidence, independently of whether enrichment succeeds."""

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
        retry_poll_seconds: float = 1.0,
        cluster_id: str = "default",
        watch_sources: tuple[ResourceWatchSource, ...] | None = None,
    ):
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if max_resolution_attempts <= 0:
            raise ValueError("max_resolution_attempts must be positive")
        self.core_api = core_api
        self.store = store
        self.namespace = namespace
        self.owner_resolver = owner_resolver
        self.resource_resolvers = resource_resolvers
        self.watch_timeout_seconds = watch_timeout_seconds
        self.max_resolution_attempts = max_resolution_attempts
        self.retry_poll_seconds = retry_poll_seconds
        self.cluster_id = cluster_id
        self.watch_sources = watch_sources
        if self.watch_sources is None:
            self.watch_sources = (pod_watch_source(core_api, namespace),) if core_api is not None else ()
        self.health = HealthState()
        self.worker_count = workers
        self.stop_event = threading.Event()

        # `enrichment_capacity` bounds the distinct Pod UIDs tracked below
        # across `_pending`, `_in_flight`, and `_retry` combined.
        self.enrichment_capacity = queue_size
        self._pending: dict[str, tuple[dict, int]] = {}
        self._in_flight: set[str] = set()
        self._retry: dict[str, tuple[dict, int]] = {}
        self._retry_attempts: dict[str, int] = {}
        self._latest_generation: dict[str, int] = {}
        self._lock = threading.Lock()
        self._work_available = threading.Condition(self._lock)
        self._initial_lists_complete: set[str] = set()
        self._initial_lists_lock = threading.Lock()

    def run(self) -> None:
        workers = [
            threading.Thread(target=self._worker, name=f"kubernetes-daemon-worker-{index}", daemon=True)
            for index in range(self.worker_count)
        ]
        retry_thread = threading.Thread(
            target=self._retry_loop, name="kubernetes-daemon-retry", daemon=True
        )
        for worker in workers:
            worker.start()
        retry_thread.start()

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
        # Drain whatever is already pending or in flight before returning;
        # anything still deferred in `_retry` is abandoned on shutdown.
        while self._pending or self._in_flight:
            with self._work_available:
                self._work_available.wait(timeout=0.5)

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
                    if source.kind == "Pod":
                        self._reconcile_active_pods(resources)
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
                    self.health.update(initial_list_complete=False)
                    continue
                if error.status == 404 and source.kind == "JobSet":
                    logger.warning(
                        f"{source.name} Kubernetes API is absent; disabling the JobSet watch"
                    )
                    self.health.disable_watch(source.name, "api-not-found")
                    self._mark_initial_list_complete(source.name)
                    return
                logger.warning(f"{source.name} watch failed: {error}")
            except RequiredPersistenceError as error:
                logger.warning(f"{error}; replaying from the last persisted resourceVersion")
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
                mark_deleted=placement is not None and event_type == "DELETED",
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
            if observation.uid:
                self._schedule(observation.uid, resource)
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

    def _reconcile_active_pods(self, pods: list[dict]) -> None:
        active_pod_uids = {
            uid
            for pod in pods
            if (uid := (pod.get("metadata") or {}).get("uid"))
        }
        self._reconcile_active_pod_uids(active_pod_uids)

    def _enqueue(self, pod: dict, *, deleted: bool = False) -> None:
        # Unbound Pod ingestion belongs to the reusable watch interface in
        # step 3. The placement foundation remains bound-Pod-only for now.
        if not (pod.get("spec") or {}).get("nodeName"):
            return
        pod_uid = (pod.get("metadata") or {}).get("uid")
        captured = self._capture_placement(pod)
        deletion_persisted = True
        if deleted and pod_uid:
            deletion_persisted = self._mark_deleted(pod_uid)
        if not captured or not deletion_persisted:
            raise RequiredPersistenceError(
                f"required placement persistence failed for Pod UID {pod_uid}"
            )
        if not pod_uid:
            return
        self._schedule(pod_uid, pod)

    def _schedule(self, pod_uid: str, pod: dict) -> None:
        """Track the latest `(pod, generation)` snapshot for `pod_uid`.

        An already-tracked UID is updated in place; there's no special-casing
        for what changed about it. New UIDs are skipped when capacity is full."""
        generation = next(_WORK_GENERATIONS)
        with self._work_available:
            admitted = self._pending.keys() | self._in_flight | self._retry.keys()
            if pod_uid in admitted:
                self._retry.pop(pod_uid, None)
                self._retry_attempts.pop(pod_uid, None)
                self._pending[pod_uid] = (pod, generation)
            elif len(admitted) < self.enrichment_capacity:
                self._pending[pod_uid] = (pod, generation)
            else:
                logger.error(
                    f"KubernetesDaemon enrichment capacity ({self.enrichment_capacity}) is saturated; "
                    f"skipped owner/scheduler enrichment for new Pod UID {pod_uid}"
                )
                return
            self._latest_generation[pod_uid] = generation
            self._work_available.notify()

    def _take_work(self, timeout: float) -> tuple[str, dict, int] | None:
        # A Pod UID already in `_in_flight` is excluded here: only one worker
        # may enrich a given Pod UID at a time. A newer snapshot for that UID
        # waits in `_pending` until the active attempt finishes, instead of
        # racing it on another worker.
        deadline = time.monotonic() + timeout
        with self._work_available:
            while True:
                pod_uid = next((uid for uid in self._pending if uid not in self._in_flight), None)
                if pod_uid is not None:
                    pod, generation = self._pending.pop(pod_uid)
                    self._in_flight.add(pod_uid)
                    return pod_uid, pod, generation
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._work_available.wait(timeout=remaining)

    def _complete_work(self, pod_uid: str) -> None:
        with self._work_available:
            self._in_flight.discard(pod_uid)
            if (
                pod_uid not in self._pending
                and pod_uid not in self._in_flight
                and pod_uid not in self._retry
            ):
                self._latest_generation.pop(pod_uid, None)
                self._retry_attempts.pop(pod_uid, None)
            self._work_available.notify()

    def _schedule_retry(self, pod_uid: str, pod: dict, generation: int) -> None:
        """Defer a failed enrichment attempt for retry, unless a newer
        snapshot for this Pod UID has already superseded it."""
        with self._work_available:
            if pod_uid in self._pending:
                return
            attempts = self._retry_attempts.get(pod_uid, 0) + 1
            if attempts >= self.max_resolution_attempts:
                logger.warning(
                    f"retained unresolved Pod placement for Pod UID {pod_uid} "
                    f"after {attempts} enrichment attempts"
                )
                self._retry_attempts.pop(pod_uid, None)
                return
            self._retry_attempts[pod_uid] = attempts
            self._retry[pod_uid] = (pod, generation)

    def _retry_loop(self) -> None:
        while not self.stop_event.is_set():
            self._promote_retries()
            self.stop_event.wait(self.retry_poll_seconds)

    def _promote_retries(self) -> None:
        """Move deferred retries back into pending.

        Called on a fixed interval by `_retry_loop`, never by a worker."""
        with self._work_available:
            if self.stop_event.is_set():
                return
            ready = list(self._retry)
            for pod_uid in ready:
                self._pending[pod_uid] = self._retry.pop(pod_uid)
            if ready:
                self._work_available.notify_all()

    def _worker(self) -> None:
        while not self.stop_event.is_set() or self._pending or self._in_flight:
            item = self._take_work(timeout=0.5)
            if item is None:
                continue
            pod_uid, pod, generation = item
            needs_retry = False
            try:
                needs_retry = self._record(pod, generation=generation)
            except Exception:
                # A malformed observation or an unexpected enrichment bug
                # must never terminate a recording worker.
                logger.exception("unexpected KubernetesDaemon recording failure")
            finally:
                if needs_retry:
                    self._schedule_retry(pod_uid, pod, generation)
                self._complete_work(pod_uid)

    def _record(
        self,
        pod: dict,
        *,
        generation: int | None = None,
    ) -> bool:
        """Enrich `pod`. Returns True when enrichment should be retried later."""
        metadata = pod.get("metadata") or {}
        try:
            base = self.store.get_by_pod_uid(metadata.get("uid"))
        except Exception:
            logger.exception("failed to read stored placement before enrichment")
            failure_reason = pod_node_failure_reason(pod) or pod_container_exit_reason(pod)
            base = PlacementRecord.for_pod(pod, node_failure_reason=failure_reason)
        if base is None:
            logger.error(
                f"cannot enrich Pod {self.namespace}/{metadata.get('name')}: "
                "base placement is not persisted"
            )
            return False
        return self._enrich_placement(pod, base, generation)

    def _capture_placement(self, pod: dict) -> bool:
        failure_reason = pod_node_failure_reason(pod) or pod_container_exit_reason(pod)
        base = PlacementRecord.for_pod(pod, node_failure_reason=failure_reason)
        result = self._persist(base, "scheduler-neutral Pod placement")
        if result is None:
            return False

        metadata = pod.get("metadata") or {}
        if result != CreateResult.DUPLICATE:
            logger.info(f"{result.value} placement for Pod {self.namespace}/{metadata.get('name')}")
        return True

    def _enrich_placement(
        self,
        pod: dict,
        base: PlacementRecord,
        generation: int | None,
    ) -> bool:
        """Resolve and persist owner/resource enrichment for `pod`.

        Each resolver is called exactly once; a failure that should be
        retried is reported via the return value, not retried inline."""
        metadata = pod.get("metadata") or {}
        if self.owner_resolver is None:
            return False
        # Unobserved relationship changes cannot be reconstructed after deletion.
        cached_owner = (
            base.owner
            if _stored_owner_matches_pod(base, pod)
            else None
        )
        try:
            owner = self.owner_resolver.resolve(pod)
        except ResolutionPending as error:
            if cached_owner is not None:
                owner = cached_owner
            else:
                logger.info(f"deferred Pod owner enrichment for retry: {error}")
                return True
        except UnsupportedOwner as error:
            logger.info(f"retained placement without supported owner: {error}")
            return False
        except StaleOwnerReference as error:
            logger.warning(f"retained placement with stale owner reference: {error}")
            return False
        except Exception as error:
            # Resolvers treat all other errors as retryable rather
            # than dropping enrichment outright.
            logger.warning(f"deferred Pod owner enrichment after an unexpected failure, will retry: {error}")
            return True

        current = PlacementRecord.for_pod(pod)
        owned = current.enriched(owner=owner)
        owner_result, owner_retry = self._persist_enrichment(
            owned,
            "Pod owner enrichment",
            generation,
        )
        if owner_retry:
            return True
        if owner_result is None:
            # A newer generation for this Pod UID already superseded this
            # write; that newer work owns any further enrichment.
            return False
        if owner_result != CreateResult.DUPLICATE:
            logger.info(f"{owner_result.value} owner metadata for Pod {self.namespace}/{metadata.get('name')}")

        enriched = owned
        needs_retry = False
        for resolver in self.resource_resolvers:
            resolver_name = type(resolver).__name__
            try:
                resources = resolver.resolve(pod, owner)
            except ResolutionPending as error:
                logger.info(
                    f"deferred Pod resources from {resolver_name} for retry: {error}"
                )
                needs_retry = True
                continue
            except UnsupportedOwner as error:
                logger.info(f"retained Pod placement without resources from {resolver_name}: {error}")
                continue
            except Exception as error:
                # Same reasoning as the owner-resolution case above: an
                # unexpected error (e.g. a transient HTTP 503) is retryable,
                # not a reason to permanently drop this resolver's evidence.
                logger.warning(
                    f"deferred Pod resources from {resolver_name} after an unexpected failure, "
                    f"will retry: {error}"
                )
                needs_retry = True
                continue
            enriched = enriched.enriched(associated_resources=resources)
            resource_result, resource_retry = self._persist_enrichment(
                enriched,
                f"resources from {resolver_name}",
                generation,
            )
            if resource_retry:
                needs_retry = True
                continue
            if resource_result is not None and resource_result != CreateResult.DUPLICATE:
                logger.info(
                    f"{resource_result.value} resources from {resolver_name} for "
                    f"Pod {self.namespace}/{metadata.get('name')}"
                )
        return needs_retry

    def _persist_enrichment(
        self,
        record: PlacementRecord,
        operation: str,
        generation: int | None,
    ) -> tuple[CreateResult | None, bool]:
        if generation is None:
            result = self._persist(record, operation)
            return result, result is None
        with self._work_available:
            if self._latest_generation.get(record.pod_uid) != generation:
                return None, False
            result = self._persist(record, operation)
            return result, result is None

    def _persist(self, record: PlacementRecord, operation: str) -> CreateResult | None:
        try:
            result = self.store.create(record)
        except Exception:
            self.health.update(persistence_failed=True)
            logger.exception(f"failed to persist {operation}")
            return None
        self.health.update(persistence_failed=False)
        return result

    def _mark_deleted(self, pod_uid: str) -> bool:
        try:
            self.store.mark_deleted(pod_uid)
        except Exception:
            self.health.update(persistence_failed=True)
            logger.exception(f"failed to persist deletion for Pod UID {pod_uid}")
            return False
        return True

    def _reconcile_active_pod_uids(self, active_pod_uids: set[str]) -> None:
        try:
            self.store.reconcile_active_pods(self.namespace, active_pod_uids)
        except Exception as error:
            self.health.update(persistence_failed=True)
            logger.exception("failed to reconcile active Pod placements")
            raise RequiredPersistenceError(
                f"required placement reconciliation failed for namespace {self.namespace}"
            ) from error
        self.health.update(persistence_failed=False)


def _stored_owner_matches_pod(record: PlacementRecord, pod: dict) -> bool:
    owner = record.owner
    controllers = [
        reference
        for reference in (pod.get("metadata") or {}).get("ownerReferences") or []
        if reference.get("controller") is True and reference.get("kind") == "Job"
    ]
    if owner is None or len(controllers) != 1:
        return False
    controller_uid = controllers[0].get("uid")
    if owner.kind == "Job":
        return owner.uid == controller_uid
    if owner.kind == "JobSet":
        job_references = [
            reference
            for reference in record.owner_references
            if reference.kind == "Job"
        ]
        return len(job_references) == 1 and job_references[0].uid == controller_uid
    return False
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
