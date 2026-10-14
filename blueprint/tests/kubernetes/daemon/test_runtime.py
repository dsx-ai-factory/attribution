# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.kubernetes.daemon.models import ResourceIdentity
from attribution_blueprint.kubernetes.daemon.resolvers import ResolutionPending
from attribution_blueprint.kubernetes.daemon.runtime import KubernetesDaemon
from attribution_blueprint.kubernetes.daemon.sqlite_store import SQLitePlacementStore
from attribution_blueprint.kubernetes.daemon.store import CreateResult


def _pod() -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "trainer-0",
            "namespace": "team-a",
            "uid": "pod-uid-1",
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": "trainer",
                    "uid": "job-uid-1",
                    "controller": True,
                }
            ],
        },
        "spec": {"nodeName": "node-a"},
        "status": {},
    }


class _OwnerResolver:
    def resolve(self, pod):
        return ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")


class _PendingResourceResolver:
    def resolve(self, pod, owner):
        raise ResolutionPending("Workload has not appeared")


class _FailOnEnrichmentStore:
    def __init__(self):
        self.calls = 0

    def create(self, record):
        self.calls += 1
        if self.calls == 1:
            return CreateResult.CREATED
        raise OSError("synthetic storage failure")


def test_records_placement_before_any_resolution(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")

    daemon._record(_pod())

    record = store.get_by_pod_uid("pod-uid-1")
    assert record is not None
    assert record.node_name == "node-a"
    assert record.owner is None
    assert record.associated_resources == ()


def test_retains_owner_when_optional_scheduler_resolution_exhausts(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
        resource_resolvers=(_PendingResourceResolver(),),
        max_resolution_attempts=1,
    )

    daemon._record(_pod())

    record = store.get_by_pod_uid("pod-uid-1")
    assert record is not None
    assert record.owner == ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")
    assert record.associated_resources == ()


def test_owner_enrichment_write_failure_does_not_escape_or_kill_recording():
    store = _FailOnEnrichmentStore()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
    )

    daemon._record(_pod())

    assert store.calls == 2
    assert daemon.health.persistence_failed is True


def test_worker_survives_unexpected_recording_failure(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    daemon.items.put(_pod())
    daemon.items.put(_pod())
    daemon.stop_event.set()
    calls = 0

    def fail_once(_pod):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic processing failure")

    daemon._record = fail_once
    daemon._worker()

    assert calls == 2
    assert daemon.items.unfinished_tasks == 0
    assert daemon.health.recording_failed is False
