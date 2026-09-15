# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import sqlite3

import pytest

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import (
    OwnerIdentity,
    PlacementRecord,
    WorkloadIdentity,
)
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.sqlite_store import (
    SQLitePlacementStore,
)
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.store import (
    CreateResult,
    PlacementConflictError,
)


def _record(*, pod_uid="pod-uid-1", node="node-a", reason=None):
    return PlacementRecord(
        namespace="ns",
        workload=WorkloadIdentity(name="wl-a", uid="wl-uid-1"),
        owner=OwnerIdentity(api_version="batch/v1", kind="Job", name="job-a", uid="job-uid-1"),
        pod_name=f"pod-{pod_uid}",
        pod_uid=pod_uid,
        node_name=node,
        observed_at="2026-09-14T00:00:00Z",
        resolver_source="owner-uid",
        node_failure_reason=reason,
    )


def test_records_survive_store_reopen(tmp_path):
    path = tmp_path / "placements.db"
    assert SQLitePlacementStore(str(path)).create(_record()) == CreateResult.CREATED

    records = SQLitePlacementStore(str(path)).list_for_workload("ns", "wl-uid-1")

    assert records == [_record()]


def test_duplicate_is_idempotent(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())

    assert store.create(_record()) == CreateResult.DUPLICATE


def test_later_failure_evidence_updates_raw_record(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())

    assert store.create(_record(reason="NodeLost")) == CreateResult.UPDATED
    assert store.list_for_workload("ns", "wl-uid-1")[0].node_failure_reason == "NodeLost"


def test_conflicting_immutable_placement_is_rejected(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())

    with pytest.raises(PlacementConflictError):
        store.create(_record(node="node-b"))


def test_queries_are_scoped_by_namespace_and_workload(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())

    assert store.list_for_workload("other", "wl-uid-1") == []
    assert store.list_for_workload("ns", "other-workload") == []


def test_default_retention_prunes_records_older_than_30_days(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    store.create(_record())
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE placement_record SET updated_at = datetime('now', '-31 days')")

    assert SQLitePlacementStore(str(path)).list_for_workload("ns", "wl-uid-1") == []
