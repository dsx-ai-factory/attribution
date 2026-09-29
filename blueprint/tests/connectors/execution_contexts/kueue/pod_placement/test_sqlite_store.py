# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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


def test_get_by_pod_uid_returns_stored_record(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())

    assert store.get_by_pod_uid("pod-uid-1") == _record()
    assert store.get_by_pod_uid("missing") is None


def test_duplicate_is_idempotent(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())

    assert store.create(_record()) == CreateResult.DUPLICATE


def test_later_failure_evidence_updates_raw_record(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())

    assert store.create(_record(reason="NodeLost")) == CreateResult.UPDATED
    assert store.list_for_workload("ns", "wl-uid-1")[0].node_failure_reason == "NodeLost"


def test_stronger_node_loss_evidence_upgrades_generic_exit_code(tmp_path):
    """Exit code arrives first, then node-loss evidence: the stronger,
    more specific evidence must overwrite the weaker one, surviving a
    store reopen."""
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    store.create(_record(reason="ExitCode:137"))

    assert store.create(_record(reason="NodeLost")) == CreateResult.UPDATED
    assert SQLitePlacementStore(str(path)).list_for_workload("ns", "wl-uid-1")[0].node_failure_reason == "NodeLost"


def test_node_loss_evidence_arriving_before_exit_code_is_not_downgraded(tmp_path):
    """Node-loss evidence arrives first, then a generic exit code: the
    weaker evidence must not overwrite the stronger evidence already
    stored, and must not raise a conflict either."""
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    store.create(_record(reason="NodeLost"))

    assert store.create(_record(reason="ExitCode:137")) == CreateResult.DUPLICATE
    assert SQLitePlacementStore(str(path)).list_for_workload("ns", "wl-uid-1")[0].node_failure_reason == "NodeLost"


def test_two_distinct_node_loss_reasons_are_treated_as_equivalent(tmp_path):
    """NodeLost and DeletionByPodGC are different Kubernetes signaling
    mechanisms for the same underlying node-loss fact. Neither should
    raise a conflict or overwrite the other; whichever is recorded first
    is kept."""
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record(reason="NodeLost"))

    assert store.create(_record(reason="DeletionByPodGC")) == CreateResult.DUPLICATE
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
