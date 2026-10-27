# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import requests

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import (
    OwnerIdentity,
    PlacementRecord,
    WorkloadIdentity,
)
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.health import HealthState
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.server import serve_api
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.sqlite_store import (
    SQLitePlacementStore,
)


def _record():
    return PlacementRecord(
        namespace="ns",
        workload=WorkloadIdentity("wl-a", "wl-uid-1"),
        owner=OwnerIdentity("batch/v1", "Job", "job-a", "job-uid-1"),
        pod_name="pod-a",
        pod_uid="pod-uid-1",
        node_name="node-a",
        observed_at="2026-09-14T00:00:00Z",
        resolver_source="owner-uid",
    )


def test_placement_endpoint_returns_scoped_records(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())
    server = serve_api(HealthState(), store, 0, "ns")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params={"namespace": "ns", "workload_uid": "wl-uid-1"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    assert response.json()["items"] == [_record().to_resource()]


def test_placement_endpoint_requires_both_selectors(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    server = serve_api(HealthState(), store, 0, "ns")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params={"namespace": "ns"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 400


def test_placement_endpoint_rejects_other_namespaces(tmp_path):
    """A recorder installation only watches one namespace. Querying a
    different namespace must fail explicitly rather than returning an empty
    list indistinguishable from "no placements recorded yet"."""
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    server = serve_api(HealthState(), store, 0, "ns")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params={"namespace": "other-ns", "workload_uid": "wl-uid-1"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 403
