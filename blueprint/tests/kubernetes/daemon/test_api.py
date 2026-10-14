# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import requests

from attribution_blueprint.kubernetes.daemon.health import HealthState
from attribution_blueprint.kubernetes.daemon.models import (
    PlacementRecord,
    ResourceIdentity,
)
from attribution_blueprint.kubernetes.daemon.server import serve_api
from attribution_blueprint.kubernetes.daemon.sqlite_store import SQLitePlacementStore


def _record() -> PlacementRecord:
    return PlacementRecord(
        namespace="team-a",
        pod_name="trainer-0",
        pod_uid="pod-uid-1",
        node_name="node-a",
        observed_at="2026-10-13T20:00:00Z",
        owner=ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1"),
        associated_resources=(
            ResourceIdentity(
                "scheduling.example.io/v1",
                "Admission",
                "team-a",
                "trainer-admission",
                "admission-uid-1",
            ),
        ),
    )


def test_placement_endpoint_queries_associated_resource(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())
    server = serve_api(HealthState(), store, 0, "team-a")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params={
                "namespace": "team-a",
                "resource_uid": "admission-uid-1",
            },
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    assert response.json()["items"] == [_record().to_resource()]


def test_placement_endpoint_requires_resource_uid(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())
    server = serve_api(HealthState(), store, 0, "team-a")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params={"namespace": "team-a"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 400
    assert response.json() == {"error": "namespace and resource_uid are required"}
