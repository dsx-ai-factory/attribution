# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
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


def test_placement_endpoint_queries_owner_without_scheduler_association(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())
    server = serve_api(HealthState(), store, 0, "team-a")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params={"namespace": "team-a", "owner_uid": "job-uid-1"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    assert response.json()["items"] == [_record().to_resource()]


def test_placement_endpoint_queries_pod_uid(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())
    server = serve_api(HealthState(), store, 0, "team-a")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params={"namespace": "team-a", "pod_uid": "pod-uid-1"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    assert response.json()["items"] == [_record().to_resource()]


class _FailingReader:
    """A `PlacementReader` whose methods always raise, for exercising the
    endpoint's 500 branch without depending on a real storage failure."""

    def list_for_pod(self, namespace, pod_uid):
        raise RuntimeError("synthetic storage failure")

    def list_for_owner(self, namespace, owner_uid):
        raise RuntimeError("synthetic storage failure")

    def list_for_resource(self, namespace, resource_uid):
        raise RuntimeError("synthetic storage failure")


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"pod_uid": "pod-uid-1"}, 400),
        ({"namespace": "team-a"}, 400),
        ({"namespace": "team-a", "pod_uid": "p", "owner_uid": "o"}, 400),
        ({"namespace": "team-b", "pod_uid": "pod-uid-1"}, 403),
    ],
)
def test_placement_endpoint_rejects_invalid_queries(tmp_path, params, status):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    server = serve_api(HealthState(), store, 0, "team-a")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params=params,
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == status


def test_placement_endpoint_returns_500_on_reader_failure():
    server = serve_api(HealthState(), _FailingReader(), 0, "team-a")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params={"namespace": "team-a", "pod_uid": "pod-uid-1"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 500


def test_unknown_path_returns_404(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    server = serve_api(HealthState(), store, 0, "team-a")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/unknown",
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 404


def test_placement_endpoint_can_enable_a_resource_query_alias(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.create(_record())
    server = serve_api(
        HealthState(),
        store,
        0,
        "team-a",
        resource_uid_aliases=("legacy_resource_uid",),
    )
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/placements",
            params={"namespace": "team-a", "legacy_resource_uid": "admission-uid-1"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    assert response.json()["items"] == [_record().to_resource()]
