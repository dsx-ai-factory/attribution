# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests

from attribution_blueprint.kubernetes.daemon.client import KubernetesDaemonClient
from attribution_blueprint.kubernetes.daemon.health import HealthState
from attribution_blueprint.kubernetes.daemon.models import (
    OwnerReference,
    PlacementRecord,
    ResourceIdentity,
    ResourceObservation,
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


def _capture(store: SQLitePlacementStore, observation: ResourceObservation) -> None:
    store.capture_observation(
        observation,
        source_name=f"{observation.kind.lower()}s",
        placement=None,
        advance_checkpoint=False,
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


class _FailingReader:
    def list_for_resource(self, namespace, resource_uid):
        raise RuntimeError("synthetic storage failure")


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"resource_uid": "admission-uid-1"}, 400),
        ({"namespace": "team-a"}, 400),
        ({"namespace": "team-b", "resource_uid": "admission-uid-1"}, 403),
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
            params={"namespace": "team-a", "resource_uid": "admission-uid-1"},
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


def test_readiness_reports_disabled_optional_watches(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    state = HealthState()
    state.update(initial_list_complete=True)
    state.disable_watch("jobsets", "api-not-found")
    server = serve_api(state, store, 0, "team-a")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/readyz",
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "disabledWatches": {"jobsets": "api-not-found"},
    }


def test_history_endpoint_returns_controller_descendants_and_relationships(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    job = ResourceObservation(
        cluster_id="cluster-a",
        api_version="batch/v1",
        kind="Job",
        namespace="team-a",
        name="trainer",
        uid="job-uid-1",
        resource_version="1",
        event_type="ADDED",
        observed_at="2026-10-14T00:00:00Z",
        status={"active": 1},
    )
    pod = ResourceObservation(
        cluster_id="cluster-a",
        api_version="v1",
        kind="Pod",
        namespace="team-a",
        name="trainer-abcde",
        uid="pod-uid-1",
        resource_version="2",
        event_type="MODIFIED",
        observed_at="2026-10-14T00:00:01Z",
        owner_references=(
            OwnerReference(
                api_version="batch/v1",
                kind="Job",
                name="trainer",
                uid="job-uid-1",
                controller=True,
            ),
        ),
        selected_metadata={
            "annotations": {"batch.kubernetes.io/job-completion-index": "0"}
        },
        status={"phase": "Running"},
    )
    _capture(store, job)
    _capture(store, pod)
    server = serve_api(
        HealthState(),
        store,
        0,
        "team-a",
        cluster_id="cluster-a",
    )
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/history",
            params={"namespace": "team-a", "controller_uid": "job-uid-1"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    history = response.json()
    assert history["clusterId"] == "cluster-a"
    assert history["namespace"] == "team-a"
    assert history["controllerUid"] == "job-uid-1"
    assert [resource["uid"] for resource in history["resources"]] == [
        "job-uid-1",
        "pod-uid-1",
    ]
    assert history["relationships"] == [
        {
            "owner": {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "namespace": "team-a",
                "name": "trainer",
                "uid": "job-uid-1",
            },
            "resource": {
                "apiVersion": "v1",
                "kind": "Pod",
                "namespace": "team-a",
                "name": "trainer-abcde",
                "uid": "pod-uid-1",
            },
            "controller": True,
        }
    ]
    assert [observation["metadata"]["uid"] for observation in history["observations"]] == [
        "job-uid-1",
        "pod-uid-1",
    ]
    assert history["observations"][1]["selectedMetadata"] == pod.selected_metadata


def test_daemon_client_reads_controller_history(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    observation = ResourceObservation(
        cluster_id="cluster-a",
        api_version="batch/v1",
        kind="Job",
        namespace="team-a",
        name="trainer",
        uid="job-uid-1",
        resource_version="1",
        event_type="SNAPSHOT",
        observed_at="2026-10-14T00:00:00Z",
    )
    _capture(store, observation)
    server = serve_api(
        HealthState(),
        store,
        0,
        "team-a",
        cluster_id="cluster-a",
    )
    try:
        client = KubernetesDaemonClient(f"http://127.0.0.1:{server.server_port}")
        history = client.get_history("team-a", "job-uid-1")
    finally:
        server.shutdown()
        server.server_close()

    assert history["controllerUid"] == "job-uid-1"
    assert [item["metadata"]["uid"] for item in history["observations"]] == ["job-uid-1"]
