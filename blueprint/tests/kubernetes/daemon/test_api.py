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


def test_history_endpoint_discovers_controller_observations(tmp_path):
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
    store.complete_watch_list(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        resource_version="2",
        completed_at="2026-10-13T23:59:00Z",
        gap_id=None,
    )
    gap_id = store.begin_coverage_gap(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        reason="RESOURCE_VERSION_EXPIRED",
        start_resource_version="2",
        started_at="2026-10-14T00:00:00Z",
        detected_at="2026-10-14T00:00:02Z",
    )
    store.complete_watch_list(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        resource_version="3",
        completed_at="2026-10-14T00:00:03Z",
        gap_id=gap_id,
    )
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
            params={
                "namespace": "team-a",
                "kind": "Job",
                "name": "trainer",
            },
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    [history] = response.json()["items"]
    assert history["namespace"] == "team-a"
    assert [observation["metadata"]["uid"] for observation in history["observations"]] == [
        "job-uid-1",
        "pod-uid-1",
    ]
    assert history["observations"][1]["selectedMetadata"] == pod.selected_metadata
    assert history["coverage"] == {
        "sources": [
            {
                "sourceName": "pods",
                "apiVersion": "v1",
                "kind": "Pod",
                "coverageStartedAt": "2026-10-13T23:59:00Z",
                "resourceVersion": "3",
                "checkpointedAt": "2026-10-14T00:00:03Z",
            }
        ],
        "gaps": [
            {
                "sourceName": "pods",
                "apiVersion": "v1",
                "kind": "Pod",
                "reason": "RESOURCE_VERSION_EXPIRED",
                "startResourceVersion": "2",
                "startedAt": "2026-10-14T00:00:00Z",
                "detectedAt": "2026-10-14T00:00:02Z",
                "endResourceVersion": "3",
                "recoveredAt": "2026-10-14T00:00:03Z",
            }
        ],
        "disabledWatches": {},
    }


def test_history_window_keeps_active_and_delayed_complete_histories(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))

    def workload(
        uid,
        resource_version,
        observed_at,
        conditions,
    ):
        return ResourceObservation(
            cluster_id="cluster-a",
            api_version="kueue.x-k8s.io/v1beta1",
            kind="Workload",
            namespace="team-a",
            name=uid,
            uid=uid,
            resource_version=resource_version,
            event_type="MODIFIED",
            observed_at=observed_at,
            status={"conditions": conditions},
        )

    active = workload(
        "active-uid",
        "1",
        "2026-10-01T00:00:00Z",
        [
            {
                "type": "Admitted",
                "status": "True",
                "lastTransitionTime": "2026-10-01T00:00:00Z",
            }
        ],
    )
    delayed_early = workload(
        "delayed-uid",
        "1",
        "2026-10-01T00:00:00Z",
        [
            {
                "type": "Admitted",
                "status": "True",
                "lastTransitionTime": "2026-10-01T00:00:00Z",
            }
        ],
    )
    delayed_terminal = workload(
        "delayed-uid",
        "2",
        "2026-10-15T00:00:00Z",
        [
            {
                "type": "Finished",
                "status": "True",
                "lastTransitionTime": "2026-10-05T00:00:00Z",
            }
        ],
    )
    stale = workload(
        "stale-uid",
        "1",
        "2026-10-05T00:00:00Z",
        [
            {
                "type": "Finished",
                "status": "True",
                "lastTransitionTime": "2026-10-05T00:00:00Z",
            }
        ],
    )
    requeued = workload(
        "requeued-uid",
        "1",
        "2026-10-05T00:00:00Z",
        [
            {
                "type": "Evicted",
                "status": "True",
                "lastTransitionTime": "2026-10-05T00:00:00Z",
            },
            {
                "type": "Requeued",
                "status": "True",
                "lastTransitionTime": "2026-10-06T00:00:00Z",
            },
        ],
    )
    for observation in (
        active,
        delayed_early,
        delayed_terminal,
        requeued,
        stale,
    ):
        _capture(store, observation)

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
            params={
                "namespace": "team-a",
                "kind": "Workload",
                "start": "2026-10-14T00:00:00Z",
                "end": "2026-10-14T23:59:59Z",
            },
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    histories = response.json()["items"]
    histories_by_uid = {
        history["observations"][0]["metadata"]["uid"]: history
        for history in histories
    }
    assert set(histories_by_uid) == {
        "active-uid",
        "delayed-uid",
        "requeued-uid",
    }
    assert [
        observation["metadata"]["resourceVersion"]
        for observation in histories_by_uid["delayed-uid"]["observations"]
    ] == ["1", "2"]


def test_history_endpoint_computes_coverage_once_for_multiple_controllers(tmp_path, mocker):
    """Coverage/gaps are scoped to (cluster_id, namespace), not to any one
    controller, so the handler must compute them once per request rather
    than once per controller in the response."""
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    for index in range(2):
        _capture(
            store,
            ResourceObservation(
                cluster_id="cluster-a",
                api_version="batch/v1",
                kind="Job",
                namespace="team-a",
                name=f"trainer-{index}",
                uid=f"job-uid-{index}",
                resource_version="1",
                event_type="ADDED",
                observed_at="2026-10-14T00:00:00Z",
            ),
        )
    coverages = mocker.spy(store, "list_watch_coverages")
    gaps = mocker.spy(store, "list_coverage_gaps")
    server = serve_api(HealthState(), store, 0, "team-a", cluster_id="cluster-a")
    try:
        response = requests.get(
            f"http://127.0.0.1:{server.server_port}/v1/history",
            params={"namespace": "team-a", "kind": "Job"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    assert len(response.json()["items"]) == 2
    assert coverages.call_count == 1
    assert gaps.call_count == 1


def test_daemon_client_finds_retained_history(tmp_path):
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
        [history] = client.find_histories(
            "team-a",
            "Job",
            name="trainer",
        )
    finally:
        server.shutdown()
        server.server_close()

    assert [item["metadata"]["uid"] for item in history["observations"]] == ["job-uid-1"]
