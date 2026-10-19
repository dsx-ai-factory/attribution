# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.connectors.execution_contexts.kueue.helpers import (
    controller_owner,
)
from attribution_blueprint.connectors.execution_contexts.kueue.resource_resolver import (
    KueueResourceResolver,
)
from attribution_blueprint.kubernetes.daemon.models import (
    ResourceIdentity,
    ResourceObservation,
)


class _Journal:
    def __init__(self, observations):
        self.observations = observations

    def list_latest_observations_for_kind(self, cluster_id, namespace, kind):
        assert (cluster_id, namespace, kind) == (
            "cluster-a",
            "team-a",
            "Workload",
        )
        return self.observations


def test_controller_owner_requires_literal_true():
    def resource(controller_marker):
        owner = {"kind": "Job", "name": "trainer", "uid": "job-uid"}
        if controller_marker != "missing":
            owner["controller"] = controller_marker
        return {"metadata": {"ownerReferences": [owner]}}

    assert controller_owner(resource("missing")) is None
    assert controller_owner(resource(None)) is None
    assert controller_owner(resource(False)) is None
    assert controller_owner(resource(True))["uid"] == "job-uid"


def test_resolves_deleted_workload_from_uid_verified_journal(mocker):
    custom_api = mocker.Mock()
    observation = ResourceObservation(
        cluster_id="cluster-a",
        api_version="kueue.x-k8s.io/v1beta1",
        kind="Workload",
        namespace="team-a",
        name="trainer-workload",
        uid="original-workload-uid",
        resource_version="12",
        event_type="DELETED",
        observed_at="2026-10-14T00:00:00Z",
        selected_metadata={
            "labels": {"kueue.x-k8s.io/job-uid": "job-uid"},
        },
    )
    pod = {
        "metadata": {
            "namespace": "team-a",
            "annotations": {"kueue.x-k8s.io/workload": "trainer-workload"},
        }
    }
    owner = ResourceIdentity(
        "batch/v1",
        "Job",
        "team-a",
        "trainer",
        "job-uid",
    )

    result = KueueResourceResolver(
        custom_api,
        journal=_Journal([observation]),
        cluster_id="cluster-a",
    ).resolve(pod, owner)

    assert result == (
        ResourceIdentity(
            "kueue.x-k8s.io/v1beta1",
            "Workload",
            "team-a",
            "trainer-workload",
            "original-workload-uid",
        ),
    )
    custom_api.get_namespaced_custom_object.assert_not_called()
