# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_blueprint.connectors.execution_contexts.kueue.resource_resolver import (
    KueueResourceResolver,
)
from attribution_blueprint.kubernetes.daemon.models import (
    ObservationEventType,
    ResourceIdentity,
    ResourceObservation,
)
from attribution_blueprint.kubernetes.daemon.resolvers import UnsupportedOwner


def _pod():
    return {
        "metadata": {
            "namespace": "team-a",
            "annotations": {"kueue.x-k8s.io/workload": "trainer-workload"},
        }
    }


def _owner():
    return ResourceIdentity(
        "batch/v1",
        "Job",
        "team-a",
        "trainer",
        "job-uid",
    )


def _retained_workload(
    uid: str,
    *,
    owner_uid: str = "job-uid",
    event_type: ObservationEventType = "DELETED",
) -> ResourceObservation:
    return ResourceObservation(
        cluster_id="cluster-a",
        api_version="kueue.x-k8s.io/v1beta1",
        kind="Workload",
        namespace="team-a",
        name="trainer-workload",
        uid=uid,
        resource_version=uid,
        event_type=event_type,
        observed_at="2026-10-14T00:00:00Z",
        selected_metadata={
            "labels": {"kueue.x-k8s.io/job-uid": owner_uid},
        },
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


def test_resolves_deleted_workload_from_uid_verified_journal(mocker):
    custom_api = mocker.Mock()
    observation = _retained_workload("original-workload-uid")

    result = KueueResourceResolver(
        custom_api,
        journal=_Journal([observation]),
        cluster_id="cluster-a",
    ).resolve(_pod(), _owner())

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


def test_unrelated_retained_workload_falls_back_to_live_lookup(mocker):
    custom_api = mocker.Mock()
    custom_api.get_namespaced_custom_object.return_value = {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "Workload",
        "metadata": {
            "namespace": "team-a",
            "name": "trainer-workload",
            "uid": "live-workload-uid",
            "labels": {"kueue.x-k8s.io/job-uid": "job-uid"},
        },
    }
    resolver = KueueResourceResolver(
        custom_api,
        journal=_Journal(
            [
                _retained_workload(
                    "unrelated-workload-uid",
                    owner_uid="other-job-uid",
                )
            ]
        ),
        cluster_id="cluster-a",
    )

    assert resolver.resolve(_pod(), _owner()) == (
        ResourceIdentity(
            "kueue.x-k8s.io/v1beta1",
            "Workload",
            "team-a",
            "trainer-workload",
            "live-workload-uid",
        ),
    )
    custom_api.get_namespaced_custom_object.assert_called_once()


def test_ambiguous_retained_workloads_preserve_uncertainty(mocker):
    custom_api = mocker.Mock()
    resolver = KueueResourceResolver(
        custom_api,
        journal=_Journal(
            [
                _retained_workload("deleted-workload-uid"),
                _retained_workload(
                    "live-workload-uid",
                    event_type="MODIFIED",
                ),
            ]
        ),
        cluster_id="cluster-a",
    )

    with pytest.raises(UnsupportedOwner, match="multiple retained"):
        resolver.resolve(_pod(), _owner())

    custom_api.get_namespaced_custom_object.assert_not_called()
