# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_blueprint.connectors.execution_contexts.kai.resource_resolver import KaiResourceResolver
from attribution_blueprint.kubernetes.daemon.models import (
    ResourceIdentity,
    ResourceObservation,
)
from attribution_blueprint.kubernetes.daemon.resolvers import UnsupportedOwner


def _pod():
    return {
        "metadata": {
            "namespace": "team-a",
            "annotations": {"pod-group-name": "pg-a"},
        }
    }


def _owner():
    return ResourceIdentity("batch/v1", "Job", "team-a", "job-a", "job-uid")


class _Journal:
    def __init__(self, observations):
        self.observations = observations

    def list_latest_observations_for_kind(self, cluster_id, namespace, kind):
        assert (cluster_id, namespace, kind) == (
            "cluster-a",
            "team-a",
            "PodGroup",
        )
        return self.observations


def test_resolves_uid_verified_pod_group_from_top_owner_annotation(mocker):
    custom_api = mocker.Mock()
    custom_api.get_namespaced_custom_object.return_value = {
        "apiVersion": "scheduling.run.ai/v2alpha2",
        "kind": "PodGroup",
        "metadata": {
            "namespace": "team-a",
            "name": "pg-a",
            "uid": "pg-uid",
            "annotations": {
                "kai.scheduler/top-owner-metadata": "kind: Job\nname: job-a\nuid: job-uid\n",
            },
        },
    }

    assert KaiResourceResolver(custom_api).resolve(_pod(), _owner()) == (
        ResourceIdentity(
            "scheduling.run.ai/v2alpha2",
            "PodGroup",
            "team-a",
            "pg-a",
            "pg-uid",
        ),
    )


def test_rejects_recreated_or_unrelated_pod_group(mocker):
    custom_api = mocker.Mock()
    custom_api.get_namespaced_custom_object.return_value = {
        "metadata": {
            "namespace": "team-a",
            "name": "pg-a",
            "uid": "pg-uid",
            "ownerReferences": [
                {
                    "kind": "Job",
                    "name": "job-a",
                    "uid": "other-job-uid",
                    "controller": True,
                }
            ],
        }
    }

    with pytest.raises(UnsupportedOwner, match="does not belong"):
        KaiResourceResolver(custom_api).resolve(_pod(), _owner())


def test_resolves_deleted_pod_group_from_uid_verified_journal(mocker):
    custom_api = mocker.Mock()
    observation = ResourceObservation(
        cluster_id="cluster-a",
        api_version="scheduling.run.ai/v2alpha2",
        kind="PodGroup",
        namespace="team-a",
        name="pg-a",
        uid="original-pg-uid",
        resource_version="12",
        event_type="DELETED",
        observed_at="2026-10-14T00:00:00Z",
        selected_metadata={
            "annotations": {
                "kai.scheduler/top-owner-metadata": (
                    "kind: Job\nname: job-a\nuid: job-uid\n"
                )
            }
        },
    )

    result = KaiResourceResolver(
        custom_api,
        journal=_Journal([observation]),
        cluster_id="cluster-a",
    ).resolve(_pod(), _owner())

    assert result == (
        ResourceIdentity(
            "scheduling.run.ai/v2alpha2",
            "PodGroup",
            "team-a",
            "pg-a",
            "original-pg-uid",
        ),
    )
    custom_api.get_namespaced_custom_object.assert_not_called()
