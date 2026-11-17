# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_blueprint.connectors.execution_contexts.kai.helpers import resolve_owner
from attribution_blueprint.connectors.execution_contexts.kai.resource_resolver import KaiResourceResolver
from attribution_blueprint.kubernetes.daemon.models import (
    ObservationEventType,
    OwnerReference,
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


def _jobset_owner():
    return ResourceIdentity(
        "jobset.x-k8s.io/v1alpha2",
        "JobSet",
        "team-a",
        "training",
        "jobset-uid",
    )


def _partial_jobset_metadata():
    return {
        "annotations": {
            "kai.scheduler/top-owner-metadata": "kind: JobSet\nuid: jobset-uid\n",
        },
        "ownerReferences": [
            {
                "kind": "Job",
                "name": "child-job",
                "uid": "child-job-uid",
                "controller": True,
            }
        ],
    }


def _retained_pod_group(
    uid: str,
    *,
    owner_uid: str = "job-uid",
    event_type: ObservationEventType = "DELETED",
) -> ResourceObservation:
    return ResourceObservation(
        cluster_id="cluster-a",
        api_version="scheduling.run.ai/v2alpha2",
        kind="PodGroup",
        namespace="team-a",
        name="pg-a",
        uid=uid,
        resource_version=uid,
        event_type=event_type,
        observed_at="2026-10-14T00:00:00Z",
        selected_metadata={
            "annotations": {
                "kai.scheduler/top-owner-metadata": (
                    f"kind: Job\nname: job-a\nuid: {owner_uid}\n"
                )
            }
        },
    )


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


def test_partial_top_owner_identity_matches_live_pod_group(mocker):
    custom_api = mocker.Mock()
    custom_api.get_namespaced_custom_object.return_value = {
        "apiVersion": "scheduling.run.ai/v2alpha2",
        "kind": "PodGroup",
        "metadata": {
            **_partial_jobset_metadata(),
            "namespace": "team-a",
            "name": "pg-a",
            "uid": "pg-uid",
        },
    }

    assert KaiResourceResolver(custom_api).resolve(
        _pod(),
        _jobset_owner(),
    ) == (
        ResourceIdentity(
            "scheduling.run.ai/v2alpha2",
            "PodGroup",
            "team-a",
            "pg-a",
            "pg-uid",
        ),
    )


def test_partial_top_owner_identity_matches_retained_pod_group(mocker):
    custom_api = mocker.Mock()
    observation = ResourceObservation(
        cluster_id="cluster-a",
        api_version="scheduling.run.ai/v2alpha2",
        kind="PodGroup",
        namespace="team-a",
        name="pg-a",
        uid="pg-uid",
        resource_version="1",
        event_type="MODIFIED",
        observed_at="2026-10-14T00:00:00Z",
        selected_metadata={
            "annotations": _partial_jobset_metadata()["annotations"],
        },
        owner_references=(
            OwnerReference(
                api_version="batch/v1",
                kind="Job",
                name="child-job",
                uid="child-job-uid",
                controller=True,
            ),
        ),
    )

    assert KaiResourceResolver(
        custom_api,
        journal=_Journal([observation]),
        cluster_id="cluster-a",
    ).resolve(_pod(), _jobset_owner()) == (
        ResourceIdentity(
            "scheduling.run.ai/v2alpha2",
            "PodGroup",
            "team-a",
            "pg-a",
            "pg-uid",
        ),
    )
    custom_api.get_namespaced_custom_object.assert_not_called()


def test_complete_owner_resolution_still_requires_top_owner_name():
    metadata = _partial_jobset_metadata()

    assert resolve_owner(metadata) == metadata["ownerReferences"][0]


def test_resolves_deleted_pod_group_from_uid_verified_journal(mocker):
    custom_api = mocker.Mock()
    observation = _retained_pod_group("original-pg-uid")

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


def test_unrelated_retained_pod_group_falls_back_to_live_lookup(mocker):
    custom_api = mocker.Mock()
    custom_api.get_namespaced_custom_object.return_value = {
        "apiVersion": "scheduling.run.ai/v2alpha2",
        "kind": "PodGroup",
        "metadata": {
            "namespace": "team-a",
            "name": "pg-a",
            "uid": "live-pg-uid",
            "annotations": {
                "kai.scheduler/top-owner-metadata": (
                    "kind: Job\nname: job-a\nuid: job-uid\n"
                )
            },
        },
    }
    resolver = KaiResourceResolver(
        custom_api,
        journal=_Journal(
            [_retained_pod_group("unrelated-pg-uid", owner_uid="other-job-uid")]
        ),
        cluster_id="cluster-a",
    )

    assert resolver.resolve(_pod(), _owner()) == (
        ResourceIdentity(
            "scheduling.run.ai/v2alpha2",
            "PodGroup",
            "team-a",
            "pg-a",
            "live-pg-uid",
        ),
    )
    custom_api.get_namespaced_custom_object.assert_called_once()


def test_ambiguous_retained_pod_groups_preserve_uncertainty(mocker):
    custom_api = mocker.Mock()
    resolver = KaiResourceResolver(
        custom_api,
        journal=_Journal(
            [
                _retained_pod_group("deleted-pg-uid"),
                _retained_pod_group("live-pg-uid", event_type="MODIFIED"),
            ]
        ),
        cluster_id="cluster-a",
    )

    with pytest.raises(UnsupportedOwner, match="multiple retained"):
        resolver.resolve(_pod(), _owner())

    custom_api.get_namespaced_custom_object.assert_not_called()
