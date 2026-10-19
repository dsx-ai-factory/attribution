# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest

from attribution_blueprint.kubernetes.daemon.models import ResourceIdentity
from attribution_blueprint.kubernetes.daemon.resolvers import (
    KubernetesOwnershipResolver,
    StaleOwnerReference,
    UnsupportedOwner,
    normalize_resource,
)


class _Resource:
    def __init__(self, resource):
        self.resource = resource

    def to_dict(self):
        return self.resource


class _BatchApi:
    def __init__(self, job):
        self.job = job

    def read_namespaced_job(self, name, namespace):
        assert (name, namespace) == ("trainer", "team-a")
        return _Resource(self.job)


class _CustomApi:
    def __init__(self, jobset):
        self.jobset = jobset

    def get_namespaced_custom_object(self, group, version, namespace, plural, name):
        assert (group, version, namespace, plural, name) == (
            "jobset.x-k8s.io",
            "v1alpha2",
            "team-a",
            "jobsets",
            "trainer-set",
        )
        return self.jobset


def _pod() -> dict:
    return {
        "metadata": {
            "namespace": "team-a",
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": "trainer",
                    "uid": "job-uid-1",
                    "controller": True,
                }
            ],
        }
    }


def test_resolves_uid_verified_job_owner():
    job = {
        "metadata": {
            "name": "trainer",
            "namespace": "team-a",
            "uid": "job-uid-1",
        }
    }
    resolver = KubernetesOwnershipResolver(_BatchApi(job), custom_api=None)

    assert resolver.resolve(_pod()) == ResourceIdentity(
        "batch/v1",
        "Job",
        "team-a",
        "trainer",
        "job-uid-1",
    )


@pytest.mark.parametrize(
    "controller_field",
    [{}, {"controller": None}, {"controller": False}],
)
def test_rejects_non_controller_job_owner_references(controller_field):
    pod = _pod()
    owner = pod["metadata"]["ownerReferences"][0]
    owner.pop("controller")
    owner.update(controller_field)
    resolver = KubernetesOwnershipResolver(object(), custom_api=None)

    with pytest.raises(UnsupportedOwner, match="not controlled"):
        resolver.resolve(pod)


def test_resolves_uid_verified_jobset_owner():
    job = {
        "metadata": {
            "name": "trainer",
            "namespace": "team-a",
            "uid": "job-uid-1",
            "owner_references": [
                {
                    "api_version": "jobset.x-k8s.io/v1alpha2",
                    "kind": "JobSet",
                    "name": "trainer-set",
                    "uid": "jobset-uid-1",
                    "controller": True,
                }
            ],
        }
    }
    jobset = {
        "metadata": {
            "name": "trainer-set",
            "namespace": "team-a",
            "uid": "jobset-uid-1",
        }
    }
    resolver = KubernetesOwnershipResolver(_BatchApi(job), _CustomApi(jobset))

    assert resolver.resolve(_pod()) == ResourceIdentity(
        "jobset.x-k8s.io/v1alpha2",
        "JobSet",
        "team-a",
        "trainer-set",
        "jobset-uid-1",
    )


def test_rejects_reincarnated_job():
    job = {
        "metadata": {
            "name": "trainer",
            "namespace": "team-a",
            "uid": "replacement-job-uid",
        }
    }
    resolver = KubernetesOwnershipResolver(_BatchApi(job), custom_api=None)

    with pytest.raises(StaleOwnerReference, match="uid mismatch"):
        resolver.resolve(_pod())


def test_normalizes_sdk_resource_fields():
    observed = datetime(2026, 10, 14, 19, 0, tzinfo=timezone.utc)
    resource = normalize_resource(
        {
            "api_version": "v1",
            "metadata": {
                "name": "trainer-0",
                "namespace": "team-a",
                "uid": "pod-uid-1",
                "resource_version": "42",
                "owner_references": [
                    {
                        "api_version": "batch/v1",
                        "kind": "Job",
                        "name": "trainer",
                        "uid": "job-uid-1",
                        "controller": True,
                    }
                ],
            },
            "spec": {"node_name": "node-a"},
            "status": {
                "observed": observed,
                "conditions": [{"type": "Ready", "last_transition_time": observed}],
                "container_statuses": [
                    {
                        "name": "trainer",
                        "state": {"terminated": {"exit_code": 17}},
                    }
                ],
            },
        },
        kind="Pod",
    )

    assert resource["apiVersion"] == "v1"
    assert resource["kind"] == "Pod"
    assert resource["metadata"]["resourceVersion"] == "42"
    assert resource["metadata"]["ownerReferences"][0]["apiVersion"] == "batch/v1"
    assert resource["spec"]["nodeName"] == "node-a"
    assert resource["status"]["observed"] == "2026-10-14T19:00:00Z"
    assert resource["status"]["conditions"][0]["lastTransitionTime"] == "2026-10-14T19:00:00Z"
    assert resource["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] == 17
