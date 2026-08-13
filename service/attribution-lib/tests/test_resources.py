# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json

import pytest
from pydantic import BaseModel, ValidationError

from attribution_lib.resources import AllJobResources, Node, Replica, Resource, ResourceSet


def test_node():
    host_a = Node(hostname="host_a")
    host_a2 = Node(hostname="host_a")

    assert host_a == host_a2

    host_b = Node(hostname="host_b")

    assert host_a != host_b

    with pytest.raises(ValidationError):
        _ = Node(hostname=12)  # type: ignore

    with pytest.raises(ValidationError):
        host_a.hostname = "host_b"


def test_base_deserialize_node():
    host_a_json = {
        "kind": "node",
        "hostname": "host_a",
    }

    host_a = Resource.validate_python(host_a_json)
    assert host_a == Node(hostname="host_a")


def test_base_deserialize_replica():
    pod_a_json = {
        "kind": "replica",
        "name": "pod_a",
    }

    pod_a = Resource.validate_python(pod_a_json)
    assert pod_a == Replica(name="pod_a")


def test_replica():
    pod_a = Replica(name="pod_a")
    pod_a2 = Replica(name="pod_a")

    assert pod_a == pod_a2

    pod_b = Replica(name="pod_b")

    assert pod_a != pod_b

    with pytest.raises(ValidationError):
        _ = Replica(name=12)  # type: ignore

    with pytest.raises(ValidationError):
        pod_a.name = "host_b"


def test_universal_set():
    host_a = Node(hostname="host_a")
    host_b = Node(hostname="host_b")
    host_c = Node(hostname="host_c")

    rset = ResourceSet([host_a, host_b, host_a])

    assert host_a in rset
    assert host_b in rset
    assert host_c not in rset

    assert host_a in AllJobResources
    assert host_b in AllJobResources
    assert host_c in AllJobResources

    assert rset.union(AllJobResources) == AllJobResources
    assert rset.intersection(AllJobResources) == rset


def test_resource_set():
    host_a = Node(hostname="host_a")
    host_a2 = Node(hostname="host_a")

    assert host_a == host_a2

    host_b = Node(hostname="host_b")

    assert host_a != host_b

    host_c = Node(hostname="host_c")

    rset = ResourceSet([host_a, host_b, host_a])

    assert host_a in rset
    assert host_b in rset
    assert host_c not in rset


def test_serialization():
    class TestModel(BaseModel):
        resources: ResourceSet

    host_a = Node(hostname="host_a")
    host_b = Node(hostname="host_b")
    pod_a = Replica(name="pod_a")

    model = TestModel(resources=ResourceSet([host_a, host_b, pod_a]))
    assert model.model_dump_json() == json.dumps(
        {
            "resources": [
                {"kind": "node", "key": "host_a", "hostname": "host_a"},
                {"kind": "node", "key": "host_b", "hostname": "host_b"},
                {"kind": "replica", "key": "pod_a", "name": "pod_a"},
            ]
        },
        separators=(",", ":"),
    )

    model = TestModel(resources=AllJobResources)
    assert model.model_dump_json() == json.dumps({"resources": "AllJobResources"}, separators=(",", ":"))


def test_deserialization():
    class TestModel(BaseModel):
        resources: ResourceSet

    str = json.dumps(
        {
            "resources": [
                {"kind": "node", "key": "host_a", "hostname": "host_a"},
                {"kind": "node", "key": "host_b", "hostname": "host_b"},
                {"kind": "replica", "key": "pod_a", "name": "pod_a"},
            ]
        },
        separators=(",", ":"),
    )

    model = TestModel.model_validate_json(str)

    host_a = Node(hostname="host_a")
    host_b = Node(hostname="host_b")
    pod_a = Replica(name="pod_a")

    assert host_a in model.resources
    assert host_b in model.resources
    assert pod_a in model.resources

    str = json.dumps({"resources": "AllJobResources"}, separators=(",", ":"))

    model = TestModel.model_validate_json(str)

    assert model.resources == AllJobResources
    assert Node(hostname="host_x") in model.resources
