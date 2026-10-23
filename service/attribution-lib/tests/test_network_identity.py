# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_lib.network_identity import network_source_identity
from attribution_lib.utils import hostname_mapping


@pytest.mark.parametrize(
    "prefix,mapping,expected,conflict",
    [
        ("[2026-01-01 00:00:00] NODE-A.example:12:34 [0] ", {}, "node-a", False),
        ("192: ", {"192.0.2.1": "Node-A.example"}, "node-a", False),
        ("", {}, None, False),
        ("[2026-01-01 00:00:00] ", {}, None, False),
        ("NODE-A:12:34 [0] ", {"192.0.2.1": "node-a.example"}, "node-a", False),
        ("NODE-A:12:34 [0] ", {"192.0.2.1": "node-b"}, "node-a", True),
        ("", {"192.0.2.1": "192.0.2.1"}, None, False),
        ("", {"192.0.2.2": "peer-only"}, None, False),
    ],
)
def test_identity_uses_independent_local_evidence(prefix, mapping, expected, conflict):
    with hostname_mapping(mapping):
        identity = network_source_identity(
            prefix + "NET/IB: Got completion from peer 192.0.2.2 localGid ::ffff:192.0.2.1"
        )
    assert identity.hostname == expected
    assert identity.conflict is conflict
    assert identity.local_ip == "192.0.2.1"
    if conflict is True:
        assert identity.log_hostname == "node-a"
        assert identity.mapped_hostname == "node-b"


def test_message_label_is_not_a_hostname_and_rank_zero_is_not_source_evidence():
    with hostname_mapping({"192.0.2.1": "node-a"}):
        line = "RuntimeError: NET/IB: Got completion from peer 192.0.2.2 localGid ::ffff:192.0.2.1"
        identity = network_source_identity(line, prefix_hostname="rank-zero-node")
    assert identity.log_hostname is None
    assert identity.hostname == "node-a"
    assert identity.conflict is False


def test_verified_hostname_prefix_remains_explicit_evidence():
    identity = network_source_identity("Node-A: NET/IB: message", prefix_hostname="node-a")
    assert identity.hostname == "node-a"


def test_explicit_nccl_hostname_does_not_require_a_gid_or_allocation():
    identity = network_source_identity("[2026-01-01 00:00:00] OUTSIDE.example:12:34 [0] NET/IB: message")
    assert identity.hostname == "outside"
    assert identity.local_ip is None
    assert identity.mapped_hostname is None
