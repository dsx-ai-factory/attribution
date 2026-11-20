# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray transport and actor ranks must not acquire scheduler node ownership."""

import pytest

from attribution_lib.exceptions import UnknownNodeError
from attribution_lib.utils import applog_extract_node_rank, hostname_mapping, ray_log_identity


@pytest.mark.parametrize("transport", ["", "0: ", "[rank0]: ", "[default0]: "])
@pytest.mark.parametrize("address", ["192.0.2.7", "2001:db8::7"])
def test_ray_worker_mapping_overrides_transport_and_actor_ranks(transport, address, simple_job_factory):
    """Resolve the emitting actor without interpreting either rank namespace.

    Args:
        transport: Optional label belonging to the collecting process.
        address: Synthetic IPv4 or IPv6 address of the actor.
        simple_job_factory: Factory for a synthetic scheduler allocation.

    Returns:
        None.
    """
    line = f"{transport}(TrainWorker[rank=27] pid=123, ip={address}) [rank27]: CUDA error: illegal memory access"
    job = simple_job_factory(nodes=["head", "worker"], ranks_per_node=1)
    with hostname_mapping({address: "worker"}):
        assert applog_extract_node_rank(line) == ("worker", None)
        assert job.applog_extract_node(line) == "worker"


@pytest.mark.parametrize(
    "envelope",
    [
        "(TrainWorker[rank=27] pid=123)",
        "(TrainWorker pid=123, ip=invalid)",
        "(TrainWorker pid=123, ip=192.0.2.7, ip=192.0.2.8)",
    ],
)
def test_unusable_ray_identity_never_becomes_rank_zero(envelope):
    """Keep a recognized but unusable actor identity explicitly unknown.

    Args:
        envelope: A Ray envelope without one valid worker address.

    Returns:
        None.
    """
    assert ray_log_identity(envelope) is not None
    assert applog_extract_node_rank(f"0: {envelope} CUDA error: illegal memory access") == (None, None)


def test_unmapped_ray_address_cannot_fall_back_to_head_rank(simple_job_factory):
    """Reject an unmapped worker address against the head's allocation.

    Args:
        simple_job_factory: Factory for a synthetic one-node head step.

    Returns:
        None.
    """
    line = "0: (TrainWorker[rank=27] pid=123, ip=192.0.2.7) CUDA error: illegal memory access"
    with hostname_mapping({}):
        assert applog_extract_node_rank(line) == ("192.0.2.7", None)
        with pytest.raises(UnknownNodeError):
            simple_job_factory(nodes=["head"]).applog_extract_node(line)


@pytest.mark.parametrize(
    "body",
    ["other:123:456 [0] NCCL WARN error", "RuntimeError: Rank 27, node other, device 0: error"],
)
def test_conflicting_ray_and_body_hostnames_remain_unknown(body):
    """Do not choose one of two conflicting explicit emitting identities.

    Args:
        body: A payload naming a different emitting host.

    Returns:
        None.
    """
    with hostname_mapping({"192.0.2.7": "worker"}):
        assert applog_extract_node_rank(f"(Worker pid=123, ip=192.0.2.7) {body}") == (None, None)


def test_ray_envelope_mentioned_in_payload_does_not_override_emitter():
    """Only a leading forwarding envelope establishes the log emitter.

    Returns:
        None.
    """
    line = "head: failed to contact (Worker pid=123, ip=192.0.2.7)"
    assert ray_log_identity(line) is None
    assert applog_extract_node_rank(line) == ("head", None)
