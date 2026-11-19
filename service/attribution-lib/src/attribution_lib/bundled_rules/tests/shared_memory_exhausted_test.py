# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribute a full /dev/shm to the job, not to the SIGBUS it causes or to a teardown peer."""

import pytest

from attribution_lib import core
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.observation.inbound import Source as ObservationSource
from test_support.helpers import line_entries

_WORKER_BUS_ERROR = (
    "0: ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)."
)
_TORCH_SHM_FULL = "0: RuntimeError: unable to write to file </torch_2914_1838475476_0>: No space left on device (28)"
_NCCL_SHM_FULL = (
    "1: NCCL WARN Error: failed to extend /dev/shm/nccl-rWUhkP to 9637892 bytes, error: No space left on device (28)"
)
_PEER_CLOSED = "0: NCCL WARN Connection closed by remote peer node02<1234>"
_PEER_RESET = "1: NCCL WARN socketProgressOpt: Call to recv from 127.0.0.1<28601> failed : Connection reset by peer"
_SHM_PATH_AFTER_ENOSPC = "0: OSError: [Errno 28] No space left on device: '/dev/shm/torch_1'"
_IB_RETRY_EXCEEDED = (
    "0: NET/IB: Got completion from peer 192.0.2.2<1234> with status=IBV_WC_RETRY_EXC_ERR(12) "
    "opcode=UNKNOWN_OPCODE(65386) vendor_err=129  localGid ::ffff:192.0.2.1 remoteGids::ffff:192.0.2.2 hca mlx5_0"
)
_HW_BUS_ERROR = "1: Caught signal 7 (Bus error: nonexistent physical address)"


@pytest.fixture
def attribute_logs(attribution_test_builder):
    """Attribute applog lines for a two-node job.

    Args:
        attribution_test_builder: Fixture that extracts events from log entries.

    Returns:
        A function that takes log lines and returns the attribution result.
    """

    def _attribute(lines):
        job = default_job_info(nodes=["node01", "node02"], exit_code_signal=None)
        extracted = attribution_test_builder(
            job_info=job, data=line_entries(lines), source=ObservationSource.APPLOG, insert_tflops_line=False
        )
        bad_nodes = core.calculate_bad_nodes(
            event_stream=extracted.event_stream, job_info=job, applog_in=extracted.observation_source
        )
        return core.attribute(
            event_stream=extracted.event_stream,
            job_info=job,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
        )

    return _attribute


def test_full_shm_is_fail_user_and_drops_teardown_peer(attribute_logs):
    result = attribute_logs([_WORKER_BUS_ERROR, _TORCH_SHM_FULL, _NCCL_SHM_FULL, _PEER_CLOSED])

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == "Shared memory exhausted"
    assert "Connection closed" not in result.symptoms
    assert "bus error" not in result.symptoms


def test_full_shm_outranks_peer_reset(attribute_logs):
    result = attribute_logs([_WORKER_BUS_ERROR, _SHM_PATH_AFTER_ENOSPC, _PEER_RESET])

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == "Shared memory exhausted"


def test_full_shm_outranks_ib_retry_on_same_node(attribute_logs):
    result = attribute_logs([_TORCH_SHM_FULL, _IB_RETRY_EXCEEDED])

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == "Shared memory exhausted"


def test_worker_bus_error_without_full_shm_stays_multicategory(attribute_logs):
    result = attribute_logs([_WORKER_BUS_ERROR])

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail MultiCategory"
    assert outcome.root_cause_symptom == "DataLoader worker bus error"
    assert "bus error" not in result.symptoms


def test_full_shm_keeps_hardware_bus_error(attribute_logs):
    result = attribute_logs([_WORKER_BUS_ERROR, _TORCH_SHM_FULL, _HW_BUS_ERROR])

    assert "bus error" in result.symptoms
