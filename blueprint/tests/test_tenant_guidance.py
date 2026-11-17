# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content regressions, not substitutes for Product or engineering approval."""

import re
from typing import Any

import pytest

from attribution_blueprint.webui.diagnose import _load_catalog


@pytest.fixture(scope="module")
def catalog() -> dict[str, Any]:
    _load_catalog.cache_clear()
    return _load_catalog()


@pytest.mark.parametrize(
    "entry_id",
    [
        "cudaerrormemoryallocation",
        "cudaerror-out-of-memory",
        "system-out-of-memory",
        "missing-key-in-checkpoint",
        "gloo-connectfullmesh-failed",
        "python-exception",
        "python-bug",
        "jsondecodeerror",
        "disk-quota-exceeded",
        "no-such-file-or-directory",
        "pyxis-enroot-mount-failure",
        "cancelled-by-timeout",
    ],
)
def test_actions_are_numbered_and_incomplete_evidence_does_not_block_help(
    catalog: dict[str, Any], entry_id: str
) -> None:
    tenant = catalog[entry_id]["personas"]["tenant_sre"]
    steps = re.findall(r"^(\d+)\. ", tenant["action"], re.MULTILINE)
    assert len(steps) >= 2
    assert steps == [str(i) for i in range(1, len(steps) + 1)]
    support = tenant["support_details"]
    assert "NVIDIA Attribution diagnosis (verdict, symptom chain, and source logs) when available" in support
    assert "send the evidence available to you and say what is missing" in support
    assert "do not delay the request" in support


@pytest.mark.parametrize("entry_id", ["cudaerrormemoryallocation", "cudaerror-out-of-memory"])
def test_gpu_memory_advice_identifies_framework_and_measurement_limits(catalog: dict[str, Any], entry_id: str) -> None:
    action = catalog[entry_id]["personas"]["tenant_sre"]["action"]
    for phrase in ("For PyTorch", "For other frameworks", "after the process exits", "live tensor"):
        assert phrase in action


def test_host_memory_is_not_confused_with_gpu_or_whole_job_memory(catalog: dict[str, Any]) -> None:
    entry = catalog["system-out-of-memory"]
    action = entry["personas"]["tenant_sre"]["action"]
    for phrase in ("exit code 137 alone", "On Slurm", "On Kubernetes", "not the combined memory of the whole job"):
        assert phrase in action
    assert "do not treat GPU memory as host RAM" in action
    assert "SIGKILL alone does not prove" in entry["technical_details"]


def test_checkpoint_recovery_requires_model_validation(catalog: dict[str, Any]) -> None:
    action = catalog["missing-key-in-checkpoint"]["personas"]["tenant_sre"]["action"]
    assert "`strict=False` only for an intentional partial load" in action
    assert "model validation before resuming" in action
    assert "omitted keys are unexplained" in action


def test_gloo_retry_preserves_backend_scope_and_has_a_stop_condition(catalog: dict[str, Any]) -> None:
    action = catalog["gloo-connectfullmesh-failed"]["personas"]["tenant_sre"]["action"]
    for phrase in (
        "`GLOO_SOCKET_IFNAME`",
        "`NCCL_SOCKET_IFNAME` controls NCCL, not Gloo",
        "Retry once",
        "all ranks initialize",
    ):
        assert phrase in action


@pytest.mark.parametrize(
    "entry_id",
    [
        "ibv-wc-success",
        "rdma-cm-event-addr-resolved",
        "rdma-cm-event-route-resolved",
        "rdma-cm-event-established",
        "cudasuccess",
        "cudaerrornotready",
    ],
)
def test_normal_statuses_do_not_request_automatic_recovery_or_tickets(catalog: dict[str, Any], entry_id: str) -> None:
    tenant = catalog[entry_id]["personas"]["tenant_sre"]
    assert "No corrective action is required for this symptom." in tenant["action"]
    assert "No support report is needed" in tenant["support_details"]
    for forbidden in ("sudo", "nvidia-smi --gpu-reset", "drain the node"):
        assert forbidden not in tenant["action"]


def test_pending_cuda_work_is_not_an_instruction_to_wait_forever(catalog: dict[str, Any]) -> None:
    action = catalog["cudaerrornotready"]["personas"]["tenant_sre"]["action"]
    assert "progress timeout" in action
    assert "stop extending the wait" in action
    assert "not confirmation" in catalog["cudasuccess"]["technical_details"]


def test_bad_token_describes_the_recorded_nonfinite_result(catalog: dict[str, Any]) -> None:
    entry = catalog["bad-token"]
    assert "Infinity is not a finite numerical result" in entry["technical_details"]
    patterns = [signal["pattern"] for signal in entry["signals"]]
    assert any(re.search(pattern, "Unexpected result inf") for pattern in patterns)
    assert not any(re.search(pattern, "tokenizer vocabulary mismatch") for pattern in patterns)


def test_kernel_memory_warning_is_not_a_cuda_kernel_diagnosis(catalog: dict[str, Any]) -> None:
    entry = catalog["kernel-invalid-memory-access"]
    action = entry["personas"]["tenant_sre"]["action"]
    assert "host-kernel" in entry["technical_details"]
    assert "cluster administrator or cloud provider" in action
    assert "compute-sanitizer" not in action


def test_ib_flush_status_is_not_described_as_a_remote_invalid_request(catalog: dict[str, Any]) -> None:
    entry = catalog["ib-error-5"]
    assert "WR_FLUSH_ERR" in entry["personas"]["tenant_sre"]["action"]
    assert "REM_INV_REQ_ERR" not in entry["summary"]
    assert "libibverbs" in entry["personas"]["tenant_sre"]["action"]


def test_rdma_invalid_argument_does_not_assume_a_gid_cause(catalog: dict[str, Any]) -> None:
    entry = catalog["call-to-ibv-modify-qp-failed-eno-22-invalid-argument"]
    assert "not the only cause" in entry["technical_details"]
    assert entry["likely_owner"] == "user"
