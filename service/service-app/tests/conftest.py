# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.attributor import (
    AttributorMetadata,
    WorkloadTypeEnum,
)
from app.factory import create_app
from app.v1.models import AttributorInfoV1, AttributorWorkloadInfoV1
from app.v2.models import AttributorInfoV2, AttributorWorkloadInfoV2


@pytest.fixture
def app():
    yield create_app()


@pytest.fixture
def client(app):
    yield TestClient(app, headers={"Content-Type": "application/json"})


@pytest.fixture(params=["v1", "v2"])
def version(request):
    """Parameterizes a test across API versions. Used for tests that exercise
    version-invariant behavior like user journeys or basic endpoint behavior."""
    return request.param


def _default_metadata() -> AttributorMetadata:
    return AttributorMetadata(
        cluster="test-cluster-a",
        agent="test",
        tenant="testing",
        ruleset="default",
    )


def make_up_attributor_info(version: str = "v1", **workload_overrides) -> AttributorInfoV1 | AttributorInfoV2:
    """Build a default ``AttributorInfo`` for the requested wire version."""
    if version == "v1":
        return _make_up_v1(workload_overrides)
    if version == "v2":
        return _make_up_v2(workload_overrides)
    raise ValueError(f"Unsupported version: {version}")


def _make_up_v1(workload_overrides: dict) -> AttributorInfoV1:
    workload = dict(
        id="120123",
        srun_cmd="",
        type=WorkloadTypeEnum.SLURM,
        cancelled_by=None,
        last_tflops_time=None,
        nranks=10,
        ranks_per_node=8,
        job_start_time=datetime.now().astimezone() - timedelta(minutes=10),
        job_end_time=datetime.now().astimezone() - timedelta(minutes=5),
        exit_code_signal=1,
        nodes=[f"node{s}" for s in range(10)],
        status="CANCELED",
        username="test-agent",
        name="test_job",
        framework="megatron",
    )
    workload.update(workload_overrides)
    return AttributorInfoV1(
        workload=AttributorWorkloadInfoV1(**workload),
        metadata=_default_metadata(),
    )


def _make_up_v2(workload_overrides: dict) -> AttributorInfoV2:
    workload = dict(
        id="120123",
        srun_cmd="",
        type="slurm",
        cancelled_by=None,
        last_tflops_time=None,
        job_start_time=datetime.now().astimezone() - timedelta(minutes=10),
        job_end_time=datetime.now().astimezone() - timedelta(minutes=5),
        timezone="America/Los_Angeles",
        exit_code_signal=1,
        nodes={f"node{s}": list(range(s * 8, (s + 1) * 8)) for s in range(10)},
        status="CANCELED",
        username="test-agent",
        name="test_job",
        framework="megatron",
    )
    workload.update(workload_overrides)
    return AttributorInfoV2(
        workload=AttributorWorkloadInfoV2(**workload),
        metadata=_default_metadata(),
    )


def post_attributor(client, attributor_info):
    """POST a new attributor and return its ID. Version is inferred from the info type."""
    version = "v2" if isinstance(attributor_info, AttributorInfoV2) else "v1"
    response = client.post(f"/{version}/attributor", content=attributor_info.model_dump_json())
    assert response.status_code == 201
    return response.json()["attributor_id"]
