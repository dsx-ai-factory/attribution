# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timezone

import pytest

from attribution_blueprint.connectors.execution_contexts.static import StaticExecutionContextAdapter


def _make_eca(**overrides):
    base = {
        "id": "sample-static-job",
        "start_time": datetime(2026, 5, 8, 12, 14, 30, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 8, 18, 1, 1, tzinfo=timezone.utc),
        "nodes": ["example-node-003", "example-node-005"],
        "state": "FAILED",
    }
    base.update(overrides)
    return StaticExecutionContextAdapter(**base)


def test_discover_by_ids_returns_single_context():
    eca = _make_eca(name="sample-training", user_name="alice", std_out="/path/to/logs/job.out")
    ctxs = eca.discover()
    assert len(ctxs) == 1
    ctx = ctxs[0]
    assert ctx["id"] == "sample-static-job"
    assert ctx["attempt_id"] == "0"
    assert ctx["workload_type"] == ""
    # `scheduler` is the BP-internal connector identity. Hardcoded "static"
    # regardless of what `workload_type` the operator passes.
    assert ctx["scheduler"] == "static"
    assert ctx["nodes"] == ["example-node-003", "example-node-005"]
    assert ctx["state"] == "FAILED"
    assert ctx["name"] == "sample-training"
    assert ctx["user_name"] == "alice"
    assert ctx["std_out"] == "/path/to/logs/job.out"


def test_discover_by_ids_validates_then_returns_single_context():
    eca = _make_eca()
    assert eca.discover_by_ids(["sample-static-job"]) == eca.discover()


def test_discover_by_ids_filters_contexts_file_mode():
    contexts = [
        {
            "id": "ctx-1",
            "attempt_id": "1",
            "workload_type": "",
            "scheduler": "static",
            "nodes": ["node-001"],
            "start_time": datetime(2026, 5, 8, 12, 14, 30, tzinfo=timezone.utc),
            "end_time": datetime(2026, 5, 8, 12, 15, 30, tzinfo=timezone.utc),
            "job_id": "ctx-1",
            "state": "FAILED",
        },
        {
            "id": "ctx-2",
            "attempt_id": "2",
            "workload_type": "",
            "scheduler": "static",
            "nodes": ["node-002"],
            "start_time": datetime(2026, 5, 8, 12, 16, 30, tzinfo=timezone.utc),
            "end_time": datetime(2026, 5, 8, 12, 17, 30, tzinfo=timezone.utc),
            "job_id": "ctx-2",
            "state": "FAILED",
        },
    ]
    eca = StaticExecutionContextAdapter(contexts=contexts)

    assert eca.discover_by_ids(["ctx-2"]) == [contexts[1]]


def test_discover_includes_configured_attempt_id():
    eca = _make_eca(attempt_id="1")
    ctx = eca.discover()[0]
    assert ctx["attempt_id"] == "1"


def test_discover_includes_configured_gpu_count():
    eca = _make_eca(gpu_count=6)
    ctx = eca.discover()[0]
    assert ctx["gpu_count"] == 6


def test_discover_includes_configured_timezone():
    eca = _make_eca(timezone="Asia/Kolkata")

    assert eca.discover()[0]["timezone"] == "Asia/Kolkata"


def test_discover_by_ids_includes_configured_workload_type():
    eca = _make_eca(workload_type="runai")
    ctx = eca.discover()[0]
    assert ctx["workload_type"] == "runai"
    # `scheduler` remains "static" even when the operator spoofs the wire
    # workload_type, connector identity is structural, not configurable.
    assert ctx["scheduler"] == "static"


def test_discover_by_ids_rejects_id_mismatch():
    eca = _make_eca()
    with pytest.raises(ValueError, match="id mismatch"):
        eca.discover_by_ids(["different"])


def test_discover_by_ids_rejects_multiple():
    eca = _make_eca()
    with pytest.raises(ValueError, match="exactly one"):
        eca.discover_by_ids(["1", "2"])


def test_discover_by_time_range_not_supported():
    eca = _make_eca()
    with pytest.raises(NotImplementedError):
        eca.discover_by_time_range(
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 2, tzinfo=timezone.utc),
            nodelist=None,
        )


def test_constructor_rejects_empty_nodes():
    with pytest.raises(ValueError, match="nodes must be non-empty"):
        StaticExecutionContextAdapter(
            id="1",
            start_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 2, tzinfo=timezone.utc),
            nodes=[],
        )


def test_constructor_rejects_empty_id():
    with pytest.raises(ValueError, match="id is required"):
        StaticExecutionContextAdapter(
            id="",
            start_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 2, tzinfo=timezone.utc),
            nodes=["n1"],
        )


def test_constructor_rejects_negative_gpu_count():
    with pytest.raises(ValueError, match="gpu_count"):
        StaticExecutionContextAdapter(
            id="1",
            start_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 2, tzinfo=timezone.utc),
            nodes=["n1"],
            gpu_count=-1,
        )


def test_constructor_rejects_invalid_timezone():
    with pytest.raises(ValueError, match="valid IANA timezone key"):
        _make_eca(timezone="Not/A/Zone")


def test_constructor_rejects_naive_start_time():
    with pytest.raises(ValueError, match="start_time must be tz-aware"):
        StaticExecutionContextAdapter(
            id="1",
            start_time=datetime(2026, 5, 1),  # naive
            end_time=datetime(2026, 5, 2, tzinfo=timezone.utc),
            nodes=["n1"],
        )


def test_constructor_rejects_naive_end_time():
    with pytest.raises(ValueError, match="end_time must be tz-aware"):
        StaticExecutionContextAdapter(
            id="1",
            start_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 2),  # naive
            nodes=["n1"],
        )


def test_workload_framework_propagates_to_emitted_context():
    """Operator-supplied workload framework reaches the ctx so it lands in workload.framework
    on the wire (the wire field name stays `framework` for service compatibility)."""
    eca = _make_eca(workload_framework="megatron")
    ctx = eca.discover()[0]
    assert ctx["workload_framework"] == "megatron"


def test_workload_framework_absent_by_default():
    """Without explicit input the workload_framework key is absent (not empty-string)."""
    ctx = _make_eca().discover()[0]
    assert "workload_framework" not in ctx
