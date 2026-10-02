# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timedelta, timezone

import pytest

from attribution_blueprint.connectors.batch import (
    supports_batch_collection,
    time_bounded_batches,
)
from attribution_blueprint.types import Source

_T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)


def _ctx(ctx_id: str, *, end_minutes: int, duration_minutes: int = 30) -> dict:
    end = _T0 + timedelta(minutes=end_minutes)
    return {
        "id": ctx_id,
        "nodes": [f"node-{ctx_id}"],
        "start_time": end - timedelta(minutes=duration_minutes),
        "end_time": end,
    }


def test_time_bounded_batches_groups_by_end_time_within_window():
    contexts = [
        _ctx("late", end_minutes=300),
        _ctx("a", end_minutes=0),
        _ctx("b", end_minutes=90),
        _ctx("c", end_minutes=120),
        _ctx("d", end_minutes=121),
    ]
    batches = time_bounded_batches(contexts, window=timedelta(hours=2))
    assert [[ctx["id"] for ctx in batch] for batch in batches] == [["a", "b", "c"], ["d"], ["late"]]


def test_time_bounded_batches_caps_batch_size():
    contexts = [_ctx(str(i), end_minutes=i) for i in range(5)]
    batches = time_bounded_batches(contexts, window=timedelta(hours=2), max_contexts=2)
    assert [[ctx["id"] for ctx in batch] for batch in batches] == [["0", "1"], ["2", "3"], ["4"]]


def test_time_bounded_batches_without_bounds_returns_single_ordered_batch():
    contexts = [_ctx("b", end_minutes=500), _ctx("a", end_minutes=1)]
    assert [[ctx["id"] for ctx in batch] for batch in time_bounded_batches(contexts, window=None)] == [["a", "b"]]


def test_time_bounded_batches_empty_input():
    assert time_bounded_batches([], window=timedelta(hours=2)) == []


def test_time_bounded_batches_orders_ties_by_start_then_id():
    first = _ctx("z", end_minutes=10, duration_minutes=5)
    second = _ctx("y", end_minutes=10, duration_minutes=5)
    third = _ctx("x", end_minutes=10, duration_minutes=60)
    batches = time_bounded_batches([first, second, third], window=timedelta(hours=1))
    assert [ctx["id"] for ctx in batches[0]] == ["x", "y", "z"]


def test_time_bounded_batches_rejects_naive_datetimes():
    ctx = _ctx("a", end_minutes=0)
    ctx["end_time"] = ctx["end_time"].replace(tzinfo=None)
    with pytest.raises(ValueError, match="end_time"):
        time_bounded_batches([ctx], window=timedelta(hours=1))


class _Batching:
    source = Source.SYSLOG

    def collect(self, context):
        return []

    def collect_batch(self, contexts):
        return {}


class _PerContext:
    source = Source.SYSLOG

    def collect(self, context):
        return []


def test_supports_batch_collection_detects_collect_batch():
    assert supports_batch_collection(_Batching()) is True
    assert supports_batch_collection(_PerContext()) is False
