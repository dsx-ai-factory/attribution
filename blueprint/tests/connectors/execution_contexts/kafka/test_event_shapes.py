# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flat and per-node allocation events on one disposable Kafka topic; no Service required."""

from uuid import uuid4

import pytest

from attribution_blueprint.connectors.execution_contexts.kafka import _build

from .test_connector import EVENT_TYPE, EVENT_TYPE_V2, _allocations_event
from .test_delivery import _assert_drained
from .test_streaming_delivery import _committed, _next_delivery

pytestmark = pytest.mark.integration

ALLOCATED_NODES = ["cpu-n07", "gpu-n01", "gpu-n02"]
ALLOCATED_POD_NODES = {
    "job-1234-head-0": "cpu-n07",
    "job-1234-worker-0": "gpu-n01",
    "job-1234-worker-1": "gpu-n01",
    "job-1234-worker-2": "gpu-n02",
}


@pytest.fixture
def migration_spec(kafka_spec):
    """Accept the flat and allocation event types together, as during a producer migration."""
    return {**kafka_spec, "event_types": [EVENT_TYPE, EVENT_TYPE_V2]}


@pytest.fixture
def allocated_event():
    """Give the allocation event a unique job so topics never share attempts."""
    event = _allocations_event()
    event["id"] = str(uuid4())
    event["data"]["id"] = event["data"]["job_id"] = f"job-{uuid4().hex}"
    event["subject"] = f"jobs/{event['data']['id']}"
    return event


def _assert_flat(context, event):
    assert context["event_id"] == event["id"]
    assert context["nodes"] == ["example-node-001", "example-node-002"]
    assert context["gpu_count"] == 16
    assert context["namespace"] == "example-team"
    assert "allocations" not in context


def _assert_allocated(context, event):
    assert context["event_id"] == event["id"]
    assert context["job_id"] == event["data"]["job_id"]
    assert context["nodes"] == ALLOCATED_NODES
    assert context["gpu_count"] == 16
    assert context["pod_nodes"] == ALLOCATED_POD_NODES
    assert context["pod_names"] == list(ALLOCATED_POD_NODES)
    assert "cluster" not in context
    assert context["allocations"] == event["data"]["allocations"]


@pytest.mark.parametrize("encoding", ["structured", "binary"])
def test_discover_returns_flat_and_allocation_events_from_one_topic(
    migration_spec, completion_event, allocated_event, publish_event, encoding
):
    publish_event(completion_event, encoding=encoding)
    publish_event(allocated_event, encoding=encoding)
    adapter = _build(migration_spec, {})
    try:
        flat, allocated = adapter.discover()
    finally:
        adapter.close()

    _assert_flat(flat, completion_event)
    _assert_allocated(allocated, allocated_event)
    assert adapter.last_discovery_stats["malformed"] == 0
    assert adapter.last_discovery_stats["filtered"] == 0
    _assert_drained(migration_spec)


def test_poll_delivers_flat_and_allocation_events_with_their_envelopes(
    migration_spec, completion_event, allocated_event, publish_event
):
    publish_event(completion_event)
    publish_event(allocated_event)
    adapter = _build(migration_spec, {})
    try:
        flat, allocated = _next_delivery(adapter), _next_delivery(adapter)
        _assert_flat(flat.context, completion_event)
        _assert_allocated(allocated.context, allocated_event)
        assert [flat.event["type"], allocated.event["type"]] == [EVENT_TYPE, EVENT_TYPE_V2]
        adapter.ack(flat)
        adapter.ack(allocated)
        assert _committed(adapter, allocated) == allocated.offset + 1
    finally:
        adapter.close()
    _assert_drained(migration_spec)


def test_allocation_event_supersedes_earlier_flat_event_for_the_same_attempt(
    migration_spec, completion_event, allocated_event, publish_event
):
    flat_data = completion_event["data"]
    allocated_event["data"].update(id=flat_data["id"], job_id=flat_data["job_id"], attempt_id=flat_data["attempt_id"])
    publish_event(completion_event)
    publish_event(allocated_event)
    adapter = _build(migration_spec, {})
    try:
        [context] = adapter.discover()
    finally:
        adapter.close()

    _assert_allocated(context, allocated_event)
    assert "namespace" not in context
    assert adapter.last_discovery_stats["superseded"] == 1
    assert adapter.last_discovery_stats["redeliveries"] == 0
    _assert_drained(migration_spec)
