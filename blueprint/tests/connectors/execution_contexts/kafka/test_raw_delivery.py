# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw delivery preserves caller custody through the existing Kafka lifecycle."""

from dataclasses import replace

import pytest

from attribution_blueprint.connectors.execution_contexts import InvalidReceiptError
from attribution_blueprint.connectors.execution_contexts.kafka import KafkaExecutionContextAdapter
from attribution_blueprint.connectors.execution_contexts.kafka.connector import KafkaConsumerError

from .test_connector import (
    TOPIC,
    TP,
    FakeConsumer,
    FakeError,
    FakeMessage,
    _events,
    _make_eca,
    _structured,
    lose,
    revoke,
)


@pytest.fixture
def raw_stream():
    """Reuse the native consumer fake and close each adapter after its test."""
    adapters = []

    def create(script, **kwargs):
        adapter, consumer = _make_eca(script, **kwargs)
        adapters.append(adapter)
        return adapter, consumer

    yield create
    for adapter in adapters:
        adapter.close()


@pytest.mark.parametrize(
    "payload",
    [b'{"kind":"snapshot"}', b"not JSON", b"\xff", b"", None],
    ids=["json", "malformed", "binary", "empty", "tombstone"],
)
def test_raw_poll_preserves_native_record_fields_without_committing(raw_stream, payload):
    """Caller-owned decoding includes malformed values, empty values and tombstones."""
    headers = [("content-type", b"application/json"), ("trace", None), ("trace", b"second")]
    adapter, consumer = raw_stream([FakeMessage(payload, 7, key=b"job-key", headers=headers)])

    [receipt] = adapter.poll_raw(1, 0)

    assert (receipt.topic, receipt.partition, receipt.offset) == (TOPIC, 0, 7)
    assert receipt.key == b"job-key"
    assert receipt.value == payload
    assert receipt.headers == headers
    headers.append(("later", b"mutation"))
    assert receipt.headers == [("content-type", b"application/json"), ("trace", None), ("trace", b"second")]
    assert consumer.commits == []
    adapter.ack(receipt)
    assert consumer.commits == [{0: 8}]


@pytest.mark.parametrize("headers", [None, []], ids=["missing", "empty"])
def test_raw_poll_preserves_absent_and_empty_headers(raw_stream, headers):
    """Native metadata is not replaced by an invented envelope or key."""
    adapter, _ = raw_stream([FakeMessage(b"value", 0, headers=headers)])
    [receipt] = adapter.poll_raw(1, 0)
    assert receipt.headers == headers
    assert receipt.key is None


def test_raw_poll_keeps_cloudevent_envelope_and_bypasses_context_mapping(raw_stream):
    """Event-type filters and execution-context validation belong to the raw caller."""
    message = FakeMessage(b'{"specversion":"1.0","type":"custom.event","data":{"native_id":12}}', 0)
    adapter, consumer = raw_stream([message, message])
    first, second = adapter.poll_raw(2, 0)
    assert first.value == second.value == message.value()
    assert adapter.last_discovery_stats["consumed"] == 2
    assert adapter.last_discovery_stats["accepted"] == 0
    assert adapter.last_discovery_stats["raw_delivered"] == 2
    assert adapter.last_discovery_stats["returned"] == 2
    assert adapter.last_discovery_stats["malformed"] == 0
    assert adapter.last_discovery_stats["filtered"] == 0
    assert consumer.commits == []


@pytest.mark.parametrize(
    "fields",
    [
        {"key": "decoded"},
        {"key": {"id": 1}},
        {"key": bytearray(b"key")},
        {"value": "decoded"},
        {"value": {"id": 1}},
        {"value": bytearray(b"value")},
        {"headers": [("trace", "decoded")]},
        {"headers": [("trace", {"id": 1})]},
        {"headers": [("trace", bytearray(b"value"))]},
        {"headers": [(b"trace", b"value")]},
        {"headers": [("trace",)]},
        {"headers": [None]},
        {"headers": {"trace": b"value"}},
    ],
    ids=[
        "string-key",
        "object-key",
        "mutable-key",
        "string-value",
        "object-value",
        "mutable-value",
        "string-header-value",
        "object-header-value",
        "mutable-header-value",
        "bytes-header-name",
        "incomplete-header",
        "null-header-entry",
        "mapping-headers",
    ],
)
def test_raw_poll_rejects_deserialized_fields_without_committing(raw_stream, fields):
    """A deserializing consumer cannot silently break the original-byte contract."""
    message = FakeMessage(**({"key": b"key", "value": b"value", "offset": 1} | fields))
    adapter, consumer = raw_stream([FakeMessage(b"pending", 0), message])
    [pending] = adapter.poll_raw(1, 0)

    with pytest.raises(InvalidReceiptError, match="raw"):
        adapter.poll_raw(1, 0)

    assert consumer.closed
    assert consumer.commits == []
    assert adapter.last_discovery_stats["returned"] == 1
    with pytest.raises(InvalidReceiptError, match="closed"):
        adapter.ack(pending)


def test_raw_receipt_validates_direct_construction_and_copies_headers(raw_stream):
    """The public receipt enforces its contract even outside the connector."""
    adapter, _ = raw_stream([FakeMessage(b"value", 0)])
    [receipt] = adapter.poll_raw(1, 0)
    with pytest.raises(InvalidReceiptError, match="raw"):
        replace(receipt, value={"decoded": True})
    headers = [("trace", b"")]
    copied = replace(receipt, headers=headers, key=b"")
    headers.clear()
    assert copied.headers == [("trace", b"")]
    assert copied.key == b""


def test_raw_acknowledgements_cannot_skip_unhandled_payloads(raw_stream):
    """Even an undecodable record holds the prefix until the caller handles it."""
    adapter, consumer = raw_stream([FakeMessage(b"not JSON", 3), FakeMessage(None, 9)])
    first, last = adapter.poll_raw(2, 0)
    adapter.ack(last)
    assert consumer.commits == []
    adapter.ack(first)
    assert consumer.commits == [{0: 10}]
    adapter.ack(first)
    assert consumer.commits == [{0: 10}]


@pytest.mark.parametrize("release", [revoke, lose], ids=["revoke", "loss"])
def test_raw_reassignment_rejects_receipts_from_previous_ownership(raw_stream, release):
    """Replayed offsets require the receipt from the current assignment."""
    adapter, consumer = raw_stream(
        [FakeMessage(b"first", 0), release(0), lambda client: client.partitions.add(0), FakeMessage(b"first", 0)]
    )
    [original] = adapter.poll_raw(1, 0)
    [replay] = adapter.poll_raw(1, 0)
    with pytest.raises(InvalidReceiptError):
        adapter.ack(original)
    assert consumer.commits == []
    adapter.ack(replay)
    assert consumer.commits == [{0: 1}]


def test_raw_partition_loss_preserves_other_owned_receipts(raw_stream):
    """An ownership change affects only its own partition."""
    adapter, consumer = raw_stream(
        [FakeMessage(b"first", 0), FakeMessage(b"second", 0, partition=1), lose(0)], partitions=(0, 1)
    )
    first, second = adapter.poll_raw(2, 0)
    assert adapter.poll_raw(1, 0) == []
    with pytest.raises(InvalidReceiptError):
        adapter.ack(first)
    adapter.ack(second)
    assert consumer.commits == [{1: 1}]


def test_raw_unknown_receipt_is_rejected_without_closing_consumer(raw_stream):
    """Invalid application acknowledgements leave healthy intake usable."""
    adapter, consumer = raw_stream([FakeMessage(b"value", 0)])
    [receipt] = adapter.poll_raw(1, 0)
    with pytest.raises(InvalidReceiptError):
        adapter.ack(replace(receipt, offset=100))
    assert not consumer.closed
    adapter.ack(receipt)
    assert consumer.commits == [{0: 1}]


def test_raw_commit_failure_closes_and_invalidates_pending_receipts(raw_stream):
    """Uncertain broker commits require recovery with a new consumer."""
    adapter, consumer = raw_stream([FakeMessage(b"value", 0)], commit_raises=True)
    [receipt] = adapter.poll_raw(1, 0)
    with pytest.raises(KafkaConsumerError):
        adapter.ack(receipt)
    assert consumer.closed
    with pytest.raises(InvalidReceiptError):
        adapter.ack(receipt)


def test_raw_poll_error_closes_without_committing_pending_work(raw_stream):
    """A transport failure leaves unfinished records available for Kafka replay."""
    adapter, consumer = raw_stream([FakeMessage(b"value", 0), FakeMessage(None, 1, error=FakeError("unavailable"))])
    [receipt] = adapter.poll_raw(1, 0)
    with pytest.raises(KafkaConsumerError):
        adapter.poll_raw(1, 0)
    assert consumer.closed
    assert consumer.commits == []
    with pytest.raises(InvalidReceiptError):
        adapter.ack(receipt)


def test_raw_pause_keeps_polling_without_delivering_or_committing(raw_stream):
    """The existing flow control also bounds undecoded intake."""
    adapter, consumer = raw_stream([lambda client: client.on_assign(client, [TP(TOPIC, 0)]), FakeMessage(b"value", 0)])
    adapter.pause()
    assert adapter.poll_raw(1, 0) == []
    assert consumer.polls == 1
    assert consumer.commits == []
    adapter.resume()
    [receipt] = adapter.poll_raw(1, 0)
    assert receipt.value == b"value"


@pytest.mark.parametrize("read", ["poll", "discover"], ids=["decoded-poll", "discovery"])
def test_raw_mode_rejects_other_read_lifecycles_without_losing_work(raw_stream, read):
    """One instance cannot switch decoding and acknowledgement policies midstream."""
    adapter, consumer = raw_stream([FakeMessage(b"value", 0)])
    [receipt] = adapter.poll_raw(1, 0)
    with pytest.raises(KafkaConsumerError):
        getattr(adapter, read)()
    assert not consumer.closed
    adapter.ack(receipt)
    assert consumer.commits == [{0: 1}]


def test_decoded_poll_rejects_raw_reads_and_preserves_default_contexts(raw_stream):
    """Existing consumers keep the decoded CloudEvent contract and mode guard."""
    adapter, consumer = raw_stream(_structured(_events()[:1]))
    [receipt] = adapter.poll(1, 0)
    assert receipt.context["id"] == "job-a1"
    with pytest.raises(KafkaConsumerError):
        adapter.poll_raw(1, 0)
    adapter.ack(receipt)
    assert consumer.commits == [{0: 1}]


def test_factory_opens_independent_raw_consumer_without_changing_source():
    """A raw consumer uses the configured native factory and retains the envelope."""
    native = FakeConsumer([FakeMessage(b"value", 0)])
    source = KafkaExecutionContextAdapter(consumer_factory=lambda: native, topic=TOPIC, topic_partition=TP)
    opened = source.open_raw_consumer()
    try:
        [receipt] = opened.poll_raw(1, 0)
        assert receipt.value == b"value"
        opened.ack(receipt)
        assert native.commits == [{0: 1}]
    finally:
        opened.close()
        source.close()


@pytest.mark.parametrize("max_records", [0, -1, True, 1.5])
def test_raw_poll_rejects_invalid_batch_limits_before_fetching(raw_stream, max_records):
    """The same bounded-poll contract applies to raw delivery."""
    adapter, consumer = raw_stream([])
    with pytest.raises(ValueError):
        adapter.poll_raw(max_records, 0)
    assert consumer.polls == 0


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True])
def test_raw_poll_rejects_invalid_timeouts_before_fetching(raw_stream, timeout):
    """Invalid timeout inputs never reach the native client."""
    adapter, consumer = raw_stream([])
    with pytest.raises(ValueError):
        adapter.poll_raw(1, timeout)
    assert consumer.polls == 0


def test_raw_receipt_is_exported_without_execution_context_fields(raw_stream):
    """Raw receipts do not pretend an execution context is ready."""
    from attribution_blueprint.connectors.execution_contexts.kafka import KafkaRawReceipt

    adapter, _ = raw_stream([FakeMessage(b"value", 0)])
    [receipt] = adapter.poll_raw(1, 0)
    assert isinstance(receipt, KafkaRawReceipt)
    assert set(receipt.__dataclass_fields__) == {
        "topic",
        "partition",
        "offset",
        "assignment_id",
        "key",
        "value",
        "headers",
    }


def test_raw_factory_requires_an_independent_native_consumer(raw_stream):
    """Opening raw mode cannot take ownership of a source's injected consumer."""
    source, native = raw_stream([])
    with pytest.raises(KafkaConsumerError, match="consumer factory"):
        source.open_raw_consumer()
    assert not native.closed


def test_closed_raw_consumer_rejects_reads_and_acknowledgements(raw_stream):
    """Close fences raw receipts just as it fences decoded deliveries."""
    adapter, native = raw_stream([FakeMessage(b"value", 0)])
    [receipt] = adapter.poll_raw(1, 0)
    adapter.close()
    with pytest.raises(KafkaConsumerError, match="closed"):
        adapter.poll_raw(1, 0)
    with pytest.raises(InvalidReceiptError, match="closed"):
        adapter.ack(receipt)
    assert native.commits == []
