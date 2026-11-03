# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native exception behavior without requiring a broker."""

import pytest

from attribution_blueprint.connectors.execution_contexts.kafka.client import KafkaConsumerClient, KafkaConsumerError

native = pytest.importorskip("confluent_kafka")


@pytest.mark.parametrize(
    "operation,args,kwargs",
    [
        ("subscribe", (["events"],), {}),
        ("poll", (0.1,), {}),
        ("assignment", (), {}),
        ("position", ([],), {}),
        ("get_watermark_offsets", (object(),), {"timeout": 1, "cached": False}),
        ("commit", (), {"offsets": [], "asynchronous": False}),
        ("seek", (object(),), {}),
        ("close", (), {}),
    ],
)
@pytest.mark.parametrize("failure", ["kafka", "programming"])
def test_native_boundary_translates_only_kafka_errors(operation, args, kwargs, failure):
    error = (
        native.KafkaException(native.KafkaError(native.KafkaError._FAIL)) if failure == "kafka" else TypeError("bug")
    )

    class FailingConsumer:
        pass

    def broken(*args, **kwargs):
        raise error

    consumer = FailingConsumer()
    setattr(consumer, operation, broken)
    client = KafkaConsumerClient(consumer)
    expected = KafkaConsumerError if failure == "kafka" else TypeError
    with pytest.raises(expected) as caught:
        getattr(client, operation)(*args, **kwargs)
    if failure == "kafka":
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error
