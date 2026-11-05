# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register the optional Kafka result sink with the output framework."""

import math

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.kafka_client import SECURITY_SCHEMA_KEYS, security_config
from attribution_blueprint.connectors.outputs.kafka.connector import (
    DEFAULT_MAX_MESSAGE_BYTES,
    EVENT_SOURCE,
    EVENT_TYPE,
    KafkaOutput,
    KafkaOutputError,
    _validate_options,
)
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM


def _output_options(spec: dict) -> dict:
    """Coerce and validate sink options before creating a native client."""
    from attribution_blueprint.mca import _coerce_float, _coerce_int

    missing = [key for key in ("bootstrap_servers", "topic") if not spec.get(key)]
    if missing:
        raise MCAError(f"output.kafka missing required keys: {missing}")
    if isinstance(spec.get("delivery_timeout_seconds"), bool):
        raise MCAError("output.kafka.delivery_timeout_seconds must be finite and positive")
    options = {
        "topic": spec["topic"],
        "event_type": spec.get("event_type", EVENT_TYPE),
        "source": spec.get("source", EVENT_SOURCE),
        "cluster": spec.get("cluster"),
        "max_message_bytes": _coerce_int(
            spec.get("max_message_bytes", DEFAULT_MAX_MESSAGE_BYTES), "output.kafka.max_message_bytes"
        ),
        "delivery_timeout_seconds": _coerce_float(
            spec.get("delivery_timeout_seconds", 30.0), "output.kafka.delivery_timeout_seconds"
        ),
    }
    try:
        _validate_options(
            options["topic"],
            options["event_type"],
            options["source"],
            options["max_message_bytes"],
            options["delivery_timeout_seconds"],
        )
    except ValueError as exc:
        raise MCAError(str(exc)) from exc
    return options


def _build(spec: dict, ctx: dict) -> KafkaOutput:
    """Construct an idempotent native producer, importing the optional extra on use."""
    options = _output_options(spec)
    config = {
        "bootstrap.servers": str(spec["bootstrap_servers"]),
        "acks": "all",
        "enable.idempotence": True,
        "message.timeout.ms": math.ceil(options["delivery_timeout_seconds"] * 1000),
        # Blueprint checks the event cap before enqueueing. Keep the native
        # 1 MB default unless the configured event limit requires more.
        "message.max.bytes": max(1_000_000, options["max_message_bytes"]),
        **security_config(spec, "output.kafka"),
    }
    try:
        from confluent_kafka import KafkaException, Producer
    except ImportError as exc:
        raise MCAError("output.kafka requires the kafka extra: install nvidia-attribution-blueprint[kafka]") from exc
    try:
        return KafkaOutput(producer=Producer(config), **options)
    except KafkaException as exc:
        raise MCAError(f"output.kafka invalid producer configuration: {exc}") from exc


_registry.register_component(
    framework="output",
    name="kafka",
    schema=dict.fromkeys(
        (
            "bootstrap_servers",
            "topic",
            "event_type",
            "source",
            "cluster",
            "max_message_bytes",
            "delivery_timeout_seconds",
            *SECURITY_SCHEMA_KEYS,
        ),
        _PARAM,
    ),
    build=_build,
)

__all__ = ["KafkaOutput", "KafkaOutputError"]
