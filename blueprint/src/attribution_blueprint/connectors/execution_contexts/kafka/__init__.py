# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.execution_contexts.kafka.connector import (
    KafkaExecutionContext,
    KafkaExecutionContextAdapter,
)
from attribution_blueprint.connectors.execution_contexts.kafka.receipts import KafkaReceipt
from attribution_blueprint.connectors.kafka_client import SECURITY_SCHEMA_KEYS, security_config
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM


def _consumer_config(spec: dict) -> dict:
    """Translate `[eca.kafka]` into librdkafka consumer settings.

    Credentials are read from the environment variables named by
    `sasl_username_env` / `sasl_password_env`, never from the TOML itself.
    """
    missing = [key for key in ("bootstrap_servers", "topic", "group_id") if not spec.get(key)]
    if missing:
        raise MCAError(f"eca.kafka missing required keys: {missing}")

    if spec.get("auto_offset_reset", "earliest") != "earliest":
        raise MCAError("eca.kafka supports only auto_offset_reset=earliest; latest needs assignment checkpointing")
    config = {
        "bootstrap.servers": str(spec["bootstrap_servers"]),
        "group.id": str(spec["group_id"]),
        "enable.auto.commit": False,
        "enable.partition.eof": True,
        "auto.offset.reset": "earliest",
    }

    config.update(security_config(spec, "eca.kafka"))
    return config


def _build(spec: dict, ctx: dict) -> KafkaExecutionContextAdapter:
    """Build the input adapter; import the optional native client only on use."""
    from attribution_blueprint.mca import _coerce_float, _coerce_int, _coerce_str_list

    config = _consumer_config(spec)
    event_types_raw = spec.get("event_types")
    event_types = _coerce_str_list(event_types_raw, "eca.kafka.event_types") if event_types_raw else []
    max_messages = _coerce_int(spec.get("max_messages", 500), "eca.kafka.max_messages")
    poll_timeout = _coerce_float(spec.get("poll_timeout_seconds", 5.0), "eca.kafka.poll_timeout_seconds")
    timeout = _coerce_float(spec.get("discovery_timeout_seconds", 60), "eca.kafka.discovery_timeout_seconds")

    try:
        from confluent_kafka import Consumer, KafkaException, TopicPartition
    except ImportError as exc:
        raise MCAError("eca.kafka requires the kafka extra: install nvidia-attribution-blueprint[kafka]") from exc
    try:
        return KafkaExecutionContextAdapter(
            consumer_factory=partial(Consumer, config),
            consumer_error=KafkaException,
            topic=str(spec["topic"]),
            topic_partition=TopicPartition,
            event_types=event_types,
            scheduler=str(spec.get("scheduler") or "kafka"),
            workload_type=str(spec.get("workload_type", "")),
            max_messages=max_messages,
            poll_timeout_seconds=poll_timeout,
            discovery_timeout_seconds=timeout,
        )
    except ValueError as exc:
        raise MCAError(str(exc)) from exc


_registry.register_component(
    framework="eca",
    name="kafka",
    schema={
        # Comma-separated host:port list.
        "bootstrap_servers": _PARAM,
        "topic": _PARAM,
        # Consumer group; offsets are committed per group, so each
        # deployment reading the same topic needs its own.
        "group_id": _PARAM,
        # Only earliest is supported; latest requires assignment checkpointing.
        "auto_offset_reset": _PARAM,
        # CloudEvent `type` values to accept. Unset accepts every type.
        "event_types": _PARAM,
        # Scheduler identity when an event's data has no `scheduler`.
        "scheduler": _PARAM,
        # Wire-format `workload.type` sent to the attribution service.
        "workload_type": _PARAM,
        "max_messages": _PARAM,
        "poll_timeout_seconds": _PARAM,
        # Bound the complete read, including assignment and empty fetches.
        "discovery_timeout_seconds": _PARAM,
        **dict.fromkeys(SECURITY_SCHEMA_KEYS, _PARAM),
    },
    build=_build,
)


__all__ = ["KafkaExecutionContext", "KafkaExecutionContextAdapter", "KafkaReceipt"]
