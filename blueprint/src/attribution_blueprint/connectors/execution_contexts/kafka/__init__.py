# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.execution_contexts.kafka.connector import (
    KafkaExecutionContext,
    KafkaExecutionContextAdapter,
)
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM

_SECURITY_PROTOCOLS = ("PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL")
_SASL_MECHANISMS = ("PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512")


def _consumer_config(spec: dict) -> dict:
    """Translate `[eca.kafka]` into librdkafka consumer settings.

    Credentials are read from the environment variables named by
    `sasl_username_env` / `sasl_password_env`, never from the TOML itself.
    """
    missing = [key for key in ("bootstrap_servers", "topic", "group_id") if not spec.get(key)]
    if missing:
        raise MCAError(f"eca.kafka missing required keys: {missing}")

    auto_offset_reset = spec.get("auto_offset_reset") or "earliest"
    if auto_offset_reset not in ("earliest", "latest"):
        raise MCAError(f"eca.kafka.auto_offset_reset must be earliest or latest, got {auto_offset_reset!r}")
    config = {
        "bootstrap.servers": str(spec["bootstrap_servers"]),
        "group.id": str(spec["group_id"]),
        "enable.auto.commit": False,
        "enable.partition.eof": True,
        "auto.offset.reset": auto_offset_reset,
    }

    security_protocol = spec.get("security_protocol") or "PLAINTEXT"
    if security_protocol not in _SECURITY_PROTOCOLS:
        raise MCAError(f"eca.kafka.security_protocol must be one of {list(_SECURITY_PROTOCOLS)}")
    config["security.protocol"] = security_protocol
    if spec.get("ssl_ca_location"):
        config["ssl.ca.location"] = str(spec["ssl_ca_location"])

    if security_protocol.startswith("SASL_"):
        mechanism = spec.get("sasl_mechanism")
        if mechanism not in _SASL_MECHANISMS:
            raise MCAError(f"eca.kafka.sasl_mechanism must be one of {list(_SASL_MECHANISMS)}")
        config["sasl.mechanism"] = mechanism
        for key, setting in (("sasl_username_env", "sasl.username"), ("sasl_password_env", "sasl.password")):
            env_name = spec.get(key)
            if not env_name:
                raise MCAError(f"eca.kafka security_protocol={security_protocol} requires {key}")
            if env_name not in os.environ:
                raise MCAError(f"eca.kafka.{key} references unset environment variable: {env_name}")
            config[setting] = os.environ[env_name]
    return config


def _build(spec: dict, ctx: dict) -> KafkaExecutionContextAdapter:
    from attribution_blueprint.mca import _coerce_float, _coerce_int, _coerce_str_list

    config = _consumer_config(spec)
    event_types_raw = spec.get("event_types")
    event_types = _coerce_str_list(event_types_raw, "eca.kafka.event_types") if event_types_raw else []
    max_messages = _coerce_int(spec.get("max_messages", 500), "eca.kafka.max_messages")
    poll_timeout = _coerce_float(spec.get("poll_timeout_seconds", 5.0), "eca.kafka.poll_timeout_seconds")
    max_transient_errors = _coerce_int(spec.get("max_transient_errors", 3), "eca.kafka.max_transient_errors")
    assignment_timeout = _coerce_float(
        spec.get("assignment_timeout_seconds", 60.0), "eca.kafka.assignment_timeout_seconds"
    )

    fetch_timeout = _coerce_float(spec.get("fetch_timeout_seconds", 60.0), "eca.kafka.fetch_timeout_seconds")

    try:
        from confluent_kafka import Consumer, TopicPartition
    except ImportError as exc:
        raise MCAError("eca.kafka requires the kafka extra: install nvidia-attribution-blueprint[kafka]") from exc
    consumer = Consumer(config)
    try:
        return KafkaExecutionContextAdapter(
            consumer=consumer,
            topic=str(spec["topic"]),
            topic_partition=TopicPartition,
            event_types=event_types,
            scheduler=str(spec.get("scheduler") or "kafka"),
            workload_type=str(spec.get("workload_type", "")),
            max_messages=max_messages,
            poll_timeout_seconds=poll_timeout,
            assignment_timeout_seconds=assignment_timeout,
            fetch_timeout_seconds=fetch_timeout,
            max_transient_errors=max_transient_errors,
        )
    except ValueError as exc:
        consumer.close()
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
        # Where a new group starts: earliest (default) or latest.
        "auto_offset_reset": _PARAM,
        # CloudEvent `type` values to accept. Unset accepts every type.
        "event_types": _PARAM,
        # Scheduler identity when an event's data has no `scheduler`.
        "scheduler": _PARAM,
        # Wire-format `workload.type` sent to the attribution service.
        "workload_type": _PARAM,
        "max_messages": _PARAM,
        "poll_timeout_seconds": _PARAM,
        # How long to wait for the consumer group to assign partitions.
        "assignment_timeout_seconds": _PARAM,
        # Maximum wait without progress while assigned partitions have unread data.
        "fetch_timeout_seconds": _PARAM,
        # Retriable consumer errors tolerated per run before it fails.
        "max_transient_errors": _PARAM,
        "security_protocol": _PARAM,
        "sasl_mechanism": _PARAM,
        "sasl_username_env": _PARAM,
        "sasl_password_env": _PARAM,
        "ssl_ca_location": _PARAM,
    },
    build=_build,
)


__all__ = ["KafkaExecutionContext", "KafkaExecutionContextAdapter"]
