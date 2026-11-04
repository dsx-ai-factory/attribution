# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish structured CloudEvents and wait for each record's delivery report.

The producer owns transport retries. A timeout is an uncertain outcome: the
broker may still have stored the record. This sink neither retries attribution
nor acknowledges input offsets. Instances are used by one writer at a time.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Callable, Protocol
from uuid import uuid4

from attribution_blueprint.types import ExecutionContext

EVENT_TYPE = "com.nvidia.attribution.job.attributed.v1"
EVENT_SOURCE = "/attribution/blueprint"
_SUMMARY_FIELDS = ("terminal_state", "sub_domain", "tenant_sre_action", "cloud_operator_sre_action")
_REPORT_FIELDS = ("connector_reports", "ingestion_reports", "evidence_status")


class KafkaProducer(Protocol):
    """The native producer operations needed by this sink."""

    def produce(self, topic: str, *, key: bytes, value: bytes, headers: list, on_delivery: Callable) -> None: ...

    def flush(self, timeout: float) -> int: ...


class KafkaOutputError(RuntimeError):
    """Publishing a result failed or its delivery could not be confirmed."""


def _validate_options(topic: str, event_type: str, source: str, max_bytes: int, timeout: float) -> None:
    """Reject invalid options before allocating a native producer."""
    for name, value in (("topic", topic), ("event_type", event_type), ("source", source)):
        if not isinstance(value, str) or not value or any(char.isspace() for char in value):
            raise ValueError(f"output.kafka.{name} must be a non-empty string without whitespace")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 0 < max_bytes <= 1_000_000_000:
        raise ValueError("output.kafka.max_message_bytes must be an integer in 1..1000000000")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 2147483.647:
        raise ValueError("output.kafka.delivery_timeout_seconds must be finite and in (0, 2147483.647]")


def _event(context: ExecutionContext, attribution: dict, *, event_type: str, source: str, cluster: str | None) -> dict:
    """Map an enriched result to the RFC envelope without changing its evidence."""
    job_id = str(context.get("job_id") or context["id"])
    attempt_id = context.get("attempt_id")
    data = {
        "job_id": job_id,
        "attempt_id": str(attempt_id) if attempt_id is not None else "0",
        "cluster": context.get("cluster") or cluster,
        "scheduler": context.get("scheduler"),
        "trigger_event_id": context.get("event_id"),
        **{field: attribution.get(field, "") for field in _SUMMARY_FIELDS},
        "result": {k: v for k, v in attribution.items() if k not in (*_SUMMARY_FIELDS, *_REPORT_FIELDS)},
        "connector_reports": attribution.get("connector_reports", []),
    }
    for field in ("ingestion_reports", "evidence_status"):
        if field in attribution:
            data[field] = attribution[field]
    subject = context.get("event_subject")
    return {
        "specversion": "1.0",
        "id": str(uuid4()),
        "source": source,
        "type": event_type,
        "subject": subject if isinstance(subject, str) and subject else f"jobs/{job_id}",
        "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "datacontenttype": "application/json",
        "data": data,
    }


class KafkaOutput:
    """Write one RFC result event per context, keyed by the input's run ID."""

    def __init__(
        self,
        *,
        producer: KafkaProducer,
        topic: str,
        event_type: str = EVENT_TYPE,
        source: str = EVENT_SOURCE,
        cluster: str | None = None,
        max_message_bytes: int = 900_000,
        delivery_timeout_seconds: float = 30.0,
    ):
        """Use an injected producer; the registry builder supplies the native client."""
        _validate_options(topic, event_type, source, max_message_bytes, delivery_timeout_seconds)
        self._producer = producer
        self._topic = topic
        self._event_type = event_type
        self._source = source
        self._cluster = cluster
        self._max_bytes = max_message_bytes
        self._timeout = delivery_timeout_seconds
        self._closed = False

    def write(self, context: ExecutionContext, attribution: dict) -> None:
        """Return only on broker acknowledgement; propagate enqueue and delivery failures."""
        if self._closed:
            raise KafkaOutputError("output.kafka is closed; create a new output connector")
        event = _event(context, attribution, event_type=self._event_type, source=self._source, cluster=self._cluster)
        value = json.dumps(event, default=str, allow_nan=False, separators=(",", ":")).encode()
        if len(value) > self._max_bytes:
            raise KafkaOutputError(
                f"output.kafka event is {len(value)} bytes, over max_message_bytes={self._max_bytes}"
            )
        delivery: list[Any] = []
        self._producer.produce(
            self._topic,
            key=str(context["id"]).encode(),
            value=value,
            headers=[("content-type", "application/cloudevents+json")],
            on_delivery=lambda error, _message: delivery.append(error),
        )
        remaining = self._producer.flush(self._timeout)
        if not delivery:
            raise KafkaOutputError(f"output.kafka delivery unconfirmed after {self._timeout}s ({remaining} pending)")
        if delivery[0] is not None:
            raise KafkaOutputError(f"output.kafka delivery failed: {delivery[0]}")

    def close(self) -> None:
        """Flush pending records once; reject writes even if that final flush fails."""
        if self._closed:
            return
        self._closed = True
        remaining = self._producer.flush(self._timeout)
        if remaining:
            raise KafkaOutputError(f"output.kafka close left {remaining} records pending; delivery is uncertain")
