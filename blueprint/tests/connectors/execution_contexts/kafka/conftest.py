# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for opt-in integration tests against disposable local services."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def kafka_topic():
    brokers = os.environ.get("ATTRIBUTION_TEST_KAFKA_BOOTSTRAP_SERVERS")
    service_url = os.environ.get("ATTRIBUTION_TEST_SERVICE_URL")
    if not brokers or not service_url:
        pytest.skip(
            "set ATTRIBUTION_TEST_KAFKA_BOOTSTRAP_SERVERS and ATTRIBUTION_TEST_SERVICE_URL to disposable services"
        )
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": brokers, "socket.timeout.ms": 10000})
    topic = f"blueprint-completion-contract-{uuid4().hex}"
    admin.create_topics([NewTopic(topic, num_partitions=2, replication_factor=1)], request_timeout=15)[topic].result(20)
    try:
        yield brokers, service_url, topic
    finally:
        admin.delete_topics([topic], request_timeout=15)[topic].result(20)


@pytest.fixture
def completion_event():
    path = Path(__file__).parents[3] / "fixtures" / "kafka_completion_events.json"
    event = json.loads(path.read_text())[0]
    event["id"] = str(uuid4())
    event["data"]["id"] = event["data"]["job_id"] = f"job-{uuid4().hex}"
    event["subject"] = f"jobs/{event['data']['id']}"
    return event


@pytest.fixture
def kafka_spec(kafka_topic):
    brokers, _, topic = kafka_topic
    return {
        "bootstrap_servers": brokers,
        "topic": topic,
        "group_id": f"completion-contract-{uuid4().hex}",
        "event_types": ["com.example.job.completed.v1"],
        "poll_timeout_seconds": 1,
        "discovery_timeout_seconds": 20,
    }


@pytest.fixture
def publish_event(kafka_topic):
    from confluent_kafka import Producer

    brokers, _, topic = kafka_topic
    producer = Producer({"bootstrap.servers": brokers})

    def publish(event, *, partition=0, encoding="structured"):
        errors = []

        def delivered(error, message):
            if error is not None:
                errors.append(str(error))

        if encoding == "binary":
            value = json.dumps(event["data"]).encode()
            headers = [
                (f"ce_{key}", value.encode()) for key, value in event.items() if key not in {"data", "datacontenttype"}
            ]
            headers.append(("content-type", b"application/json"))
        else:
            value = json.dumps(event).encode()
            headers = None
        producer.produce(
            topic, key=event["data"]["id"], value=value, headers=headers, partition=partition, on_delivery=delivered
        )
        assert producer.flush(15) == 0
        assert errors == []

    return publish


@pytest.fixture
def unavailable_endpoint():
    # Reserve a local port without listening: no unrelated process can claim
    # it and accidentally turn a failure-path test into a successful request.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        yield f"127.0.0.1:{listener.getsockname()[1]}"


@pytest.fixture
def run_kafka_cli(kafka_spec, kafka_topic, tmp_path):
    _, service_url, _ = kafka_topic

    def run(*, service=service_url, database=None, spec=None):
        config = tmp_path / f"config-{uuid4().hex}.toml"
        settings = kafka_spec if spec is None else spec
        database = tmp_path / "results.db" if database is None else database
        lines = ["[eca]", 'component = "kafka"', "[eca.kafka]"]
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in settings.items())
        lines.extend(
            [
                "[attribution]",
                f"service_url = {json.dumps(service)}",
                "timeout_seconds = 2",
                "[output.db]",
                'component = "sqlite"',
                "[output.db.sqlite]",
                f"path = {json.dumps(str(database))}",
            ]
        )
        config.write_text("\n".join(lines) + "\n")
        return subprocess.run(
            [sys.executable, "-m", "attribution_blueprint.cli", "--config", str(config)],
            capture_output=True,
            text=True,
            timeout=45,
        )

    return run
