# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest
from loguru import logger
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter  # NOT InMemoryLogExporter (deprecated)

import app.logging_config as logcfg

otlp_endpoint_key = "OTEL_EXPORTER_OTLP_ENDPOINT"
otlp_token_key = "OTEL_EXPORTER_OTLP_TOKEN"
otlp_protocol_key = "OTEL_EXPORTER_OTLP_PROTOCOL"
otlp_endpoint_value = "http://localhost:4317"
otlp_token_value = "dummy"


@pytest.fixture(autouse=True)
def loguru_isolation():
    before = set(logger._core.handlers)
    logcfg._provider = None  # clear the module singleton so each test gets a fresh setup
    yield
    for hid in set(logger._core.handlers) - before:
        logger.remove(hid)
    logcfg._provider = None


def test_loguru_message_reaches_exporter_with_service_name(monkeypatch):
    monkeypatch.setenv(otlp_endpoint_key, otlp_endpoint_value)
    monkeypatch.setenv(otlp_token_key, otlp_token_value)
    mem = InMemoryLogRecordExporter()
    monkeypatch.setattr(logcfg, "_build_exporter", lambda endpoint, token: mem)
    provider = logcfg.setup_logging()
    assert provider is not None

    msg = "otlp smoke test"
    logger.info(msg)
    provider.force_flush()
    recs = mem.get_finished_logs()

    assert any(str(r.log_record.body) == msg for r in recs)


def test_noop_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv(otlp_endpoint_key, raising=False)

    assert logcfg._get_or_create_provider() is None


def test_exports_without_token_when_only_endpoint_set(monkeypatch):
    monkeypatch.setenv(otlp_endpoint_key, otlp_endpoint_value)
    monkeypatch.delenv(otlp_token_key, raising=False)
    seen = {}

    def fake_build(endpoint, token):
        seen["endpoint"], seen["token"] = endpoint, token
        return InMemoryLogRecordExporter()

    monkeypatch.setattr(logcfg, "_build_exporter", fake_build)

    assert logcfg._get_or_create_provider() is not None
    assert seen["token"] is None


def _collect_warnings(monkeypatch):
    monkeypatch.setenv(otlp_endpoint_key, otlp_endpoint_value)
    monkeypatch.setenv(otlp_token_key, otlp_token_value)
    monkeypatch.setattr(logcfg, "_build_exporter", lambda endpoint, token: InMemoryLogRecordExporter())
    warnings = []
    logger.add(warnings.append, level="WARNING", format="{message}")
    logcfg._get_or_create_provider()  # factory emits the protocol warning; no sink teardown to clobber our capture
    return warnings


def test_warns_when_protocol_not_grpc(monkeypatch):
    monkeypatch.setenv(otlp_protocol_key, "http/protobuf")
    assert any("gRPC only" in str(w) for w in _collect_warnings(monkeypatch))


def test_no_warn_when_protocol_unset(monkeypatch):
    monkeypatch.delenv(otlp_protocol_key, raising=False)
    assert not _collect_warnings(monkeypatch)


@pytest.mark.parametrize("protocol", ["grpc", "GRPC"])
def test_no_warn_when_protocol_grpc(monkeypatch, protocol):
    monkeypatch.setenv(otlp_protocol_key, protocol)
    assert not _collect_warnings(monkeypatch)
