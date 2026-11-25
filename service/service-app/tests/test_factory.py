# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import Request, Response
from loguru import logger

import app.factory as factory
from app import __version__


def test_importing_factory_does_not_initialize_rules():
    code = """
import attribution_lib

assert attribution_lib.initialized is False
import app.factory
assert attribution_lib.initialized is False
"""

    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def _write_config(tmp_path: Path, text: str) -> Path:
    """Write a library TOML file.

    Args:
        tmp_path: Directory for the file.
        text: TOML content.

    Returns:
        Path to the written file.
    """
    config = tmp_path / "attribution.toml"
    config.write_text(text)
    return config


def test_factory_reads_library_settings_from_config_file(monkeypatch, tmp_path):
    """Load all five library settings through the normal service factory.

    Args:
        monkeypatch: Pytest fixture for setting process variables.
        tmp_path: Directory for the test configuration, mapping, and additional rules.
    """
    (tmp_path / "nodes.csv").write_text("id,instanceId,ip\nnode-a,instance-a,127.0.0.1\n")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "extra.py").write_text(
        'from attribution_lib.rules import declare_event_types\ndeclare_event_types("test.additional")\n'
    )
    config = _write_config(
        tmp_path,
        """
applog_parse_timestamp = false
ignored_hostname_prefixes = ["scanner-"]
node_mapping_file = "nodes.csv"
node_domain_regex = '(?P<domain>rack\\d+)-node\\d+$'
rules_paths = ["rules"]
""",
    )
    monkeypatch.setenv("ATTRIBUTION_CONFIG_FILE", str(config))
    code = """
import attribution_lib
from app.factory import create_app
from attribution_lib.observation.inbound import Source
from attribution_lib.rules_engine import SOURCE_CONFIGS

create_app()
assert SOURCE_CONFIGS[Source.APPLOG].body_fallbacks.parse_timestamp is False
assert attribution_lib.ignored_hostname_prefixes == ["scanner-"]
assert attribution_lib.ip_to_nodenames == {"127.0.0.1": "node-a"}
assert attribution_lib.node_domain_resolver("RACK12-NODE1.example.org") == "rack12"
assert attribution_lib.node_domain_resolver("node-a") is None
assert attribution_lib.get_rules_engine().declared_event_types == {"test.additional"}
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_factory_warns_and_starts_with_empty_rules_and_ip_only_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Warn about missing rules and instance IDs while keeping IP lookup available.

    Args:
        monkeypatch: Pytest fixture for setting process variables.
        tmp_path: Directory for the test configuration, rule directory, and node mapping.
    """
    rules_path = tmp_path / "rules"
    rules_path.mkdir()
    mapping_file = tmp_path / "nodes.csv"
    mapping_file.write_text("id,ip\nnode-a,192.0.2.1\n")
    config = _write_config(
        tmp_path, f"rules_paths = [{str(rules_path)!r}]\nnode_mapping_file = {str(mapping_file)!r}\n"
    )
    monkeypatch.setenv("ATTRIBUTION_CONFIG_FILE", str(config))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    code = """
import attribution_lib
from app.factory import create_app
from fastapi.testclient import TestClient

with TestClient(create_app()) as client:
    assert client.get("/v2/healthz").status_code == 200
    assert attribution_lib.get_rules_engine().all_rule_names == set()
    assert attribution_lib.ip_to_nodenames == {"192.0.2.1": "node-a"}
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=30)
    assert "WARNING" in result.stdout
    assert "rules_paths loaded no attribution rules" in result.stdout
    assert "node_mapping_file has no instanceId column" in result.stdout


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('applog_parse_timestamp = "yes"', "applog_parse_timestamp"),
        ('ignored_hostname_prefixes = "scanner-"', "ignored_hostname_prefixes"),
        ('ignored_hostname_prefixes = [" "]', "ignored_hostname_prefixes"),
        ('node_mapping_file = ""', "node_mapping_file"),
        ('node_domain_regex = "("', "node_domain_regex"),
        ('node_domain_regex = "rack"', "node_domain_regex"),
        ("rules_paths = [1]", "rules_paths"),
        ('rules_paths = ["missing"]', "rules_paths"),
        ("rules_path = []", "rules_path"),
        ("not toml", "ATTRIBUTION_CONFIG_FILE"),
    ],
)
def test_factory_rejects_invalid_library_config(monkeypatch, tmp_path, text, error):
    """Name the invalid key when the service cannot start.

    Args:
        monkeypatch: Pytest fixture for setting process variables.
        tmp_path: Directory for the test configuration.
        text: Invalid TOML content.
        error: Text that the startup error must contain.
    """
    monkeypatch.setenv("ATTRIBUTION_CONFIG_FILE", str(_write_config(tmp_path, text)))
    monkeypatch.setattr(factory, "setup_logging", lambda: None)
    with pytest.raises(ValueError, match=error):
        factory.create_app()


def test_factory_rejects_missing_config_file(monkeypatch, tmp_path):
    """Fail startup when ATTRIBUTION_CONFIG_FILE names a file that does not exist.

    Args:
        monkeypatch: Pytest fixture for setting process variables.
        tmp_path: Directory without the configuration file.
    """
    monkeypatch.setenv("ATTRIBUTION_CONFIG_FILE", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(factory, "setup_logging", lambda: None)
    with pytest.raises(ValueError, match="ATTRIBUTION_CONFIG_FILE"):
        factory.create_app()


def test_lifespan_cancels_and_awaits_cleanup_task(monkeypatch):
    """Verify application shutdown stops the background cleanup task.

    Args:
        monkeypatch: Pytest fixture for replacing service dependencies.
    """
    provider = MagicMock()
    monkeypatch.setattr(factory, "setup_logging", lambda: provider)
    app = factory.create_app()

    async def exercise_lifespan():
        """Run the lifespan and observe cleanup task shutdown."""
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def cleanup(_aim):
            """Wait until application shutdown cancels this task.

            Args:
                _aim: Unused attributor manager supplied by the application.
            """
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        monkeypatch.setattr(factory, "_cleanup_expired_attributors", cleanup)

        async with app.router.lifespan_context(app):
            task = app.state.cleanup_task
            await asyncio.wait_for(started.wait(), timeout=1)
            assert task.done() is False

        assert stopped.is_set() is True
        assert task.done() is True
        assert task.cancelled() is True

    asyncio.run(exercise_lifespan())
    provider.force_flush.assert_called_once_with(timeout_millis=5000)
    provider.shutdown.assert_called_once_with()


def test_http_logging_omits_bodies_and_preserves_request_handling(app, client):
    """Verify HTTP logs omit payloads without changing request handling."""

    @app.post("/_test/logging")
    async def echo_body(request: Request):
        """Return the request body to exercise the successful response path."""
        return Response(content=await request.body())

    @app.post("/_test/logging-error")
    async def fail_after_reading_body(request: Request):
        """Raise after reading the body to exercise exception logging."""
        await request.body()
        raise RuntimeError("expected test failure")

    messages = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    payload = b"private-request-body\r\nforged-log-entry"
    try:
        response = client.post("/_test/logging", content=payload)
        error_response = client.post("/_test/logging-error", content=payload)
    finally:
        logger.remove(sink_id)

    assert response.content == payload
    assert error_response.status_code == 500
    assert error_response.json() == {"detail": "Internal Server Error"}

    logged = "".join(map(str, messages))
    assert "private-request-body" not in logged
    assert "forged-log-entry" not in logged
    assert "REQUEST: POST /_test/logging" in logged
    assert "RESPONSE: POST /_test/logging Status: 200" in logged
    assert "Error handling POST /_test/logging-error" in logged
    assert "expected test failure" in logged


def test_app_and_openapi_version_follow_literal(app):
    assert app.version == __version__
    assert app.openapi()["info"]["version"] == __version__
