# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import subprocess
import sys
from unittest.mock import MagicMock

from fastapi import Request, Response
from loguru import logger

import app.factory as factory


def test_importing_factory_does_not_initialize_rules():
    code = """
import attribution_lib

assert attribution_lib.initialized is False
import app.factory
assert attribution_lib.initialized is False
"""

    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_lifespan_cancels_and_awaits_cleanup_task(monkeypatch):
    """Verify application shutdown stops the background cleanup task.

    Args:
        monkeypatch: Pytest fixture for replacing service dependencies.
    """
    provider = MagicMock()
    monkeypatch.setattr(factory, "setup_logging", lambda: provider)
    monkeypatch.setattr(factory, "setup_rules", lambda: None)
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
