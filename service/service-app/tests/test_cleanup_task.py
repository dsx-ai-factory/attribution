# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import pytest

import app.factory as factory_module
from app.attributor_manager import AttributorManager


def test_cleanup_task_survives_exceptions(monkeypatch):
    """A single failed cleanup pass must not permanently stop the background
    cleanup loop -- otherwise expired attributors accumulate forever."""
    call_count = 0
    aim = AttributorManager()

    def flaky_cleanup():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated transient error")

    monkeypatch.setattr(aim, "cleanup_expired_attributors", flaky_cleanup)

    real_sleep = asyncio.sleep

    async def fast_sleep(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(factory_module.asyncio, "sleep", fast_sleep)

    async def run():
        task = asyncio.create_task(factory_module._cleanup_expired_attributors(aim))
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())

    assert call_count >= 2, "cleanup loop stopped after the first exception instead of continuing"


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX SIGTERM")
@pytest.mark.parametrize("failure", ["none", "transient", "persistent"])
def test_sigterm_cleanup(tmp_path, monkeypatch, failure):
    import signal
    import subprocess
    import sys
    import time

    root = tmp_path / "spill"
    root.mkdir()
    monkeypatch.setenv("ATTRIBUTOR_ALLOW_DISK_SPILL", "true")
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_TEMP_ROOT", str(root))
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "16MB")
    ready, report, log = (tmp_path / name for name in ("ready", "report", "server.log"))
    script = r"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from unittest.mock import Mock, patch
import uvicorn
from app.factory import create_app
from tests.conftest import make_up_attributor_info

ready, report = map(Path, sys.argv[1:3])
failure = sys.argv[3]
app = create_app()
aim = app.state.aim
lifespan = app.router.lifespan_context

@asynccontextmanager
async def checked_lifespan(app):
    try:
        async with lifespan(app):
            retired = aim.add_attributor(make_up_attributor_info(id='retired'))
            if failure != 'none':
                with patch('shutil.rmtree', side_effect=PermissionError('injected')):
                    try:
                        aim.delete_attributor(retired.id)
                    except PermissionError:
                        pass
                assert aim._pending_cleanup
            active = aim.add_attributor(make_up_attributor_info(id='active'))
            active._event_manager._conn.execute("CREATE TABLE big AS SELECT repeat('x', 1000000) FROM range(50)")
            assert list(Path(active._event_manager._spill_dir).iterdir())
            if failure == 'persistent':
                retired._event_manager.close = Mock(side_effect=PermissionError('injected'))
            yield
    finally:
        assert not aim.attributors
        assert active._event_manager._closed
        assert bool(aim._pending_cleanup) == (failure == 'persistent')
        report.write_text('checked')

class Server(uvicorn.Server):
    async def startup(self, sockets=None):
        await super().startup(sockets)
        if self.started:
            ready.write_text('ready')

app.router.lifespan_context = checked_lifespan
Server(uvicorn.Config(app, host='127.0.0.1', port=0, lifespan='on')).run()
"""
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(ready), str(report), failure], stdout=output, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 30
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ready.exists(), log.read_text()
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert report.read_text() == "checked", log.read_text()
    assert len(list(root.iterdir())) == (1 if failure == "persistent" else 0)
    message = "Application shutdown failed" if failure == "persistent" else "Application shutdown complete"
    assert message in log.read_text()
