# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB defaults its per-connection `threads` to the host's full CPU count
and `memory_limit` to ~80% of system RAM. Each attributor gets its own
private in-memory DuckDB connection (see `app/attributor.py`), and that
dataset is always tiny (one job's worth of events). It never benefits
from intra-query parallelism, so the host-sized defaults are pure
per-attributor overhead that multiplies with concurrent load. This is
especially bad in Kubernetes, where a pod's CPU *limit* is often invisible
to DuckDB's host-core-count detection, so a 2-core-limited pod on a
64-core node still gets threads=64 per connection.

`attribution_lib.events.EventManager` now accepts threads/memory_limit
directly (see `attribution-lib`'s `events.py`), so this is wired in as a
plain constructor argument. No monkeypatching required.
"""

import os

import pytest

from tests.conftest import make_up_attributor_info, post_attributor


def test_attributor_duckdb_connection_has_bounded_resources(app, client):
    info = make_up_attributor_info(id="duckdb-limits-1")
    att_id = post_attributor(client, info)

    wrapper = app.state.aim.attributors[att_id]
    conn = wrapper.attributor.event_stream.conn

    (threads,) = conn.execute("SELECT current_setting('threads')").fetchone()
    assert threads == 1, f"expected a single-threaded per-attributor DuckDB connection, got threads={threads}"

    (memory_limit,) = conn.execute("SELECT current_setting('memory_limit')").fetchone()
    # DuckDB reports this as a human string (e.g. "128.0 MiB"); just confirm
    # it's nowhere near the multi-GiB host-RAM-based default.
    assert "GiB" not in memory_limit, f"expected a small bounded memory_limit, got {memory_limit!r}"


def test_disk_spill_disabled_by_default(app, client):
    """This service is primarily deployed in containers with little or no
    writable disk, so exceeding memory_limit should fail fast and be
    catchable (see test_event_storage_overflow.py) rather than silently
    spill to a local .tmp directory."""
    info = make_up_attributor_info(id="duckdb-limits-2")
    att_id = post_attributor(client, info)

    conn = app.state.aim.attributors[att_id].attributor.event_stream.conn
    (temp_dir,) = conn.execute("SELECT current_setting('temp_directory')").fetchone()
    assert temp_dir == "", f"expected disk spill disabled by default, got temp_directory={temp_dir!r}"


def test_disk_spill_can_be_enabled_via_env_var(app, client, monkeypatch):
    """Deployments where local disk can be assumed (e.g. bare metal) can opt
    back into DuckDB's own default graceful-degradation-via-disk-spill
    behavior."""
    monkeypatch.setenv("ATTRIBUTOR_ALLOW_DISK_SPILL", "true")
    info = make_up_attributor_info(id="duckdb-limits-3")
    att_id = post_attributor(client, info)

    conn = app.state.aim.attributors[att_id].attributor.event_stream.conn
    (temp_dir,) = conn.execute("SELECT current_setting('temp_directory')").fetchone()
    assert temp_dir != "", "expected disk spill enabled when ATTRIBUTOR_ALLOW_DISK_SPILL=true"


def test_memory_limit_can_be_configured_via_env_var(app, client, monkeypatch):
    """The 128MB per-attributor default is a guess at a reasonable ceiling,
    not a value every deployment should be stuck with; ATTRIBUTOR_DUCKDB_MEMORY_LIMIT
    lets it be tuned without a code change."""
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "256MB")
    info = make_up_attributor_info(id="duckdb-limits-4")
    att_id = post_attributor(client, info)

    conn = app.state.aim.attributors[att_id].attributor.event_stream.conn
    (memory_limit,) = conn.execute("SELECT current_setting('memory_limit')").fetchone()
    value, unit = memory_limit.split()
    assert unit == "MiB", f"expected a MiB-scale memory_limit, got {memory_limit!r}"
    assert float(value) == pytest.approx(256_000_000 / 1024**2, rel=0.01)


def test_spill_dir_is_private_under_configured_root(app, client, monkeypatch, tmp_path):
    """Each attributor spills into its own directory under
    ATTRIBUTOR_DUCKDB_TEMP_ROOT: attributors sharing DuckDB's default `.tmp`
    overwrite each other's fixed-name spill files."""
    monkeypatch.setenv("ATTRIBUTOR_ALLOW_DISK_SPILL", "true")
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_TEMP_ROOT", str(tmp_path))
    infos = [make_up_attributor_info(id=f"duckdb-limits-temp-{i}") for i in range(2)]
    temp_dirs = []
    for info in infos:
        att_id = post_attributor(client, info)
        conn = app.state.aim.attributors[att_id].attributor.event_stream.conn
        (temp_dir,) = conn.execute("SELECT current_setting('temp_directory')").fetchone()
        temp_dirs.append(temp_dir)
    assert temp_dirs[0] != temp_dirs[1]
    assert all(os.path.dirname(d) == str(tmp_path) for d in temp_dirs), temp_dirs

    # A rejected duplicate must not leave its own spill dir behind.
    assert client.post("/v1/attributor", content=infos[0].model_dump_json()).status_code == 409
    assert sorted(str(p) for p in tmp_path.iterdir()) == sorted(temp_dirs)


@pytest.mark.parametrize(("enabled", "limit"), [(True, "64MiB"), (True, ""), (True, "   "), (False, "invalid")])
def test_spill_capacity_setting(app, client, monkeypatch, tmp_path, enabled, limit):
    monkeypatch.setenv("ATTRIBUTOR_ALLOW_DISK_SPILL", str(enabled).lower())
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_TEMP_ROOT", str(tmp_path))
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", limit)
    att_id = post_attributor(client, make_up_attributor_info(id="spill-capacity"))
    conn = app.state.aim.attributors[att_id].attributor.event_stream.conn
    (directory,) = conn.execute("SELECT current_setting('temp_directory')").fetchone()
    if enabled:
        assert os.path.dirname(directory) == str(tmp_path)
        if limit.strip():
            assert conn.execute("SELECT current_setting('max_temp_directory_size')").fetchone()[0] == "64.0 MiB"
    else:
        assert directory == ""
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", ""),
        ("ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "   "),
        ("ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "128"),
        ("ATTRIBUTOR_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", "1Gi"),
    ],
)
def test_invalid_size_fails_creation_without_registering_or_leaking(monkeypatch, tmp_path, version, variable, value):
    from fastapi.testclient import TestClient

    from app.factory import create_app

    monkeypatch.setenv("ATTRIBUTOR_ALLOW_DISK_SPILL", "true")
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_TEMP_ROOT", str(tmp_path))
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        # Keep the request-time diagnostic if configuration changes after boot.
        monkeypatch.setenv(variable, value)
        response = client.post(
            f"/{version}/attributor",
            content=make_up_attributor_info(version=version, id="invalid-size").model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 500
        assert response.json() == {
            "detail": (
                f"Invalid server setting {variable}. Use a DuckDB size such as 128MB or 1GiB. "
                "Correct the service environment and restart the service."
            )
        }
        assert app.state.aim.attributors == {}
        assert list(tmp_path.iterdir()) == []


def test_unrelated_initialization_error_stays_generic(client, monkeypatch, version):
    from unittest.mock import Mock

    from app.attributor import AttributorWrapper

    monkeypatch.setattr(AttributorWrapper, "initialize", Mock(side_effect=RuntimeError("internal details")))
    response = client.post(f"/{version}/attributor", content=make_up_attributor_info(version=version).model_dump_json())
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}


@pytest.mark.parametrize(
    ("variable", "value", "spill", "invalid"),
    [
        ("ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "", False, True),
        ("ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "128", True, True),
        ("ATTRIBUTOR_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", "1Gi", True, True),
        ("ATTRIBUTOR_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", "1Gi", False, False),
        ("ATTRIBUTOR_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", "64MiB", True, False),
        ("ATTRIBUTOR_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", "", True, False),
    ],
)
def test_storage_configuration_checked_before_startup(monkeypatch, tmp_path, variable, value, spill, invalid):
    from fastapi.testclient import TestClient

    from app.factory import create_app

    monkeypatch.setenv(variable, value)
    monkeypatch.setenv("ATTRIBUTOR_ALLOW_DISK_SPILL", str(spill).lower())
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_TEMP_ROOT", str(tmp_path))
    app = create_app()
    if invalid:
        with pytest.raises(RuntimeError, match=variable), TestClient(app):
            pytest.fail("Invalid configuration must not finish startup")
        assert not hasattr(app.state, "cleanup_task")
    else:
        with TestClient(app) as client:
            assert client.get("/v2/healthz").status_code == 200
            assert not list(tmp_path.iterdir())  # The startup probe is already closed.
    assert not app.state.aim.attributors
    assert not list(tmp_path.iterdir())
