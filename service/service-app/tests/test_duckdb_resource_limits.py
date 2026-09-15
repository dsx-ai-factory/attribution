# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
