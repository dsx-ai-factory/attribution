# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for the shared, read-only rule/event-type lookup database.

EventManager used to copy the full static rule-name and event-type name set
into a private DuckDB table on every instance. These tests prove that (a)
multiple EventManagers for the same rules_engine actually share the
underlying data instead of duplicating it, (b) lookups still work correctly
through the shared+local-overlay view, and (c) different rules_engine
instances get independent, non-interfering lookup data.
"""

import duckdb
import pytest

import attribution_lib
from attribution_lib.events import EventManager
from attribution_lib.exceptions import UndeclaredRuleError
from attribution_lib.rules_engine import RuleLoader


def test_multiple_managers_share_the_same_lookup_file():
    """Two EventManagers for the same rules_engine must ATTACH the same
    on-disk shared lookup database rather than each building their own."""
    rules_engine = RuleLoader().build()
    mgr1 = EventManager(rules_engine)
    mgr2 = EventManager(rules_engine)
    try:
        (path1,) = mgr1._conn.execute("SELECT path FROM duckdb_databases() WHERE database_name='lookup'").fetchone()
        (path2,) = mgr2._conn.execute("SELECT path FROM duckdb_databases() WHERE database_name='lookup'").fetchone()
        assert path1 == path2
    finally:
        mgr1.close()
        mgr2.close()


def test_closing_one_manager_does_not_delete_a_shared_lookup_file():
    """Only `from_snapshot()`'s one-off lookup db is owned by a single
    EventManager and safe to delete on close(). A real rules_engine's
    lookup db is shared by every EventManager constructed with it, so
    closing one must not unlink the file a sibling instance still has
    ATTACHed -- checked on disk directly, since a POSIX unlink of an
    already-open file wouldn't break mgr2's existing file descriptor
    and so wouldn't be caught by mgr2 continuing to work."""
    import os

    rules_engine = RuleLoader().build()
    mgr1 = EventManager(rules_engine)
    mgr2 = EventManager(rules_engine)
    try:
        (lookup_path,) = mgr2._conn.execute(
            "SELECT path FROM duckdb_databases() WHERE database_name='lookup'"
        ).fetchone()
        mgr1.close()
        assert os.path.exists(lookup_path), "mgr1.close() must not delete a lookup file mgr2 still shares"
    finally:
        mgr2.close()


def test_multiple_managers_share_the_same_python_cache_dicts():
    """Two EventManagers for the same rules_engine must share the *same*
    Python-level id/str dict objects for the shared tier, not just equal
    values in separately-copied dicts -- that copying is exactly the
    per-instance duplication this shared cache exists to avoid.
    """
    rules_engine = RuleLoader().build()
    mgr1 = EventManager(rules_engine)
    mgr2 = EventManager(rules_engine)
    try:
        assert mgr1._rules._shared_id_cache is mgr2._rules._shared_id_cache
        assert mgr1._rules._shared_str_cache is mgr2._rules._shared_str_cache
        assert mgr1._types._shared_id_cache is mgr2._types._shared_id_cache
        assert mgr1._types._shared_str_cache is mgr2._types._shared_str_cache
    finally:
        mgr1.close()
        mgr2.close()


def test_rule_names_resolve_consistently_across_instances():
    # `RuleLoader().build()` alone yields an empty rules_engine, which would
    # make the loops below iterate zero times and pass vacuously regardless
    # of whether ids actually resolve consistently. Use the populated engine
    # instead.
    rules_engine = attribution_lib.get_rules_engine()
    assert rules_engine.all_rule_names, "rules_engine.all_rule_names must be non-empty for this test to be meaningful"
    assert rules_engine.all_event_types, "rules_engine.all_event_types must be non-empty for this test to be meaningful"

    mgr1 = EventManager(rules_engine)
    mgr2 = EventManager(rules_engine)
    try:
        for name in rules_engine.all_rule_names:
            assert mgr1.rules.get_id(name) == mgr2.rules.get_id(name)
        for name in rules_engine.all_event_types:
            assert mgr1.types.get_id(name) == mgr2.types.get_id(name)
    finally:
        mgr1.close()
        mgr2.close()


def test_shared_lookup_is_not_writable_by_construction():
    """The shared table is ATTACHed READ_ONLY. Writing to it directly
    (bypassing `preload()`, which only ever targets the local overlay) must fail."""
    rules_engine = RuleLoader().build()
    mgr = EventManager(rules_engine)
    try:
        with pytest.raises(duckdb.Error, match="read-only"):
            mgr._conn.execute("INSERT INTO lookup.rules (id, name) VALUES (999999999, 'should_not_be_allowed')")
    finally:
        mgr.close()


def test_post_construction_preload_is_instance_local():
    """A name registered via `preload_rule_names()` on one instance must not
    leak into a second, independently constructed instance for the same
    rules_engine. It only affects the caller's own local overlay."""
    rules_engine = RuleLoader().build()
    mgr1 = EventManager(rules_engine)
    mgr2 = EventManager(rules_engine)
    try:
        mgr1.preload_rule_names({"only_on_mgr1"})
        assert mgr1.rules.get_id("only_on_mgr1") is not None
        with pytest.raises(UndeclaredRuleError):
            mgr2.rules.get_id("only_on_mgr1")

        # Sharper than the above: mgr1/mgr2 share the same _shared_id_cache
        # object (see test_multiple_managers_share_the_same_python_cache_dicts),
        # so a preload() that accidentally wrote into that shared dict instead
        # of its own local one would corrupt it for every instance, not just
        # leak into mgr2 specifically. Assert directly against the shared dict.
        assert "only_on_mgr1" not in mgr1._rules._shared_id_cache
    finally:
        mgr1.close()
        mgr2.close()


def test_different_rules_engines_get_independent_lookups():
    engine_a = RuleLoader().build()
    engine_b = RuleLoader().build()
    mgr_a = EventManager(engine_a)
    mgr_b = EventManager(engine_b)
    try:
        (path_a,) = mgr_a._conn.execute("SELECT path FROM duckdb_databases() WHERE database_name='lookup'").fetchone()
        (path_b,) = mgr_b._conn.execute("SELECT path FROM duckdb_databases() WHERE database_name='lookup'").fetchone()
        assert path_a != path_b
    finally:
        mgr_a.close()
        mgr_b.close()


def test_joins_against_rules_view_see_both_tiers():
    """Raw SQL elsewhere in the codebase (`event_helpers.py`, `filter_rules.py`)
    joins against the plain `rules`/`event_types` names. Both the shared
    (construction-time) and local (`preload_rule_names`) tiers must be visible
    through that unqualified name."""
    rules_engine = attribution_lib.get_rules_engine()
    mgr = EventManager(rules_engine)
    try:
        mgr.preload_rule_names({"late_addition_rule"})
        shared_name = next(iter(rules_engine.all_rule_names))
        rows = {
            row[0]
            for row in mgr._conn.execute(
                "SELECT name FROM rules WHERE name IN (?, ?)", [shared_name, "late_addition_rule"]
            ).fetchall()
        }
        assert rows == {shared_name, "late_addition_rule"}
    finally:
        mgr.close()


class TestSharedLookupTmpdirForkSafety:
    """`_get_shared_lookup_tmpdir()`'s `atexit` cleanup is registered once per
    process and, along with the module globals it reads, inherited by any
    `os.fork()`'d child (e.g. a pre-fork server's worker process). A child
    must not delete the directory on its own exit: it didn't create it, and
    the parent or sibling workers may still have `EventManager`s ATTACHed to
    files inside it.
    """

    def test_forked_child_does_not_delete_shared_tmpdir(self, monkeypatch, tmp_path):
        import os

        import attribution_lib.events as events_module

        fake_tmpdir = str(tmp_path / "shared_lookup")
        os.makedirs(fake_tmpdir)
        monkeypatch.setattr(events_module, "_shared_lookup_tmpdir", fake_tmpdir)
        monkeypatch.setattr(events_module, "_shared_lookup_tmpdir_owner_pid", os.getpid())

        pid = os.fork()
        if pid == 0:
            # Child: simulate its inherited atexit handler firing on its own exit.
            try:
                events_module._cleanup_shared_lookup_tmpdir()
            finally:
                os._exit(0)

        _, status = os.waitpid(pid, 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, f"child exited abnormally: {status}"

        # The child must not have deleted the directory the parent still owns.
        assert os.path.isdir(fake_tmpdir)

        # The actual owning process's cleanup must still work.
        events_module._cleanup_shared_lookup_tmpdir()
        assert not os.path.exists(fake_tmpdir)
