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


def test_rule_names_resolve_consistently_across_instances():
    rules_engine = RuleLoader().build()
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
    """The shared table is ATTACHed READ_ONLY -- writing to it directly
    (bypassing preload(), which only ever targets the local overlay) must fail."""
    rules_engine = RuleLoader().build()
    mgr = EventManager(rules_engine)
    try:
        with pytest.raises(duckdb.Error):
            mgr._conn.execute("INSERT INTO lookup.rules (name) VALUES ('should_not_be_allowed')")
    finally:
        mgr.close()


def test_post_construction_preload_is_instance_local():
    """A name registered via preload_rule_names() on one instance must not
    leak into a second, independently constructed instance for the same
    rules_engine -- it only affects the caller's own local overlay."""
    rules_engine = RuleLoader().build()
    mgr1 = EventManager(rules_engine)
    mgr2 = EventManager(rules_engine)
    try:
        mgr1.preload_rule_names({"only_on_mgr1"})
        assert mgr1.rules.get_id("only_on_mgr1") is not None
        with pytest.raises(Exception):
            mgr2.rules.get_id("only_on_mgr1")
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
    """Raw SQL elsewhere in the codebase (event_helpers.py, filter_rules.py)
    joins against the plain `rules`/`event_types` names -- both the shared
    (construction-time) and local (preload_rule_names) tiers must be visible
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
