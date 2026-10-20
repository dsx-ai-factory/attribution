# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import atexit
import contextlib
import json
import os
import shutil
import tempfile
import threading
import weakref
import zlib
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypeAlias

import duckdb
from loguru import logger

from attribution_lib.exceptions import (
    EventStorageOutOfMemoryError,
    InvalidEventStorageSizeError,
    InvalidSnapshotError,
    SnapshotSchemaVersionError,
    StableIdCollisionError,
    UndeclaredEventTypeError,
    UndeclaredRuleError,
    UnexpectedEmptyQueryResultError,
)

if TYPE_CHECKING:
    from attribution_lib.rules_engine import RuleEngine

from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import (
    Format as ObsFormat,
    Source as ObsSource,
)
from attribution_lib.resources.resource_set import serialize_resource_set
from attribution_lib.serialization import serialize
from attribution_lib.sources import Source

if TYPE_CHECKING:
    from attribution_lib.rules_engine import RuleEngine


class DemotionTypes(str, Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

    AFTER_MFU_DEMOTION = auto()
    NOT_NEAR_WATCHDOG_EVENT_DEMOTION = auto()
    BEFORE_DATALOADER_DEMOTION = auto()


class TableName(str, Enum):
    EVENT_TYPES = "event_types"
    RULES = "rules"
    CONTEXTS = "contexts"
    OBSERVATION_SOURCES = "obs_sources"
    OBSERVATION_FORMATS = "obs_formats"
    OBSERVATION_CONTEXTS = "obs_contexts"
    EVENTS = "events"


# SHARED event type constants
BAD_PAIR_EVENT_TYPES = ["Application.bad_pair.partial", "Application.bad_pair.complete"]


@contextlib.contextmanager
def _translate_out_of_memory():
    """Re-raise duckdb's OutOfMemoryException as EventStorageOutOfMemoryError.

    Deployments with no writable disk construct EventManager with
    allow_disk_spill=False, so exceeding memory_limit raises instead of
    DuckDB silently spilling to a local .tmp directory. Callers should be
    able to catch a stable, library-level exception type rather than needing
    to know about duckdb's own exception hierarchy.
    """
    try:
        yield
    except duckdb.OutOfMemoryException as e:
        raise EventStorageOutOfMemoryError(str(e)) from e


@contextlib.contextmanager
def _atomic(conn: duckdb.DuckDBPyConnection):
    """Run a block of statements as one transaction, rolling back on any
    failure, including a failing `conn.commit()` itself, so a mid-block
    error can't leave a paired DROP/CREATE or DELETE/INSERT half-applied.
    Callers should nest this *inside* `_translate_out_of_memory()` (i.e.
    `with _translate_out_of_memory(), _atomic(conn):`) so an
    `OutOfMemoryException` raised by `commit()` still gets translated to
    `EventStorageOutOfMemoryError`: that only happens if this context
    manager's own `__exit__` (where `commit()` runs) exits *before*
    `_translate_out_of_memory()`'s does.
    """
    conn.begin()
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _get_or_create_id(conn: duckdb.DuckDBPyConnection, table: TableName, name: str) -> int:
    """Get or create an ID for a string in a lookup table. Returns None if name is None."""
    with _translate_out_of_memory():
        # Insert if not exists, then get the ID
        conn.execute(f"INSERT INTO {table.value} (name) VALUES (?) ON CONFLICT (name) DO NOTHING", [name])
        return execute_fetch_one(conn, f"SELECT id FROM {table.value} WHERE name=?", [name])


def sql_placeholders(count: int) -> str:
    """Generate SQL placeholder string for parameterized IN clauses."""
    return ",".join(["?"] * count)


_SNAPSHOT_LOOKUP_REFERENCES = (
    ("observation_contexts", "source", "obs_sources"),
    ("observation_contexts", "format", "obs_formats"),
    ("events", "event_type", "event_types"),
    ("events", "source_rule", "rules"),
)


def _validate_snapshot_lookup_references(conn: duckdb.DuckDBPyConnection) -> None:
    """Reject a restored snapshot whose `events`/`observation_contexts` rows
    reference a lookup id that doesn't resolve to any name in the restored
    `event_types`/`rules`/`obs_sources`/`obs_formats` tables.

    Normal writes always go through `_ReadOnlyLookup.get_id()`, which only
    ever returns an id for a name actually present in the lookup tables, so
    this can't happen for data written through the normal API. But
    `from_snapshot()` restores `events`/`observation_contexts` via a raw
    `INSERT ... SELECT * FROM read_parquet(...)`, bypassing that path
    entirely, so a corrupted or hand-edited snapshot could otherwise import
    silently-dangling ids.
    """
    with _translate_out_of_memory():
        for table, column, lookup_table in _SNAPSHOT_LOOKUP_REFERENCES:
            rows = conn.execute(
                f"SELECT DISTINCT {column} FROM {table} "
                f"WHERE {column} IS NOT NULL AND {column} NOT IN (SELECT id FROM {lookup_table})"
            ).fetchall()
            if rows:
                raise InvalidSnapshotError(table, column, [row[0] for row in rows])


def _assert_no_id_collision(conn: duckdb.DuckDBPyConnection, view_name: str) -> None:
    """Raise `StableIdCollisionError` if any id in `view_name` resolves to
    more than one distinct name.

    `from_snapshot()`'s restore of `obs_sources`/`obs_formats` local-overlay
    rows is a raw SQL `INSERT` that only filters out rows whose name already
    resolves, bypassing `_ReadOnlyLookup.preload()`'s per-name collision
    check entirely. A snapshot row whose id collides with an existing
    shared-tier name under a different name would otherwise import silently,
    aliasing two names to the same id.
    """
    with _translate_out_of_memory():
        rows = conn.execute(f"SELECT id, name FROM {view_name}").fetchall()
    names_by_id: dict[int, set[str]] = {}
    for id_val, name in rows:
        names = names_by_id.setdefault(id_val, set())
        names.add(name)
        if len(names) > 1:
            other = next(iter(names - {name}))
            raise StableIdCollisionError(view_name, id_val, name, other)


def _sql_string_literal(value: object) -> str:
    """Quote `value` (typically a `Path`) as a single-quoted SQL string
    literal, doubling any embedded single quotes. DuckDB has no parameter
    binding for identifiers/file paths passed to `ATTACH`/`COPY`/
    `read_parquet()`, so these have to be interpolated directly; a bare
    f-string would break (or, for a maliciously crafted path, inject) on a
    path containing a single quote, which is a valid filename character on
    Linux.
    """
    return "'" + str(value).replace("'", "''") + "'"


def _stable_id(name: str) -> int:
    """Content-addressed id for a lookup-table (event_types/rules/obs_sources/
    obs_formats) name: a function of the name's own bytes only, stable across
    processes and independent of insertion order or what other names are
    present in the same build.

    Python's built-in hash() is randomized per-process for strings
    (PYTHONHASHSEED) specifically to prevent this kind of reliance, so
    it cannot be used here; zlib.crc32 is not randomized. Masked into
    DuckDB INTEGER's signed 32-bit range. A collision between two
    different names is detected by preload(), which raises rather than
    silently misattributing.

    This is what lets every EventManager for a given rules_engine share one
    read-only lookup database (see _build_shared_lookup_db below) and what
    lets from_snapshot() restore a snapshot's lookup rows directly, without
    needing them to match whatever a freshly built EventManager happens to
    assign: a name's id is the same regardless of which process, in what
    order, or how many other names were computed alongside it.
    """
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


# Per-connection schema for the tables that are genuinely unique to this
# EventManager instance. event_types/rules/obs_sources/obs_formats live in a
# shared, read-only ATTACHed database instead (see _build_shared_lookup_db /
# _LOCAL_OVERLAY_SCHEMA below), since their content is static for the
# process's lifetime and duplicating it per instance is pure overhead. FK
# references to those tables are dropped here since they now live in a
# different attached catalog; the _ReadOnlyLookup class is the sole path by
# which ids for those tables are produced (via get_id()), so referential
# integrity is enforced there instead of by the database.
SCHEMA = """
CREATE SEQUENCE contexts_seq START 1;
CREATE TABLE contexts (
    id INTEGER PRIMARY KEY DEFAULT nextval('contexts_seq'),
    name VARCHAR UNIQUE NOT NULL
);

CREATE TABLE observation_contexts (
    id BIGINT PRIMARY KEY,
    time_start TIMESTAMPTZ,
    time_end TIMESTAMPTZ,
    source INTEGER,
    format INTEGER,
    timezone VARCHAR
);

CREATE SEQUENCE events_seq START 1;
CREATE TABLE events (
    id INTEGER DEFAULT nextval('events_seq'),
    context_id INTEGER NOT NULL REFERENCES contexts(id),
    event_type INTEGER NOT NULL,
    timestamp TIMESTAMPTZ,
    status INTEGER DEFAULT 1,
    line_num INTEGER,
    source_datum VARCHAR,
    source_rule INTEGER,
    resources VARCHAR,
    obs_context_id BIGINT REFERENCES observation_contexts(id),
    payload JSON,
    PRIMARY KEY (id, context_id)
);

-- Index for time-range queries (most common analytical pattern)
CREATE INDEX idx_events_ctx_timestamp ON events(context_id, timestamp);
"""

_COLS = "context_id,event_type,timestamp,status,line_num,source_datum,source_rule,resources,obs_context_id,payload"

# Schema for the shared, read-only lookup database ATTACHed by every
# EventManager for a given rules_engine. Built once (see
# _build_shared_lookup_db) and never written to again. ids are
# content-addressed (_stable_id), not sequence-assigned, so no DEFAULT clause.
_SHARED_LOOKUP_SCHEMA = """
CREATE TABLE event_types (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE NOT NULL);
CREATE TABLE rules (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE NOT NULL);
CREATE TABLE obs_sources (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE NOT NULL);
CREATE TABLE obs_formats (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE NOT NULL);
"""

# Small per-instance overlay for names registered after construction
# (`preload_rule_names`/`preload_event_types`) or restored from a snapshot. ids
# are content-addressed (`_stable_id`) exactly like the shared tier, so a
# name's id is the same whichever tier it ends up in. The two tiers can
# only "collide" by holding an identical (id, name) row for the same name,
# which `preload()` and `from_snapshot()` both avoid by skipping names already
# resolvable from the shared tier. A UNION ALL view combines both tiers
# under the plain unqualified table name (`rules`, `event_types`, ...) so
# existing raw-SQL joins/queries elsewhere in the codebase keep working
# unmodified.
_LOCAL_OVERLAY_SCHEMA = """
CREATE TABLE local_event_types (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE NOT NULL);
CREATE TABLE local_rules (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE NOT NULL);
CREATE TABLE local_obs_sources (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE NOT NULL);
CREATE TABLE local_obs_formats (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE NOT NULL);

CREATE VIEW event_types AS SELECT id, name FROM lookup.event_types UNION ALL SELECT id, name FROM local_event_types;
CREATE VIEW rules AS SELECT id, name FROM lookup.rules UNION ALL SELECT id, name FROM local_rules;
CREATE VIEW obs_sources AS SELECT id, name FROM lookup.obs_sources UNION ALL SELECT id, name FROM local_obs_sources;
CREATE VIEW obs_formats AS SELECT id, name FROM lookup.obs_formats UNION ALL SELECT id, name FROM local_obs_formats;
"""

# name -> local overlay table, for the 2 lookup tables from_snapshot() still
# restores via the local-overlay-merge path (see from_snapshot() for why
# rules/event_types don't need this: their names are folded into the shared
# lookup db at construction time instead, via _SnapshotRuleNames).
_LOOKUP_LOCAL_TABLES = {
    "obs_sources": "local_obs_sources",
    "obs_formats": "local_obs_formats",
}

# table name -> (name -> id, id -> name), the Python-level mirror of a shared
# lookup table's rows. Immutable once built; shared by reference across every
# `_ReadOnlyLookup` for a given rules_engine (see `_build_shared_lookup_db`),
# never mutated after `EventManager.__init__` returns.
_LookupPythonCache: TypeAlias = dict[str, tuple[dict[str, int], dict[int, str]]]

_shared_lookup_lock = threading.Lock()
_shared_lookup_cache: "weakref.WeakKeyDictionary[RuleEngine, tuple[Path, _LookupPythonCache]]" = (
    weakref.WeakKeyDictionary()
)
_shared_lookup_tmpdir: Optional[str] = None
# PID of the process that created `_shared_lookup_tmpdir`. A `os.fork()`'d
# child (e.g. a pre-fork server's worker process) inherits both this global
# and the `atexit` registration below, but must not delete the directory on
# its own exit: it didn't create it, and the parent or sibling workers may
# still have EventManagers ATTACHed to files inside it. Only the owning PID
# performs cleanup; a forked child's exit is a no-op here.
_shared_lookup_tmpdir_owner_pid: Optional[int] = None
# Monotonically increasing; NOT len(_shared_lookup_cache), which can shrink
# as entries are garbage-collected and would then produce a reused filename
# for an unrelated rules_engine, colliding with that file's existing schema.
_shared_lookup_counter = 0


def _cleanup_shared_lookup_tmpdir() -> None:
    if _shared_lookup_tmpdir is not None and os.getpid() == _shared_lookup_tmpdir_owner_pid:
        shutil.rmtree(_shared_lookup_tmpdir, ignore_errors=True)


def _get_shared_lookup_tmpdir() -> str:
    global _shared_lookup_tmpdir, _shared_lookup_tmpdir_owner_pid
    if _shared_lookup_tmpdir is None:
        _shared_lookup_tmpdir = tempfile.mkdtemp(prefix="attribution_lib_shared_lookup_")
        _shared_lookup_tmpdir_owner_pid = os.getpid()
        atexit.register(_cleanup_shared_lookup_tmpdir)
    return _shared_lookup_tmpdir


def _bulk_insert_names(conn: duckdb.DuckDBPyConnection, table: str, names: set[str]) -> dict[str, int]:
    """Seed `table` with `names`, each keyed by its `_stable_id()`. Returns the
    computed `name -> id` mapping so callers can reuse it (e.g. to warm a
    Python-level cache) without recomputing `_stable_id()`.

    `_stable_id()` is a 31-bit hash, so two distinct names can in principle
    collide. Detected here up front, before the INSERT, so a collision
    raises the library's own `StableIdCollisionError` naming both names
    instead of surfacing as an opaque DuckDB primary-key constraint
    violation (this table's initial seed is the only bulk multi-row insert
    of `_stable_id()`-derived ids; `preload()`'s incremental path already
    has its own collision check, see `_ReadOnlyLookup.preload`).
    """
    if not names:
        return {}
    name_by_id: dict[int, str] = {}
    for name in names:
        id_val = _stable_id(name)
        existing = name_by_id.get(id_val)
        if existing is not None:
            raise StableIdCollisionError(table, id_val, name, existing)
        name_by_id[id_val] = name
    name_list = list(names)
    values_clause = ",".join(["(?, ?)"] * len(name_list))
    params = [param for name in name_list for param in (_stable_id(name), name)]
    conn.execute(f"INSERT INTO {table} (id, name) VALUES {values_clause}", params)
    return {name: id_val for id_val, name in name_by_id.items()}


def _build_shared_lookup_db(rules_engine: "RuleEngine") -> tuple[Path, _LookupPythonCache]:
    """Build (once per rules_engine instance, cached) a small file-backed
    DuckDB database holding the event_types/rules/obs_sources/obs_formats
    lookup tables, fully preloaded from rules_engine and the observation
    source/format enums, plus a Python-level mirror of those same rows.
    Every EventManager constructed with this rules_engine ATTACHes the file
    read-only and passes the Python-level cache straight into its
    `_ReadOnlyLookup`s, instead of each instance separately rebuilding its
    own private copy of this static data (SQL round trip and Python dict
    alike).

    Both are cached together in a WeakKeyDictionary keyed by the
    rules_engine object itself (not id()) so a GC'd-and-reused id can never
    spuriously match a live, different rules_engine.
    """
    with _shared_lookup_lock:
        cached = _shared_lookup_cache.get(rules_engine)
        if cached is not None:
            return cached

        global _shared_lookup_counter
        _shared_lookup_counter += 1
        path = Path(_get_shared_lookup_tmpdir()) / f"lookup_{_shared_lookup_counter}.duckdb"
        conn = None
        try:
            conn = duckdb.connect(str(path))
            conn.execute(_SHARED_LOOKUP_SCHEMA)
            id_by_name_per_table = {
                "obs_sources": _bulk_insert_names(conn, "obs_sources", {m.value for m in ObsSource if m.value}),
                "obs_formats": _bulk_insert_names(conn, "obs_formats", {m.value for m in ObsFormat if m.value}),
                "rules": _bulk_insert_names(conn, "rules", set(rules_engine.all_rule_names)),
                "event_types": _bulk_insert_names(conn, "event_types", set(rules_engine.all_event_types)),
            }
            # Release the write lock before publishing for ATTACH ... READ_ONLY.
            conn.close()
            python_cache: _LookupPythonCache = {
                table: (id_by_name, {id_val: name for name, id_val in id_by_name.items()})
                for table, id_by_name in id_by_name_per_table.items()
            }
            result = (path, python_cache)
            _shared_lookup_cache[rules_engine] = result
            return result
        except BaseException:
            try:
                if conn is not None:
                    conn.close()
                path.unlink(missing_ok=True)
                Path(f"{path}.wal").unlink(missing_ok=True)
            except Exception:
                # The process-owned temp directory remains covered by atexit.
                logger.exception("Failed to remove incomplete lookup database {}", path)
            raise


class EventStream:
    """SQL-based event storage.

    Events are stored in DuckDB with auto-generated IDs. Schema:
        id INTEGER          - auto-generated unique event ID
        context_id INTEGER  - context ID
        event_type INTEGER  - event type ID (foreign key to event_types)
        timestamp TIMESTAMPTZ - timestamp with timezone
        status INTEGER      - 1=active, 0=inactive
        line_num INTEGER    - source line number
        source_datum VARCHAR - source datum string
        source_rule INTEGER - rule ID (foreign key to rules)
        resources VARCHAR   - resource identifier ("ANY" or hostname)
        obs_context_id BIGINT - observation context ID
        payload JSON        - event payload as JSON

    Use query_raw() for SELECT queries, execute() for mutations.
    """

    Adder: TypeAlias = Callable[[str, dict[str, Any], Source], None]

    def __init__(self, ctx_id: int, ctx_str: str, conn: duckdb.DuckDBPyConnection, mgr: "EventManager"):
        self._ctx_id = ctx_id
        self._ctx_str = ctx_str
        self._conn = conn
        self._mgr = mgr
        self._batch: list[tuple[str, dict[str, Any], Source]] = []
        self._batch_size_threshold = 500  # Auto-flush after this many events

    @property
    def ctx_id(self) -> int:
        """Context ID for use in SQL WHERE clauses."""
        return self._ctx_id

    @property
    def context_id(self) -> str:
        """Context ID as string."""
        return self._ctx_str

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Direct DuckDB connection for SQL queries."""
        return self._conn

    @property
    def mgr(self) -> "EventManager":
        """EventManager for accessing database."""
        return self._mgr

    def _flush(self) -> None:
        """Flush batched events to the database efficiently."""
        if not self._batch:
            return

        # Step 1: Collect all unique event types and rules
        event_types = set()
        rules = set()
        for event_type, _, source in self._batch:
            event_types.add(event_type)
            if source.rule:
                rules.add(source.rule)

        # Step 2: Resolve event type and rule name strings to integer IDs.
        # Both are read-only — get_id raises on unknown names.
        types_lookup = self._mgr._types
        type_ids: dict[str, int] = {t: types_lookup.get_id(t) for t in event_types}

        rules_lookup = self._mgr._rules
        rule_ids: dict[str, int] = {r: rules_lookup.get_id(r) for r in rules}

        # Step 3: Prepare all event rows for bulk insert
        event_rows = []
        for event_type, payload, source in self._batch:
            typ_id = type_ids.get(event_type)
            rule_id = rule_ids.get(source.rule) if source.rule else None
            # Serialize resources using Pydantic (returns "AllJobResources" or list of Resource models)
            serialized = serialize_resource_set(source.resources)
            if isinstance(serialized, str):
                resources_str = json.dumps(serialized)
            else:
                # Convert Pydantic models to dicts for JSON serialization
                resources_str = json.dumps([r.model_dump() for r in serialized])
            payload_json = json.dumps(
                {(k.value if isinstance(k, Enum) else k): serialize(v) for k, v in payload.items()},
                default=serialize,
            )

            event_rows.append(
                (
                    self._ctx_id,
                    typ_id,
                    source.time,
                    source.line_num,
                    source.datum,
                    rule_id,
                    resources_str,
                    source.source_context.id,
                    payload_json,
                )
            )

        # Step 4: Bulk insert all events using flat tuple insert
        if event_rows:
            values_clause = ",".join(["(?, ?, ?, 1, ?, ?, ?, ?, ?, ?)"] * len(event_rows))
            # Flatten all rows into a single parameter list
            flat_params = [param for row in event_rows for param in row]
            with _translate_out_of_memory():
                self._conn.execute(
                    f"INSERT INTO events ({_COLS}) VALUES {values_clause}",
                    flat_params,
                )

        # Clear the batch
        self._batch.clear()

    def flush(self) -> None:
        """Flush pending events to the database."""
        self._flush()

    def register_observation_context(self, obs: ObservationContext) -> None:
        """Register an observation context upfront before adding events.

        This should be called once per observation context before adding any events.
        Observation contexts are NOT batched---they're inserted immediately to ensure
        foreign key constraints are satisfied when events are flushed.
        """

        # Insert the observation context immediately (not batched)
        # This ensures the foreign key exists when events are later flushed
        with _translate_out_of_memory():
            self._conn.execute(
                """INSERT OR REPLACE INTO observation_contexts (id, time_start, time_end, source, format, timezone)
                   SELECT ?, ?, ?, os.id, of.id, ?
                   FROM obs_sources os, obs_formats of
                   WHERE os.name = ? AND of.name = ?""",
                [
                    obs.id,
                    obs.time_interval.start,
                    obs.time_interval.end,
                    obs.timezone.key,
                    obs.source.value,
                    obs.format.value,
                ],
            )

    def add_event(self, event_type: str, payload: dict[str, Any], source: Source) -> None:
        """Add an event to the stream. Events are batched and flushed periodically.

        IMPORTANT: Observation contexts MUST be registered via register_observation_context()
        before adding events that reference them, otherwise flush() will fail with a
        foreign key constraint violation.

        Returns:
            None (batched mode) - call flush() to persist events.
        """
        self._batch.append((event_type, payload, source))

        if len(self._batch) >= self._batch_size_threshold:
            self._flush()

        return None

    def event_adder(self) -> "EventStream.Adder":
        """Return a callable for adding events."""
        return self.add_event

    def query_raw(self, sql: str, params: list = []) -> list[tuple]:
        """Execute SQL query and return raw rows."""
        # Flush pending events to ensure query sees all events
        self._flush()
        with _translate_out_of_memory():
            return self._conn.execute(sql, params).fetchall()

    def execute(self, sql: str, params: list = []) -> None:
        """Execute SQL statement (for mutations)."""
        with _translate_out_of_memory():
            self._conn.execute(sql, params)


def execute_fetch_one(conn: duckdb.DuckDBPyConnection, sql: str, params: list = []) -> Any:
    """Execute SQL statement and fetch one. Raises if empty"""
    result = conn.execute(sql, params).fetchone()
    if result is None:
        raise UnexpectedEmptyQueryResultError(sql, params)
    return result[0]


class _ReadOnlyLookup:
    """Read-only string<->ID lookup, backed by two tiers of DuckDB storage:

    - a shared, read-only table (ATTACHed from another database, see
      `_build_shared_lookup_db`) holding the static names known at
      `EventManager` construction time, shared across every `EventManager` for
      the same rules_engine, never written to by this class.
    - a small local overlay table, private to this instance, for names
      registered after construction (`preload()`) or restored from a snapshot.

    ``view_name`` is a UNION ALL view combining both tiers under the plain
    unqualified table name, so raw-SQL joins elsewhere in the codebase don't
    need to know about the split. ``get_id()`` raises *error_type* for
    unknown names.

    The Python-level cache mirrors this same split: ``shared_id_cache``/
    ``shared_str_cache`` are the *same dict objects* passed in by every
    `_ReadOnlyLookup` for this rules_engine's shared tier (built once by
    `_build_shared_lookup_db`), fully populated before `__init__` returns --
    this class never writes to them, only reads. ``_local_id_cache``/
    ``_local_str_cache`` start empty and are private to this instance,
    populated only by `preload()` (or, for a restored snapshot's
    `obs_sources`/`obs_formats` local overlay, `_reload_local_overlay()`).
    An instance is always fully valid and queryable immediately after
    construction: no separate "warm the cache" call is required, unlike the
    old `load_from_view()`, whose result depended on when the caller
    happened to invoke it relative to the local overlay's own state.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        view_name: str,
        local_table: str,
        shared_id_cache: dict[str, int],
        shared_str_cache: dict[int, str],
        *,
        error_type: type[Exception] = KeyError,
    ):
        self._conn = conn
        self._view_name = view_name
        self._local_table = local_table
        self._error_type = error_type
        self._shared_id_cache = shared_id_cache
        self._shared_str_cache = shared_str_cache
        self._local_id_cache: dict[str, int] = {}
        self._local_str_cache: dict[int, str] = {}

    def _reload_local_overlay(self) -> None:
        """Resync the local-overlay cache from `local_table` after it was
        written to directly via raw SQL (`from_snapshot()`'s restore of
        `obs_sources`/`obs_formats`, which bypasses `preload()` for
        efficiency). Reads only `local_table`, never the shared tier, so it
        can't reintroduce the ambiguity `load_from_view()` used to have.
        """
        with _translate_out_of_memory():
            rows = self._conn.execute(f"SELECT name, id FROM {self._local_table}").fetchall()
        for name, id_val in rows:
            self._local_id_cache[name] = id_val
            self._local_str_cache[id_val] = name

    def preload(self, names: set[str]) -> None:
        """Register additional names not already known, warming the local cache.

        Writes only to this instance's local overlay table and local cache.
        The shared table/cache are never touched. ids are computed directly
        via `_stable_id()` rather than assigned by the database, so unlike a
        sequence-based scheme there's no need to read the id back after
        insert to learn what id a name got. But `_stable_id()` is a 31-bit
        hash, so two distinct names can collide: if that happens, `INSERT OR
        IGNORE` silently keeps whichever name claimed the id first, leaving
        the second name's row missing. We verify each computed id actually
        persisted under the requested name, checked against the combined
        view since the id may have already been claimed by a name in
        either tier, before caching anything, so a collision raises
        instead of silently aliasing two names to one id.
        """
        missing = [n for n in names if n not in self._local_id_cache and n not in self._shared_id_cache]
        if not missing:
            return

        id_by_name = {name: _stable_id(name) for name in missing}

        values_clause = ",".join(["(?, ?)"] * len(missing))
        params = [param for name in missing for param in (id_by_name[name], name)]
        with _translate_out_of_memory():
            self._conn.execute(
                f"INSERT OR IGNORE INTO {self._local_table} (id, name) VALUES {values_clause}",
                params,
            )

            ids = list(id_by_name.values())
            id_placeholders = ",".join(["?"] * len(ids))
            rows = self._conn.execute(
                f"SELECT id, name FROM {self._view_name} WHERE id IN ({id_placeholders})",
                ids,
            ).fetchall()
        # Retain every persisted name per id, not just one: the shared tier
        # and this instance's local overlay are separate physical tables, so
        # the same id can be claimed by a different name in each without
        # either table's own PRIMARY KEY constraint ever seeing the conflict.
        # A dict keyed by id alone would silently drop one side of that,
        # with the outcome depending on SQL row return order.
        persisted_names_by_id: dict[int, set[str]] = {}
        for id_val, persisted_name in rows:
            persisted_names_by_id.setdefault(id_val, set()).add(persisted_name)

        for name in missing:
            id_val = id_by_name[name]
            persisted_names = persisted_names_by_id.get(id_val, set())
            other_names = persisted_names - {name}
            if other_names or name not in persisted_names:
                other_name = next(iter(other_names), None)
                raise StableIdCollisionError(self._view_name, id_val, name, other_name)

        for name in missing:
            id_val = id_by_name[name]
            self._local_id_cache[name] = id_val
            self._local_str_cache[id_val] = name

    def get_id(self, name: str) -> int:
        """Get ID for a preloaded name. Raises *error_type* if unknown."""
        try:
            return self._local_id_cache[name]
        except KeyError:
            pass
        try:
            return self._shared_id_cache[name]
        except KeyError:
            raise self._error_type(f"Unknown {self._view_name} name: {name!r}")

    def get_ids(self, names: list[str]) -> list[int]:
        """Get IDs for multiple preloaded names. Raises *error_type* if any are unknown."""
        return [self.get_id(n) for n in names]

    def get_str(self, id: int) -> Optional[str]:
        """Get string for an ID, or None if not found."""
        if id in self._local_str_cache:
            return self._local_str_cache[id]
        return self._shared_str_cache.get(id)


class _SnapshotRuleNames:
    """Adapts a live rules_engine plus a restored snapshot's own rule/event-type
    names into the interface `_build_shared_lookup_db` expects (`all_rule_names`/
    `all_event_types`).

    Used only by `from_snapshot()`: a restored snapshot's events can reference a
    rule or event-type name the live rules_engine no longer declares (e.g. a
    retired rule). Building the shared lookup db from the union up front means
    every name the restored data needs is already there, with no separate
    write-into-a-local-overlay-and-filter-duplicates step required. This is
    unlike `obs_sources`/`obs_formats` (see `_LOOKUP_LOCAL_TABLES`), which aren't
    rules_engine-derived and so still go through that path.

    A fresh instance each call, so it never collides with the real
    rules_engine's entry in `_shared_lookup_cache`. Restoring a snapshot always
    builds its own one-off shared lookup db rather than reusing (or polluting)
    the cache shared by every other `EventManager` for that rules_engine.
    """

    def __init__(self, rules_engine: "RuleEngine", extra_rule_names: set[str], extra_event_types: set[str]):
        self.all_rule_names = set(rules_engine.all_rule_names) | extra_rule_names
        self.all_event_types = set(rules_engine.all_event_types) | extra_event_types


class EventManager:
    def __init__(
        self,
        rules_engine: "RuleEngine",
        *,
        threads: Optional[int] = None,
        memory_limit: Optional[str] = None,
        allow_disk_spill: bool = True,
        temp_root: Optional[str] = None,
        max_temp_directory_size: Optional[str] = None,
    ):
        """
        Args:
            rules_engine: rules engine to preload rule/event-type names from.
            threads: DuckDB worker thread count for this connection. Defaults to
                DuckDB's own default (the host's full CPU count) when omitted.
                Each EventManager's dataset is one unit of work's worth of events
                and rarely benefits from intra-query parallelism, so callers that
                create many EventManagers (e.g. one per job) should pass a small
                value (often 1) to avoid host-sized per-connection overhead.
            memory_limit: DuckDB memory_limit for this connection (e.g. "128MB").
                Defaults to DuckDB's own default (~80% of system RAM) when
                omitted.
            allow_disk_spill: whether DuckDB may spill data exceeding
                memory_limit to disk instead of raising. Spill files go in a
                directory private to this instance (DuckDB's default `.tmp`
                uses fixed file names, so instances sharing it corrupt each
                other), removed on close.
                Defaults to True, matching DuckDB's own default. Suitable
                for deployments where local disk can be assumed (e.g. bare
                metal). Set to False for deployments with no writable disk
                (e.g. read-only containers): exceeding memory_limit then
                raises `EventStorageOutOfMemoryError` instead of writing to
                disk. Independent of memory_limit: DuckDB always has some
                memory_limit (its host-RAM-based default if none is given
                here), so this can matter even without setting memory_limit
                explicitly.
            temp_root: directory to create the private spill directory in.
                Defaults to the system temp directory (`tempfile`, which
                honors TMPDIR). Ignored when allow_disk_spill is False.
            max_temp_directory_size: Maximum spill storage per instance as a
                DuckDB size string (e.g. "1GB"). None preserves DuckDB's
                default. Ignored when allow_disk_spill is False.
        """
        self._initialize_ownership()
        self.initialize(
            rules_engine,
            threads=threads,
            memory_limit=memory_limit,
            allow_disk_spill=allow_disk_spill,
            temp_root=temp_root,
            max_temp_directory_size=max_temp_directory_size,
        )

    @classmethod
    def unopened(cls) -> "EventManager":
        """Create a resource-free owner before attempting initialization.

        Retain this object, then call initialize(). If initialization fails,
        close(flush=False) releases partial state and can be retried on failure.
        """
        instance = cls.__new__(cls)
        instance._initialize_ownership()
        return instance

    def _initialize_ownership(self) -> None:
        self._initialization_started = False
        self._streams: dict[str, EventStream] = {}
        self._closed = True
        self._spill_dir: Optional[str] = None
        self._owned_lookup_path: Optional[Path] = None
        self._lookup_detached = True

    def initialize(
        self,
        rules_engine: "RuleEngine",
        *,
        threads: Optional[int] = None,
        memory_limit: Optional[str] = None,
        allow_disk_spill: bool = True,
        temp_root: Optional[str] = None,
        max_temp_directory_size: Optional[str] = None,
    ) -> None:
        """Open an owned instance once, using the constructor's settings.

        Failure always attempts close(flush=False). The owner retains this
        instance to retry if cleanup also fails. Never reinitialize partial or
        closed state: it may still own files whose removal needs retrying.
        """
        if self._initialization_started:
            raise RuntimeError("EventManager initialization has already been attempted")
        self._initialization_started = True
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(self._close_after_error)
            self._conn = duckdb.connect(":memory:")
            self._closed = False
            if threads is not None:
                self._conn.execute("SET threads = ?", [threads])
            if memory_limit is not None:
                self._set_size("memory_limit", memory_limit)
            self._conn.execute(SCHEMA)

            lookup_path, shared_lookup_python_cache = _build_shared_lookup_db(rules_engine)
            # `_SnapshotRuleNames` builds its own one-off shared lookup db per
            # `from_snapshot()` call (see its docstring), keyed in
            # `_shared_lookup_cache` by an object nothing else keeps alive once
            # `from_snapshot()` returns. Nothing else ever unlinks that file, so
            # this instance -- the only thing that will ever ATTACH it -- owns
            # cleaning it up in close(). A real rules_engine's lookup db is
            # shared by every EventManager constructed with it, so it must never
            # be deleted here.

            self._owned_lookup_path = lookup_path if isinstance(rules_engine, _SnapshotRuleNames) else None
            # Path is generated internally (tempfile), but escape defensively
            # in case a caller's platform ever produces a path with a quote.
            self._conn.execute(f"ATTACH {_sql_string_literal(lookup_path)} AS lookup (READ_ONLY)")
            self._lookup_detached = False
            self._conn.execute(_LOCAL_OVERLAY_SCHEMA)

            # Provide convenient lookup helpers. Each shares the same Python-level
            # id/str dicts for its table's shared tier (built once by
            # _build_shared_lookup_db, immutable from here on) instead of copying
            # them into a private per-instance cache.
            event_types_shared_id, event_types_shared_str = shared_lookup_python_cache["event_types"]
            self._types = _ReadOnlyLookup(
                self._conn,
                "event_types",
                "local_event_types",
                event_types_shared_id,
                event_types_shared_str,
                error_type=UndeclaredEventTypeError,
            )
            rules_shared_id, rules_shared_str = shared_lookup_python_cache["rules"]
            self._rules = _ReadOnlyLookup(
                self._conn, "rules", "local_rules", rules_shared_id, rules_shared_str, error_type=UndeclaredRuleError
            )
            obs_sources_shared_id, obs_sources_shared_str = shared_lookup_python_cache["obs_sources"]
            self._obs_sources = _ReadOnlyLookup(
                self._conn, "obs_sources", "local_obs_sources", obs_sources_shared_id, obs_sources_shared_str
            )
            obs_formats_shared_id, obs_formats_shared_str = shared_lookup_python_cache["obs_formats"]
            self._obs_formats = _ReadOnlyLookup(
                self._conn, "obs_formats", "local_obs_formats", obs_formats_shared_id, obs_formats_shared_str
            )

            # Last, so a setup failure above leaves no spill directory behind.
            # abspath: before Python 3.12, mkdtemp returns a relative path for a relative temp_root.
            self._spill_dir = (
                os.path.abspath(tempfile.mkdtemp(prefix="attribution_lib_spill_", dir=temp_root))
                if allow_disk_spill
                else None
            )
            self._conn.execute("SET temp_directory = ?", [self._spill_dir or ""])
            if allow_disk_spill and max_temp_directory_size is not None:
                self._set_size("max_temp_directory_size", max_temp_directory_size)
            cleanup.pop_all()

    def _set_size(self, setting: str, value: str) -> None:
        try:
            self._conn.execute(f"SET {setting} = ?", [value])
        except (duckdb.ParserException, duckdb.InvalidInputException) as error:
            raise InvalidEventStorageSizeError(setting, value) from error

    def preload_rule_names(self, names: set[str]) -> None:
        """Pre-populate the rules lookup table and warm caches.

        After this call, ``rules.get_id()`` for any preloaded name is a pure
        dict lookup with no SQL or stream flushing.
        """
        self._rules.preload(names)

    def preload_event_types(self, names: set[str]) -> None:
        """Pre-populate the event types lookup table and warm caches.

        Event types not in this set can still be created dynamically via
        ``_flush()`` (read-write behavior), but preloaded names avoid SQL.
        """
        self._types.preload(names)

    @property
    def types(self) -> _ReadOnlyLookup:
        """Helper for event type lookups (read-only, raises on unknown names)."""
        return self._types

    @property
    def rules(self) -> _ReadOnlyLookup:
        """Helper for rule lookups (read-only, raises on unknown names)."""
        return self._rules

    def create_stream(self, context_id: str) -> EventStream:
        if context_id is None:
            context_id = f"_auto_{id(self)}_{len(self._streams)}"
        if context_id not in self._streams:
            ctx_id = _get_or_create_id(self._conn, TableName.CONTEXTS, context_id)
            self._streams[context_id] = EventStream(ctx_id, context_id, self._conn, self)
        return self._streams[context_id]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures close() is called."""
        self.close()
        return False  # Don't suppress exceptions

    def _close_after_error(self) -> None:
        """Unwind without replacing the initialization/import exception."""
        try:
            self.close(flush=False)
        except Exception:
            logger.exception("EventManager cleanup failed while handling another error")

    def __del__(self):
        """Fallback cleanup discards buffered events; explicit close() flushes by default."""
        if not getattr(self, "_closed", True):  # unset when __init__ failed early
            # No caller can use this data now. Do not let an unflushable buffer
            # prevent the last opportunity to release connection and spill files.
            try:
                self.close(flush=False)
            except Exception:
                # Suppress exceptions in destructor since we can't propagate them
                pass

    # Bump when DuckDB table schemas change (columns, types, new tables).
    # Snapshots written with a different version are rejected on load.
    #
    # 1 -> 2: event_types/rules/obs_sources/obs_formats ids switched from
    # sequence-assigned (nextval()) to content-addressed (_stable_id()). A
    # version-1 snapshot's ids for those tables are arbitrary and have no
    # relationship to _stable_id(name), so importing one under version-2 code
    # would silently mix unrelated ids -- reject it instead.
    # 2 -> 3: legacy observation formats were removed from `Format`. A
    # version-2 `obs_formats` table can carry a retired name that the overlay
    # restore keeps and `Format(obs_format_str)` in event_helpers.py then
    # rejects, so treat such snapshots as a cache miss.
    _SNAPSHOT_SCHEMA_VERSION = 3
    _SCHEMA_VERSION_FILE = "schema_version"

    # All DuckDB tables in dependency-safe import order.
    # NOTE: TableName.OBSERVATION_CONTEXTS has value "obs_contexts" but the
    # actual SQL table is "observation_contexts", so we use real table names here.
    _SNAPSHOT_TABLES = [
        "event_types",
        "rules",
        "contexts",
        "obs_sources",
        "obs_formats",
        "observation_contexts",
        "events",
    ]

    def export_snapshot(self, snapshot_dir: Path) -> None:
        """Export all tables to Parquet files for later re-import.

        Flushes all pending events before exporting. Each table is written
        to a separate Parquet file under snapshot_dir.
        """
        for stream in self._streams.values():
            stream.flush()

        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for table in self._SNAPSHOT_TABLES:
            dest = snapshot_dir / f"{table}.parquet"
            try:
                self._conn.execute(f"COPY {table} TO {_sql_string_literal(dest)} (FORMAT PARQUET)")
            except (PermissionError, OSError) as e:
                raise PermissionError(f"Cannot write snapshot file {dest}: {e}") from e
            except duckdb.IOException as e:
                raise RuntimeError(f"DuckDB I/O error exporting '{table}' to {dest}: {e}") from e
            except duckdb.Error as e:
                raise RuntimeError(f"DuckDB error exporting '{table}' to {dest}: {e}") from e

        version_path = snapshot_dir / self._SCHEMA_VERSION_FILE
        version_path.write_text(str(self._SNAPSHOT_SCHEMA_VERSION))

    @classmethod
    def from_snapshot(cls, snapshot_dir: Path, rules_engine: "RuleEngine") -> "EventManager":
        """Create an EventManager pre-populated from a previous snapshot.

        The returned instance has the same DuckDB state as when export_snapshot()
        was called, ready for filters/refiners/outcome without re-extraction.
        """
        snapshot_dir = Path(snapshot_dir)
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot_dir}")

        try:
            version_file = snapshot_dir / cls._SCHEMA_VERSION_FILE
            if version_file.exists():
                stored_version = int(version_file.read_text().strip())
                if stored_version != cls._SNAPSHOT_SCHEMA_VERSION:
                    raise SnapshotSchemaVersionError(
                        f"Snapshot schema version mismatch: stored={stored_version}, "
                        f"current={cls._SNAPSHOT_SCHEMA_VERSION}"
                    )
        except ValueError:
            raise
        except (PermissionError, OSError) as e:
            raise RuntimeError(f"Cannot read schema version from {snapshot_dir}: {e}") from e

        for table in ("rules", "event_types"):
            src = snapshot_dir / f"{table}.parquet"
            if not src.exists():
                raise FileNotFoundError(f"Snapshot file missing: {src}")

        try:
            with duckdb.connect(":memory:") as scratch:
                extra_rule_names = {
                    row[0]
                    for row in scratch.execute(
                        f"SELECT name FROM read_parquet({_sql_string_literal(snapshot_dir / 'rules.parquet')})"
                    ).fetchall()
                }
                extra_event_types = {
                    row[0]
                    for row in scratch.execute(
                        f"SELECT name FROM read_parquet({_sql_string_literal(snapshot_dir / 'event_types.parquet')})"
                    ).fetchall()
                }
        except (PermissionError, OSError) as e:
            raise PermissionError(f"Cannot read snapshot rule/event-type names: {e}") from e
        except duckdb.IOException as e:
            raise RuntimeError(f"DuckDB I/O error reading snapshot rule/event-type names: {e}") from e
        except duckdb.Error as e:
            raise RuntimeError(f"DuckDB error reading snapshot rule/event-type names: {e}") from e

        # Building the shared lookup db from rules_engine's names union'd with
        # whatever this snapshot's events actually reference means every name
        # the restored data needs is already there, including a retired
        # rule/event-type the live rules_engine no longer declares. No
        # separate write-into-local-overlay-and-dedup step needed for these
        # two tables (contrast with `obs_sources`/`obs_formats` below, which
        # aren't rules_engine-derived and so still need one).
        mgr = cls(_SnapshotRuleNames(rules_engine, extra_rule_names, extra_event_types))

        with contextlib.ExitStack() as cleanup:
            cleanup.callback(mgr._close_after_error)
            mgr._restore_snapshot(snapshot_dir)
            cleanup.pop_all()
        return mgr

    def _restore_snapshot(self, snapshot_dir: Path) -> None:
        for table in self._SNAPSHOT_TABLES:
            if table in ("rules", "event_types"):
                continue  # already fully seeded above, via the shared lookup db itself
            src = snapshot_dir / f"{table}.parquet"
            if not src.exists():
                raise FileNotFoundError(f"Snapshot file missing: {src}")
            try:
                local_table = _LOOKUP_LOCAL_TABLES.get(table)
                if local_table is not None:
                    # `table` (`obs_sources`/`obs_formats`) is a read-only view
                    # (shared tier ∪ this instance's local overlay), so
                    # DELETE/INSERT can't target it directly. Write into the
                    # local overlay instead, skipping any name already
                    # resolvable from the shared tier: a name's id is
                    # content-addressed (`_stable_id`), so it's identical
                    # whichever tier it ends up in. Skipping duplicates isn't
                    # lossy, it just avoids a harmless-but-wasteful duplicate
                    # row for names the current enum already covers.
                    # (The schema-version bump to 2 is what guarantees every
                    # snapshot reaching this point already used
                    # content-addressed ids, see `_SNAPSHOT_SCHEMA_VERSION`.)
                    self._conn.execute(f"DELETE FROM {local_table}")
                    self._conn.execute(
                        f"""INSERT INTO {local_table}
                            SELECT * FROM read_parquet({_sql_string_literal(src)})
                            WHERE name NOT IN (SELECT name FROM {table})"""
                    )
                    _assert_no_id_collision(self._conn, table)
                else:
                    self._conn.execute(f"DELETE FROM {table}")
                    self._conn.execute(f"INSERT INTO {table} SELECT * FROM read_parquet({_sql_string_literal(src)})")
            except (PermissionError, OSError) as e:
                raise PermissionError(f"Cannot read snapshot file {src}: {e}") from e
            except duckdb.IOException as e:
                raise RuntimeError(f"DuckDB I/O error importing '{table}' from {src}: {e}") from e
            except duckdb.Error as e:
                raise RuntimeError(f"DuckDB error importing '{table}' from {src}: {e}") from e

        _validate_snapshot_lookup_references(self._conn)

        # Advance sequences past imported IDs so new inserts don't collide.
        # DuckDB lacks setval(), so we consume nextval() until we're past max(id).
        # Only contexts/events still use sequences -- event_types/rules/
        # obs_sources/obs_formats ids are content-addressed (_stable_id), so
        # they need no advancement, and the loop below skips them
        # automatically (no matching sequence to find). observation_contexts
        # uses an externally-supplied ID, no sequence either.
        for table in self._SNAPSHOT_TABLES:
            seq_name = f"{table}_seq"
            if not self._conn.execute("SELECT 1 FROM duckdb_sequences() WHERE sequence_name=?", [seq_name]).fetchone():
                continue
            row = self._conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()
            max_id = row[0] if row else 0
            if max_id > 0:
                cur = 0
                while cur <= max_id:
                    next_row = self._conn.execute(f"SELECT nextval('{seq_name}')").fetchone()
                    assert next_row is not None, f"nextval('{seq_name}') returned no row"
                    cur = next_row[0]

        # self._types/self._rules need no resync here: their shared cache was
        # already built from _SnapshotRuleNames's rules_engine-union-snapshot
        # names at construction. Only obs_sources/obs_formats have a local
        # overlay actually written above (via raw SQL, not preload()), so
        # only their local caches need catching up.
        self._obs_sources._reload_local_overlay()
        self._obs_formats._reload_local_overlay()

    def checkpoint(self, name: str) -> None:
        """Snapshot the events table to a shadow table within the same connection.

        Uses DuckDB-internal CREATE TABLE AS SELECT for zero Python serialization
        overhead. Only the events table needs checkpointing — lookup tables
        (event_types, rules, obs_sources, obs_formats) and observation_contexts
        don't change during filters/refiners/outcome assignment.
        """
        for stream in self._streams.values():
            stream.flush()
        shadow = f"_ckpt_{name}"
        with _translate_out_of_memory(), _atomic(self._conn):
            self._conn.execute(f'DROP TABLE IF EXISTS "{shadow}"')
            self._conn.execute(f'CREATE TABLE "{shadow}" AS SELECT * FROM events')

    def restore(self, name: str) -> None:
        """Restore the events table from a previous checkpoint."""
        shadow = f"_ckpt_{name}"
        with _translate_out_of_memory(), _atomic(self._conn):
            self._conn.execute("DELETE FROM events")
            self._conn.execute(f'INSERT INTO events SELECT * FROM "{shadow}"')

    def drop_checkpoint(self, name: str) -> None:
        """Remove the shadow table for a checkpoint."""
        self._conn.execute(f'DROP TABLE IF EXISTS "_ckpt_{name}"')

    def close(self, *, flush: bool = True):
        """Release owned resources; retain unfinished steps for a later retry.

        Ordinary callers flush buffered events first. A retiring service wrapper
        uses flush=False: its data is being discarded, and an OOM during flush
        must not prevent releasing the connection and spill files.

        Close DuckDB before removing files. Clear ownership only after removal
        succeeds (or the path is already absent), and mark closed last. Thus a
        failed removal can be retried without losing the path or deleting another
        instance's resources. The spill root and shared lookup DB are never owned.
        """
        if self._closed:
            return

        if flush:
            for stream in self._streams.values():
                stream.flush()
        self._streams.clear()

        if self._owned_lookup_path is not None and not self._lookup_detached:
            self._conn.execute("DETACH lookup")
            self._lookup_detached = True
        self._conn.close()

        if self._owned_lookup_path is not None:
            self._owned_lookup_path.unlink(missing_ok=True)
            self._owned_lookup_path = None

        if self._spill_dir is not None:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self._spill_dir)
            self._spill_dir = None

        self._closed = True
