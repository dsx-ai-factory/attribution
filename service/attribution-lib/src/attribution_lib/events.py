# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import json
import zlib
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypeAlias

import duckdb

from attribution_lib.exceptions import (
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


def _get_or_create_id(conn: duckdb.DuckDBPyConnection, table: TableName, name: str) -> int:
    """Get or create an ID for a string in a lookup table. Returns None if name is None."""
    # Insert if not exists, then get the ID
    conn.execute(f"INSERT INTO {table.value} (name) VALUES (?) ON CONFLICT (name) DO NOTHING", [name])
    return execute_fetch_one(conn, f"SELECT id FROM {table.value} WHERE name=?", [name])


def sql_placeholders(count: int) -> str:
    """Generate SQL placeholder string for parameterized IN clauses."""
    return ",".join(["?"] * count)


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

    This is what lets from_snapshot() restore a snapshot's lookup rows
    directly rather than needing them to match whatever a freshly built
    EventManager happens to assign: a name's id is the same regardless of
    which process or in what order it was computed.

    """
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


# Schema with lookup tables for string->int mappings. event_types/rules/
# obs_sources/obs_formats ids are content-addressed (_stable_id), not
# sequence-assigned, so they need no DEFAULT clause.
SCHEMA = """
CREATE TABLE event_types (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL
);

CREATE TABLE rules (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL
);

CREATE SEQUENCE contexts_seq START 1;
CREATE TABLE contexts (
    id INTEGER PRIMARY KEY DEFAULT nextval('contexts_seq'),
    name VARCHAR UNIQUE NOT NULL
);

CREATE TABLE obs_sources (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL
);

CREATE TABLE obs_formats (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL
);

CREATE TABLE observation_contexts (
    id BIGINT PRIMARY KEY,
    time_start TIMESTAMPTZ,
    time_end TIMESTAMPTZ,
    source INTEGER REFERENCES obs_sources(id),
    format INTEGER REFERENCES obs_formats(id),
    timezone VARCHAR
);

CREATE SEQUENCE events_seq START 1;
CREATE TABLE events (
    id INTEGER DEFAULT nextval('events_seq'),
    context_id INTEGER NOT NULL REFERENCES contexts(id),
    event_type INTEGER NOT NULL REFERENCES event_types(id),
    timestamp TIMESTAMPTZ,
    status INTEGER DEFAULT 1,
    line_num INTEGER,
    source_datum VARCHAR,
    source_rule INTEGER REFERENCES rules(id),
    resources VARCHAR,
    obs_context_id BIGINT REFERENCES observation_contexts(id),
    payload JSON,
    PRIMARY KEY (id, context_id)
);

-- Index for time-range queries (most common analytical pattern)
CREATE INDEX idx_events_ctx_timestamp ON events(context_id, timestamp);
"""

_COLS = "context_id,event_type,timestamp,status,line_num,source_datum,source_rule,resources,obs_context_id,payload"


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
        return self._conn.execute(sql, params).fetchall()

    def execute(self, sql: str, params: list = []) -> None:
        """Execute SQL statement (for mutations)."""
        self._conn.execute(sql, params)


def execute_fetch_one(conn: duckdb.DuckDBPyConnection, sql: str, params: list = []) -> Any:
    """Execute SQL statement and fetch one. Raises if empty"""
    result = conn.execute(sql, params).fetchone()
    if result is None:
        raise UnexpectedEmptyQueryResultError(sql, params)
    return result[0]


class _ReadOnlyLookup:
    """Read-only string<->ID lookup backed by a preloaded DuckDB table.

    All names must be registered via ``preload()`` before use.
    ``get_id()`` raises *error_type* for unknown names.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, table: TableName, *, error_type: type[Exception] = KeyError):
        self._conn = conn
        self._table = table
        self._error_type = error_type
        self._id_cache: dict[str, int] = {}
        self._str_cache: dict[int, str] = {}

    def preload(self, names: set[str]) -> None:
        """Bulk-insert names and warm both caches.

        ids are computed directly via _stable_id() rather than assigned by
        the database, so unlike a sequence-based scheme there's no need to
        read the id back after insert to learn what id a name got. But
        _stable_id() is a 31-bit hash, so two distinct names can collide: if
        that happens, INSERT OR IGNORE silently keeps whichever name claimed
        the id first, leaving the second name's row missing. We verify each
        computed id actually persisted under the requested name before
        caching anything, so a collision raises instead of silently
        aliasing two names to one id.
        """
        missing = [n for n in names if n not in self._id_cache]
        if not missing:
            return

        id_by_name = {name: _stable_id(name) for name in missing}

        values_clause = ",".join(["(?, ?)"] * len(missing))
        params = [param for name in missing for param in (id_by_name[name], name)]
        self._conn.execute(
            f"INSERT OR IGNORE INTO {self._table.value} (id, name) VALUES {values_clause}",
            params,
        )

        ids = list(id_by_name.values())
        id_placeholders = ",".join(["?"] * len(ids))
        rows = self._conn.execute(
            f"SELECT id, name FROM {self._table.value} WHERE id IN ({id_placeholders})",
            ids,
        ).fetchall()
        persisted_name_by_id = {id_val: persisted_name for id_val, persisted_name in rows}

        for name in missing:
            id_val = id_by_name[name]
            persisted_name = persisted_name_by_id.get(id_val)
            if persisted_name != name:
                raise StableIdCollisionError(self._table.value, id_val, name, persisted_name)

        for name in missing:
            id_val = id_by_name[name]
            self._id_cache[name] = id_val
            self._str_cache[id_val] = name

    def get_id(self, name: str) -> int:
        """Get ID for a preloaded name. Raises *error_type* if unknown."""
        try:
            return self._id_cache[name]
        except KeyError:
            raise self._error_type(f"Unknown {self._table.value} name: {name!r}")

    def get_ids(self, names: list[str]) -> list[int]:
        """Get IDs for multiple preloaded names. Raises *error_type* if any are unknown."""
        return [self.get_id(n) for n in names]

    def get_str(self, id: int) -> Optional[str]:
        """Get string for an ID, or None if not found."""
        return self._str_cache.get(id)

    def rebuild_cache(self) -> None:
        """Repopulate caches from the underlying DuckDB table."""
        rows = self._conn.execute(f"SELECT name, id FROM {self._table.value}").fetchall()
        self._id_cache.clear()
        self._str_cache.clear()
        for name, id_val in rows:
            self._id_cache[name] = id_val
            self._str_cache[id_val] = name


class EventManager:
    def __init__(self, rules_engine: "RuleEngine"):
        self._conn = duckdb.connect(":memory:")
        self._conn.execute(SCHEMA)
        self._streams: dict[str, EventStream] = {}
        self._closed = False  # Track if already closed to prevent double-logging

        # Provide convenient lookup helpers
        self._types = _ReadOnlyLookup(self._conn, TableName.EVENT_TYPES, error_type=UndeclaredEventTypeError)
        self._rules = _ReadOnlyLookup(self._conn, TableName.RULES, error_type=UndeclaredRuleError)
        self._obs_sources = _ReadOnlyLookup(self._conn, TableName.OBSERVATION_SOURCES)
        self._obs_formats = _ReadOnlyLookup(self._conn, TableName.OBSERVATION_FORMATS)

        # Populate obs_sources and obs_formats from their enums
        self._populate_enum_tables()

        # Preload rule names and event types from the rules engine
        self._rules.preload(rules_engine.all_rule_names)
        self._types.preload(rules_engine.all_event_types)

    def _populate_enum_tables(self):
        """Populate obs_sources and obs_formats tables with enum values."""
        self._obs_sources.preload({m.value for m in ObsSource if m.value})
        self._obs_formats.preload({m.value for m in ObsFormat if m.value})

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

    def __del__(self):
        """Destructor - fallback cleanup if close() not explicitly called."""
        if not self._closed:
            # Try to close gracefully, but don't fail if there are issues
            try:
                self.close()
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
    _SNAPSHOT_SCHEMA_VERSION = 2
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
                self._conn.execute(f"COPY {table} TO '{dest}' (FORMAT PARQUET)")
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

        mgr = cls(rules_engine)

        for table in cls._SNAPSHOT_TABLES:
            src = snapshot_dir / f"{table}.parquet"
            if not src.exists():
                raise FileNotFoundError(f"Snapshot file missing: {src}")
            try:
                mgr._conn.execute(f"DELETE FROM {table}")
                mgr._conn.execute(f"INSERT INTO {table} SELECT * FROM read_parquet('{src}')")
            except (PermissionError, OSError) as e:
                raise PermissionError(f"Cannot read snapshot file {src}: {e}") from e
            except duckdb.IOException as e:
                raise RuntimeError(f"DuckDB I/O error importing '{table}' from {src}: {e}") from e
            except duckdb.Error as e:
                raise RuntimeError(f"DuckDB error importing '{table}' from {src}: {e}") from e

        # Advance sequences past imported IDs so new inserts don't collide.
        # DuckDB lacks setval(), so we consume nextval() until we're past max(id).
        # Only contexts/events still use sequences -- event_types/rules/
        # obs_sources/obs_formats ids are content-addressed (_stable_id), so
        # they need no advancement: a restored id and a freshly-computed one
        # for the same name always agree. observation_contexts uses an
        # externally-supplied ID, no sequence either.
        for table in cls._SNAPSHOT_TABLES:
            seq_name = f"{table}_seq"
            if not mgr._conn.execute("SELECT 1 FROM duckdb_sequences() WHERE sequence_name=?", [seq_name]).fetchone():
                continue
            row = mgr._conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()
            max_id = row[0] if row else 0
            if max_id > 0:
                cur = 0
                while cur <= max_id:
                    next_row = mgr._conn.execute(f"SELECT nextval('{seq_name}')").fetchone()
                    assert next_row is not None, f"nextval('{seq_name}') returned no row"
                    cur = next_row[0]

        for lookup in (mgr._types, mgr._rules, mgr._obs_sources, mgr._obs_formats):
            lookup.rebuild_cache()

        return mgr

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
        self._conn.execute(f'DROP TABLE IF EXISTS "{shadow}"')
        self._conn.execute(f'CREATE TABLE "{shadow}" AS SELECT * FROM events')

    def restore(self, name: str) -> None:
        """Restore the events table from a previous checkpoint."""
        shadow = f"_ckpt_{name}"
        self._conn.execute("DELETE FROM events")
        self._conn.execute(f'INSERT INTO events SELECT * FROM "{shadow}"')

    def drop_checkpoint(self, name: str) -> None:
        """Remove the shadow table for a checkpoint."""
        self._conn.execute(f'DROP TABLE IF EXISTS "_ckpt_{name}"')

    def close(self):
        """Close the event manager, flushing all streams and logging statistics.

        This method is idempotent - it's safe to call multiple times.
        """
        if self._closed:
            return

        self._closed = True

        # Flush all streams before closing
        for stream in self._streams.values():
            stream.flush()
        self._streams.clear()
        self._conn.close()
