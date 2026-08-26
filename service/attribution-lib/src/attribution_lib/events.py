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
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, TypeAlias

import duckdb

from attribution_lib.exceptions import (
    UndeclaredEventTypeError,
    UndeclaredRuleError,
    UnexpectedEmptyQueryResultError,
)
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import (
    Format as ObsFormat,
    Source as ObsSource,
)
from attribution_lib.resources.resource_set import serialize_resource_set
from attribution_lib.serialization import serialize
from attribution_lib.sources import Source

if TYPE_CHECKING:
    from attribution_lib.rule_types.definitions import RuleEmission
    from attribution_lib.rules_engine import RuleEngine


SourceProvenanceKey: TypeAlias = tuple[Optional[int], Optional[int], Optional[str], Optional[float]]
ExtractionProvenanceKey: TypeAlias = tuple[SourceProvenanceKey, str]


def source_provenance_key(source: Source) -> SourceProvenanceKey:
    """Return the observation identity used only for Decision DAG provenance.

    Args:
        source: Attribution evidence source.

    Returns:
        Stable key independent of the credited attribution rule.
    """
    observation_id = int(source.source_context.id) if source.source_context is not None else None
    timestamp = source.time.timestamp() if source.time is not None else None
    return observation_id, source.line_num, source.datum, timestamp


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


# Schema with lookup tables for string->int mappings
SCHEMA = """
CREATE SEQUENCE event_types_seq START 1;
CREATE TABLE event_types (
    id INTEGER PRIMARY KEY DEFAULT nextval('event_types_seq'),
    name VARCHAR UNIQUE NOT NULL
);

CREATE SEQUENCE rules_seq START 1;
CREATE TABLE rules (
    id INTEGER PRIMARY KEY DEFAULT nextval('rules_seq'),
    -- Internal emission rows encode (credited rule, extractor rule) in name.
    name VARCHAR UNIQUE NOT NULL
);

CREATE SEQUENCE contexts_seq START 1;
CREATE TABLE contexts (
    id INTEGER PRIMARY KEY DEFAULT nextval('contexts_seq'),
    name VARCHAR UNIQUE NOT NULL
);

CREATE SEQUENCE obs_sources_seq START 1;
CREATE TABLE obs_sources (
    id INTEGER PRIMARY KEY DEFAULT nextval('obs_sources_seq'),
    name VARCHAR UNIQUE NOT NULL
);

CREATE SEQUENCE obs_formats_seq START 1;
CREATE TABLE obs_formats (
    id INTEGER PRIMARY KEY DEFAULT nextval('obs_formats_seq'),
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
        source_rule INTEGER - reference to either an ordinary credited rule or
            an exact credited-rule/extractor metadata row in rules
        resources VARCHAR   - resource identifier ("ANY" or hostname)
        obs_context_id BIGINT - observation context ID
        payload JSON        - event payload as JSON

    Use query_raw() for SELECT queries, execute() for mutations.
    """

    class Adder(Protocol):
        def __call__(
            self,
            event_type: str,
            payload: dict[str, Any],
            source: Source,
            *,
            emitter_rule_id: Optional[str] = None,
        ) -> None: ...

    def __init__(self, ctx_id: int, ctx_str: str, conn: duckdb.DuckDBPyConnection, mgr: "EventManager"):
        self._ctx_id = ctx_id
        self._ctx_str = ctx_str
        self._conn = conn
        self._mgr = mgr
        self._batch: list[tuple[str, dict[str, Any], Source, Optional[str]]] = []
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

        # Step 1: Resolve the event types used by this batch. Rule and credit
        # names were preloaded from the rule engine and declarations.
        event_types = set()
        for event_type, _, _, _ in self._batch:
            event_types.add(event_type)

        # Step 2: Resolve event type strings to integer IDs.
        types_lookup = self._mgr._types
        type_ids: dict[str, int] = {t: types_lookup.get_id(t) for t in event_types}
        rules_lookup = self._mgr._rules

        # Step 3: Prepare all event rows for bulk insert
        event_rows = []
        for event_type, payload, source, emitter_rule_name in self._batch:
            typ_id = type_ids[event_type]
            rule_id: Optional[int]
            if emitter_rule_name is not None:
                if source.rule is None:
                    raise UndeclaredRuleError(
                        f"Extractor {emitter_rule_name!r} emitted an event without a credited rule"
                    )
                rule_id = rules_lookup.get_emission_id(source.rule, emitter_rule_name)
            else:
                rule_id = rules_lookup.get_id(source.rule) if source.rule is not None else None
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

    def add_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: Source,
        *,
        emitter_rule_id: Optional[str] = None,
    ) -> None:
        """Add an event to the stream. Events are batched and flushed periodically.

        IMPORTANT: Observation contexts MUST be registered via register_observation_context()
        before adding events that reference them, otherwise flush() will fail with a
        foreign key constraint violation.

        Args:
            event_type: Event type identifier.
            payload: Event data.
            source: Evidence and its attribution credit label.
            emitter_rule_id: Executable rule that emitted the event, when available.

        Returns:
            None (batched mode) - call flush() to persist events.
        """
        self._batch.append((event_type, payload, source, emitter_rule_id))

        if len(self._batch) >= self._batch_size_threshold:
            self._flush()

        return None

    def event_adder(self) -> "EventStream.Adder":
        """Return a callable for adding events."""
        return self.add_event

    def emitter_rules_by_extraction(self) -> dict[ExtractionProvenanceKey, tuple[str, ...]]:
        """Load emitting-rule identities for Decision DAG extraction nodes.

        Returns:
            Mapping from source-and-symptom identity to emitting rules.
        """
        from attribution_lib.rule_types.definitions import RuleEmission

        self._flush()
        rows = self._conn.execute(
            """SELECT e.obs_context_id, e.line_num, e.source_datum,
                      e.timestamp, e.payload, r.name
               FROM events e
               JOIN rules r ON e.source_rule = r.id
               WHERE e.context_id=?""",
            [self._ctx_id],
        ).fetchall()
        emitters: dict[ExtractionProvenanceKey, set[str]] = defaultdict(set)
        for observation_id, line_num, datum, timestamp, payload, storage_name in rows:
            emission = RuleEmission.from_storage_key(storage_name)
            if emission is None:
                continue
            payload_value = json.loads(payload) if isinstance(payload, str) else payload
            if not isinstance(payload_value, dict):
                continue
            symptom = payload_value.get("symptom")
            if not isinstance(symptom, str):
                continue
            timestamp_value = timestamp.timestamp() if timestamp is not None else None
            source_key: SourceProvenanceKey = observation_id, line_num, datum, timestamp_value
            key = source_key, symptom
            emitters[key].add(emission.emitter_rule_id)
        return {key: tuple(sorted(rule_ids)) for key, rule_ids in emitters.items()}

    def deactivate_with_rule(
        self,
        credited_rule: str,
        predicate_sql: str,
        params: list[Any],
    ) -> None:
        """Deactivate matching events while preserving extractor provenance.

        Extractor events use ``source_rule`` to reference an exact
        credit/extractor relationship. A filter or refiner that changes the
        visible credit must therefore create the corresponding relationship for
        the original extractor instead of replacing it with an ordinary rule ID.

        Args:
            credited_rule: New audit credit assigned by the filter or refiner.
            predicate_sql: SQL predicate scoped to the ``e`` events alias.
            params: Values for placeholders in ``predicate_sql``.

        Returns:
            None.
        """
        from attribution_lib.rule_types.definitions import RuleEmission

        self._flush()
        ordinary_rule_id = self._mgr.rules.get_id(credited_rule)
        matching_params = [self._ctx_id, *params]
        source_rule_rows = self._conn.execute(
            f"""SELECT DISTINCT e.source_rule, current_rule.name
                FROM events e
                LEFT JOIN rules current_rule ON e.source_rule = current_rule.id
                WHERE e.context_id=? AND e.status=1
                  AND ({predicate_sql})""",
            matching_params,
        ).fetchall()
        emissions_by_source_rule = {
            source_rule_id: emission
            for source_rule_id, storage_name in source_rule_rows
            if storage_name is not None and (emission := RuleEmission.from_storage_key(storage_name)) is not None
        }
        self._mgr._rules.preload_emissions(
            tuple(
                RuleEmission(
                    credited_rule_id=credited_rule,
                    emitter_rule_id=emission.emitter_rule_id,
                )
                for emission in emissions_by_source_rule.values()
            )
        )
        rewritten_ids = {
            source_rule_id: self._mgr._rules.get_emission_id(credited_rule, emission.emitter_rule_id)
            for source_rule_id, emission in emissions_by_source_rule.items()
        }
        if len(rewritten_ids) > 0:
            cases = " ".join("WHEN ? THEN ?" for _ in rewritten_ids)
            source_rule_sql = f"CASE e.source_rule {cases} ELSE ? END"
            source_rule_params = [value for item in rewritten_ids.items() for value in item]
            source_rule_params.append(ordinary_rule_id)
        else:
            source_rule_sql = "?"
            source_rule_params = [ordinary_rule_id]
        self._conn.execute(
            f"""UPDATE events AS e
                SET status=0,
                    source_rule={source_rule_sql}
                WHERE e.context_id=? AND e.status=1
                  AND ({predicate_sql})""",
            [*source_rule_params, *matching_params],
        )

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
        """Bulk-insert names and warm both caches."""
        if not names:
            return

        name_list = list(names)
        values_clause = ",".join(["(?)"] * len(name_list))
        # INSERT OR IGNORE + RETURNING gives back only newly inserted rows
        new_rows = self._conn.execute(
            f"INSERT OR IGNORE INTO {self._table.value} (name) VALUES {values_clause} RETURNING name, id",
            name_list,
        ).fetchall()
        for name, id_val in new_rows:
            self._id_cache[name] = id_val
            self._str_cache[id_val] = name

        # Any names that already existed won't appear in RETURNING;
        # look them up only if some were skipped.
        if len(new_rows) < len(name_list):
            missing = [n for n in name_list if n not in self._id_cache]
            if missing:
                placeholders = ",".join(["?"] * len(missing))
                existing = self._conn.execute(
                    f"SELECT name, id FROM {self._table.value} WHERE name IN ({placeholders})",
                    missing,
                ).fetchall()
                for name, id_val in existing:
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

    def get_str_required(self, id: int) -> str:
        """Get a preloaded string by ID, raising if the lookup is inconsistent."""
        try:
            return self._str_cache[id]
        except KeyError:
            raise self._error_type(f"Unknown {self._table.value} id: {id!r}")

    def rebuild_cache(self) -> None:
        """Repopulate caches from the underlying DuckDB table."""
        rows = self._conn.execute(f"SELECT name, id FROM {self._table.value}").fetchall()
        self._id_cache.clear()
        self._str_cache.clear()
        for name, id_val in rows:
            self._id_cache[name] = id_val
            self._str_cache[id_val] = name


class _RuleLookup(_ReadOnlyLookup):
    """Rule lookup with exact credited-rule to extractor metadata."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        """Initialize rules-table caches.

        Args:
            conn: DuckDB connection containing the rules table.
        """
        super().__init__(conn, TableName.RULES, error_type=UndeclaredRuleError)
        self._emission_id_cache: dict[tuple[str, str], int] = {}

    def preload_emissions(self, emissions: tuple["RuleEmission", ...]) -> None:
        """Insert immutable emission metadata into the existing rules table.

        Args:
            emissions: Credited-rule and extractor relationships.

        Returns:
            None.
        """
        if len(emissions) == 0:
            return
        values_clause = ",".join(["(?)"] * len(emissions))
        params = [emission.storage_key for emission in emissions]
        self._conn.execute(
            f"""INSERT INTO rules (name)
                VALUES {values_clause}
                ON CONFLICT (name) DO NOTHING""",
            params,
        )
        self.rebuild_cache()

    def get_emission_id(self, credited_rule: str, emitter_rule: str) -> int:
        """Return the rules-table ID for one exact emission relationship.

        Args:
            credited_rule: Attribution credit carried by ``Source.rule``.
            emitter_rule: Executable extractor rule ID.

        Returns:
            Preloaded rules-table ID.
        """
        try:
            return self._emission_id_cache[(credited_rule, emitter_rule)]
        except KeyError:
            raise UndeclaredRuleError(f"Extractor {emitter_rule!r} did not declare credited rule {credited_rule!r}")

    def rebuild_cache(self) -> None:
        """Repopulate ordinary and emission-rule caches.

        Returns:
            None.
        """
        from attribution_lib.rule_types.definitions import RuleEmission

        rows = self._conn.execute("SELECT name, id FROM rules").fetchall()
        self._id_cache.clear()
        self._str_cache.clear()
        self._emission_id_cache.clear()
        for storage_name, id_val in rows:
            emission = RuleEmission.from_storage_key(storage_name)
            self._id_cache[storage_name] = id_val
            self._str_cache[id_val] = emission.credited_rule_id if emission is not None else storage_name
            if emission is not None:
                self._emission_id_cache[(emission.credited_rule_id, emission.emitter_rule_id)] = id_val


class EventManager:
    def __init__(self, rules_engine: "RuleEngine"):
        self._conn = duckdb.connect(":memory:")
        self._conn.execute(SCHEMA)
        self._streams: dict[str, EventStream] = {}
        self._rules_engine = rules_engine
        self._closed = False  # Track if already closed to prevent double-logging

        # Provide convenient lookup helpers
        self._types = _ReadOnlyLookup(self._conn, TableName.EVENT_TYPES, error_type=UndeclaredEventTypeError)
        self._rules = _RuleLookup(self._conn)
        self._obs_sources = _ReadOnlyLookup(self._conn, TableName.OBSERVATION_SOURCES)
        self._obs_formats = _ReadOnlyLookup(self._conn, TableName.OBSERVATION_FORMATS)

        # Populate obs_sources and obs_formats from their enums
        self._populate_enum_tables()

        # Preload rule names and event types from the rules engine
        self._rules.preload(rules_engine.all_rule_names)
        self._rules.preload_emissions(rules_engine.rule_emissions)
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

    @property
    def rules_engine(self) -> "RuleEngine":
        """Return the rule engine that produced this manager's events."""
        return self._rules_engine

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
    # Supported older versions are migrated while loading; others are rejected.
    # Schema v3 replaces the v2 events.emitter_rule column with metadata rows
    # in the existing rules table.
    _RULE_METADATA_SCHEMA_VERSION = 3
    _LEGACY_EMITTER_SCHEMA_VERSION = 2
    _SNAPSHOT_SCHEMA_VERSION = _RULE_METADATA_SCHEMA_VERSION
    _SCHEMA_VERSION_FILE = "schema_version"
    _MIGRATABLE_SNAPSHOT_VERSIONS = frozenset({1, _LEGACY_EMITTER_SCHEMA_VERSION})

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
                self._conn.execute(f"COPY {table} TO ? (FORMAT PARQUET)", [str(dest)])
            except (PermissionError, OSError) as e:
                raise PermissionError(f"Cannot write snapshot file {dest}: {e}") from e
            except duckdb.IOException as e:
                raise RuntimeError(f"DuckDB I/O error exporting '{table}' to {dest}: {e}") from e
            except duckdb.Error as e:
                raise RuntimeError(f"DuckDB error exporting '{table}' to {dest}: {e}") from e

        version_path = snapshot_dir / self._SCHEMA_VERSION_FILE
        version_path.write_text(str(self._SNAPSHOT_SCHEMA_VERSION))

    def _migrate_legacy_emitter_rules(
        self,
        events_src: Path,
    ) -> None:
        """Convert v2 event emitter IDs into rules-table emission metadata.

        Args:
            events_src: Path to the v2 events Parquet file.

        Returns:
            None.
        """
        from attribution_lib.rule_types.definitions import RuleEmission

        legacy_pairs = self._conn.execute(
            """SELECT DISTINCT COALESCE(credit.name, emitter.name), emitter.name
               FROM read_parquet(?) legacy
               LEFT JOIN rules credit ON legacy.source_rule = credit.id
               JOIN rules emitter ON legacy.emitter_rule = emitter.id
               WHERE legacy.emitter_rule IS NOT NULL""",
            [str(events_src)],
        ).fetchall()
        emissions = tuple(
            RuleEmission(
                credited_rule_id=credited_rule,
                emitter_rule_id=emitter_rule,
            )
            for credited_rule, emitter_rule in legacy_pairs
        )
        self._rules.preload_emissions(emissions)
        if len(emissions) == 0:
            return
        mapping_values = ",".join(["(?, ?, ?)"] * len(emissions))
        mapping_params = [
            value
            for emission in emissions
            for value in (
                emission.credited_rule_id,
                emission.emitter_rule_id,
                self._rules.get_emission_id(emission.credited_rule_id, emission.emitter_rule_id),
            )
        ]
        self._conn.execute(
            f"""UPDATE events current
               SET source_rule = mapping.emission_id
               FROM read_parquet(?) legacy
               LEFT JOIN rules credit ON legacy.source_rule = credit.id
               JOIN rules emitter ON legacy.emitter_rule = emitter.id
               JOIN (VALUES {mapping_values})
                 AS mapping(credited_rule, emitter_rule, emission_id)
                 ON mapping.credited_rule = COALESCE(credit.name, emitter.name)
                AND mapping.emitter_rule = emitter.name
               WHERE current.id = legacy.id
                 AND current.context_id = legacy.context_id
                 AND legacy.emitter_rule IS NOT NULL""",
            [str(events_src), *mapping_params],
        )

    @classmethod
    def from_snapshot(cls, snapshot_dir: Path, rules_engine: "RuleEngine") -> "EventManager":
        """Create an EventManager pre-populated from a previous snapshot.

        The returned instance has the same DuckDB state as when export_snapshot()
        was called, ready for filters/refiners/outcome without re-extraction.
        """
        snapshot_dir = Path(snapshot_dir)
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot_dir}")

        version_file = snapshot_dir / cls._SCHEMA_VERSION_FILE
        # Missing version files were historically accepted and have the v1 event shape.
        stored_version = 1
        if version_file.exists():
            try:
                raw_version = version_file.read_text().strip()
            except (PermissionError, OSError) as e:
                raise RuntimeError(f"Cannot read schema version from {snapshot_dir}: {e}") from e
            try:
                stored_version = int(raw_version)
            except ValueError as e:
                raise ValueError(f"Invalid snapshot schema version: {raw_version!r}") from e
        supported_versions = cls._MIGRATABLE_SNAPSHOT_VERSIONS | {cls._SNAPSHOT_SCHEMA_VERSION}
        if stored_version not in supported_versions:
            raise ValueError(
                f"Snapshot schema version mismatch: stored={stored_version}, current={cls._SNAPSHOT_SCHEMA_VERSION}"
            )

        mgr = cls(rules_engine)

        for table in reversed(cls._SNAPSHOT_TABLES):
            mgr._conn.execute(f"DELETE FROM {table}")

        for table in cls._SNAPSHOT_TABLES:
            src = snapshot_dir / f"{table}.parquet"
            if not src.exists():
                raise FileNotFoundError(f"Snapshot file missing: {src}")
            try:
                if table == "rules" and stored_version < cls._RULE_METADATA_SCHEMA_VERSION:
                    mgr._conn.execute(
                        "INSERT INTO rules (id, name) SELECT id, name FROM read_parquet(?)",
                        [str(src)],
                    )
                elif table == "events" and stored_version < cls._RULE_METADATA_SCHEMA_VERSION:
                    # Schema v3 moved emitter identity out of events. Import the
                    # common event shape; v2 provenance is migrated below.
                    legacy_columns = (
                        "id, context_id, event_type, timestamp, status, line_num, "
                        "source_datum, source_rule, resources, obs_context_id, payload"
                    )
                    mgr._conn.execute(
                        f"INSERT INTO events ({legacy_columns}) SELECT {legacy_columns} FROM read_parquet(?)",
                        [str(src)],
                    )
                else:
                    mgr._conn.execute(f"INSERT INTO {table} SELECT * FROM read_parquet(?)", [str(src)])
            except (PermissionError, OSError) as e:
                raise PermissionError(f"Cannot read snapshot file {src}: {e}") from e
            except duckdb.IOException as e:
                raise RuntimeError(f"DuckDB I/O error importing '{table}' from {src}: {e}") from e
            except duckdb.Error as e:
                raise RuntimeError(f"DuckDB error importing '{table}' from {src}: {e}") from e

        mgr._rules.rebuild_cache()
        if stored_version < cls._RULE_METADATA_SCHEMA_VERSION:
            mgr._rules.preload_emissions(rules_engine.rule_emissions)
        if stored_version == cls._LEGACY_EMITTER_SCHEMA_VERSION:
            mgr._migrate_legacy_emitter_rules(snapshot_dir / "events.parquet")

        # Advance sequences past imported IDs so new inserts don't collide.
        # DuckDB lacks setval(), so we consume nextval() until we're past max(id).
        # observation_contexts uses an externally-supplied ID, no sequence.
        # TODO: switch to content-based IDs to eliminate sequence advancement and avoid initialization order issues.
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
        overhead. Lookup tables and observation contexts remain immutable.
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
