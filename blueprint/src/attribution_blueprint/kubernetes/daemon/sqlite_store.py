# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from attribution_blueprint.kubernetes.daemon.models import (
    CoverageGap,
    CoverageGapReason,
    OwnerReference,
    PlacementRecord,
    ResourceObservation,
    WatchCoverage,
)
from attribution_blueprint.kubernetes.daemon.store import (
    CaptureResult,
    CreateResult,
    ObservationAppendResult,
    ObservationConflictError,
    PlacementConflictError,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS placement_record (
    pod_uid TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    pod_name TEXT NOT NULL,
    node_name TEXT NOT NULL,
    node_failure_reason TEXT,
    raw_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS placement_resource (
    pod_uid TEXT NOT NULL,
    namespace TEXT NOT NULL,
    resource_uid TEXT NOT NULL,
    PRIMARY KEY(pod_uid, resource_uid),
    FOREIGN KEY(pod_uid) REFERENCES placement_record(pod_uid) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS placement_resource_identity
    ON placement_resource(namespace, resource_uid);
CREATE TABLE IF NOT EXISTS resource_observation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id TEXT NOT NULL,
    api_version TEXT NOT NULL,
    kind TEXT NOT NULL,
    namespace TEXT NOT NULL,
    name TEXT NOT NULL,
    uid TEXT NOT NULL,
    resource_version TEXT NOT NULL,
    event_type TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    source_timestamp TEXT,
    owner_references_json TEXT NOT NULL,
    selected_metadata_json TEXT,
    status_json TEXT,
    UNIQUE(cluster_id, api_version, kind, uid, resource_version)
);
CREATE INDEX IF NOT EXISTS resource_observation_identity
    ON resource_observation(cluster_id, namespace, uid, id);
CREATE TABLE IF NOT EXISTS observation_owner_reference (
    observation_id INTEGER NOT NULL,
    owner_uid TEXT NOT NULL,
    controller INTEGER NOT NULL,
    PRIMARY KEY(observation_id, owner_uid),
    FOREIGN KEY(observation_id) REFERENCES resource_observation(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS observation_owner_identity
    ON observation_owner_reference(owner_uid, observation_id);
CREATE TABLE IF NOT EXISTS watch_coverage (
    cluster_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    source_name TEXT NOT NULL,
    api_version TEXT NOT NULL,
    kind TEXT NOT NULL,
    coverage_started_at TEXT NOT NULL,
    resource_version TEXT NOT NULL,
    checkpointed_at TEXT NOT NULL,
    PRIMARY KEY(cluster_id, namespace, source_name)
);
CREATE TABLE IF NOT EXISTS coverage_gap (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    source_name TEXT NOT NULL,
    api_version TEXT NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT NOT NULL,
    start_resource_version TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    end_resource_version TEXT,
    recovered_at TEXT
);
CREATE INDEX IF NOT EXISTS coverage_gap_scope
    ON coverage_gap(cluster_id, namespace, detected_at);
"""


def _failure_evidence_rank(reason: str | None) -> int:
    if not reason:
        return 0
    if reason.startswith("ExitCode:"):
        return 1
    return 2


class SQLitePlacementStore:
    """Scheduler-neutral placement view backed by a persistent SQLite file."""

    def __init__(self, path: str, *, timeout_seconds: float = 30.0, retention_days: int = 30):
        if not path:
            raise ValueError("SQLite placement path must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if isinstance(retention_days, bool) or not isinstance(retention_days, int) or retention_days <= 0:
            raise ValueError("retention_days must be a positive integer")
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.retention_days = retention_days
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute("PRAGMA user_version = 3")
            self._prune_expired(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def create(self, record: PlacementRecord) -> CreateResult:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prune_expired(connection)
            return self._create(connection, record)

    def get_by_pod_uid(self, pod_uid: str) -> PlacementRecord | None:
        with self._connect() as connection:
            self._prune_expired(connection)
            row = connection.execute(
                "SELECT raw_json FROM placement_record WHERE pod_uid = ?",
                (pod_uid,),
            ).fetchone()
        return PlacementRecord.from_resource(json.loads(row["raw_json"])) if row else None

    def list_for_resource(self, namespace: str, resource_uid: str) -> list[PlacementRecord]:
        with self._connect() as connection:
            self._prune_expired(connection)
            rows = connection.execute(
                """
                SELECT record.raw_json
                FROM placement_record AS record
                JOIN placement_resource AS resource USING (pod_uid)
                WHERE resource.namespace = ?
                  AND resource.resource_uid = ?
                ORDER BY record.pod_uid
                """,
                (namespace, resource_uid),
            ).fetchall()
        return [PlacementRecord.from_resource(json.loads(row["raw_json"])) for row in rows]

    def capture_observation(
        self,
        observation: ResourceObservation,
        *,
        source_name: str,
        placement: PlacementRecord | None,
        advance_checkpoint: bool,
    ) -> CaptureResult:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            observation_result = self._append_observation(connection, observation)
            placement_result = self._create(connection, placement) if placement else None
            if advance_checkpoint:
                self._mark_watch_checkpoint(
                    connection,
                    cluster_id=observation.cluster_id,
                    namespace=observation.namespace,
                    source_name=source_name,
                    resource_version=observation.resource_version,
                    checkpointed_at=observation.observed_at,
                )
            return CaptureResult(observation_result, placement_result)

    def _append_observation(
        self,
        connection: sqlite3.Connection,
        observation: ResourceObservation,
    ) -> ObservationAppendResult:
        owner_references_json = _owner_references_json(observation.owner_references)
        selected_metadata_json = _optional_json(observation.selected_metadata)
        status_json = _optional_json(observation.status)
        row = connection.execute(
            """
            SELECT *
            FROM resource_observation
            WHERE cluster_id = ? AND api_version = ? AND kind = ?
              AND uid = ? AND resource_version = ?
            """,
            (
                observation.cluster_id,
                observation.api_version,
                observation.kind,
                observation.uid,
                observation.resource_version,
            ),
        ).fetchone()
        if row is not None:
            existing = _observation_from_row(row)
            if _observation_content(existing) != _observation_content(observation):
                raise ObservationConflictError(
                    f"resource {observation.kind} {observation.namespace}/{observation.name} "
                    f"UID {observation.uid} version {observation.resource_version} changed during replay"
                )
            return ObservationAppendResult.DUPLICATE

        cursor = connection.execute(
            """
            INSERT INTO resource_observation(
                cluster_id, api_version, kind, namespace, name, uid,
                resource_version, event_type, observed_at, source_timestamp,
                owner_references_json, selected_metadata_json, status_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation.cluster_id,
                observation.api_version,
                observation.kind,
                observation.namespace,
                observation.name,
                observation.uid,
                observation.resource_version,
                observation.event_type,
                observation.observed_at,
                observation.source_timestamp,
                owner_references_json,
                selected_metadata_json,
                status_json,
            ),
        )
        connection.executemany(
            """
            INSERT INTO observation_owner_reference(observation_id, owner_uid, controller)
            VALUES (?, ?, ?)
            """,
            [
                (cursor.lastrowid, owner.uid, int(owner.controller))
                for owner in observation.owner_references
            ],
        )
        return ObservationAppendResult.APPENDED

    def list_observations_for_uid(
        self,
        cluster_id: str,
        namespace: str,
        uid: str,
    ) -> list[ResourceObservation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM resource_observation
                WHERE cluster_id = ? AND namespace = ? AND uid = ?
                ORDER BY id
                """,
                (cluster_id, namespace, uid),
            ).fetchall()
        return [_observation_from_row(row) for row in rows]

    def list_observations_for_owner(
        self,
        cluster_id: str,
        namespace: str,
        owner_uid: str,
    ) -> list[ResourceObservation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT observation.*
                FROM resource_observation AS observation
                JOIN observation_owner_reference AS owner
                  ON owner.observation_id = observation.id
                WHERE observation.cluster_id = ?
                  AND observation.namespace = ?
                  AND owner.owner_uid = ?
                ORDER BY observation.id
                """,
                (cluster_id, namespace, owner_uid),
            ).fetchall()
        return [_observation_from_row(row) for row in rows]

    def list_latest_observations_for_kind(
        self,
        cluster_id: str,
        namespace: str,
        kind: str,
    ) -> list[ResourceObservation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                WITH latest AS (
                    SELECT uid, MAX(id) AS observation_id
                    FROM resource_observation
                    WHERE cluster_id = ? AND namespace = ? AND kind = ?
                    GROUP BY uid
                )
                SELECT observation.*
                FROM resource_observation AS observation
                JOIN latest ON latest.observation_id = observation.id
                ORDER BY observation.id
                """,
                (cluster_id, namespace, kind),
            ).fetchall()
        return [_observation_from_row(row) for row in rows]

    def get_watch_coverage(
        self,
        cluster_id: str,
        namespace: str,
        source_name: str,
    ) -> WatchCoverage | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM watch_coverage
                WHERE cluster_id = ? AND namespace = ? AND source_name = ?
                """,
                (cluster_id, namespace, source_name),
            ).fetchone()
        return _watch_coverage_from_row(row) if row else None

    def begin_coverage_gap(
        self,
        *,
        cluster_id: str,
        namespace: str,
        source_name: str,
        api_version: str,
        kind: str,
        reason: CoverageGapReason,
        start_resource_version: str,
        detected_at: str,
    ) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT id
                FROM coverage_gap
                WHERE cluster_id = ? AND namespace = ? AND source_name = ?
                  AND recovered_at IS NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (cluster_id, namespace, source_name),
            ).fetchone()
            if existing:
                return int(existing["id"])
            cursor = connection.execute(
                """
                INSERT INTO coverage_gap(
                    cluster_id, namespace, source_name, api_version, kind,
                    reason, start_resource_version, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cluster_id,
                    namespace,
                    source_name,
                    api_version,
                    kind,
                    reason,
                    start_resource_version,
                    detected_at,
                ),
            )
            return int(cursor.lastrowid)

    def complete_watch_list(
        self,
        *,
        cluster_id: str,
        namespace: str,
        source_name: str,
        api_version: str,
        kind: str,
        resource_version: str,
        completed_at: str,
        gap_id: int | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO watch_coverage(
                    cluster_id, namespace, source_name, api_version, kind,
                    coverage_started_at, resource_version, checkpointed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cluster_id, namespace, source_name) DO UPDATE SET
                    api_version = excluded.api_version,
                    kind = excluded.kind,
                    resource_version = excluded.resource_version,
                    checkpointed_at = excluded.checkpointed_at
                """,
                (
                    cluster_id,
                    namespace,
                    source_name,
                    api_version,
                    kind,
                    completed_at,
                    resource_version,
                    completed_at,
                ),
            )
            if gap_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE coverage_gap
                    SET end_resource_version = ?, recovered_at = ?
                    WHERE id = ? AND cluster_id = ? AND namespace = ?
                      AND source_name = ? AND recovered_at IS NULL
                    """,
                    (
                        resource_version,
                        completed_at,
                        gap_id,
                        cluster_id,
                        namespace,
                        source_name,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"coverage gap {gap_id} is missing or already completed")

    def mark_watch_checkpoint(
        self,
        *,
        cluster_id: str,
        namespace: str,
        source_name: str,
        resource_version: str,
        checkpointed_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._mark_watch_checkpoint(
                connection,
                cluster_id=cluster_id,
                namespace=namespace,
                source_name=source_name,
                resource_version=resource_version,
                checkpointed_at=checkpointed_at,
            )

    def list_coverage_gaps(
        self,
        cluster_id: str,
        namespace: str,
    ) -> list[CoverageGap]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM coverage_gap
                WHERE cluster_id = ? AND namespace = ?
                ORDER BY id
                """,
                (cluster_id, namespace),
            ).fetchall()
        return [_coverage_gap_from_row(row) for row in rows]

    @staticmethod
    def _mark_watch_checkpoint(
        connection: sqlite3.Connection,
        *,
        cluster_id: str,
        namespace: str,
        source_name: str,
        resource_version: str,
        checkpointed_at: str,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE watch_coverage
            SET resource_version = ?, checkpointed_at = ?
            WHERE cluster_id = ? AND namespace = ? AND source_name = ?
            """,
            (
                resource_version,
                checkpointed_at,
                cluster_id,
                namespace,
                source_name,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"watch coverage for source {source_name!r} is not initialized")

    def _create(
        self,
        connection: sqlite3.Connection,
        record: PlacementRecord,
    ) -> CreateResult:
        row = connection.execute(
            "SELECT raw_json FROM placement_record WHERE pod_uid = ?",
            (record.pod_uid,),
        ).fetchone()
        if row is None:
            self._insert(connection, record)
            return CreateResult.CREATED

        existing = PlacementRecord.from_resource(json.loads(row["raw_json"]))
        if _placement_identity(existing) != _placement_identity(record):
            raise PlacementConflictError(
                f"placement for Pod UID {record.pod_uid} already exists with different immutable data"
            )
        if (
            existing.scheduled_at
            and record.scheduled_at
            and existing.scheduled_at != record.scheduled_at
        ):
            raise PlacementConflictError(
                f"placement for Pod UID {record.pod_uid} has conflicting scheduling timestamps"
            )
        failure_reason = existing.node_failure_reason
        if _failure_evidence_rank(record.node_failure_reason) > _failure_evidence_rank(failure_reason):
            failure_reason = record.node_failure_reason
        merged = existing.enriched(
            scheduled_at=record.scheduled_at,
            owner=record.owner,
            associated_resources=record.associated_resources,
            node_failure_reason=failure_reason,
        )
        if merged == existing:
            return CreateResult.DUPLICATE
        self._update(connection, merged)
        return CreateResult.UPDATED

    def _insert(self, connection: sqlite3.Connection, record: PlacementRecord) -> None:
        raw_json = json.dumps(record.to_resource(), separators=(",", ":"), sort_keys=True)
        connection.execute(
            """
            INSERT INTO placement_record(
                pod_uid, namespace, pod_name, node_name,
                node_failure_reason, raw_json, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.pod_uid,
                record.namespace,
                record.pod_name,
                record.node_name,
                record.node_failure_reason,
                raw_json,
                record.observed_at,
            ),
        )
        self._replace_resources(connection, record)

    def _update(self, connection: sqlite3.Connection, record: PlacementRecord) -> None:
        raw_json = json.dumps(record.to_resource(), separators=(",", ":"), sort_keys=True)
        connection.execute(
            """
            UPDATE placement_record
            SET node_failure_reason = ?, raw_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE pod_uid = ?
            """,
            (
                record.node_failure_reason,
                raw_json,
                record.pod_uid,
            ),
        )
        self._replace_resources(connection, record)

    @staticmethod
    def _replace_resources(connection: sqlite3.Connection, record: PlacementRecord) -> None:
        connection.execute("DELETE FROM placement_resource WHERE pod_uid = ?", (record.pod_uid,))
        connection.executemany(
            """
            INSERT INTO placement_resource(
                pod_uid, namespace, resource_uid
            ) VALUES (?, ?, ?)
            """,
            [
                (record.pod_uid, record.namespace, resource.uid)
                for resource in record.associated_resources
            ],
        )

    def _prune_expired(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "DELETE FROM placement_record WHERE updated_at < datetime('now', ?)",
            (f"-{self.retention_days} days",),
        )


def _placement_identity(record: PlacementRecord) -> tuple:
    return (
        record.namespace,
        record.pod_name,
        record.pod_uid,
        record.node_name,
    )


def _owner_references_json(owner_references: tuple[OwnerReference, ...]) -> str:
    return json.dumps(
        [
            {
                "apiVersion": owner.api_version,
                "kind": owner.kind,
                "name": owner.name,
                "uid": owner.uid,
                "controller": owner.controller,
            }
            for owner in owner_references
        ],
        separators=(",", ":"),
        sort_keys=True,
    )


def _optional_json(value: dict | None) -> str | None:
    return json.dumps(value, separators=(",", ":"), sort_keys=True) if value is not None else None


def _observation_from_row(row: sqlite3.Row) -> ResourceObservation:
    return ResourceObservation(
        cluster_id=row["cluster_id"],
        api_version=row["api_version"],
        kind=row["kind"],
        namespace=row["namespace"],
        name=row["name"],
        uid=row["uid"],
        resource_version=row["resource_version"],
        event_type=row["event_type"],
        observed_at=row["observed_at"],
        source_timestamp=row["source_timestamp"],
        owner_references=tuple(
            OwnerReference(
                api_version=owner.get("apiVersion") or "",
                kind=owner["kind"],
                name=owner["name"],
                uid=owner["uid"],
                controller=bool(owner["controller"]),
            )
            for owner in json.loads(row["owner_references_json"])
        ),
        selected_metadata=(
            json.loads(row["selected_metadata_json"])
            if row["selected_metadata_json"] is not None
            else None
        ),
        status=json.loads(row["status_json"]) if row["status_json"] is not None else None,
    )


def _observation_content(observation: ResourceObservation) -> tuple:
    return (
        observation.cluster_id,
        observation.api_version,
        observation.kind,
        observation.namespace,
        observation.name,
        observation.uid,
        observation.resource_version,
        observation.owner_references,
        _optional_json(observation.selected_metadata),
        observation.source_timestamp,
        _optional_json(observation.status),
    )


def _watch_coverage_from_row(row: sqlite3.Row) -> WatchCoverage:
    return WatchCoverage(
        cluster_id=row["cluster_id"],
        source_name=row["source_name"],
        api_version=row["api_version"],
        kind=row["kind"],
        namespace=row["namespace"],
        coverage_started_at=row["coverage_started_at"],
        resource_version=row["resource_version"],
        checkpointed_at=row["checkpointed_at"],
    )


def _coverage_gap_from_row(row: sqlite3.Row) -> CoverageGap:
    return CoverageGap(
        id=row["id"],
        cluster_id=row["cluster_id"],
        source_name=row["source_name"],
        api_version=row["api_version"],
        kind=row["kind"],
        namespace=row["namespace"],
        reason=row["reason"],
        start_resource_version=row["start_resource_version"],
        detected_at=row["detected_at"],
        end_resource_version=row["end_resource_version"],
        recovered_at=row["recovered_at"],
    )
