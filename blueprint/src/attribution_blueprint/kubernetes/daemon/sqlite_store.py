# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from attribution_blueprint.kubernetes.daemon.models import (
    OwnerReference,
    PlacementRecord,
    ResourceObservation,
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
CREATE TABLE IF NOT EXISTS placement_deletion (
    pod_uid TEXT PRIMARY KEY,
    deleted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(pod_uid) REFERENCES placement_record(pod_uid) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS placement_deletion_expiration
    ON placement_deletion(deleted_at);
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
CREATE INDEX IF NOT EXISTS resource_observation_kind
    ON resource_observation(cluster_id, namespace, kind, id);
CREATE TABLE IF NOT EXISTS observation_owner_reference (
    observation_id INTEGER NOT NULL,
    owner_uid TEXT NOT NULL,
    controller INTEGER NOT NULL,
    PRIMARY KEY(observation_id, owner_uid),
    FOREIGN KEY(observation_id) REFERENCES resource_observation(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS observation_owner_identity
    ON observation_owner_reference(owner_uid, observation_id);
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
        with self._connection() as connection:
            connection.executescript(_SCHEMA)
            self._migrate_observation_schema(connection)
            connection.execute("PRAGMA user_version = 4")
            self._prune_expired(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, record: PlacementRecord) -> CreateResult:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prune_expired(connection)
            return self._create(connection, record)

    def get_by_pod_uid(self, pod_uid: str) -> PlacementRecord | None:
        with self._connection() as connection:
            self._prune_expired(connection)
            row = connection.execute(
                "SELECT raw_json FROM placement_record WHERE pod_uid = ?",
                (pod_uid,),
            ).fetchone()
        return PlacementRecord.from_resource(json.loads(row["raw_json"])) if row else None

    def list_for_resource(self, namespace: str, resource_uid: str) -> list[PlacementRecord]:
        with self._connection() as connection:
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

    def mark_deleted(self, pod_uid: str) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO placement_deletion(pod_uid)
                SELECT pod_uid FROM placement_record WHERE pod_uid = ?
                ON CONFLICT(pod_uid) DO NOTHING
                """,
                (pod_uid,),
            )
            self._prune_expired(connection)

    def reconcile_active_pods(
        self,
        namespace: str,
        active_pod_uids: set[str],
    ) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TEMP TABLE active_pod_uid(pod_uid TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            connection.executemany(
                "INSERT INTO active_pod_uid(pod_uid) VALUES (?)",
                [(pod_uid,) for pod_uid in active_pod_uids],
            )
            connection.execute(
                """
                INSERT INTO placement_deletion(pod_uid)
                SELECT record.pod_uid
                FROM placement_record AS record
                LEFT JOIN active_pod_uid AS active USING (pod_uid)
                WHERE record.namespace = ? AND active.pod_uid IS NULL
                ON CONFLICT(pod_uid) DO NOTHING
                """,
                (namespace,),
            )
            self._prune_expired(connection)

    def append_observation(self, observation: ResourceObservation) -> ObservationAppendResult:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = self._append_observation(connection, observation)
            self._prune_expired(connection)
        return result

    def capture_observation(
        self,
        observation: ResourceObservation,
        *,
        placement: PlacementRecord | None,
        mark_deleted: bool = False,
    ) -> CaptureResult:
        if mark_deleted and placement is None:
            raise ValueError("mark_deleted requires a placement")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            observation_result = self._append_observation(connection, observation)
            placement_result = self._create(connection, placement) if placement is not None else None
            if mark_deleted:
                connection.execute(
                    """
                    INSERT INTO placement_deletion(pod_uid)
                    VALUES (?)
                    ON CONFLICT(pod_uid) DO NOTHING
                    """,
                    (placement.pod_uid,),
                )
            self._prune_expired(connection)
            return CaptureResult(observation_result, placement_result)

    def list_observations_for_uid(
        self,
        cluster_id: str,
        namespace: str,
        uid: str,
    ) -> list[ResourceObservation]:
        with self._connection() as connection:
            self._prune_expired(connection)
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
        with self._connection() as connection:
            self._prune_expired(connection)
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
        if _failure_evidence_rank(record.node_failure_reason) > _failure_evidence_rank(
            failure_reason
        ):
            failure_reason = record.node_failure_reason
        merge_base = existing
        if record.owner is not None and record.owner != existing.owner:
            merge_base = replace(existing, associated_resources=())
        merged = merge_base.enriched(
            scheduled_at=record.scheduled_at,
            owner_references=record.owner_references if record.owner is not None else None,
            owner=record.owner,
            associated_resources=record.associated_resources,
            node_failure_reason=failure_reason,
        )
        if merged == existing:
            return CreateResult.DUPLICATE
        self._update(connection, merged)
        return CreateResult.UPDATED

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

    @staticmethod
    def _migrate_observation_schema(connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(resource_observation)").fetchall()
        }
        if "selected_metadata_json" not in columns:
            connection.execute(
                "ALTER TABLE resource_observation ADD COLUMN selected_metadata_json TEXT"
            )

    def _prune_expired(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM placement_record
            WHERE pod_uid IN (
                SELECT pod_uid
                FROM placement_deletion
                WHERE deleted_at < datetime('now', ?)
            )
            """,
            (f"-{self.retention_days} days",),
        )
        rows = connection.execute(
            """
            WITH latest AS (
                SELECT cluster_id, namespace, uid, MAX(id) AS observation_id
                FROM resource_observation
                GROUP BY cluster_id, namespace, uid
            )
            SELECT observation.*
            FROM resource_observation AS observation
            JOIN latest ON latest.observation_id = observation.id
            """
        ).fetchall()
        observed = {
            (row["cluster_id"], row["namespace"], row["uid"])
            for row in rows
        }
        candidates: dict[tuple[str, str, str], sqlite3.Row] = {}
        for row in rows:
            controller_uid = _controller_uid(row)
            if row["kind"] in {"Workload", "PodGroup"}:
                root_uid = controller_uid or row["uid"]
                key = (row["cluster_id"], row["namespace"], root_uid)
                current = candidates.get(key)
                if (
                    current is None
                    or current["kind"] not in {"Workload", "PodGroup"}
                    or row["id"] > current["id"]
                ):
                    candidates[key] = row
            elif controller_uid is None or (
                row["cluster_id"],
                row["namespace"],
                controller_uid,
            ) not in observed:
                key = (row["cluster_id"], row["namespace"], row["uid"])
                candidates.setdefault(key, row)
        for (cluster_id, namespace, controller_uid), candidate in candidates.items():
            if not _expired_terminal_resource(
                connection,
                candidate,
                self.retention_days,
            ):
                continue
            connection.execute(
                """
                WITH RECURSIVE related(uid) AS (
                    SELECT ?
                    UNION
                    SELECT observation.uid
                    FROM resource_observation AS observation
                    JOIN observation_owner_reference AS owner
                      ON owner.observation_id = observation.id
                    JOIN related ON related.uid = owner.owner_uid
                    WHERE observation.cluster_id = ?
                      AND observation.namespace = ?
                )
                DELETE FROM resource_observation
                WHERE cluster_id = ? AND namespace = ?
                  AND uid IN (SELECT uid FROM related)
                """,
                (
                    controller_uid,
                    cluster_id,
                    namespace,
                    cluster_id,
                    namespace,
                ),
            )


def _controller_uid(row: sqlite3.Row) -> str | None:
    controllers = [
        owner.get("uid")
        for owner in json.loads(row["owner_references_json"])
        if owner.get("controller") is True and owner.get("uid")
    ]
    return controllers[0] if len(controllers) == 1 else None


def _expired_terminal_resource(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    retention_days: int,
) -> bool:
    terminal_at = _terminal_at(row)
    if terminal_at is None:
        return False
    return bool(
        connection.execute(
            "SELECT datetime(?) < datetime('now', ?)",
            (terminal_at, f"-{retention_days} days"),
        ).fetchone()[0]
    )


def _terminal_at(row: sqlite3.Row) -> str | None:
    if row["event_type"] == "DELETED":
        return row["observed_at"]
    status = json.loads(row["status_json"]) if row["status_json"] else {}
    transitions = [
        condition.get("lastTransitionTime")
        for condition in status.get("conditions") or []
        if condition.get("type") in {
            "Complete",
            "Completed",
            "Failed",
            "Finished",
        }
        and condition.get("status") == "True"
        and condition.get("lastTransitionTime")
    ]
    if transitions:
        return max(transitions)
    if status.get("phase") in {"Failed", "Succeeded"} or status.get(
        "terminalState"
    ):
        return row["source_timestamp"] or row["observed_at"]
    return status.get("completionTime")


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
