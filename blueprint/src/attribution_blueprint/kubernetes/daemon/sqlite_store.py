# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from attribution_blueprint.kubernetes.daemon.models import PlacementRecord
from attribution_blueprint.kubernetes.daemon.store import CreateResult, PlacementConflictError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS placement_record (
    pod_uid TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    pod_name TEXT NOT NULL,
    node_name TEXT NOT NULL,
    owner_uid TEXT,
    node_failure_reason TEXT,
    raw_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS placement_record_owner
    ON placement_record(namespace, owner_uid);
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
            self._validate_schema(connection)
            connection.executescript(_SCHEMA)
            connection.execute("PRAGMA user_version = 2")
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

    def get_by_pod_uid(self, pod_uid: str) -> PlacementRecord | None:
        with self._connection() as connection:
            self._prune_expired(connection)
            row = connection.execute(
                "SELECT raw_json FROM placement_record WHERE pod_uid = ?",
                (pod_uid,),
            ).fetchone()
        return PlacementRecord.from_resource(json.loads(row["raw_json"])) if row else None

    def list_for_pod(self, namespace: str, pod_uid: str) -> list[PlacementRecord]:
        record = self.get_by_pod_uid(pod_uid)
        return [record] if record is not None and record.namespace == namespace else []

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

    def list_for_owner(self, namespace: str, owner_uid: str) -> list[PlacementRecord]:
        with self._connection() as connection:
            self._prune_expired(connection)
            rows = connection.execute(
                """
                SELECT raw_json FROM placement_record
                WHERE namespace = ? AND owner_uid = ?
                ORDER BY pod_uid
                """,
                (namespace, owner_uid),
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

    def _insert(self, connection: sqlite3.Connection, record: PlacementRecord) -> None:
        raw_json = json.dumps(record.to_resource(), separators=(",", ":"), sort_keys=True)
        connection.execute(
            """
            INSERT INTO placement_record(
                pod_uid, namespace, pod_name, node_name, owner_uid,
                node_failure_reason, raw_json, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.pod_uid,
                record.namespace,
                record.pod_name,
                record.node_name,
                record.owner.uid if record.owner else None,
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
            SET owner_uid = ?, node_failure_reason = ?, raw_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE pod_uid = ?
            """,
            (
                record.owner.uid if record.owner else None,
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
    def _validate_schema(connection: sqlite3.Connection) -> None:
        columns = connection.execute("PRAGMA table_info(placement_record)").fetchall()
        if not columns or any(row["name"] == "pod_name" for row in columns):
            return
        raise RuntimeError("existing placement database requires compatibility migration before KubernetesDaemon can use it")

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


def _placement_identity(record: PlacementRecord) -> tuple:
    return (
        record.namespace,
        record.pod_name,
        record.pod_uid,
        record.node_name,
    )
