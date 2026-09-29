# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import PlacementRecord
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.store import (
    CreateResult,
    PlacementConflictError,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS placement_record (
    pod_uid TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    workload_uid TEXT NOT NULL,
    node_name TEXT NOT NULL,
    node_failure_reason TEXT,
    raw_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS placement_record_workload
    ON placement_record(namespace, workload_uid);
"""


def _failure_evidence_rank(reason: str | None) -> int:
    """Rank failure evidence strength: no evidence < generic container exit
    < explicit node-loss.

    Two node-loss-tier reasons (e.g. `NodeLost` vs `DeletionByPodGC`) rank
    equally: they are different Kubernetes signaling mechanisms for the
    same underlying fact and are semantically equivalent for attribution.
    """
    if not reason:
        return 0
    if reason.startswith("ExitCode:"):
        return 1
    return 2


class SQLitePlacementStore:
    """Single-writer placement store backed by a persistent SQLite file."""

    def __init__(self, path: str, *, timeout_seconds: float = 30.0, retention_days: int = 30):
        if not path:
            raise ValueError("SQLite placement path must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("SQLite timeout_seconds must be positive")
        if isinstance(retention_days, bool) or not isinstance(retention_days, int) or retention_days <= 0:
            raise ValueError("SQLite retention_days must be a positive integer")
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.retention_days = retention_days
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute("PRAGMA user_version = 1")
            self._prune_expired(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def create(self, record: PlacementRecord) -> CreateResult:
        resource = record.to_resource()
        raw_json = json.dumps(resource, separators=(",", ":"), sort_keys=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prune_expired(connection)
            row = connection.execute(
                "SELECT raw_json FROM placement_record WHERE pod_uid = ?",
                (record.pod_uid,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO placement_record(
                        pod_uid, namespace, workload_uid, node_name,
                        node_failure_reason, raw_json, observed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.pod_uid,
                        record.namespace,
                        record.workload.uid,
                        record.node_name,
                        record.node_failure_reason,
                        raw_json,
                        record.observed_at,
                    ),
                )
                return CreateResult.CREATED

            existing = PlacementRecord.from_resource(json.loads(row["raw_json"]))
            if _placement_identity(existing) != _placement_identity(record):
                raise PlacementConflictError(
                    f"placement for Pod UID {record.pod_uid} already exists with different immutable data"
                )
            if not record.node_failure_reason or record.node_failure_reason == existing.node_failure_reason:
                return CreateResult.DUPLICATE
            if _failure_evidence_rank(record.node_failure_reason) <= _failure_evidence_rank(
                existing.node_failure_reason
            ):
                # A downgrade (e.g. NodeLost already stored, ExitCode:137
                # arrives later), or a different-but-equivalent-tier reason
                # (e.g. NodeLost vs DeletionByPodGC): keep the evidence
                # already stored rather than raising a conflict or churning
                # the row for a change that wouldn't affect attribution.
                return CreateResult.DUPLICATE

            connection.execute(
                """
                UPDATE placement_record
                SET node_failure_reason = ?, raw_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE pod_uid = ?
                """,
                (record.node_failure_reason, raw_json, record.pod_uid),
            )
            return CreateResult.UPDATED

    def get_by_pod_uid(self, pod_uid: str) -> PlacementRecord | None:
        with self._connect() as connection:
            self._prune_expired(connection)
            row = connection.execute(
                "SELECT raw_json FROM placement_record WHERE pod_uid = ?",
                (pod_uid,),
            ).fetchone()
        if row is None:
            return None
        return PlacementRecord.from_resource(json.loads(row["raw_json"]))

    def list_for_workload(self, namespace: str, workload_uid: str) -> list[PlacementRecord]:
        with self._connect() as connection:
            self._prune_expired(connection)
            rows = connection.execute(
                """
                SELECT raw_json
                FROM placement_record
                WHERE namespace = ? AND workload_uid = ?
                ORDER BY pod_uid
                """,
                (namespace, workload_uid),
            ).fetchall()
        return [PlacementRecord.from_resource(json.loads(row["raw_json"])) for row in rows]

    def _prune_expired(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "DELETE FROM placement_record WHERE updated_at < datetime('now', ?)",
            (f"-{self.retention_days} days",),
        )


def _placement_identity(record: PlacementRecord) -> tuple:
    return (
        record.namespace,
        record.workload,
        record.owner,
        record.pod_name,
        record.pod_uid,
        record.node_name,
        record.scheduled_at,
    )
