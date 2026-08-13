# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Event recorder for the attribution decision graph."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from attribution_lib.exceptions import DecisionGraphError
from attribution_lib.sources import Source

if TYPE_CHECKING:
    from attribution_lib.attr_types import JobNodeAttributions


class NodeType(str, Enum):
    EXTRACTION = "EXTRACTION"
    FILTER = "FILTER"
    SCRUB = "SCRUB"
    REFINER = "REFINER"
    CASCADE_PRUNE = "CASCADE_PRUNE"
    OUTCOME = "OUTCOME"


class Action(str, Enum):
    CREATED = "CREATED"
    MATCHED = "MATCHED"
    REMOVED = "REMOVED"
    PRUNED = "PRUNED"


@dataclass
class DecisionEvent:
    node_type: NodeType
    rule_name: str
    action: Action
    symptom: str
    hostname: Optional[str] = None
    reason: Optional[str] = None
    evidence: Optional[dict[str, Any]] = None
    sequence_num: int = 0


EvidencePayload = dict[str, Any]


@dataclass(frozen=True)
class SourceEvidence:
    """Serializable evidence for one symptom source in a decision trace."""

    rule_name: Optional[str]
    datum: Optional[str]
    line_num: Optional[int]
    timestamp: Optional[str]
    obs_source: Optional[str]
    hostname: Optional[str] = None

    @classmethod
    def from_source(cls, source: Source, hostname: Optional[str] = None) -> "SourceEvidence":
        observation_source = source.source_context.source if source.source_context is not None else None
        timestamp = source.time
        return cls(
            rule_name=source.rule,
            datum=source.datum,
            line_num=source.line_num,
            timestamp=timestamp.isoformat() if timestamp is not None else None,
            obs_source=observation_source.value if observation_source is not None else None,
            hostname=hostname,
        )

    @classmethod
    def from_payload(cls, payload: EvidencePayload, hostname: Optional[str] = None) -> "SourceEvidence":
        """Build source evidence from a recorder payload.

        Args:
            payload: Serialized source evidence.
            hostname: Fallback hostname when the payload does not provide one.

        Returns:
            Validated source evidence.

        Raises:
            DecisionGraphError: If the payload cannot form a stable extraction identity.
        """
        if not isinstance(payload, dict):
            raise DecisionGraphError("Decision source evidence must be a mapping")

        field_types = {
            "rule_name": str,
            "datum": str,
            "line_num": int,
            "timestamp": str,
            "obs_source": str,
            "hostname": str,
        }
        for field_name, expected_type in field_types.items():
            value = payload.get(field_name)
            if value is not None and not isinstance(value, expected_type):
                raise DecisionGraphError(
                    f"Decision source evidence field {field_name!r} must be {expected_type.__name__} or None"
                )

        payload_hostname = payload.get("hostname", hostname)
        return cls(
            rule_name=payload.get("rule_name"),
            datum=payload.get("datum"),
            line_num=payload.get("line_num"),
            timestamp=payload.get("timestamp"),
            obs_source=payload.get("obs_source"),
            hostname=payload_hostname,
        )

    def to_payload(self) -> EvidencePayload:
        return {
            "rule_name": self.rule_name,
            "datum": self.datum,
            "line_num": self.line_num,
            "timestamp": self.timestamp,
            "obs_source": self.obs_source,
            "hostname": self.hostname,
        }


NodeSymptomSnapshot = dict[str, dict[str, list[SourceEvidence]]]


def _snapshot_sources(snapshot: NodeSymptomSnapshot, node: str, symptom: str) -> list[EvidencePayload]:
    per_node = snapshot.get(node, {})
    return [source.to_payload() for source in per_node.get(symptom, [])]


def _snapshot_sources_for_symptom(snapshot: NodeSymptomSnapshot, symptom: str) -> list[EvidencePayload]:
    sources: list[EvidencePayload] = []
    for node in sorted(snapshot):
        sources.extend(_snapshot_sources(snapshot, node, symptom))
    return sources


class DecisionRecorder:
    """Collects DecisionEvents as the attribution pipeline runs.

    Optional — when absent, instrumented code skips recording.

    NOTE: The recorder is currently threaded through the pipeline as an
    Optional[DecisionRecorder] parameter.  A future evolution is to store the
    active recorder in a ``ContextVar`` (similar to the rules-engine context
    var in ``rules_engine.py``) so instrumented functions can access it
    implicitly, eliminating the parameter threading.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._events: list[DecisionEvent] = []
        self._seq: int = 0

    def record(
        self,
        node_type: NodeType,
        rule_name: str,
        action: Action,
        symptom: str,
        *,
        hostname: Optional[str] = None,
        reason: Optional[str] = None,
        evidence: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record one decision event."""
        event = DecisionEvent(
            node_type=node_type,
            rule_name=rule_name,
            action=action,
            symptom=symptom,
            hostname=hostname,
            reason=reason,
            evidence=evidence,
            sequence_num=self._seq + 1,
        )

        self._seq += 1
        self._events.append(event)

    def record_refiner_diff(
        self,
        rule_name: str,
        before_symptoms: set[str],
        after_symptoms: set[str],
        before_bad: NodeSymptomSnapshot,
        after_bad: NodeSymptomSnapshot,
    ) -> None:
        """Compare before/after snapshots and record REFINER/SCRUB events for any changes.

        Args:
            rule_name: Name of the refiner that ran.
            before_symptoms: Job-level symptom set before the refiner.
            after_symptoms: Job-level symptom set after the refiner.
            before_bad: Per-node symptom snapshot before the refiner.
            after_bad: Per-node symptom snapshot after the refiner.
        """
        removed = before_symptoms - after_symptoms
        for symptom in sorted(removed):
            self.record(
                node_type=NodeType.REFINER,
                rule_name=rule_name,
                action=Action.REMOVED,
                symptom=symptom,
                evidence={"sources": _snapshot_sources_for_symptom(before_bad, symptom)},
            )

        added = after_symptoms - before_symptoms
        for symptom in sorted(added):
            self.record(
                node_type=NodeType.REFINER,
                rule_name=rule_name,
                action=Action.CREATED,
                symptom=symptom,
                evidence={"sources": _snapshot_sources_for_symptom(after_bad, symptom)},
            )

        for node in sorted(set(before_bad) | set(after_bad)):
            node_removed = set(before_bad.get(node, {})) - set(after_bad.get(node, {}))
            for symptom in sorted(node_removed):
                self.record(
                    node_type=NodeType.SCRUB,
                    rule_name=rule_name,
                    action=Action.REMOVED,
                    symptom=symptom,
                    hostname=node,
                    evidence={"sources": _snapshot_sources(before_bad, node, symptom)},
                )

            node_added = set(after_bad.get(node, {})) - set(before_bad.get(node, {}))
            for symptom in sorted(node_added):
                self.record(
                    node_type=NodeType.SCRUB,
                    rule_name=rule_name,
                    action=Action.CREATED,
                    symptom=symptom,
                    hostname=node,
                    evidence={"sources": _snapshot_sources(after_bad, node, symptom)},
                )

    @property
    def events(self) -> list[DecisionEvent]:
        return list(self._events)


def snapshot_bad_nodes(my_bad_nodes: Optional["JobNodeAttributions"]) -> NodeSymptomSnapshot:
    """Capture per-node symptom sets for before/after diffing across a refiner run.

    Args:
        my_bad_nodes: Per-node attribution state, or None when no node state is available.

    Returns:
        Mapping of node hostname to symptom source evidence.
    """
    if my_bad_nodes is None:
        return {}

    snapshot: NodeSymptomSnapshot = {}
    for node, group in my_bad_nodes.items():
        node_snapshot: dict[str, list[SourceEvidence]] = {}
        for symptom, source_set in group.attributions.all_items():
            if not source_set.has_bad():
                continue
            node_snapshot[symptom] = [
                SourceEvidence.from_source(source, hostname=node) for source in source_set.all_sources()
            ]
        if node_snapshot:
            snapshot[node] = node_snapshot
    return snapshot
