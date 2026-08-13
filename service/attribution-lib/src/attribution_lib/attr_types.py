# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import DemotionTypes, EventStream
from attribution_lib.exceptions import (
    MismatchedNodeAttributionGroupUpdateError,
)
from attribution_lib.filter import filter_events
from attribution_lib.job_info import JobInfo
from attribution_lib.observation.inbound import (
    Source as ObservationSource,
)
from attribution_lib.sources import (
    Source,
    SourceSet,
    SymptomSet,
    json_helper,
)

# NCCL log lines embed a PID in hostnames (``host<pid>``), sometimes with
# trailing JSON artifacts (``\n"``).  This regex strips everything from the
# first ``<digits>`` onward.  ``re.DOTALL`` lets ``.`` match newlines.
_NCCL_PID_SUFFIX_RE = re.compile(r"<\d+>.*", re.DOTALL)


def _normalize_node(name: str) -> str:
    """Normalize a node name by stripping NCCL ``<pid>`` suffixes and lowercasing.

    Lowercasing ensures that the same physical node (e.g. ``Node-042.Example.COM``
    from log-line parsing vs ``node-042.example.com`` from labels) maps to a
    single key in ``JobNodeAttributions``.  Canonical casing is restored
    later by ``update_to_canonical`` in the report layer.

    Args:
        name: Raw node hostname, potentially containing a ``<pid>`` suffix
            and trailing JSON artifacts (e.g. ``"host-001<33751>\\n"``).

    Returns:
        The cleaned, lowercased hostname.  If stripping would produce an
        empty string, the original *name* (lowercased) is returned.
    """
    cleaned = _NCCL_PID_SUFFIX_RE.sub("", name).strip()
    return (cleaned if cleaned else name).lower()


@dataclass(eq=True, frozen=True)
class BadPairInfo:
    local_node: str
    remote_node: str
    local_dev_ip: Optional[str] = None
    local_device: Optional[str] = None
    remote_dev_ip: Optional[str] = None
    remote_device: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BadPairInfo):
            return False
        return (self.local_node == other.local_node and self.remote_node == other.remote_node) or (
            self.local_node == other.remote_node and self.remote_node == other.local_node
        )

    def __hash__(self) -> int:
        return hash(frozenset([self.local_dev_ip, self.remote_dev_ip]))

    def update(self, other: "BadPairInfo") -> "BadPairInfo":
        if other.local_node == self.local_node:
            if other.local_device and not self.local_device:
                object.__setattr__(self, "local_device", other.local_device)
            if other.local_dev_ip and not self.local_dev_ip:
                object.__setattr__(self, "local_dev_ip", other.local_dev_ip)
            if other.remote_device and not self.remote_device:
                object.__setattr__(self, "remote_device", other.remote_device)
            if other.remote_dev_ip and not self.remote_dev_ip:
                object.__setattr__(self, "remote_dev_ip", other.remote_dev_ip)
        elif other.remote_node == self.local_node:
            if other.remote_device and not self.local_device:
                object.__setattr__(self, "local_device", other.remote_device)
            if other.remote_dev_ip and not self.local_dev_ip:
                object.__setattr__(self, "local_dev_ip", other.remote_dev_ip)
            if other.local_device and not self.remote_device:
                object.__setattr__(self, "remote_device", other.local_device)
            if other.local_dev_ip and not self.remote_dev_ip:
                object.__setattr__(self, "remote_dev_ip", other.local_dev_ip)
        return self


@dataclass
class NodeAttributionGroup:
    """Structure to track the attributions for a node. Attributions are grouped by their symptoms."""

    hostname: str
    attributions: SymptomSet[str] = field(default_factory=SymptomSet)

    def to_dict(self) -> dict[str, Any]:
        return {"hostname": self.hostname, "attributions": self.attributions}

    def __len__(self) -> int:
        return len(self.attributions)

    def is_void(self) -> bool:
        return self.attributions.is_void()

    def __contains__(self, symptom: str) -> bool:
        return symptom in self.attributions

    # def __iter__(self):
    #     return iter(self.attributions)

    def symptoms(self) -> Iterable[str]:
        return self.attributions.symptoms()

    def add(self, source_set: SourceSet[str]) -> None:
        """Add an attribution to node, unless
        the symptom is already present, in which case we raise DuplicateAttribution"""
        self.attributions.add(source_set)

    def discard(self, symptom: str, purge: bool = False) -> None:
        """Remove all attributions matching symptoms from this container.
        If purge is True, then symptoms are removed and not moved to interesting.
        """
        self.attributions.discard(symptom, purge)

    def has_bad(self) -> bool:
        return self.attributions.has_bad()

    def discard_all(self, purge: bool = False) -> None:
        """Remove all attributions from this container. (mark all as "not bad")
        If purge is True, then symptoms are removed and not moved to interesting.
        """
        self.attributions.discard_all(purge)

    def update(self, other: "NodeAttributionGroup") -> "NodeAttributionGroup":
        """Append the symptoms and their sources in other to this."""
        if self.hostname != other.hostname:
            raise MismatchedNodeAttributionGroupUpdateError(self.hostname, other.hostname)
        self.attributions.update(other.attributions)
        return self


@dataclass
class JobNodeAttributions:
    """Container for a group of sources associated with a job.  This
    replaces the previous approach that used three dicts-of-sets for
    bad, interesting, and epilog nodes.  We use the default
    add/update/discard behavior of SourceSet and SymptomSet to handle
    'bad', and we allow for things to be added as 'interesting'. Also,
    we always move discarded 'bad' entries to 'interesting' rather
    than really discard them.  Epilog is handled through our ability
    to find the source context and the source field thereof
    (ObservationSource.HEALTH_CHECK) Maps nodes to the
    NodeAttributionGroup for them.

    """

    attributions: dict[str, NodeAttributionGroup] = field(default_factory=dict)

    @classmethod
    def from_event_stream(cls, event_stream: EventStream, job_info: JobInfo) -> "JobNodeAttributions":
        filter_events(event_stream, job_info)

        res = cls()

        # Get all type IDs upfront
        type_configs = {
            "Attribution.node.bad": (False, False),
            "Attribution.node.interesting": (True, False),
            "Attribution.node.bad_peers": (False, True),
            "Attribution.node.interesting_peers": (True, True),
        }

        type_ph = ",".join(["?"] * len(type_configs))
        names = list(type_configs.keys())

        rows = event_stream.query_raw(
            f"""SELECT e.name, e.id
                FROM event_types e
                WHERE e.name IN ({type_ph})""",
            names,
        )
        type_id_map = {row[1]: type_configs[row[0]] for row in rows}

        if len(type_id_map) == 0:
            return res

        event_ids = []
        for eid, event_type, payload, source in query_events_with_source(event_stream, list(type_id_map.keys())):
            symptom = payload.get("symptom")
            if symptom is None:
                continue

            # Build source_set
            source_set = SourceSet(symptom)
            interesting, peers = type_id_map[event_type]
            if interesting:
                source_set.add_interesting(source)
            else:
                source_set.add(source)

            # Extract nodes
            if peers:
                peer_nodes = [payload.get("node"), payload.get("peer")]
                nodes: list[str] = [n for n in peer_nodes if n is not None]
            else:
                node = payload.get("node")
                nodes = [node] if node is not None else []

            if nodes:
                res.add(nodes, source_set)

            event_ids.append(eid)

        # Delete all processed events in ONE query
        if event_ids:
            ph = ",".join(["?"] * len(event_ids))
            event_stream.execute(
                f"DELETE FROM events WHERE context_id=? AND id IN ({ph})",
                [event_stream.ctx_id] + event_ids,
            )

        return res

    def to_dict(self) -> dict[str, Any]:
        return {"attributions": json_helper(self.attributions)}

    def add(self, nodes: Iterable[str], attribution: SourceSet[str]) -> None:
        for node in nodes:
            node = _normalize_node(node)
            self.attributions.setdefault(node, NodeAttributionGroup(node)).add(attribution)

    def clear(self) -> None:
        for group in self.attributions.values():
            group.discard_all()

    def discard(self, nodes: Iterable[str], symptoms: list[str], purge: bool = False) -> None:
        for node in nodes:
            if node in self.attributions:
                for symptom in symptoms:
                    self.attributions[node].discard(symptom, purge)
                if self.attributions[node].is_void():
                    del self.attributions[node]

    def __iter__(self) -> Iterator[str]:
        return iter(self.attributions.keys())

    def num_nodes(self) -> int:
        """Return number of nodes in this object."""
        return len(self.attributions)

    def __getitem__(self, key: str) -> NodeAttributionGroup:
        return self.attributions[key]

    def keys(self) -> Iterator[str]:
        return iter(self.attributions.keys())

    def items(self) -> Iterator[tuple[str, NodeAttributionGroup]]:
        return iter(self.attributions.items())

    def discard_node(self, node: str, purge: bool = False) -> None:
        self.attributions[node].discard_all(purge)
        if self.attributions[node].is_void():
            del self.attributions[node]

    def discard_matching_from_all_nodes(self, substrings: list[str], purge: bool = False) -> None:
        """Discard symptoms containing any of the given substrings, from every node."""
        for node_group in list(self.attributions.values()):
            to_discard = [symptom for symptom in node_group.symptoms() if any(sub in symptom for sub in substrings)]
            if to_discard:
                node_group.attributions.discard_seq(to_discard, purge=purge)

    def purge_from_node(self, node: str, symptom: str) -> None:
        if node in self.attributions:
            self.attributions[node].discard(symptom, True)
            if self.attributions[node].is_void():
                del self.attributions[node]

    def update(self, other: "JobNodeAttributions") -> "JobNodeAttributions":
        """Append/add the attributions in other to this."""
        for node, att_grp in other.attributions.items():
            node = _normalize_node(node)
            self.attributions.setdefault(node, NodeAttributionGroup(node)).update(att_grp)
        return self

    def extract_bad_interesting_epilog(
        self,
        bad_nodes: defaultdict[str, set[str]],
        interesting_nodes: defaultdict[str, set[str]],
        epilog_nodes: defaultdict[str, set[str]],
    ) -> None:
        for node, attr_group in self.attributions.items():
            for symptom, attributions in attr_group.attributions._data.items():
                for attribution in attributions._bad:
                    if attribution.source_context.source is ObservationSource.HEALTH_CHECK:
                        epilog_nodes[node].add(symptom)
                    bad_nodes[node].add(symptom)
                    interesting_nodes[node].add(symptom)
                for attribution in attributions._interesting:
                    if attribution.source_context.source is ObservationSource.HEALTH_CHECK:
                        epilog_nodes[node].add(symptom)
                    interesting_nodes[node].add(symptom)

    def extract_evidence_and_xid(
        self,
        node_to_evidence: defaultdict[str, set[str]],
        node_to_xid: defaultdict[str, set[str]],
    ) -> None:
        for node, attr_group in self.attributions.items():
            for symptom, attributions in attr_group.attributions._data.items():
                for attribution in attributions._bad:
                    if attribution.source_context.source is ObservationSource.SYSLOG:
                        node_to_evidence[node].add(attribution.datum)
                        if "Xid" in attribution.datum:
                            node_to_xid[node].add(attribution.datum)
                for attribution in attributions._interesting:
                    if attribution.source_context.source is ObservationSource.SYSLOG:
                        node_to_evidence[node].add(attribution.datum)
                        if "Xid" in attribution.datum:
                            node_to_xid[node].add(attribution.datum)


@dataclass
class EventHelper:
    """Helper class intended to be used in line-walking observations. Makes adding node attributions from srcs less verbose."""

    event_adder: EventStream.Adder
    src_base: Source

    def other_event(
        self, event_type: str, payload: dict[str, Any], rule: str, *, time_override: Optional[datetime] = None
    ) -> None:
        self.event_adder(event_type, payload, self.src_base.derived(rule=rule, time=time_override))

    def bad_peers(
        self,
        node: str,
        peer: str,
        symptom: str,
        rule: str,
        *,
        demotion: dict[DemotionTypes, str] = {},
        time_override: Optional[datetime] = None,
    ) -> None:
        payload = {"node": node, "peer": peer, "symptom": symptom}
        for dt, drule in demotion.items():
            payload[dt] = drule
        self.event_adder(
            "Attribution.node.bad_peers",
            payload,
            self.src_base.derived(rule=rule, time=time_override),
        )

    def bad(
        self,
        node: str,
        symptom: str,
        rule: str,
        *,
        demotion: dict[DemotionTypes, str] = {},
        time_override: Optional[datetime] = None,
    ) -> None:
        payload = {"node": node, "symptom": symptom}
        for dt, drule in demotion.items():
            payload[dt] = drule
        self.event_adder(
            "Attribution.node.bad",
            payload,
            self.src_base.derived(rule=rule, time=time_override),
        )

    def interesting(
        self,
        node: str,
        symptom: str,
        rule: str,
        *,
        demotion: dict[DemotionTypes, str] = {},
        time_override: Optional[datetime] = None,
    ) -> None:
        payload = {"node": node, "symptom": symptom}
        for dt, drule in demotion.items():
            payload[dt] = drule
        self.event_adder(
            "Attribution.node.interesting",
            payload,
            self.src_base.derived(rule=rule, time=time_override),
        )

    def interesting_peers(
        self,
        node: str,
        peer: str,
        symptom: str,
        rule: str,
        *,
        demotion: dict[DemotionTypes, str] = {},
        time_override: Optional[datetime] = None,
    ) -> None:
        payload = {"node": node, "peer": peer, "symptom": symptom}
        for dt, drule in demotion.items():
            payload[dt] = drule
        self.event_adder(
            "Attribution.node.interesting_peers",
            payload,
            self.src_base.derived(rule=rule, time=time_override),
        )
