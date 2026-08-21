# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime
from typing import Optional
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest

from attribution_lib.attr_types import JobNodeAttributions, NodeAttributionGroup
from attribution_lib.core import AttributionResult, CalcBadNodesAttributionResult
from attribution_lib.decision_trace import (
    DecisionDAG,
    DecisionEdge,
    DecisionNode,
    SymptomDetail,
    build_decision_graph,
    build_symptom_details,
)
from attribution_lib.decision_trace_recorder import Action, DecisionRecorder, NodeType, snapshot_bad_nodes
from attribution_lib.exceptions import DecisionGraphError
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
    Id,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.sources import Source, SourceSet, SymptomSet
from attribution_lib.symptoms.base import BaseSymptom

TZ = ZoneInfo("UTC")
T1 = datetime(2025, 6, 15, 10, 0, 0, tzinfo=TZ)
T2 = datetime(2025, 6, 15, 10, 0, 5, tzinfo=TZ)
TIME_INTERVAL = TimeInterval(T1, T2)


def _make_obs_context(source: ObservationSource, fmt: ObservationFormat) -> ObservationContext:
    return ObservationContext(
        time_interval=TIME_INTERVAL,
        resources=AllJobResources,
        source=source,
        format=fmt,
        timezone=TZ,
        id=Id(99999),
    )


def _make_source(
    obs_source: ObservationSource = ObservationSource.SYSLOG,
    rule: str = "test_rule",
    time: datetime = T1,
    line_num: int = 100,
    datum: str = "test evidence",
) -> Source:
    ctx = _make_obs_context(obs_source, ObservationFormat.LOKI_DICT_LIST)
    return Source(
        source_context=ctx,
        line_num=line_num,
        datum=datum,
        time=time,
        resources=AllJobResources,
        rule=rule,
    )


def _make_attribution_result(
    node_symptoms: dict[str, dict[str, list[Source]]],
    interesting_symptoms: Optional[dict[str, dict[str, list[Source]]]] = None,
    job_symptoms: Optional[dict[str, list[Source]]] = None,
    interesting_job_symptoms: Optional[dict[str, list[Source]]] = None,
    node_symptoms_base: Optional[dict] = None,
    skipped_symptom_ids: Optional[set] = None,
) -> AttributionResult:
    """Build an AttributionResult from a simple spec.

    Args:
        node_symptoms: {node: {symptom: [Source, ...]}} for bad symptoms
        interesting_symptoms: {node: {symptom: [Source, ...]}} for interesting-only symptoms
        job_symptoms: {symptom: [Source, ...]} for job-level bad symptoms
        interesting_job_symptoms: {symptom: [Source, ...]} for job-level interesting symptoms
        node_symptoms_base: pre-built dict for cascade info
        skipped_symptom_ids: set of pruned symptom ids
    """
    bad_nodes = JobNodeAttributions()
    for node, symptoms in node_symptoms.items():
        for symptom, sources in symptoms.items():
            ss = SourceSet(symptom)
            for src in sources:
                ss.add(src)
            bad_nodes.add([node], ss)

    if interesting_symptoms:
        for node, symptoms in interesting_symptoms.items():
            for symptom, sources in symptoms.items():
                ss = SourceSet(symptom)
                for src in sources:
                    ss.add_interesting(src)
                bad_nodes.attributions.setdefault(node, NodeAttributionGroup(node)).attributions._data.setdefault(
                    symptom, SourceSet(symptom)
                ).update(ss)

    bad_node_result = CalcBadNodesAttributionResult(
        my_bad_nodes=bad_nodes,
        node_symptoms_base=node_symptoms_base or {},
        skipped_symptom_ids=skipped_symptom_ids or set(),
    )

    event_stream = Mock()
    result = AttributionResult(
        bad_nodes_attr=bad_node_result,
        event_stream=event_stream,
    )
    if job_symptoms:
        for symptom, sources in job_symptoms.items():
            ss = SourceSet(symptom)
            for src in sources:
                ss.add(src)
            result.symptoms.add(ss)
    if interesting_job_symptoms:
        for symptom, sources in interesting_job_symptoms.items():
            ss = SourceSet(symptom)
            for src in sources:
                ss.add_interesting(src)
            result.symptoms.add(ss)
    return result


class TestBuildSymptomDetails:
    def test_basic_bad_symptom(self):
        """Single bad symptom on one node produces correct structure."""
        src = _make_source(
            obs_source=ObservationSource.SYSLOG,
            rule="xid_48_ecc",
            time=T1,
            line_num=4821,
            datum="Xid (PCI:0000:b1:00): 48, pid=12345",
        )
        att_res = _make_attribution_result({"node-042": {"XID-48": [src]}})

        result = build_symptom_details(att_res)

        assert "node-042" in result
        assert "XID-48" in result["node-042"]
        detail = result["node-042"]["XID-48"]
        assert isinstance(detail, SymptomDetail)
        assert detail.status == "bad"
        assert len(detail.sources) == 1
        assert detail.sources[0].observation_source == "syslog"
        assert detail.sources[0].rule == "xid_48_ecc"
        assert detail.sources[0].line_num == 4821
        assert detail.sources[0].datum == "Xid (PCI:0000:b1:00): 48, pid=12345"
        assert detail.cascade_discards == []

    def test_interesting_only_symptom(self):
        """Symptom with only interesting sources gets status 'interesting'."""
        src = _make_source(rule="ib_error_rule", datum="IB Error-12")
        att_res = _make_attribution_result(
            node_symptoms={},
            interesting_symptoms={"node-001": {"IB Error-12": [src]}},
        )

        result = build_symptom_details(att_res)

        assert "node-001" in result
        detail = result["node-001"]["IB Error-12"]
        assert detail.status == "interesting"

    def test_multiple_sources_same_symptom(self):
        """Multiple sources for the same symptom are all included."""
        src1 = _make_source(time=T1, line_num=100, datum="first occurrence")
        src2 = _make_source(time=T2, line_num=200, datum="second occurrence")
        att_res = _make_attribution_result({"node-001": {"XID-48": [src1, src2]}})

        result = build_symptom_details(att_res)

        detail = result["node-001"]["XID-48"]
        assert len(detail.sources) == 2
        assert detail.sources[0].line_num == 100
        assert detail.sources[1].line_num == 200

    def test_cascade_discards_from_node_symptoms_base(self):
        """cascade_discards populated from node_symptoms_base BaseSymptom."""
        src = _make_source(datum="XID-48 evidence")
        base_symptom = BaseSymptom(
            symptom_id="XID-48",
            cascade_discards=["XID-43", "XID-79"],
        )
        per_node_set = SymptomSet()
        bs_source_set = SourceSet(base_symptom)
        bs_source_set.add(src)
        per_node_set.add(bs_source_set)

        att_res = _make_attribution_result(
            {"node-042": {"XID-48": [src]}},
            node_symptoms_base={"node-042": per_node_set},
        )

        result = build_symptom_details(att_res)

        detail = result["node-042"]["XID-48"]
        assert detail.cascade_discards == ["XID-43", "XID-79"]

    def test_empty_when_no_bad_nodes(self):
        """Returns empty dict when there are no bad nodes."""
        att_res = _make_attribution_result({})

        result = build_symptom_details(att_res)

        assert result == {}

    def test_canonical_node_mapping(self):
        """Node names are mapped through canonical_from_normalized."""
        src = _make_source()
        att_res = _make_attribution_result({"normalized-node": {"XID-48": [src]}})

        result = build_symptom_details(att_res, canonical_from_normalized={"normalized-node": "real-node"})

        assert "real-node" in result
        assert "normalized-node" not in result

    def test_multiple_nodes(self):
        """Multiple nodes each get their own entry."""
        src1 = _make_source(datum="node1 evidence")
        src2 = _make_source(obs_source=ObservationSource.HEALTH_CHECK, datum="node2 evidence")
        att_res = _make_attribution_result(
            {
                "node-001": {"XID-48": [src1]},
                "node-002": {"LustreError": [src2]},
            }
        )

        result = build_symptom_details(att_res)

        assert len(result) == 2
        assert "node-001" in result
        assert "node-002" in result
        assert "XID-48" in result["node-001"]
        assert "LustreError" in result["node-002"]
        assert result["node-002"]["LustreError"].sources[0].observation_source == "health_check"

    def test_different_observation_sources(self):
        """Observation source enum values are correctly serialized."""
        sources_map = {
            ObservationSource.SYSLOG: "syslog",
            ObservationSource.APPLOG: "applog",
            ObservationSource.HEALTH_CHECK: "health_check",
            ObservationSource.NODE_METRICS: "node_metrics",
            ObservationSource.UFMLOG: "ufmlog",
        }
        for obs_source, expected_str in sources_map.items():
            src = _make_source(obs_source=obs_source)
            att_res = _make_attribution_result({"node-001": {"symptom": [src]}})
            result = build_symptom_details(att_res)
            assert result["node-001"]["symptom"].sources[0].observation_source == expected_str

    def test_source_with_none_timestamp(self):
        """Source with None timestamp serializes correctly."""
        ctx = _make_obs_context(ObservationSource.SYSLOG, ObservationFormat.LOKI_DICT_LIST)
        src = Source(ctx, None, "evidence", None, AllJobResources, "rule")
        att_res = _make_attribution_result({"node-001": {"XID-48": [src]}})

        result = build_symptom_details(att_res)

        detail = result["node-001"]["XID-48"]
        assert detail.sources[0].timestamp is None
        assert detail.sources[0].line_num is None


class TestDecisionRecorder:
    def test_record_creates_sequenced_events(self):
        recorder = DecisionRecorder("job-123")
        recorder.record(NodeType.EXTRACTION, "xid_rule", Action.CREATED, "XID-48", hostname="node-042")
        recorder.record(NodeType.OUTCOME, "symptom_to_outcome", Action.MATCHED, "XID-48")

        events = recorder.events
        assert len(events) == 2
        assert events[0].sequence_num == 1
        assert events[1].sequence_num == 2
        assert events[0].node_type == NodeType.EXTRACTION
        assert events[1].node_type == NodeType.OUTCOME

    def test_empty_recorder(self):
        recorder = DecisionRecorder("job-123")
        assert recorder.events == []

    def test_record_failure_does_not_advance_sequence(self):
        recorder = DecisionRecorder("job-123")

        with (
            patch("attribution_lib.decision_trace_recorder.DecisionEvent", side_effect=ValueError("bad event")),
            pytest.raises(ValueError, match="bad event"),
        ):
            recorder.record(NodeType.EXTRACTION, "xid_rule", Action.CREATED, "XID-48")

        assert recorder.events == []

        recorder.record(NodeType.EXTRACTION, "xid_rule", Action.CREATED, "XID-48")
        assert [e.sequence_num for e in recorder.events] == [1]

    def test_snapshot_bad_nodes_preserves_source_evidence(self):
        src = _make_source(
            obs_source=ObservationSource.SYSLOG,
            rule="xid_48_ecc",
            datum="Xid 48",
            line_num=42,
        )
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node-042"], SourceSet.new_one("XID-48", src))

        snapshot = snapshot_bad_nodes(bad_nodes)

        evidence = snapshot["node-042"]["XID-48"][0]
        assert evidence.obs_source == "syslog"
        assert evidence.rule_name == "xid_48_ecc"
        assert evidence.datum == "Xid 48"
        assert evidence.line_num == 42
        assert evidence.hostname == "node-042"

    def test_record_refiner_diff_attaches_removed_node_evidence(self):
        src = _make_source(
            obs_source=ObservationSource.APPLOG,
            rule="PythonExceptionFromFile",
            datum="Traceback",
            line_num=700,
        )
        before_nodes = JobNodeAttributions()
        before_nodes.add(["node-001"], SourceSet.new_one("Python Exception", src))

        recorder = DecisionRecorder("job-123")
        recorder.record_refiner_diff(
            "PythonExceptionFromFile",
            before_symptoms=set(),
            after_symptoms=set(),
            before_bad=snapshot_bad_nodes(before_nodes),
            after_bad={},
        )

        event = next(e for e in recorder.events if e.hostname == "node-001")
        assert event.symptom == "Python Exception"
        assert event.evidence is not None
        assert event.evidence["sources"][0]["hostname"] == "node-001"
        assert event.evidence["sources"][0]["obs_source"] == "applog"


class TestBuildDecisionDAG:
    @pytest.mark.parametrize(
        ("evidence", "message"),
        [
            ({"sources": None}, "sources for 'XID-48' must be a list"),
            ({"sources": ["not-a-mapping"]}, "source evidence must be a mapping"),
            ({"sources": [{"datum": {"nested": "value"}}]}, "field 'datum' must be str or None"),
        ],
    )
    def test_build_rejects_malformed_event_evidence(self, evidence, message):
        recorder = DecisionRecorder("job-123")
        recorder.record(NodeType.REFINER, "test_rule", Action.CREATED, "XID-48", evidence=evidence)
        att_res = _make_attribution_result({})
        att_res.outcome_attr = SourceSet(Mock(outcome="Incomplete Info", root_cause_symptom=""))

        with pytest.raises(DecisionGraphError, match=message):
            build_decision_graph(recorder, att_res)

    def test_basic_graph_structure(self):
        """Single symptom produces extraction + outcome nodes with edges."""
        src = _make_source(
            obs_source=ObservationSource.SYSLOG,
            rule="xid_48_ecc",
            time=T1,
            line_num=42,
            datum="Xid 48",
        )
        att_res = _make_attribution_result({"node-042": {"XID-48": [src]}})
        att_res.outcome_attr = SourceSet(Mock(outcome="Fail InfraHW", root_cause_symptom="XID-48"))

        recorder = DecisionRecorder("job-123")
        graph = build_decision_graph(recorder, att_res)

        assert isinstance(graph, DecisionDAG)

        ext_nodes = [n for n in graph.nodes if n.node_type == NodeType.EXTRACTION]
        assert len(ext_nodes) >= 1
        assert isinstance(ext_nodes[0], DecisionNode)
        assert ext_nodes[0].symptom == "XID-48"
        assert ext_nodes[0].hostname == "node-042"

        outcome_nodes = [n for n in graph.nodes if n.node_type == NodeType.OUTCOME]
        assert len(outcome_nodes) == 1
        assert outcome_nodes[0].reason is not None
        assert "XID-48" in outcome_nodes[0].reason

        matches_edges = [e for e in graph.edges if e.edge_type == "MATCHES"]
        assert len(matches_edges) >= 1
        assert isinstance(matches_edges[0], DecisionEdge)

    def test_graph_with_cascade_prune_events(self):
        """Recorder cascade events produce CASCADE_PRUNE nodes with PRUNES edges."""
        src1 = _make_source(rule="xid_171", datum="Xid 171", line_num=10)
        src2 = _make_source(rule="xid_48", datum="Xid 48", line_num=20)
        att_res = _make_attribution_result(
            {
                "node-042": {"XID-171": [src1], "XID-48": [src2]},
            }
        )
        att_res.outcome_attr = SourceSet(Mock(outcome="Fail InfraHW", root_cause_symptom="XID-171"))

        recorder = DecisionRecorder("job-123")
        recorder.record(
            NodeType.CASCADE_PRUNE,
            "DagCascadePruning",
            Action.PRUNED,
            "XID-48",
            hostname="node-042",
            reason="cascade child of XID-171",
        )

        graph = build_decision_graph(recorder, att_res)

        cascade_nodes = [n for n in graph.nodes if n.node_type == NodeType.CASCADE_PRUNE]
        assert len(cascade_nodes) == 1
        assert cascade_nodes[0].symptom == "XID-48"
        assert cascade_nodes[0].action == Action.PRUNED

        prune_edges = [e for e in graph.edges if e.edge_type == "PRUNES"]
        assert len(prune_edges) >= 1

    def test_graph_with_refiner_events(self):
        """Refiner removal events produce REFINER nodes."""
        src = _make_source(rule="ib_rule", datum="IB Error-12", line_num=300)
        att_res = _make_attribution_result({"node-001": {"IB Error-12": [src]}})
        att_res.outcome_attr = SourceSet(Mock(outcome="Incomplete Info", root_cause_symptom=""))

        recorder = DecisionRecorder("job-123")
        recorder.record(
            NodeType.REFINER,
            "IBSymptomCleanup",
            Action.REMOVED,
            "IB Error-12",
            reason="IB errors cascaded from link down",
        )

        graph = build_decision_graph(recorder, att_res)

        refiner_nodes = [n for n in graph.nodes if n.node_type == NodeType.REFINER]
        assert len(refiner_nodes) == 1
        assert refiner_nodes[0].rule_name == "IBSymptomCleanup"
        assert refiner_nodes[0].action == Action.REMOVED

    def test_graph_includes_job_level_symptom_extractions(self):
        """Job-level symptoms without node attribution still produce extraction nodes."""
        src = _make_source(
            obs_source=ObservationSource.APPLOG,
            rule="PythonExceptionFromFile",
            datum="Traceback (most recent call last)",
            line_num=700,
        )
        att_res = _make_attribution_result(
            node_symptoms={},
            job_symptoms={"Python Exception": [src]},
        )
        att_res.outcome_attr = SourceSet(Mock(outcome="Fail User", root_cause_symptom="Python Exception"))

        recorder = DecisionRecorder("job-123")
        graph = build_decision_graph(recorder, att_res)

        ext_nodes = [n for n in graph.nodes if n.node_type == NodeType.EXTRACTION and n.symptom == "Python Exception"]
        assert len(ext_nodes) == 1
        assert ext_nodes[0].hostname is None
        assert ext_nodes[0].obs_source == "applog"
        assert ext_nodes[0].rule_name == "PythonExceptionFromFile"

        outcome_nodes = [n for n in graph.nodes if n.node_type == NodeType.OUTCOME]
        matches_edges = [e for e in graph.edges if e.edge_type == "MATCHES"]
        assert any(
            e.parent_id == ext_nodes[0].node_id and e.child_id == outcome_nodes[0].node_id for e in matches_edges
        )

    def test_graph_preserves_same_source_on_multiple_hosts(self):
        """Same source evidence on multiple hosts should not collapse to one extractor."""
        src = _make_source(rule="node_down", datum="node down")
        att_res = _make_attribution_result(
            {
                "node-001": {"Node not responding": [src]},
                "node-002": {"Node not responding": [src]},
            }
        )
        att_res.outcome_attr = SourceSet(Mock(outcome="Fail InfraHW", root_cause_symptom="Node not responding"))

        recorder = DecisionRecorder("job-123")
        graph = build_decision_graph(recorder, att_res)

        ext_nodes = [
            n for n in graph.nodes if n.node_type == NodeType.EXTRACTION and n.symptom == "Node not responding"
        ]
        assert sorted(n.hostname for n in ext_nodes) == ["node-001", "node-002"]

    def test_graph_does_not_fallback_to_unrelated_host_extraction(self):
        """Host-specific events synthesize evidence instead of linking to another host."""
        src = _make_source(rule="node_down", datum="node down")
        att_res = _make_attribution_result({"node-001": {"Node not responding": [src]}})
        att_res.outcome_attr = SourceSet(Mock(outcome="Incomplete Info", root_cause_symptom=""))

        recorder = DecisionRecorder("job-123")
        recorder.record(
            NodeType.REFINER,
            "NodeSpecificRefiner",
            Action.REMOVED,
            "Node not responding",
            hostname="node-002",
        )

        graph = build_decision_graph(recorder, att_res)

        ext_nodes = {n.hostname: n for n in graph.nodes if n.node_type == NodeType.EXTRACTION}
        refiner_node = next(n for n in graph.nodes if n.node_type == NodeType.REFINER)
        parent_ids = {e.parent_id for e in graph.edges if e.child_id == refiner_node.node_id}

        assert "node-001" in ext_nodes
        assert "node-002" in ext_nodes
        assert ext_nodes["node-001"].node_id not in parent_ids
        assert ext_nodes["node-002"].node_id in parent_ids

    def test_graph_uses_recorder_snapshot_for_purged_symptom(self):
        """Symptoms absent from final snapshots can still be wired from recorder evidence."""
        src = _make_source(
            obs_source=ObservationSource.APPLOG,
            rule="PythonExceptionFromFile",
            datum="Traceback",
            line_num=700,
        )
        before_nodes = JobNodeAttributions()
        before_nodes.add(["node-001"], SourceSet.new_one("Python Exception", src))
        att_res = _make_attribution_result(node_symptoms={})
        att_res.outcome_attr = SourceSet(Mock(outcome="Incomplete Info", root_cause_symptom=""))

        recorder = DecisionRecorder("job-123")
        recorder.record_refiner_diff(
            "PythonExceptionFromFile",
            before_symptoms=set(),
            after_symptoms=set(),
            before_bad=snapshot_bad_nodes(before_nodes),
            after_bad={},
        )

        graph = build_decision_graph(recorder, att_res)

        ext_nodes = [n for n in graph.nodes if n.node_type == NodeType.EXTRACTION and n.symptom == "Python Exception"]
        scrub_nodes = [n for n in graph.nodes if n.node_type == NodeType.SCRUB and n.hostname == "node-001"]
        linked_targets = {
            e.child_id
            for e in graph.edges
            if e.parent_id in {n.node_id for n in ext_nodes} and e.edge_type == "DEMOTES"
        }

        assert len(ext_nodes) == 1
        assert ext_nodes[0].hostname == "node-001"
        assert ext_nodes[0].obs_source == "applog"
        assert len(scrub_nodes) == 1
        assert scrub_nodes[0].node_id in linked_targets

    def test_graph_synthesizes_extraction_for_refiner_only_symptom(self):
        """Refiner events referencing removed-only symptoms are not orphaned."""
        att_res = _make_attribution_result(node_symptoms={})
        att_res.outcome_attr = SourceSet(Mock(outcome="Incomplete Info", root_cause_symptom=""))

        recorder = DecisionRecorder("job-123")
        recorder.record(
            NodeType.REFINER,
            "PythonExceptionFromFile",
            Action.REMOVED,
            "Python Exception",
            reason="Python Exception was superseded by file-specific symptom",
        )

        graph = build_decision_graph(recorder, att_res)

        ext_nodes = [n for n in graph.nodes if n.node_type == NodeType.EXTRACTION and n.symptom == "Python Exception"]
        refiner_nodes = [
            n for n in graph.nodes if n.node_type == NodeType.REFINER and n.rule_name == "PythonExceptionFromFile"
        ]
        demote_edges = [e for e in graph.edges if e.edge_type == "DEMOTES"]

        assert len(ext_nodes) == 1
        assert len(refiner_nodes) == 1
        assert any(e.parent_id == ext_nodes[0].node_id and e.child_id == refiner_nodes[0].node_id for e in demote_edges)

    def test_graph_uses_final_reclassified_outcome(self):
        """Intermediate outcome recorder events must not override final outcome_attr."""
        src = _make_source(
            obs_source=ObservationSource.APPLOG,
            rule="EmptySymptomsFallback",
            datum="PermissionError",
            line_num=12,
        )
        pass_source = src.derived(rule="reclassified: slurm preempted", datum="PREEMPTED")
        att_res = _make_attribution_result(
            node_symptoms={},
            job_symptoms={"log permission error": [src]},
        )
        att_res.outcome_attr = SourceSet(Mock(outcome="Pass", root_cause_symptom="SLURM preempted")).add(pass_source)

        recorder = DecisionRecorder("job-123")
        recorder.record(
            NodeType.OUTCOME,
            "symptom_to_outcome",
            Action.MATCHED,
            "log permission error",
            reason="log permission error matched tier 15 → Incomplete Info",
        )

        graph = build_decision_graph(recorder, att_res)

        outcome_nodes = [n for n in graph.nodes if n.node_type == NodeType.OUTCOME]
        assert len(outcome_nodes) == 1
        assert outcome_nodes[0].symptom == "SLURM preempted"
        assert outcome_nodes[0].reason == "SLURM preempted → Pass"
        assert "Incomplete Info" not in outcome_nodes[0].reason

        pass_ext_nodes = [
            n for n in graph.nodes if n.node_type == NodeType.EXTRACTION and n.symptom == "SLURM preempted"
        ]
        assert len(pass_ext_nodes) == 1
        assert pass_ext_nodes[0].rule_name == "reclassified: slurm preempted"

    def test_graph_dataclasses_serializable(self):
        """DecisionDAG dataclasses can be round-tripped through asdict + JSON."""
        import json
        from dataclasses import asdict

        src = _make_source(rule="xid_48_ecc", datum="Xid 48")
        att_res = _make_attribution_result({"node-042": {"XID-48": [src]}})
        att_res.outcome_attr = SourceSet(Mock(outcome="Fail InfraHW", root_cause_symptom="XID-48"))

        recorder = DecisionRecorder("job-123")
        recorder.record(NodeType.OUTCOME, "symptom_to_outcome", Action.MATCHED, "XID-48")

        graph = build_decision_graph(recorder, att_res)
        graph_dict = asdict(graph)
        json_str = json.dumps(graph_dict, default=str)
        parsed = json.loads(json_str)
        assert len(parsed["nodes"]) > 0
        assert len(parsed["edges"]) > 0
