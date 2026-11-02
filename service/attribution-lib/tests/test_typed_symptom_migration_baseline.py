# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import attribution_lib
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.core import CalcBadNodesAttributionResult, attribute
from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import EventManager
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.resources import AllJobResources
from attribution_lib.sources import Source, SourceSet
from test_support.helpers import make_node_to_ranks


def _source(context: ObservationContext, *, datum: str, line_num: int) -> Source:
    """Build stable source evidence for migration characterization.

    Args:
        context: Observation context registered with the event stream.
        datum: Evidence text to preserve through the pipeline.
        line_num: Evidence line number to preserve through the pipeline.

    Returns:
        Source evidence using the shared test rule.
    """
    return Source(
        source_context=context,
        line_num=line_num,
        datum=datum,
        time=context.time_interval.start,
        resources=AllJobResources,
        rule="test_rule",
    )


def test_legacy_event_consumption_handles_mixed_payloads_and_preserves_sources(
    event_stream,
    dummy_ctx,
    simple_job_factory,
):
    stream = event_stream("typed-symptom-event-baseline")
    stream.register_observation_context(dummy_ctx)
    source = _source(dummy_ctx, datum="ECC evidence", line_num=17)
    stream.add_event(
        "Attribution.node.bad",
        {"node": "node-001", "symptom": "ECC Error"},
        source,
    )
    stream.add_event(
        "Attribution.node.bad",
        {"node": "node-001", "symptom": "kernel panic"},
        _source(dummy_ctx, datum="kernel panic evidence", line_num=18),
    )
    stream.add_event(
        "Attribution.node.bad",
        {"node": "node-001"},
        _source(dummy_ctx, datum="missing symptom evidence", line_num=19),
    )
    job_info = simple_job_factory(
        state="FAILED",
        node_to_ranks=make_node_to_ranks(["node-001"], 1),
    )

    attributions = JobNodeAttributions.from_event_stream(stream, job_info)

    group = attributions["node-001"]
    assert set(group.symptoms()) == {"ECC Error", "kernel panic"}

    ecc_source = group.attributions["ECC Error"].one_source()
    assert ecc_source is not None
    assert ecc_source.datum == "ECC evidence"
    assert ecc_source.line_num == 17
    assert ecc_source.rule == "test_rule"

    panic_source = group.attributions["kernel panic"].one_source()
    assert panic_source is not None
    assert panic_source.datum == "kernel panic evidence"
    assert panic_source.line_num == 18
    assert panic_source.rule == "test_rule"

    event_type = stream.mgr.types.get_id("Attribution.node.bad")
    remaining_events = list(query_events_with_source(stream, event_type))
    assert len(remaining_events) == 1
    _, _, remaining_payload, remaining_source = remaining_events[0]
    assert remaining_payload == {"node": "node-001"}
    assert remaining_source is not None
    assert remaining_source.datum == "missing symptom evidence"
    assert remaining_source.line_num == 19
    assert remaining_source.rule == "test_rule"


def test_snapshot_preserves_nested_symptom_identity_payload(tmp_path, dummy_ctx):
    payload = {
        "node": "node-001",
        "symptom": "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
        "symptom_identity": {
            "symptom_id": "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
            "family": {
                "name": "XID",
                "properties": {
                    "number": 149,
                    "subcode": 4,
                    "port_down_reason": "ALIGNMENT_LOSS",
                },
            },
            "family_schema_fingerprint": "baseline-fingerprint",
        },
    }
    manager = EventManager(rules_engine=attribution_lib.get_rules_engine())
    manager.preload_rule_names({"test_rule"})
    stream = manager.create_stream("typed-symptom-snapshot-baseline")
    stream.register_observation_context(dummy_ctx)
    stream.add_event(
        "Attribution.node.bad",
        payload,
        _source(dummy_ctx, datum="XID evidence", line_num=149),
    )
    stream.flush()
    snapshot_dir = tmp_path / "snapshot"
    manager.export_snapshot(snapshot_dir)
    manager.close()

    restored = EventManager.from_snapshot(snapshot_dir, attribution_lib.get_rules_engine())
    try:
        restored_stream = restored.create_stream("typed-symptom-snapshot-baseline")
        event_type = restored.types.get_id("Attribution.node.bad")
        events = list(query_events_with_source(restored_stream, event_type))

        assert len(events) == 1
        assert events[0][2] == payload
        assert events[0][3].datum == "XID evidence"
    finally:
        restored.close()


def test_attribute_preserves_public_strings_outcome_and_source(
    event_stream,
    dummy_ctx,
    simple_job_factory,
):
    stream = event_stream("typed-symptom-runtime-baseline")
    source = _source(dummy_ctx, datum="double-bit ECC", line_num=48)
    bad_nodes = JobNodeAttributions()
    bad_nodes.add(["node-001"], SourceSet.new_one("ECC Error", source))
    bad_node_result = CalcBadNodesAttributionResult(my_bad_nodes=bad_nodes)
    job_info = simple_job_factory(
        state="FAILED",
        node_to_ranks=make_node_to_ranks(["node-001"], 1),
    )
    recorder = DecisionRecorder(job_info.id)

    result = attribute(
        event_stream=stream,
        job_info=job_info,
        bad_node_result=bad_node_result,
        extract_job_metadata_from_logfile=False,
        recorder=recorder,
    )

    assert list(result.symptoms.symptoms()) == ["ECC Error"]
    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert result.outcome_attr.value().root_cause_symptom == "ECC Error"
    assert list(result.bad_nodes_attr.my_bad_nodes["node-001"].symptoms()) == ["ECC Error"]
    stored_source = result.bad_nodes_attr.my_bad_nodes["node-001"].attributions["ECC Error"].one_source()
    assert stored_source is not None
    assert stored_source.datum == "double-bit ECC"
    assert recorder.events
    assert all(event.symptom is None or isinstance(event.symptom, str) for event in recorder.events)
