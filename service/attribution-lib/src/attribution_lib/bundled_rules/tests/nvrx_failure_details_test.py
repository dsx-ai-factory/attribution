# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize Megatron batch-size assertions using normal outcome priorities."""

import pytest

from attribution_lib import core
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.decision_trace import build_decision_graph
from attribution_lib.decision_trace_recorder import DecisionRecorder, NodeType
from attribution_lib.job_info import NVRX_TAG
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.resources import AllJobResources
from attribution_lib.sources import SourceSet
from test_support.helpers import line_entries, make_applog_entry


_ERROR = (
    "AssertionError: eval_global_batch_size (12) must be divisible by "
    "eval_micro_batch_size (4) * data_parallel_size (8)"
)
_SYMPTOM = "Python Exception: AssertionError: batch size not divisible"
_UNAVAILABLE_REASON = "Underlying failure reason unavailable from the collected evidence."
_NVRX = "0: Cycle: 1 Event: failure_detected Node: node01 Rank: 0 Time: 2024-01-01 10:01:00.000 UTC"
_NO_PROGRESS = "0: progress_tracker.py:248 EARLY TERMINATION: 3 consecutive training cycles without progress"
_STEP_TIMEOUT = "0: rank_monitor_server.py:523 [Cycle 1] Section 'step' has been open for 121 seconds. Timeout is 120"


@pytest.fixture
def attribute_cycle(attribution_test_builder):
    def _attribute(entries, *, additional_symptom=None, **job_options):
        options = {"nodes": ["node01", "node02"], "tags": frozenset({NVRX_TAG}), "exit_code_signal": None}
        options.update(job_options)
        job = default_job_info(**options)
        extracted = attribution_test_builder(
            job_info=job,
            data=entries,
            source=ObservationSource.APPLOG,
            insert_tflops_line=False,
        )
        bad_nodes = core.calculate_bad_nodes(
            event_stream=extracted.event_stream,
            job_info=job,
            applog_in=extracted.observation_source,
        )
        if additional_symptom:
            bad_nodes.my_bad_nodes.add(
                ["node01"],
                SourceSet.new_one(additional_symptom, job.source("additional evidence", "test")),
            )
        recorder = DecisionRecorder(job.id)
        result = core.attribute(
            event_stream=extracted.event_stream,
            job_info=job,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
            recorder=recorder,
        )
        return result, build_decision_graph(recorder, result)

    return _attribute


@pytest.mark.parametrize(
    "job_options, expected_outcome",
    [
        ({}, "Fail User"),
        ({"state": "TIMEOUT", "cancelled_by": "Timeout"}, "Fail User"),
        ({"state": "CANCELLED", "cancelled_by": "test-user"}, "Cancelled by User"),
        # Older projected records can carry cancellation even on a FAILED
        # cycle. Preserve that existing policy and retain the assertion details.
        ({"state": "FAILED", "cancelled_by": "test-user"}, "Cancelled by User"),
    ],
)
def test_assertion_survives_nvrx_and_teardown(attribute_cycle, job_options, expected_outcome):
    entries = line_entries(
        [
            "1: Traceback (most recent call last):",
            '1:   File "/app/train.py", line 20, in validate_args',
            f"1: {_ERROR}",
            _NVRX,
            "1: torch.distributed.DistBackendError: process group aborted during shutdown",
        ]
    )
    result, graph = attribute_cycle(entries, **job_options)

    assert result.outcome_attr.value().outcome == expected_outcome
    assert _SYMPTOM in result.symptoms
    assert "NVRx detected failure" in result.symptoms
    assert "Python Exception" not in result.symptoms
    source = result.symptoms[_SYMPTOM].one_source()
    assert source.rule == "BatchSizeAssertion"
    assert source.datum == f"1: {_ERROR}"
    assert source.line_num == 3
    assert source.time is None
    assert {resource.hostname for resource in source.resources} == {"node02"}
    evidence = [node for node in graph.nodes if node.rule_name == "BatchSizeAssertion" and node.hostname == "node02"]
    assert len(evidence) == 1
    assert evidence[0].node_type == NodeType.EXTRACTION
    assert evidence[0].symptom == _SYMPTOM
    assert evidence[0].datum == source.datum
    assert evidence[0].hostname == "node02"
    assert evidence[0].line_num == 3
    assert evidence[0].timestamp is None


@pytest.mark.parametrize("prefix", ["1: ", "[rank1]: ", "[rank 1]: ", "[rank: 1]: ", "[rank : 1]: "])
@pytest.mark.parametrize("batch_kind", ["training", "evaluation"])
def test_assertion_accepts_supported_rank_prefixes(attribute_cycle, prefix, batch_kind):
    """Retain the specific failure and source node for supported rank prefixes."""
    error = _ERROR if batch_kind == "evaluation" else _ERROR.replace("eval_", "")
    line = prefix + error
    result, graph = attribute_cycle(line_entries([line, _NVRX]))

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == _SYMPTOM
    assert outcome.doc == "python-exception"
    source = result.symptoms[_SYMPTOM].one_source()
    assert source.datum == line
    assert {resource.hostname for resource in source.resources} == {"node02"}
    evidence = [node for node in graph.nodes if node.rule_name == "BatchSizeAssertion" and node.hostname == "node02"]
    assert len(evidence) == 1
    assert evidence[0].datum == line
    assert evidence[0].hostname == "node02"


@pytest.mark.parametrize("has_timestamp", [False, True])
def test_assertion_preserves_available_source_without_guessing(attribute_cycle, has_timestamp):
    # AllJobResources means the collector did not identify a node; an unmapped
    # rank in the shared log must remain evidence, not become the first job node.
    timestamp_ns = 1704103260000000000
    entry = {"body": f"99: {_ERROR}", "resources": "AllJobResources"}
    if has_timestamp:
        entry["ts"] = timestamp_ns
    result, graph = attribute_cycle([entry])

    assert _SYMPTOM in result.symptoms
    assert result.outcome_attr.value().outcome == "Fail User"
    source = result.symptoms[_SYMPTOM].one_source()
    assert source.resources == AllJobResources
    assert source.datum == f"99: {_ERROR}"
    assert source.line_num == 1
    assert (source.time.timestamp() if source.time else None) == (timestamp_ns / 1e9 if has_timestamp else None)
    assert not result.bad_nodes_attr.my_bad_nodes.attributions
    evidence = [node for node in graph.nodes if node.rule_name == "BatchSizeAssertion"]
    assert len(evidence) == 1
    assert evidence[0].hostname is None
    assert evidence[0].timestamp == (source.time.isoformat() if source.time else None)


def test_infrastructure_failure_keeps_priority(attribute_cycle):
    result, graph = attribute_cycle(
        [
            make_applog_entry(f"1: {_ERROR}", 1704103260000000000, "node02"),
            {"body": _NVRX},
        ],
        additional_symptom="XID-79-FALLEN_OFF_BUS",
    )

    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert any(node.datum == f"1: {_ERROR}" for node in graph.nodes)


@pytest.mark.parametrize(
    "line",
    [
        f"1: handled {_ERROR}",
        "1: AssertionError: process group is not initialized",
        "1: torch.distributed.DistBackendError: process group aborted during shutdown",
    ],
)
def test_generic_teardown_does_not_become_configuration_failure(attribute_cycle, line):
    result, graph = attribute_cycle(line_entries([line, _NVRX]))

    assert _SYMPTOM not in result.symptoms
    assert not any(node.rule_name == "BatchSizeAssertion" for node in graph.nodes)
    assert result.outcome_attr.value().outcome == "Fail Undetermined"


@pytest.mark.parametrize("nvrx_line", [None, _NVRX, _NO_PROGRESS, _STEP_TIMEOUT])
def test_assertion_priority_and_decision_graph(attribute_cycle, nvrx_line):
    line = f"1: {_ERROR}"
    lines = [line]
    if nvrx_line is not None:
        lines.append(nvrx_line)
    result, graph = attribute_cycle(
        line_entries(lines), tags=frozenset() if nvrx_line is None else frozenset({NVRX_TAG})
    )

    assert result.outcome_attr.value().outcome == "Fail User"
    assert result.outcome_attr.value().root_cause_symptom == _SYMPTOM
    assert result.outcome_attr.value().hw_cause != _UNAVAILABLE_REASON
    assert result.outcome_attr.one_source().datum == line
    if nvrx_line is not None:
        assert any(s.startswith("NVRx ") for s in result.symptoms.symptoms())
    winner = next(n for n in graph.nodes if n.node_type == NodeType.OUTCOME)
    assert winner.rule_name == "symptom_to_outcome"
    assert winner.symptom == _SYMPTOM
    evidence_ids = {n.node_id for n in graph.nodes if n.node_type == NodeType.EXTRACTION and n.datum == line}
    assert any(e.parent_id in evidence_ids and e.child_id == winner.node_id for e in graph.edges)


@pytest.mark.parametrize(
    "line",
    [
        "1: FileNotFoundError: [Errno 2] No such file or directory: '/data/checkpoint.pt'",
        "1: wandb.Error: API request failed",
        "1: Exception: Failed to write dataset materials to the data cache directory",
        "1: runtime.sh: line 42: 12345 Killed unsquashfs -f -d /work/cache image.sqsh",
    ],
)
def test_other_application_errors_keep_existing_nvrx_priority(attribute_cycle, line):
    """This rule changes only the batch-size assertion's detection and priority."""
    result, _ = attribute_cycle(line_entries([line, _NVRX]))

    assert _SYMPTOM not in result.symptoms
    assert result.outcome_attr.value().outcome == "Fail Undetermined"
    assert result.outcome_attr.value().root_cause_symptom == "NVRx detected failure"


@pytest.mark.parametrize("watchdog", ["NCCL watchdog thread terminated", "Exception raised from ncclCommWatchdog"])
@pytest.mark.parametrize(
    "nvrx_line, fallback_symptom, fallback_outcome",
    [
        (_NVRX, "NVRx detected failure", "Fail Undetermined"),
        (_NO_PROGRESS, "NVRx no progress across restarts", "Fail MultiCategory"),
        (_STEP_TIMEOUT, "NVRx step section timeout", "Fail MultiCategory"),
    ],
)
@pytest.mark.parametrize("has_assertion", [False, True])
def test_watchdog_teardown_does_not_explain_nvrx_failure(
    attribute_cycle, watchdog, nvrx_line, fallback_symptom, fallback_outcome, has_assertion
):
    lines = [nvrx_line, f"1: {watchdog}"]
    if has_assertion:
        lines.insert(0, f"1: {_ERROR}")
    result, graph = attribute_cycle(line_entries(lines))

    assert result.outcome_attr.value().outcome == ("Fail User" if has_assertion else fallback_outcome)
    assert result.outcome_attr.value().root_cause_symptom == (_SYMPTOM if has_assertion else fallback_symptom)
    assert fallback_symptom in result.symptoms


def test_batch_sizes_share_a_label_and_preserve_each_source(attribute_cycle):
    # Mirror rank padding and the assertion-line shape seen in shared logs, with
    # synthetic ranks, configuration values, and paths only.
    lines = [f"    1: {_ERROR}\n", f"    1: {_ERROR.replace('(12)', '(18)')}\n", _NVRX]
    result, graph = attribute_cycle(line_entries(lines))

    assert result.outcome_attr.value().root_cause_symptom == _SYMPTOM
    assert [s for s in result.symptoms.symptoms() if s.startswith("Python Exception")] == [_SYMPTOM]
    assert {s.datum for s in result.symptoms[_SYMPTOM].all_sources()} == set(lines[:2])
    assert {n.datum for n in graph.nodes if n.rule_name == "BatchSizeAssertion"} == set(lines[:2])


@pytest.mark.parametrize(
    "lines, job_options",
    [
        ([], {}),
        (["1: Traceback (most recent call last):", '1:   File "/app/train.py", line 20, in worker'], {}),
        (["1: fatal"], {}),
        (["1: ValueError: unrecognized application-specific error"], {}),
        (["1: torch.distributed.DistBackendError: process group aborted during shutdown"], {}),
        (["1: RuntimeError: Gloo connectFullMesh failed"], {}),
        ([], {"state": "TIMEOUT", "cancelled_by": "Timeout"}),
        ([], {"state": "CANCELLED", "cancelled_by": "test-user"}),
        ([], {"exit_code": 93}),
        ([], {"exit_code_signal": 15}),
    ],
)
def test_generic_nvrx_failure_remains_undetermined(attribute_cycle, lines, job_options):
    """Report an unavailable reason explicitly without inventing a cause."""
    result, _ = attribute_cycle(line_entries([*lines, _NVRX]), **job_options)

    assert result.outcome_attr.value().outcome == "Fail Undetermined"
    assert result.outcome_attr.value().root_cause_symptom == "NVRx detected failure"
    assert result.outcome_attr.value().hw_cause == _UNAVAILABLE_REASON
    assert "NVRx detected failure" in result.symptoms


@pytest.mark.parametrize("line", [_NO_PROGRESS, _STEP_TIMEOUT])
def test_other_nvrx_mechanisms_keep_their_existing_fallback(attribute_cycle, line):
    result, _ = attribute_cycle(line_entries([line]))

    assert result.outcome_attr.value().outcome == "Fail MultiCategory"
    assert result.outcome_attr.value().root_cause_symptom.startswith("NVRx ")


def test_multiple_nvrx_messages_do_not_mask_specific_error(attribute_cycle):
    result, _ = attribute_cycle(line_entries([f"1: {_ERROR}", _NVRX, _NO_PROGRESS, _STEP_TIMEOUT]))

    assert result.outcome_attr.value().outcome == "Fail User"
    assert result.outcome_attr.value().root_cause_symptom == _SYMPTOM
    assert {_SYMPTOM, "NVRx detected failure", "NVRx no progress across restarts", "NVRx step section timeout"} <= set(
        result.symptoms.symptoms()
    )


@pytest.mark.parametrize("parent_exit", [(0, 0), (93, 0), (0, 15)])
def test_specific_failure_survives_launcher_exit_policy(attribute_cycle, parent_exit):
    result, _ = attribute_cycle(
        line_entries([f"1: {_ERROR}", _NVRX]), exit_code=parent_exit[0], exit_code_signal=parent_exit[1]
    )

    assert result.outcome_attr.value().outcome == "Fail User"
    assert result.outcome_attr.value().root_cause_symptom == _SYMPTOM


@pytest.mark.parametrize(
    "additional_symptom, expected_symptom, expected_outcome",
    [
        ("the process completes", "the process completes", "Pass"),
        ("unmapped-runtime-error", "unknown-symptom-unmapped-runtime-error", "Fail Undetermined"),
    ],
)
def test_assertion_does_not_override_higher_priority_verdicts(
    attribute_cycle, additional_symptom, expected_symptom, expected_outcome
):
    result, _ = attribute_cycle(line_entries([f"1: {_ERROR}", _NVRX]), additional_symptom=additional_symptom)

    assert result.outcome_attr.value().outcome == expected_outcome
    assert result.outcome_attr.value().root_cause_symptom == expected_symptom
