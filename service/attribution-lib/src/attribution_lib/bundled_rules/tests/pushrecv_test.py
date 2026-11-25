# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep pushrecv evidence recognizable through refinement and outcome selection."""

import pytest

from attribution_lib import core
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.observation.inbound import Source as ObservationSource
from test_support.helpers import line_entries


@pytest.mark.parametrize(
    ("line", "expected_nodes"),
    [
        pytest.param("0: pushrecv timeout", {"node01"}, id="reporting-node"),
        pytest.param("0: [0 from 1] pushrecv: stalled", {"node01", "node02"}, id="resolved-peer"),
    ],
)
@pytest.mark.parametrize(
    ("companions", "expected_outcome", "expected_root"),
    [
        pytest.param((), "Fail MultiCategory", "UserBuffer timeout", id="pushrecv-only"),
        pytest.param(("CE deadlock DETECTED",), "Fail InfraSW", "CE deadlock DETECTED", id="ce-deadlock"),
        pytest.param(
            ("PyTorch NCCL watchdog", "UserBuffer timeout"),
            "Fail MultiCategory",
            "UserBuffer timeout",
            id="watchdog-and-userbuffer",
        ),
        pytest.param(
            ("CE deadlock DETECTED", "unregistered test symptom"),
            "Fail Undetermined",
            "unknown-symptom-unregistered test symptom",
            id="genuinely-unknown-companion",
        ),
    ],
)
def test_pushrecv_preserves_existing_outcomes_through_full_pipeline(
    attribution_test_builder, line, expected_nodes, companions, expected_outcome, expected_root
):
    """Recognize pushrecv without masking a known diagnosis or weakening unknown evidence."""
    job = default_job_info(nodes=["node01", "node02"], exit_code_signal=None)
    extracted = attribution_test_builder(
        job_info=job,
        data=line_entries([line]),
        source=ObservationSource.APPLOG,
        insert_tflops_line=False,
    )
    bad_nodes = core.calculate_bad_nodes(
        event_stream=extracted.event_stream, job_info=job, applog_in=extracted.observation_source
    )
    assert set(bad_nodes.my_bad_nodes.attributions) == expected_nodes
    for node in expected_nodes:
        assert "pushrecv" in bad_nodes.my_bad_nodes[node]
    for symptom in companions:
        bad_nodes.my_bad_nodes.add(["node01"], job.source_value(symptom, symptom, "test"))

    result = core.attribute(
        event_stream=extracted.event_stream,
        job_info=job,
        bad_node_result=bad_nodes,
        extract_job_metadata_from_logfile=False,
    )

    outcome = result.outcome_attr.value()
    assert outcome.outcome == expected_outcome
    assert outcome.root_cause_symptom == expected_root
    assert expected_root in result.symptoms
    assert "pushrecv" not in result.symptoms
    assert "unknown-symptom-pushrecv" not in result.symptoms
    for node in expected_nodes:
        evidence = result.bad_nodes_attr.my_bad_nodes[node].attributions["pushrecv"]
        assert evidence.has_interesting()
        assert {source.datum for source in evidence.all_sources()} == {line}
