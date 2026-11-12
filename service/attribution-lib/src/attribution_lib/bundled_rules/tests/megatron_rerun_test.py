# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Megatron rerun verdicts replace "Bad Token" on the node that logged them."""

import pytest

from attribution_lib import core
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.observation.inbound import Source as ObservationSource
from test_support.helpers import line_entries

_PREFIX = "1: ERROR:megatron.core.rerun_state_machine:"
_UNEXPECTED = (
    f"{_PREFIX}Unexpected result inf on rank 1 at iteration #56427 invocation #12 "
    "(message='found Inf in local grad norm for bucket #0 in backward pass before data-parallel "
    "communication collective')"
)
_FIRST_NOT_REPRODUCIBLE = (
    f"{_PREFIX}Rank 1, node host-f85954d2, device 1, iteration #56427: First rerun: unexpected result "
    "is not reproducible within the tolerance (11520.0 != inf)!"
)
_TRANSIENT = f"{_PREFIX}Rank 1, node host-f85954d2, device 1, iteration #56427: Possible transient error!!"
_FIRST_REPRODUCIBLE = (
    f"{_PREFIX}Rank 1, node host-f85954d2, device 1, iteration #56427: First rerun: unexpected result "
    "is reproducible within the tolerance (inf = inf). Need to rerun on a different GPU to verify correctness.!"
)
# Megatron logs the second-rerun lines at WARNING. The applog pre-filter drops
# lines that contain "WARNING:megatron", so these tests use a log format without
# that text. The "Possible persistent error" and "Correct result" lines that
# follow are always logged at ERROR.
_SECOND_NOT_REPRODUCIBLE = (
    "0: [WARNING | megatron.core.rerun_state_machine]: Second rerun: unexpected result is not reproducible on a "
    "different GPU, therefore was likely incorrect (11520.0 != inf)"
)
_PERSISTENT = "0: ERROR:megatron.core.rerun_state_machine:Rank 0, node host-0c1d2e3f, device 0, iteration #56427: Possible persistent error!!"
_SECOND_REPRODUCIBLE = (
    "0: [WARNING | megatron.core.rerun_state_machine]: Second rerun: unexpected result is reproducible on a "
    "different GPU, therefore it was likely correct (inf = inf)"
)
_CORRECT_RESULT = (
    "0: ERROR:megatron.core.rerun_state_machine:Rank 0, node host-0c1d2e3f, device 0, iteration #56427: "
    "Correct result (but possible Application error) (found Inf in local grad norm)!"
)


@pytest.fixture
def attribute_lines(attribution_test_builder):
    """Return a function that attributes application log lines on two nodes.

    Args:
        attribution_test_builder: Fixture that builds an event stream from entries.

    Returns:
        A function that takes log lines and returns the AttributionResult.
    """

    def _attribute(lines):
        """Attribute the given application log lines.

        Args:
            lines: Application log lines; the "R:" prefix names the global rank.

        Returns:
            The AttributionResult.
        """
        job = default_job_info(nodes=["node01", "node02"])
        extracted = attribution_test_builder(
            job_info=job,
            data=line_entries(lines),
            source=ObservationSource.APPLOG,
        )
        bad_nodes = core.calculate_bad_nodes(
            event_stream=extracted.event_stream,
            job_info=job,
            applog_in=extracted.observation_source,
        )
        return core.attribute(
            event_stream=extracted.event_stream,
            job_info=job,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
        )

    return _attribute


def _interesting_symptoms(result, node):
    """Return the interesting-only symptoms on a node.

    Args:
        result: AttributionResult.
        node: Hostname.

    Returns:
        The set of symptoms on the node that have no bad source.
    """
    group = result.bad_nodes_attr.my_bad_nodes[node].attributions
    return {symptom for symptom, sources in group.all_items() if sources.has_bad() is False}


def test_bad_token_stays_without_rerun_verdict(attribute_lines):
    result = attribute_lines([_UNEXPECTED])

    assert "Bad Token" in result.symptoms
    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == "Bad Token"


def test_transient_verdict_discards_bad_token(attribute_lines):
    result = attribute_lines([_UNEXPECTED, _FIRST_NOT_REPRODUCIBLE, _TRANSIENT])

    assert "Bad Token" not in result.symptoms
    assert "node02" not in result.bad_nodes_attr.my_bad_nodes or (
        "Bad Token" not in result.bad_nodes_attr.my_bad_nodes["node02"]
    )
    assert {
        "Bad Token",
        "Unexpected value not reproduced on same GPU",
        "Transient error reported by Megatron",
    } <= _interesting_symptoms(result, "node02")
    # Nothing else explains the failed job, so the generic fallback wins.
    assert result.outcome_attr.value().root_cause_symptom == "job failed"


def test_first_rerun_reproducible_is_undetermined(attribute_lines):
    result = attribute_lines([_UNEXPECTED, _FIRST_REPRODUCIBLE])

    assert "Bad Token" not in result.symptoms
    assert "Unexpected value reproduced on same GPU" in _interesting_symptoms(result, "node02")
    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail Undetermined"
    assert outcome.root_cause_symptom == "Unexpected value reproduced on same GPU"


def test_second_rerun_reproducible_is_user_without_blaming_rerun_node(attribute_lines):
    result = attribute_lines([_SECOND_REPRODUCIBLE, _CORRECT_RESULT])

    assert result.bad_nodes_attr.my_bad_nodes["node01"].has_bad() is False
    assert _interesting_symptoms(result, "node01") == {
        "Unexpected value reproduced on other GPU",
        "Application error reported by Megatron",
    }
    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == "Application error reported by Megatron"


def test_second_rerun_verdict_discards_bad_token_on_every_node(attribute_lines):
    """The second rerun runs on node01, but "Bad Token" came from rank 1 on node02."""
    result = attribute_lines([_UNEXPECTED, _SECOND_NOT_REPRODUCIBLE, _PERSISTENT])

    assert "Bad Token" not in result.symptoms
    assert "Bad Token" in _interesting_symptoms(result, "node02")
    assert {
        "Unexpected value not reproduced on other GPU",
        "Persistent error reported by Megatron",
    } <= _interesting_symptoms(result, "node01")
    assert result.outcome_attr.value().root_cause_symptom == "job failed"
