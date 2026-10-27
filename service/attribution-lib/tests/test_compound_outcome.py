# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import attribution_lib
from attribution_lib.compound_outcome import (
    CancellationActor,
    Cancelled,
    CompoundOutcome,
    ExpectedFail,
    Fail,
    FailureDomain,
    Pass,
    TerminalState,
    UncertaintyType,
    combine_candidates,
    compound_outcome_for,
)

EXPECTED_TABLE: dict[str, CompoundOutcome] = {
    "Fail User": CompoundOutcome(possibilities=frozenset({Fail(FailureDomain.USER)})),
    "Fail InfraSW": CompoundOutcome(possibilities=frozenset({Fail(FailureDomain.INFRA_SW)})),
    "Fail InfraHW": CompoundOutcome(possibilities=frozenset({Fail(FailureDomain.INFRA_HW)})),
    "Pass": CompoundOutcome(possibilities=frozenset({Pass()})),
    "Artificial": CompoundOutcome(possibilities=frozenset({ExpectedFail()})),
    "Cancelled by User": CompoundOutcome(possibilities=frozenset({Cancelled(CancellationActor.USER)})),
    "Cancelled by cluster operators": CompoundOutcome(possibilities=frozenset({Cancelled(CancellationActor.OPERATOR)})),
    "Incomplete Info": CompoundOutcome(
        possibilities=frozenset({Pass(), Fail(), ExpectedFail(), Cancelled()}),
        uncertainty=UncertaintyType.ALEATORIC,
    ),
    "Fail MultiCategory": CompoundOutcome(
        possibilities=frozenset({Fail(FailureDomain.USER), Fail(FailureDomain.INFRA_HW), Fail(FailureDomain.INFRA_SW)}),
        uncertainty=UncertaintyType.ALEATORIC,
    ),
    "Fail Undetermined": CompoundOutcome(
        possibilities=frozenset({Fail(FailureDomain.USER), Fail(FailureDomain.INFRA_HW), Fail(FailureDomain.INFRA_SW)}),
        uncertainty=UncertaintyType.EPISTEMIC,
    ),
}


@pytest.mark.parametrize("outcome_str", sorted(EXPECTED_TABLE))
def test_compound_outcome_for_matches_rfc_table(outcome_str: str) -> None:
    assert compound_outcome_for(outcome_str) == EXPECTED_TABLE[outcome_str]


def test_compound_outcome_for_table_is_exhaustive() -> None:
    """EXPECTED_TABLE covers exactly the outcome strings the active rule set can
    produce, so table drift (a new outcome added upstream, or a stale entry left
    behind) fails this test instead of surfacing as a silent fallback later."""
    registry = attribution_lib.get_symptom_registry()
    produced_outcomes = {outcome for tier in registry.outcome_rules for outcome, _hw_cause in tier.values()}
    assert produced_outcomes == set(EXPECTED_TABLE)


def test_compound_outcome_for_unmapped_falls_back_to_fail_undetermined_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        result = compound_outcome_for("some new outcome string nobody declared")
    assert result == EXPECTED_TABLE["Fail Undetermined"]
    assert "unrecognized outcome" in caplog.text
    assert "some new outcome string nobody declared" in caplog.text


def test_compound_outcome_terminal_states_property() -> None:
    result = CompoundOutcome(possibilities=frozenset({Pass(), Cancelled()}))
    assert result.terminal_states == frozenset({TerminalState.PASS, TerminalState.CANCELLED})


def test_compound_outcome_failure_domains_property_empty_when_no_fail() -> None:
    assert CompoundOutcome(possibilities=frozenset({Pass()})).failure_domains == frozenset()


def test_compound_outcome_cancellation_actors_property_empty_when_no_cancelled() -> None:
    assert CompoundOutcome(possibilities=frozenset({Pass()})).cancellation_actors == frozenset()


def test_combine_candidates_single_passthrough() -> None:
    candidate = CompoundOutcome(possibilities=frozenset({Fail(FailureDomain.USER)}))
    assert combine_candidates([candidate]) == candidate


def test_combine_candidates_unions_distinct_candidates() -> None:
    a = CompoundOutcome(possibilities=frozenset({Pass()}))
    b = CompoundOutcome(possibilities=frozenset({Fail(FailureDomain.INFRA_HW)}))
    result = combine_candidates([a, b])
    assert result.terminal_states == frozenset({TerminalState.PASS, TerminalState.FAIL})
    assert result.failure_domains == frozenset({FailureDomain.INFRA_HW})
    assert result.cancellation_actors == frozenset()


def test_combine_candidates_unions_fail_domains_across_candidates() -> None:
    a = CompoundOutcome(possibilities=frozenset({Fail(FailureDomain.USER)}))
    b = CompoundOutcome(possibilities=frozenset({Fail(FailureDomain.INFRA_SW)}))
    result = combine_candidates([a, b])
    assert result.terminal_states == frozenset({TerminalState.FAIL})
    assert result.failure_domains == frozenset({FailureDomain.USER, FailureDomain.INFRA_SW})


def test_combine_candidates_applies_given_uncertainty() -> None:
    a = CompoundOutcome(possibilities=frozenset({Fail()}))
    result = combine_candidates([a], uncertainty=UncertaintyType.EPISTEMIC)
    assert result.uncertainty == UncertaintyType.EPISTEMIC


def test_combine_candidates_empty_uncertainty_defaults_to_none() -> None:
    a = CompoundOutcome(possibilities=frozenset({Pass()}))
    assert combine_candidates([a]).uncertainty is None


def test_compound_outcome_rejects_no_terminal_state() -> None:
    with pytest.raises(ValueError, match="at least one terminal state"):
        CompoundOutcome(possibilities=frozenset())


def test_combine_candidates_rejects_empty_candidate_list() -> None:
    with pytest.raises(ValueError, match="at least one terminal state"):
        combine_candidates([])


def test_compound_outcome_allows_fail_and_cancelled_with_no_domain_or_actor() -> None:
    """Unspecified domains and actors preserve the Incomplete Info mapping."""
    result = CompoundOutcome(possibilities=frozenset({Fail(), Cancelled()}))
    assert result.terminal_states == frozenset({TerminalState.FAIL, TerminalState.CANCELLED})
    assert result.failure_domains == frozenset()
    assert result.cancellation_actors == frozenset()


def test_fail_shortcut_preserves_alternatives_and_deduplicates() -> None:
    result = CompoundOutcome.fail(
        FailureDomain.USER,
        FailureDomain.INFRA_SW,
        FailureDomain.USER,
        uncertainty=UncertaintyType.ALEATORIC,
    )
    assert result.possibilities == frozenset({Fail(FailureDomain.USER), Fail(FailureDomain.INFRA_SW)})
    assert result.uncertainty is UncertaintyType.ALEATORIC


def test_fail_shortcut_without_domains_keeps_failure_plausible() -> None:
    result = CompoundOutcome.fail()
    assert result.possibilities == frozenset({Fail()})
    assert result.terminal_states == frozenset({TerminalState.FAIL})
    assert result.failure_domains == frozenset()


def test_combine_candidates_retains_all_variants_and_unspecified_payloads() -> None:
    broad = compound_outcome_for("Incomplete Info")
    specific = CompoundOutcome(
        frozenset(
            {
                Fail(FailureDomain.INFRA_HW),
                Cancelled(CancellationActor.USER),
                Cancelled(CancellationActor.OPERATOR),
            }
        )
    )
    result = combine_candidates([broad, specific, specific], uncertainty=UncertaintyType.EPISTEMIC)
    assert result.possibilities == broad.possibilities | specific.possibilities
    assert result.terminal_states == frozenset(TerminalState)
    assert result.failure_domains == frozenset({FailureDomain.INFRA_HW})
    assert result.cancellation_actors == frozenset(CancellationActor)
    assert result.uncertainty is UncertaintyType.EPISTEMIC


def test_same_possibilities_can_have_different_uncertainty_sources() -> None:
    ambiguous = compound_outcome_for("Fail MultiCategory")
    inadequate = compound_outcome_for("Fail Undetermined")
    assert ambiguous.possibilities == inadequate.possibilities
    assert ambiguous.uncertainty is UncertaintyType.ALEATORIC
    assert inadequate.uncertainty is UncertaintyType.EPISTEMIC
