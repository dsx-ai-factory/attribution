# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static translation from today's ten flat outcome strings to a typed compound
representation, additive to the existing `OutcomeAttribution.outcome: str`.

A flat outcome string like "Fail InfraHW" conflates three separate questions:
what happened to the job (terminal state), who or what is responsible for a
failure or cancellation (failure domain / cancellation actor), and whether
NVIDIA Attribution is uncertain about that attribution, and why (uncertainty
type).
`CompoundOutcome` splits those back out into typed fields so downstream
consumers can reason about them independently instead of pattern-matching
the string.

This module is additive only: it does not change how outcome.py computes
outcomes, and `OutcomeAttribution.outcome: str` remains the source of truth.
It is a pure translation layer over that existing string.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

from loguru import logger


class TerminalState(str, Enum):
    """What happened to the job. Always present; every outcome maps to at
    least one of these (possibly more than one, when NVIDIA Attribution can't
    narrow it down further -- see "Incomplete Info" in `_OUTCOME_TABLE`)."""

    PASS = "Pass"
    """Job ran successfully."""
    FAIL = "Fail"
    """Job encountered an unexpected failure."""
    EXPECTED_FAIL = "ExpectedFail"
    """Job failed, but failure was expected (likely because it was artificially made to do so.)"""
    CANCELLED = "Cancelled"
    """Job did not complete successfully because it was cancelled by the scheduler."""


class FailureDomain(str, Enum):
    """Possible domains of responsibility carried by `Fail`."""

    USER = "User"
    INFRA_HW = "InfraHW"
    INFRA_SW = "InfraSW"


class CancellationActor(str, Enum):
    """Possible initiators carried by `Cancelled`."""

    USER = "User"
    OPERATOR = "Operator"


class UncertaintyType(str, Enum):
    """NVIDIA Attribution's qualitative classification of the source of uncertainty.

    ALEATORIC describes ambiguity in the available job evidence, including
    insufficient data ("Incomplete Info"). It does not assert that the job
    behaved randomly or that better observations could never resolve it.
    EPISTEMIC describes limitations of NVIDIA Attribution itself, such as missing rules or
    bugs ("Fail Undetermined"). These are the operational definitions from
    the compound-outcome RFC, not a general statistical uncertainty model.

    Neither value measures likelihood, confidence, severity, or an error rate.
    None means no uncertainty source is reported, not guaranteed correctness.
    The source is supplied explicitly, not inferred from the possibility count.
    """

    ALEATORIC = "Aleatoric"
    EPISTEMIC = "Epistemic"


@dataclass(frozen=True, slots=True)
class Pass:
    """The job plausibly passed."""

    state: ClassVar[Literal[TerminalState.PASS]] = TerminalState.PASS


@dataclass(frozen=True, slots=True)
class Fail:
    """One plausible failure domain; None leaves the domain unspecified."""

    state: ClassVar[Literal[TerminalState.FAIL]] = TerminalState.FAIL
    domain: FailureDomain | None = None


@dataclass(frozen=True, slots=True)
class ExpectedFail:
    """The job plausibly failed as expected."""

    state: ClassVar[Literal[TerminalState.EXPECTED_FAIL]] = TerminalState.EXPECTED_FAIL


@dataclass(frozen=True, slots=True)
class Cancelled:
    """One plausible cancellation actor; None leaves the actor unspecified."""

    state: ClassVar[Literal[TerminalState.CANCELLED]] = TerminalState.CANCELLED
    actor: CancellationActor | None = None


Possibility = Pass | Fail | ExpectedFail | Cancelled


@dataclass(frozen=True, slots=True)
class CompoundOutcome:
    """A nonempty set of plausible alternatives and the source of uncertainty.

    Each possibility carries only the payload applicable to its terminal state.
    Multiple Fail or Cancelled values describe alternative domains or actors;
    they do not assert simultaneous causes. Identical possibilities deduplicate
    through set equality. The set describes what NVIDIA Attribution considers plausible:
    omitted values are ruled out by its assessment, except where a payload is
    explicitly unspecified (Fail() or Cancelled()). A singleton selects one
    possibility at the represented level of detail, not certainty of correctness.

    Possibilities have no weights or ranking; multiple entries do not imply
    equal probabilities, and set size is not a confidence score. The optional
    uncertainty field explains why NVIDIA Attribution cannot narrow its assessment, using
    UncertaintyType's NVA-specific definitions. The same alternatives can have
    different sources: "Fail MultiCategory" and "Fail Undetermined" both list
    all failure domains, but report ALEATORIC and EPISTEMIC respectively.

    A possibility's payload can later become a structured domain without
    changing the outer set or its union operation. The flattened properties
    preserve the existing API projection. Unspecified payloads contribute no
    domain or actor to that projection, as required for "Incomplete Info".
    """

    possibilities: frozenset[Possibility]
    uncertainty: UncertaintyType | None = None

    def __post_init__(self) -> None:
        if not self.possibilities:
            raise ValueError("CompoundOutcome must have at least one terminal state present")

    @classmethod
    def fail(
        cls,
        *domains: FailureDomain,
        uncertainty: UncertaintyType | None = None,
    ) -> "CompoundOutcome":
        """Describe plausible failure domains, or unspecified failure if omitted."""
        return cls(
            possibilities=frozenset(Fail(domain) for domain in domains) if domains else frozenset({Fail()}),
            uncertainty=uncertainty,
        )

    @property
    def terminal_states(self) -> frozenset[TerminalState]:
        """Return the distinct terminal states represented by the possibilities."""
        return frozenset(possibility.state for possibility in self.possibilities)

    @property
    def failure_domains(self) -> frozenset[FailureDomain]:
        """Return plausible failure domains, omitting unspecified domains."""
        return frozenset(
            possibility.domain
            for possibility in self.possibilities
            if isinstance(possibility, Fail) and possibility.domain is not None
        )

    @property
    def cancellation_actors(self) -> frozenset[CancellationActor]:
        """Return plausible cancellation actors, omitting unspecified actors."""
        return frozenset(
            possibility.actor
            for possibility in self.possibilities
            if isinstance(possibility, Cancelled) and possibility.actor is not None
        )


def combine_candidates(
    candidates: Sequence[CompoundOutcome],
    *,
    uncertainty: UncertaintyType | None = None,
) -> CompoundOutcome:
    """Union plausible alternatives, using the caller's uncertainty reason.

    Input uncertainty is not inferred or propagated. Broad and specific
    possibilities are retained without subsumption or normalization.
    An empty candidate sequence raises ValueError.
    """
    return CompoundOutcome(
        possibilities=frozenset(possibility for candidate in candidates for possibility in candidate.possibilities),
        uncertainty=uncertainty,
    )


# Kept exhaustive against the outcome strings produced by the active rule set.
# Unknown strings use the Fail Undetermined entry through compound_outcome_for.
_OUTCOME_TABLE: dict[str, CompoundOutcome] = {
    "Fail User": CompoundOutcome.fail(FailureDomain.USER),
    "Fail InfraSW": CompoundOutcome.fail(FailureDomain.INFRA_SW),
    "Fail InfraHW": CompoundOutcome.fail(FailureDomain.INFRA_HW),
    "Pass": CompoundOutcome(frozenset({Pass()})),
    "Artificial": CompoundOutcome(frozenset({ExpectedFail()})),
    "Cancelled by User": CompoundOutcome(frozenset({Cancelled(CancellationActor.USER)})),
    "Cancelled by cluster operators": CompoundOutcome(frozenset({Cancelled(CancellationActor.OPERATOR)})),
    "Incomplete Info": CompoundOutcome(
        frozenset({Pass(), Fail(), ExpectedFail(), Cancelled()}),
        uncertainty=UncertaintyType.ALEATORIC,
    ),
    "Fail MultiCategory": CompoundOutcome.fail(*FailureDomain, uncertainty=UncertaintyType.ALEATORIC),
    "Fail Undetermined": CompoundOutcome.fail(*FailureDomain, uncertainty=UncertaintyType.EPISTEMIC),
}


def compound_outcome_for(outcome: str) -> CompoundOutcome:
    """Translate one of today's ten flat outcome strings to its CompoundOutcome.

    An unrecognized string falls back to exactly the table's "Fail Undetermined"
    entry (the same CompoundOutcome value, not a separately maintained copy) and
    logs a warning, rather than raising: consistent with what that entry already
    means ("NVIDIA Attribution's rules are inadequate here"), and consistent with
    attribution-lib's general preference for degrading gracefully at this
    boundary rather than breaking attribution on an unrecognized value.
    """
    result = _OUTCOME_TABLE.get(outcome)
    if result is not None:
        return result
    logger.warning(f"compound_outcome_for: unrecognized outcome {outcome!r}, falling back to Fail Undetermined shape")
    return _OUTCOME_TABLE["Fail Undetermined"]
