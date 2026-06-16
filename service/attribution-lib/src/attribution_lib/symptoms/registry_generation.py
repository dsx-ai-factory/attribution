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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

from attribution_lib.exceptions import (
    DuplicateSymptomDeclarationError,
    DuplicateSymptomMetadataError,
    SymptomValidationError,
    UnknownSymptomError,
)
from attribution_lib.symptoms.declarations import OutcomePriority, SymptomDeclaration

OUTCOME_PRIORITIES: tuple[OutcomePriority, ...] = tuple(sorted(OutcomePriority, key=lambda priority: priority.value))


@dataclass(frozen=True, slots=True)
class GeneratedSymptomMetadata:
    """Cascade metadata for a declared symptom, with string-typed targets."""

    id: str
    cascade_discards: tuple[str, ...] = ()
    conditional_discards: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class GeneratedOutcomeMatch:
    """Outcome match selected from generated symptom registry rules."""

    symptom: str
    matched_prefix: str
    outcome: str
    hw_cause: str
    tier_index: int


@dataclass(frozen=True, slots=True)
class GeneratedSymptomRegistry:
    """Symptom registry generated from declarations.

    This is the runtime-facing snapshot used for outcome selection, symptom
    vocabulary checks, and cascade pruning.
    """

    registered_symptoms: frozenset[str]
    outcome_rules: tuple[Mapping[str, tuple[str, str]], ...]
    metadata: Mapping[str, GeneratedSymptomMetadata]
    # Runtime fallback prefixes for dynamic symptom ids. For example,
    # "NV_ERR_INSUFFICIENT_RESOURCES ..." resolves to the declared "NV_ERR_"
    # prefix. Add new runtime fallbacks with fallback_prefix=True in declarations.
    # Stored longest-first so resolve_symptom picks the most specific prefix.
    fallback_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        registered_symptoms = frozenset(self.registered_symptoms)
        outcome_rules = tuple(
            MappingProxyType({prefix: (outcome_data[0], outcome_data[1]) for prefix, outcome_data in tier.items()})
            for tier in self.outcome_rules
        )
        metadata = MappingProxyType(dict(self.metadata))
        # Sort longest prefix first so the most specific generic prefix wins
        # during resolve_symptom; the declared fallback prefixes are disjoint, so
        # order is otherwise moot.
        fallback_prefixes = tuple(sorted(self.fallback_prefixes, key=lambda symptom: (-len(symptom), symptom)))
        object.__setattr__(self, "registered_symptoms", registered_symptoms)
        object.__setattr__(self, "outcome_rules", outcome_rules)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "fallback_prefixes", fallback_prefixes)

    def contains(self, symptom: str) -> bool:
        """Return whether a symptom belongs to the declared vocabulary.

        Args:
            symptom: Symptom id.

        Returns:
            True when the normalized symptom is declared.
        """
        return symptom in self.registered_symptoms

    def resolve_symptom(self, symptom: str) -> str:
        """Resolve a raw symptom id to a declared vocabulary entry.

        Exact matches win. Dynamic symptom variants fall back to declared generic
        prefixes (those marked ``fallback_prefix=True``).

        Args:
            symptom: Symptom id.

        Returns:
            Declared symptom id.

        Raises:
            UnknownSymptomError: If no declared symptom matches.
        """
        symptom = symptom.strip()
        if symptom in self.registered_symptoms:
            return symptom
        for prefix in self.fallback_prefixes:
            if symptom.startswith(prefix):
                return prefix
        raise UnknownSymptomError(f"{symptom!r} is not a known symptom id")

    def metadata_for(self, symptom: str) -> Optional[GeneratedSymptomMetadata]:
        """Return cascade metadata for a symptom if one exists.

        Args:
            symptom: Symptom id.

        Returns:
            Generated metadata for the resolved symptom, or None.
        """
        try:
            resolved = self.resolve_symptom(symptom)
        except UnknownSymptomError:
            return None
        return self.metadata.get(resolved)

    def get(self, symptom: str) -> Optional[GeneratedSymptomMetadata]:
        """Return cascade metadata by exact symptom id, without prefix resolution.

        Used by cascade traversal and the symptom factory, where the id is
        already a declared symptom value.

        Args:
            symptom: Symptom id.

        Returns:
            Generated metadata for the exact symptom id, or None.
        """
        return self.metadata.get(symptom)

    def has_metadata(self, symptom: str) -> bool:
        """Return whether cascade metadata exists for an exact symptom id.

        Args:
            symptom: Symptom id.

        Returns:
            True when the exact symptom id has registered cascade metadata.
        """
        return symptom in self.metadata

    def cascade_discards_for(self, symptom: str) -> list[str]:
        """Return unconditional cascade discard targets for a symptom.

        Args:
            symptom: Symptom id.

        Returns:
            List of discarded symptom ids.
        """
        metadata = self.metadata_for(symptom)
        if metadata is None:
            return []
        return list(metadata.cascade_discards)

    def outcome_for(
        self,
        symptoms: Iterable[str],
        *,
        ignored_prefixes: tuple[str, ...] = ("plr_tx_bw_loss_high",),
    ) -> Optional[GeneratedOutcomeMatch]:
        """Return the highest-priority outcome match for symptoms.

        Iterates outcome tiers in priority order, and within each tier examines
        symptoms in sorted order and mapping prefixes in insertion order.

        Args:
            symptoms: Symptom ids to classify.
            ignored_prefixes: Symptom prefixes excluded from outcome selection.

        Returns:
            Selected outcome match, or None if no mapping matches.
        """
        normalized_symptoms = sorted(symptoms)
        for tier_index, mapping_list in enumerate(self.outcome_rules):
            for symptom in normalized_symptoms:
                if any(symptom.startswith(prefix) for prefix in ignored_prefixes):
                    continue
                for mapping, outcome_data in mapping_list.items():
                    if symptom.startswith(mapping):
                        outcome, hw_cause = outcome_data
                        return GeneratedOutcomeMatch(
                            symptom=symptom,
                            matched_prefix=mapping,
                            outcome=outcome,
                            hw_cause=hw_cause,
                            tier_index=tier_index,
                        )
        return None


def outcome_rules_from_declarations(declarations: Iterable[SymptomDeclaration]) -> list[dict[str, list[str]]]:
    """Generate ordered outcome tiers from declarations.

    Args:
        declarations: Symptom declarations in insertion order.

    Returns:
        Ordered tiers preserving declaration order within each tier.
    """
    tiers: list[dict[str, list[str]]] = [dict() for _ in OUTCOME_PRIORITIES]
    seen: set[str] = set()
    for declaration in declarations:
        if declaration.outcome is None:
            continue
        if declaration.priority is None:
            raise SymptomValidationError(f"Outcome declaration {declaration.prefix!r} is missing priority")
        if declaration.hw_cause is None:
            raise SymptomValidationError(f"Outcome declaration {declaration.prefix!r} is missing hw_cause")
        if declaration.prefix in seen:
            raise DuplicateSymptomDeclarationError(f"Duplicate outcome declaration for {declaration.prefix!r}")
        seen.add(declaration.prefix)
        tier_index = _tier_for_priority(declaration.priority)
        tiers[tier_index][declaration.prefix] = [declaration.outcome, declaration.hw_cause]
    return tiers


def metadata_from_declarations(
    declarations: Iterable[SymptomDeclaration],
) -> dict[str, GeneratedSymptomMetadata]:
    """Generate normalized cascade metadata from declarations.

    Args:
        declarations: Symptom declarations.

    Returns:
        Metadata keyed by symptom prefix for declarations with cascade data.
    """
    generated: dict[str, GeneratedSymptomMetadata] = {}
    for declaration in declarations:
        if declaration.prefix in generated:
            raise DuplicateSymptomMetadataError(f"Duplicate symptom metadata for {declaration.prefix!r}")
        if (
            declaration.cascade_discards == ()
            and declaration.conditional_discards == ()
            and not declaration.cascade_noop
        ):
            continue
        generated[declaration.prefix] = GeneratedSymptomMetadata(
            id=declaration.prefix,
            cascade_discards=declaration.cascade_discards,
            conditional_discards=declaration.conditional_discards,
        )
    return generated


def registry_from_declarations(
    declarations: Iterable[SymptomDeclaration],
) -> GeneratedSymptomRegistry:
    """Generate the runtime symptom registry from declarations.

    Args:
        declarations: Symptom declarations loaded by the rule engine.

    Returns:
        Runtime-shaped registry snapshot generated from declarative inputs.
    """
    declaration_list = list(declarations)
    return GeneratedSymptomRegistry(
        registered_symptoms=frozenset(known_prefixes_from_declarations(declaration_list)),
        outcome_rules=_immutable_outcome_rules(outcome_rules_from_declarations(declaration_list)),
        metadata=metadata_from_declarations(declaration_list),
        fallback_prefixes=fallback_prefixes_from_declarations(declaration_list),
    )


def known_prefixes_from_declarations(declarations: Iterable[SymptomDeclaration]) -> set[str]:
    """Return all known prefixes declared through the new representation.

    Args:
        declarations: Symptom declarations.

    Returns:
        Set of declared symptom prefixes.
    """
    return {declaration.prefix for declaration in declarations}


def fallback_prefixes_from_declarations(declarations: Iterable[SymptomDeclaration]) -> tuple[str, ...]:
    """Return generic fallback prefixes declared through the new representation.

    Args:
        declarations: Symptom declarations.

    Returns:
        Tuple of prefixes flagged as generic fallbacks, in declaration order.
    """
    return tuple(declaration.prefix for declaration in declarations if declaration.fallback_prefix)


def _tier_for_priority(priority: OutcomePriority) -> int:
    try:
        return OUTCOME_PRIORITIES.index(priority)
    except ValueError as exc:
        raise SymptomValidationError(f"Unknown OutcomePriority {priority!r}") from exc


def _immutable_outcome_rules(
    rules: Iterable[Mapping[str, Sequence[str]]],
) -> tuple[Mapping[str, tuple[str, str]], ...]:
    return tuple(
        MappingProxyType({prefix: (outcome_data[0], outcome_data[1]) for prefix, outcome_data in tier.items()})
        for tier in rules
    )
