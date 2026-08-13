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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Optional

from attribution_lib.exceptions import (
    DuplicateSymptomDeclarationError,
    DuplicateSymptomMetadataError,
    SymptomValidationError,
    UnknownSymptomError,
)
from attribution_lib.symptoms.declarations import (
    CascadeScope,
    DagDiscardBehavior,
    OutcomePriority,
    SymptomCategory,
    SymptomDeclaration,
)
from attribution_lib.symptoms.doc_validation import symptom_requires_doc
from attribution_lib.symptoms.predicate_registry import SymptomPredicate

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
    doc: Optional[str] = None


@dataclass(frozen=True, slots=True)
class GeneratedSymptomRegistry:
    """Symptom registry generated from declarations.

    This is the runtime-facing snapshot used for outcome selection, symptom
    vocabulary checks, and cascade pruning.
    """

    registered_symptoms: frozenset[str]
    outcome_rules: tuple[Mapping[str, tuple[str, str]], ...]
    metadata: Mapping[str, GeneratedSymptomMetadata]
    scoped_cascade_discards: Mapping[CascadeScope, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    symptoms_by_category: Mapping[SymptomCategory, frozenset[str]] = field(default_factory=dict)
    # Runtime fallback prefixes for dynamic symptom ids. For example,
    # "NV_ERR_INSUFFICIENT_RESOURCES ..." resolves to the declared "NV_ERR_"
    # prefix. Add new runtime fallbacks with fallback_prefix=True in declarations.
    # Stored longest-first so resolve_symptom picks the most specific prefix.
    fallback_prefixes: tuple[str, ...] = ()
    # Per-prefix projection behavior when the DAG discards a symptom (default DEMOTE
    # for anything absent here). Read by project_node_symptoms_to_bad_nodes.
    dag_discard_behaviors: Mapping[str, DagDiscardBehavior] = field(default_factory=dict)
    # Substring aliases: {substring: target prefix}. resolve_symptom resolves any id
    # containing a substring to its target (e.g. "non-fatal" -> the NON_FATAL prefix).
    substring_aliases: Mapping[str, str] = field(default_factory=dict)
    # Documentation slug per symptom prefix. Not used for matching.
    symptom_docs: Mapping[str, str] = field(default_factory=dict)
    # Conditional-discard predicates resolved by name, registered onto the engine via
    # @symptom_predicate. symptom_set_operator resolves conditional_discards through this.
    predicates: Mapping[str, SymptomPredicate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        registered_symptoms = frozenset(self.registered_symptoms)
        outcome_rules = tuple(
            MappingProxyType({prefix: (outcome_data[0], outcome_data[1]) for prefix, outcome_data in tier.items()})
            for tier in self.outcome_rules
        )
        metadata = MappingProxyType(dict(self.metadata))
        scoped_cascade_discards = MappingProxyType(
            {
                scope: MappingProxyType(dict(discards_by_symptom))
                for scope, discards_by_symptom in self.scoped_cascade_discards.items()
            }
        )
        symptoms_by_category = MappingProxyType(
            {category: frozenset(symptoms) for category, symptoms in self.symptoms_by_category.items()}
        )
        # Sort longest prefix first so the most specific generic prefix wins
        # during resolve_symptom; the declared fallback prefixes are disjoint, so
        # order is otherwise moot.
        fallback_prefixes = tuple(sorted(self.fallback_prefixes, key=lambda symptom: (-len(symptom), symptom)))
        object.__setattr__(self, "registered_symptoms", registered_symptoms)
        object.__setattr__(self, "outcome_rules", outcome_rules)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "scoped_cascade_discards", scoped_cascade_discards)
        object.__setattr__(self, "symptoms_by_category", symptoms_by_category)
        object.__setattr__(self, "fallback_prefixes", fallback_prefixes)
        object.__setattr__(self, "dag_discard_behaviors", MappingProxyType(dict(self.dag_discard_behaviors)))
        object.__setattr__(self, "substring_aliases", MappingProxyType(dict(self.substring_aliases)))
        object.__setattr__(self, "symptom_docs", MappingProxyType(dict(self.symptom_docs)))
        object.__setattr__(self, "predicates", MappingProxyType(dict(self.predicates)))

    def contains(self, symptom: str) -> bool:
        """Return whether a symptom belongs to the declared vocabulary.

        Args:
            symptom: Symptom id.

        Returns:
            True when the normalized symptom is declared.
        """
        return symptom in self.registered_symptoms

    def symptoms_in_category(self, category: SymptomCategory) -> frozenset[str]:
        """Return declared symptoms assigned to a semantic category.

        Args:
            category: Category to query.

        Returns:
            Immutable set of symptom ids assigned to the category.
        """
        return self.symptoms_by_category.get(category, frozenset())

    def predicate_for(self, name: str) -> Optional[SymptomPredicate]:
        """Return the conditional-discard predicate registered under ``name``, or None.

        Args:
            name: Predicate name referenced by a ``conditional_discards`` entry.

        Returns:
            The registered predicate callable, or None when unregistered.
        """
        return self.predicates.get(name)

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
        # Substring aliases win first (e.g. any "(non-fatal) X" resolves to NON_FATAL),
        # matching the legacy in-line check that ran before exact/prefix resolution.
        for substring, target in self.substring_aliases.items():
            if substring in symptom:
                return target
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

    def dag_discard_for(self, symptom: str) -> DagDiscardBehavior:
        """Return the projection behavior when the DAG discards this symptom.

        Defaults to DEMOTE for any symptom without an explicit dag_discard
        declaration (or that does not resolve to a known prefix).
        """
        try:
            resolved = self.resolve_symptom(symptom)
        except UnknownSymptomError:
            return DagDiscardBehavior.DEMOTE
        return self.dag_discard_behaviors.get(resolved, DagDiscardBehavior.DEMOTE)

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

    def cascade_discards_for(
        self,
        symptom: str,
        *,
        scope: CascadeScope = CascadeScope.NODE,
    ) -> list[str]:
        """Return unconditional cascade discard targets for a symptom and scope.

        Args:
            symptom: Symptom id.
            scope: Resource scope where the cascade applies.

        Returns:
            List of discarded symptom ids.
        """
        if scope is not CascadeScope.NODE:
            resolved = self.resolve_symptom(symptom)
            return list(self.scoped_cascade_discards.get(scope, {}).get(resolved, ()))
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
                        doc = self.symptom_docs.get(mapping)
                        return GeneratedOutcomeMatch(
                            symptom=symptom,
                            matched_prefix=mapping,
                            outcome=outcome,
                            hw_cause=hw_cause,
                            tier_index=tier_index,
                            doc=doc,
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


def symptom_docs_from_declarations(declarations: Iterable[SymptomDeclaration]) -> dict[str, str]:
    """Map symptom prefixes to their documentation slug.

    Mirrors outcome_rules_from_declarations, carrying only the documentation slug
    for declarations that require documentation and actually declare one.

    Args:
        declarations: Symptom declarations in insertion order.

    Returns:
        Documentation slug keyed by declaration prefix.
    """
    docs: dict[str, str] = {}
    for declaration in declarations:
        if not symptom_requires_doc(declaration):
            continue
        if declaration.doc is None or declaration.doc == "":
            continue
        docs[declaration.prefix] = declaration.doc
    return docs


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
        if declaration.scope is not CascadeScope.NODE:
            continue
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


def scoped_cascade_discards_from_declarations(
    declarations: Iterable[SymptomDeclaration],
) -> dict[CascadeScope, dict[str, tuple[str, ...]]]:
    """Generate non-node cascade targets grouped by scope and primary symptom.

    Args:
        declarations: Symptom declarations.

    Returns:
        Cascade targets keyed first by scope, then by primary symptom.
    """
    generated: dict[CascadeScope, dict[str, tuple[str, ...]]] = {}
    for declaration in declarations:
        if declaration.scope is CascadeScope.NODE:
            continue
        if declaration.conditional_discards != ():
            raise SymptomValidationError(
                f"Conditional discards are not supported for {declaration.scope.value!r} scope"
            )
        if declaration.cascade_discards == ():
            continue
        by_symptom = generated.setdefault(declaration.scope, {})
        if declaration.prefix in by_symptom:
            raise DuplicateSymptomMetadataError(
                f"Duplicate {declaration.scope.value!r}-scoped symptom metadata for {declaration.prefix!r}"
            )
        by_symptom[declaration.prefix] = declaration.cascade_discards
    return generated


def symptom_categories_from_declarations(
    declarations: Iterable[SymptomDeclaration],
) -> dict[SymptomCategory, frozenset[str]]:
    """Generate category membership from symptom declarations.

    Args:
        declarations: Symptom declarations.

    Returns:
        Symptom ids grouped by category.
    """
    category_by_symptom: dict[str, SymptomCategory] = {}
    symptoms_by_category: dict[SymptomCategory, set[str]] = {}
    for declaration in declarations:
        category = declaration.category
        if category is None:
            continue
        existing = category_by_symptom.get(declaration.prefix)
        if existing is not None and existing is not category:
            raise SymptomValidationError(
                f"Conflicting categories for {declaration.prefix!r}: {existing.value!r} and {category.value!r}"
            )
        category_by_symptom[declaration.prefix] = category
        symptoms_by_category.setdefault(category, set()).add(declaration.prefix)
    return {category: frozenset(symptoms) for category, symptoms in symptoms_by_category.items()}


def registry_from_declarations(
    declarations: Iterable[SymptomDeclaration],
    *,
    predicates: Optional[Mapping[str, SymptomPredicate]] = None,
) -> GeneratedSymptomRegistry:
    """Generate the runtime symptom registry from declarations.

    Args:
        declarations: Symptom declarations loaded by the rule engine.
        predicates: Conditional-discard predicates registered onto the engine, keyed
            by the name that ``conditional_discards`` references. Defaults to empty.

    Returns:
        Runtime-shaped registry snapshot generated from declarative inputs.
    """
    declaration_list = list(declarations)
    return GeneratedSymptomRegistry(
        registered_symptoms=frozenset(known_prefixes_from_declarations(declaration_list)),
        outcome_rules=_immutable_outcome_rules(outcome_rules_from_declarations(declaration_list)),
        metadata=metadata_from_declarations(declaration_list),
        scoped_cascade_discards=scoped_cascade_discards_from_declarations(declaration_list),
        symptoms_by_category=symptom_categories_from_declarations(declaration_list),
        fallback_prefixes=fallback_prefixes_from_declarations(declaration_list),
        dag_discard_behaviors=dag_discard_behaviors_from_declarations(declaration_list),
        substring_aliases=substring_aliases_from_declarations(declaration_list),
        symptom_docs=symptom_docs_from_declarations(declaration_list),
        predicates=predicates if predicates is not None else {},
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


def dag_discard_behaviors_from_declarations(
    declarations: Iterable[SymptomDeclaration],
) -> dict[str, DagDiscardBehavior]:
    """Return per-prefix DAG-discard projection behavior for non-default declarations.

    Args:
        declarations: Symptom declarations.

    Returns:
        Mapping of prefix to behavior for every prefix declared non-DEMOTE.
    """
    return {
        declaration.prefix: declaration.dag_discard
        for declaration in declarations
        if declaration.dag_discard is not DagDiscardBehavior.DEMOTE
    }


def substring_aliases_from_declarations(declarations: Iterable[SymptomDeclaration]) -> dict[str, str]:
    """Return {substring: target prefix} for declarations with resolve_contains set.

    Args:
        declarations: Symptom declarations.

    Returns:
        Mapping of substring to the declaring prefix.

    Raises:
        SymptomValidationError: If a declaration sets resolve_contains to the empty
            string. An empty needle is contained in every symptom, so it would hijack
            resolution of the entire vocabulary (aliases are checked before exact and
            prefix matches).
        DuplicateSymptomDeclarationError: If two prefixes claim the same substring,
            or if one active substring is nested inside another. resolve_symptom
            returns on the first matching substring, so a nested pair would make
            resolution depend on declaration-file load order; rejecting it at build
            time keeps resolution deterministic as the alias set grows.
    """
    aliases: dict[str, str] = {}
    for declaration in declarations:
        substring = declaration.resolve_contains
        if substring is None:
            continue
        if substring == "":
            raise SymptomValidationError(
                f"Empty resolve_contains alias declared by {declaration.prefix!r}: "
                "an empty substring matches every symptom."
            )
        existing = aliases.get(substring)
        if existing is not None and existing != declaration.prefix:
            raise DuplicateSymptomDeclarationError(
                f"Conflicting resolve_contains alias {substring!r}: "
                f"declared by both {existing!r} and {declaration.prefix!r}"
            )
        aliases[substring] = declaration.prefix
    substrings = list(aliases)
    for index, outer in enumerate(substrings):
        for inner in substrings[index + 1 :]:
            if outer in inner or inner in outer:
                raise DuplicateSymptomDeclarationError(
                    f"Overlapping resolve_contains substrings {outer!r} "
                    f"(prefix {aliases[outer]!r}) and {inner!r} (prefix {aliases[inner]!r}): "
                    "one is nested in the other, making resolution order-dependent."
                )
    return aliases


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
