# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decorators that declare symptom metadata from a rule file."""

from typing import Optional

from attribution_lib.declarative.context import _optional_rule_loader, _require_rule_loader
from attribution_lib.exceptions import RuleIngestionError
from attribution_lib.rule_types.registration import _get_caller_location
from attribution_lib.symptoms.declarations import (
    CascadeScope,
    DagDiscardBehavior,
    OutcomePriority,
    SymptomCategory,
    SymptomDeclaration,
)
from attribution_lib.symptoms.family import SymptomFamily
from attribution_lib.symptoms.predicate_registry import SymptomPredicate


def _module_location_marker() -> None:
    pass


# _get_caller_location lives in rule_types.registration, so its own frame-skip
# would stop here. Pass this file too, or every symptom would be reported as
# declared at this module.
_THIS_FILE = _module_location_marker.__code__.co_filename


def outcome_symptom(
    prefix: str,
    *,
    outcome: str,
    priority: OutcomePriority,
    doc: Optional[str],
    hw_cause: str = "Not Applicable",
    cascade_discards: tuple[str, ...] = (),
    conditional_discards: tuple[tuple[str, str], ...] = (),
    cascade_noop: bool = False,
    category: Optional[SymptomCategory] = None,
    fallback_prefix: bool = False,
    dag_discard: DagDiscardBehavior = DagDiscardBehavior.DEMOTE,
    family: Optional[SymptomFamily] = None,
) -> None:
    """Declare a symptom that can drive final outcome selection.

    Args:
        prefix: Symptom prefix matched via startswith during outcome selection.
        outcome: NVIDIA Attribution outcome assigned when this symptom wins.
        priority: Outcome selection priority tier (lower wins).
        doc: Documentation anchor, or None for a fallback without a catalog entry.
        hw_cause: Hardware cause text preserved in outcome attribution.
        cascade_discards: Symptoms discarded when this symptom is present.
        conditional_discards: Predicate-gated cascade discards.
        cascade_noop: Preserve this symptom as a cascade graph node that has no
            discard edges.
        category: Optional semantic category for registry queries.
        fallback_prefix: Mark this prefix as a generic fallback used to resolve
            dynamic symptom variants (such as "XID-" and "NV_ERR_").
        dag_discard: Projection behavior when the DAG discards this symptom.
        family: Optional structured family schema or member identity.
    """
    if (doc is not None and doc.strip() == "") or (doc is None and not fallback_prefix):
        raise RuleIngestionError(f"outcome_symptom({prefix!r}) requires a non-empty doc anchor")
    loader = _require_rule_loader("declaring outcome symptom")
    loader.register_symptom_declaration(
        SymptomDeclaration(
            prefix=prefix,
            outcome=outcome,
            hw_cause=hw_cause,
            priority=priority,
            doc=doc,
            cascade_discards=cascade_discards,
            conditional_discards=conditional_discards,
            cascade_noop=cascade_noop,
            category=category,
            fallback_prefix=fallback_prefix,
            dag_discard=dag_discard,
            location=_get_caller_location(skip_files=frozenset([_THIS_FILE])),
            family=family,
        )
    )


def symptom_predicate(func: SymptomPredicate) -> SymptomPredicate:
    """Register a conditional-discard predicate onto the active loader.

    Init-scoped like the symptom-declaration decorators: it attaches the callable
    to whichever :class:`RuleLoader` is loading the rule file, keyed by the function
    name that ``conditional_discards`` references. Unlike ``known_symptom``, this may
    also run when the predicate module is imported directly for its function objects
    (e.g. unit tests), where no loader is active; that is a no-op, since registration
    only matters during a rule load.
    """
    loader = _optional_rule_loader()
    if loader is not None:
        # getattr rather than func.__name__: the parameter is typed as a Callable alias,
        # which is not guaranteed a __name__ by strict type checkers.
        name: str = getattr(func, "__name__")
        loader.register_symptom_predicate(name, func)
    return func


def known_symptom(
    prefix: str,
    *,
    category: Optional[SymptomCategory] = None,
    fallback_prefix: bool = False,
    dag_discard: DagDiscardBehavior = DagDiscardBehavior.DEMOTE,
    family: Optional[SymptomFamily] = None,
) -> None:
    """Declare a symptom that belongs to the vocabulary only.

    Args:
        prefix: Symptom prefix known to the registry.
        category: Optional semantic category for registry queries.
        fallback_prefix: Mark this prefix as a generic fallback used to resolve
            dynamic symptom variants (such as "RuntimeError").
        dag_discard: Projection behavior when the DAG discards this symptom.
        family: Optional structured family schema or member identity.
    """
    loader = _require_rule_loader("declaring known symptom")
    loader.register_symptom_declaration(
        SymptomDeclaration(
            prefix=prefix,
            category=category,
            fallback_prefix=fallback_prefix,
            dag_discard=dag_discard,
            location=_get_caller_location(skip_files=frozenset([_THIS_FILE])),
            family=family,
        )
    )


def cascade_symptom(
    prefix: str,
    *,
    cascade_discards: tuple[str, ...],
    conditional_discards: tuple[tuple[str, str], ...] = (),
    scope: CascadeScope = CascadeScope.NODE,
    category: Optional[SymptomCategory] = None,
    dag_discard: DagDiscardBehavior = DagDiscardBehavior.DEMOTE,
    resolve_contains: Optional[str] = None,
    family: Optional[SymptomFamily] = None,
) -> None:
    """Declare a cascade-only symptom used for pruning, not outcome selection.

    Args:
        prefix: Symptom prefix that triggers cascade pruning.
        cascade_discards: Symptoms discarded when this symptom is present.
        conditional_discards: Predicate-gated cascade discards.
        scope: Resource scope where cascade discards apply.
        category: Optional semantic category for registry queries.
        dag_discard: Projection behavior when the DAG discards this symptom.
        resolve_contains: Any raw symptom id containing this substring resolves to
            this prefix.
        family: Optional structured family schema or member identity.
    """
    loader = _require_rule_loader("declaring cascade symptom")
    loader.register_symptom_declaration(
        SymptomDeclaration(
            prefix=prefix,
            cascade_discards=cascade_discards,
            conditional_discards=conditional_discards,
            scope=scope,
            category=category,
            dag_discard=dag_discard,
            resolve_contains=resolve_contains,
            location=_get_caller_location(skip_files=frozenset([_THIS_FILE])),
            family=family,
        )
    )
