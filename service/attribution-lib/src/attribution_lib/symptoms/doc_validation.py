# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Validate that every symptom maps to a real documentation entry.

Symptoms declare a documentation slug on ``SymptomDeclaration.doc``. The published documentation exposes the set of valid slugs via its
generated ``url-map.json``. A symptom whose ``doc`` slug is empty or does not
resolve to a published entry produces a dead ("404") documentation link.

This module holds the pure resolution logic so it can be unit-tested without a
rules engine or the docs repository. The CI gate (``scripts/validate_symptom_docs.py``)
supplies the live declarations and slug set and enforces the result.
"""

from __future__ import annotations

from collections.abc import Iterable

from attribution_lib.symptoms.declarations import SymptomDeclaration


def symptom_requires_doc(declaration: SymptomDeclaration) -> bool:
    """Return whether a declaration is a real symptom that must have documentation.

    Excluded (they legitimately have no single doc entry):
      - cascade-only declarations (``cascade_noop``): pruning edges, not symptoms.
      - generic fallback prefixes (``fallback_prefix``, e.g. ``XID-``, ``NV_ERR_``):
        raw ids resolve to these dynamically at runtime, so they do not map to one
        documentation slug.
      - declarations with no ``outcome``: not user-facing symptoms.

    Args:
        declaration: The symptom declaration to classify.

    Returns:
        True if the declaration should carry a resolving documentation slug.
    """
    if declaration.cascade_noop:
        return False
    if declaration.fallback_prefix:
        return False
    return declaration.outcome is not None


def find_unresolved_symptom_docs(
    declarations: Iterable[SymptomDeclaration],
    valid_doc_ids: set[str],
    baseline: frozenset[str] = frozenset(),
) -> list[SymptomDeclaration]:
    """Return real symptoms whose documentation slug does not resolve.

    A slug resolves when it is non-empty and present in ``valid_doc_ids``. Slugs
    listed in ``baseline`` are treated as known, grandfathered gaps and are not
    reported, so the check can be enabled without first backfilling every existing
    missing doc — new additions are still caught.

    Args:
        declarations: Symptom declarations to check.
        valid_doc_ids: The set of documentation ids that resolve (e.g. from url-map.json).
        baseline: Slugs already known to be missing; excluded from the result.

    Returns:
        Declarations, in input order, that require documentation but do not resolve
        and are not baselined.
    """
    unresolved: list[SymptomDeclaration] = []
    for declaration in declarations:
        if not symptom_requires_doc(declaration):
            continue
        slug = declaration.doc
        if slug is not None and slug != "" and slug in valid_doc_ids:
            continue
        if slug is not None and slug in baseline:
            continue
        unresolved.append(declaration)
    return unresolved


def stale_baseline_entries(
    declarations: Iterable[SymptomDeclaration],
    valid_doc_ids: set[str],
    baseline: frozenset[str],
) -> list[str]:
    """Return baseline slugs that now resolve and should be removed from the baseline.

    Keeps the baseline a strict ratchet: once documentation is added for a
    previously-missing symptom, its slug must be dropped from the baseline so the
    gap cannot silently reopen.

    Args:
        declarations: Symptom declarations to check.
        valid_doc_ids: The set of documentation ids that resolve.
        baseline: The current baseline slug set.

    Returns:
        Sorted baseline slugs that are now resolved by a real symptom.
    """
    resolving = {
        declaration.doc
        for declaration in declarations
        if symptom_requires_doc(declaration) and declaration.doc is not None and declaration.doc in valid_doc_ids
    }
    return sorted(baseline & resolving)
