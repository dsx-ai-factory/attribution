# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that every symptom maps to a real documentation entry.

Symptoms declare a documentation slug on ``SymptomDeclaration.doc``. The
published documentation exposes valid slugs and names via a generated manifest.
A symptom whose ``doc`` slug is empty or does not resolve to a published entry
produces a dead ("404") documentation link.

This module holds the pure resolution logic so it can be unit-tested without a
rules engine or documentation checkout. An integration layer supplies the live
declarations and generated documentation manifest and enforces the result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from attribution_lib.symptoms.declarations import SymptomDeclaration

SymptomDocKey = tuple[str, str]


def symptom_doc_key(declaration: SymptomDeclaration) -> SymptomDocKey | None:
    """Return the unchanged symptom-name/doc-slug pair used by the baseline."""
    if declaration.doc is None or declaration.doc == "":
        return None
    return declaration.prefix, declaration.doc


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
    baseline: frozenset[SymptomDocKey] = frozenset(),
) -> list[SymptomDeclaration]:
    """Return real symptoms whose documentation slug does not resolve.

    A slug resolves when it is non-empty and present in ``valid_doc_ids``.
    Exact ``(symptom name, doc slug)`` pairs listed in ``baseline`` are
    treated as known, grandfathered gaps and are not reported. Pairing the name
    and slug without normalization prevents a distinct declaration from reusing
    an old missing slug, even if its name differs only in case or punctuation.

    Args:
        declarations: Symptom declarations to check.
        valid_doc_ids: The set of documentation ids that resolve (e.g. from url-map.json).
        baseline: Exact declaration pairs already known to be missing.

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
        key = symptom_doc_key(declaration)
        if key is not None and key in baseline:
            continue
        unresolved.append(declaration)
    return unresolved


def stale_baseline_entries(
    declarations: Iterable[SymptomDeclaration],
    valid_doc_ids: set[str],
    baseline: frozenset[SymptomDocKey],
) -> list[SymptomDocKey]:
    """Return baseline pairs that disappeared or now resolve.

    Keeps the baseline a strict ratchet: an entry must be removed when its exact
    declaration disappears or when documentation is added for it. This prevents
    stale entries from becoming reusable suppression slots.

    Args:
        declarations: Symptom declarations to check.
        valid_doc_ids: The set of documentation ids that resolve.
        baseline: The current baseline declaration-pair set.

    Returns:
        Sorted baseline pairs that are no longer active unresolved declarations.
    """
    active = {
        key
        for declaration in declarations
        if symptom_requires_doc(declaration)
        and (key := symptom_doc_key(declaration)) is not None
        and declaration.doc not in valid_doc_ids
    }
    return sorted(baseline - active)


def build_doc_name_index(
    manifest_entries: Iterable[Mapping[str, object]],
) -> dict[str, frozenset[str]]:
    """Map every resolving documentation id to its exact declared names.

    Each docs-owned manifest entry provides a canonical id, title, aliases, and
    redirect ids. Aliases mean additional emitted names covered by that page; they do
    not imply equivalent outcomes or runtime deduplication. Redirect ids inherit the
    canonical page's declared names while also declaring their own exact id.
    Names retain their case, punctuation, whitespace, and Unicode characters.
    Alternate spellings must be explicit aliases, not inferred from a URL slug.
    """
    names_by_id: dict[str, set[str]] = {}
    redirect_ids_by_id: dict[str, list[str]] = {}
    for entry in manifest_entries:
        doc_id = entry.get("id")
        title = entry.get("title")
        if not isinstance(doc_id, str) or doc_id == "":
            continue
        if doc_id in names_by_id:
            raise ValueError(f"duplicate documentation id: {doc_id!r}")
        declared_names = {doc_id}
        if isinstance(title, str) and title != "":
            declared_names.add(title)
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(f"documentation entry {doc_id!r} has non-list aliases")
        for alias in aliases:
            if not isinstance(alias, str) or alias == "":
                raise ValueError(f"documentation entry {doc_id!r} has an invalid alias: {alias!r}")
            declared_names.add(alias)
        redirect_ids = entry.get("redirect_ids", [])
        if not isinstance(redirect_ids, list):
            raise ValueError(f"documentation entry {doc_id!r} has non-list redirect_ids")
        for redirect_id in redirect_ids:
            if not isinstance(redirect_id, str) or redirect_id == "":
                raise ValueError(f"documentation entry {doc_id!r} has an invalid redirect id: {redirect_id!r}")
        names_by_id[doc_id] = declared_names
        redirect_ids_by_id[doc_id] = redirect_ids

    redirect_owner_by_id: dict[str, str] = {}
    for doc_id, redirect_ids in redirect_ids_by_id.items():
        for redirect_id in redirect_ids:
            if redirect_id in names_by_id and redirect_id != doc_id:
                raise ValueError(
                    f"redirect id {redirect_id!r} for {doc_id!r} conflicts with a canonical documentation id"
                )
            existing_owner = redirect_owner_by_id.get(redirect_id)
            if existing_owner is not None and existing_owner != doc_id:
                raise ValueError(f"redirect id {redirect_id!r} maps to both {existing_owner!r} and {doc_id!r}")
            redirect_owner_by_id[redirect_id] = doc_id

    for redirect_id, doc_id in redirect_owner_by_id.items():
        names_by_id[redirect_id] = names_by_id[doc_id] | {redirect_id}

    return {doc_id: frozenset(names) for doc_id, names in names_by_id.items()}


def find_undeclared_symptom_names(
    declarations: Iterable[SymptomDeclaration],
    doc_names_by_id: Mapping[str, frozenset[str]],
) -> list[SymptomDeclaration]:
    """Return resolving targets that do not declare their emitted symptom name.

    This is deliberately an exact string-equality check, not a review of the
    document's technical correctness. Missing targets are reported separately
    by :func:`find_unresolved_symptom_docs`.
    """
    gaps: list[SymptomDeclaration] = []
    for declaration in declarations:
        if not symptom_requires_doc(declaration) or declaration.doc is None:
            continue
        target_names = doc_names_by_id.get(declaration.doc)
        if target_names is None:
            continue
        if declaration.prefix in target_names:
            continue
        gaps.append(declaration)
    return gaps
