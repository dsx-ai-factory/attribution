# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for symptom-to-documentation resolution."""

from attribution_lib.symptoms.declarations import SymptomDeclaration
from attribution_lib.symptoms.doc_validation import (
    find_unresolved_symptom_docs,
    stale_baseline_entries,
    symptom_requires_doc,
)

# The published slugs a symptom's doc must resolve against (stand-in for url-map.json ids).
VALID_DOC_IDS = {"xid-48", "oom-kill-of-user-task"}


def _symptom(prefix, *, outcome="Fail InfraHW", doc="xid-48", cascade_noop=False, fallback_prefix=False):
    return SymptomDeclaration(
        prefix=prefix,
        outcome=outcome,
        doc=doc,
        cascade_noop=cascade_noop,
        fallback_prefix=fallback_prefix,
    )


class TestSymptomRequiresDoc:
    def test_outcome_bearing_symptom_requires_doc(self):
        assert symptom_requires_doc(_symptom("XID-48")) is True

    def test_cascade_only_declaration_is_exempt(self):
        # Cascade-only declarations describe pruning edges, not user-facing symptoms.
        assert symptom_requires_doc(_symptom("cascade", outcome=None, cascade_noop=True)) is False

    def test_generic_fallback_prefix_is_exempt(self):
        # Dynamic families like "XID-" resolve raw ids at runtime and have no single doc.
        assert symptom_requires_doc(_symptom("XID-", fallback_prefix=True)) is False

    def test_declaration_without_outcome_is_exempt(self):
        assert symptom_requires_doc(_symptom("noop", outcome=None)) is False


class TestFindUnresolvedSymptomDocs:
    def test_resolving_slug_is_not_flagged(self):
        assert find_unresolved_symptom_docs([_symptom("XID-48", doc="xid-48")], VALID_DOC_IDS) == []

    def test_missing_doc_field_is_flagged(self):
        result = find_unresolved_symptom_docs([_symptom("XID-99", doc=None)], VALID_DOC_IDS)
        assert [d.prefix for d in result] == ["XID-99"]

    def test_empty_doc_string_is_flagged(self):
        result = find_unresolved_symptom_docs([_symptom("XID-99", doc="")], VALID_DOC_IDS)
        assert [d.prefix for d in result] == ["XID-99"]

    def test_unresolvable_slug_is_flagged(self):
        result = find_unresolved_symptom_docs([_symptom("kernel-oops", doc="kernel-oops")], VALID_DOC_IDS)
        assert [d.prefix for d in result] == ["kernel-oops"]

    def test_dynamic_and_cascade_symptoms_are_not_flagged(self):
        # Aaron's caveat: dynamic/generic symptoms must not be treated as missing docs.
        declarations = [
            _symptom("XID-", doc=None, fallback_prefix=True),
            _symptom("cascade", outcome=None, doc=None, cascade_noop=True),
        ]
        assert find_unresolved_symptom_docs(declarations, VALID_DOC_IDS) == []

    def test_baselined_slug_is_grandfathered(self):
        # A known-missing slug in the baseline is not reported, so the gate can land
        # without backfilling every existing gap first.
        declarations = [_symptom("kernel-oops", doc="kernel-oops")]
        result = find_unresolved_symptom_docs(declarations, VALID_DOC_IDS, baseline=frozenset({"kernel-oops"}))
        assert result == []

    def test_new_unresolved_still_flagged_despite_baseline(self):
        declarations = [
            _symptom("kernel-oops", doc="kernel-oops"),
            _symptom("brand-new", doc="brand-new"),
        ]
        result = find_unresolved_symptom_docs(declarations, VALID_DOC_IDS, baseline=frozenset({"kernel-oops"}))
        assert [d.prefix for d in result] == ["brand-new"]

    def test_missing_doc_not_grandfathered_by_placeholder(self):
        # The baseline is matched by raw slug only; a symptom with no doc slug has
        # nothing to match on, so a placeholder-style entry must not suppress it -
        # it keeps failing until it declares a real slug.
        declarations = [_symptom("XID-99", doc=None)]
        result = find_unresolved_symptom_docs(declarations, VALID_DOC_IDS, baseline=frozenset({"<missing-doc:XID-99>"}))
        assert [d.prefix for d in result] == ["XID-99"]


class TestStaleBaselineEntries:
    def test_baseline_slug_that_now_resolves_is_reported(self):
        # Documentation was added for a previously-missing symptom; its slug must be
        # dropped from the baseline so the ratchet tightens.
        declarations = [_symptom("oom", doc="oom-kill-of-user-task")]
        stale = stale_baseline_entries(declarations, VALID_DOC_IDS, frozenset({"oom-kill-of-user-task"}))
        assert stale == ["oom-kill-of-user-task"]

    def test_still_missing_baseline_slug_is_not_stale(self):
        declarations = [_symptom("kernel-oops", doc="kernel-oops")]
        stale = stale_baseline_entries(declarations, VALID_DOC_IDS, frozenset({"kernel-oops"}))
        assert stale == []
