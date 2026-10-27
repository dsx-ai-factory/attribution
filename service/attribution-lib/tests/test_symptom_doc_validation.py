# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for symptom-to-documentation resolution."""

import pytest

from attribution_lib.symptoms.declarations import SymptomDeclaration
from attribution_lib.symptoms.doc_validation import (
    build_doc_name_index,
    find_undeclared_symptom_names,
    find_unresolved_symptom_docs,
    stale_baseline_entries,
    symptom_doc_key,
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

    def test_baselined_declaration_pair_is_grandfathered(self):
        declarations = [_symptom("kernel-oops", doc="kernel-oops")]
        baseline = frozenset({("kernel-oops", "kernel-oops")})
        result = find_unresolved_symptom_docs(declarations, VALID_DOC_IDS, baseline=baseline)
        assert result == []

    def test_new_unresolved_still_flagged_despite_baseline(self):
        declarations = [
            _symptom("kernel-oops", doc="kernel-oops"),
            _symptom("brand-new", doc="brand-new"),
        ]
        baseline = frozenset({("kernel-oops", "kernel-oops")})
        result = find_unresolved_symptom_docs(declarations, VALID_DOC_IDS, baseline=baseline)
        assert [d.prefix for d in result] == ["brand-new"]

    def test_new_symptom_cannot_reuse_baselined_slug(self):
        declarations = [_symptom("Brand new symptom", doc="kernel-oops")]
        baseline = frozenset({("kernel-oops", "kernel-oops")})
        result = find_unresolved_symptom_docs(declarations, VALID_DOC_IDS, baseline=baseline)
        assert [d.prefix for d in result] == ["Brand new symptom"]

    @pytest.mark.parametrize("prefix", ["GPU A/B", "gpu a+b", "GPU A B", "GPU A+B "])
    def test_similar_name_cannot_reuse_baseline(self, prefix):
        original = _symptom("GPU A+B", doc="gpu-error")
        added = _symptom(prefix, doc="gpu-error")
        baseline = frozenset({("GPU A+B", "gpu-error")})

        assert symptom_doc_key(original) == ("GPU A+B", "gpu-error")
        assert find_unresolved_symptom_docs([original, added], set(), baseline) == [added]


class TestStaleBaselineEntries:
    def test_baseline_pair_that_now_resolves_is_reported(self):
        declarations = [_symptom("oom", doc="oom-kill-of-user-task")]
        baseline = frozenset({("oom", "oom-kill-of-user-task")})
        stale = stale_baseline_entries(declarations, VALID_DOC_IDS, baseline)
        assert stale == [("oom", "oom-kill-of-user-task")]

    def test_still_missing_baseline_pair_is_not_stale(self):
        declarations = [_symptom("kernel-oops", doc="kernel-oops")]
        baseline = frozenset({("kernel-oops", "kernel-oops")})
        stale = stale_baseline_entries(declarations, VALID_DOC_IDS, baseline)
        assert stale == []

    def test_baseline_pair_is_stale_when_declaration_disappears(self):
        baseline = frozenset({("retired-symptom", "retired-symptom")})
        assert stale_baseline_entries([], VALID_DOC_IDS, baseline) == [("retired-symptom", "retired-symptom")]

    @pytest.mark.parametrize("prefix", ["GPU A/B", "gpu a+b", "GPU A B", "GPU A+B "])
    def test_similar_replacement_does_not_keep_baseline_active(self, prefix):
        declarations = [_symptom(prefix, doc="gpu-error")]
        baseline = frozenset({("GPU A+B", "gpu-error")})

        assert stale_baseline_entries(declarations, set(), baseline) == [("GPU A+B", "gpu-error")]


def _name_index(*, aliases: list[str] | None = None) -> dict[str, frozenset[str]]:
    return build_doc_name_index(
        [
            {
                "id": "not-responding",
                "title": "Not responding",
                "aliases": aliases or [],
                "redirect_ids": [],
            }
        ]
    )


class TestDocNameIndexValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("aliases", None),
            ("aliases", "not-a-list"),
            ("aliases", [""]),
            ("aliases", [123]),
            ("redirect_ids", None),
            ("redirect_ids", "not-a-list"),
            ("redirect_ids", [""]),
            ("redirect_ids", [123]),
        ],
    )
    def test_rejects_malformed_manifest_collections(self, field: str, value: object):
        entry: dict[str, object] = {
            "id": "not-responding",
            "title": "Not responding",
            "aliases": [],
            "redirect_ids": [],
        }
        entry[field] = value

        with pytest.raises(ValueError):
            build_doc_name_index([entry])

    def test_omitted_manifest_collections_default_to_empty(self):
        index = build_doc_name_index([{"id": "not-responding", "title": "Not responding"}])

        assert index["not-responding"] == frozenset({"not-responding", "Not responding"})

    def test_rejects_duplicate_canonical_document_ids(self):
        entries = [
            {"id": "not-responding", "title": "Not responding"},
            {"id": "not-responding", "title": "Node not responding"},
        ]

        with pytest.raises(ValueError, match="duplicate documentation id"):
            build_doc_name_index(entries)

    def test_rejects_redirect_owned_by_multiple_documents(self):
        entries = [
            {"id": "first", "title": "First", "redirect_ids": ["legacy"]},
            {"id": "second", "title": "Second", "redirect_ids": ["legacy"]},
        ]

        with pytest.raises(ValueError, match="maps to both"):
            build_doc_name_index(entries)

    def test_rejects_redirect_that_shadows_another_canonical_id(self):
        entries = [
            {"id": "first", "title": "First", "redirect_ids": ["second"]},
            {"id": "second", "title": "Second"},
        ]

        with pytest.raises(ValueError, match="conflicts with a canonical documentation id"):
            build_doc_name_index(entries)


class TestSymptomNameDeclarations:
    @pytest.mark.parametrize("name", ["gpu-error", "GPU A+B", "GPU A/B", "legacy-gpu-error"])
    def test_exact_manifest_names_pass(self, name):
        index = build_doc_name_index(
            [{"id": "gpu-error", "title": "GPU A+B", "aliases": ["GPU A/B"], "redirect_ids": ["legacy-gpu-error"]}]
        )

        assert find_undeclared_symptom_names([_symptom(name, doc="legacy-gpu-error")], index) == []

    @pytest.mark.parametrize("field", ["id", "title", "aliases", "redirect_ids"])
    @pytest.mark.parametrize("name", ["GPU A/B", "gpu a+b", "GPU A B", "GPU A+B ", "GPU A+Bé"])
    def test_similar_names_require_explicit_declaration(self, field, name):
        entry: dict[str, object] = {"id": "gpu-error"}
        entry[field] = ["GPU A+B"] if field in ("aliases", "redirect_ids") else "GPU A+B"
        index = build_doc_name_index([entry])
        slug = "GPU A+B" if field in ("id", "redirect_ids") else "gpu-error"
        declarations = [_symptom(name, doc=slug)]

        assert find_undeclared_symptom_names(declarations, index) == declarations

    def test_slug_shaped_name_does_not_replace_an_explicit_alias(self):
        entry = {"id": "xid-48", "title": "XID 48"}
        declarations = [_symptom("XID-48-", doc="xid-48")]

        assert find_undeclared_symptom_names(declarations, build_doc_name_index([entry])) == declarations
        index = build_doc_name_index([{**entry, "aliases": ["XID-48-"]}])
        assert find_undeclared_symptom_names(declarations, index) == []

    def test_canonical_document_name_passes(self):
        assert find_undeclared_symptom_names([_symptom("Not responding", doc="not-responding")], _name_index()) == []

    def test_declared_alias_passes(self):
        declarations = [_symptom("Corroborated node failure", doc="not-responding")]
        assert (
            find_undeclared_symptom_names(
                declarations,
                _name_index(aliases=["Corroborated node failure"]),
            )
            == []
        )

    def test_existing_but_undeclared_document_is_flagged(self):
        declarations = [_symptom("Corroborated node failure", doc="not-responding")]
        assert find_undeclared_symptom_names(declarations, _name_index()) == declarations

    def test_dynamic_fallback_is_exempt(self):
        declarations = [_symptom("XID-", doc="not-responding", fallback_prefix=True)]
        assert find_undeclared_symptom_names(declarations, _name_index()) == []

    def test_redirect_inherits_target_names(self):
        index = build_doc_name_index(
            [
                {
                    "id": "xid-48",
                    "title": "XID 48",
                    "aliases": ["XID-48"],
                    "redirect_ids": ["xid-48-gpu-ecc-dbe"],
                }
            ]
        )
        assert (
            find_undeclared_symptom_names(
                [_symptom("XID-48", doc="xid-48-gpu-ecc-dbe")],
                index,
            )
            == []
        )
