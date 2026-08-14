# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Branch-complete coverage for declarative rules (leaves AND combinators).

Every emit element carries a set of obligations derived from its structure; each
obligation is a unique path-prefixed STRING so nested nodes never collide:

- a **leaf** with a ``when`` must be exercised on both sides
  (``{"when:true", "when:false"}``); a leaf without a ``when`` needs at least one
  case (``{"emit"}``).
- a **pick_one** requires each branch to be *selected* (``"branch[i]:selected"``)
  and the default to be *reached* (``"default:reached"``); nested obligations of a
  combinator branch and of a dict default recurse under that child's path.
- a **try_all** unions each event's obligations (an event's own ``when`` yields its
  event's true/false pair). A combinator-level ``when`` adds this node's
  ``"when:true"/"when:false"`` pair.

Paths use ``.`` between structural segments (``event[1].branch[0]``) and ``:`` before
the obligation kind (``event[1].branch[0]:selected``). The root path is empty, so a
top-level leaf keeps the bare obligations (``"emit"``, ``"when:true"``, ...).

``check_coverage`` fails the load (``LoadError``) unless every obligation is met by
the co-located ``[[extractor.cases]]`` or the element carries an ``untested`` waiver.
Cases and ``untested`` are mutually exclusive.
"""

import re
from collections.abc import Callable
from typing import Any

from attribution_lib.declarative.emit import LoadError
from attribution_lib.declarative.expressions import _FIXED_FUNCS, eval_bool, eval_expr
from attribution_lib.declarative.families import GeneratedRule

__all__ = [
    "leaf_obligations",
    "element_obligations",
    "covered_obligations",
    "check_coverage",
    "catalog_obligations",
    "check_catalog_coverage",
    "family_obligations",
    "check_family_coverage",
]

# `expect` values that assert the leaf does NOT emit (the ``when:false`` side).
_NO_EMIT_EXPECTS = {"none", "no_match", "nothing"}


def _overlay_lifecycle(
    functions: dict[str, Any] | None,
    lifecycle: set[str] | None,
    case: dict[str, Any],
) -> dict[str, Any] | None:
    """Per-case ``functions`` map: the base plus each lifecycle predicate bound to
    the case's declared boolean (default False). Returns ``functions`` unchanged
    when there are no lifecycle predicates."""
    if not lifecycle:
        return functions
    base = functions if functions is not None else _FIXED_FUNCS
    return {**base, **{p: (lambda v=bool(case.get(p, False)): v) for p in lifecycle}}


def _child(path: str, seg: str) -> str:
    """Extend a structural path by one segment (``.``-separated; root is empty).

    Example:
        >>> _child("", "branch[0]")
        'branch[0]'
        >>> _child("event[1]", "branch[0]")
        'event[1].branch[0]'
    """
    return seg if not path else f"{path}.{seg}"


def _obligation(path: str, kind: str) -> str:
    """Name an obligation ``kind`` at ``path`` (``:``-separated; bare at the root).

    Example:
        >>> _obligation("", "emit")
        'emit'
        >>> _obligation("branch[0]", "selected")
        'branch[0]:selected'
    """
    return kind if not path else f"{path}:{kind}"


def leaf_obligations(emit: dict[str, Any]) -> set[str]:
    """Obligations a leaf must cover: both ``when`` sides, or a single ``emit``.

    Example:
        >>> leaf_obligations({"type": "bad_node", "symptom": "..."})
        {'emit'}
        >>> leaf_obligations({"type": "bad_node", "when": "x > 0", "symptom": "..."})
        {'when:true', 'when:false'}
    """
    return {"when:true", "when:false"} if emit.get("when") is not None else {"emit"}


def element_obligations(emit: dict[str, Any], path: str = "") -> set[str]:
    """All obligations for ``emit`` (a leaf or a combinator), path-prefixed.

    Recurses the emit tree. A combinator branch's own ``when`` is its *selection*
    guard, subsumed by ``"branch[i]:selected"``; only its deeper structure adds
    obligations (see :func:`_branch_child_obligations`). A ``try_all`` event, by
    contrast, carries a real ``when`` whose true/false sides are both required.

    Example:
        >>> element_obligations({"type": "bad_node", "symptom": "..."})
        {'emit'}
        >>> sorted(element_obligations({"type": "bad_node", "when": "x", "symptom": "..."}, path="event[0]"))
        ['event[0]:when:false', 'event[0]:when:true']
    """
    match emit.get("type"):
        case "pick_one":
            obs: set[str] = set()
            if emit.get("when") is not None:
                obs |= {_obligation(path, "when:true"), _obligation(path, "when:false")}
            for i, br in enumerate(emit.get("branches", [])):
                bpath = _child(path, f"branch[{i}]")
                obs.add(f"{bpath}:selected")
                obs |= _branch_child_obligations(br, bpath)
            dpath = _child(path, "default")
            obs.add(f"{dpath}:reached")
            default = emit.get("default")
            if isinstance(default, dict):
                obs |= element_obligations(default, dpath)
            return obs
        case "try_all":
            obs = set()
            if emit.get("when") is not None:
                obs |= {_obligation(path, "when:true"), _obligation(path, "when:false")}
            for i, ev in enumerate(emit.get("events", [])):
                obs |= element_obligations(ev, _child(path, f"event[{i}]"))
            return obs
        case _:
            return {_obligation(path, o) for o in leaf_obligations(emit)}


def _branch_child_obligations(br: dict[str, Any], bpath: str) -> set[str]:
    """Obligations INSIDE a pick_one branch, beyond its ``:selected`` marker.

    A leaf branch adds nothing (its ``when`` is the selection guard, already
    covered by selection; once selected it emits unconditionally). A combinator
    branch contributes its inner branch/event obligations, but NOT its own
    top-level ``when`` pair, which is likewise the selection guard.
    """
    match br.get("type"):
        case "pick_one" | "try_all":
            inner = element_obligations(br, bpath)
            return inner - {f"{bpath}:when:true", f"{bpath}:when:false"}
        case _:
            return set()


def _hit(
    emit: dict[str, Any],
    case: dict[str, Any],
    groups: dict[str, Any] | None,
    name: str = "?",
    *,
    cross_check_expect: bool = True,
    functions: dict[str, Any] | None = None,
) -> str:
    """The single obligation ``case`` covers for ``emit``.

    ``groups`` is the case line's captures (from the rule's ``match_re``) or ``None``
    when no regex is available (the unit-test path).

    When captures ARE available the covered side is MACHINE-VERIFIED: ``when`` is
    evaluated against the case's captures rather than trusting the author's
    ``expect`` label. If the case also carries an ``expect``, it is cross-checked
    against the evaluated side and a contradiction (mislabeled case) fails the load.

    ``cross_check_expect`` is ``False`` when this leaf is one of several independent
    ``try_all`` events: a whole-rule ``expect`` describes the emitted SET, not any one
    sibling event, so cross-checking it against a single event's ``when`` is a category
    error (a case that makes one guard true necessarily makes an independent guard
    false). The covered side is still machine-verified from the captures; only the
    ``expect`` contradiction check is skipped.

    A line that did not match the rule's ``match_re`` (empty ``groups``) cannot emit,
    so it covers the ``when:false`` side.
    """
    when = emit.get("when")
    if when is None:
        return "emit"

    if groups is None:
        # Unit-test path: no regex to evaluate against, so fall back to the author's
        # `expect` label. An emitting `expect` covers the true side.
        return "when:false" if case.get("expect") in _NO_EMIT_EXPECTS else "when:true"

    # Loader path: captures are available, so verify the covered side directly.
    # A non-matching line cannot emit. Evaluation errors for a matching line
    # propagate because a bad name or operation means the rule is malformed.
    side_true = False if not groups else eval_bool(when, groups, functions)

    expect = case.get("expect")
    if expect is not None and cross_check_expect:
        no_emit = expect in _NO_EMIT_EXPECTS
        if no_emit and side_true:
            raise LoadError(
                f"{name}: case {case.get('line')!r} is labeled expect={expect!r} (no emit) "
                f"but its `when` evaluates TRUE over the captured line; the case is mislabeled"
            )
        if not no_emit and not side_true:
            raise LoadError(
                f"{name}: case {case.get('line')!r} is labeled expect={expect!r} (emit) "
                f"but its `when` evaluates FALSE over the captured line; the case is mislabeled"
            )
    return "when:true" if side_true else "when:false"


def _cover(
    emit: dict[str, Any],
    path: str,
    case: dict[str, Any],
    groups: dict[str, Any] | None,
    name: str,
    hits: set[str],
    *,
    cross_check_expect: bool = True,
    functions: dict[str, Any] | None = None,
) -> None:
    """Record which obligations of ``emit`` the single ``case`` exercises.

    Walks the same control flow the compiled processor takes over ``groups``:

    - **pick_one**: an outer ``when`` false records ``when:false`` and stops (nothing
      emitted). Otherwise branch ``when``s are evaluated in order; the first true
      records ``branch[i]:selected`` and recurses into a combinator branch's inner
      structure; if none match, ``default:reached`` is recorded and a dict default
      recursed into.
    - **try_all**: an outer ``when`` false records ``when:false`` and stops; otherwise
      every event is recursed into with ``cross_check_expect=False`` (a whole-rule
      ``expect`` cannot describe any single one of the independent sibling emissions).
    - **leaf**: delegates to :func:`_hit` (covered-side derivation + ``expect`` cross-check,
      the latter suppressed under a ``try_all``).

    ``cross_check_expect`` propagates into nested subtrees: once inside a ``try_all``
    event it stays ``False`` all the way down, since the whole-rule ``expect`` is just
    as meaningless against a deeply nested guard as against a top-level sibling.

    A non-matching line (empty ``groups``) means the rule did not fire, so a
    combinator records nothing; a leaf's :func:`_hit` still maps it to ``when:false``.
    """
    match emit.get("type"):
        case "pick_one" | "try_all" as etype:
            if groups is not None and not groups:
                # The line did not match the rule at all; the combinator emitted nothing.
                return
            if groups is None:
                raise LoadError(f"{name}: combinator coverage requires captured groups; supply match_groups")
            outer = emit.get("when")
            if outer is not None:
                side = eval_bool(outer, groups, functions)
                hits.add(_obligation(path, "when:true" if side else "when:false"))
                if not side:
                    return
            match etype:
                case "pick_one":
                    branches = emit.get("branches")
                    if not branches:
                        raise LoadError(f"{name}: pick_one requires a non-empty 'branches' list")
                    if "default" not in emit:
                        raise LoadError(f"{name}: pick_one requires a 'default' (an element, or \"none\")")
                    for i, br in enumerate(branches):
                        when = br.get("when")
                        if not when:
                            raise LoadError(f"{name}: pick_one branch {i} requires a 'when'")
                        if eval_bool(when, groups, functions):
                            bpath = _child(path, f"branch[{i}]")
                            hits.add(f"{bpath}:selected")
                            match br.get("type"):
                                case "pick_one" | "try_all":
                                    _cover(
                                        br,
                                        bpath,
                                        case,
                                        groups,
                                        name,
                                        hits,
                                        cross_check_expect=cross_check_expect,
                                        functions=functions,
                                    )
                                case _:
                                    # A selected leaf branch emits unconditionally, so it adds no
                                    # obligation of its own; but still cross-check its `expect` label
                                    # so a mislabeled leaf-branch case is caught. Discard the result.
                                    _hit(
                                        br,
                                        case,
                                        groups,
                                        name,
                                        cross_check_expect=cross_check_expect,
                                        functions=functions,
                                    )
                            return
                    dpath = _child(path, "default")
                    hits.add(f"{dpath}:reached")
                    default = emit["default"]
                    if isinstance(default, dict):
                        _cover(
                            default,
                            dpath,
                            case,
                            groups,
                            name,
                            hits,
                            cross_check_expect=cross_check_expect,
                            functions=functions,
                        )
                case "try_all":
                    # each event is an independent emission, so a whole-rule `expect`
                    # cannot be cross-checked against any single event's `when`.
                    events = emit.get("events")
                    if not events:
                        raise LoadError(f"{name}: try_all requires a non-empty 'events' list")
                    for i, ev in enumerate(events):
                        _cover(
                            ev,
                            _child(path, f"event[{i}]"),
                            case,
                            groups,
                            name,
                            hits,
                            cross_check_expect=False,
                            functions=functions,
                        )
        case _:
            hits.add(
                _obligation(
                    path, _hit(emit, case, groups, name, cross_check_expect=cross_check_expect, functions=functions)
                )
            )


def covered_obligations(
    emit: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    match_groups: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    name: str = "?",
    functions: dict[str, Any] | None = None,
    lifecycle: set[str] | None = None,
) -> set[str]:
    """Obligations covered by ``cases``.

    ``match_groups(case) -> dict`` supplies each case line's captures (from the
    rule's ``match_re``); when supplied, each covered side is machine-verified by
    evaluating ``when`` and any ``expect`` label is cross-checked. When omitted,
    coverage is inferred from each case's ``expect`` alone (leaf-only unit path).

    ``lifecycle`` (a set of predicate names) binds each named lifecycle predicate
    per case to the case's declared boolean, so a ``when`` conjunct like
    ``at_end_of_job()`` is evaluated rather than raising.
    """
    hits: set[str] = set()
    for case in cases:
        groups = match_groups(case) if match_groups is not None else None
        case_fns = _overlay_lifecycle(functions, lifecycle, case)
        _cover(emit, "", case, groups, name, hits, functions=case_fns)
    return hits


# --- catalog_lookup: row-complete coverage --------------------------------------


def catalog_obligations(table: dict[str, Any]) -> set[str]:
    """Row-complete obligations for a ``catalog_lookup``: one ``row:<key>`` per
    table row PLUS ``not_found`` (the miss path). ``catalog_lookup`` is root-only,
    so obligations are bare (no structural path prefix).

    Example:
        >>> sorted(catalog_obligations({"A": {}, "B": {}}))
        ['not_found', 'row:A', 'row:B']
    """
    obs = {f"row:{key}" for key in table}
    obs.add("not_found")
    return obs


def expand_catalog_cases(cases: list[dict[str, Any]], table: dict[str, Any], name: str = "?") -> list[dict[str, Any]]:
    """Expand any ``expansion = "catalog"`` case into one hit case per table row.

    The case's ``line_template`` has ``{key}`` replaced by each row key, yielding a
    concrete line whose ``key`` expression resolves to that row -- so one stanza
    covers every ``row:<key>``. Non-expansion cases pass through unchanged.

    Example:
        >>> expand_catalog_cases([{"line": "hit"}], {"A": {}})
        [{'line': 'hit'}]
        >>> rows = expand_catalog_cases(
        ...     [{"expansion": "catalog", "line_template": "key={key}"}],
        ...     {"X": {}, "Y": {}},
        ... )
        >>> [r["line"] for r in rows]
        ['key=X', 'key=Y']
    """
    out: list[dict[str, Any]] = []
    for case in cases:
        if case.get("expansion") == "catalog":
            tmpl = case.get("line_template")
            if tmpl is None:
                raise LoadError(f"{name}: a catalog expansion case requires a 'line_template'")
            out.extend({**case, "line": tmpl.replace("{key}", key)} for key in table)
        else:
            out.append(case)
    return out


def covered_catalog_obligations(
    emit: dict[str, Any],
    table: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    match_groups: Callable[[dict[str, Any]], dict[str, Any]],
    name: str = "?",
    functions: dict[str, Any] | None = None,
    lifecycle: set[str] | None = None,
) -> set[str]:
    """Obligations the ``cases`` cover for a ``catalog_lookup``.

    For each case the ``key`` expression is evaluated over the case line's captures
    and ``str()``-coerced: a key present in ``table`` covers ``row:<key>``, else
    ``not_found``. The lookup's ``when`` is honored -- when it is false the lookup
    emits nothing, so the case covers neither a row nor ``not_found`` (a no-op for
    row coverage), matching the compiled closure's control flow.
    """
    when = emit.get("when")
    key_expr = emit["key"]
    hits: set[str] = set()
    for case in cases:
        groups = match_groups(case)
        if len(groups) == 0:
            continue
        case_fns = _overlay_lifecycle(functions, lifecycle, case)
        if when is not None and not eval_bool(when, groups, case_fns):
            continue
        key = str(eval_expr(key_expr, groups, case_fns))
        hits.add(f"row:{key}" if key in table else "not_found")
    return hits


def check_catalog_coverage(
    name: str,
    emit: dict[str, Any],
    table: dict[str, Any],
    cases: list[dict[str, Any]],
    untested: str | None,
    untested_rows: list[str] | None,
    *,
    match_groups: Callable[[dict[str, Any]], dict[str, Any]],
    functions: dict[str, Any] | None = None,
    lifecycle: set[str] | None = None,
) -> None:
    """Raise ``LoadError`` unless every catalog-row obligation is covered or waived.

    - ``untested`` waives the whole rule but is mutually exclusive with ``cases``.
    - ``untested_rows`` waives named ``row:<key>`` obligations (co-exists with cases).
    - a rule with neither cases nor ``untested`` fails the load.
    - otherwise every ``row:<key>`` (minus ``untested_rows``) plus ``not_found``
      must be covered by the (expanded) cases.
    """
    if untested is not None:
        if cases:
            raise LoadError(f"{name}: has both cases and `untested`; they are mutually exclusive")
        return
    need = catalog_obligations(table)
    if untested_rows is not None:
        need -= {f"row:{key}" for key in untested_rows}
    cases = expand_catalog_cases(cases, table, name)
    if not cases:
        raise LoadError(f"{name}: no cases and no `untested` waiver")
    have = covered_catalog_obligations(
        emit, table, cases, match_groups=match_groups, name=name, functions=functions, lifecycle=lifecycle
    )
    missing = need - have
    if missing:
        raise LoadError(f"{name}: uncovered obligations {sorted(missing)}")


# --- families: row-complete coverage --------------------------------------------


def family_obligations(generated: list[GeneratedRule]) -> set[str]:
    """Row-complete obligations for a ``[[family]]``: one ``rule:<gen-name>`` per
    generated rule (per row, per pattern). A family is a flat set of generated
    extractors, so obligations are bare (no structural path prefix)."""
    return {f"rule:{g.name}" for g in generated}


def expand_family_cases(
    cases: list[dict[str, Any]], generated: list[GeneratedRule], name: str = "?"
) -> list[dict[str, Any]]:
    """Expand any ``expansion = "rows"`` case into one hit case per generated rule.

    Each generated rule's synthesized line is the literal text its ``match_re``
    matches (the substituted pattern before ``re.escape``), so one stanza covers
    every ``rule:<gen-name>``. Non-expansion cases pass through unchanged.
    """
    out: list[dict[str, Any]] = []
    for case in cases:
        if case.get("expansion") == "rows":
            out.extend({**case, "line": g.line} for g in generated)
        else:
            out.append(case)
    return out


def covered_family_obligations(
    generated: list[GeneratedRule], cases: list[dict[str, Any]], name: str = "?"
) -> set[str]:
    """Obligations the ``cases`` cover for a family.

    A case whose ``line`` matches a generated rule's (literal, ``re.escape``-d)
    ``match_re`` covers ``rule:<that gen-name>``. A single line may cover several
    generated rules when one literal is a substring of another (harmless overlap).
    """
    # ``expansion = "rows"`` creates one case from every generated rule's own
    # synthesized matching line. Each rule is therefore covered by
    # construction; comparing every generated case with every generated rule
    # would only repeat that proof with O(n^2) regex searches.
    if any(case.get("expansion") == "rows" for case in cases):
        return family_obligations(generated)

    hits: set[str] = set()
    for case in cases:
        line = case["line"]
        for g in generated:
            if re.search(g.match_re, line):
                hits.add(f"rule:{g.name}")
    return hits


def check_family_coverage(
    name: str,
    generated: list[GeneratedRule],
    cases: list[dict[str, Any]],
    untested: str | None,
    untested_rows: list[str] | None,
    *,
    functions: dict[str, Any] | None = None,
) -> None:
    """Raise ``LoadError`` unless every generated rule is covered or waived.

    - ``untested`` waives the whole family but is mutually exclusive with ``cases``.
    - ``untested_rows`` waives every generated rule stamped from a named row
      (co-exists with cases).
    - a family with neither cases nor ``untested`` fails the load.
    - otherwise every ``rule:<gen-name>`` (minus ``untested_rows``) must be covered
      by the (expanded) cases.
    """
    if untested is not None:
        if cases:
            raise LoadError(f"{name}: has both cases and `untested`; they are mutually exclusive")
        return
    need = family_obligations(generated)
    if untested_rows is not None:
        waived = set(untested_rows)
        need -= {f"rule:{g.name}" for g in generated if g.row_key in waived}
    cases = expand_family_cases(cases, generated, name)
    if not cases:
        raise LoadError(f"{name}: no cases and no `untested` waiver")
    have = covered_family_obligations(generated, cases, name)
    missing = need - have
    if missing:
        raise LoadError(f"{name}: uncovered generated rules {sorted(missing)}")


def check_coverage(
    name: str,
    emit: dict[str, Any],
    cases: list[dict[str, Any]],
    untested: str | None,
    *,
    match_groups: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    functions: dict[str, Any] | None = None,
    lifecycle: set[str] | None = None,
) -> None:
    """Raise ``LoadError`` unless every obligation is covered or waived.

    - ``untested`` waives coverage but is mutually exclusive with ``cases``.
    - an element with neither cases nor ``untested`` fails the load.
    - otherwise every obligation in :func:`element_obligations` must be covered.
    """
    if untested is not None:
        if cases:
            raise LoadError(f"{name}: has both cases and `untested`; they are mutually exclusive")
        return
    if not cases:
        raise LoadError(f"{name}: no cases and no `untested` waiver")
    need = element_obligations(emit)
    have = covered_obligations(
        emit, cases, match_groups=match_groups, name=name, functions=functions, lifecycle=lifecycle
    )
    missing = need - have
    if missing:
        raise LoadError(f"{name}: uncovered obligations {sorted(missing)}")
