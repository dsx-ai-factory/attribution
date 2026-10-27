# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import os
import re
import tomllib
import types
from collections.abc import Callable, Collection, Iterator
from typing import TYPE_CHECKING, Any, NoReturn, cast

from attribution_lib.declarative.catalogs import Tables, parse_tables, resolve_catalog_table
from attribution_lib.declarative.coverage import (
    check_catalog_coverage,
    check_coverage,
    check_family_coverage,
)
from attribution_lib.declarative.emit import (
    INJECTED_NAMES,
    compile_catalog_lookup,
    compile_element,
    compile_processor,
)
from attribution_lib.declarative.errors import FinalizeError, LoadError, RegistrationError, UnknownKeyError
from attribution_lib.declarative.expressions import (
    _FIXED_FUNCS,
    _free_names_in_template,
    called_names_in,
    check_consts_resolvable,
    check_guarded,
    check_helpers_resolvable,
    check_template_syntax,
    guards_established_by,
    merge_functions,
    optional_captures,
    render_template,
)
from attribution_lib.declarative.families import stamp_family
from attribution_lib.declarative.lifecycle import LIFECYCLE_INJECTIONS, LIFECYCLE_PREDICATES
from attribution_lib.exceptions import RuleIngestionError
from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.prefilters import PreFilterSpec
from attribution_lib.rule_types.definitions import ExtractorRule, Rule, RuleId, RuleLocation
from attribution_lib.rule_types.registration import normalize_tags

if TYPE_CHECKING:
    from attribution_lib.rules_engine import RuleLoader


# ---------------------------------------------------------------------------
# Load context and pending-stanza carriers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _LoadContext:
    """Immutable build context derived from the file header and loader state.

    Produced once by :func:`_build_load_context` and shared across all per-stanza
    finalize calls so each stanza sees the same helper/const/table snapshot.
    """

    functions: dict[str, Any]
    injecting_registry: dict[str, tuple[Callable[..., Any], frozenset[str]]]
    allowed_helpers: set[str]
    lifecycle_names: set[str]
    static_names: set[str]
    const_ns: types.SimpleNamespace
    allowed_consts: set[str]
    non_injecting: dict[str, Callable[..., Any]]
    tables: Tables
    file_sources: list[str]
    file_author: str | None
    file_tags: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _PendingProcessor:
    """A recorded processor stanza awaiting finalization."""

    index: int
    stanza: dict[str, Any]
    name: str


@dataclasses.dataclass(frozen=True)
class _PendingEmit:
    """A recorded emit stanza awaiting finalization."""

    index: int
    stanza: dict[str, Any]
    name: str


# ---------------------------------------------------------------------------
# Helper / validation functions (used by both phases)
# ---------------------------------------------------------------------------

_FILE_KEYS = frozenset({"sources", "author", "tags", "extractor", "family", "catalog", "rows", "const", "prefilter"})

# TODO: `description` is grammar-legal on both `[[extractor]]` and `[[family]]` (see
# docs/declarative-grammar.md) but not yet supported, so both sets below whitelist it and never read it.
_EXTRACTOR_KEYS = frozenset(
    {"name", "match_re", "sources", "author", "description", "tags", "emit", "cases", "untested", "untested_rows"}
)
_FAMILY_KEYS = frozenset(
    {
        "name",
        "sources",
        "author",
        "description",
        "tags",
        "rows",
        "rows_from",
        "pattern",
        "patterns",
        "emit",
        "cases",
        "untested",
        "untested_rows",
    }
)

_PROCESSOR_KEYS = frozenset({"processor", "derived_rules", "derived_event_types"})

_PREFILTER_KEYS = frozenset({"sources", "include", "exclude"})


def _reject_unknown_keys(declared: dict[str, Any], known: Collection[str], *, where: str) -> None:
    """Raise on the first declared key the grammar does not define at this level."""
    for key in declared:
        if key not in known:
            raise UnknownKeyError(key, known, where)


def _check_declared_keys(doc: dict[str, Any], filepath: str) -> None:
    """Reject grammar-unknown keys at file, ``[[extractor]]``, ``[[family]]``, and ``[prefilter]`` level."""
    _reject_unknown_keys(doc, _FILE_KEYS, where=filepath)
    for section, section_keys in (
        ("extractor", _EXTRACTOR_KEYS),
        ("family", _FAMILY_KEYS),
    ):
        for i, stanza in enumerate(doc.get(section, [])):
            if not isinstance(stanza, dict):
                continue
            known = (section_keys | _PROCESSOR_KEYS) if "processor" in stanza else section_keys
            label = repr(stanza["name"]) if "name" in stanza else f"#{i}"
            _reject_unknown_keys(stanza, known, where=f"{filepath}: [[{section}]] {label}")

    prefilter = doc.get("prefilter")
    if isinstance(prefilter, dict):
        _reject_unknown_keys(prefilter, _PREFILTER_KEYS, where=f"{filepath}: [prefilter]")


def _resolve_sources(
    names: list[str],
    *,
    where: str,
) -> frozenset[ObsSource]:
    """Resolve ObservationSource member NAMES (spec: the ``sources`` key)."""
    out = set()
    for n in names:
        try:
            out.add(ObsSource[n])
        except KeyError as e:
            raise LoadError(f"{where}: unknown source {n!r}; use an ObservationSource member name") from e
    return frozenset(out)


def _resolve_tags(
    declared: Any,
    *,
    ctx: _LoadContext,
    where: str,
    filepath: str,
) -> tuple[str, ...]:
    """Combine a stanza's own ``tags`` with the file-level defaults.

    Unlike ``sources`` and ``author``, which a stanza *replaces*, tags **union** with
    the file-level value. A stanza cannot drop a tag its file declared.

    Args:
        declared: The stanza's raw ``tags`` value, exactly as written in the rule file.
        ctx: Load context supplying the file-level tags to merge in.
        where: Rule or family name, for error messages.
        filepath: Rule file being loaded, prefixed to the error so the failure is locatable.

    Returns:
        The finalized, normalized tags as a tuple of strings.

    Raises:
        FinalizeError: If the tags are invalid.
    """
    try:
        # Validate before merging: splatting an unvalidated `tags = "external"` into
        # the merge list would silently yield one tag per character.
        own = normalize_tags(declared, where=where)
        merged_where = f"{where} (file-level tags: {list(ctx.file_tags)})" if len(ctx.file_tags) > 0 else where
        return normalize_tags([*ctx.file_tags, *own], where=merged_where)
    except RuleIngestionError as exc:
        raise FinalizeError(f"{filepath}: {exc}") from exc


def _collect_declarations(
    emit: dict[str, Any],
    rules: set[str],
    event_types: set[str],
) -> None:
    """Walk an emit tree collecting the secondary names it credits/emits.

    Mirrors the Python decorators' ``derived_rules``/``derived_event_types``: a
    leaf ``derived_rule`` credits a finding under a name other than the stanza's,
    a ``demotion`` dict/string names the rule(s) demoted events get attributed
    to (see ``_bad``/``_interesting`` and ``_sql_demote_events``), and a
    ``generic_event`` emits an ``event_type``; all must be declared so the
    event store's rule/type lookups resolve them. Recurses combinator arms
    (``pick_one`` branches + dict ``default``; ``try_all`` events).

    Example:
        >>> rules, types_ = set(), set()
        >>> _collect_declarations({"type": "generic_event", "event_type": "gpu_reset"}, rules, types_)
        >>> types_
        {'gpu_reset'}
    """
    if (derived := emit.get("derived_rule")) is not None:
        rules.add(derived)
    demotion = emit.get("demotion")
    if isinstance(demotion, dict):
        rules.update(demotion.values())
    elif isinstance(demotion, str):
        rules.add(demotion)
    if emit.get("type") == "generic_event" and emit.get("event_type") is not None:
        event_types.add(emit["event_type"])
    for br in emit.get("branches", []):
        _collect_declarations(br, rules, event_types)
    for ev in emit.get("events", []):
        _collect_declarations(ev, rules, event_types)
    if isinstance(default := emit.get("default"), dict):
        _collect_declarations(default, rules, event_types)


def _collect_catalog_declarations(
    emit: dict[str, Any],
    table: dict[str, Any],
    is_element: bool,
    rules: set[str],
    event_types: set[str],
) -> None:
    """Declare the credited rule NAMES a ``catalog_lookup`` can emit on a hit.

    Unlike ``_collect_declarations`` (which would add the literal ``derived_rule``
    TEMPLATE, e.g. ``"X/{key}"``), a catalog_lookup credits a per-row RENDERED name:
    the matched row's own ``derived_rule`` if set (element catalogs only), else the
    lookup's ``derived_rule`` template rendered with that row's key. Both must be
    declared or ``process_check`` raises ``UndeclaredRuleError`` at match time. Also
    collects any element-row/project ``event_type`` and the ``not_found`` leaf's
    ``derived_rule``/``event_type``.

    Example:
        >>> rules, types_ = set(), set()
        >>> _collect_catalog_declarations(
        ...     {"derived_rule": "{key}_credit"},
        ...     {"A": {}, "B": {}},
        ...     False,
        ...     rules,
        ...     types_,
        ... )
        >>> sorted(rules)
        ['A_credit', 'B_credit']
    """
    lookup_derived = emit.get("derived_rule")
    for key, row in table.items():
        row_derived = row.get("derived_rule") if is_element else None
        credit = row_derived or (render_template(lookup_derived, {"key": key}) if lookup_derived else None)
        if credit is not None:
            rules.add(credit)
        if is_element and row.get("type") == "generic_event" and row.get("event_type") is not None:
            event_types.add(row["event_type"])
    project = emit.get("project")
    if isinstance(project, dict) and project.get("type") == "generic_event" and project.get("event_type") is not None:
        event_types.add(project["event_type"])
    if isinstance(project, dict) and project.get("derived_rule"):
        rules.add(project["derived_rule"])
    if isinstance(nf := emit.get("not_found"), dict):
        _collect_declarations(nf, rules, event_types)


def _catalog_post_when_optional(emit: dict[str, Any], optional: set[str]) -> set[str]:
    """Return the optional-capture set that dominates expressions evaluated after ``when``.

    ``when`` is checked first; any capture it proves non-``None`` is removed from the
    set that the ``key``, ``project``, ``not_found``, and element-row expressions must
    guard against.
    """
    if when := emit.get("when"):
        check_guarded(when, optional, is_template=False)
        return optional - guards_established_by(when)
    return optional


def _check_catalog_guards(
    emit: dict[str, Any],
    optional: set[str],
    *,
    where: str,
) -> None:
    """Reject un-guard-dominated uses of possibly-absent captures in a catalog_lookup.

    The ``when`` gate is evaluated first (its own uses must be self-guarded); a
    non-``None`` capture it proves then dominates the ``key`` expression and the
    ``project``/``not_found`` leaf templates (all evaluated after ``when``).
    """
    if not optional:
        return
    templates_optional = _catalog_post_when_optional(emit, optional)
    if (key_expr := emit.get("key")) is None:
        raise LoadError(f"{where}: catalog_lookup is missing the required 'key' field")
    check_guarded(key_expr, templates_optional, is_template=False)
    if isinstance(project := emit.get("project"), dict):
        _check_emit_guards(project, templates_optional, where=f"{where}.project")
    if isinstance(nf := emit.get("not_found"), dict):
        _check_emit_guards(nf, templates_optional, where=f"{where}.not_found")


def _check_emit_guards(
    emit: dict[str, Any],
    optional: set[str],
    *,
    where: str,
) -> None:
    """Reject un-guard-dominated uses of possibly-absent captures in an emit element.

    Recurses into combinators (``pick_one`` branches + ``default``; ``try_all``
    events) using the SAME ``optional`` set, since the captures all come from the
    stanza's one ``match_re``. For a leaf: ``when`` is checked against the full
    optional set (it is evaluated first, so its own uses must be self-guarded). A
    leaf's own ``when`` then dominates its symptom/payload templates: any capture it
    proves non-``None`` is excluded from the optional set for those templates (spec:
    "a leaf's own `when` dominates its templates").
    """
    if not optional:
        return

    if when := emit.get("when"):
        check_guarded(when, optional, is_template=False)
    etype = emit.get("type")
    match etype:
        case "pick_one":
            for i, br in enumerate(emit.get("branches", [])):
                _check_emit_guards(br, optional, where=f"{where}.branch[{i}]")
            if isinstance(default := emit.get("default"), dict):
                _check_emit_guards(default, optional, where=f"{where}.default")
        case "try_all":
            for i, ev in enumerate(emit.get("events", [])):
                _check_emit_guards(ev, optional, where=f"{where}.event[{i}]")
        case _:
            templates_optional = optional - guards_established_by(when) if when else optional
            if etype in ("bad_node", "interesting_node") and "symptom" in emit:
                check_guarded(emit["symptom"], templates_optional)
            if etype == "generic_event":
                for key, value in emit.get("payload", {}).items():
                    check_guarded(key, templates_optional)
                    check_guarded(value, templates_optional)


def _iter_emit_expressions(emit: dict[str, Any]) -> Iterator[tuple[str, bool]]:
    """Yield each expression in an emit tree as ``(text, is_template)``.

    Helper validation, const validation, and injection discovery all need the
    same traversal. Guard and name checks remain separate because their allowed
    names depend on control flow and catalog-row scope.

    Example:
        >>> list(_iter_emit_expressions({"type": "bad_node", "when": "x > 0", "symptom": "{x}"}))
        [('x > 0', False), ('{x}', True)]
    """
    if when := emit.get("when"):
        yield when, False

    match emit.get("type"):
        case "bad_node" | "interesting_node":
            if symptom := emit.get("symptom"):
                yield symptom, True
        case "generic_event":
            for key, value in emit.get("payload", {}).items():
                yield key, True
                yield value, True
        case "catalog_lookup":
            if (key := emit.get("key")) is not None:
                yield key, False

    for child in _iter_emit_children(emit):
        yield from _iter_emit_expressions(child)


def _iter_emit_children(emit: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the direct child elements of a combinator or catalog lookup.

    Example:
        >>> events = [{"type": "bad_node"}, {"type": "interesting_node"}]
        >>> list(_iter_emit_children({"type": "try_all", "events": events})) == events
        True
    """
    match emit.get("type"):
        case "pick_one":
            yield from emit.get("branches", [])
            if isinstance(default := emit.get("default"), dict):
                yield default
        case "try_all":
            yield from emit.get("events", [])
        case "catalog_lookup":
            for field in ("project", "not_found"):
                if isinstance(child := emit.get(field), dict):
                    yield child


def _check_emit_helpers(emit: dict[str, Any], allowed: set[str]) -> None:
    """Reject a call to an unregistered helper anywhere in an emit tree."""
    for text, is_template in _iter_emit_expressions(emit):
        check_helpers_resolvable(text, allowed, is_template=is_template)


def _check_emit_consts(emit: dict[str, Any], allowed: set[str]) -> None:
    """Reject an undefined ``const.<name>`` anywhere in an emit tree."""
    for text, is_template in _iter_emit_expressions(emit):
        check_consts_resolvable(text, allowed, is_template=is_template)


def _reject_undeclared_expr_names(
    expr: str,
    allowed: set[str],
    *,
    field: str,
    where: str,
) -> None:
    """Raise :class:`LoadError` if a bare expression (e.g. a ``when`` guard) references
    a free name not in ``allowed``.

    Unlike :func:`_reject_undeclared_names`, this accepts a bare expression rather than
    a template string, so it skips the brace-syntax check (which would incorrectly flag
    set literals or other inner braces).

    Example:
        >>> _reject_undeclared_expr_names("count > 0", {"count"}, field="when", where="test")
        >>> _reject_undeclared_expr_names("typo > 0", {"count"}, field="when", where="test")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        LoadError: ...
    """
    for nm in _free_names_in_template(f"{{{expr}}}"):
        if nm not in allowed:
            raise LoadError(
                f"{where}: {field!r} expression references undeclared name {nm!r} "
                f"(not a capture, datum, const, helper, or lifecycle predicate)"
            )


def _reject_undeclared_names(
    template: str,
    allowed: set[str],
    *,
    field: str,
    where: str,
) -> None:
    """Raise :class:`LoadError` if ``template`` has stray braces or references a free
    name not in ``allowed``.

    ``allowed`` is the union of match captures, ``datum``/``node``/``const``,
    registered helpers, fixed functions, and lifecycle predicates. A name outside
    that set is a typo that would raise ``NameNotDefined`` at match time.

    Example:
        >>> _reject_undeclared_names("{count}", {"count", "datum"}, field="symptom", where="test")
        >>> _reject_undeclared_names("{typo}", {"count"}, field="symptom", where="test")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        LoadError: ...
    """
    check_template_syntax(template, where=f"{where}: field {field!r}")
    for nm in _free_names_in_template(template):
        if nm not in allowed:
            raise LoadError(
                f"{where}: template field {field!r} references undeclared name {nm!r} "
                f"(not a capture, datum, const, helper, or lifecycle predicate)"
            )


def _check_emit_names(
    emit: dict[str, Any],
    allowed: set[str],
    *,
    where: str,
) -> None:
    """Reject a template field (a ``symptom`` / a ``generic_event`` payload key or
    value) that references a free name resolving to neither a match capture,
    ``datum``/``node``, ``const``, a helper/fixed function, nor a lifecycle
    predicate -- a typo that would otherwise load clean and raise ``NameNotDefined``
    at match time. Mirrors the sibling guards' recursion (pick_one branches +
    default; try_all events). Also checks ``when`` bare expressions at every level.
    """
    if when := emit.get("when"):
        _reject_undeclared_expr_names(when, allowed, field="when", where=where)
    etype = emit.get("type")
    match etype:
        case "pick_one":
            for br in emit.get("branches", []):
                _check_emit_names(br, allowed, where=where)
            if isinstance(default := emit.get("default"), dict):
                _check_emit_names(default, allowed, where=where)
        case "try_all":
            for ev in emit.get("events", []):
                _check_emit_names(ev, allowed, where=where)
        case _:
            if etype in ("bad_node", "interesting_node") and "symptom" in emit:
                _reject_undeclared_names(emit["symptom"], allowed, field="symptom", where=where)
            if etype == "generic_event":
                for key, value in emit.get("payload", {}).items():
                    _reject_undeclared_names(key, allowed, field=f"payload key {key!r}", where=where)
                    _reject_undeclared_names(value, allowed, field=f"payload[{key!r}]", where=where)


def _check_catalog_names(
    emit: dict[str, Any],
    table: dict[str, Any],
    match_captures: set[str],
    static: set[str],
    *,
    where: str,
) -> None:
    """Name-resolve a catalog_lookup's ``project`` (a record leaf whose templates
    reference the RECORD's column names) and its ``not_found`` leaf (whose templates
    reference the match captures). ``static`` is the capture-independent allowed set
    (``datum``/``node``/``const`` + helpers + fixed funcs + lifecycle predicates).
    """
    if when := emit.get("when"):
        _reject_undeclared_expr_names(when, static | match_captures, field="when", where=where)
    if key_expr := emit.get("key", ""):
        _reject_undeclared_expr_names(key_expr, static | match_captures, field="key", where=where)
    if isinstance(project := emit.get("project"), dict):
        for row in table.values():
            _check_emit_names(project, static | set(row.keys()), where=f"{where}.project")
    if isinstance(nf := emit.get("not_found"), dict):
        _check_emit_names(nf, static | match_captures, where=f"{where}.not_found")


def _referenced_calls(emit: dict[str, Any]) -> set[str]:
    """Every function name CALLED anywhere in an emit element's expressions
    (when / symptom / payload key+value; recursing combinators + catalog_lookup).

    Example:
        >>> sorted(_referenced_calls({"type": "bad_node", "when": "int(x) > 0", "symptom": "err"}))
        ['int']
    """
    names: set[str] = set()
    for text, is_template in _iter_emit_expressions(emit):
        names.update(called_names_in(text, is_template=is_template))
    return names


# ---------------------------------------------------------------------------
# Phase 1: build load context
# ---------------------------------------------------------------------------


def _build_load_context(
    loader: "RuleLoader",
    doc: dict[str, Any],
    filepath: str,
) -> _LoadContext:
    """Build the shared load context from loader state and file-level declarations.

    Partitions the @expr_helper registry, merges the inline ``[const]`` table into
    the loader's flat namespace, parses element catalogs and record tables, and merges
    ``@rows``-provided record tables.

    Raises :class:`RegistrationError` for const-name collisions, table-name
    collisions, or invalid file-level ``tags``, all detectable before any stanza
    is inspected.
    """
    all_helpers = dict(loader.declarative_ctx.expr_helpers)
    helper_injections = dict(loader.declarative_ctx.expr_helper_injections)
    non_injecting = {n: f for n, f in all_helpers.items() if not helper_injections.get(n)}
    functions = merge_functions(non_injecting)
    injecting_registry: dict[str, tuple[Callable[..., Any], frozenset[str]]] = {
        n: (all_helpers[n], helper_injections[n]) for n in all_helpers if helper_injections.get(n)
    }
    for n, pred in LIFECYCLE_PREDICATES.items():
        injecting_registry[n] = (pred, LIFECYCLE_INJECTIONS[n])
    allowed_helpers = set(all_helpers)
    lifecycle_names = set(LIFECYCLE_PREDICATES)
    static_names = set(INJECTED_NAMES) | allowed_helpers | set(_FIXED_FUNCS) | lifecycle_names

    new_consts = doc.get("const", {})
    for cname in new_consts:
        if cname in loader.declarative_ctx.const_values:
            raise RegistrationError(
                f"const name {cname!r} (inline [const] in {filepath}) is already "
                f"defined; const names are a single flat namespace"
            )
    loader.declarative_ctx.const_values.update(new_consts)
    const_ns = types.SimpleNamespace(**loader.declarative_ctx.const_values)
    allowed_consts = set(loader.declarative_ctx.const_values)

    try:
        tables = parse_tables(doc)
    except LoadError as exc:
        _wrap_load_error(exc)

    for rname, records in loader.declarative_ctx.record_tables.items():
        if rname in tables.element_catalogs:
            raise RegistrationError(f"record table {rname!r} (@rows provider) collides with [catalog.{rname}]")
        if rname in tables.record_tables:
            raise RegistrationError(f"record table {rname!r} (@rows provider) collides with an inline [rows.{rname}]")
        tables.record_tables[rname] = records

    try:
        file_tags = normalize_tags(doc.get("tags"), where=f"file-level tags in {filepath}")
    except RuleIngestionError as exc:
        raise RegistrationError(str(exc)) from exc

    return _LoadContext(
        functions=functions,
        injecting_registry=injecting_registry,
        allowed_helpers=allowed_helpers,
        lifecycle_names=lifecycle_names,
        static_names=static_names,
        const_ns=const_ns,
        allowed_consts=allowed_consts,
        non_injecting=non_injecting,
        tables=tables,
        file_sources=doc.get("sources", []),
        file_author=doc.get("author"),
        file_tags=file_tags,
    )


# ---------------------------------------------------------------------------
# Phase 1: registration pass
# ---------------------------------------------------------------------------


def _register_extractor_stanzas(
    doc: dict[str, Any],
    loader: "RuleLoader",
) -> "list[_PendingProcessor | _PendingEmit]":
    """Record each ``[[extractor]]`` stanza for later finalization.

    Performs only structural checks that are detectable from the stanza alone:

    - exactly one of ``emit`` or ``processor`` must be present
    - a ``processor`` reference must name a registered ``@rule_processor``

    Raises :class:`RegistrationError` for any structural problem found here.
    Returns the stanzas in order as a list of pending descriptors.
    """
    pending: list[_PendingProcessor | _PendingEmit] = []
    for i, stanza in enumerate(doc.get("extractor", [])):
        if "name" not in stanza:
            raise RegistrationError(f"[[extractor]] stanza #{i}: missing required 'name'")
        name = stanza["name"]
        has_processor = "processor" in stanza
        has_emit = "emit" in stanza
        if has_processor and has_emit:
            raise RegistrationError(f"{name}: a stanza has exactly one of `emit` or `processor`, not both")
        if not has_processor and not has_emit:
            raise RegistrationError(f"{name}: a stanza requires either `emit` or `processor`")
        if has_processor:
            ns_name = stanza["processor"]
            if loader.declarative_ctx.rule_processors.get(ns_name) is None:
                raise RegistrationError(
                    f"{name}: processor {ns_name!r} is not registered "
                    f"(define it with @rule_processor({ns_name!r}) in a loaded .py rule file)"
                )
            pending.append(_PendingProcessor(index=i, stanza=stanza, name=name))
        else:
            pending.append(_PendingEmit(index=i, stanza=stanza, name=name))
    return pending


# ---------------------------------------------------------------------------
# Phase 2: finalize pass
# ---------------------------------------------------------------------------


def _wrap_load_error(exc: LoadError) -> NoReturn:
    """Raise exc as a FinalizeError, preserving the original cause chain."""
    raise FinalizeError(str(exc)) from exc


def _finalize_processor(
    pending: _PendingProcessor,
    loader: "RuleLoader",
    ctx: _LoadContext,
    filepath: str,
) -> "Rule":
    """Compile and register a processor stanza.

    Raises :class:`FinalizeError` for source-resolution or compilation failures.
    """
    stanza = pending.stanza
    name = pending.name
    try:
        func = loader.declarative_ctx.rule_processors[stanza["processor"]]
        proc = compile_processor(func, rule_name=name)
        sources = _resolve_sources(stanza.get("sources", ctx.file_sources), where=name)
        tags = _resolve_tags(stanza.get("tags"), ctx=ctx, where=name, filepath=filepath)
        if "match_re" not in stanza:
            raise LoadError(f"{name}: missing required 'match_re'")
        rule = ExtractorRule(
            name=name,
            author=stanza.get("author") or ctx.file_author or "unknown",
            location=RuleLocation(filepath, pending.index),
            match_re=stanza["match_re"],
            processor=proc,
            source_filters=sources,
            tags=tags,
        )
        loader.register_rule(rule)
        loader.declared_rule_names.update(stanza.get("derived_rules", []))
        loader.declared_event_types.update(stanza.get("derived_event_types", []))
        loader.declarative_emit[name] = json.dumps(
            {
                "processor": stanza["processor"],
                "derived_rules": sorted(stanza.get("derived_rules", [])),
                "derived_event_types": sorted(stanza.get("derived_event_types", [])),
            },
            sort_keys=True,
        )
        return rule
    except LoadError as exc:
        if type(exc) is LoadError:
            _wrap_load_error(exc)
        raise


def _finalize_emit_stanza(
    pending: _PendingEmit,
    loader: "RuleLoader",
    ctx: _LoadContext,
    filepath: str,
) -> "Rule":
    """Validate, compile, and register an emit stanza.

    Compiles ``match_re``, checks reserved captures, runs guard/helper/const/name
    checks, verifies coverage, compiles the emit tree, and registers the rule.

    Raises :class:`FinalizeError` for any validation or compilation failure.
    """
    stanza = pending.stanza
    name = pending.name
    emit = stanza["emit"]
    try:
        if "match_re" not in stanza:
            raise LoadError(f"{name}: missing required 'match_re'")
        try:
            compiled = re.compile(stanza["match_re"])
        except (re.error, TypeError) as exc:
            raise LoadError(f"{name}: invalid 'match_re': {exc}") from exc
        for reserved in INJECTED_NAMES:
            if reserved in compiled.groupindex:
                raise LoadError(
                    f"{name}: capture group {reserved!r} is reserved (it names a value "
                    f"injected into the expression environment)"
                )

        def match_groups(
            case: dict[str, Any], _compiled: re.Pattern[str] = compiled, _const: types.SimpleNamespace = ctx.const_ns
        ) -> dict[str, Any]:
            m = _compiled.search(case["line"])
            if not m:
                return {}
            g = m.groupdict()
            g["datum"] = case["line"]
            g["const"] = _const
            g["node"] = None
            return g

        is_catalog = emit.get("type") == "catalog_lookup"
        table: dict[str, Any] = {}
        is_element: bool = False
        referenced = _referenced_calls(emit)
        rule_injecting = {n: ctx.injecting_registry[n] for n in referenced if n in ctx.injecting_registry}

        if is_catalog:
            table, is_element = resolve_catalog_table(emit, ctx.tables, rule_name=name)
            optional = optional_captures(stanza["match_re"])
            _check_catalog_guards(emit, optional, where=name)
            _check_emit_helpers(emit, ctx.allowed_helpers)
            _check_emit_consts(emit, ctx.allowed_consts)
            _check_catalog_names(emit, table, set(compiled.groupindex), ctx.static_names, where=name)
            if is_element:
                allowed = ctx.static_names | set(compiled.groupindex)
                post_when = _catalog_post_when_optional(emit, optional)
                for key, row in table.items():
                    _check_emit_names(row, allowed, where=f"{name}.row[{key!r}]")
                    _check_emit_guards(row, post_when, where=f"{name}.row[{key!r}]")
            check_catalog_coverage(
                name,
                emit,
                table,
                stanza.get("cases", []),
                stanza.get("untested"),
                stanza.get("untested_rows"),
                match_groups=match_groups,
                functions=ctx.functions,
                lifecycle=ctx.lifecycle_names,
            )
            proc = compile_catalog_lookup(
                emit,
                rule_name=name,
                tables=ctx.tables,
                helpers=ctx.non_injecting,
                consts=ctx.const_ns,
                injecting=rule_injecting,
            )
        else:
            _check_emit_guards(emit, optional_captures(stanza["match_re"]), where=name)
            _check_emit_helpers(emit, ctx.allowed_helpers)
            _check_emit_consts(emit, ctx.allowed_consts)
            _check_emit_names(emit, set(compiled.groupindex) | ctx.static_names, where=name)
            check_coverage(
                name,
                emit,
                stanza.get("cases", []),
                stanza.get("untested"),
                match_groups=match_groups,
                functions=ctx.functions,
                lifecycle=ctx.lifecycle_names,
            )
            proc = compile_element(
                emit,
                rule_name=name,
                helpers=ctx.non_injecting,
                consts=ctx.const_ns,
                injecting=rule_injecting,
            )

        sources = _resolve_sources(stanza.get("sources", ctx.file_sources), where=name)
        tags = _resolve_tags(stanza.get("tags"), ctx=ctx, where=name, filepath=filepath)
        rule = ExtractorRule(
            name=name,
            author=stanza.get("author") or ctx.file_author or "unknown",
            location=RuleLocation(filepath, pending.index),
            match_re=stanza["match_re"],
            processor=proc,
            source_filters=sources,
            tags=tags,
        )
        loader.register_rule(rule)

        declared_rules: set[str] = set()
        declared_event_types: set[str] = set()
        if is_catalog:
            _collect_catalog_declarations(emit, table, is_element, declared_rules, declared_event_types)
        else:
            _collect_declarations(emit, declared_rules, declared_event_types)
        loader.declared_rule_names.update(declared_rules)
        loader.declared_event_types.update(declared_event_types)
        loader.declarative_emit[name] = json.dumps(emit, sort_keys=True)
        return rule
    except LoadError as exc:
        if type(exc) is LoadError:
            _wrap_load_error(exc)
        raise


def _finalize_family(
    family: dict[str, Any],
    fi: int,
    loader: "RuleLoader",
    ctx: _LoadContext,
    filepath: str,
) -> "dict[RuleId, Rule]":
    """Validate, compile, and register a family stanza.

    Stamps the family into generated rules, runs helper/const checks and coverage,
    compiles each generated rule, and registers them.

    Raises :class:`FinalizeError` for any validation or compilation failure.
    """
    fam_name = family.get("name", "family")
    added: "dict[RuleId, Rule]" = {}
    try:
        fam_sources = _resolve_sources(family.get("sources", ctx.file_sources), where=fam_name)
        fam_author = family.get("author") or ctx.file_author or "unknown"
        fam_tags = _resolve_tags(family.get("tags"), ctx=ctx, where=fam_name, filepath=filepath)
        generated = stamp_family(family, ctx.tables)
        referenced = _referenced_calls(family["emit"])
        rule_injecting = {n: ctx.injecting_registry[n] for n in referenced if n in ctx.injecting_registry}
        _check_emit_helpers(family["emit"], ctx.allowed_helpers)
        _check_emit_consts(family["emit"], ctx.allowed_consts)
        check_family_coverage(
            fam_name,
            generated,
            family.get("cases", []),
            family.get("untested"),
            family.get("untested_rows"),
            functions=ctx.functions,
        )
        declared_rules: set[str] = set()
        declared_event_types: set[str] = set()
        for gen in generated:
            _check_emit_names(gen.emit, ctx.static_names, where=gen.name)
            proc = compile_element(
                gen.emit,
                rule_name=gen.name,
                helpers=ctx.non_injecting,
                consts=ctx.const_ns,
                injecting=rule_injecting,
            )
            rule = ExtractorRule(
                name=gen.name,
                author=fam_author,
                location=RuleLocation(filepath, fi),
                match_re=gen.match_re,
                processor=proc,
                source_filters=fam_sources,
                tags=fam_tags,
            )
            loader.register_rule(rule)
            _collect_declarations(gen.emit, declared_rules, declared_event_types)
            loader.declarative_emit[gen.name] = json.dumps(gen.emit, sort_keys=True)
            added[gen.name] = rule
        loader.declared_rule_names.update(declared_rules)
        loader.declared_event_types.update(declared_event_types)
        return added
    except LoadError as exc:
        if type(exc) is LoadError:
            _wrap_load_error(exc)
        raise


# ---------------------------------------------------------------------------
# Prefilter loader
# ---------------------------------------------------------------------------


def _load_prefilter(loader: "RuleLoader", pf_doc: dict[str, Any], ctx: _LoadContext, filepath: str) -> None:
    """Register a [prefilter] section: include/exclude term lists for each source."""
    raw_sources = pf_doc.get("sources", ctx.file_sources)
    if not raw_sources:
        raise RegistrationError(
            f"{filepath}: [prefilter] has no sources -- set file-level 'sources' or add 'sources' to [prefilter]"
        )
    try:
        sources = _resolve_sources(raw_sources, where=f"{filepath}: [prefilter]")
    except LoadError as exc:
        raise RegistrationError(str(exc)) from exc

    def _checked_str_list(key: str) -> list[str]:
        val = pf_doc.get(key, [])
        if not isinstance(val, list) or not all(isinstance(s, str) for s in val):
            raise RegistrationError(
                f"{filepath}: [prefilter].{key} must be a list of strings, got {type(val).__name__!r}"
            )
        return list(cast(list[str], val))  # cast: all() guard proved str but checker can't narrow list[object]

    spec = PreFilterSpec(
        include_terms=_checked_str_list("include"),
        exclude_terms=_checked_str_list("exclude"),
    )
    for source in sources:
        loader.register_pre_filter(source, spec)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_toml_rules(
    loader: "RuleLoader",
    filepath: str | os.PathLike[str],
) -> "dict[RuleId, Rule]":
    """Parse a declarative .toml rule file and register its extractors.

    Returns the rules added by this file (mirrors ``RuleLoader.load``'s return).

    The load process has two phases:

    - **Registration**: build the shared context (helpers, consts, tables) and
      record each stanza, catching structural problems early as
      :class:`RegistrationError`.
    - **Finalize**: for each recorded stanza, compile ``match_re``, validate
      guards/helpers/consts/names, check coverage, compile the emit tree, and
      register the rule, raising :class:`FinalizeError` on any failure.
    """
    filepath = os.fspath(filepath)
    with open(filepath, "rb") as fh:
        doc = tomllib.load(fh)

    _check_declared_keys(doc, filepath)

    ctx = _build_load_context(loader, doc, filepath)
    pending = _register_extractor_stanzas(doc, loader)

    added: "dict[RuleId, Rule]" = {}
    for p in pending:
        if isinstance(p, _PendingProcessor):
            added[p.name] = _finalize_processor(p, loader, ctx, filepath)
        else:
            added[p.name] = _finalize_emit_stanza(p, loader, ctx, filepath)

    for fi, family in enumerate(doc.get("family", [])):
        added.update(_finalize_family(family, fi, loader, ctx, filepath))

    if "prefilter" in doc:
        _load_prefilter(loader, doc["prefilter"], ctx, filepath)

    return added
