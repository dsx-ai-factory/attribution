# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Compile a declarative ``emit`` stanza into an ExtractorProcessor closure."""

import inspect
import types
from collections.abc import Callable
from typing import Any

from attribution_lib.declarative.catalogs import Tables, resolve_catalog_table
from attribution_lib.declarative.errors import (
    FinalizeError as FinalizeError,
    LoadError as LoadError,
    RegistrationError as RegistrationError,
)
from attribution_lib.declarative.expressions import (
    bind_functions,
    eval_bool,
    eval_expr,
    merge_functions,
    render_template,
)
from attribution_lib.events import DemotionTypes
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules_engine import EventTriple, ExtractorProcessor, RuleContext
from attribution_lib.sources import Source

# Re-exported for callers that import ``LoadError`` from this module (loader, tests).
__all__ = [
    "LoadError",
    "RegistrationError",
    "FinalizeError",
    "INJECTED_NAMES",
    "compile_leaf",
    "compile_element",
    "compile_catalog_lookup",
    "compile_processor",
]

# Names injected into every expression environment at runtime. Capture groups
# with these names are rejected at load time to prevent silent shadowing.
INJECTED_NAMES: frozenset[str] = frozenset({"datum", "node", "const"})


def _env(
    groups: dict[str, Any],
    source: Source,
    node: str | None = None,
    const: object | None = None,
) -> dict[str, Any]:
    """The expression namespace for a compiled closure: the match captures plus
    `datum` (the full log line), `node` (the source hostname), and `const` (the
    engine's constant namespace, a SimpleNamespace read as `const.<name>`).
    Centralized so every compiler builds it identically. A capture literally named
    `node`/`const` keeps precedence over the injected value (the loader rejects a
    capture named `const`, so in practice there is no conflict)."""
    names = dict(groups)
    names["datum"] = source.datum
    if node is not None and "node" not in names:
        names["node"] = node
    if const is not None and "const" not in names:
        names["const"] = const
    return names


# Keys that only combinators may carry; a leaf carrying any of them is a LoadError
# (no silently-ignored arms on a leaf).
_ARM_KEYS = ("branches", "events", "default")


def compile_element(
    emit: dict[str, Any],
    *,
    rule_name: str,
    helpers: dict[str, Callable] | None = None,
    consts: types.SimpleNamespace | None = None,
    injecting: dict[str, tuple[Callable[..., Any], frozenset[str]]] | None = None,
) -> ExtractorProcessor:
    """Recursive entry point: compile any emit element into a processor closure.

    Leaf types delegate to :func:`compile_leaf`; ``pick_one`` to
    :func:`compile_pick_one`; ``try_all`` to :func:`compile_try_all`.
    Anything else (including ``catalog_lookup``) raises :class:`LoadError`.
    """
    match emit.get("type"):
        case "bad_node" | "interesting_node" | "generic_event" as etype:
            for k in _ARM_KEYS:
                if k in emit:
                    raise LoadError(f"{rule_name}: leaf {etype!r} must not carry {k!r} (no arms on a leaf)")
            return compile_leaf(emit, rule_name=rule_name, helpers=helpers, consts=consts, injecting=injecting)
        case "pick_one":
            return compile_pick_one(emit, rule_name=rule_name, helpers=helpers, consts=consts, injecting=injecting)
        case "try_all":
            return compile_try_all(emit, rule_name=rule_name, helpers=helpers, consts=consts, injecting=injecting)
        case etype:
            raise LoadError(
                f"{rule_name}: emit type {etype!r} is not supported here "
                f"(expected bad_node/interesting_node/generic_event/pick_one/try_all)"
            )


def compile_catalog_lookup(
    emit: dict[str, Any],
    *,
    rule_name: str,
    tables: Tables,
    helpers: dict[str, Callable] | None = None,
    consts: types.SimpleNamespace | None = None,
    injecting: dict[str, tuple[Callable[..., Any], frozenset[str]]] | None = None,
) -> ExtractorProcessor:
    """Compile a root-only ``catalog_lookup`` into a processor closure.

    Looks a ``str()``-coerced key up in an element catalog (emit the matched row
    directly as a leaf) or a record table (project the record's columns into a
    ``project`` leaf). ``not_found`` (required) is a leaf, or ``"none"`` (emit
    nothing). An optional ``when`` gates the whole lookup. An optional
    ``derived_rule`` is a template over ``{key}`` crediting a hit; a matched
    element-catalog row's own ``derived_rule`` takes precedence over it.

    Every emitted leaf (each catalog row, the record ``project``, and
    ``not_found``) is compiled ONCE at load time, so a malformed row or leaf
    raises :class:`LoadError` at load, not at match, and no per-match recompile
    happens.
    """
    base_functions = merge_functions(helpers)
    table, is_element = resolve_catalog_table(emit, tables, rule_name=rule_name)
    name = emit["catalog"]

    project = emit.get("project")
    if not is_element and project is None:
        raise LoadError(f"{rule_name}: catalog_lookup over record table {name!r} requires a 'project'")
    if is_element and project is not None:
        raise LoadError(f"{rule_name}: catalog_lookup over element catalog {name!r} must not have 'project'")

    key_expr = emit.get("key")
    if key_expr is None:
        raise LoadError(f"{rule_name}: catalog_lookup is missing the required 'key' field")
    when = emit.get("when")
    lookup_derived = emit.get("derived_rule")  # a template over {key}
    nf = emit.get("not_found")
    if nf is None:
        raise LoadError(f"{rule_name}: catalog_lookup requires 'not_found' (a leaf, or \"none\")")
    if isinstance(nf, str) and nf != "none":
        raise LoadError(f"{rule_name}: catalog_lookup 'not_found' must be a leaf element or \"none\", got {nf!r}")
    # not_found is a LEAF (or "none"); a combinator here is a LoadError.
    not_found_proc = (
        None
        if nf == "none"
        else compile_leaf(nf, rule_name=f"{rule_name}.not_found", helpers=helpers, consts=consts, injecting=injecting)
    )

    if is_element:
        # Precompile each row's final leaf at LOAD, resolving credit precedence:
        # the row's own derived_rule wins; else the lookup's derived_rule template
        # rendered with {key}; else nothing (ExtractorRule.process fills the
        # stanza name). A malformed row raises LoadError here, at load.
        compiled_rows: dict[str, ExtractorProcessor] = {}
        for key, row in table.items():
            row_derived = row.get("derived_rule")
            credit = row_derived or (render_template(lookup_derived, {"key": key}) if lookup_derived else None)
            leaf = {k: v for k, v in row.items() if k != "derived_rule"}
            if credit is not None:
                leaf["derived_rule"] = credit
            compiled_rows[key] = compile_leaf(
                leaf, rule_name=f"{rule_name}:{key}", helpers=helpers, consts=consts, injecting=injecting
            )

        def _proc(groups: dict[str, Any], node: str | None, source: Source, ctx: RuleContext) -> list[EventTriple]:
            names = _env(groups, source, node, const=consts)
            functions = bind_functions(base_functions, injecting, source, ctx)
            if when and not eval_bool(when, names, functions):
                return []
            key = str(eval_expr(key_expr, names, functions))
            proc = compiled_rows.get(key)
            if proc is None:
                return not_found_proc(groups, node, source, ctx) if not_found_proc else []
            return proc(groups, node, source, ctx)

        return _proc

    # Record table: the project leaf is fixed, so compile it ONCE at load.
    # Credit precedence mirrors the element path: project's own derived_rule wins
    # over the lookup-level template.  If project carries derived_rule, compile_leaf
    # bakes it in and we skip the runtime lookup credit entirely.
    project_leaf = project or {}
    project_has_credit = isinstance(project_leaf, dict) and project_leaf.get("derived_rule") is not None
    project_proc = compile_leaf(
        project_leaf, rule_name=f"{rule_name}.project", helpers=helpers, consts=consts, injecting=injecting
    )

    def _proc_record(groups: dict[str, Any], node: str | None, source: Source, ctx: RuleContext) -> list[EventTriple]:
        names = _env(groups, source, node, const=consts)
        functions = bind_functions(base_functions, injecting, source, ctx)
        if when and not eval_bool(when, names, functions):
            return []
        key = str(eval_expr(key_expr, names, functions))
        row = table.get(key)
        if row is None:
            return not_found_proc(groups, node, source, ctx) if not_found_proc else []
        # Render the project leaf using the RECORD's columns as groups.
        events = project_proc(row, node, source, ctx)
        if not project_has_credit and lookup_derived is not None:
            credit = render_template(lookup_derived, {"key": key})
            events = [e._replace(source=e.source.derived(rule=credit)) for e in events]
        return events

    return _proc_record


def compile_pick_one(
    emit: dict[str, Any],
    *,
    rule_name: str,
    helpers: dict[str, Callable] | None = None,
    consts: types.SimpleNamespace | None = None,
    injecting: dict[str, tuple[Callable[..., Any], frozenset[str]]] | None = None,
) -> ExtractorProcessor:
    """Compile a ``pick_one``: ordered when-guarded branches, first-match-wins.

    Every branch must carry a ``when``; ``default`` must be present (an element
    dict, or the string ``"none"`` to emit nothing). An optional combinator-level
    ``when`` gates the whole element (emit nothing if false, before any branch).
    """
    branches = emit.get("branches")
    if not branches:
        raise LoadError(f"{rule_name}: pick_one requires a non-empty 'branches' list")
    if "default" not in emit:
        raise LoadError(f"{rule_name}: pick_one requires a 'default' (an element, or \"none\")")

    base_functions = merge_functions(helpers)
    compiled_branches = []
    for i, br in enumerate(branches):
        when = br.get("when")
        if not when:
            raise LoadError(f"{rule_name}: pick_one branch {i} requires a 'when'")
        # Strip `when` before compiling the branch body: pick_one selects the
        # branch via eval_bool above; the body should emit unconditionally once
        # selected, not re-evaluate the same condition.
        body = {k: v for k, v in br.items() if k != "when"}
        compiled_branches.append(
            (
                when,
                compile_element(
                    body, rule_name=f"{rule_name}.branch[{i}]", helpers=helpers, consts=consts, injecting=injecting
                ),
            )
        )

    default = emit["default"]
    if isinstance(default, str) and default != "none":
        raise LoadError(f"{rule_name}: pick_one 'default' must be an element or \"none\", got {default!r}")
    default_proc = (
        None
        if default == "none"
        else compile_element(
            default, rule_name=f"{rule_name}.default", helpers=helpers, consts=consts, injecting=injecting
        )
    )
    outer_when = emit.get("when")

    def _proc(groups: dict[str, Any], node: str | None, source: Source, ctx: RuleContext) -> list[EventTriple]:
        names = _env(groups, source, node, const=consts)
        functions = bind_functions(base_functions, injecting, source, ctx)
        if outer_when and not eval_bool(outer_when, names, functions):
            return []
        for when, proc in compiled_branches:
            if eval_bool(when, names, functions):
                return proc(groups, node, source, ctx)
        return default_proc(groups, node, source, ctx) if default_proc else []

    return _proc


def compile_try_all(
    emit: dict[str, Any],
    *,
    rule_name: str,
    helpers: dict[str, Callable] | None = None,
    consts: types.SimpleNamespace | None = None,
    injecting: dict[str, tuple[Callable[..., Any], frozenset[str]]] | None = None,
) -> ExtractorProcessor:
    """Compile a ``try_all``: emit every event whose ``when`` holds, in order.

    Each nested element already handles its own ``when`` (an unguarded event
    always emits), so this simply concatenates each child's emissions. An
    optional combinator-level ``when`` gates the whole list (emit nothing if
    false, before evaluating any event).
    """
    events = emit.get("events")
    if not events:
        raise LoadError(f"{rule_name}: try_all requires a non-empty 'events' list")
    base_functions = merge_functions(helpers)
    compiled = [
        compile_element(ev, rule_name=f"{rule_name}.event[{i}]", helpers=helpers, consts=consts, injecting=injecting)
        for i, ev in enumerate(events)
    ]
    outer_when = emit.get("when")

    def _proc(groups: dict[str, Any], node: str | None, source: Source, ctx: RuleContext) -> list[EventTriple]:
        functions = bind_functions(base_functions, injecting, source, ctx)
        if outer_when and not eval_bool(outer_when, _env(groups, source, node, const=consts), functions):
            return []
        out: list[EventTriple] = []
        for proc in compiled:
            out.extend(proc(groups, node, source, ctx))
        return out

    return _proc


_INJECTABLE_PARAMS = frozenset({"node", "source", "ctx"})


def compile_processor(func: Callable[..., Any], *, rule_name: str) -> ExtractorProcessor:
    """Wrap a ``@rule_processor`` function into an ExtractorProcessor closure.

    ``groups`` is always passed as the first positional argument; ``node`` /
    ``source`` / ``ctx`` are injected as keyword arguments only for the ones the
    function declares (the engine already supplies the current node = ctx.hostname,
    source, and ctx to every ExtractorProcessor, so a migrated processor's body is
    unchanged). Raises :class:`LoadError` if the signature contains a required
    parameter that cannot be satisfied, or if an injectable name is positional-only."""
    params = inspect.signature(func).parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not positional or positional[0].name != "groups":
        raise LoadError(f"{rule_name}: processor must accept 'groups' as its first positional parameter")
    allowed = {"groups"} | _INJECTABLE_PARAMS
    for pname, param in params.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if pname not in allowed and param.default is inspect.Parameter.empty:
            raise LoadError(
                f"{rule_name}: processor has unsatisfied required parameter {pname!r}; "
                f"only groups/node/source/ctx are supplied by the engine"
            )
        if pname in _INJECTABLE_PARAMS and param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise LoadError(
                f"{rule_name}: processor parameter {pname!r} is positional-only "
                f"and cannot be injected as a keyword argument"
            )
    inject_node = "node" in params
    inject_source = "source" in params
    inject_ctx = "ctx" in params

    def _proc(groups: dict[str, Any], node: str | None, source: Source, ctx: RuleContext) -> list[EventTriple]:
        kwargs: dict[str, Any] = {}
        if inject_node:
            kwargs["node"] = node
        if inject_source:
            kwargs["source"] = source
        if inject_ctx:
            kwargs["ctx"] = ctx
        return func(groups, **kwargs)  # type: ignore[no-any-return]

    return _proc


def _resolve_demotion(
    demotion: dict[str, Any] | str | None, *, rule_name: str
) -> dict[DemotionTypes, Any] | str | None:
    """Map a demotion dict's keys (DemotionTypes member NAMES) to enum members.

    A str shorthand or ``None`` passes through unchanged (handled downstream by
    ``_bad``/``_interesting``). An unknown member name raises :class:`LoadError`.

    Example:
        >>> _resolve_demotion("full", rule_name="test")
        'full'
        >>> _resolve_demotion(None, rule_name="test") is None
        True
    """
    if not isinstance(demotion, dict):
        return demotion
    out = {}
    for key, val in demotion.items():
        try:
            out[DemotionTypes[key]] = val
        except KeyError as e:
            raise LoadError(f"{rule_name}: unknown demotion type {key!r}") from e
    return out


def compile_leaf(
    emit: dict[str, Any],
    *,
    rule_name: str,
    helpers: dict[str, Callable] | None = None,
    consts: types.SimpleNamespace | None = None,
    injecting: dict[str, tuple[Callable[..., Any], frozenset[str]]] | None = None,
) -> ExtractorProcessor:
    """Compile a leaf ``emit`` dict into an ExtractorProcessor closure.

    Raises :class:`LoadError` for a non-leaf emit type (combinators/catalog_lookup).
    """
    base_functions = merge_functions(helpers)
    when = emit.get("when")
    # A leaf ``derived_rule`` is a LITERAL slug (not a template, not guard-checked):
    # it credits a finding under an explicit name instead of the stanza name that
    # ``ExtractorRule.process`` would otherwise fill in.
    derived_rule = emit.get("derived_rule")

    match emit.get("type"):
        case "bad_node" | "interesting_node" as etype:
            symptom_tmpl = emit["symptom"]
            demotion = _resolve_demotion(emit.get("demotion"), rule_name=rule_name)
            make = _bad if etype == "bad_node" else _interesting

            def _proc(groups: dict[str, Any], node: str | None, source: Source, ctx: RuleContext) -> list[EventTriple]:
                names = _env(groups, source, node, const=consts)
                functions = bind_functions(base_functions, injecting, source, ctx)
                if when and not eval_bool(when, names, functions):
                    return []
                src = source.derived(rule=derived_rule) if derived_rule else source
                return [make(node, render_template(symptom_tmpl, names, functions), src, demotion=demotion)]

            return _proc

        case "generic_event":
            event_type = emit["event_type"]
            payload_tmpl = dict(emit.get("payload", {}))

            def _proc_generic(
                groups: dict[str, Any], node: str | None, source: Source, ctx: RuleContext
            ) -> list[EventTriple]:
                names = _env(groups, source, node, const=consts)
                functions = bind_functions(base_functions, injecting, source, ctx)
                if when and not eval_bool(when, names, functions):
                    return []
                payload = {
                    render_template(k, names, functions): render_template(v, names, functions)
                    for k, v in payload_tmpl.items()
                }
                src = source.derived(rule=derived_rule) if derived_rule else source
                return [EventTriple(type=event_type, payload=payload, source=src)]

            return _proc_generic

        case etype:
            raise LoadError(f"{rule_name}: emit type {etype!r} is not a leaf (bad_node/interesting_node/generic_event)")
