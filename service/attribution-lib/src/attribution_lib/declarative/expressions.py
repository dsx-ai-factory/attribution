# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""``{expr}`` templating for declarative ``symptom``/``payload`` values.

Placeholders are ``{expr}`` where ``expr`` is a simpleeval expression evaluated
over the matched capture ``groups`` plus a fixed set of functions. Literal
braces are written ``{{`` / ``}}``. Plain text with no braces passes through.
"""

from __future__ import annotations

import ast
import functools
import keyword
import re
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from simpleeval import EvalWithCompoundTypes  # type: ignore[import-untyped]

from attribution_lib.declarative.errors import LoadError
from attribution_lib.declarative.lifecycle import LIFECYCLE_PREDICATES

try:
    from re import _parser as _sre_parse  # type: ignore[attr-defined]  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover
    import sre_parse as _sre_parse  # older fallback

if TYPE_CHECKING:
    from attribution_lib.rule_types.definitions import RuleContext
    from attribution_lib.sources import Source


def _scan_to_matching_brace(s: str, start: int, n: int) -> tuple[int, bool]:
    """Scan from ``start`` (just after an opening ``{``) to its matching ``}``.

    String literals are tracked so that braces within them (e.g. quantifiers in
    ``r'\\d{3}'``) are not counted as placeholder delimiters.

    Returns ``(end, True)`` where ``s[end - 1] == '}'``, or ``(start, False)``
    if no matching brace is found before the end of ``s``.
    """
    i = start
    depth = 1
    while i < n and depth > 0:
        c = s[i]
        if c in ('"', "'"):
            triple = s[i : i + 3]
            if triple in ('"""', "'''"):
                q, qlen = triple, 3
            else:
                q, qlen = c, 1
            i += qlen
            while i < n:
                if s[i : i + qlen] == q:
                    i += qlen
                    break
                if s[i] == "\\" and qlen == 1:
                    i += 2  # skip escaped char in non-raw strings
                else:
                    i += 1
        elif c == "{":
            depth += 1
            i += 1
        elif c == "}":
            depth -= 1
            i += 1
        else:
            i += 1
    return (i, True) if depth == 0 else (start, False)


def _scan_template(template: str) -> Iterator[tuple[str, str]]:
    """Tokenize a ``{expr}`` template into ``(kind, value)`` pairs.

    Kinds: ``'{{'``, ``'}}'``, ``'expr'`` (a placeholder expression), ``'stray'``
    (an unmatched ``{`` or ``}``), and ``'text'`` (plain literal text).

    Braces inside quoted string literals within a placeholder expression are not
    treated as delimiters, so ``r'\\d{3}'`` in an expression is handled correctly.
    """
    i = 0
    n = len(template)
    text_start = 0
    while i < n:
        c = template[i]
        if c == "{":
            if i > text_start:
                yield "text", template[text_start:i]
            if i + 1 < n and template[i + 1] == "{":
                yield "{{", ""
                i += 2
            else:
                end, ok = _scan_to_matching_brace(template, i + 1, n)
                if ok:
                    yield "expr", template[i + 1 : end - 1]
                    i = end
                else:
                    i += 1
                    yield "stray", "{"
            text_start = i
        elif c == "}":
            if i > text_start:
                yield "text", template[text_start:i]
            if i + 1 < n and template[i + 1] == "}":
                yield "}}", ""
                i += 2
            else:
                i += 1
                yield "stray", "}"
            text_start = i
        else:
            i += 1
    if text_start < n:
        yield "text", template[text_start:]


_FIXED_FUNCS: dict[str, Callable[..., Any]] = {
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "bool": bool,
    "round": round,
    "search": re.search,
    "match": re.match,
}


def _eval(
    expr: str,
    names: dict[str, Any],
    functions: dict[str, Callable] | None = None,
) -> Any:
    """Evaluate ``expr`` with simpleeval over ``names`` and ``functions``.

    Falls back to ``_FIXED_FUNCS`` when ``functions`` is ``None``; callers that
    have already merged helpers pass their combined map directly."""
    ev = EvalWithCompoundTypes(names=names, functions=dict(functions or _FIXED_FUNCS))
    return ev.eval(expr)


def merge_functions(helpers: dict[str, Callable] | None) -> dict[str, Callable]:
    """The simpleeval ``functions`` map for a compiled rule: the fixed functions
    plus any registered ``@expr_helper``s. Returns the shared ``_FIXED_FUNCS``
    object unchanged when there are no helpers (the common case: no per-call dict
    build, and callers may identity-check the empty case).

    Example:
        >>> merge_functions(None) is merge_functions(None)
        True
        >>> "my_fn" in merge_functions({"my_fn": len})
        True
    """
    if not helpers:
        return _FIXED_FUNCS
    return {**_FIXED_FUNCS, **helpers}


def render_template(
    template: str,
    groups: dict[str, Any],
    functions: dict[str, Callable] | None = None,
) -> str:
    """Expand ``{expr}`` placeholders in ``template`` over ``groups``.

    Literal ``{{`` / ``}}`` produce a single brace. A template with no placeholders
    is returned unchanged. Each placeholder expression is evaluated with
    :func:`_eval` and ``str()``-coerced.

    Example:
        >>> render_template("GPU {name} down", {"name": "A100"})
        'GPU A100 down'
        >>> render_template("{{escaped}}", {})
        '{escaped}'
    """
    parts: list[str] = []
    for kind, value in _scan_template(template):
        if kind == "{{":
            parts.append("{")
        elif kind == "}}":
            parts.append("}")
        elif kind == "expr":
            parts.append(str(_eval(value, dict(groups), functions)))
        else:  # 'text' or 'stray' — pass through unchanged
            parts.append(value)
    return "".join(parts)


def eval_bool(
    expr: str,
    groups: dict[str, Any],
    functions: dict[str, Callable] | None = None,
) -> bool:
    """Evaluate ``expr`` over ``groups`` and coerce to bool (used for ``when``).

    Example:
        >>> eval_bool("count > 0", {"count": 3})
        True
        >>> eval_bool("flag", {"flag": None})
        False
    """
    return bool(_eval(expr, dict(groups), functions))


def eval_expr(
    expr: str,
    groups: dict[str, Any],
    functions: dict[str, Callable] | None = None,
) -> Any:
    """Evaluate a bare ``expr`` over ``groups`` and return the raw value.

    Same namespace/functions as :func:`eval_bool` but without bool-coercion (used
    for a ``catalog_lookup`` key expression, whose result is ``str()``-coerced by
    the caller before the table lookup).

    Example:
        >>> eval_expr("int(score)", {"score": "42"})
        42
        >>> eval_expr("x + y", {"x": 1, "y": 2})
        3
    """
    return _eval(expr, dict(groups), functions)


# --- optional-capture detection -------------------------------------------------


def optional_captures(match_re: str) -> set[str]:
    """Names of capture groups the regex can leave unmatched (bind to None).

    A group is optional if it sits under a repeat with min-count 0 (``?``, ``*``,
    ``{0,n}``) or inside an alternation (``BRANCH``) branch. Parsed structurally with
    the stdlib regex parser so bounded-repeat and alternation cases are caught, not
    just a trailing ``?``/``*``.

    Example:
        >>> optional_captures(r"(?P<a>x)?(?P<b>y)")
        {'a'}
        >>> optional_captures(r"(?P<a>x)(?P<b>y)")
        set()
    """
    index_to_name = {idx: name for name, idx in re.compile(match_re).groupindex.items()}
    parsed = _sre_parse.parse(match_re)
    optional: set[str] = set()

    def walk(seq: Any, in_optional: bool) -> None:
        for op, av in seq:
            opname = getattr(op, "name", str(op))
            if opname == "SUBPATTERN":
                gid = av[0]
                name = index_to_name.get(gid)
                if name is not None and in_optional:
                    optional.add(name)
                walk(av[3], in_optional)  # the subpattern body
            elif opname in ("MAX_REPEAT", "MIN_REPEAT"):
                min_count, _max, sub = av
                walk(sub, in_optional or min_count == 0)
            elif opname == "BRANCH":
                _, branches = av
                for br in branches:
                    walk(br, True)  # any branch may not participate
            else:
                # LITERAL, IN, AT, ANY, etc. carry no subpattern with groups here.
                pass

    walk(parsed, False)
    return optional


# --- load-time guard checker ----------------------------------------------------


def _parse_expr(expr: str) -> ast.expr:
    """Parse a single expression (``mode="eval"``), turning a ``SyntaxError`` into
    a :class:`LoadError`. A capture named after a Python keyword (``class`` /
    ``return`` / ...) renders a ``{class}`` template that cannot be parsed as
    Python; point the author at renaming the capture group."""
    try:
        return ast.parse(expr, mode="eval").body
    except SyntaxError as e:
        kw = next(
            (t for t in re.findall(r"[A-Za-z_]\w*", expr) if keyword.iskeyword(t)),
            None,
        )
        hint = (
            f" (it uses the Python keyword {kw!r}; a capture group named after a "
            f"keyword cannot be used in an expression -- rename the group)"
            if kw
            else ""
        )
        raise LoadError(f"cannot parse expression {expr!r}{hint}") from e


def check_guarded(
    text: str,
    optional: set[str],
    *,
    is_template: bool = True,
) -> None:
    """Raise :class:`LoadError` if ``text`` references an optional capture that is
    not guard-dominated.

    ``is_template`` (the default) treats ``text`` as a ``{expr}`` template and only
    inspects its placeholder expressions; plain literal text (no ``{...}``)
    references no captures and is passed through untouched (so a literal ``symptom``
    like ``"Fabric manager lost connection to GPU"`` is never parsed as Python).
    Pass ``is_template=False`` for a bare expression (a ``when`` guard), where the
    whole string is a single expression.

    Recognized guards (Python short-circuit semantics): the right operands of
    ``C is not None and ...`` / ``C and ...`` (and-chains, where each conjunct may
    assume the earlier ones held); ``C or <default>``; the true-arm of
    ``X if C [is not None] else Y`` and the else-arm of ``X if C is None else Y``.
    A bare ``C`` (truthiness) or ``C is/is not/==/!= None`` in a test position is the
    guard itself, not a dangerous use.
    """
    exprs = _placeholder_exprs(text) if is_template else [text]
    for expr in exprs:
        tree = _parse_expr(expr)
        for name in _unguarded_names(tree):
            if name in optional:
                raise LoadError(
                    f"expression {expr!r} uses possibly-absent capture {name!r} without a "
                    f"guard (use `{name} is not None and ...`, `{name} or ...`, or a ternary)"
                )


def guards_established_by(expr: str) -> set[str]:
    """Names an expression proves non-``None`` when it is truthy.

    Used by the loader to let a leaf's own ``when`` dominate its templates.

    Example:
        >>> guards_established_by("name is not None")
        {'name'}
        >>> sorted(guards_established_by("a and b"))
        ['a', 'b']
    """
    tree = _parse_expr(expr)
    return _collect_true_guards(tree)


def _collect_true_guards(node: ast.AST) -> set[str]:
    """All names proven non-``None`` by ``node`` being truthy, including across ``and`` conjuncts."""
    g = _guards_true(node)
    match node:
        case ast.BoolOp(op=ast.And(), values=values):
            for v in values:
                g |= _collect_true_guards(v)
    return g


def check_template_syntax(template: str, *, where: str) -> None:
    """Raise :class:`LoadError` if ``template`` contains a stray ``{`` or ``}`` that is
    not part of a valid ``{{``, ``}}``, or ``{expr}`` sequence.

    Example:
        >>> check_template_syntax("{name}", where="test")
        >>> check_template_syntax("{", where="test")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        LoadError: ...
    """
    for kind, ch in _scan_template(template):
        if kind == "stray":
            raise LoadError(
                f"{where}: stray {ch!r} in template {template!r}; "
                f"use {{{{ / }}}} for literal braces or {{expr}} for placeholders"
            )


def _placeholder_exprs(text: str) -> list[str]:
    """Every ``{expr}`` placeholder expression inside a template (empty when the
    template is pure literal text).

    Example:
        >>> _placeholder_exprs("{count} errors")
        ['count']
        >>> _placeholder_exprs("no placeholders here")
        []
    """
    return [value for kind, value in _scan_template(text) if kind == "expr"]


def _free_names_in_template(template: str) -> set[str]:
    """Every ``Name`` id referenced in a template's ``{expr}`` placeholders.

    Collects ALL ``ast.Name`` ids (including a call's callee ``f`` in ``f(...)``);
    the caller subtracts the known-good capture/``datum``/``const``/helper/fixed/
    lifecycle names, so a placeholder like ``{valeu}`` whose name resolves to none of
    those is caught at LOAD instead of raising ``NameNotDefined`` at match. A
    ``const.<name>`` contributes only the base ``const`` (the ``<name>`` is an
    attribute, validated separately by :func:`check_consts_resolvable`).

    Example:
        >>> sorted(_free_names_in_template("{count} {name}"))
        ['count', 'name']
        >>> _free_names_in_template("plain text")
        set()
    """
    out: set[str] = set()
    for expr in _placeholder_exprs(template):
        out.update(node.id for node in ast.walk(_parse_expr(expr)) if isinstance(node, ast.Name))
    return out


def _unguarded_names(node: ast.AST, guarded: frozenset[str] = frozenset()) -> Iterator[str]:
    """Yield ``Name`` ids used in positions NOT dominated by a None/truthiness guard."""
    match node:
        case ast.Name(id=name_id):
            if name_id not in guarded:
                yield name_id
        case ast.Compare() if _is_none_test(node) is not None:
            # `C is/is not/==/!= None` only tests C; it never uses C dangerously.
            return
        case ast.BoolOp(op=ast.And(), values=values):
            # `A and B and C`: each operand is in test position; reaching a later operand
            # means the earlier ones were truthy, so their guards accumulate.
            g = set(guarded)
            for v in values:
                yield from _check_test(v, frozenset(g))
                g |= _guards_true(v)
        case ast.BoolOp(op=ast.Or(), values=values):
            # `A or B or default`: all but the last are tested for truthiness (their
            # truthy value is what propagates, so a bare optional there is safe); the
            # final operand is the fallback value and is checked normally. No guard
            # carries forward (reaching a later operand means earlier ones were falsy).
            *tested, last = values
            for v in tested:
                yield from _check_test(v, guarded)
            yield from _unguarded_names(last, guarded)
        case ast.IfExp(test=test, body=body, orelse=orelse):  # body if test else orelse
            yield from _check_test(test, guarded)
            g_true = guarded | _guards_true(test)
            g_false = guarded | _guards_false(test)
            yield from _unguarded_names(body, frozenset(g_true))
            yield from _unguarded_names(orelse, frozenset(g_false))
        case _:
            for child in ast.iter_child_nodes(node):
                yield from _unguarded_names(child, guarded)


def _check_test(node: ast.AST, guarded: frozenset[str]) -> Iterator[str]:
    """Check a sub-expression that appears in a boolean-test position.

    A bare ``Name`` here is a truthiness test (safe even if ``None``); anything else
    is checked normally (``_unguarded_names`` still treats a ``... is None`` compare as
    safe on its own).
    """
    match node:
        case ast.Name():
            return
        case _:
            yield from _unguarded_names(node, guarded)


def _is_none_test(node: ast.AST) -> str | None:
    """Return the ``Name`` id compared against ``None`` (any of is/is not/==/!=), else None."""
    match node:
        case ast.Compare(
            left=ast.Name(id=name_id),
            ops=[ast.Is() | ast.IsNot() | ast.Eq() | ast.NotEq()],
            comparators=[right],
        ) if _is_none_const(right):
            return name_id
        case ast.Compare(
            left=left,
            ops=[ast.Is() | ast.IsNot() | ast.Eq() | ast.NotEq()],
            comparators=[ast.Name(id=name_id)],
        ) if _is_none_const(left):
            return name_id
    return None


def _is_none_const(node: ast.AST) -> bool:
    """True when ``node`` is an AST ``Constant`` whose value is ``None``."""
    return isinstance(node, ast.Constant) and node.value is None


def _guards_true(node: ast.AST) -> set[str]:
    """Names proven non-``None`` when ``node`` is truthy (``C``, ``C is not None``, ``C != None``)."""
    match node:
        case ast.Name(id=name_id):
            return {name_id}
        case ast.Compare(
            left=ast.Name(id=name_id),
            ops=[ast.IsNot() | ast.NotEq()],
            comparators=[right],
        ) if _is_none_const(right):
            return {name_id}
        case ast.Compare(
            left=left,
            ops=[ast.IsNot() | ast.NotEq()],
            comparators=[ast.Name(id=name_id)],
        ) if _is_none_const(left):
            return {name_id}
    return set()


def _guards_false(node: ast.AST) -> set[str]:
    """Names proven non-``None`` when ``node`` is falsy (``C is None``, ``C == None``)."""
    match node:
        case ast.Compare(
            left=ast.Name(id=name_id),
            ops=[ast.Is() | ast.Eq()],
            comparators=[right],
        ) if _is_none_const(right):
            return {name_id}
        case ast.Compare(
            left=left,
            ops=[ast.Is() | ast.Eq()],
            comparators=[ast.Name(id=name_id)],
        ) if _is_none_const(left):
            return {name_id}
    return set()


def called_function_names(expr: str) -> set[str]:
    """Names invoked as a call (``f(...)``) whose callee is a bare ``Name``.

    A call through an attribute/subscript (``a.b()``) is ignored: only bare names
    are fixed functions or ``@expr_helper``s. Used by the load guard to reject a
    call to an unregistered helper.

    Example:
        >>> sorted(called_function_names("int(x) + len(y)"))
        ['int', 'len']
        >>> called_function_names("a.b()")
        set()
    """
    tree = _parse_expr(expr)
    return {
        func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(func := node.func, ast.Name)
    }


def check_helpers_resolvable(
    text: str,
    allowed: set[str],
    *,
    is_template: bool = True,
) -> None:
    """Raise :class:`LoadError` if ``text`` calls a function that is neither a
    fixed function nor a registered helper.

    ``allowed`` is the registered ``@expr_helper`` name set (the fixed functions
    are always allowed). ``is_template`` (default) inspects only the ``{expr}``
    placeholders of a template; pass ``is_template=False`` for a bare expression
    (a ``when`` guard / catalog ``key``)."""
    exprs = _placeholder_exprs(text) if is_template else [text]
    for expr in exprs:
        for fname in called_function_names(expr):
            if fname not in _FIXED_FUNCS and fname not in allowed and fname not in LIFECYCLE_PREDICATES:
                raise LoadError(
                    f"expression {expr!r} calls unknown function {fname!r}; "
                    f"register it with @expr_helper({fname!r}) or use a fixed function"
                )


def bind_functions(
    base: dict[str, Callable],
    injecting: dict[str, tuple[Callable[..., Any], frozenset[str]]] | None,
    source: Source,
    ctx: RuleContext,
) -> dict[str, Callable]:
    """The per-match ``functions`` map: ``base`` plus each injecting callable with
    its declared subset of ``{source, ctx}`` bound.

    ``injecting`` maps ``name -> (func, declared)`` where ``declared`` is the
    frozenset subset of ``{"source", "ctx"}`` to inject. Returns ``base`` UNCHANGED
    when ``injecting`` is empty (the fast path: a rule referencing no injecting
    callable rebuilds nothing per match)."""
    if not injecting:
        return base
    bound = dict(base)
    for name, (func, declared) in injecting.items():
        kwargs: dict[str, Any] = {}
        if "source" in declared:
            kwargs["source"] = source
        if "ctx" in declared:
            kwargs["ctx"] = ctx
        bound[name] = functools.partial(func, **kwargs)
    return bound


def called_names_in(
    text: str,
    *,
    is_template: bool = True,
) -> set[str]:
    """Every function name called in ``text``: a template's ``{expr}`` placeholders,
    or (``is_template=False``) the whole string as one bare expression. Public so the
    loader can find a rule's referenced injecting callables.

    Example:
        >>> called_names_in("{int(score)}")
        {'int'}
        >>> called_names_in("len(x)", is_template=False)
        {'len'}
    """
    exprs = _placeholder_exprs(text) if is_template else [text]
    out: set[str] = set()
    for e in exprs:
        out.update(called_function_names(e))
    return out


def referenced_const_names(expr: str) -> set[str]:
    """Attribute names read off a ``const`` base in a single expression.

    ``const.plr_x`` yields ``{"plr_x"}``. Only a ``const``-based attribute access
    is collected; any other attribute access is ignored here (the const guard
    only validates const names).

    Example:
        >>> referenced_const_names("const.plr_x + 1")
        {'plr_x'}
        >>> referenced_const_names("other.attr")
        set()
    """
    tree = _parse_expr(expr)
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(val := node.value, ast.Name) and val.id == "const"
    }


def check_consts_resolvable(
    text: str,
    allowed: set[str],
    *,
    is_template: bool = True,
) -> None:
    """Raise :class:`LoadError` if ``text`` reads a ``const.<name>`` whose name is
    not in ``allowed`` (the registered const-name set).

    ``is_template`` (default) inspects only the ``{expr}`` placeholders of a
    template; pass ``is_template=False`` for a bare expression (a ``when`` guard /
    catalog ``key``)."""
    exprs = _placeholder_exprs(text) if is_template else [text]
    for expr in exprs:
        for name in referenced_const_names(expr):
            if name not in allowed:
                raise LoadError(
                    f"expression {expr!r} reads unknown const {name!r}; "
                    f"define it in a [const] table or a @const provider"
                )
