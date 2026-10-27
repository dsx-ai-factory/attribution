# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load-time state and registration decorators for declarative (TOML) rules.

Holds the ContextVars that scope a rule-module load, and the four provider
decorators a Python rule file uses to publish helpers, constants, record
tables, and whole-rule processors into the active rule loader.
"""

import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, TypeAlias

from attribution_lib.declarative.expressions import _FIXED_FUNCS
from attribution_lib.declarative.lifecycle import INJECTABLE_PARAMS, LIFECYCLE_PREDICATES
from attribution_lib.exceptions import RuleIngestionError

if TYPE_CHECKING:
    from attribution_lib.rules_engine import RuleLoader

# Zero-arg provider returning a flat {name: value} or {key: record} dict.
_NamespaceProvider: TypeAlias = Callable[[], dict[str, Any]]
# Decorator that accepts and returns a _NamespaceProvider unchanged (init-scoped @const / @rows).
_ProviderDecorator: TypeAlias = Callable[[_NamespaceProvider], _NamespaceProvider]


# Context-local state for a rule-module load.
_module_globals_context: ContextVar[dict[str, Any]] = ContextVar("_module_globals_context")
_rule_loader_context: ContextVar["RuleLoader"] = ContextVar("_rule_loader_context")

# Fixed-function names an @expr_helper may not shadow.
_EXPR_HELPER_RESERVED = frozenset(_FIXED_FUNCS)

# Parameters the engine injects into an @expr_helper that declares them.
EXPR_HELPER_INJECTED_PARAMS: tuple[str, ...] = INJECTABLE_PARAMS


@contextmanager
def _using_module_globals(module_globals: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Make a rule module's globals available within this load scope."""
    token = _module_globals_context.set(module_globals)
    try:
        yield module_globals
    finally:
        _module_globals_context.reset(token)


def _current_module_globals() -> dict[str, Any]:
    """Return the scoped rule-module globals, or an empty mapping outside a load."""
    try:
        return _module_globals_context.get()
    except LookupError:
        return {}


@contextmanager
def _using_rule_loader(loader: "RuleLoader") -> Iterator["RuleLoader"]:
    """Make ``loader`` active for registrations within this scope."""
    token = _rule_loader_context.set(loader)
    try:
        yield loader
    finally:
        _rule_loader_context.reset(token)


def _require_rule_loader(operation: str) -> "RuleLoader":
    """Return the scoped loader or raise a registration-specific error."""
    try:
        return _rule_loader_context.get()
    except LookupError:
        raise RuleIngestionError(f"No active rule loader when {operation}") from None


def _optional_rule_loader() -> Optional["RuleLoader"]:
    """Return the scoped loader when registration is intentionally optional."""
    try:
        return _rule_loader_context.get()
    except LookupError:
        return None


@dataclass
class DeclarativeContext:
    """State a load pass accumulates for declarative (TOML) rules.

    The helpers, constants, row tables and named processors that TOML rules
    reference in their when/emit expressions. Lives on RuleLoader and is read
    by the declarative loader while compiling each file. Has no connection to
    the execution-time RuleEngine.
    """

    expr_helpers: dict[str, Callable[..., Any]] = field(default_factory=dict)
    expr_helper_injections: dict[str, frozenset] = field(default_factory=dict)
    const_values: dict[str, object] = field(default_factory=dict)
    record_tables: dict[str, dict] = field(default_factory=dict)
    rule_processors: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "expr_helpers": dict(self.expr_helpers),
            "expr_helper_injections": dict(self.expr_helper_injections),
            "const_values": dict(self.const_values),
            "record_tables": dict(self.record_tables),
            "rule_processors": dict(self.rule_processors),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.expr_helpers = dict(snap["expr_helpers"])
        self.expr_helper_injections = dict(snap["expr_helper_injections"])
        self.const_values = dict(snap["const_values"])
        self.record_tables = dict(snap["record_tables"])
        self.rule_processors = dict(snap["rule_processors"])


def expr_helper(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a Python function as a named expression helper, callable bare
    inside a declarative ``when`` / symptom / payload expression (e.g.
    ``is_user_path(path)``).

    Init-scoped like :func:`declare_rule_names`: it registers into the current
    loader during the load pass and is rebuilt on each ``setup_rules`` (a fresh
    ``RuleLoader`` per init). The registration NAME is independent of the
    function's ``__name__``; the function is returned unchanged so it stays
    directly callable in Python (a helper shared with un-migrated Python rules
    keeps working).

    A helper may declare ``source`` and/or ``ctx`` parameters; the engine injects
    the current :class:`Source` / :class:`RuleContext` for whichever it declares
    (see :mod:`attribution_lib.declarative.lifecycle`). Every other argument comes
    from the calling expression.

    Raises :class:`RuleIngestionError` if there is no active loader, the name
    shadows a fixed function, or the name is already registered in this pass.
    """

    def register(func: Callable[..., Any]) -> Callable[..., Any]:
        loader = _require_rule_loader(f"registering expr_helper {name!r}")
        if name in _EXPR_HELPER_RESERVED or name in LIFECYCLE_PREDICATES:
            raise RuleIngestionError(
                f"expr_helper {name!r} shadows a fixed function or lifecycle predicate; choose another name"
            )
        existing = loader.declarative_ctx.expr_helpers.get(name)
        if existing is not None:
            raise RuleIngestionError(
                f"expr_helper {name!r} is already registered by "
                f"{getattr(existing, '__module__', '?')}:{getattr(existing, '__qualname__', repr(existing))}; "
                f"duplicate at {getattr(func, '__module__', '?')}:{getattr(func, '__qualname__', repr(func))}"
            )
        sig = inspect.signature(func)
        for pname in EXPR_HELPER_INJECTED_PARAMS:
            param = sig.parameters.get(pname)
            if param is not None and param.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise RuleIngestionError(
                    f"expr_helper {name!r}: parameter {pname!r} is positional-only but "
                    f"the engine injects it as a keyword argument; remove the '/' separator"
                )
        loader.declarative_ctx.expr_helpers[name] = func
        loader.declarative_ctx.expr_helper_injections[name] = frozenset(
            p for p in EXPR_HELPER_INJECTED_PARAMS if p in sig.parameters
        )
        return func

    return register


def const(ns: str) -> _ProviderDecorator:
    """Register a `@const` provider: a zero-arg function returning a flat
    ``{name: value}`` dict merged into the loader's single ``const`` namespace.

    The ``ns`` argument names the PROVIDER (for error messages), not a
    sub-namespace: a rule reads ``const.<name>``, never ``const.<ns>.<name>``.
    Init-scoped like :func:`expr_helper`; the provider is called at registration
    (its values are static for v1). A ``const`` name defined more than once
    (across two providers, or a provider and an inline ``[const]`` table) is a
    :class:`RuleIngestionError`. The provider function is returned unchanged.
    """

    def register(func: _NamespaceProvider) -> _NamespaceProvider:
        loader = _require_rule_loader(f"registering @const provider {ns!r}")
        values = func()
        if not isinstance(values, dict):
            raise RuleIngestionError(
                f"@const provider {ns!r} must return a dict of {{name: value}}, got {type(values).__name__}"
            )
        for name, value in values.items():
            if name in loader.declarative_ctx.const_values:
                raise RuleIngestionError(
                    f"const name {name!r} from provider {ns!r} is already defined; "
                    f"const names are a single flat namespace"
                )
            loader.declarative_ctx.const_values[name] = value
        return func

    return register


def rows(name: str) -> _ProviderDecorator:
    """Register a `@rows` provider: a zero-arg function returning a ``{key: record}``
    dict that becomes a record table named ``name`` (usable by a ``catalog_lookup``
    ``project`` exactly like an inline ``[rows.<name>]`` TOML table).

    Init-scoped like :func:`const`; the provider is called at registration (its
    records are static for v1). Record keys are ``str()``-coerced so an int key
    from Python matches a string TOML lookup key (mirrors ``parse_tables``). A
    table ``name`` registered by more than one ``@rows`` provider is a
    :class:`RuleIngestionError`; a collision with an inline ``[rows.<name>]`` or a
    ``[catalog.<name>]`` is caught at load. The provider function is returned
    unchanged."""

    def register(func: _NamespaceProvider) -> _NamespaceProvider:
        loader = _require_rule_loader(f"registering @rows provider {name!r}")
        if name in loader.declarative_ctx.record_tables:
            raise RuleIngestionError(f"record table {name!r} is already registered by another @rows provider")
        records = func()
        if not isinstance(records, dict):
            raise RuleIngestionError(
                f"@rows provider {name!r} must return a dict of {{key: record}}, got {type(records).__name__}"
            )
        loader.declarative_ctx.record_tables[name] = {str(k): v for k, v in records.items()}
        return func

    return register


def rule_processor(ns_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a whole-rule Python processor under a NAMESPACED name (``ns:name``,
    e.g. ``applog:pushrecv``), referenced by a declarative stanza's
    ``processor = "ns:name"``. The function is an ordinary extractor body; the
    declarative loader wraps it to inject the ``{node, source, ctx}`` it declares.

    Init-scoped like the other providers. A duplicate ``ns:name`` is a
    :class:`RuleIngestionError`. The function is returned unchanged."""

    def register(func: Callable[..., Any]) -> Callable[..., Any]:
        loader = _require_rule_loader(f"registering @rule_processor {ns_name!r}")
        if ns_name in loader.declarative_ctx.rule_processors:
            existing = loader.declarative_ctx.rule_processors[ns_name]
            raise RuleIngestionError(
                f"@rule_processor {ns_name!r} is already registered by {getattr(existing, '__module__', '?')}:{getattr(existing, '__qualname__', repr(existing))}"
            )
        loader.declarative_ctx.rule_processors[ns_name] = func
        return func

    return register
