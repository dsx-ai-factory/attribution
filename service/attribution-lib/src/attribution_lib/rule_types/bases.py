# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared bases for the per-rule-type loader and sub-engine pairs.

Every rule type contributes two classes: a loader that accumulates rules
during the load pass, and a sub-engine that executes them afterwards. The
split is the point: a loader is mutable and validates, a sub-engine is built
once and only runs.

``@rule_type`` records the pair in ``_RULE_TYPE_REGISTRY``, which is the only
place rule types are named. The coordinator dispatches through that registry
rather than through isinstance chains, so adding a rule type means adding one
module, not editing five.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import groupby
from typing import Any, ClassVar, Generic, Optional, TypeVar

from attribution_lib.exceptions import DuplicateRuleError
from attribution_lib.rule_types.definitions import ExecutionContext, Rule, RuleId, RuleLocation

R = TypeVar("R", bound=Rule)
P = TypeVar("P")


class SubEngineBase(Generic[R]):
    """Execution half of a rule-type pair: built once, then only read."""

    def __init__(self, rules: Optional[dict[RuleId, R]] = None) -> None:
        self.rules: dict[RuleId, R] = dict(rules) if rules is not None else {}

    def handles(self, context: ExecutionContext) -> bool:
        """True when execute() will act on this context. Subclasses override.

        The coordinator uses this to detect a context no sub-engine claims,
        which would otherwise be a silent no-op.
        """
        return False

    def execute(self, context: ExecutionContext) -> None:
        """Execute against context. Subclasses override; default is a no-op."""

    def extractor_hash_contribution(self) -> str:
        """Stable hash string for extraction cache keying. Empty means no contribution."""
        return ""

    def attribution_hash_contribution(self) -> bytes:
        """Stable bytes for attribution cache keying. Empty means no contribution."""
        return b""

    def extra_rules(self) -> Mapping[RuleId, Rule]:
        """Additional rules beyond self.rules (e.g. extractor-accumulator rules)."""
        return {}

    def extra_rule_names(self) -> set:
        """Additional declared rule names beyond self.rules.keys()."""
        return set()


class LoaderBase(Generic[R]):
    """Load half of a rule-type pair: mutable, validates, then builds an engine."""

    engine_class: ClassVar[type[SubEngineBase]]

    def __init__(self) -> None:
        self.rules: dict[RuleId, R] = {}

    def _check_duplicate(self, name: RuleId, location: RuleLocation) -> None:
        """Reject a name already registered in this loader.

        Cross-type duplicates are the coordinator's job; it scans every loader
        before dispatching.
        """
        existing = self.rules.get(name)
        if existing is not None:
            raise DuplicateRuleError(
                f"Duplicate rule '{name}': "
                f"first at {existing.location.file}:{existing.location.line}, "
                f"duplicate at {location.file}:{location.line}"
            )

    def register(self, rule: R) -> None:
        self._check_duplicate(rule.name, rule.location)
        self.rules[rule.name] = rule

    def find_by_name(self, name: RuleId) -> Optional[Rule]:
        """Return the rule registered under ``name``, or None if absent."""
        return self.rules.get(name)

    def extra_rules(self) -> Mapping[RuleId, Rule]:
        """Additional rules beyond ``self.rules`` held by this loader."""
        return {}

    def snapshot(self) -> dict[str, Any]:
        """Capture every field a load may mutate, for rollback on failure."""
        return {"rules": dict(self.rules)}

    def restore(self, snap: dict[str, Any]) -> None:
        self.rules = dict(snap["rules"])

    def build(self) -> SubEngineBase:
        raise NotImplementedError

    def declarative_emit_store(self) -> Optional[dict[str, str]]:
        """Return this loader's declarative_emit dict, or None if it has no such concept."""
        return None


class OrderedSubEngine(SubEngineBase[R], Generic[R, P]):
    """Sub-engine base for rule types with priority-ordered execution (Filter, Refiner).

    Sorting and phase bucketing live here rather than on the loader: they are
    only meaningful at execution time, and the loader would have to redo them
    on every registration.
    """

    def __init__(
        self,
        rules: Optional[dict[RuleId, R]] = None,
        execution_order: Optional[list[tuple[int, int, str, P]]] = None,
    ) -> None:
        super().__init__(rules)
        self._execution_order: list[tuple[int, int, str, P]] = (
            list(execution_order) if execution_order is not None else []
        )
        self._sorted: bool = len(self._execution_order) == 0
        self._by_phase: dict[int, list[tuple[str, P]]] = {}

    def _ensure_sorted(self) -> None:
        if self._sorted is False:
            self._execution_order.sort(key=lambda e: (e[0], e[1]))
            self._by_phase = {
                priority: [(name, proc) for _, _, name, proc in group]
                for priority, group in groupby(self._execution_order, key=lambda e: e[0])
            }
            self._sorted = True


class OrderedLoader(LoaderBase[R], Generic[R, P]):
    """Loader base for rule types with priority-ordered execution.

    Holds only the order rules were registered in. The second element of each
    entry is the registration sequence, which makes the engine's sort stable
    for rules sharing a priority.
    """

    def __init__(self) -> None:
        super().__init__()
        self._execution_order: list[tuple[int, int, str, P]] = []

    def _add_to_order(self, priority: int, name: str, processor: P) -> None:
        self._execution_order.append((priority, len(self._execution_order), name, processor))

    def snapshot(self) -> dict[str, Any]:
        return {**super().snapshot(), "_execution_order": list(self._execution_order)}

    def restore(self, snap: dict[str, Any]) -> None:
        super().restore(snap)
        self._execution_order = list(snap["_execution_order"])


@dataclass(frozen=True)
class RuleTypeEntry:
    """Links a set of Rule subclasses to their loader and engine classes."""

    rule_types: tuple[type[Rule], ...]
    loader_class: type[LoaderBase]
    engine_class: type[SubEngineBase]


_RULE_TYPE_REGISTRY: list[RuleTypeEntry] = []


def rule_type(*rule_types: type[Rule]) -> Callable[[type[LoaderBase]], type[LoaderBase]]:
    """Class decorator: registers a loader and its engine for the given rule types.

    This is the only place a rule type is named outside its own module. The
    coordinator builds its dispatch table from the registry, so nothing else
    needs to know the set of rule types.
    """

    def decorator(loader_class: type[LoaderBase]) -> type[LoaderBase]:
        registered_rule_types = {
            registered_rule_type for entry in _RULE_TYPE_REGISTRY for registered_rule_type in entry.rule_types
        }
        duplicates = registered_rule_types.intersection(rule_types)
        if duplicates:
            names = ", ".join(sorted(rule_type_.__qualname__ for rule_type_ in duplicates))
            raise DuplicateRuleError(f"Rule type already registered: {names}")
        _RULE_TYPE_REGISTRY.append(RuleTypeEntry(rule_types, loader_class, loader_class.engine_class))
        return loader_class

    return decorator
