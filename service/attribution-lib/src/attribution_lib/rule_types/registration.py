# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Decorators that register rules from a rule file into the active loader."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import currentframe
from typing import Any, Optional

from attribution_lib.declarative.context import _current_module_globals, _require_rule_loader
from attribution_lib.exceptions import RuleIngestionError
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rule_types.definitions import (
    CollectorFormatter,
    CollectorProcessor,
    CollectorRule,
    ExtractorAccumulator,
    ExtractorAccumulatorRule,
    FilterPriority,
    FilterProcessor,
    FilterRule,
    RefinerProcessor,
    RefinerRule,
    Rule,
    RuleId,
    RuleLocation,
    RuleParamsBase,
)


def _get_caller_location(skip_files: frozenset[str] = frozenset()) -> RuleLocation:
    """Return the file/line of the rule's definition site.

    The rule-registration helpers all live in this module, so walk past every
    frame inside this file and return the first caller outside it — i.e. where
    the rule is actually defined (a ``bundled_rules/*.py`` file, or the generator
    loop that mints it), not the decorator machinery here.
    """
    frame = currentframe()
    if frame is None:
        raise RuleIngestionError("Cannot determine call frame")

    all_skip = skip_files | {frame.f_code.co_filename}
    frame = frame.f_back
    # Skip the registration machinery's own frames; the first frame outside
    # them is the definition site.
    while frame is not None and frame.f_code.co_filename in all_skip:
        frame = frame.f_back
    if frame is None:
        raise RuleIngestionError("Cannot access caller frame")

    return RuleLocation(frame.f_code.co_filename, frame.f_lineno)


def normalize_tags(tags: Any = None, *, where: str) -> tuple[str, ...]:
    """Validate, dedupe and sort tag values into a stable immutable tuple.

    Tags are free-form labels; the only semantic rule is that a single rule cannot be
    tagged both "internal" and "external". The check is exact and case-sensitive.

    Args:
        tags: Declared tag values, as written in a rule file and therefore unvalidated.
            None or omitted means untagged.
        where: Rule name, plus the defining file when known, to name in error messages.

    Returns:
        The normalized tags as a tuple of strings.

    Raises:
        RuleIngestionError: If the tags are invalid.
    """
    if tags is None:
        return ()

    # str is the one sequence to reject: iterating it would yield one tag per character.
    if isinstance(tags, str) or not isinstance(tags, Sequence):
        raise RuleIngestionError(f"{where}: tags must be a list of strings, got {type(tags).__name__} ({tags!r})")

    cleaned: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            raise RuleIngestionError(f"{where}: every tag must be a string, got {type(tag).__name__} ({tag!r})")
        stripped = tag.strip()
        if stripped == "":
            raise RuleIngestionError(f"{where}: tags must not be empty or whitespace-only, got {tag!r}")
        cleaned.add(stripped)

    # Only up to one tag in each mutually exclusive tag set is allowed.
    mutually_exclusive_tag_sets = ({"internal", "external"},)
    for exclusive_tags in mutually_exclusive_tag_sets:
        exclusive_tags_present = exclusive_tags.intersection(cleaned)
        if len(exclusive_tags_present) > 1:
            raise RuleIngestionError(
                f"{where}: tags {sorted(exclusive_tags_present)} are mutually exclusive; declare only one of them"
            )

    return tuple(sorted(cleaned))


@dataclass(frozen=True, slots=True)
class _Registration:
    """Metadata and final registration shared by every ``*_def`` decorator."""

    name: RuleId
    declared_author: Optional[str]
    location: RuleLocation
    derived_rules: tuple[str, ...]
    derived_event_types: tuple[str, ...]
    tags: tuple[str, ...]

    @classmethod
    def prepare(
        cls,
        name: RuleId,
        author: Optional[str],
        derived_rules: Sequence[str],
        derived_event_types: Sequence[str],
        tags: Sequence[str],
    ) -> "_Registration":
        location = _get_caller_location()
        return cls(
            name=name,
            declared_author=author,
            location=location,
            derived_rules=tuple(derived_rules),
            derived_event_types=tuple(derived_event_types),
            tags=normalize_tags(tags, where=f"{name} ({location.file}:{location.line})"),
        )

    def resolve_author(self, module_globals: Optional[dict[str, Any]] = None) -> str:
        if module_globals is None:
            module_globals = _current_module_globals()
        author: str = self.declared_author or module_globals.get("rule_author", "unknown")
        return author

    def register(self, rule: Rule) -> None:
        loader = _require_rule_loader(f"defining rule '{self.name}'")
        loader.register_rule(rule)
        if self.derived_rules:
            loader.declared_rule_names.update(self.derived_rules)
        if self.derived_event_types:
            loader.declared_event_types.update(self.derived_event_types)


def declare_rule_names(*names: str) -> None:
    """Declare synthetic rule names that will be used at runtime.

    Call at module scope in rule files to register rule names that aren't
    associated with a registered rule (e.g. names used for cascade tracking
    in filter/refiner processors). These names are pre-populated into the
    EventManager's lookup tables at finalize time.

    Multiple calls (including from different rule files) with the same name
    are safe — names are collected as a set union.
    """
    loader = _require_rule_loader("declaring rule names")
    loader.declared_rule_names.update(names)


def declare_event_types(*names: str) -> None:
    """Declare event type names that will be emitted at runtime.

    Call at module scope in rule files to register event types that aren't
    co-located with a decorator (e.g. types emitted by library helpers like
    ``_bad()`` or ``_other_event()``).

    Multiple calls (including from different rule files) with the same name
    are safe — names are collected as a set union.
    """
    loader = _require_rule_loader("declaring event types")
    loader.declared_event_types.update(names)


def filter_def(
    name: RuleId,
    author: Optional[str] = None,
    priority: int = FilterPriority.STANDARD,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> Callable[[FilterProcessor], FilterProcessor]:
    """Define a filter rule. No regex needed.

    Args:
        name: Unique rule identifier.
        author: Rule author. Falls back to module-level rule_author if not set.
        priority: Execution priority. Lower values execute first.
        derived_rules: Rule names produced by source.derived(rule=...) inside
            this rule's processor. Declaring them here registers them automatically.
        derived_event_types: Event type names emitted by this rule's processor.
        tags: Free-form classification labels, deduped and sorted at declaration.
    """
    registration = _Registration.prepare(name, author, derived_rules, derived_event_types, tags)

    def _register(proc: FilterProcessor) -> FilterProcessor:
        rule = FilterRule(
            name=name,
            processor=proc,
            author=registration.resolve_author(),
            location=registration.location,
            tags=registration.tags,
            priority=priority,
        )
        registration.register(rule)
        return proc

    return _register


def extractor_accumulator_def(
    name: RuleId,
    match_re: str,
    source_filters: Optional[set[ObservationSource]] = None,
    author: Optional[str] = None,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> Callable[[type[ExtractorAccumulator]], type[ExtractorAccumulator]]:
    """Define an extractor-accumulator rule. Used as a class decorator on ExtractorAccumulator subclasses.

    Lines are only passed to the accumulator when match_re matches,
    unless the accumulator's is_consuming property returns True.

    Args:
        derived_rules: Rule names produced by source.derived(rule=...) inside
            this rule's processor. Declaring them here registers them automatically.
        derived_event_types: Event type names emitted by this rule's processor.
        tags: Free-form classification labels, deduped and sorted at declaration.
    """
    registration = _Registration.prepare(name, author, derived_rules, derived_event_types, tags)

    def _register(cls: type[ExtractorAccumulator]) -> type[ExtractorAccumulator]:
        mod_globals = _current_module_globals()

        final_source_filters = source_filters if source_filters is not None else mod_globals.get("rule_sources", set())

        rule = ExtractorAccumulatorRule(
            name=name,
            match_re=match_re,
            factory=cls,
            source_filters=frozenset(final_source_filters),
            author=registration.resolve_author(mod_globals),
            location=registration.location,
            tags=registration.tags,
        )
        registration.register(rule)
        return cls

    return _register


def refiner_def(
    name: RuleId,
    author: Optional[str] = None,
    priority: int = FilterPriority.FINAL,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> Callable[[RefinerProcessor], RefinerProcessor]:
    """Define a refiner rule (runs on materialized symptoms / my_bad_nodes).

    Default priority is FINAL (runs in the final phase, after event stream).
    Ordering is among refiner rules only (not mixed with filters).

    Args:
        name: Unique rule identifier.
        author: Rule author. Falls back to module-level rule_author if not set.
        priority: Execution priority starts from FINAL
        derived_rules: Rule names produced by source.derived(rule=...) inside
            this rule's processor. Declaring them here registers them automatically.
        derived_event_types: Event type names emitted by this rule's processor.
        tags: Free-form classification labels, deduped and sorted at declaration.
    """
    registration = _Registration.prepare(name, author, derived_rules, derived_event_types, tags)

    def _register(proc: RefinerProcessor) -> RefinerProcessor:
        rule = RefinerRule(
            name=name,
            processor=proc,
            author=registration.resolve_author(),
            location=registration.location,
            tags=registration.tags,
            priority=priority,
        )
        registration.register(rule)
        return proc

    return _register


def collector_def(
    name: RuleId,
    formatter: CollectorFormatter,
    author: Optional[str] = None,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
    params_model: Optional[type[RuleParamsBase]] = None,
    tags: Sequence[str] = (),
) -> Callable[[CollectorProcessor], CollectorProcessor]:
    """Define a collector rule (post-attribution digest collection).

    Collectors query the event stream after attribution and produce
    digest key-value pairs for the report.

    Args:
        name: Unique rule identifier.
        formatter: Function that converts raw collector output to digest dict.
        author: Rule author. Falls back to module-level rule_author if not set.
        derived_rules: Rule names produced by source.derived(rule=...) inside
            this rule's processor. Declaring them here registers them automatically.
        derived_event_types: Event type names emitted by this rule's processor.
        params_model: Optional Pydantic model declaring the schema of per-request
            params for this collector.
        tags: Free-form classification labels, deduped and sorted at declaration.
    """
    registration = _Registration.prepare(name, author, derived_rules, derived_event_types, tags)

    def _register(proc: CollectorProcessor) -> CollectorProcessor:
        rule = CollectorRule(
            name=name,
            processor=proc,
            formatter=formatter,
            author=registration.resolve_author(),
            location=registration.location,
            tags=registration.tags,
            params_model=params_model,
        )
        registration.register(rule)
        return proc

    return _register
