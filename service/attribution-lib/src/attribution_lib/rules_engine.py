# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import hashlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from inspect import currentframe
from itertools import groupby
from pathlib import Path
from re import Match
from typing import Any, NamedTuple, Optional, Sequence, TypeAlias, Union, cast, overload

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from attribution_lib.decision_trace_recorder import DecisionRecorder, NodeSymptomSnapshot, snapshot_bad_nodes
from attribution_lib.events import EventStream
from attribution_lib.exceptions import (
    DuplicateRuleError,
    InvalidRuleParams,
    RuleAcceptsNoParamsError,
    RuleIngestionError,
    UninitializedRuleEngineError,
    UnknownResourceError,
    UnknownRuleTypeError,
)
from attribution_lib.job_info import JobInfo
from attribution_lib.logline_filtering import log_ignored_node_line, log_ignored_time_line
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.observation.processing import EntryProtocol, EntrySourceBase
from attribution_lib.regex_accelerator.two_stage_matcher import TwoStageMatcher
from attribution_lib.resources import get_single_node
from attribution_lib.sources import Source, SourceSet
from attribution_lib.symptoms.declarations import OutcomePriority, SymptomDeclaration


class EventTriple(NamedTuple):
    """Output of a rule"""

    type: str
    payload: dict[str, Any]
    source: Source


class RuleLocation(NamedTuple):
    """Where a rule was defined"""

    file: str
    line: int


@dataclass(slots=True)
class RuleContext:
    """Base context available to rule processors.

    Extractors are write-only with respect to events - they produce events
    but don't read from the event stream. Hence no event_stream here.
    """

    job_info: Optional[Any] = None
    hostname: Optional[str] = None

    def get_node_from_rank(self, rank: int) -> Optional[str]:
        """Resolve a rank number to a node hostname."""
        if self.job_info is None:
            return None
        result: Optional[str] = self.job_info.get_name_from_rank(rank)
        return result

    @property
    def job_script_content(self) -> Optional[str]:
        """Get job script content for parameter checking."""
        if self.job_info is None:
            return None
        content: Optional[str] = self.job_info.job_script_content
        return content


@dataclass(slots=True)
class FilterContext(RuleContext):
    """Context for filter rule processors (event-stream phase).

    Filters run during from_event_stream() and query/modify the event stream.
    """

    event_stream: Optional[EventStream] = None


@dataclass(slots=True)
class RefinerContext(RuleContext):
    """Context for refiner rule processors (post–event-stream phase).

    Refiner rules run during calculate_symptoms() after events are consumed.
    They operate on materialized symptoms and my_bad_nodes. event_stream
    is still provided for non-attribution queries (e.g. pytorch timeout, bad pairs).
    """

    event_stream: Optional[EventStream] = None
    symptoms: Optional[Any] = None  # SymptomSet[str]
    my_bad_nodes: Optional[Any] = None  # JobNodeAttributions
    raw_keywords: Optional[Any] = None  # SymptomSet[str]
    extract_job_metadata_from_logfile: bool = False
    # Populated by DagCascadePruning refiner for downstream reporting
    node_symptoms_base: Optional[dict] = None
    skipped_symptom_ids: Optional[set] = None
    # Optional recorder for decision graph tracing
    recorder: Optional[DecisionRecorder] = None
    # Populated for POST_OUTCOME refiners (after calculate_outcome)
    outcome: Optional["SourceSet"] = None
    cancelled_by: Optional[str] = None
    attribution_result: Optional[Any] = None
    bad_node_result: Optional[Any] = None


RuleId: TypeAlias = str


@dataclass(frozen=True, slots=True)
class RuleParameters:
    """Generic envelope for rule parameters."""

    params: dict[str, Any] = field(default_factory=dict)


class RuleParamsBase(BaseModel):
    """Base class for all rule ``params_model`` declarations.

    Frozen (immutable + content-hashable) and strict (unknown fields raise).
    Every collector/extractor/filter/refiner params class should inherit this.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


ExtractorProcessor: TypeAlias = Callable[[dict[str, str], Optional[str], Source, RuleContext], list[EventTriple]]

FilterProcessor: TypeAlias = Callable[[FilterContext], None]

RefinerProcessor: TypeAlias = Callable[[RefinerContext], None]

CollectorProcessor: TypeAlias = Callable[[EventStream, "JobInfo", Optional[RuleParamsBase]], Any]
CollectorFormatter: TypeAlias = Callable[[Any], dict[str, Any]]


class FilterPriority(IntEnum):
    """Shared execution priority. Lower values execute first.

    EARLY..LATE: filter rules (during from_event_stream()).
    SCRUB: per-node scrubbing refiners (before aggregation into job-level symptoms).
    FINAL: refiner rules (during calculate_symptoms(), after aggregation).
    POST_OUTCOME: refiner rules that run after outcome calculation.
    """

    EARLY = 100
    MID = 500
    STANDARD = 1000
    LATE = 1500
    SCRUB = 1900
    FINAL = 2000
    POST_OUTCOME = 3000


@dataclass(kw_only=True, slots=True)
class RuleDefinitionBase:
    """Base definition containing shared metadata for all rules."""

    name: RuleId
    author: Optional[str] = None
    location: Optional[RuleLocation] = None
    derived_rules: tuple[str, ...] = ()
    """Rule names produced by source.derived(rule=...) inside this rule's processor.

    These are secondary rule names that differ from the rule's own name. Declaring
    them here keeps them co-located with the rule definition and allows them to be
    registered automatically during load.
    """
    derived_event_types: tuple[str, ...] = ()
    """Event type names emitted by this rule's processor via EventTriple(type=...).

    Declaring them here keeps them co-located with the rule definition and allows
    them to be registered automatically during load.
    """


@dataclass(kw_only=True, slots=True)
class ExtractorDefinition(RuleDefinitionBase):
    """Definition for an extractor rule (input to decorator)."""

    match_re: str
    processor: ExtractorProcessor
    source_filters: set[ObservationSource] = field(default_factory=set)


@dataclass(kw_only=True, slots=True)
class FilterDefinition(RuleDefinitionBase):
    """Definition for a filter rule (input to decorator)."""

    processor: FilterProcessor
    priority: int = FilterPriority.STANDARD


@dataclass(kw_only=True, slots=True)
class ExtractorAccumulatorDefinition(RuleDefinitionBase):
    """Definition for an extractor-accumulator rule (input to decorator)."""

    match_re: str
    factory: Callable[[], "ExtractorAccumulator"]
    source_filters: set[ObservationSource] = field(default_factory=set)


@dataclass(kw_only=True, slots=True)
class RefinerDefinition(RuleDefinitionBase):
    """Definition for a refiner rule (input to decorator)."""

    processor: RefinerProcessor
    priority: int = FilterPriority.FINAL


@dataclass(kw_only=True, slots=True)
class CollectorDefinition(RuleDefinitionBase):
    """Definition for a collector rule (post-attribution digest collection)."""

    processor: CollectorProcessor
    formatter: CollectorFormatter
    params_model: Optional[type[RuleParamsBase]] = None


@dataclass(frozen=True, slots=True)
class Rule(ABC):
    """Base rule containing shared rule metadata."""

    name: RuleId
    author: str
    location: RuleLocation

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Process the rule. Signature varies by subtype."""
        ...


@dataclass(frozen=True, slots=True)
class ExtractorRule(Rule):
    """Single rule that matches and extracts"""

    match_re: str
    processor: ExtractorProcessor
    source_filters: frozenset[ObservationSource] = field(default_factory=frozenset)

    @classmethod
    def from_definition(cls, defn: ExtractorDefinition) -> "ExtractorRule":
        """Create an ExtractorRule from an ExtractorDefinition."""
        return cls(
            name=defn.name,
            author=defn.author or "unknown",
            location=defn.location or RuleLocation("unknown", 0),
            match_re=defn.match_re,
            processor=defn.processor,
            source_filters=frozenset(defn.source_filters),
        )

    def process(self, match: Match, source: Source, context: RuleContext) -> list[EventTriple]:
        """Try to match and extract from source"""
        events = self.processor(match.groupdict(), context.hostname, source, context)

        # Use the source from the processor if it set a rule, otherwise derive from input source
        return [
            EventTriple(
                type=evt.type,
                payload=evt.payload,
                source=evt.source if evt.source.rule else source.derived(rule=self.name),
            )
            for evt in events
        ]


@dataclass(frozen=True, slots=True)
class FilterRule(Rule):
    """Rule that transforms/filters the event stream.

    Attributes:
        priority: Execution priority. Lower values execute first.
            Default is FilterPriority.STANDARD.  Priority conventions:
            100-499=early cleanup, 500-999=mid, 1000=standard, 1001-1999=late.
    """

    processor: FilterProcessor
    priority: int = FilterPriority.STANDARD

    @classmethod
    def from_definition(cls, defn: FilterDefinition) -> "FilterRule":
        """Create a FilterRule from a FilterDefinition."""
        return cls(
            name=defn.name,
            author=defn.author or "unknown",
            location=defn.location or RuleLocation("unknown", 0),
            processor=defn.processor,
            priority=defn.priority,
        )

    def process(self, context: FilterContext) -> None:
        """Process the filter on the event stream."""
        self.processor(context)


class ExtractorAccumulator(ABC):
    """Stateful per-line processor created by a factory when a trigger regex matches.

    An extractor-accumulator is a type of extractor that needs multiple lines
    to produce its events. Each instance is created by its rule's factory when
    the trigger regex fires. The instance then receives **every subsequent
    line** via ``process_line()`` until ``is_consuming`` returns False,
    at which point ``finalize()`` is called and the instance is removed.

    The trigger regex stays active and can spawn overlapping instances.
    At end-of-stream any still-consuming instances are force-finalized.

    Lifecycle:
        1. Trigger regex matches -> factory creates a new instance.
        2. ``start(source)`` is called with the triggering line.
        3. Every subsequent line is passed to ``process_line()``.
        4. When ``is_consuming`` becomes False -> ``finalize()`` is called,
           events are emitted, instance is discarded.
        5. At end-of-stream, remaining instances are finalized.
    """

    def start(self, source: Source) -> None:
        """Called with the triggering line. Default: delegates to process_line."""
        self.process_line(source)

    @abstractmethod
    def process_line(self, source: Source) -> None:
        """Process a line while consuming."""
        ...

    @property
    @abstractmethod
    def is_consuming(self) -> bool:
        """True while this instance wants to keep receiving lines."""
        ...

    @abstractmethod
    def finalize(self) -> list[EventTriple]:
        """Emit accumulated events. Called when is_consuming->False or at end-of-stream."""
        ...


@dataclass(frozen=True, slots=True)
class ExtractorAccumulatorRule(Rule):
    """Factory rule: trigger regex fires -> factory creates new ExtractorAccumulator instance.

    The instance receives all subsequent lines until it finishes. Overlapping
    instances are allowed (trigger can fire while another instance is active).
    """

    match_re: str
    factory: Callable[[], ExtractorAccumulator]
    source_filters: frozenset[ObservationSource] = field(default_factory=frozenset)

    @classmethod
    def from_definition(cls, defn: ExtractorAccumulatorDefinition) -> "ExtractorAccumulatorRule":
        """Create an ExtractorAccumulatorRule from an ExtractorAccumulatorDefinition."""
        return cls(
            name=defn.name,
            author=defn.author or "unknown",
            location=defn.location or RuleLocation("unknown", 0),
            match_re=defn.match_re,
            factory=defn.factory,
            source_filters=frozenset(defn.source_filters),
        )

    def process(self) -> ExtractorAccumulator:
        """Create a new extractor-accumulator instance."""
        return self.factory()


@dataclass(frozen=True, slots=True)
class RefinerRule(Rule):
    """Rule that runs on materialized symptoms and my_bad_nodes (post–event-stream)."""

    processor: RefinerProcessor
    priority: int = FilterPriority.FINAL

    @classmethod
    def from_definition(cls, defn: RefinerDefinition) -> "RefinerRule":
        """Create a RefinerRule from a RefinerDefinition."""
        return cls(
            name=defn.name,
            author=defn.author or "unknown",
            location=defn.location or RuleLocation("unknown", 0),
            processor=defn.processor,
            priority=defn.priority,
        )

    def process(self, context: RefinerContext) -> None:
        """Process the refiner rule."""
        self.processor(context)


@dataclass(frozen=True, slots=True)
class CollectorRule(Rule):
    """Post-attribution collector: queries the event stream and formats digest data.

    Each collector has a processor (queries event stream) and a formatter
    (converts raw data to digest key-value pairs). An optional ``params_model``
    declares a Pydantic schema for per-request configuration.
    """

    processor: CollectorProcessor
    formatter: CollectorFormatter
    params_model: Optional[type[RuleParamsBase]] = None

    @classmethod
    def from_definition(cls, defn: CollectorDefinition) -> "CollectorRule":
        """Create a CollectorRule from a CollectorDefinition."""
        return cls(
            name=defn.name,
            author=defn.author or "unknown",
            location=defn.location or RuleLocation("unknown", 0),
            processor=defn.processor,
            formatter=defn.formatter,
            params_model=defn.params_model,
        )

    def process(
        self, event_stream: EventStream, job_info: "JobInfo", params: Optional[RuleParamsBase] = None
    ) -> dict[str, Any]:
        """Run the collector and format results."""
        raw = self.processor(event_stream, job_info, params)
        return self.formatter(raw)


# Context variable - thread-safe, no global state
_module_globals: ContextVar[dict] = ContextVar("_module_globals")
_rules_engine: ContextVar["RuleEngine"] = ContextVar("_rules_engine")


def declare_rule_names(*names: str) -> None:
    """Declare synthetic rule names that will be used at runtime.

    Call at module scope in rule files to register rule names that aren't
    associated with a registered rule (e.g. names used for cascade tracking
    in filter/refiner processors). These names are pre-populated into the
    EventManager's lookup tables at finalize time.

    Multiple calls (including from different rule files) with the same name
    are safe — names are collected as a set union.
    """
    engine = _rules_engine.get(None)
    if engine is None:
        raise RuleIngestionError("No active rules engine when declaring rule names")
    engine.declared_rule_names.update(names)


def declare_event_types(*names: str) -> None:
    """Declare event type names that will be emitted at runtime.

    Call at module scope in rule files to register event types that aren't
    co-located with a decorator (e.g. types emitted by library helpers like
    ``_bad()`` or ``_other_event()``).

    Multiple calls (including from different rule files) with the same name
    are safe — names are collected as a set union.
    """
    engine = _rules_engine.get(None)
    if engine is None:
        raise RuleIngestionError("No active rules engine when declaring event types")
    engine.declared_event_types.update(names)


def outcome_symptom(
    prefix: str,
    *,
    outcome: str,
    priority: OutcomePriority,
    doc: str,
    hw_cause: str = "Not Applicable",
    cascade_discards: tuple[str, ...] = (),
    conditional_discards: tuple[tuple[str, str], ...] = (),
    cascade_noop: bool = False,
    fallback_prefix: bool = False,
) -> None:
    """Declare a symptom that can drive final outcome selection.

    Args:
        prefix: Symptom prefix matched via startswith during outcome selection.
        outcome: FACT outcome assigned when this symptom wins.
        priority: Outcome selection priority tier (lower wins).
        doc: User-facing attribution documentation anchor for this symptom.
        hw_cause: Hardware cause text preserved in outcome attribution.
        cascade_discards: Symptoms discarded when this symptom is present.
        conditional_discards: Predicate-gated cascade discards.
        cascade_noop: Preserve this symptom as a cascade graph node that has no
            discard edges.
        fallback_prefix: Mark this prefix as a generic fallback used to resolve
            dynamic symptom variants (such as "XID-" and "NV_ERR_").
    """
    if doc == "":
        raise RuleIngestionError(f"outcome_symptom({prefix!r}) requires a non-empty doc anchor")
    engine = _rules_engine.get(None)
    if engine is None:
        raise RuleIngestionError("No active rules engine when declaring outcome symptom")
    engine.register_symptom_declaration(
        SymptomDeclaration(
            prefix=prefix,
            outcome=outcome,
            hw_cause=hw_cause,
            priority=priority,
            doc=doc,
            cascade_discards=cascade_discards,
            conditional_discards=conditional_discards,
            cascade_noop=cascade_noop,
            fallback_prefix=fallback_prefix,
            location=_get_caller_location(),
        )
    )


def known_symptom(prefix: str, *, fallback_prefix: bool = False) -> None:
    """Declare a symptom that belongs to the vocabulary only.

    Args:
        prefix: Symptom prefix known to the registry.
        fallback_prefix: Mark this prefix as a generic fallback used to resolve
            dynamic symptom variants (such as "RuntimeError").
    """
    engine = _rules_engine.get(None)
    if engine is None:
        raise RuleIngestionError("No active rules engine when declaring known symptom")
    engine.register_symptom_declaration(
        SymptomDeclaration(
            prefix=prefix,
            fallback_prefix=fallback_prefix,
            location=_get_caller_location(),
        )
    )


def cascade_symptom(
    prefix: str,
    *,
    cascade_discards: tuple[str, ...],
    conditional_discards: tuple[tuple[str, str], ...] = (),
) -> None:
    """Declare a cascade-only symptom used for pruning, not outcome selection.

    Args:
        prefix: Symptom prefix that triggers cascade pruning.
        cascade_discards: Symptoms discarded when this symptom is present.
        conditional_discards: Predicate-gated cascade discards.
    """
    engine = _rules_engine.get(None)
    if engine is None:
        raise RuleIngestionError("No active rules engine when declaring cascade symptom")
    engine.register_symptom_declaration(
        SymptomDeclaration(
            prefix=prefix,
            cascade_discards=cascade_discards,
            conditional_discards=conditional_discards,
            location=_get_caller_location(),
        )
    )


def _get_caller_location() -> RuleLocation:
    """Return the file/line of the rule's definition site.

    The rule-registration helpers (``extractor_def``, the symptom declarators,
    etc.) all live in this module, so walk past every frame inside this file and
    return the first caller outside it — i.e. where the rule is actually defined
    (a ``bundled_rules/*.py`` file, or the generator loop that mints it), not the
    decorator machinery here.
    """
    frame = currentframe()
    if frame is None:
        raise RuleIngestionError("Cannot determine call frame")

    this_file = frame.f_code.co_filename
    frame = frame.f_back
    # Skip the engine's own registration frames; the first frame in another file
    # is the definition site.
    while frame is not None and frame.f_code.co_filename == this_file:
        frame = frame.f_back
    if frame is None:
        raise RuleIngestionError("Cannot access caller frame")

    return RuleLocation(frame.f_code.co_filename, frame.f_lineno)


# Overload: when processor is provided, returns ExtractorDefinition
@overload
def extractor_def(
    name: RuleId,
    match_re: str,
    processor: ExtractorProcessor,
    source_filters: set[ObservationSource] = set(),
    author: Optional[str] = None,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
) -> ExtractorDefinition: ...


# Overload: when processor is None (decorator usage), returns a decorator
@overload
def extractor_def(
    name: RuleId,
    match_re: str,
    processor: None = None,
    source_filters: set[ObservationSource] = set(),
    author: Optional[str] = None,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
) -> Callable[[ExtractorProcessor], ExtractorProcessor]: ...


def extractor_def(
    name: RuleId,
    match_re: str,
    processor: Optional[ExtractorProcessor] = None,
    source_filters: set[ObservationSource] = set(),
    author: Optional[str] = None,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
) -> ExtractorDefinition | Callable[[ExtractorProcessor], ExtractorProcessor]:
    """Define a extractor rule. Can be used as a decorator or called directly.

    As decorator:
        @extractor_def(name="MyRule", match_re=r"pattern")
        def my_processor(groups, node, source, ctx):
            ...

    Direct call (legacy):
        extractor_def(name="MyRule", match_re=r"pattern", processor=my_processor)

    Args:
        derived_rules: Rule names produced by source.derived(rule=...) inside
            this rule's processor. Declaring them here registers them automatically.
        derived_event_types: Event type names emitted by this rule's processor.
    """
    location = _get_caller_location()
    _derived = tuple(derived_rules)
    _event_types = tuple(derived_event_types)

    def _register(proc: ExtractorProcessor) -> ExtractorProcessor:
        mod_globals = _module_globals.get({})

        final_author = author or mod_globals.get("rule_author", "unknown")
        final_source_filters = source_filters or mod_globals.get("rule_sources", set())

        defn_instance = ExtractorDefinition(
            name=name,
            match_re=match_re,
            processor=proc,
            source_filters=final_source_filters,
            author=final_author,
            location=location,
            derived_rules=_derived,
            derived_event_types=_event_types,
        )
        rule = ExtractorRule.from_definition(defn_instance)

        engine = _rules_engine.get()
        if engine is None:
            raise RuleIngestionError(f"No active rules engine when defining rule '{name}'")

        engine.register_rule(rule)
        if _derived:
            engine.declared_rule_names.update(_derived)
        if _event_types:
            engine.declared_event_types.update(_event_types)
        return proc

    # If processor provided directly, register immediately (legacy usage)
    if processor is not None:
        _register(processor)
        mod_globals = _module_globals.get({})
        final_author = author or mod_globals.get("rule_author", "unknown")
        final_source_filters = source_filters or mod_globals.get("rule_sources", set())
        return ExtractorDefinition(
            name=name,
            match_re=match_re,
            processor=processor,
            source_filters=final_source_filters,
            author=final_author,
            location=location,
            derived_rules=_derived,
            derived_event_types=_event_types,
        )

    # Otherwise return decorator
    return _register


def filter_def(
    name: RuleId,
    author: Optional[str] = None,
    priority: int = FilterPriority.STANDARD,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
) -> Callable[[FilterProcessor], FilterProcessor]:
    """Define a filter rule. No regex needed.

    Args:
        name: Unique rule identifier.
        author: Rule author. Falls back to module-level rule_author if not set.
        priority: Execution priority. Lower values execute first.
        derived_rules: Rule names produced by source.derived(rule=...) inside
            this rule's processor. Declaring them here registers them automatically.
        derived_event_types: Event type names emitted by this rule's processor.
    """
    location = _get_caller_location()
    _derived = tuple(derived_rules)
    _event_types = tuple(derived_event_types)

    def _register(proc: FilterProcessor) -> FilterProcessor:
        mod_globals = _module_globals.get({})

        final_author = author or mod_globals.get("rule_author", "unknown")

        defn = FilterDefinition(
            name=name,
            processor=proc,
            author=final_author,
            location=location,
            priority=priority,
            derived_rules=_derived,
            derived_event_types=_event_types,
        )
        rule = FilterRule.from_definition(defn)

        engine = _rules_engine.get()
        if engine is None:
            raise RuleIngestionError(f"No active rules engine when defining rule '{name}'")

        engine.register_rule(rule)
        if _derived:
            engine.declared_rule_names.update(_derived)
        if _event_types:
            engine.declared_event_types.update(_event_types)
        return proc

    return _register


def extractor_accumulator_def(
    name: RuleId,
    match_re: str,
    source_filters: set[ObservationSource] = set(),
    author: Optional[str] = None,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
) -> Callable[[type[ExtractorAccumulator]], type[ExtractorAccumulator]]:
    """Define an extractor-accumulator rule. Used as a class decorator on ExtractorAccumulator subclasses.

    Lines are only passed to the accumulator when match_re matches,
    unless the accumulator's is_consuming property returns True.

    Args:
        derived_rules: Rule names produced by source.derived(rule=...) inside
            this rule's processor. Declaring them here registers them automatically.
        derived_event_types: Event type names emitted by this rule's processor.
    """
    location = _get_caller_location()
    _derived = tuple(derived_rules)
    _event_types = tuple(derived_event_types)

    def _register(cls: type[ExtractorAccumulator]) -> type[ExtractorAccumulator]:
        mod_globals = _module_globals.get({})

        final_author = author or mod_globals.get("rule_author", "unknown")
        final_source_filters = source_filters or mod_globals.get("rule_sources", set())

        defn = ExtractorAccumulatorDefinition(
            name=name,
            match_re=match_re,
            factory=cls,
            source_filters=final_source_filters,
            author=final_author,
            location=location,
            derived_rules=_derived,
            derived_event_types=_event_types,
        )
        rule = ExtractorAccumulatorRule.from_definition(defn)

        engine = _rules_engine.get()
        if engine is None:
            raise RuleIngestionError(f"No active rules engine when defining rule '{name}'")

        engine.register_rule(rule)
        if _derived:
            engine.declared_rule_names.update(_derived)
        if _event_types:
            engine.declared_event_types.update(_event_types)
        return cls

    return _register


def refiner_def(
    name: RuleId,
    author: Optional[str] = None,
    priority: int = FilterPriority.FINAL,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
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
    """
    location = _get_caller_location()
    _derived = tuple(derived_rules)
    _event_types = tuple(derived_event_types)

    def _register(proc: RefinerProcessor) -> RefinerProcessor:
        mod_globals = _module_globals.get({})
        final_author = author or mod_globals.get("rule_author", "unknown")
        defn = RefinerDefinition(
            name=name,
            processor=proc,
            author=final_author,
            location=location,
            priority=priority,
            derived_rules=_derived,
            derived_event_types=_event_types,
        )
        rule = RefinerRule.from_definition(defn)
        engine = _rules_engine.get()
        if engine is None:
            raise RuleIngestionError(f"No active rules engine when defining rule '{name}'")
        engine.register_rule(rule)
        if _derived:
            engine.declared_rule_names.update(_derived)
        if _event_types:
            engine.declared_event_types.update(_event_types)
        return proc

    return _register


def collector_def(
    name: RuleId,
    formatter: CollectorFormatter,
    author: Optional[str] = None,
    derived_rules: Sequence[str] = (),
    derived_event_types: Sequence[str] = (),
    params_model: Optional[type[RuleParamsBase]] = None,
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
    """
    location = _get_caller_location()
    _derived = tuple(derived_rules)
    _event_types = tuple(derived_event_types)

    def _register(proc: CollectorProcessor) -> CollectorProcessor:
        mod_globals = _module_globals.get({})

        final_author = author or mod_globals.get("rule_author", "unknown")

        defn = CollectorDefinition(
            name=name,
            processor=proc,
            formatter=formatter,
            author=final_author,
            location=location,
            derived_rules=_derived,
            derived_event_types=_event_types,
            params_model=params_model,
        )
        rule = CollectorRule.from_definition(defn)

        engine = _rules_engine.get()
        if engine is None:
            raise RuleIngestionError(f"No active rules engine when defining rule '{name}'")

        engine.register_rule(rule)
        if _derived:
            engine.declared_rule_names.update(_derived)
        if _event_types:
            engine.declared_event_types.update(_event_types)
        return proc

    return _register


class TwoStageSourceMatcher:
    """Per-source-type Aho-Corasick accelerator. Knows nothing about rule types."""

    def __init__(self, patterns_by_source: dict[ObservationSource, dict[RuleId, str]]):
        # dict is invariant in its key type, so dict[RuleId, str] is not
        # assignable to dict[Hashable, str] even though RuleId (str) satisfies
        # Hashable. Cast widens the key type; safe because TwoStageMatcher only
        # reads keys.
        self.by_source: dict[ObservationSource, TwoStageMatcher] = {
            source: TwoStageMatcher(cast("dict[Hashable, str]", patterns))
            for source, patterns in patterns_by_source.items()
        }

    def get_candidates(self, source_type: ObservationSource, datum: str) -> set[RuleId]:
        """Stage 1 only: get candidate rule IDs via Aho-Corasick pre-filter."""
        matcher = self.by_source.get(source_type)
        if matcher is None:
            return set()
        # matcher is typed with Hashable keys but we only ever build it with
        # RuleId (str) keys, so narrowing the return type back to set[RuleId]
        # is safe.
        return cast("set[RuleId]", matcher.candidates(datum))

    def confirm_match(self, source_type: ObservationSource, rule_id: RuleId, datum: str) -> Optional[Match]:
        """Stage 2: confirm a specific candidate with full regex."""
        matcher = self.by_source.get(source_type)
        if matcher is None:
            return None
        return matcher.confirm_match(rule_id, datum)


class RefinerEntry(NamedTuple):
    """A refiner rule with its execution metadata."""

    priority: int
    sequence: int
    name: str
    processor: "RefinerProcessor"


class RuleEngine:
    """Execute rules against sources"""

    def __init__(self) -> None:
        self.extractor_rules: dict[RuleId, ExtractorRule] = {}
        self.filter_rules: dict[RuleId, FilterRule] = {}
        self.extractor_accumulator_rules: dict[RuleId, ExtractorAccumulatorRule] = {}
        self.refiner_rules: dict[RuleId, RefinerRule] = {}
        self.collector_rules: dict[RuleId, CollectorRule] = {}
        self.declared_rule_names: set[str] = set()
        self.declared_event_types: set[str] = set()
        self.symptom_declarations: list[SymptomDeclaration] = []
        self.matcher: Optional[TwoStageSourceMatcher] = None
        self._filter_execution_order: list[tuple[int, int, FilterProcessor]] = []
        self._filters_sorted: bool = True
        self._filters_by_phase: dict[int, list[FilterProcessor]] = {}
        self._refiner_execution_order: list[RefinerEntry] = []
        self._refiners_sorted: bool = True
        self._refiners_by_phase: dict[int, list[tuple[str, RefinerProcessor]]] = {}

    @property
    def rules(self) -> dict[RuleId, Rule]:
        """Combined view of all rules (for backward compatibility)."""
        all_rules: dict[RuleId, Rule] = {}
        all_rules.update(self.extractor_rules)
        all_rules.update(self.filter_rules)
        all_rules.update(self.extractor_accumulator_rules)
        all_rules.update(self.refiner_rules)
        all_rules.update(self.collector_rules)
        return all_rules

    @property
    def all_rule_names(self) -> set[str]:
        """All known rule names: registered rules + explicitly declared synthetic names."""
        return set(self.rules.keys()) | self.declared_rule_names

    @property
    def all_event_types(self) -> set[str]:
        """All declared event type names."""
        return set(self.declared_event_types)

    def register_rule(self, rule: Rule) -> None:
        """Register a rule, dispatching to correct storage based on type."""
        if isinstance(rule, ExtractorRule):
            self._register_extractor(rule)
        elif isinstance(rule, FilterRule):
            self._register_filter(rule)
        elif isinstance(rule, ExtractorAccumulatorRule):
            self._register_extractor_accumulator(rule)
        elif isinstance(rule, RefinerRule):
            self._register_refiner(rule)
        elif isinstance(rule, CollectorRule):
            self._register_collector(rule)
        else:
            raise UnknownRuleTypeError(f"Unknown rule type: {type(rule)}")

    def register_symptom_declaration(self, declaration: SymptomDeclaration) -> None:
        """Register declarative symptom metadata loaded from a rule module."""
        for existing in self.symptom_declarations:
            if existing.prefix != declaration.prefix:
                continue
            if existing == declaration:
                return
            existing_location = existing.location if existing.location is not None else RuleLocation("unknown", 0)
            new_location = declaration.location if declaration.location is not None else RuleLocation("unknown", 0)
            raise RuleIngestionError(
                f"Conflicting symptom declaration for {declaration.prefix!r}: "
                f"first at {existing_location.file}:{existing_location.line}, "
                f"conflict at {new_location.file}:{new_location.line}"
            )
        self.symptom_declarations.append(declaration)

    def _check_duplicate(self, name: RuleId, location: RuleLocation) -> None:
        """Check for duplicate rule names across all rule types."""
        existing: Optional[Rule] = (
            self.extractor_rules.get(name)
            or self.filter_rules.get(name)
            or self.extractor_accumulator_rules.get(name)
            or self.refiner_rules.get(name)
            or self.filter_rules.get(name)
            or self.extractor_accumulator_rules.get(name)
            or self.collector_rules.get(name)
        )
        if existing is not None:
            raise DuplicateRuleError(
                f"Duplicate rule '{name}': "
                f"first at {existing.location.file}:{existing.location.line}, "
                f"duplicate at {location.file}:{location.line}"
            )

    def _register_extractor(self, rule: ExtractorRule) -> None:
        """Register an extractor rule."""
        self._check_duplicate(rule.name, rule.location)
        self.matcher = None  # Invalidate matcher
        self.extractor_rules[rule.name] = rule

    def _register_filter(self, rule: FilterRule) -> None:
        """Register a filter rule."""
        self._check_duplicate(rule.name, rule.location)
        self.filter_rules[rule.name] = rule
        self._filter_execution_order.append((rule.priority, len(self._filter_execution_order), rule.processor))
        self._filters_sorted = False

    def _register_extractor_accumulator(self, rule: ExtractorAccumulatorRule) -> None:
        """Register an extractor-accumulator rule."""
        self._check_duplicate(rule.name, rule.location)
        self.matcher = None  # Invalidate matcher
        self.extractor_accumulator_rules[rule.name] = rule

    def _register_refiner(self, rule: RefinerRule) -> None:
        """Register a refiner rule."""
        self._check_duplicate(rule.name, rule.location)
        self.refiner_rules[rule.name] = rule
        self._refiner_execution_order.append(
            RefinerEntry(rule.priority, len(self._refiner_execution_order), rule.name, rule.processor)
        )
        self._refiners_sorted = False

    def _register_collector(self, rule: CollectorRule) -> None:
        """Register a collector rule."""
        self._check_duplicate(rule.name, rule.location)
        self.collector_rules[rule.name] = rule

    def run_collectors(
        self,
        event_stream: EventStream,
        job_info: "JobInfo",
        rule_params: Optional[Mapping[RuleId, RuleParameters]] = None,
    ) -> dict[str, Any]:
        """Run all registered collectors and merge their digest outputs.

        Each collector's formatter returns a dict of digest key-value pairs.
        Duplicate keys across collectors are logged as errors (first value kept).
        """
        params_by_rule = rule_params or {}
        digest: dict[str, Any] = {}
        for rule_id, rule in self.collector_rules.items():
            try:
                params = self.parse_collector_params(rule, params_by_rule.get(rule_id))
                result = rule.process(event_stream, job_info, params)
            except Exception:
                logger.opt(exception=True).error(f"Collector '{rule_id}' failed")
                continue
            for key, value in result.items():
                if key in digest:
                    logger.error(
                        f"Duplicate digest key '{key}': collector '{rule_id}' "
                        f"would overwrite existing value (keeping first)"
                    )
                else:
                    digest[key] = value
        return digest

    @staticmethod
    def parse_collector_params(rule: CollectorRule, raw: Optional[RuleParameters]) -> Optional[RuleParamsBase]:
        """Parse a ``RuleParameters`` envelope against the rule's params_model.

        Returns:
            ``None`` if the rule declares no ``params_model``; otherwise
            constructs the model (with defaults when ``raw`` is ``None``).
        Raises:
            ``RuleAcceptsNoParamsError``: If params are supplied for a rule with
                no ``params_model``.
            ``InvalidRuleParams``: If the payload fails ``params_model`` validation.
        """
        if rule.params_model is None:
            if raw is not None and raw.params != {}:
                raise RuleAcceptsNoParamsError(rule.name)
            return None

        params_data = raw.params if raw is not None else {}
        try:
            return rule.params_model(**params_data)
        except ValidationError as e:
            raise InvalidRuleParams(rule.name, e) from e

    def finalize(self) -> None:
        """Build the TwoStageMatcher and sort filter execution order."""
        if self.matcher is not None:
            logger.warning("Redundant rules engine finalization")

        # Collect all patterns by source type — matcher is rule-type-agnostic
        patterns_by_source: dict[ObservationSource, dict[RuleId, str]] = {}
        for rule in self.extractor_rules.values():
            if not rule.match_re:
                continue
            for source in rule.source_filters:
                patterns_by_source.setdefault(source, {})[rule.name] = rule.match_re
        for acc_rule in self.extractor_accumulator_rules.values():
            if not acc_rule.match_re:
                continue
            for source in acc_rule.source_filters:
                patterns_by_source.setdefault(source, {})[acc_rule.name] = acc_rule.match_re

        self.matcher = TwoStageSourceMatcher(patterns_by_source)
        self._ensure_filters_sorted()
        self._ensure_refiners_sorted()

    def _ensure_filters_sorted(self) -> None:
        """Sort filter execution order and build phase buckets for efficient dispatch."""
        if not self._filters_sorted:
            self._filter_execution_order.sort()
            self._filters_by_phase = {
                priority: [proc for _, _, proc in group]
                for priority, group in groupby(
                    self._filter_execution_order,
                    key=lambda x: x[0],
                )
            }
            self._filters_sorted = True

    def _ensure_refiners_sorted(self) -> None:
        """Sort refiner execution order and build phase buckets for efficient dispatch."""
        if not self._refiners_sorted:
            self._refiner_execution_order.sort()
            self._refiners_by_phase = {
                priority: [(entry.name, entry.processor) for entry in group]
                for priority, group in groupby(
                    self._refiner_execution_order,
                    key=lambda x: x.priority,
                )
            }
            self._refiners_sorted = True

    def process_event_filters(
        self,
        event_stream: EventStream,
        job_info: JobInfo,
        *,
        phase: Optional[int] = None,
    ) -> None:
        """Process filter rules in priority order (lower priority values first).

        Args:
            phase: If set, run only filters at this exact priority.
                   If None, run all filter phases in order.
        """
        self._ensure_filters_sorted()
        context = FilterContext(job_info=job_info, event_stream=event_stream)
        if phase is not None:
            for processor in self._filters_by_phase.get(phase, []):
                processor(context)
        else:
            for priority in sorted(self._filters_by_phase.keys()):
                for processor in self._filters_by_phase[priority]:
                    processor(context)

    def process_refiner_rules(
        self,
        event_stream: EventStream,
        job_info: JobInfo,
        *,
        symptoms: Any,
        my_bad_nodes: Any,
        raw_keywords: Optional[Any] = None,
        extract_job_metadata_from_logfile: bool = False,
        phase: Optional[int] = None,
        outcome: Optional[Any] = None,
        cancelled_by: Optional[str] = None,
        attribution_result: Optional[Any] = None,
        bad_node_result: Optional[Any] = None,
        recorder: Optional[DecisionRecorder] = None,
    ) -> RefinerContext:
        """Process refiner rules in priority order (lower priority values first).

        Args:
            event_stream: Event stream (for non-attribution queries, e.g. timeout).
            job_info: Job metadata.
            symptoms: Materialized SymptomSet[str].
            my_bad_nodes: Materialized JobNodeAttributions.
            raw_keywords: Raw keyword SymptomSet[str] from log scanning.
            extract_job_metadata_from_logfile: Whether job metadata was extracted from log file.
            phase: Run only refiners in this phase. The upper bound is derived
                   automatically from the next FilterPriority value.
                   If None, run all pre-outcome phases (SCRUB + FINAL).
            outcome: SourceSet[OutcomeAttribution] (for POST_OUTCOME refiners).
            cancelled_by: Cancellation source (for POST_OUTCOME refiners).
            attribution_result: AttributionResult (for POST_OUTCOME refiners).
            bad_node_result: CalcBadNodesAttributionResult (for POST_OUTCOME refiners).
            recorder: Optional DecisionRecorder for decision graph.

        Returns:
            The RefinerContext after all rules have executed.
        """
        self._ensure_refiners_sorted()
        context = RefinerContext(
            job_info=job_info,
            event_stream=event_stream,
            symptoms=symptoms,
            my_bad_nodes=my_bad_nodes,
            raw_keywords=raw_keywords,
            extract_job_metadata_from_logfile=extract_job_metadata_from_logfile,
            recorder=recorder,
            outcome=outcome,
            cancelled_by=cancelled_by,
            attribution_result=attribution_result,
            bad_node_result=bad_node_result,
        )

        def _run_refiners(phase_rules: list[tuple[str, "RefinerProcessor"]]) -> None:
            cached_bad: Optional[NodeSymptomSnapshot] = None
            for name, processor in phase_rules:
                before: Optional[set[str]] = None
                if recorder is not None:
                    before = set(symptoms.symptoms()) if symptoms is not None else set()
                    if cached_bad is None:
                        cached_bad = snapshot_bad_nodes(my_bad_nodes)
                processor(context)
                if recorder is not None and before is not None:
                    after = set(symptoms.symptoms()) if symptoms is not None else set()
                    after_bad = snapshot_bad_nodes(my_bad_nodes)
                    if after != before or after_bad != cached_bad:
                        recorder.record_refiner_diff(name, before, after, cached_bad or {}, after_bad)
                    cached_bad = after_bad

        if phase is not None:
            upper = min((p for p in FilterPriority if p > phase), default=None)
            for priority in sorted(self._refiners_by_phase.keys()):
                if priority < phase:
                    continue
                if upper is not None and priority >= upper:
                    break
                _run_refiners(self._refiners_by_phase[priority])
        else:
            for priority in sorted(self._refiners_by_phase.keys()):
                if priority >= FilterPriority.POST_OUTCOME:
                    break
                _run_refiners(self._refiners_by_phase[priority])
        return context

    def observation_processor(
        self,
        source_type: ObservationSource,
        job_info: JobInfo,
        event_stream: EventStream,
    ) -> "ObservationProcessor":
        """Create an ObservationProcessor for the given source type.

        Ensures the matcher is built before creating the processor.
        """
        if self.matcher is None:
            self.finalize()
        return ObservationProcessor(
            engine=self,
            source_type=source_type,
            context=RuleContext(job_info=job_info),
            event_adder=event_stream.event_adder(),
        )

    def load(self, filepath: str) -> dict[RuleId, Rule]:
        """
        Load a rules module and return dict of Rule objects.

        The rule file's parent directory is temporarily added to sys.path
        so that rule modules can import from sibling packages (e.g.
        ``from common.syslog_helpers import ...``).
        """
        engine_token = _rules_engine.set(self)
        rules_before = set(self.rules.keys())
        symptom_declarations_before = list(self.symptom_declarations)
        load_succeeded = False

        try:
            # Load module
            spec = importlib.util.spec_from_file_location("rules_module", filepath)
            if spec is None or spec.loader is None:
                raise RuleIngestionError(f"Cannot load module from {filepath}")

            module = importlib.util.module_from_spec(spec)

            # Temporarily add rule file's parent directory to sys.path so
            # rules can import from sibling packages (e.g. common/).
            parent_dir = str(Path(filepath).resolve().parent)
            sys.path.insert(0, parent_dir)

            globals_token = _module_globals.set(module.__dict__)
            try:
                spec.loader.exec_module(module)
            finally:
                _module_globals.reset(globals_token)
                sys.path.remove(parent_dir)

            # Return only the rules that were added by this module
            rules_after = set(self.rules.keys())
            new_rules = {name: self.rules[name] for name in (rules_after - rules_before)}
            load_succeeded = True
            return new_rules

        finally:
            if not load_succeeded:
                self.symptom_declarations = symptom_declarations_before
            _rules_engine.reset(engine_token)

    def extractor_hash(self) -> str:
        """Stable hash of all extractor rules. Changes when extraction behavior changes.

        TODO: evolve into a versioned scheme that can detect compatible vs breaking
        rule changes (e.g. additive rules could reuse existing cache + incremental extract).
        """
        h = hashlib.sha256()
        for name in sorted(self.extractor_rules):
            rule = self.extractor_rules[name]
            h.update(name.encode())
            h.update(rule.match_re.encode())
            h.update(str(sorted(sf.value for sf in rule.source_filters)).encode())
        for name in sorted(self.extractor_accumulator_rules):
            acc_rule = self.extractor_accumulator_rules[name]
            h.update(name.encode())
            h.update(acc_rule.match_re.encode())
        return h.hexdigest()

    def attribution_hash(self) -> str:
        """Stable hash of filter/refiner/collector rules. Changes when attribution logic changes."""
        h = hashlib.sha256()
        for name in sorted(self.filter_rules):
            filter_rule = self.filter_rules[name]
            h.update(name.encode())
            h.update(str(filter_rule.priority).encode())
        for name in sorted(self.refiner_rules):
            refiner_rule = self.refiner_rules[name]
            h.update(name.encode())
            h.update(str(refiner_rule.priority).encode())
        for name in sorted(self.collector_rules):
            h.update(name.encode())
        return h.hexdigest()


_SOFT_LINE_LIMIT = 100
_ACTIVE_COUNT_WARN = 10

_logger = logger


@dataclass(slots=True)
class ObservationProcessor:
    """Stateful per-observation processor created by RuleEngine.

    Owns extractor-accumulator state. Single public method ``process_line()`` does:
    1. Feed line to active extractor-accumulators + clean up finished ones
    2. Single Aho-Corasick pass → unified candidate loop dispatching to
       extractors OR spawning extractor-accumulators
    3. Emit events via the event adder

    Safeguards:
    - Soft line limit: warns once per extractor-accumulator exceeding threshold
    - Active count high-water mark: warns once if too many active
    """

    engine: "RuleEngine"
    source_type: ObservationSource
    context: RuleContext
    event_adder: Callable
    _acc_rules: dict[RuleId, ExtractorAccumulatorRule] = field(init=False)
    _active: dict[RuleId, ExtractorAccumulator] = field(init=False, default_factory=dict)
    _lines_fed: dict[RuleId, int] = field(init=False, default_factory=dict)
    _warned_line_limit: set[RuleId] = field(init=False, default_factory=set)
    _warned_active_count: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._acc_rules = {
            r.name: r for r in self.engine.extractor_accumulator_rules.values() if self.source_type in r.source_filters
        }

    # -- Extractor-accumulator lifecycle helpers --

    def _emit_events(self, acc: ExtractorAccumulator) -> None:
        """Finalize an extractor-accumulator and emit its events."""
        for evt in acc.finalize():
            self.event_adder(evt.type, evt.payload, evt.source)

    def _deactivate(self, rid: RuleId) -> None:
        """Remove an active extractor-accumulator, clean up tracking state, and emit its events."""
        acc = self._active.pop(rid)
        self._lines_fed.pop(rid, None)
        self._emit_events(acc)

    def _try_spawn(self, rule_id: RuleId, source: Source) -> None:
        """Spawn a new extractor-accumulator if the rule isn't already active.

        If the new instance is still consuming after start(), it becomes
        active and will be fed subsequent lines. Otherwise it is finalized
        immediately and discarded.
        """
        if rule_id in self._active:
            return
        new_acc = self._acc_rules[rule_id].process()
        new_acc.start(source)
        if new_acc.is_consuming:
            self._active[rule_id] = new_acc
            self._check_active_count()
        else:
            self._emit_events(new_acc)

    def _feed_active(self, source: Source) -> None:
        """Feed a line to every active extractor-accumulator, then finalize any that finished."""
        if not self._active:
            return
        for rid, acc in self._active.items():
            acc.process_line(source)
            self._lines_fed[rid] = self._lines_fed.get(rid, 0) + 1
            if rid not in self._warned_line_limit and self._lines_fed[rid] >= _SOFT_LINE_LIMIT:
                self._warned_line_limit.add(rid)
                _logger.warning(
                    f"Extractor-accumulator {rid!r} has consumed {self._lines_fed[rid]} lines without finalizing"
                )
        finished = [rid for rid, a in self._active.items() if not a.is_consuming]
        for rid in finished:
            self._deactivate(rid)

    def _check_active_count(self) -> None:
        if not self._warned_active_count and len(self._active) >= _ACTIVE_COUNT_WARN:
            self._warned_active_count = True
            _logger.warning(f"High extractor-accumulator count: {len(self._active)} active")

    # -- Main entry points --

    def process_line(self, source: Source) -> None:
        """Process a single line through extractors and extractor-accumulators."""
        try:
            self.context.hostname = get_single_node(source.resources).hostname
        except UnknownResourceError:
            self.context.hostname = None

        self._feed_active(source)

        assert self.engine.matcher is not None
        for rule_id in self.engine.matcher.get_candidates(self.source_type, source.datum):
            match = self.engine.matcher.confirm_match(self.source_type, rule_id, source.datum)
            if not match:
                continue
            if (ext_rule := self.engine.extractor_rules.get(rule_id)) is not None:
                for evt in ext_rule.process(match, source, self.context):
                    self.event_adder(evt.type, evt.payload, evt.source)
            elif rule_id in self._acc_rules:
                self._try_spawn(rule_id, source)

    def finalize(self) -> None:
        """End-of-stream: finalize all remaining active extractor-accumulators."""
        for rid in list(self._active):
            self._deactivate(rid)


@dataclass(frozen=True, slots=True)
class SourceProcessingConfig:
    """Per-source-type configuration for process_observation_source."""

    time_start: Optional[Callable[[JobInfo], Optional[datetime]]] = None
    time_end: Optional[Callable[[JobInfo], Optional[datetime]]] = None
    fallback_unknown_nodes: bool = False
    requires_nodes: bool = False


# Populated by core.py via register_source_config(); keeps rules_engine
# free of source-specific imports.
SOURCE_CONFIGS: dict[ObservationSource, SourceProcessingConfig] = {}


def register_source_config(source: ObservationSource, config: SourceProcessingConfig) -> None:
    """Register processing configuration for an observation source type."""
    SOURCE_CONFIGS[source] = config


PreFilterCallable: TypeAlias = Callable[["EntryProtocol", "JobInfo"], Union[bool, tuple[bool, list[str]]]]


@dataclass
class PreFilterSpec:
    """Pre-filter specification registered by rules for a given source type."""

    include_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    callables: list[PreFilterCallable] = field(default_factory=list)


@dataclass
class MetricSpec:
    """Metric collection specification registered by rules."""

    thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)


_PRE_FILTERS: dict[ObservationSource, PreFilterSpec] = {}
_METRIC_SPECS: dict[ObservationSource, MetricSpec] = {}


def register_pre_filter(source: ObservationSource, spec: PreFilterSpec) -> None:
    """Register pre-filter spec for an observation source. Multiple calls merge."""
    if source in _PRE_FILTERS:
        existing = _PRE_FILTERS[source]
        existing.include_terms.extend(spec.include_terms)
        existing.exclude_terms.extend(spec.exclude_terms)
        existing.callables.extend(spec.callables)
    else:
        _PRE_FILTERS[source] = PreFilterSpec(
            include_terms=list(spec.include_terms),
            exclude_terms=list(spec.exclude_terms),
            callables=list(spec.callables),
        )


def register_metric_spec(source: ObservationSource, spec: MetricSpec) -> None:
    """Register metric spec for an observation source. Multiple calls merge."""
    if source in _METRIC_SPECS:
        _METRIC_SPECS[source].thresholds.update(spec.thresholds)
    else:
        _METRIC_SPECS[source] = MetricSpec(thresholds=dict(spec.thresholds))


def get_pre_filter(source: ObservationSource) -> PreFilterSpec:
    """Return pre-filter spec for source. Raises if not registered."""
    if source not in _PRE_FILTERS:
        raise UninitializedRuleEngineError(f"No pre-filter registered for {source.name}. Was setup_rules() called?")
    return _PRE_FILTERS[source]


def get_metric_spec(source: ObservationSource) -> MetricSpec:
    """Return metric spec for source. Raises if not registered."""
    if source not in _METRIC_SPECS:
        raise UninitializedRuleEngineError(f"No metric spec registered for {source.name}. Was setup_rules() called?")
    return _METRIC_SPECS[source]


def _apply_pre_filter(
    entry: EntryProtocol,
    job_info: JobInfo,
    exclude_set: Optional[set[str]],
    include_set: Optional[set[str]],
    callables: Optional[list[PreFilterCallable]],
) -> tuple[bool, list[str]]:
    """Apply pre-filter logic to a single entry.

    Returns (skip, keywords) where skip=True means the entry should be
    discarded, and keywords is a list of matched include terms or
    callable-contributed keywords to add to raw_keywords.
    """
    body = entry.body()
    skip = False
    keywords: list[str] = []

    # 1. Callable pre-filters run first — they can collect keywords
    #    from entries that will later be excluded by keyword rules.
    if callables:
        for fn in callables:
            result = fn(entry, job_info)
            if isinstance(result, tuple):
                should_skip, kw = result
                keywords.extend(kw)
                if should_skip:
                    skip = True
            elif result:
                skip = True

    if skip:
        return True, keywords

    # 2. Keyword exclude
    if exclude_set:
        for term in exclude_set:
            if term in body:
                return True, keywords

    # 3. Keyword include — collect matched terms
    if include_set:
        for term in include_set:
            if term in body:
                keywords.append(term)

    return False, keywords


def process_observation_source(
    *,
    event_stream: EventStream,
    observation_source: EntrySourceBase,
    job_info: JobInfo,
    rules_engine: RuleEngine,
    pre_filtered: bool,
) -> set[str]:
    """Process an observation source through the rules engine.

    Args:
        pre_filtered: If True, skip all pre-filter logic (keyword exclude/include
            and callables). Caller is responsible for any filtering.

    Returns:
        Set of content-derived raw_keywords collected from include term matches
        and callable pre-filter returns. Empty if pre_filtered=True.
    """
    source_type = observation_source.context.source
    config = SOURCE_CONFIGS.get(source_type)
    time_start = config.time_start(job_info) if config and config.time_start else None
    time_end = config.time_end(job_info) if config and config.time_end else None
    fallback_unknown_nodes = config.fallback_unknown_nodes if config else False

    raw_keywords: set[str] = set()

    # Build pre-filter state (once, not per-entry)
    pre_filter_spec = None if pre_filtered else _PRE_FILTERS.get(source_type)
    exclude_set = set(pre_filter_spec.exclude_terms) if pre_filter_spec else None
    include_set = set(pre_filter_spec.include_terms) if pre_filter_spec else None
    pf_callables = pre_filter_spec.callables if pre_filter_spec else None

    processor = rules_engine.observation_processor(source_type, job_info, event_stream)

    for entry in iter(observation_source):
        # Pre-filter phase (when not pre_filtered)
        if pre_filter_spec is not None:
            skip, entry_keywords = _apply_pre_filter(entry, job_info, exclude_set, include_set, pf_callables)
            raw_keywords.update(entry_keywords)
            if skip:
                continue

        # Time filtering
        if time_start is not None or time_end is not None:
            entry_time = entry.time()
            if time_start is not None and entry_time < time_start:
                log_ignored_time_line(entry, job_info, "before_start", entry.context().source)
                continue
            if time_end is not None and entry_time > time_end:
                log_ignored_time_line(entry, job_info, "after_end", entry.context().source)
                continue

        # Node resolution and source building
        if fallback_unknown_nodes:
            try:
                att_src_base = entry.attribution_source_base(job_info)
            except UnknownResourceError as e:
                log_ignored_node_line(e, entry, job_info, entry.context().source)
                att_src_base = entry.attribution_source_base(None)
        else:
            try:
                node = entry.node(job_info)
            except UnknownResourceError as e:
                log_ignored_node_line(e, entry, job_info, entry.context().source)
                continue

            if not node:
                continue

            att_src_base = entry.attribution_source_base(job_info)

        processor.process_line(att_src_base)

    processor.finalize()
    return raw_keywords
