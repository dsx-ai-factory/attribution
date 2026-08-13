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
import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Hashable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import (
    dataclass,
    field,
    fields as dc_fields,
)
from datetime import datetime
from enum import IntEnum
from inspect import currentframe
from itertools import groupby
from pathlib import Path
from re import Match
from typing import Any, NamedTuple, Optional, Sequence, TypeAlias, Union, cast

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
from attribution_lib.symptoms.declarations import (
    CascadeScope,
    DagDiscardBehavior,
    OutcomePriority,
    SymptomCategory,
    SymptomDeclaration,
)
from attribution_lib.symptoms.predicate_registry import SymptomPredicate

# Zero-arg provider returning a flat {name: value} or {key: record} dict.
_NamespaceProvider: TypeAlias = Callable[[], dict[str, Any]]
# Decorator that accepts and returns a _NamespaceProvider unchanged (init-scoped @const / @rows).
_ProviderDecorator: TypeAlias = Callable[[_NamespaceProvider], _NamespaceProvider]


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
        """Resolve a rank number to a node hostname. Returns None if rank is unknown."""
        if self.job_info is None:
            return None
        if not self.job_info.rank_exists(rank):
            logger.debug(
                "rank {} not in job {} topology; cannot resolve to a node",
                rank,
                self.job_info.id,
            )
            return None

        result: str = self.job_info.get_name_from_rank(rank)
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
    tags: tuple[str, ...] = ()
    """Free-form classification labels, already normalized by normalize_tags()."""
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
    tags: tuple[str, ...] = field(default=(), kw_only=True)

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
            tags=defn.tags,
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
            tags=defn.tags,
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
            tags=defn.tags,
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
            tags=defn.tags,
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


# Fixed-function names an @expr_helper may not shadow. MUST stay in sync with
# attribution_lib.declarative.expressions._FIXED_FUNCS.
_EXPR_HELPER_RESERVED = frozenset({"int", "float", "str", "len", "bool", "round", "search", "match"})

# Mirrors lifecycle.INJECTABLE_PARAMS; duplicated here because rules_engine is
# imported by lifecycle's transitive dependencies, making a top-level import circular.
EXPR_HELPER_INJECTED_PARAMS: tuple[str, ...] = ("source", "ctx")


def expr_helper(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a Python function as a named expression helper, callable bare
    inside a declarative ``when`` / symptom / payload expression (e.g.
    ``is_user_path(path)``).

    Init-scoped like :func:`declare_rule_names`: it registers into the current
    engine during the load pass and is rebuilt on each ``setup_rules`` (a fresh
    ``RuleEngine`` per init). The registration NAME is independent of the
    function's ``__name__``; the function is returned unchanged so it stays
    directly callable in Python (a helper shared with un-migrated Python rules
    keeps working).

    A helper may declare ``source`` and/or ``ctx`` parameters; the engine injects
    the current :class:`Source` / :class:`RuleContext` for whichever it declares
    (see :mod:`attribution_lib.declarative.lifecycle`). Every other argument comes
    from the calling expression.

    Raises :class:`RuleIngestionError` if there is no active engine, the name
    shadows a fixed function, or the name is already registered in this pass.
    """

    def register(func: Callable[..., Any]) -> Callable[..., Any]:
        engine = _rules_engine.get(None)
        if engine is None:
            raise RuleIngestionError(f"No active rules engine when registering expr_helper {name!r}")
        from attribution_lib.declarative.lifecycle import LIFECYCLE_PREDICATES as _lc_preds

        if name in _EXPR_HELPER_RESERVED or name in _lc_preds:
            raise RuleIngestionError(
                f"expr_helper {name!r} shadows a fixed function or lifecycle predicate; choose another name"
            )
        existing = engine.expr_helpers.get(name)
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
        engine.expr_helpers[name] = func
        engine.expr_helper_injections[name] = frozenset(p for p in EXPR_HELPER_INJECTED_PARAMS if p in sig.parameters)
        return func

    return register


def const(ns: str) -> _ProviderDecorator:
    """Register a `@const` provider: a zero-arg function returning a flat
    ``{name: value}`` dict merged into the engine's single ``const`` namespace.

    The ``ns`` argument names the PROVIDER (for error messages), not a
    sub-namespace: a rule reads ``const.<name>``, never ``const.<ns>.<name>``.
    Init-scoped like :func:`expr_helper`; the provider is called at registration
    (its values are static for v1). A ``const`` name defined more than once
    (across two providers, or a provider and an inline ``[const]`` table) is a
    :class:`RuleIngestionError`. The provider function is returned unchanged.
    """

    def register(func: _NamespaceProvider) -> _NamespaceProvider:
        engine = _rules_engine.get(None)
        if engine is None:
            raise RuleIngestionError(f"No active rules engine when registering @const provider {ns!r}")
        values = func()
        if not isinstance(values, dict):
            raise RuleIngestionError(
                f"@const provider {ns!r} must return a dict of {{name: value}}, got {type(values).__name__}"
            )
        for name, value in values.items():
            if name in engine.const_values:
                raise RuleIngestionError(
                    f"const name {name!r} from provider {ns!r} is already defined; "
                    f"const names are a single flat namespace"
                )
            engine.const_values[name] = value
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
        engine = _rules_engine.get(None)
        if engine is None:
            raise RuleIngestionError(f"No active rules engine when registering @rows provider {name!r}")
        if name in engine.record_tables:
            raise RuleIngestionError(f"record table {name!r} is already registered by another @rows provider")
        records = func()
        if not isinstance(records, dict):
            raise RuleIngestionError(
                f"@rows provider {name!r} must return a dict of {{key: record}}, got {type(records).__name__}"
            )
        engine.record_tables[name] = {str(k): v for k, v in records.items()}
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
        engine = _rules_engine.get(None)
        if engine is None:
            raise RuleIngestionError(f"No active rules engine when registering @rule_processor {ns_name!r}")
        if ns_name in engine.rule_processors:
            existing = engine.rule_processors[ns_name]
            raise RuleIngestionError(
                f"@rule_processor {ns_name!r} is already registered by {getattr(existing, '__module__', '?')}:{getattr(existing, '__qualname__', repr(existing))}"
            )
        engine.rule_processors[ns_name] = func
        return func

    return register


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
    category: Optional[SymptomCategory] = None,
    fallback_prefix: bool = False,
    dag_discard: DagDiscardBehavior = DagDiscardBehavior.DEMOTE,
) -> None:
    """Declare a symptom that can drive final outcome selection.

    Args:
        prefix: Symptom prefix matched via startswith during outcome selection.
        outcome: NVIDIA Attribution outcome assigned when this symptom wins.
        priority: Outcome selection priority tier (lower wins).
        doc: User-facing attribution documentation anchor for this symptom.
        hw_cause: Hardware cause text preserved in outcome attribution.
        cascade_discards: Symptoms discarded when this symptom is present.
        conditional_discards: Predicate-gated cascade discards.
        cascade_noop: Preserve this symptom as a cascade graph node that has no
            discard edges.
        category: Optional semantic category for registry queries.
        fallback_prefix: Mark this prefix as a generic fallback used to resolve
            dynamic symptom variants (such as "XID-" and "NV_ERR_").
        dag_discard: Projection behavior when the DAG discards this symptom.
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
            category=category,
            fallback_prefix=fallback_prefix,
            dag_discard=dag_discard,
            location=_get_caller_location(),
        )
    )


def symptom_predicate(func: SymptomPredicate) -> SymptomPredicate:
    """Register a conditional-discard predicate onto the active engine.

    Init-scoped like the symptom-declaration decorators: it attaches the callable
    to whichever :class:`RuleEngine` is loading the rule file, keyed by the function
    name that ``conditional_discards`` references. Unlike ``known_symptom``, this may
    also run when the predicate module is imported directly for its function objects
    (e.g. unit tests), where no engine is loading; that is a no-op, since registration
    only matters during a rule load.
    """
    engine = _rules_engine.get(None)
    if engine is not None:
        # getattr rather than func.__name__: the parameter is typed as a Callable alias,
        # which is not guaranteed a __name__ by strict type checkers.
        name: str = getattr(func, "__name__")
        engine.register_symptom_predicate(name, func)
    return func


def known_symptom(
    prefix: str,
    *,
    category: Optional[SymptomCategory] = None,
    fallback_prefix: bool = False,
    dag_discard: DagDiscardBehavior = DagDiscardBehavior.DEMOTE,
) -> None:
    """Declare a symptom that belongs to the vocabulary only.

    Args:
        prefix: Symptom prefix known to the registry.
        category: Optional semantic category for registry queries.
        fallback_prefix: Mark this prefix as a generic fallback used to resolve
            dynamic symptom variants (such as "RuntimeError").
        dag_discard: Projection behavior when the DAG discards this symptom.
    """
    engine = _rules_engine.get(None)
    if engine is None:
        raise RuleIngestionError("No active rules engine when declaring known symptom")
    engine.register_symptom_declaration(
        SymptomDeclaration(
            prefix=prefix,
            category=category,
            fallback_prefix=fallback_prefix,
            dag_discard=dag_discard,
            location=_get_caller_location(),
        )
    )


def cascade_symptom(
    prefix: str,
    *,
    cascade_discards: tuple[str, ...],
    conditional_discards: tuple[tuple[str, str], ...] = (),
    scope: CascadeScope = CascadeScope.NODE,
    category: Optional[SymptomCategory] = None,
    dag_discard: DagDiscardBehavior = DagDiscardBehavior.DEMOTE,
    resolve_contains: Optional[str] = None,
) -> None:
    """Declare a cascade-only symptom used for pruning, not outcome selection.

    Args:
        prefix: Symptom prefix that triggers cascade pruning.
        cascade_discards: Symptoms discarded when this symptom is present.
        conditional_discards: Predicate-gated cascade discards.
        scope: Resource scope where cascade discards apply.
        category: Optional semantic category for registry queries.
        dag_discard: Projection behavior when the DAG discards this symptom.
        resolve_contains: Any raw symptom id containing this substring resolves to
            this prefix.
    """
    engine = _rules_engine.get(None)
    if engine is None:
        raise RuleIngestionError("No active rules engine when declaring cascade symptom")
    engine.register_symptom_declaration(
        SymptomDeclaration(
            prefix=prefix,
            cascade_discards=cascade_discards,
            conditional_discards=conditional_discards,
            scope=scope,
            category=category,
            dag_discard=dag_discard,
            resolve_contains=resolve_contains,
            location=_get_caller_location(),
        )
    )


def _get_caller_location() -> RuleLocation:
    """Return the file/line of the rule's definition site.

    The rule-registration helpers all live in this module, so walk past every
    frame inside this file and return the first caller outside it — i.e. where
    the rule is actually defined (a ``bundled_rules/*.py`` file, or the generator
    loop that mints it), not the decorator machinery here.
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
    location = _get_caller_location()
    _derived = tuple(derived_rules)
    _event_types = tuple(derived_event_types)
    _tags = normalize_tags(tags, where=f"{name} ({location.file}:{location.line})")

    def _register(proc: FilterProcessor) -> FilterProcessor:
        mod_globals = _module_globals.get({})

        final_author = author or mod_globals.get("rule_author", "unknown")

        defn = FilterDefinition(
            name=name,
            processor=proc,
            author=final_author,
            location=location,
            tags=_tags,
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
    location = _get_caller_location()
    _derived = tuple(derived_rules)
    _event_types = tuple(derived_event_types)
    _tags = normalize_tags(tags, where=f"{name} ({location.file}:{location.line})")

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
            tags=_tags,
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
    location = _get_caller_location()
    _derived = tuple(derived_rules)
    _event_types = tuple(derived_event_types)
    _tags = normalize_tags(tags, where=f"{name} ({location.file}:{location.line})")

    def _register(proc: RefinerProcessor) -> RefinerProcessor:
        mod_globals = _module_globals.get({})
        final_author = author or mod_globals.get("rule_author", "unknown")
        defn = RefinerDefinition(
            name=name,
            processor=proc,
            author=final_author,
            location=location,
            tags=_tags,
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
    location = _get_caller_location()
    _derived = tuple(derived_rules)
    _event_types = tuple(derived_event_types)
    _tags = normalize_tags(tags, where=f"{name} ({location.file}:{location.line})")

    def _register(proc: CollectorProcessor) -> CollectorProcessor:
        mod_globals = _module_globals.get({})

        final_author = author or mod_globals.get("rule_author", "unknown")

        defn = CollectorDefinition(
            name=name,
            processor=proc,
            formatter=formatter,
            author=final_author,
            location=location,
            tags=_tags,
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


@dataclass
class _LoadSnapshot:
    """Snapshot of all RuleEngine fields that may be mutated during load.

    Field names match the corresponding RuleEngine attribute names exactly so
    capture() and restore() can iterate dc_fields() without a name mapping.
    The special fields ``_pre_filters`` and ``_metric_specs`` capture the module-level
    ``_PRE_FILTERS`` and ``_METRIC_SPECS`` dicts (not engine attributes) and are
    handled explicitly in capture/restore.
    """

    const_values: dict
    extractor_rules: dict
    filter_rules: dict
    extractor_accumulator_rules: dict
    refiner_rules: dict
    collector_rules: dict
    declared_rule_names: set
    declared_event_types: set
    declarative_emit: dict
    record_tables: dict
    expr_helpers: dict
    expr_helper_injections: dict
    rule_processors: dict
    symptom_declarations: list
    symptom_predicates: dict
    _filter_execution_order: list
    _refiner_execution_order: list
    _filters_sorted: bool
    _filters_by_phase: dict
    _refiners_sorted: bool
    _refiners_by_phase: dict
    matcher: object  # Optional[TwoStageSourceMatcher]; copied by reference (immutable identity)
    _pre_filters: dict  # snapshot of the module-level _PRE_FILTERS
    _metric_specs: dict  # snapshot of the module-level _METRIC_SPECS

    @classmethod
    def capture(cls, e: "RuleEngine") -> "_LoadSnapshot":
        copy: dict[str, Any] = {}
        for f in dc_fields(cls):
            if f.name == "_pre_filters":
                copy["_pre_filters"] = {
                    k: PreFilterSpec(
                        include_terms=list(v.include_terms),
                        exclude_terms=list(v.exclude_terms),
                        callables=list(v.callables),
                    )
                    for k, v in _PRE_FILTERS.items()
                }
            elif f.name == "_metric_specs":
                copy["_metric_specs"] = {k: MetricSpec(thresholds=dict(v.thresholds)) for k, v in _METRIC_SPECS.items()}
            else:
                v = getattr(e, f.name)
                copy[f.name] = type(v)(v) if isinstance(v, (dict, list, set)) else v
        return cls(**copy)

    def restore(self, e: "RuleEngine") -> None:
        for f in dc_fields(self):
            if f.name == "_pre_filters":
                _PRE_FILTERS.clear()
                _PRE_FILTERS.update(self._pre_filters)
            elif f.name == "_metric_specs":
                _METRIC_SPECS.clear()
                _METRIC_SPECS.update(self._metric_specs)
            else:
                setattr(e, f.name, getattr(self, f.name))


class RuleEngine:
    """Execute rules against sources"""

    def __init__(self) -> None:
        self.extractor_rules: dict[RuleId, ExtractorRule] = {}
        self.filter_rules: dict[RuleId, FilterRule] = {}
        self.extractor_accumulator_rules: dict[RuleId, ExtractorAccumulatorRule] = {}
        self.refiner_rules: dict[RuleId, RefinerRule] = {}
        self.collector_rules: dict[RuleId, CollectorRule] = {}
        self.declarative_emit: dict[str, str] = {}
        self.declared_rule_names: set[str] = set()
        self.declared_event_types: set[str] = set()
        self.expr_helpers: dict[str, "Callable[..., object]"] = {}
        self.const_values: dict[str, object] = {}
        self.record_tables: dict[str, dict] = {}
        self.rule_processors: dict[str, "Callable[..., object]"] = {}
        self.expr_helper_injections: dict[str, frozenset] = {}
        self.symptom_declarations: list[SymptomDeclaration] = []
        self.symptom_predicates: dict[str, SymptomPredicate] = {}
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
            if existing.prefix != declaration.prefix or existing.scope is not declaration.scope:
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

    def register_symptom_predicate(self, name: str, predicate: SymptomPredicate) -> None:
        """Register a conditional-discard predicate under ``name`` for this engine.

        Cascade rules reference the predicate by name in ``conditional_discards``;
        the compiled registry resolves the name to this callable.
        """
        existing = self.symptom_predicates.get(name)
        if existing is not None and existing is not predicate:
            raise RuleIngestionError(f"Conflicting symptom predicate registered for {name!r}")
        self.symptom_predicates[name] = predicate

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

    @contextmanager
    def _load_guard(self) -> Generator[None, None, None]:
        """Snapshot all load-mutable engine state; restore on any exception."""
        snap = _LoadSnapshot.capture(self)
        try:
            yield
        except Exception:
            snap.restore(self)
            raise

    def load(self, filepath: str) -> dict[RuleId, Rule]:
        """
        Load a rules module and return dict of Rule objects.

        The rule file's parent directory is temporarily added to sys.path
        so that rule modules can import from sibling packages (e.g.
        ``from common.syslog_helpers import ...``).
        """
        if str(filepath).endswith(".toml"):
            # Import here (not at module top) to avoid a circular import:
            # declarative.loader imports ExtractorRule/RuleLocation from this
            # module, so a top-level import would cycle back into rules_engine.
            from attribution_lib.declarative import load_toml_rules

            with self._load_guard():
                engine_token = _rules_engine.set(self)
                try:
                    return load_toml_rules(self, filepath)
                finally:
                    _rules_engine.reset(engine_token)

        rules_before = set(self.rules.keys())
        with self._load_guard():
            engine_token = _rules_engine.set(self)
            try:
                spec = importlib.util.spec_from_file_location("rules_module", filepath)
                if spec is None or spec.loader is None:
                    raise RuleIngestionError(f"Cannot load module from {filepath}")

                module = importlib.util.module_from_spec(spec)

                # Temporarily add rule file's parent directory to sys.path so
                # rules can import from sibling packages (e.g. common/).
                parent_dir = str(Path(filepath).resolve().parent)
                sys_path_snapshot = sys.path.copy()
                sys.path.insert(0, parent_dir)

                globals_token = _module_globals.set(module.__dict__)
                try:
                    spec.loader.exec_module(module)
                finally:
                    _module_globals.reset(globals_token)
                    sys.path[:] = sys_path_snapshot

                # Return only the rules that were added by this module
                rules_after = set(self.rules.keys())
                return {name: self.rules[name] for name in (rules_after - rules_before)}
            finally:
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
            h.update(self.declarative_emit.get(name, "").encode())
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
    """Return the registered pre-filter spec for source, or an empty spec if it has none."""
    return _PRE_FILTERS.get(source, PreFilterSpec())


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
