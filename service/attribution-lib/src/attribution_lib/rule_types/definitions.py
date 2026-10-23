# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule values, execution contexts, processor protocols, and matchers."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from re import Match
from typing import Any, NamedTuple, Optional, TypeAlias, cast

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.events import EventStream
from attribution_lib.exceptions import InvalidRuleParams, RuleAcceptsNoParamsError
from attribution_lib.job_info import JobInfo
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.regex_accelerator.two_stage_matcher import TwoStageMatcher
from attribution_lib.sources import Source, SourceSet


class EventTriple(NamedTuple):
    """Output of a rule"""

    type: str
    payload: dict[str, Any]
    source: Source


class RuleLocation(NamedTuple):
    """Where a rule was defined"""

    file: str
    line: int


class ExecutionContext:
    """Marker base for all execution phase contexts."""

    __slots__ = ()


@dataclass(slots=True)
class RuleContext(ExecutionContext):
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
    When phase is not None, only processors registered at that priority will run.
    """

    event_stream: Optional[EventStream] = None
    phase: Optional[int] = None


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
    # Inclusive lower and exclusive upper priority bounds. None is unbounded.
    # The default preserves the historical "all pre-outcome refiners" behavior.
    phase_window: tuple[Optional[int], Optional[int]] = field(
        default_factory=lambda: (None, FilterPriority.POST_OUTCOME)
    )


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


@dataclass(slots=True)
class ExtractionContext(ExecutionContext):
    """Context for the extraction phase (processing observation source lines)."""

    source_type: ObservationSource
    job_info: Optional[Any]
    event_stream: Optional[EventStream]
    result: Optional[Any] = None  # holds ObservationProcessor; typed Any to avoid circular import


@dataclass(slots=True)
class CollectorContext(ExecutionContext):
    """Context for the collection phase (post-attribution digest)."""

    event_stream: Optional[EventStream]
    job_info: Optional[Any]
    rule_params: Optional[Mapping[RuleId, RuleParameters]] = None
    result: dict[str, Any] = field(default_factory=dict)


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


ExtractorProcessor: TypeAlias = Callable[[dict[str, str], Optional[str], Source, RuleContext], list[EventTriple]]
FilterProcessor: TypeAlias = Callable[[FilterContext], None]
RefinerProcessor: TypeAlias = Callable[[RefinerContext], None]
CollectorProcessor: TypeAlias = Callable[[EventStream, "JobInfo", Optional[RuleParamsBase]], Any]
CollectorFormatter: TypeAlias = Callable[[Any], dict[str, Any]]


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
    """Factory rule: trigger regex fires -> factory creates new ExtractorAccumulator instance."""

    match_re: str
    factory: Callable[[], ExtractorAccumulator]
    source_filters: frozenset[ObservationSource] = field(default_factory=frozenset)

    def process(self) -> ExtractorAccumulator:
        """Create a new extractor-accumulator instance."""
        return self.factory()


@dataclass(frozen=True, slots=True)
class RefinerRule(Rule):
    """Rule that runs on materialized symptoms and my_bad_nodes (post–event-stream)."""

    processor: RefinerProcessor
    priority: int = FilterPriority.FINAL

    def process(self, context: RefinerContext) -> None:
        """Process the refiner rule."""
        self.processor(context)


@dataclass(frozen=True, slots=True)
class CollectorRule(Rule):
    """Post-attribution collector: queries the event stream and formats digest data."""

    processor: CollectorProcessor
    formatter: CollectorFormatter
    params_model: Optional[type[RuleParamsBase]] = None

    def parse_params(self, raw: Optional[RuleParameters]) -> Optional[RuleParamsBase]:
        """Validate a parameter envelope against this rule's declared schema."""
        if self.params_model is None:
            if raw is not None and raw.params != {}:
                raise RuleAcceptsNoParamsError(self.name)
            return None

        params_data = raw.params if raw is not None else {}
        try:
            return self.params_model(**params_data)
        except ValidationError as e:
            raise InvalidRuleParams(self.name, e) from e

    def process(
        self, event_stream: EventStream, job_info: "JobInfo", params: Optional[RuleParamsBase] = None
    ) -> dict[str, Any]:
        """Run the collector and format results."""
        raw = self.processor(event_stream, job_info, params)
        return self.formatter(raw)


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
        return cast("set[RuleId]", matcher.candidates(datum))

    def confirm_match(self, source_type: ObservationSource, rule_id: RuleId, datum: str) -> Optional[Match]:
        """Stage 2: confirm a specific candidate with full regex."""
        matcher = self.by_source.get(source_type)
        if matcher is None:
            return None
        return matcher.confirm_match(rule_id, datum)
