# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob
import hashlib
import importlib.util
import sys
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import (
    dataclass,
)
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.context import (  # noqa: F401 (re-exported public API)
    _EXPR_HELPER_RESERVED,
    EXPR_HELPER_INJECTED_PARAMS,
    DeclarativeContext,
    _using_module_globals,
    _using_rule_loader,
    const,
    expr_helper,
    rows,
    rule_processor,
)
from attribution_lib.events import EventStream
from attribution_lib.exceptions import (
    DuplicateRuleError,
    RuleIngestionError,
    UnknownResourceError,
    UnknownRuleTypeError,
)
from attribution_lib.job_info import JobInfo
from attribution_lib.logline_filtering import log_ignored_node_line, log_ignored_time_line
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.observation.processing import EntryProtocol, EntrySourceBase
from attribution_lib.prefilters import (  # noqa: F401 (re-exported public API)
    _METRIC_SPECS,
    _PRE_FILTERS,
    MetricSpec,
    PreFilterCallable,
    PreFilterSpec,
    register_metric_spec,
    register_pre_filter,
)
from attribution_lib.rule_types.bases import (  # noqa: F401 (re-exported public API)
    _RULE_TYPE_REGISTRY,
    LoaderBase,
    SubEngineBase,
)
from attribution_lib.rule_types.bootstrap import load_builtin_rule_types
from attribution_lib.rule_types.collector import CollectorEngine
from attribution_lib.rule_types.definitions import (  # noqa: F401 (re-exported public API)
    CollectorContext,
    CollectorRule,
    EventTriple,
    ExecutionContext,
    ExtractionContext,
    ExtractorAccumulator,
    ExtractorAccumulatorRule,
    ExtractorRule,
    FilterContext,
    FilterPriority,
    FilterRule,
    RefinerContext,
    RefinerRule,
    Rule,
    RuleContext,
    RuleId,
    RuleLocation,
    RuleParameters,
    RuleParamsBase,
)
from attribution_lib.rule_types.observation import ObservationProcessor  # noqa: F401 (re-exported)
from attribution_lib.rule_types.registration import (  # noqa: F401 (re-exported public API)
    _get_caller_location,
    collector_def,
    declare_event_types,
    declare_rule_names,
    extractor_accumulator_def,
    filter_def,
    normalize_tags,
    refiner_def,
)
from attribution_lib.symptoms.declarations import (  # noqa: F401 (re-exported public API)
    CascadeScope,
    DagDiscardBehavior,
    OutcomePriority,
    SymptomCategory,
    SymptomDeclaration,
)
from attribution_lib.symptoms.predicate_registry import SymptomPredicate  # noqa: F401 (re-exported)
from attribution_lib.symptoms.registration import (  # noqa: F401 (re-exported public API)
    cascade_symptom,
    known_symptom,
    outcome_symptom,
    symptom_predicate,
)


@dataclass(frozen=True, slots=True)
class RuleDeclarations:
    """Everything a load pass declares that is not itself a Rule."""

    rule_names: frozenset[str]
    event_types: frozenset[str]
    symptoms: tuple[SymptomDeclaration, ...]
    predicates: Mapping[str, SymptomPredicate]


class RuleEngine:
    """Execution coordinator. Built by RuleLoader.build(), then read-only.

    Holds one sub-engine per registered rule type and fans execute() out to
    whichever claims the context. Nothing here mutates after construction, so
    the derived views are computed once.
    """

    def __init__(self, sub_engines: list[SubEngineBase], declarations: RuleDeclarations) -> None:
        self._engines = sub_engines
        self.declarations = declarations

        combined: dict[RuleId, Rule] = {}
        names: set[str] = set()
        by_type: dict[type, dict[RuleId, Rule]] = {}
        for engine in sub_engines:
            for store in (engine.rules, engine.extra_rules()):
                for rule_id, rule in store.items():
                    combined[rule_id] = rule
                    by_type.setdefault(type(rule), {})[rule_id] = rule
            names.update(engine.rules.keys())
            names.update(engine.extra_rule_names())
        self._rules = combined
        self._by_type = by_type
        self._all_rule_names = names | set(declarations.rule_names)

    # -- Declarations ------------------------------------------------------

    @property
    def declared_rule_names(self) -> set[str]:
        return set(self.declarations.rule_names)

    @property
    def declared_event_types(self) -> set[str]:
        return set(self.declarations.event_types)

    @property
    def symptom_declarations(self) -> list[SymptomDeclaration]:
        return list(self.declarations.symptoms)

    @property
    def symptom_predicates(self) -> dict[str, SymptomPredicate]:
        return dict(self.declarations.predicates)

    # -- Rule views (computed once in __init__) ----------------------------

    @property
    def rules(self) -> dict[RuleId, Rule]:
        return self._rules

    @property
    def all_rule_names(self) -> set[str]:
        return self._all_rule_names

    @property
    def all_event_types(self) -> set[str]:
        return set(self.declarations.event_types)

    def rules_of_type(self, rule_type_: type) -> dict[RuleId, Rule]:
        """Every registered rule of exactly ``rule_type_``."""
        return self._by_type.get(rule_type_, {})

    @property
    def extractor_rules(self) -> "dict[RuleId, ExtractorRule]":
        return cast("dict[RuleId, ExtractorRule]", self.rules_of_type(ExtractorRule))

    @property
    def extractor_accumulator_rules(self) -> "dict[RuleId, ExtractorAccumulatorRule]":
        return cast("dict[RuleId, ExtractorAccumulatorRule]", self.rules_of_type(ExtractorAccumulatorRule))

    @property
    def filter_rules(self) -> "dict[RuleId, FilterRule]":
        return cast("dict[RuleId, FilterRule]", self.rules_of_type(FilterRule))

    @property
    def refiner_rules(self) -> "dict[RuleId, RefinerRule]":
        return cast("dict[RuleId, RefinerRule]", self.rules_of_type(RefinerRule))

    @property
    def collector_rules(self) -> "dict[RuleId, CollectorRule]":
        return cast("dict[RuleId, CollectorRule]", self.rules_of_type(CollectorRule))

    # -- Execution ---------------------------------------------------------

    def execute(self, context: ExecutionContext) -> None:
        """Fan out to every sub-engine claiming this context type.

        A context nothing claims is an error, not a silent no-op: it means a
        rule type is missing from the registry.
        """
        claimed = False
        for engine in self._engines:
            if engine.handles(context) is False:
                continue
            claimed = True
            engine.execute(context)
        if claimed is False:
            raise UnknownRuleTypeError(
                f"No sub-engine handles {type(context).__name__}; a rule type may be unregistered"
            )

    def observation_processor(
        self,
        source_type: ObservationSource,
        job_info: JobInfo,
        event_stream: EventStream,
    ) -> "ObservationProcessor":
        """Create an ObservationProcessor for the given source type."""
        ctx = ExtractionContext(source_type=source_type, job_info=job_info, event_stream=event_stream)
        self.execute(ctx)
        return cast("ObservationProcessor", ctx.result)

    def process_event_filters(
        self,
        event_stream: EventStream,
        job_info: JobInfo,
        *,
        phase: Optional[int] = None,
    ) -> None:
        """Run filter rules as a compatibility wrapper over ``execute()``."""
        self.execute(FilterContext(job_info=job_info, event_stream=event_stream, phase=phase))

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
        """Run refiner rules as a compatibility wrapper over ``execute()``."""
        ctx = RefinerContext(
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
            phase=phase,
        )
        self.execute(ctx)
        return ctx

    def run_collectors(
        self,
        event_stream: EventStream,
        job_info: JobInfo,
        rule_params: Optional[Mapping[RuleId, RuleParameters]] = None,
    ) -> dict[str, Any]:
        """Run collectors as a compatibility wrapper over ``execute()``."""
        ctx = CollectorContext(event_stream=event_stream, job_info=job_info, rule_params=rule_params)
        self.execute(ctx)
        return ctx.result

    @staticmethod
    def parse_collector_params(rule: CollectorRule, raw: Optional[RuleParameters]) -> Optional[RuleParamsBase]:
        """Delegates to CollectorEngine.parse_params()."""
        return CollectorEngine.parse_params(rule, raw)

    # -- Cache keys --------------------------------------------------------

    def extractor_hash(self) -> str:
        """Stable hash of all extractor rules. Changes when extraction behavior changes.

        Contributions are folded in sorted order rather than sub-engine order,
        so renaming a rule_types module cannot silently change every key.

        TODO: evolve into a versioned scheme that can detect compatible vs breaking
        rule changes (e.g. additive rules could reuse existing cache + incremental extract).
        """
        h = hashlib.sha256()
        for contribution in sorted(c for e in self._engines if len(c := e.extractor_hash_contribution()) > 0):
            h.update(contribution.encode())
        return h.hexdigest()

    def attribution_hash(self) -> str:
        """Stable hash of filter/refiner/collector rules. Changes when attribution logic changes."""
        h = hashlib.sha256()
        for contribution in sorted(c for e in self._engines if len(c := e.attribution_hash_contribution()) > 0):
            h.update(contribution)
        return h.hexdigest()


class RuleLoader:
    """Accumulates rules during a load pass, then builds a RuleEngine.

    Mutable and validating, where RuleEngine is neither. One sub-loader per
    registered rule type; dispatch goes through a table built from the
    registry, so no rule type is named here.
    """

    def __init__(self) -> None:
        load_builtin_rule_types()
        self._loaders: list[LoaderBase] = [entry.loader_class() for entry in _RULE_TYPE_REGISTRY]
        self._dispatch: dict[type[Rule], LoaderBase] = {
            rt: loader for entry, loader in zip(_RULE_TYPE_REGISTRY, self._loaders) for rt in entry.rule_types
        }
        self.declared_rule_names: set[str] = set()
        self.declared_event_types: set[str] = set()
        self.symptom_declarations: list[SymptomDeclaration] = []
        self.symptom_predicates: dict[str, SymptomPredicate] = {}
        self.declarative_ctx = DeclarativeContext()
        self._context_scopes: list[AbstractContextManager["RuleLoader"]] = []

    # -- Registration ------------------------------------------------------

    def register_rule(self, rule: Rule) -> None:
        """Register a rule, dispatching to its sub-loader by exact type.

        Dispatch is by exact type rather than isinstance: a Rule subclass with
        no registered pair is a mistake worth reporting, not something to
        silently route to its base's loader.
        """
        loader = self._dispatch.get(type(rule))
        if loader is None:
            raise UnknownRuleTypeError(f"Unknown rule type: {type(rule)}")
        for other in self._loaders:
            existing = other.find_by_name(rule.name)
            if existing is not None:
                raise DuplicateRuleError(
                    f"Duplicate rule '{rule.name}': "
                    f"first at {existing.location.file}:{existing.location.line}, "
                    f"duplicate at {rule.location.file}:{rule.location.line}"
                )
        loader.register(rule)

    def register_symptom_declaration(self, declaration: SymptomDeclaration) -> None:
        """Register declarative symptom metadata; rejects conflicting declarations."""
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
        """Register a conditional-discard predicate; rejects a conflicting callable."""
        existing = self.symptom_predicates.get(name)
        if existing is not None and existing is not predicate:
            raise RuleIngestionError(f"Conflicting symptom predicate registered for {name!r}")
        self.symptom_predicates[name] = predicate

    # -- Sub-loader access -------------------------------------------------

    @property
    def rules(self) -> dict[RuleId, Rule]:
        """Every rule registered so far, across all sub-loaders."""
        combined: dict[RuleId, Rule] = {}
        for loader in self._loaders:
            combined.update(cast(dict[RuleId, Rule], loader.rules))
            combined.update(cast(dict[RuleId, Rule], loader.extra_rules()))
        return combined

    @property
    def extractor_rules(self) -> "dict[RuleId, ExtractorRule]":
        """The extractor sub-loader's rule dict."""
        return cast("dict[RuleId, ExtractorRule]", self._dispatch[ExtractorRule].rules)

    @property
    def declarative_emit(self) -> dict[str, str]:
        """The extractor sub-loader's declarative_emit store."""
        store = self._dispatch[ExtractorRule].declarative_emit_store()
        if store is None:
            raise UnknownRuleTypeError("The extractor loader exposes no declarative_emit store")
        return store

    # -- Snapshot and rollback --------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Capture every field a load may mutate."""
        return {
            "loaders": [loader.snapshot() for loader in self._loaders],
            "declared_rule_names": set(self.declared_rule_names),
            "declared_event_types": set(self.declared_event_types),
            "symptom_declarations": list(self.symptom_declarations),
            "symptom_predicates": dict(self.symptom_predicates),
            "declarative_ctx": self.declarative_ctx.snapshot(),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore state captured by snapshot()."""
        for loader, loader_snap in zip(self._loaders, snap["loaders"]):
            loader.restore(loader_snap)
        self.declared_rule_names = snap["declared_rule_names"]
        self.declared_event_types = snap["declared_event_types"]
        self.symptom_declarations = snap["symptom_declarations"]
        self.symptom_predicates = snap["symptom_predicates"]
        self.declarative_ctx.restore(snap["declarative_ctx"])

    @contextmanager
    def _load_guard(self) -> Generator[None, None, None]:
        """Roll back every load-mutable field if the load raises."""
        snap = _LoadSnapshot.capture(self)
        try:
            yield
        except Exception:
            snap.restore(self)
            raise

    # -- Loading -----------------------------------------------------------

    def __enter__(self) -> "RuleLoader":
        scope = _using_rule_loader(self)
        scope.__enter__()
        self._context_scopes.append(scope)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if not self._context_scopes:
            raise RuntimeError("RuleLoader context is not active")
        scope = self._context_scopes.pop()
        scope.__exit__(*exc_info)

    def load(self, filepath: str | Path) -> dict[RuleId, Rule]:
        """Load one rule file, TOML or Python, and return the rules it added.

        Binds itself for the duration, so a bare ``loader.load()`` works as
        well as a ``with`` block.
        """
        path = Path(filepath)
        if path.suffix == ".toml":
            with self._load_guard(), _using_rule_loader(self):
                return load_toml_rules(self, str(path))

        rules_before = set(self.rules)
        with self._load_guard(), _using_rule_loader(self):
            spec = importlib.util.spec_from_file_location("rules_module", str(path))
            if spec is None or spec.loader is None:
                raise RuleIngestionError(f"Cannot load module from {path}")

            module = importlib.util.module_from_spec(spec)

            # Temporarily add rule file's parent directory to sys.path so
            # rules can import from sibling packages (e.g. common/).
            parent_dir = str(path.resolve().parent)
            sys_path_snapshot = sys.path.copy()
            sys.path.insert(0, parent_dir)

            try:
                with _using_module_globals(module.__dict__):
                    spec.loader.exec_module(module)
            finally:
                sys.path[:] = sys_path_snapshot

            # Return only the rules that were added by this module
            return {name: rule for name, rule in self.rules.items() if name not in rules_before}

    def build(self) -> RuleEngine:
        """Produce the immutable engine for everything registered so far."""
        return RuleEngine(
            sub_engines=[loader.build() for loader in self._loaders],
            declarations=RuleDeclarations(
                rule_names=frozenset(self.declared_rule_names),
                event_types=frozenset(self.declared_event_types),
                symptoms=tuple(self.symptom_declarations),
                predicates=dict(self.symptom_predicates),
            ),
        )

    @classmethod
    def load_all(cls, rules_paths: Iterable[Path]) -> RuleEngine:
        """Load every rule file under each path and build the engine."""
        with cls() as loader:
            for path in rules_paths:
                for pattern in ("*.py", "*.toml"):
                    for file in sorted(glob.glob(pattern, root_dir=path)):
                        if file.endswith(".test.toml"):
                            continue  # co-located test files are run by the suite, not loaded
                        loader.load(str(path / file))
            return loader.build()


@dataclass
class _LoadSnapshot:
    """All state a single load() may mutate, for rollback on failure.

    The loader captures its own fields; the two module-level registries in
    prefilters are captured here as part of the same transaction.
    """

    loader_snap: dict[str, Any]
    pre_filters: dict[ObservationSource, PreFilterSpec]
    metric_specs: dict[ObservationSource, MetricSpec]

    @classmethod
    def capture(cls, loader: "RuleLoader") -> "_LoadSnapshot":
        return cls(
            loader_snap=loader.snapshot(),
            pre_filters={
                source: PreFilterSpec(
                    include_terms=list(spec.include_terms),
                    exclude_terms=list(spec.exclude_terms),
                    callables=list(spec.callables),
                )
                for source, spec in _PRE_FILTERS.items()
            },
            metric_specs={
                source: MetricSpec(thresholds=dict(spec.thresholds)) for source, spec in _METRIC_SPECS.items()
            },
        )

    def restore(self, loader: "RuleLoader") -> None:
        loader.restore(self.loader_snap)
        _PRE_FILTERS.clear()
        _PRE_FILTERS.update(self.pre_filters)
        _METRIC_SPECS.clear()
        _METRIC_SPECS.update(self.metric_specs)


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


__all__ = [
    # Rule types and the accumulator protocol
    "CollectorRule",
    "ExtractorAccumulator",
    "ExtractorAccumulatorRule",
    "ExtractorRule",
    "FilterRule",
    "RefinerRule",
    "Rule",
    # Execution contexts
    "CollectorContext",
    "ExecutionContext",
    "ExtractionContext",
    "FilterContext",
    "RefinerContext",
    "RuleContext",
    # Values a rule file declares or receives
    "EventTriple",
    "FilterPriority",
    "RuleId",
    "RuleLocation",
    "RuleParameters",
    "RuleParamsBase",
    # Processor and formatter signatures
    # Rule registration
    "collector_def",
    "declare_event_types",
    "declare_rule_names",
    "extractor_accumulator_def",
    "filter_def",
    "normalize_tags",
    "refiner_def",
    # Declarative (TOML) registration
    "EXPR_HELPER_INJECTED_PARAMS",
    "const",
    "expr_helper",
    "rows",
    "rule_processor",
    # Symptom declaration
    "cascade_symptom",
    "known_symptom",
    "outcome_symptom",
    "symptom_predicate",
    # Pre-filter, metric and source configuration
    "MetricSpec",
    "PreFilterCallable",
    "PreFilterSpec",
    "SOURCE_CONFIGS",
    "SourceProcessingConfig",
    "register_metric_spec",
    "register_pre_filter",
    "register_source_config",
    # Engine and execution
    "CollectorEngine",
    "LoaderBase",
    "ObservationProcessor",
    "SubEngineBase",
    "RuleEngine",
    "RuleLoader",
    "process_observation_source",
    # Compatibility exports owned by exceptions and symptoms.declarations.
    "CascadeScope",
    "DagDiscardBehavior",
    "DuplicateRuleError",
    "OutcomePriority",
    "RuleIngestionError",
    "SymptomCategory",
    "SymptomDeclaration",
    "SymptomPredicate",
]
