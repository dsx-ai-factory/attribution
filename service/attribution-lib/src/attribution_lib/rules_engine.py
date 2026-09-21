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
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.context import DeclarativeContext, _using_module_globals, _using_rule_loader
from attribution_lib.events import EventStream
from attribution_lib.exceptions import (
    DuplicateRuleError,
    RuleIngestionError,
    UninitializedRuleEngineError,
    UnknownResourceError,
    UnknownRuleTypeError,
)
from attribution_lib.job_info import JobInfo
from attribution_lib.logline_filtering import log_ignored_node_line, log_ignored_time_line
from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
    Source as ObservationSource,
)
from attribution_lib.observation.processing import BodyFallbacks, EntryProtocol, EntrySourceBase
from attribution_lib.prefilters import MetricSpec, PreFilterCallable, PreFilterSpec
from attribution_lib.rule_types.bases import (
    _RULE_TYPE_REGISTRY,
    LoaderBase,
    SubEngineBase,
)
from attribution_lib.rule_types.bootstrap import load_builtin_rule_types
from attribution_lib.rule_types.definitions import (
    ExecutionContext,
    ExtractionContext,
    Rule,
    RuleId,
    RuleLocation,
)
from attribution_lib.rule_types.observation import ObservationProcessor
from attribution_lib.symptoms.declarations import SymptomDeclaration
from attribution_lib.symptoms.predicate_registry import SymptomPredicate


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

    def __init__(
        self,
        sub_engines: list[SubEngineBase],
        declarations: RuleDeclarations,
        pre_filters: Mapping[ObservationSource, PreFilterSpec],
        metric_specs: Mapping[ObservationSource, MetricSpec],
        declarative_emit: Mapping[str, str],
    ) -> None:
        self._engines = sub_engines
        self.declarations = declarations
        self.pre_filters = dict(pre_filters)
        self.metric_specs = dict(metric_specs)
        self.declarative_emit = dict(declarative_emit)

        combined: dict[RuleId, Rule] = {}
        names: set[str] = set()
        for engine in sub_engines:
            for store in (engine.rules, engine.extra_rules()):
                for rule_id, rule in store.items():
                    combined[rule_id] = rule
            names.update(engine.rules.keys())
            names.update(engine.extra_rule_names())
        self._rules = combined
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

    def get_pre_filter(self, source: ObservationSource) -> PreFilterSpec:
        """Return the pre-filter for ``source``, or an empty specification."""
        return self.pre_filters.get(source, PreFilterSpec())

    def get_metric_spec(self, source: ObservationSource) -> MetricSpec:
        """Return the metric specification for ``source``."""
        spec = self.metric_specs.get(source)
        if spec is None:
            raise UninitializedRuleEngineError(
                f"No metric spec registered for {source.name}. Was setup_rules() called?"
            )
        return spec

    # -- Rule views (computed once in __init__) ----------------------------

    @property
    def rules(self) -> dict[RuleId, Rule]:
        return self._rules

    def get_rule(self, rule_id: RuleId) -> Optional[Rule]:
        """Return a rule by ID, independent of its concrete rule type."""
        return self._rules.get(rule_id)

    @property
    def all_rule_names(self) -> set[str]:
        return self._all_rule_names

    @property
    def all_event_types(self) -> set[str]:
        return set(self.declarations.event_types)

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
        self.pre_filters: dict[ObservationSource, PreFilterSpec] = {}
        self.metric_specs: dict[ObservationSource, MetricSpec] = {}
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

    def register_pre_filter(self, source: ObservationSource, spec: PreFilterSpec) -> None:
        """Register a pre-filter specification; repeated declarations merge."""
        existing = self.pre_filters.get(source, PreFilterSpec())
        self.pre_filters[source] = PreFilterSpec(
            include_terms=[*existing.include_terms, *spec.include_terms],
            exclude_terms=[*existing.exclude_terms, *spec.exclude_terms],
            callables=[*existing.callables, *spec.callables],
        )

    def register_metric_spec(self, source: ObservationSource, spec: MetricSpec) -> None:
        """Register a metric specification; repeated declarations merge."""
        existing = self.metric_specs.get(source, MetricSpec())
        self.metric_specs[source] = MetricSpec(thresholds={**existing.thresholds, **spec.thresholds})

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
    def declarative_emit(self) -> dict[str, str]:
        """The declarative emit store exposed by the loader that owns it."""
        for loader in self._loaders:
            store = loader.declarative_emit_store()
            if store is not None:
                return store
        raise UnknownRuleTypeError("No registered loader exposes a declarative_emit store")

    # -- Snapshot and rollback --------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Capture every field a load may mutate."""
        return {
            "loaders": [loader.snapshot() for loader in self._loaders],
            "declared_rule_names": set(self.declared_rule_names),
            "declared_event_types": set(self.declared_event_types),
            "symptom_declarations": list(self.symptom_declarations),
            "symptom_predicates": dict(self.symptom_predicates),
            "pre_filters": dict(self.pre_filters),
            "metric_specs": dict(self.metric_specs),
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
        self.pre_filters = snap["pre_filters"]
        self.metric_specs = snap["metric_specs"]
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
            pre_filters=dict(self.pre_filters),
            metric_specs=dict(self.metric_specs),
            declarative_emit=dict(self.declarative_emit),
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
    """All state a single load() may mutate, for rollback on failure."""

    loader_snap: dict[str, Any]

    @classmethod
    def capture(cls, loader: "RuleLoader") -> "_LoadSnapshot":
        return cls(loader_snap=loader.snapshot())

    def restore(self, loader: "RuleLoader") -> None:
        loader.restore(self.loader_snap)


@dataclass(frozen=True, slots=True)
class SourceProcessingConfig:
    """Per-source-type configuration for process_observation_source."""

    time_start: Optional[Callable[[JobInfo], Optional[datetime]]] = None
    time_end: Optional[Callable[[JobInfo], Optional[datetime]]] = None
    """Legacy formats only: `entry_timeseries` filters on the producer-declared `context.time_interval` instead.

    TODO: drop both once every source accepts only `entry_timeseries`.
    """
    body_fallbacks: BodyFallbacks = BodyFallbacks()
    """Per line: what an `entry_timeseries` entry may parse out of its body; `year_fallback` is filled per job."""
    fallback_unknown_nodes: bool = False
    """Per line: a node that fails to resolve against the job keeps the line as resource ANY instead of dropping it."""
    requires_job_nodes: bool = False
    """Per job: skip the whole source when `job_info.nodes` is empty, since node checks pass anything then.

    Not `BodyFallbacks.require_node`, which drops a single entry that declares no node.
    """


# Populated by core.py via register_source_config(); keeps rules_engine
# free of source-specific imports.
SOURCE_CONFIGS: dict[ObservationSource, SourceProcessingConfig] = {}


def register_source_config(source: ObservationSource, config: SourceProcessingConfig) -> None:
    """Register processing configuration for an observation source type."""
    SOURCE_CONFIGS[source] = config


def body_fallbacks_for(source: ObservationSource, job_info: JobInfo) -> BodyFallbacks:
    """The source's `body_fallbacks`, with the year reference resolved against this job."""
    config = SOURCE_CONFIGS.get(source)
    fallbacks = config.body_fallbacks if config is not None else BodyFallbacks()
    if not fallbacks.parse_timestamp:
        return fallbacks
    # job_info.job_end_time, not the interval end: entry_timeseries intervals may be unbounded.
    return replace(fallbacks, year_fallback=job_info.job_end_time)


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
    if observation_source.context.format is ObservationFormat.ENTRY_TIMESERIES:
        # The declared interval is what the producer meant to cover, so the per-source window in
        # SOURCE_CONFIGS does not apply. A null bound is unbounded on that side.
        time_start = observation_source.context.time_interval.start
        time_end = observation_source.context.time_interval.end
    else:
        time_start = config.time_start(job_info) if config and config.time_start else None
        time_end = config.time_end(job_info) if config and config.time_end else None
    fallback_unknown_nodes = config.fallback_unknown_nodes if config else False

    raw_keywords: set[str] = set()

    # Build pre-filter state (once, not per-entry)
    pre_filter_spec = None if pre_filtered else rules_engine.pre_filters.get(source_type)
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
            # An entry with no timestamp is never dropped.
            if entry_time is not None:
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
