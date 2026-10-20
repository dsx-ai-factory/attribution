# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

from loguru import logger

import attribution_lib
from attribution_lib import nodename_utils
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.bundled_rules.common.scrub_helpers import get_clean_bad_nodes
from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.events import EventStream
from attribution_lib.filter import run_post_outcome_refiners, run_pre_outcome_refiners, run_scrub_phase_refiners
from attribution_lib.job_info import JobInfo, Scheduler
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.observation.processing import BodyFallbacks, EntrySourceBase
from attribution_lib.outcome import (
    OutcomeAttribution,
    calculate_outcome,
)
from attribution_lib.rule_types.definitions import CollectorContext, RuleId, RuleParameters
from attribution_lib.rules_engine import (
    SOURCE_CONFIGS,
    RuleEngine,
    RuleLoader,
    SourceProcessingConfig,
    process_observation_source,
    register_source_config,
)
from attribution_lib.sources import (
    SourceSet,
    SymptomSet,
)
from attribution_lib.symptoms import (
    BaseSymptom,
)
from attribution_lib.symptoms.registry_generation import registry_from_declarations

# Register source-type processing configs
register_source_config(
    ObservationSource.APPLOG,
    SourceProcessingConfig(
        body_fallbacks=BodyFallbacks(parse_node=True, parse_timestamp=True),
        fallback_unknown_nodes=True,
    ),
)
register_source_config(
    ObservationSource.HEALTH_CHECK,
    SourceProcessingConfig(
        # A line declaring no node is read for one and dropped if that fails.
        # An explicit timestamp is expected, else the entry is dropped.
        body_fallbacks=BodyFallbacks(parse_node=True, require_node=True, require_timestamp=True),
    ),
)
register_source_config(
    ObservationSource.SYSLOG,
    SourceProcessingConfig(
        # A line declaring no node is read for one and dropped if that fails.
        # An explicit timestamp is expected, else the entry is dropped.
        body_fallbacks=BodyFallbacks(parse_node=True, require_node=True, require_timestamp=True),
        requires_job_nodes=True,
    ),
)
register_source_config(
    ObservationSource.UFMLOG,
    SourceProcessingConfig(
        # The job node is named inside the line; the producer's own node is the fabric manager host.
        # Therefore, attempt to parse the node, but drop it if it's not found.
        body_fallbacks=BodyFallbacks(parse_node=True, require_node=True),
        requires_job_nodes=True,
    ),
)
register_source_config(
    ObservationSource.NODE_METRICS,
    SourceProcessingConfig(
        # A sample is about one node; one that names none must not count against the whole job.
        body_fallbacks=BodyFallbacks(require_node=True),
        requires_job_nodes=True,
    ),
)
register_source_config(
    ObservationSource.NVLINK_METRICS,
    SourceProcessingConfig(
        # Same as node_metrics: a sample naming no node must not count against the whole job.
        body_fallbacks=BodyFallbacks(require_node=True),
        requires_job_nodes=True,
    ),
)
register_source_config(
    ObservationSource.CSP_EVENT,
    SourceProcessingConfig(
        # An event is about one host at one moment; the body names neither, so an entry
        # declaring no node or no timestamp is dropped rather than parsed.
        body_fallbacks=BodyFallbacks(require_node=True, require_timestamp=True),
        requires_job_nodes=True,
    ),
)


def process_check(
    *,
    event_stream: EventStream,
    observation_source: EntrySourceBase,
    job_info: JobInfo,
    rules_engine: RuleEngine,
    pre_filtered: bool = True,
) -> set[str]:
    """Backward-compat wrapper — delegates to process_observation_source with pre_filtered defaulting to True."""
    return process_observation_source(
        event_stream=event_stream,
        observation_source=observation_source,
        job_info=job_info,
        rules_engine=rules_engine,
        pre_filtered=pre_filtered,
    )


def get_bundled_rules_path() -> Path:
    """Return the filesystem path to the bundled plugin rules shipped with this package."""
    return Path(str(files("attribution_lib").joinpath("bundled_rules")))


def setup_rules(
    *,
    node_mapping_file: Optional[str] = None,
    ignored_hostname_prefixes: Optional[list[str]] = None,
    node_domain_resolver: Optional[attribution_lib.NodeDomainResolver] = None,
    rules_paths: Optional[list[Path]] = None,
    force: bool = False,
):
    """
    Initialize the attribution library rules and configuration.

    Args:
        node_mapping_file: Optional path to CSV file containing node name to cloud instance ID & IP address mappings.
            CSV Format for node_mapping_file:
                Required columns:
                - id: Node hostname
                - instanceId: Cloud instance identifier
                - [optional] Any columns with IPv4 addresses for IP-to-node mapping
        ignored_hostname_prefixes: Optional list of hostname prefixes to ignore during attribution
            (e.g., ['monitor-', 'scanner-'])
        node_domain_resolver: Optional callback that maps a node hostname to a
            hardware domain identifier. Return None for nodes without a domain.
        rules_paths: Directories to scan for rule plugin files (``*.py`` and
            ``*.toml``, excluding ``*.test.toml``).
            - ``None`` (default): loads the bundled rules shipped with this package.
            - ``[]``: loads no rule files and creates an empty ``RuleEngine``.
            - Explicit list: loads only from the given paths. Use
              ``[get_bundled_rules_path(), my_extra_path]`` to extend the
              bundled set with additional rules.
        force: If True, re-initialize even if already initialized. Used for re-attribution with updated rules.

    """
    if attribution_lib.initialized and not force:
        logger.debug("attribution_lib already initialized, skipping (use force=True to re-initialize)")
        return

    if rules_paths is None:
        rules_paths = [get_bundled_rules_path()]

    attribution_lib.initialized = True
    attribution_lib.node_domain_resolver = node_domain_resolver

    # Reconfiguration must also clear a mapping from an earlier attribution run.
    nodename_utils.configure_node_mapping_file(node_mapping_file)
    attribution_lib.ip_to_nodenames = nodename_utils.get_ip_to_node_name_mapping() if node_mapping_file else {}

    if ignored_hostname_prefixes:
        attribution_lib.ignored_hostname_prefixes = ignored_hostname_prefixes

    attribution_lib._rules_engine = RuleLoader.load_all(rules_paths)

    # Compile loaded declarations into the runtime symptom registry once per
    # initialization. Per-job attribution reads this cached snapshot; it is only
    # rebuilt when rules are (re)loaded.
    attribution_lib._symptom_registry = registry_from_declarations(
        attribution_lib._rules_engine.symptom_declarations,
        predicates=attribution_lib._rules_engine.symptom_predicates,
    )


@dataclass(kw_only=True)
class CalcBadNodesAttributionResult:
    """The result of a calculate_bad_nodes call. Is intended to be filled incrementally, so that we could conceivably repeatedly fold new results together."""

    my_bad_nodes: JobNodeAttributions = field(
        default_factory=JobNodeAttributions
    )  ### See JobNodeAttributions doc string for more info on how we use this
    raw_keywords: SymptomSet[str] = field(default_factory=SymptomSet)
    node_symptoms_base: dict[str, SymptomSet[BaseSymptom]] = field(default_factory=dict)
    skipped_symptom_ids: set[str] = field(default_factory=set)

    def raw_keywords_update(self, raw_keywords: SymptomSet[str]):
        self.raw_keywords.update(raw_keywords)

    def update(self, other: "CalcBadNodesAttributionResult") -> "CalcBadNodesAttributionResult":
        self.my_bad_nodes.update(other.my_bad_nodes)
        self.raw_keywords.update(other.raw_keywords)
        self.node_symptoms_base.update(other.node_symptoms_base)
        self.skipped_symptom_ids.update(other.skipped_symptom_ids)
        return self


def calculate_bad_nodes(
    *,
    event_stream: EventStream,
    job_info: JobInfo,
    applog_in: Optional[EntrySourceBase] = None,
    epilog_in: Optional[EntrySourceBase] = None,
    syslog_in: Optional[EntrySourceBase] = None,
    ufmlog_in: Optional[EntrySourceBase] = None,
    prometheus_in: Optional[EntrySourceBase] = None,
    raw_keywords_in: Optional[EntrySourceBase] = None,
    node_metrics_in: Optional[EntrySourceBase] = None,
    nvlink_metrics_in: Optional[EntrySourceBase] = None,
    csp_event_in: Optional[EntrySourceBase] = None,
) -> CalcBadNodesAttributionResult:
    result = CalcBadNodesAttributionResult()
    if not job_info.job_start_time:
        return result

    if (raw_keywords := raw_keywords_in) is not None:
        for item in raw_keywords:
            result.raw_keywords.add(
                SourceSet(item.body()).add(item.attribution_source_base(None).derived(rule="raw keywords"))
            )

    # Process all observation sources through the rules engine
    observation_sources: list[Optional[EntrySourceBase]] = [
        applog_in,
        epilog_in,
        syslog_in,
        ufmlog_in,
        node_metrics_in,
        nvlink_metrics_in,
        csp_event_in,
    ]
    for obs_source in observation_sources:
        if obs_source is None:
            continue
        source_type = obs_source.context.source
        config = SOURCE_CONFIGS.get(source_type)
        if config is not None and config.requires_job_nodes and not job_info.nodes:
            continue
        source_raw_keywords = process_observation_source(
            event_stream=event_stream,
            observation_source=obs_source,
            job_info=job_info,
            rules_engine=attribution_lib.get_rules_engine(),
            pre_filtered=False,
        )
        for kw in source_raw_keywords:
            result.raw_keywords.add(job_info.source_value(kw, datum=None, rule="pre-filter include"))

    result.my_bad_nodes = JobNodeAttributions.from_event_stream(event_stream, job_info)

    return result


@dataclass(kw_only=True)
class AttributionResult:
    """The result of a call to "attribute"; contains all of the outputs. Has a reporting mechanism to feed outputs to the reporting mechansim in existing NVIDIA Attribution workloads."""

    bad_nodes_attr: CalcBadNodesAttributionResult
    symptoms: SymptomSet[str] = field(default_factory=SymptomSet)
    outcome_attr: SourceSet[OutcomeAttribution] = field(default_factory=lambda: SourceSet(OutcomeAttribution()))
    cancelled_by: Optional[str] = None
    event_stream: EventStream
    collectors: dict[str, Any] = field(default_factory=dict)


def calculate_symptoms(
    *,
    event_stream: EventStream,
    job_info: JobInfo,
    my_bad_nodes: JobNodeAttributions,
    raw_keywords: Optional[SymptomSet[str]] = None,
    extract_job_metadata_from_logfile: bool = False,
    bad_node_result: Optional["CalcBadNodesAttributionResult"] = None,
    recorder: Optional[DecisionRecorder] = None,
) -> SymptomSet[str]:
    """Calculate job-level symptoms from per-node attributions and raw keywords.

    Aggregates all per-node symptoms into a job-level SymptomSet, then runs
    refiner rules (including DAG cascade pruning as the final refiner).
    """
    raw_keywords = raw_keywords or SymptomSet()

    symptoms: SymptomSet[str] = SymptomSet()

    # Job-lifecycle fallbacks ("Job didn't start" for 0 nodes, "running without error"
    # for RUNNING) are the discrete JobStateFallback refiner, which runs last and
    # supersedes everything else.
    run_scrub_phase_refiners(event_stream, job_info, my_bad_nodes=my_bad_nodes, recorder=recorder)

    # Aggregate per-node symptoms into a job-level SymptomSet[str].
    # Refiners see the full unpruned set; the SuppressNonFatalJobSymptoms refiner
    # then drops "(non-fatal)" informational symptoms from the job level (they stay
    # on my_bad_nodes), and DAG cascade pruning runs last.
    for _node, node_group in my_bad_nodes.items():
        for symptom_str in list(node_group.symptoms()):
            symptoms.add(node_group.attributions[symptom_str])
    for symptom_str in list(my_bad_nodes.job_level.symptoms()):
        symptoms.add(my_bad_nodes.job_level[symptom_str])

    node_symptoms_base, skipped_symptom_ids = run_pre_outcome_refiners(
        event_stream,
        job_info,
        symptoms=symptoms,
        my_bad_nodes=my_bad_nodes,
        raw_keywords=raw_keywords,
        extract_job_metadata_from_logfile=extract_job_metadata_from_logfile,
        recorder=recorder,
    )

    # Populate the BaseSymptom view from DagCascadePruning results for reporting
    if bad_node_result is not None and node_symptoms_base is not None:
        bad_node_result.node_symptoms_base.update(node_symptoms_base)
        bad_node_result.skipped_symptom_ids.update(skipped_symptom_ids or set())

    return symptoms


def attribute(
    *,
    event_stream: EventStream,
    job_info: JobInfo,
    bad_node_result: CalcBadNodesAttributionResult,
    extract_job_metadata_from_logfile: bool,
    recorder: Optional[DecisionRecorder] = None,
    rule_params: Optional[Mapping[RuleId, RuleParameters]] = None,
    run_collectors: bool = False,
) -> AttributionResult:
    # KAI is accepted on the wire, but no KAI rules or logic exist yet.
    # Fail loudly rather than incorrectlyfallback to Slurm behavior.
    if job_info.scheduler is Scheduler.KAI:
        raise NotImplementedError("Attribution for scheduler 'kai' is not implemented yet")

    result = AttributionResult(event_stream=event_stream, bad_nodes_attr=bad_node_result)

    symp = calculate_symptoms(
        event_stream=event_stream,
        job_info=job_info,
        my_bad_nodes=bad_node_result.my_bad_nodes,
        raw_keywords=bad_node_result.raw_keywords,
        extract_job_metadata_from_logfile=extract_job_metadata_from_logfile,
        bad_node_result=bad_node_result,
        recorder=recorder,
    )

    result.symptoms.update(symp)

    # root_cause_symptom is just a placeholder. It will be updated later with rule engine 2.0
    result.outcome_attr = calculate_outcome(result.symptoms, recorder=recorder)

    # "Incomplete Info -> demote bad nodes to interesting" is the discrete
    # DemoteBadNodesOnIncompleteInfo post-outcome refiner (runs first below).
    result.bad_nodes_attr.my_bad_nodes = get_clean_bad_nodes(bad_node_result.my_bad_nodes, job_info.cluster)

    run_post_outcome_refiners(
        event_stream,
        job_info,
        symptoms=result.symptoms,
        my_bad_nodes=bad_node_result.my_bad_nodes,
        outcome=result.outcome_attr,
        cancelled_by=job_info.cancelled_by,
        attribution_result=result,
        bad_node_result=bad_node_result,
        recorder=recorder,
    )

    if run_collectors:
        collector_context = CollectorContext(event_stream=event_stream, job_info=job_info, rule_params=rule_params)
        attribution_lib.get_rules_engine().execute(collector_context)
        result.collectors = collector_context.result

    return result
