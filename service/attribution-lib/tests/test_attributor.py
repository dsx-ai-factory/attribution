# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import attribution_lib
from attribution_lib.attributor import (
    Attributor,
    ConsumedObservation,
    IngestedObservation,
)
from attribution_lib.checkpoint_backends import MemoryCheckpointBackend
from attribution_lib.core import CalcBadNodesAttributionResult
from attribution_lib.events import EventManager
from attribution_lib.exceptions import (
    DuplicateObservationError,
    MismatchedSourceAndFormatError,
    UnhandledObservationSourceError,
)
from attribution_lib.hostlist import parse_hostlist
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation import (
    Observation,
)
from attribution_lib.observation.inbound import (
    Context as ObservationContext,
    Format as ObservationFormat,
    Observation as InboundObservation,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rules_engine import (
    _PRE_FILTERS,
    EventTriple,
    ExtractorRule,
    PreFilterSpec,
    RuleEngine,
    RuleLocation,
    register_pre_filter,
)
from test_support.helpers import make_node_to_ranks, make_syslog_entry

# APPLOG registers "error" as an include-keyword.
APPLOG_INCLUDE_TERM = "error"


@pytest.fixture
def job_info():
    tz = ZoneInfo("America/Los_Angeles")
    return (
        JobInfoBuilder(
            id="test_job_123",
            node_to_ranks=make_node_to_ranks(parse_hostlist("node[1-3]"), 1),
            framework="pytorch",
            type="batch",
            state="FAILED",
            cluster="test_cluster",
            job_start_time=datetime(2024, 1, 1, 12, 0, tzinfo=tz),
            source_name="artifical slurm",
            timezone=tz,
        )
        .srun_cmd("mpirun -n 3 ./test_app")
        .exit_code_signal(1)
        .slurm_node_infos({"node1": {"state": "idle"}, "node2": {"state": "idle"}, "node3": {"state": "idle"}})
        .job_end_time(datetime(2024, 1, 1, 13, 0, tzinfo=tz))
        .build()
    )


def make_inbound_observation(source: ObservationSource, fmt: ObservationFormat, *, pre_filtered: bool = False):
    timezone = ZoneInfo("America/Los_Angeles")
    time_interval = TimeInterval(
        start=datetime(2024, 1, 1, 12, 0, tzinfo=timezone), end=datetime(2024, 1, 1, 13, 0, tzinfo=timezone)
    )
    resources = AllJobResources
    context = ObservationContext(
        time_interval=time_interval,
        resources=resources,
        source=source,
        format=fmt,
        timezone=timezone,
        pre_filtered=pre_filtered,
    )
    lines = [APPLOG_INCLUDE_TERM, "failure", "crash"]
    return InboundObservation(context, lines)


def make_observation(source: ObservationSource, fmt: ObservationFormat):
    return Observation.from_inbound_observation(make_inbound_observation(source, fmt))


@pytest.fixture
def sample_observation():
    return make_inbound_observation(ObservationSource.JOB_KEYWORDS, fmt=ObservationFormat.STRING_LIST)


def test_attributor_creation(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    assert attributor.job_info == job_info
    assert attributor.ingested_observations == {}


def test_observation_ingestion(request, event_manager, job_info, sample_observation):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    ingest_id = attributor.ingest(sample_observation)

    assert ingest_id in attributor.ingested_observations
    assert len(attributor.ingested_observations) == 1

    ingested = attributor.ingested_observations[ingest_id]
    assert isinstance(ingested, IngestedObservation)
    assert ingested.consumed is not None
    assert ingested.unconsumed is None


def test_duplicate_observation(request, event_manager, job_info, sample_observation):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    attributor.ingest(sample_observation)

    with pytest.raises(DuplicateObservationError):
        attributor.ingest(sample_observation)


def test_successful_attribution(request, event_manager, job_info, sample_observation):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    attributor.ingest(sample_observation)

    result = attributor.attribute()
    assert result is not None
    # Add more specific assertions about the attribution result based on your needs


@pytest.mark.parametrize(
    "line",
    [
        "System is rebooting.",
        "sudo: alice : TTY=pts/0 ; PWD=/tmp ; USER=root ; COMMAND=/usr/bin/systemctl reboot",
    ],
)
def test_initiated_reboot_log_produces_infra_sw_with_raw_evidence(request, event_manager, job_info, line):
    """Attribute explicit reboot logs to infrastructure software.

    Args:
        request: Pytest request fixture used to name the event stream.
        event_manager: Event manager fixture used to create the event stream.
        job_info: Job information fixture for the attribution run.
        line: Reboot log line that must retain raw evidence.
    """
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    reboot_time = job_info.job_start_time + timedelta(minutes=1)
    context = ObservationContext(
        time_interval=TimeInterval(job_info.job_start_time, job_info.job_end_time),
        resources=AllJobResources,
        source=ObservationSource.SYSLOG,
        format=ObservationFormat.RAW_LOKI_STREAMS,
        timezone=job_info.timezone,
    )
    observation = InboundObservation(
        context,
        [
            make_syslog_entry(
                line,
                node="node1",
                timestamp_ns=int(reboot_time.timestamp() * 1e9),
            )
        ],
    )

    attributor.ingest(observation)
    result = attributor.attribute()

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail InfraSW"
    assert outcome.hw_cause == "Not Applicable"
    assert "Node reboot initiated" in result.bad_nodes_attr.my_bad_nodes["node1"]
    assert result.event_stream.query_raw("SELECT source_datum FROM events WHERE source_datum = ?", [line])


def test_ingested_observation_state(sample_observation):
    observation = Observation.from_inbound_observation(sample_observation)
    ingested = IngestedObservation(observation)
    assert ingested.unconsumed is not None
    assert ingested.unconsumed.context.id == observation.context.id
    assert ingested.consumed is None
    assert isinstance(ingested.unconsumed.context.id, int)
    assert isinstance(ingested.ingestion_time, datetime)


def test_consumed_observation_creation(request, event_manager, job_info, sample_observation):
    event_stream = event_manager.create_stream(request.node.name)
    consumed = ConsumedObservation.new(event_stream, job_info, sample_observation)
    assert isinstance(consumed, ConsumedObservation)
    assert consumed.context == sample_observation.context
    assert isinstance(consumed.result, CalcBadNodesAttributionResult)


def test_source_format_job_keywords_mismatch(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    bad_formats = frozenset(ObservationFormat) - frozenset([ObservationFormat.UNDEFINED, ObservationFormat.STRING_LIST])

    for fmt in bad_formats:
        observation = make_observation(ObservationSource.JOB_KEYWORDS, fmt)
        with pytest.raises(MismatchedSourceAndFormatError):
            _ = ConsumedObservation.new(event_stream, job_info, observation)


def test_source_format_applog_mismatch(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    bad_formats = frozenset(ObservationFormat) - frozenset(
        [
            ObservationFormat.UNDEFINED,
            ObservationFormat.RAW_LOG_LINES,
            ObservationFormat.LOKI_DICT_LIST,
            ObservationFormat.RAW_LOKI_STREAMS,
        ]
    )

    for fmt in bad_formats:
        observation = make_observation(ObservationSource.APPLOG, fmt)
        with pytest.raises(MismatchedSourceAndFormatError):
            _ = ConsumedObservation.new(event_stream, job_info, observation)


def test_source_format_syslog_mismatch(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    bad_formats = frozenset(ObservationFormat) - frozenset(
        [
            ObservationFormat.UNDEFINED,
            ObservationFormat.LOKI_DICT_LIST,
            ObservationFormat.RAW_LOKI_STREAMS,
        ]
    )

    for fmt in bad_formats:
        observation = make_observation(ObservationSource.SYSLOG, fmt)
        with pytest.raises(MismatchedSourceAndFormatError):
            _ = ConsumedObservation.new(event_stream, job_info, observation)


def test_source_format_health_check_mismatch(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    bad_formats = frozenset(ObservationFormat) - frozenset(
        [
            ObservationFormat.UNDEFINED,
            ObservationFormat.LOKI_DICT_LIST,
            ObservationFormat.RAW_LOKI_STREAMS,
        ]
    )

    for fmt in bad_formats:
        observation = make_observation(ObservationSource.HEALTH_CHECK, fmt)
        with pytest.raises(MismatchedSourceAndFormatError):
            _ = ConsumedObservation.new(event_stream, job_info, observation)


def test_source_format_metrics_mismatch(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    bad_formats = frozenset(ObservationFormat) - frozenset(
        [ObservationFormat.UNDEFINED, ObservationFormat.PROMETHEUS_METRICS]
    )

    for fmt in bad_formats:
        observation = make_observation(ObservationSource.NODE_METRICS, fmt)
        with pytest.raises(MismatchedSourceAndFormatError):
            _ = ConsumedObservation.new(event_stream, job_info, observation)


def test_source_format_csp_event_mismatch(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    bad_formats = frozenset(ObservationFormat) - frozenset(
        [ObservationFormat.UNDEFINED, ObservationFormat.CSP_EVENT_LIST]
    )

    for fmt in bad_formats:
        observation = make_observation(ObservationSource.CSP_EVENT, fmt)
        with pytest.raises(MismatchedSourceAndFormatError):
            _ = ConsumedObservation.new(event_stream, job_info, observation)


def test_unhandled_sources(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    unhandled_sources = frozenset((list(ObservationSource))) - frozenset(
        [
            ObservationSource.UNDEFINED,
            ObservationSource.APPLOG,
            ObservationSource.SYSLOG,
            ObservationSource.HEALTH_CHECK,
            ObservationSource.NODE_METRICS,
            ObservationSource.NVLINK_METRICS,
            ObservationSource.JOB_KEYWORDS,
            ObservationSource.UFMLOG,
            ObservationSource.CSP_EVENT,
        ]
    )
    for source in unhandled_sources:
        observation = make_observation(source, ObservationFormat.RAW_LOG_LINES)
        with pytest.raises(UnhandledObservationSourceError):
            _ = ConsumedObservation.new(event_stream, job_info, observation)


def test_ingest_stores_fingerprints(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    obs = make_inbound_observation(ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES)
    attributor.ingest(obs)
    assert "applog" in attributor._source_fingerprints
    assert attributor._source_fingerprints["applog"].count == 3


def test_ingest_skips_fingerprint_for_keywords(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    obs = make_inbound_observation(ObservationSource.JOB_KEYWORDS, ObservationFormat.STRING_LIST)
    attributor.ingest(obs)
    assert attributor._source_fingerprints == {}


def test_attribute_preingested(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    attributor.ingest(make_inbound_observation(ObservationSource.JOB_KEYWORDS, ObservationFormat.STRING_LIST))

    backend = MemoryCheckpointBackend()
    result = attributor.attribute(backend)
    assert result is not None
    assert result.outcome_attr is not None


def test_attribute_preingested_interleave(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    backend = MemoryCheckpointBackend()

    attributor.ingest(make_inbound_observation(ObservationSource.JOB_KEYWORDS, ObservationFormat.STRING_LIST))
    result1 = attributor.attribute(backend)

    attributor.ingest(make_inbound_observation(ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES))
    result2 = attributor.attribute(backend)

    assert result1 is not None
    assert result2 is not None


def test_attribute_preingested_consistent(request, event_manager, job_info):
    """Calling attribute twice without new data produces the same outcome."""
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    backend = MemoryCheckpointBackend()

    attributor.ingest(make_inbound_observation(ObservationSource.JOB_KEYWORDS, ObservationFormat.STRING_LIST))

    result1 = attributor.attribute(backend)
    result2 = attributor.attribute(backend)

    assert result1.outcome_attr.value().outcome == result2.outcome_attr.value().outcome


def test_attribute_preingested_mutation_safe(request, event_manager, job_info):
    """Event count is preserved after attribute (mutation undone)."""
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    attributor.ingest(make_inbound_observation(ObservationSource.JOB_KEYWORDS, ObservationFormat.STRING_LIST))
    attributor.ingest(make_inbound_observation(ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES))
    event_stream.flush()

    count_before = event_manager._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    backend = MemoryCheckpointBackend()
    attributor.attribute(backend)

    count_after = event_manager._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count_after == count_before


def test_attribute_preingested_skips_save_when_unchanged(request, event_manager, job_info):
    """Second call without new data skips saving to backend."""
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    backend = MemoryCheckpointBackend()

    attributor.ingest(make_inbound_observation(ObservationSource.JOB_KEYWORDS, ObservationFormat.STRING_LIST))
    attributor.attribute(backend)

    metadata_after_first = backend.load_metadata(attributor.get_cache_key(job_info))
    assert metadata_after_first is not None

    attributor.attribute(backend)

    metadata_after_second = backend.load_metadata(attributor.get_cache_key(job_info))
    assert metadata_after_second == metadata_after_first


def test_attribute_batch(request, event_manager, job_info):
    """Batch path: ingest_batch() + attribute() produces a result."""
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    backend = MemoryCheckpointBackend()

    observations = [make_inbound_observation(ObservationSource.JOB_KEYWORDS, ObservationFormat.STRING_LIST)]
    attributor.ingest_batch(observations, backend)
    result = attributor.attribute(backend)
    assert result is not None


def test_running(request, event_manager):
    tz = ZoneInfo("America/Los_Angeles")
    job_info = (
        JobInfoBuilder(
            id="test_job_123",
            node_to_ranks=make_node_to_ranks(parse_hostlist("node[1-3]"), 1),
            framework="pytorch",
            type="batch",
            state="RUNNING",
            cluster="test_cluster",
            job_start_time=datetime(2024, 1, 1, 12, 0, tzinfo=tz),
            source_name="artifical slurm",
            timezone=tz,
        )
        .srun_cmd("mpirun -n 3 ./test_app")
        .slurm_node_infos({"node1": {"state": "idle"}, "node2": {"state": "idle"}, "node3": {"state": "idle"}})
        .interest_end_time(datetime(2024, 1, 2, 12, 0, tzinfo=tz))
        .build()
    )
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)

    result = attributor.attribute()
    outcome_attr = result.outcome_attr.value()
    assert outcome_attr.root_cause_symptom == "running without error"
    assert outcome_attr.hw_cause == "Not Applicable"
    assert outcome_attr.outcome == "Pass"


_EXCLUDE_SENTINEL = "zzz_test_exclude_sentinel"
_EXTRACTOR_TRIGGER = "zzz_test_extractor_trigger"
_TEST_RULE_NAME = "zzz_test_prefilter_extractor"
_TEST_EVENT_TYPE = "zzz_test_prefilter_event"


def _emit_one_event(groups, hostname, source, context):
    return [EventTriple(type=_TEST_EVENT_TYPE, payload={}, source=source)]


@pytest.fixture
def prefilter_env():
    """Process-global rules engine + prefilter carrying only test-local rules.

    Installs a fresh engine with one APPLOG extractor (fires on the trigger sentinel) and an
    exclude-only prefilter, then restores both globals on teardown so nothing leaks."""
    engine = RuleEngine()
    engine.register_rule(
        ExtractorRule(
            name=_TEST_RULE_NAME,
            author="test",
            location=RuleLocation(__file__, 0),
            match_re=_EXTRACTOR_TRIGGER,
            processor=_emit_one_event,
            source_filters=frozenset({ObservationSource.APPLOG}),
        )
    )
    engine.declared_event_types.add(_TEST_EVENT_TYPE)

    saved_engine = attribution_lib._rules_engine
    saved_filters = copy.deepcopy(_PRE_FILTERS)
    attribution_lib._rules_engine = engine
    _PRE_FILTERS.clear()
    register_pre_filter(ObservationSource.APPLOG, PreFilterSpec(exclude_terms=[_EXCLUDE_SENTINEL]))

    manager = EventManager(rules_engine=engine)
    try:
        yield manager
    finally:
        manager.close()
        _PRE_FILTERS.clear()
        _PRE_FILTERS.update(saved_filters)
        attribution_lib._rules_engine = saved_engine


def _count_events_after_applog_ingest(event_manager, job_info, line, *, pre_filtered, pre_filtered_sources=None):
    tz = ZoneInfo("America/Los_Angeles")
    stream = event_manager.create_stream(f"prefilter_{pre_filtered}_{bool(pre_filtered_sources)}")
    attributor = Attributor.new(stream, job_info, pre_filtered_sources=pre_filtered_sources)
    context = ObservationContext(
        time_interval=TimeInterval(
            start=datetime(2024, 1, 1, 12, 0, tzinfo=tz),
            end=datetime(2024, 1, 1, 13, 0, tzinfo=tz),
        ),
        resources=AllJobResources,
        source=ObservationSource.APPLOG,
        format=ObservationFormat.RAW_LOG_LINES,
        timezone=tz,
        pre_filtered=pre_filtered,
    )
    attributor.ingest(InboundObservation(context, [line]))
    stream.flush()
    return event_manager._conn.execute("SELECT COUNT(*) FROM events WHERE context_id=?", [stream.ctx_id]).fetchone()[0]


def test_prefiltered_context_keeps_excluded_extractor_line(prefilter_env, job_info):
    """pre_filtered=True skips the exclude phase, so an APPLOG line carrying the registered exclude
    term survives to emit an event; pre_filtered=False drops it in the pre-filter pass first."""
    line = f"[rank 0] {_EXCLUDE_SENTINEL} {_EXTRACTOR_TRIGGER}"
    kept = _count_events_after_applog_ingest(prefilter_env, job_info, line, pre_filtered=True)
    dropped = _count_events_after_applog_ingest(prefilter_env, job_info, line, pre_filtered=False)
    assert dropped == 0
    assert kept >= 1


def test_prefilter_only_affects_excluded_lines(prefilter_env, job_info):
    """The extractor trigger with no exclude term present emits an event regardless of the flag"""
    non_excluded_line = f"[rank 0] {_EXTRACTOR_TRIGGER}"
    kept = _count_events_after_applog_ingest(prefilter_env, job_info, non_excluded_line, pre_filtered=True)
    also_kept = _count_events_after_applog_ingest(prefilter_env, job_info, non_excluded_line, pre_filtered=False)
    assert kept == also_kept
    assert kept >= 1


def test_legacy_prefiltered_sources_set_skips_prefilter(prefilter_env, job_info):
    """A silent observation (pre_filtered unset) defers to the deprecated pre_filtered_sources set,
    which still forces the skip so the excluded line survives."""
    line = f"[rank 0] {_EXCLUDE_SENTINEL} {_EXTRACTOR_TRIGGER}"
    kept = _count_events_after_applog_ingest(
        prefilter_env,
        job_info,
        line,
        pre_filtered=None,
        pre_filtered_sources={ObservationSource.APPLOG},
    )
    assert kept >= 1


def test_explicit_not_prefiltered_conflicts_with_legacy_set(prefilter_env, job_info):
    """Explicit pre_filtered=False for a source the set marks pre-filtered is a contradiction"""
    line = f"[rank 0] {_EXCLUDE_SENTINEL} {_EXTRACTOR_TRIGGER}"
    with pytest.raises(ValueError):
        _count_events_after_applog_ingest(
            prefilter_env,
            job_info,
            line,
            pre_filtered=False,
            pre_filtered_sources={ObservationSource.APPLOG},
        )
