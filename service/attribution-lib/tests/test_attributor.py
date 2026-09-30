# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import attribution_lib
from attribution_lib.attributor import (
    Attributor,
    ConsumedObservation,
    IngestedObservation,
    _extract_observations,
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
from attribution_lib.prefilters import PreFilterSpec
from attribution_lib.resources.node import Node as NodeResource
from attribution_lib.resources.resource_set import AllJobResources, ResourceSetClass
from attribution_lib.rule_types.definitions import EventTriple, ExtractorRule, RuleLocation
from attribution_lib.rules_engine import RuleLoader
from test_support.helpers import make_node_to_ranks, make_syslog_entry

# APPLOG registers "error" as an include-keyword.
APPLOG_INCLUDE_TERM = "error"
DEFAULT_LINES = [APPLOG_INCLUDE_TERM, "failure", "crash"]
PYTHON_TRACEBACK_LINES = [
    "Traceback (most recent call last):",
    '  File "/workspace/train.py", line 42, in train',
    "RuntimeError: training failed",
]


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


def make_inbound_observation(
    source: ObservationSource, fmt: ObservationFormat, *, lines: list[str], pre_filtered: bool = False
):
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
    return InboundObservation(context, lines)


def make_observation(source: ObservationSource, fmt: ObservationFormat):
    return Observation.from_inbound_observation(make_inbound_observation(source, fmt, lines=DEFAULT_LINES))


@pytest.fixture
def sample_observation():
    return make_inbound_observation(
        ObservationSource.JOB_KEYWORDS,
        fmt=ObservationFormat.STRING_LIST,
        lines=DEFAULT_LINES,
    )


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
            ObservationFormat.ENTRY_TIMESERIES,
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
            ObservationFormat.ENTRY_TIMESERIES,
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
            ObservationFormat.ENTRY_TIMESERIES,
        ]
    )

    for fmt in bad_formats:
        observation = make_observation(ObservationSource.HEALTH_CHECK, fmt)
        with pytest.raises(MismatchedSourceAndFormatError):
            _ = ConsumedObservation.new(event_stream, job_info, observation)


def test_source_format_metrics_mismatch(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    bad_formats = frozenset(ObservationFormat) - frozenset(
        [ObservationFormat.UNDEFINED, ObservationFormat.PROMETHEUS_METRICS, ObservationFormat.ENTRY_TIMESERIES]
    )

    for fmt in bad_formats:
        observation = make_observation(ObservationSource.NODE_METRICS, fmt)
        with pytest.raises(MismatchedSourceAndFormatError):
            _ = ConsumedObservation.new(event_stream, job_info, observation)


def test_source_format_csp_event_mismatch(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    bad_formats = frozenset(ObservationFormat) - frozenset(
        [ObservationFormat.UNDEFINED, ObservationFormat.CSP_EVENT_LIST, ObservationFormat.ENTRY_TIMESERIES]
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
    obs = make_inbound_observation(ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES, lines=DEFAULT_LINES)
    attributor.ingest(obs)
    assert "applog" in attributor._source_fingerprints
    assert attributor._source_fingerprints["applog"].count == 3


def test_ingest_skips_fingerprint_for_keywords(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    obs = make_inbound_observation(ObservationSource.JOB_KEYWORDS, ObservationFormat.STRING_LIST, lines=DEFAULT_LINES)
    attributor.ingest(obs)
    assert attributor._source_fingerprints == {}


def test_attribute_preingested(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    attributor.ingest(
        make_inbound_observation(
            ObservationSource.JOB_KEYWORDS,
            ObservationFormat.STRING_LIST,
            lines=DEFAULT_LINES,
        )
    )

    backend = MemoryCheckpointBackend()
    result = attributor.attribute(backend)
    assert result is not None
    assert result.outcome_attr is not None


def test_attribute_applog_prefilter_keywords(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.RAW_LOG_LINES,
        lines=[
            "Traceback (most recent call last):",
            '  File "/workspace/train.py", line 42, in train',
            "RuntimeError: training failed",
        ],
    )

    attributor.ingest(observation)
    result = attributor.attribute()

    assert "Python Exception" in result.symptoms


def test_attribute_preingested_interleave(request, event_manager, job_info):
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    backend = MemoryCheckpointBackend()

    attributor.ingest(
        make_inbound_observation(
            ObservationSource.JOB_KEYWORDS,
            ObservationFormat.STRING_LIST,
            lines=DEFAULT_LINES,
        )
    )
    result1 = attributor.attribute(backend)

    attributor.ingest(
        make_inbound_observation(ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES, lines=DEFAULT_LINES)
    )
    result2 = attributor.attribute(backend)

    assert result1 is not None
    assert result2 is not None


def test_attribute_preingested_consistent(request, event_manager, job_info):
    """Calling attribute twice without new data produces the same outcome."""
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    backend = MemoryCheckpointBackend()

    attributor.ingest(
        make_inbound_observation(
            ObservationSource.JOB_KEYWORDS,
            ObservationFormat.STRING_LIST,
            lines=DEFAULT_LINES,
        )
    )

    result1 = attributor.attribute(backend)
    result2 = attributor.attribute(backend)

    assert result1.outcome_attr.value().outcome == result2.outcome_attr.value().outcome


def test_attribute_preingested_mutation_safe(request, event_manager, job_info):
    """Event count is preserved after attribute (mutation undone)."""
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    attributor.ingest(
        make_inbound_observation(
            ObservationSource.JOB_KEYWORDS,
            ObservationFormat.STRING_LIST,
            lines=DEFAULT_LINES,
        )
    )
    attributor.ingest(
        make_inbound_observation(ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES, lines=DEFAULT_LINES)
    )
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

    attributor.ingest(
        make_inbound_observation(
            ObservationSource.JOB_KEYWORDS,
            ObservationFormat.STRING_LIST,
            lines=DEFAULT_LINES,
        )
    )
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

    observations = [
        make_inbound_observation(
            ObservationSource.JOB_KEYWORDS,
            ObservationFormat.STRING_LIST,
            lines=DEFAULT_LINES,
        )
    ]
    attributor.ingest_batch(observations, backend)
    result = attributor.attribute(backend)
    assert result is not None


def test_attribute_batch_applog_prefilter_keywords(request, event_manager, job_info):
    """Promote raw keywords collected during a full batch extraction.

    Args:
        request: Pytest request used to name the event stream.
        event_manager: Event manager fixture.
        job_info: Job information fixture.
    """
    event_stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(event_stream, job_info)
    backend = MemoryCheckpointBackend()
    observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.RAW_LOG_LINES,
        lines=PYTHON_TRACEBACK_LINES,
    )

    attributor.ingest_batch([observation], backend)
    result = attributor.attribute(backend)

    assert "Python Exception" in result.symptoms


def test_attribute_batch_recomputes_incompatible_snapshot(request, event_manager, job_info, tmp_path, monkeypatch):
    """Recover a v1 snapshot with the same attribution as cold extraction.

    Args:
        request: Pytest request used to name the event stream.
        event_manager: Event manager fixture.
        job_info: Job information fixture.
        tmp_path: Temporary snapshot directory.
        monkeypatch: Replaces only the backend's snapshot storage read.
    """
    backend = MemoryCheckpointBackend()
    observation = make_inbound_observation(
        ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES, lines=PYTHON_TRACEBACK_LINES
    )
    cold = Attributor.new(event_manager.create_stream(request.node.name), job_info)
    cold.ingest_batch([observation], backend)
    cold_result = cold.attribute(backend)
    event_manager.export_snapshot(tmp_path)
    (tmp_path / EventManager._SCHEMA_VERSION_FILE).write_text("1")
    monkeypatch.setattr(
        backend,
        "load_event_manager",
        lambda key: EventManager.from_snapshot(tmp_path, attribution_lib.get_rules_engine()),
    )

    recovered = Attributor.new(event_manager.create_stream(f"{request.node.name}_recovered"), job_info)
    recovered.ingest_batch([observation], backend)
    result = recovered.attribute(backend)

    assert "Python Exception" in result.symptoms
    assert set(result.symptoms.symptoms()) == set(cold_result.symptoms.symptoms())
    assert result.outcome_attr.value() == cold_result.outcome_attr.value()


def test_attribute_batch_restores_prefilter_keywords_from_cache(request, event_manager, job_info):
    """Restore raw keywords when batch extraction is skipped on a cache hit.

    Args:
        request: Pytest request used to name the event stream.
        event_manager: Event manager fixture.
        job_info: Job information fixture.
    """
    backend = MemoryCheckpointBackend()
    observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.RAW_LOG_LINES,
        lines=PYTHON_TRACEBACK_LINES,
    )
    first_attributor = Attributor.new(event_manager.create_stream(request.node.name), job_info)
    first_attributor.ingest_batch([observation], backend)

    cached_attributor = Attributor.new(event_manager.create_stream(f"{request.node.name}_cached"), job_info)
    cached_attributor.ingest_batch([observation], backend)
    result = cached_attributor.attribute(backend)

    assert "Python Exception" in result.symptoms


def test_attribute_batch_reextracts_when_prefilter_directive_changes(request, event_manager, job_info):
    """Reject cached extraction performed under a different prefilter directive.

    Args:
        request: Pytest request used to name the event stream.
        event_manager: Event manager fixture.
        job_info: Job information fixture.
    """
    backend = MemoryCheckpointBackend()
    unfiltered_observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.RAW_LOG_LINES,
        lines=PYTHON_TRACEBACK_LINES,
        pre_filtered=False,
    )
    first_attributor = Attributor.new(event_manager.create_stream(request.node.name), job_info)
    first_attributor.ingest_batch([unfiltered_observation], backend)
    first_result = first_attributor.attribute(backend)

    prefiltered_observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.RAW_LOG_LINES,
        lines=PYTHON_TRACEBACK_LINES,
        pre_filtered=True,
    )
    second_attributor = Attributor.new(event_manager.create_stream(f"{request.node.name}_prefiltered"), job_info)
    second_attributor.ingest_batch([prefiltered_observation], backend)
    second_result = second_attributor.attribute(backend)

    assert "Python Exception" in first_result.symptoms
    assert "Python Exception" not in second_result.symptoms


def test_saved_fingerprint_includes_extraction_context(request, event_manager, job_info):
    """Change cache fingerprints when extraction-affecting context changes.

    Args:
        request: Pytest request used to name event streams.
        event_manager: Event manager fixture.
        job_info: Job information fixture.
    """
    observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.RAW_LOG_LINES,
        lines=PYTHON_TRACEBACK_LINES,
    )
    baseline_backend = MemoryCheckpointBackend()
    baseline_attributor = Attributor.new(event_manager.create_stream(request.node.name), job_info)
    baseline_attributor.ingest(observation)
    baseline_attributor.attribute(baseline_backend)
    cache_key = baseline_attributor.get_cache_key(job_info)
    baseline_metadata = baseline_backend.load_metadata(cache_key)
    assert baseline_metadata is not None
    baseline_fingerprint = baseline_metadata.data_fingerprints[ObservationSource.APPLOG.value]

    context = observation.context
    context_variants = [
        replace(context, pre_filtered=True),
        replace(context, timezone=ZoneInfo("UTC")),
        replace(
            context,
            time_interval=TimeInterval(
                start=context.time_interval.start + timedelta(days=1),
                end=context.time_interval.end + timedelta(days=1),
            ),
        ),
        replace(context, resources=ResourceSetClass([NodeResource(hostname="node1")])),
    ]
    for index, changed_context in enumerate(context_variants):
        backend = MemoryCheckpointBackend()
        attributor = Attributor.new(event_manager.create_stream(f"{request.node.name}_{index}"), job_info)
        attributor.ingest(InboundObservation(changed_context, PYTHON_TRACEBACK_LINES))
        attributor.attribute(backend)
        metadata = backend.load_metadata(cache_key)
        assert metadata is not None
        assert metadata.data_fingerprints[ObservationSource.APPLOG.value] != baseline_fingerprint


def test_batch_fingerprint_tracks_job_end_for_undated_timeseries_entries(request, event_manager, job_info):
    """Undated applog entries take their year from job end, so a later job end must not hit the cache."""
    observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.ENTRY_TIMESERIES,
        lines=[
            {"body": f"[E1231 23:57:09.800744155 ProcessGroupNCCL.cpp:1923] {line}"} for line in PYTHON_TRACEBACK_LINES
        ],
    )
    backend = MemoryCheckpointBackend()
    Attributor.new(event_manager.create_stream(request.node.name), job_info).ingest_batch([observation], backend)
    cache_key = Attributor.get_cache_key(job_info)
    baseline_metadata = backend.load_metadata(cache_key)
    assert baseline_metadata is not None
    baseline_fingerprint = baseline_metadata.data_fingerprints[ObservationSource.APPLOG.value]

    ended_a_year_later = job_info._replace(job_end_time=job_info.job_end_time + timedelta(days=366))
    Attributor.new(event_manager.create_stream(f"{request.node.name}_later"), ended_a_year_later).ingest_batch(
        [observation], backend
    )
    metadata = backend.load_metadata(cache_key)
    assert metadata is not None
    assert metadata.data_fingerprints[ObservationSource.APPLOG.value] != baseline_fingerprint


def test_attribute_batch_restores_prefilter_keywords_saved_by_ingest(request, event_manager, job_info):
    """Restore direct-ingestion raw keywords through the batch cache path.

    Args:
        request: Pytest request used to name the event stream.
        event_manager: Event manager fixture.
        job_info: Job information fixture.
    """
    backend = MemoryCheckpointBackend()
    observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.RAW_LOG_LINES,
        lines=PYTHON_TRACEBACK_LINES,
    )
    direct_attributor = Attributor.new(event_manager.create_stream(request.node.name), job_info)
    direct_attributor.ingest(observation)
    direct_attributor.attribute(backend)

    cached_attributor = Attributor.new(event_manager.create_stream(f"{request.node.name}_cached"), job_info)
    cached_attributor.ingest_batch([observation], backend)
    result = cached_attributor.attribute(backend)

    assert "Python Exception" in result.symptoms


def test_attribute_batch_retains_prefilter_keywords_during_incremental_extraction(request, event_manager, job_info):
    """Union cached raw keywords with an incrementally extracted suffix.

    Args:
        request: Pytest request used to name the event stream.
        event_manager: Event manager fixture.
        job_info: Job information fixture.
    """
    backend = MemoryCheckpointBackend()
    initial_observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.RAW_LOG_LINES,
        lines=PYTHON_TRACEBACK_LINES,
    )
    first_attributor = Attributor.new(event_manager.create_stream(request.node.name), job_info)
    first_attributor.ingest_batch([initial_observation], backend)

    appended_observation = make_inbound_observation(
        ObservationSource.APPLOG,
        ObservationFormat.RAW_LOG_LINES,
        lines=[*PYTHON_TRACEBACK_LINES, "training shutdown complete"],
    )
    incremental_attributor = Attributor.new(event_manager.create_stream(f"{request.node.name}_incremental"), job_info)
    incremental_attributor.ingest_batch([appended_observation], backend)
    result = incremental_attributor.attribute(backend)

    assert "Python Exception" in result.symptoms


def test_incremental_xid140_replaces_poisoned_gsp_and_retains_evidence(request, event_manager, job_info):
    """Replace a poisoned-GSP diagnosis when XID-140 is appended to cached syslog.

    Args:
        request: Pytest request used to name the event streams.
        event_manager: Event manager fixture.
        job_info: Job information fixture shared by both attribution passes.

    Returns:
        None.
    """
    lines = [
        "NVRM: kgspServiceFatalHwError_GB100: NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR FB Poison pending (mask: 0x1)",
        "NVRM: Xid (PCI:0000:06:00): 94, Contained: GSP. RST: Yes, D-RST: No",
        "NVRM: nvCheckOkFailedNoLog: Check failed: [NV_ERR_RESET_REQUIRED] (0x00000062) @ kernel_gsp.c:100",
        "NVRM: Xid (PCI:0000:06:00): 140, pid=1234, name=test, Poison error in GSP.",
    ]
    context = ObservationContext(
        time_interval=TimeInterval(job_info.job_start_time, job_info.job_end_time),
        resources=AllJobResources,
        source=ObservationSource.SYSLOG,
        format=ObservationFormat.RAW_LOKI_STREAMS,
        timezone=job_info.timezone,
    )
    base_ns = int((job_info.job_start_time + timedelta(minutes=1)).timestamp() * 1e9)
    entries = [
        make_syslog_entry(line, node="node1", timestamp_ns=base_ns + index * 1_000_000)
        for index, line in enumerate(lines)
    ]
    backend = MemoryCheckpointBackend()
    first_attributor = Attributor.new(event_manager.create_stream(request.node.name), job_info)
    first_attributor.ingest_batch([InboundObservation(context, entries[:3])], backend)
    first_result = first_attributor.attribute(backend)

    assert first_result.outcome_attr.value().outcome == "Fail InfraHW"
    assert first_result.outcome_attr.value().root_cause_symptom == "poisoned gsp"
    assert set(first_result.symptoms.symptoms()) == {"poisoned gsp"}
    assert {source.datum for source in first_result.symptoms["poisoned gsp"]} == set(lines[:3])

    incremental_attributor = Attributor.new(event_manager.create_stream(f"{request.node.name}_incremental"), job_info)
    with patch("attribution_lib.attributor._extract_observations", wraps=_extract_observations) as extract:
        incremental_attributor.ingest_batch([InboundObservation(context, entries)], backend)
    extract.assert_called_once()
    assert [observation.lines for observation in extract.call_args.args[1]] == [entries[3:]]
    result = incremental_attributor.attribute(backend)

    xid140 = "XID-140-UNRECOVERABLE_ECC_ERROR_ESCAPE"
    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert result.outcome_attr.value().root_cause_symptom == xid140
    assert set(result.bad_nodes_attr.my_bad_nodes) == {"node1"}
    node_symptoms = result.bad_nodes_attr.my_bad_nodes["node1"].attributions
    for symptoms in [result.symptoms, node_symptoms]:
        assert set(symptoms.symptoms()) == {xid140, "NV_ERR_RESET_REQUIRED kernel_gsp.c"}
    for symptom in ["NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR", "XID-94-CONTAINED_ERROR"]:
        assert node_symptoms[symptom].has_interesting()
    for symptom, line in zip(
        ["NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR", "XID-94-CONTAINED_ERROR", "NV_ERR_RESET_REQUIRED kernel_gsp.c", xid140],
        lines,
        strict=True,
    ):
        assert [source.datum for source in node_symptoms[symptom].all_sources()] == [line]


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

    Builds and installs an engine with one APPLOG extractor (fires on the trigger sentinel) and
    an exclude-only prefilter, then restores both globals on teardown so nothing leaks."""
    loader = RuleLoader()
    loader.register_rule(
        ExtractorRule(
            name=_TEST_RULE_NAME,
            author="test",
            location=RuleLocation(__file__, 0),
            match_re=_EXTRACTOR_TRIGGER,
            processor=_emit_one_event,
            source_filters=frozenset({ObservationSource.APPLOG}),
        )
    )
    loader.declared_event_types.add(_TEST_EVENT_TYPE)
    loader.register_pre_filter(
        ObservationSource.APPLOG,
        PreFilterSpec(exclude_terms=[_EXCLUDE_SENTINEL]),
    )

    built = loader.build()
    saved_engine = attribution_lib._rules_engine
    attribution_lib._rules_engine = built

    manager = EventManager(rules_engine=built)
    try:
        yield manager
    finally:
        manager.close()
        attribution_lib._rules_engine = saved_engine


def _applog_observation(line, *, pre_filtered):
    return make_inbound_observation(
        ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES, pre_filtered=pre_filtered, lines=[line]
    )


def _count_events_after_applog_ingest(event_manager, job_info, line, *, pre_filtered):
    stream = event_manager.create_stream(f"prefilter_{pre_filtered}")
    attributor = Attributor.new(stream, job_info)
    attributor.ingest(_applog_observation(line, pre_filtered=pre_filtered))
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


def _count_events_after_applog_batch(event_manager, job_info, line, *, pre_filtered):
    """Same as _count_events_after_applog_ingest, but through the batch extraction path."""
    stream = event_manager.create_stream(f"prefilter_batch_{pre_filtered}")
    attributor = Attributor.new(stream, job_info)
    attributor.ingest_batch([_applog_observation(line, pre_filtered=pre_filtered)], MemoryCheckpointBackend())
    batch_stream = attributor.event_stream
    batch_stream.flush()
    return batch_stream.mgr._conn.execute(
        "SELECT COUNT(*) FROM events WHERE context_id=?", [batch_stream.ctx_id]
    ).fetchone()[0]


def test_prefiltered_context_keeps_excluded_extractor_line_in_batch(prefilter_env, job_info):
    """The batch extraction path honors the same per-observation directive as ingest()"""
    line = f"[rank 0] {_EXCLUDE_SENTINEL} {_EXTRACTOR_TRIGGER}"
    kept = _count_events_after_applog_batch(prefilter_env, job_info, line, pre_filtered=True)
    dropped = _count_events_after_applog_batch(prefilter_env, job_info, line, pre_filtered=False)
    assert dropped == 0
    assert kept >= 1
