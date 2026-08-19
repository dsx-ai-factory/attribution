# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Contract tests for maintenance-fault observations."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic_core import PydanticCustomError

import attribution_lib
from attribution_lib.attributor import Attributor, ConsumedObservation
from attribution_lib.events import EventManager
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation import Observation as ProcessedObservation
from attribution_lib.observation.inbound import (
    Context,
    Format,
    Observation,
    Source,
    TimeInterval,
)
from attribution_lib.observation.processing import (
    EntrySourceBase,
    MaintenanceFaultEntryObj,
    MaintenanceFaultEntrySource,
)
from attribution_lib.resources import AllJobResources
from attribution_lib.resources.node import Node as NodeResource
from attribution_lib.resources.resource_set import ResourceSetClass
from attribution_lib.rules_engine import SOURCE_CONFIGS, EventTriple, ExtractorRule, RuleLoader, RuleLocation
from attribution_lib.symptoms.declarations import OutcomePriority, SymptomCategory
from attribution_lib.symptoms.registry_generation import OUTCOME_PRIORITIES

TIMEZONE = ZoneInfo("America/Los_Angeles")
JOB_START = datetime(2025, 1, 1, tzinfo=TIMEZONE)


def _observation(lines: list[object]) -> Observation:
    """Build a real inbound maintenance-fault observation.

    Args:
        lines: Maintenance-fault records to process.

    Returns:
        An inbound observation with the maintenance-fault contract.
    """
    return Observation(
        Context(
            TimeInterval(JOB_START, JOB_START + timedelta(hours=1)),
            AllJobResources,
            Source.MAINTENANCE_FAULT,
            Format.MAINTENANCE_FAULT_LIST,
            TIMEZONE,
        ),
        lines,
    )


def _job_info(*, node_to_ranks: dict[str, list[int]] | None = None):
    """Create a job whose real node validation recognizes ``node-a``.

    Returns:
        A JobInfo instance for resource and node assertions.
    """
    return (
        JobInfoBuilder(
            id="job-1",
            node_to_ranks={"node-a": [0]} if node_to_ranks is None else node_to_ranks,
            framework="test",
            type="test",
            state="COMPLETED",
            job_start_time=JOB_START,
            cluster="cluster-1",
            source_name="test",
            timezone=TIMEZONE,
        )
        .exit_code_signal(0)
        .job_end_time(JOB_START + timedelta(minutes=1))
        .time_limit(1)
        .build()
    )


def _valid_entry(**overrides: object) -> dict[str, object]:
    """Return a hand-authored valid maintenance-fault record.

    Args:
        **overrides: Values replacing the standard record fields.

    Returns:
        The maintenance-fault record used by a test.
    """
    entry: dict[str, object] = {
        "event_timestamp_s": 1_735_721_200.0,
        "hostname": "NODE-A",
        "event_id": "event-1",
        "fault_id": "fault-1",
        "fault_component": "GPU",
    }
    entry.update(overrides)
    return entry


def _entry(line: object):
    """Process one record through the public entry-source factory.

    Args:
        line: The input record to process.

    Returns:
        The processed maintenance-fault entry.
    """
    return next(EntrySourceBase.from_observation(_observation([line])))


def _fault_at(time: datetime, event_id: str, hostname: str = "node-a") -> dict[str, object]:
    """Build a valid fault record at a precise test time."""
    return _valid_entry(event_timestamp_s=time.timestamp(), event_id=event_id, hostname=hostname)


def _emit_maintenance_test_event(groups, hostname, source, context):
    """Produce one test-only event for each retained maintenance entry."""
    return [EventTriple(type="maintenance_test_event", payload={}, source=source)]


@pytest.fixture
def maintenance_test_engine():
    """Install a temporary extractor so source filtering has observable output."""
    loader = RuleLoader()
    loader.register_rule(
        ExtractorRule(
            name="maintenance_test_extractor",
            author="test",
            location=RuleLocation(__file__, 0),
            match_re="maintenance_fault",
            processor=_emit_maintenance_test_event,
            source_filters=frozenset({Source.MAINTENANCE_FAULT}),
        )
    )
    loader.declared_event_types.add("maintenance_test_event")
    engine = loader.build()
    saved_engine = attribution_lib._rules_engine
    attribution_lib._rules_engine = engine
    manager = EventManager(rules_engine=engine)
    try:
        yield manager
    finally:
        manager.close()
        attribution_lib._rules_engine = saved_engine


def test_maintenance_fault_enums_have_stable_values():
    """The source and format strings remain stable for public callers."""
    assert Source.MAINTENANCE_FAULT.value == "maintenance_fault"
    assert Format.MAINTENANCE_FAULT_LIST.value == "maintenance_fault_list"


def test_maintenance_fault_factory_emits_structured_entry_body():
    """A valid entry serializes its public body with compact sorted JSON."""
    source = EntrySourceBase.from_observation(_observation([_valid_entry()]))

    assert isinstance(source, MaintenanceFaultEntrySource)
    entry = next(source)
    assert isinstance(entry, MaintenanceFaultEntryObj)
    assert entry.body() == 'maintenance_fault:{"event_id":"event-1","fault_component":"GPU","fault_id":"fault-1"}'


def test_maintenance_fault_body_omits_missing_optional_component():
    """The body leaves out fault_component when no component was supplied."""
    entry_data = _valid_entry()
    del entry_data["fault_component"]

    assert _entry(entry_data).body() == 'maintenance_fault:{"event_id":"event-1","fault_id":"fault-1"}'


def test_maintenance_fault_time_node_and_resources_use_entry_contract():
    """Time respects the observation timezone and nodes use JobInfo validation."""
    entry = _entry(_valid_entry())
    job_info = _job_info()

    assert entry.time() == datetime.fromtimestamp(1_735_721_200.0, TIMEZONE)
    assert entry.node(job_info) == "node-a"
    assert entry.resources() == ResourceSetClass.ANY
    assert entry.resources(job_info) == ResourceSetClass([NodeResource(hostname="node-a")])


def test_maintenance_fault_source_configuration_filters_before_job_and_keeps_later_entries(maintenance_test_engine):
    """The registered source starts at the job and has no end-time cutoff."""
    job_info = _job_info()
    observation = _observation(
        [
            _fault_at(JOB_START - timedelta(seconds=1), "before-job"),
            _fault_at(JOB_START + timedelta(seconds=1), "during-job"),
            _fault_at(JOB_START + timedelta(hours=1), "after-job"),
        ]
    )
    stream = maintenance_test_engine.create_stream("maintenance_time_filter")
    processed_observation = ProcessedObservation.from_inbound_observation(observation)

    stream.register_observation_context(processed_observation.context)
    ConsumedObservation.new(stream, job_info, processed_observation)
    stream.flush()

    count = maintenance_test_engine._conn.execute(
        "SELECT COUNT(*) FROM events WHERE context_id=?", [stream.ctx_id]
    ).fetchone()[0]
    assert count == 2


def test_maintenance_fault_source_requires_known_job_nodes():
    """Maintenance faults are node-scoped and only begin at the job start."""
    config = SOURCE_CONFIGS[Source.MAINTENANCE_FAULT]

    assert config.time_start is not None
    assert config.time_start(_job_info()) == JOB_START
    assert config.time_end is None
    assert config.requires_nodes is True


def test_maintenance_hardware_fault_registry_metadata_is_public_contract():
    """The public symptom declaration preserves the internal rule's output contract."""
    registry = attribution_lib.get_symptom_registry()
    outcome = registry.outcome_for(["Maintenance Hardware Fault"])

    assert outcome is not None
    assert outcome.outcome == "Fail InfraHW"
    assert outcome.hw_cause == "Known HW Cause - Node"
    assert outcome.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.CSP)
    assert registry.symptom_docs["Maintenance Hardware Fault"] == "csp-event-hardware-fault"
    assert "Maintenance Hardware Fault" in registry.symptoms_in_category(SymptomCategory.CSP_EVENT)
    assert registry.metadata["Maintenance Hardware Fault"].cascade_discards == (
        "Node failure reported by slurm",
        "Not responding",
    )


@pytest.mark.parametrize(
    "observation",
    [
        _observation([_valid_entry(event_id="Maintenance Hardware Fault")]),
        Observation(
            Context(
                TimeInterval(JOB_START, JOB_START + timedelta(hours=1)),
                AllJobResources,
                Source.APPLOG,
                Format.RAW_LOG_LINES,
                TIMEZONE,
            ),
            ["[rank 0] Maintenance Hardware Fault"],
        ),
    ],
    ids=["maintenance-evidence", "non-maintenance-evidence"],
)
def test_public_extractors_do_not_emit_maintenance_hardware_fault_from_any_evidence(observation):
    """The public registry declares the symptom without shipping an extractor for it."""
    manager = EventManager(rules_engine=attribution_lib.get_rules_engine())
    try:
        attributor = Attributor.new(manager.create_stream("no_public_maintenance_extractor"), _job_info())
        attributor.ingest(observation)
        root_cause = attributor.attribute().outcome_attr.value().root_cause_symptom
    finally:
        manager.close()

    assert root_cause != "Maintenance Hardware Fault"


@pytest.mark.parametrize("line", [None, [], "not a record"])
def test_maintenance_fault_rejects_non_dictionary_entries(line: object):
    """Each fault record must be a dictionary.

    A missing dictionary check would permit malformed extractor output to reach
    attribution processing.
    """
    with pytest.raises(PydanticCustomError):
        _entry(line)


@pytest.mark.parametrize("field", ["event_timestamp_s", "hostname", "event_id", "fault_id"])
def test_maintenance_fault_rejects_each_missing_required_field(field: str):
    """Every required field is necessary to construct an attributable fault."""
    line = _valid_entry()
    del line[field]

    with pytest.raises(PydanticCustomError):
        _entry(line)


@pytest.mark.parametrize("field", ["hostname", "event_id", "fault_id", "fault_component"])
@pytest.mark.parametrize("value", ["", "   ", 1, None])
def test_maintenance_fault_rejects_blank_or_non_string_text_fields(field: str, value: object):
    """Supplied fault text must be meaningful strings, not empty or coerced values."""
    with pytest.raises(PydanticCustomError):
        _entry(_valid_entry(**{field: value}))


@pytest.mark.parametrize(
    "event_timestamp_s",
    [
        True,
        "1735721200.0",
        float("nan"),
        float("inf"),
        float("-inf"),
        10**1000,
        1e100,
        -1e100,
    ],
)
def test_maintenance_fault_rejects_invalid_timestamps(event_timestamp_s: object):
    """Timestamps must be finite numeric values representable as datetimes."""
    with pytest.raises(PydanticCustomError):
        _entry(_valid_entry(event_timestamp_s=event_timestamp_s))
