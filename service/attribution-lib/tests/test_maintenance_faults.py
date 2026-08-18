# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Contract tests for maintenance-fault observations."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic_core import PydanticCustomError

from attribution_lib.job_info import JobInfoBuilder
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


def _job_info():
    """Create a job whose real node validation recognizes ``node-a``.

    Returns:
        A JobInfo instance for resource and node assertions.
    """
    return (
        JobInfoBuilder(
            id="job-1",
            node_to_ranks={"node-a": [0]},
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
