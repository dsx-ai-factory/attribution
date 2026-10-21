# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import attribution_lib
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.core import get_bundled_rules_path
from attribution_lib.events import DemotionTypes, EventManager
from attribution_lib.exceptions import UndeclaredRuleError
from attribution_lib.bundled_rules.common.megatron import get_pytorch_timeout_minutes
from attribution_lib.filter import filter_events
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import (
    Format as ObsFormat,
    Id,
    Source as ObsSource,
    TimeInterval,
)
from attribution_lib.resources import AllJobResources, Node, ResourceSet
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.sources import Source


@pytest.fixture(autouse=True)
def rules_engine():
    original_engine = attribution_lib._rules_engine
    loader = RuleLoader()

    # Load filter rules from the bundled rules directory
    rules_dir = get_bundled_rules_path()
    loader.load(str(rules_dir / "filter_rules.py"))
    attribution_lib._rules_engine = loader.build()

    yield attribution_lib.get_rules_engine()

    attribution_lib._rules_engine = original_engine


@pytest.fixture
def mgr(rules_engine):
    m = EventManager(rules_engine=rules_engine)
    m.preload_rule_names(
        {"some_rule", "fm-lost-connection-to-gpu", "NcclWatchdog", "test_demote", "discard", "no-symptom-demotion"}
    )
    m.preload_event_types(
        {
            "Attribution.node.bad",
            "Attribution.node.interesting",
            "Application.mfu",
            "Application.pytorch.nccl_watchdog",
            "Application.dataloader.exception",
            "Application.megatron-lm.runtime_args",
            "FM_parser.lost_gpu_connection",
            "Application.NCCLWatchdog.report",
        }
    )
    yield m
    m.close()


@pytest.fixture
def stream(mgr):
    return mgr.create_stream("test_job")


@pytest.fixture
def obs_ctx():
    return ObservationContext(
        time_interval=TimeInterval(
            datetime(2024, 1, 1, 9, 0, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
        ),
        source=ObsSource.APPLOG,
        format=ObsFormat.ENTRY_TIMESERIES,
        timezone=ZoneInfo("UTC"),
        id=Id(999),
    )


@pytest.fixture
def job_info(obs_ctx):
    start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    return default_job_info(job_end_time=start + timedelta(hours=1))


@pytest.fixture(autouse=True)
def _register_ctx(stream, obs_ctx):
    stream.register_observation_context(obs_ctx)


def create_event(stream, obs_ctx, event_type, payload, time, line_num, rule="some_rule", datum="dummy line"):
    source = Source(
        source_context=obs_ctx, line_num=line_num, datum=datum, time=time, resources=AllJobResources, rule=rule
    )
    stream.add_event(event_type, payload, source)
    stream.flush()


def test_demote_tflops_symptoms(stream, obs_ctx, job_info):
    # 1. Add bad event with AFTER_MFU_DEMOTION at 10:05
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    payload = {"node": "node01", "symptom": "some error", DemotionTypes.AFTER_MFU_DEMOTION: "test_demote"}
    create_event(stream, obs_ctx, "Attribution.node.bad", payload, t1, 100)

    # 2. Add MFU event at 10:10
    t_mfu = datetime(2024, 1, 1, 10, 10, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Application.mfu", {"tflops": 100}, t_mfu, 150)

    # 3. Add bad event after MFU at 10:15
    t2 = datetime(2024, 1, 1, 10, 15, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Attribution.node.bad", payload, t2, 200)

    filter_events(stream, job_info)
    stream.flush()

    # Event at 10:05 is before MFU. Should be demoted.
    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    int_id = stream.mgr.types.get_id("Attribution.node.interesting")

    # Original bad event at 100 should be deactivated
    rows_100 = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert rows_100[0][0] == 0

    # New interesting event should preserve source data and use the demotion
    # rule as provenance, while removing the marker from its payload.
    rows_int = stream.query_raw(
        """
        SELECT
            e.status,
            e.timestamp,
            e.line_num,
            e.source_datum,
            e.resources,
            e.obs_context_id,
            json_extract_string(e.payload, '$.node'),
            json_extract_string(e.payload, '$.symptom'),
            json_exists(e.payload, '$.AFTER_MFU_DEMOTION'),
            r.name
        FROM events e
        LEFT JOIN rules r ON r.id=e.source_rule
        WHERE e.event_type=? AND e.line_num=100
        """,
        [int_id],
    )
    assert rows_int == [
        (
            1,
            t1,
            100,
            "dummy line",
            '"AllJobResources"',
            obs_ctx.id,
            "node01",
            "(non-fatal) some error",
            False,
            "test_demote",
        )
    ]

    # Event at 200 is after MFU. Should remain bad.
    rows_200 = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=200", [bad_id])
    assert rows_200[0][0] == 1


def test_demote_with_undeclared_rule_name_raises(stream, obs_ctx, job_info):
    """Demotion rule names embedded in event payloads must already be known
    (declared up front via `declare_rule_names()`). Discovering a new one
    from live event data at demotion time is a bug in whichever rule stamped
    it, not something to silently register.
    """
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    payload = {"node": "node01", "symptom": "some error", DemotionTypes.AFTER_MFU_DEMOTION: "never_declared_rule"}
    create_event(stream, obs_ctx, "Attribution.node.bad", payload, t1, 100)

    t_mfu = datetime(2024, 1, 1, 10, 10, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Application.mfu", {"tflops": 100}, t_mfu, 150)

    with pytest.raises(UndeclaredRuleError, match="never_declared_rule"):
        filter_events(stream, job_info)


def test_demote_with_multiple_undeclared_rule_names_reports_stable_name(stream, obs_ctx, job_info):
    """When more than one undeclared demotion rule name surfaces in the same
    batch, the reported name must be deterministic across runs, not whatever
    order Python's (hash-seed randomized) set iteration happens to produce.
    """
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": "err1", DemotionTypes.AFTER_MFU_DEMOTION: "zzz_never_declared"},
        t1,
        100,
    )
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": "err2", DemotionTypes.AFTER_MFU_DEMOTION: "aaa_never_declared"},
        t1,
        101,
    )

    t_mfu = datetime(2024, 1, 1, 10, 10, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Application.mfu", {"tflops": 100}, t_mfu, 150)

    with pytest.raises(UndeclaredRuleError, match="aaa_never_declared"):
        filter_events(stream, job_info)


def test_demote_tflops_symptoms_edge_cases(stream, obs_ctx, job_info):
    before_mfu = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", DemotionTypes.AFTER_MFU_DEMOTION: "no-symptom-demotion"},
        before_mfu,
        80,
    )
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": "keep active", DemotionTypes.AFTER_MFU_DEMOTION: ""},
        before_mfu,
        90,
    )
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.interesting",
        {"node": "node01", "symptom": "already interesting", DemotionTypes.AFTER_MFU_DEMOTION: "discard"},
        before_mfu,
        100,
    )
    create_event(
        stream,
        obs_ctx,
        "Application.mfu",
        {"tflops": 100},
        datetime(2024, 1, 1, 10, 10, 0, tzinfo=ZoneInfo("UTC")),
        150,
    )

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    interesting_id = stream.mgr.types.get_id("Attribution.node.interesting")

    # A replacement without a symptom still drops the demotion marker.
    replacement = stream.query_raw(
        """
        SELECT e.status,
               json_exists(e.payload, '$.symptom'),
               json_exists(e.payload, '$.AFTER_MFU_DEMOTION'),
               r.name
        FROM events e
        LEFT JOIN rules r ON r.id=e.source_rule
        WHERE e.event_type=? AND e.line_num=80
        """,
        [interesting_id],
    )
    assert replacement == [(1, False, False, "no-symptom-demotion")]
    assert stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=80", [bad_id]) == [(0,)]

    # An empty demotion value is not actionable and leaves the event untouched.
    assert stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=90", [bad_id]) == [(1,)]
    assert stream.query_raw("SELECT count(*) FROM events WHERE event_type=? AND line_num=90", [interesting_id]) == [
        (0,)
    ]

    # Preserve the established behavior for selected types that have no
    # bad-to-interesting mapping: deactivate them without making a replacement.
    assert stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [interesting_id]) == [(0,)]


def test_demote_watchdog_errors(stream, obs_ctx, job_info):
    # 1. Add bad event with NOT_NEAR_WATCHDOG_EVENT_DEMOTION
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    payload = {"node": "node01", "symptom": "some error", DemotionTypes.NOT_NEAR_WATCHDOG_EVENT_DEMOTION: "test_demote"}
    create_event(stream, obs_ctx, "Attribution.node.bad", payload, t1, 100)

    # 2. Add MFU event (required by _demote_watchdog_errors to construct error_time)
    t_mfu = datetime(2024, 1, 1, 10, 10, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Application.mfu", {"tflops": 100}, t_mfu, 150)

    # 3. Add watchdog event at 10:15
    # The error_time will be constructed from MFU's day + watchdog's raw_error_time
    t_wd = datetime(2024, 1, 1, 10, 15, 0, tzinfo=ZoneInfo("UTC"))
    wd_payload = {"raw_error_time": "10:15:00", "timeout_seconds": 600}
    create_event(stream, obs_ctx, "Application.pytorch.nccl_watchdog", wd_payload, t_wd, 200)

    # Run filter
    filter_events(stream, job_info)
    stream.flush()

    # Event at 10:05 is 10 mins before 10:15. timeout_seconds=600 (10 mins).
    # time_diff = (10:15:00 - 10:05:00) = 600 seconds.
    # condition: 0 < time_diff < timeout_seconds + 30 (630).
    # 600 is in (0, 630), so it should NOT be demoted.

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert rows[0][0] == 1  # Still active

    # 4. Add another bad event at 10:00 (too far from 10:15)
    t2 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Attribution.node.bad", payload, t2, 50)

    filter_events(stream, job_info)
    stream.flush()

    # Event at 10:00 is 15 mins before 10:15 (900 seconds). 900 > 630. Should be demoted.
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=50", [bad_id])
    assert rows[0][0] == 0  # Deactivated


def test_demote_watchdog_errors_no_mfu(stream, obs_ctx, job_info):
    """Demote all eligible events when no MFU event exists."""
    payload = {
        "node": "node01",
        "symptom": "LustreError",
        DemotionTypes.NOT_NEAR_WATCHDOG_EVENT_DEMOTION: "test_demote",
    }
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        payload,
        datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC")),
        50,
    )
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        payload,
        datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC")),
        100,
    )

    wd_payload = {"raw_error_time": "10:15:00", "timeout_seconds": 600}
    create_event(
        stream,
        obs_ctx,
        "Application.pytorch.nccl_watchdog",
        wd_payload,
        datetime(2024, 1, 1, 10, 15, 0, tzinfo=ZoneInfo("UTC")),
        200,
    )

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    distant_rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=50", [bad_id])
    assert distant_rows[0][0] == 0

    near_rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert near_rows[0][0] == 0


def test_demote_watchdog_errors_mfu_only_after_watchdog(stream, obs_ctx, job_info):
    """Retain eligible events when MFU exists but cannot anchor watchdog timing."""
    payload = {
        "node": "node01",
        "symptom": "LustreError",
        DemotionTypes.NOT_NEAR_WATCHDOG_EVENT_DEMOTION: "test_demote",
    }
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        payload,
        datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC")),
        100,
    )

    wd_payload = {"raw_error_time": "10:15:00", "timeout_seconds": 600}
    create_event(
        stream,
        obs_ctx,
        "Application.pytorch.nccl_watchdog",
        wd_payload,
        datetime(2024, 1, 1, 10, 15, 0, tzinfo=ZoneInfo("UTC")),
        200,
    )
    create_event(
        stream,
        obs_ctx,
        "Application.mfu",
        {"tflops": 100},
        datetime(2024, 1, 1, 10, 20, 0, tzinfo=ZoneInfo("UTC")),
        250,
    )

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert rows[0][0] == 1


def test_demote_watchdog_errors_no_watchdog(stream, obs_ctx, job_info):
    """Test that events with NOT_NEAR_WATCHDOG_EVENT_DEMOTION are demoted when NO watchdog exists.

    If there's no watchdog, events are "not near watchdog" by definition, so they should be demoted.
    This works regardless of whether MFU exists (MFU is only needed when there IS a watchdog).
    """
    # Add bad events with NOT_NEAR_WATCHDOG_EVENT_DEMOTION
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    payload = {
        "node": "node01",
        "symptom": "LustreError",
        DemotionTypes.NOT_NEAR_WATCHDOG_EVENT_DEMOTION: "test_demote",
    }
    create_event(stream, obs_ctx, "Attribution.node.bad", payload, t1, 100)

    t2 = datetime(2024, 1, 1, 10, 10, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Attribution.node.bad", payload, t2, 150)

    # Notably: NO watchdog event and NO MFU event are added

    # Run filter
    filter_events(stream, job_info)
    stream.flush()

    # Both events should be demoted since there's no watchdog
    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert rows[0][0] == 0  # Deactivated

    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=150", [bad_id])
    assert rows[0][0] == 0  # Deactivated


def test_demote_dataloader_errors(stream, obs_ctx, job_info):
    # 1. Add bad event with BEFORE_DATALOADER_DEMOTION at 10:05
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    payload = {"node": "node01", "symptom": "some error", DemotionTypes.BEFORE_DATALOADER_DEMOTION: "test_demote"}
    create_event(stream, obs_ctx, "Attribution.node.bad", payload, t1, 100)

    # 2. Add MFU event at 10:10 (This will become 'last_time')
    t_mfu = datetime(2024, 1, 1, 10, 10, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Application.mfu", {"tflops": 100}, t_mfu, 150)

    # 3. Add dataloader event at 10:15
    t_dl = datetime(2024, 1, 1, 10, 15, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Application.dataloader.exception", {}, t_dl, 200)

    filter_events(stream, job_info)
    stream.flush()

    # Event at 10:05 is before MFU at 10:10. Should be demoted.
    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert rows[0][0] == 0


def test_demote_process_completes_cascade(stream, obs_ctx, job_info):
    # 1. Add "process completes" event
    t1 = datetime(2024, 1, 1, 10, 10, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream, obs_ctx, "Attribution.node.bad", {"node": "node01", "symptom": "the process completes"}, t1, 100
    )

    # 2. Add noise event after completion
    t2 = datetime(2024, 1, 1, 10, 15, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream, obs_ctx, "Attribution.node.bad", {"node": "node01", "symptom": "No such file or directory"}, t2, 200
    )

    filter_events(stream, job_info)
    stream.flush()

    # Noise event at 200 should be deactivated
    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=200", [bad_id])
    assert rows[0][0] == 0


def test_clean_informational_xids_demoted(stream, obs_ctx, job_info):
    """Test that informational XIDs are demoted from bad to interesting (non-fatal)."""
    informational_xids = [
        "XID-63-INFOROM_DRAM_RETIREMENT_EVENT",
        "XID-108-NVLINK_REMOTE_TRANSLATION_ERROR",
        "XID-162-PSHC_REENGAGED",
        "XID-163-PSHC_DISENGAGED",
    ]

    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    for i, xid in enumerate(informational_xids):
        payload = {"node": "node01", "symptom": xid}
        create_event(stream, obs_ctx, "Attribution.node.bad", payload, t1, 100 + i)

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    int_id = stream.mgr.types.get_id("Attribution.node.interesting")

    for i, xid in enumerate(informational_xids):
        # Original bad event should be deactivated
        rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=?", [bad_id, 100 + i])
        assert rows[0][0] == 0, f"{xid} bad event should be deactivated"
        # discard(purge=False) moves bad → interesting, so we expect an interesting replacement
        rows = stream.query_raw(
            "SELECT status, payload FROM events WHERE event_type=? AND line_num=?", [int_id, 100 + i]
        )
        assert rows[0][0] == 1, f"{xid} should have an interesting replacement"
        assert xid in rows[0][1], f"{xid} should appear in interesting payload"


def test_row_remap_demoted_without_affecting_other_symptoms(stream, obs_ctx, job_info):
    """Keep row-remap telemetry as context and preserve other failures.

    Args:
        stream: Event stream fixture.
        obs_ctx: Observation context fixture.
        job_info: Job information fixture.
    """
    symptom = "Row Remap"
    other_symptom = "XID-94 (Containment error)"
    timestamp = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": symptom},
        timestamp,
        100,
    )
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": other_symptom},
        timestamp,
        200,
    )

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    interesting_id = stream.mgr.types.get_id("Attribution.node.interesting")

    bad_rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert bad_rows[0][0] == 0

    interesting_rows = stream.query_raw(
        "SELECT status, payload FROM events WHERE event_type=? AND line_num=100",
        [interesting_id],
    )
    assert interesting_rows[0][0] == 1
    assert symptom in interesting_rows[0][1]

    other_bad_rows = stream.query_raw(
        "SELECT status, payload FROM events WHERE event_type=? AND line_num=200",
        [bad_id],
    )
    assert other_bad_rows[0][0] == 1
    assert other_symptom in other_bad_rows[0][1]


@pytest.mark.parametrize(
    "symptom",
    [
        "NV_ERR_INSUFFICIENT_PERMISSIONS",
        "NV_ERR_INSUFFICIENT_PERMISSIONS kern_profiler_v2_ctrl.c",
    ],
)
def test_nv_err_insufficient_permissions_demoted_to_interesting(stream, obs_ctx, job_info, symptom):
    """Keep insufficient-permissions telemetry without treating it as a failure.

    Args:
        stream: Event stream fixture.
        obs_ctx: Observation context fixture.
        job_info: Job information fixture.
        symptom: Insufficient-permissions symptom to demote.
    """
    timestamp = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": symptom},
        timestamp,
        100,
    )

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    interesting_id = stream.mgr.types.get_id("Attribution.node.interesting")

    bad_rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert bad_rows[0][0] == 0

    interesting_rows = stream.query_raw(
        "SELECT status, payload FROM events WHERE event_type=? AND line_num=100",
        [interesting_id],
    )
    assert interesting_rows[0][0] == 1
    assert symptom in interesting_rows[0][1]


@pytest.mark.parametrize(
    "symptom",
    [
        "NV_ERR_BUFFER_TOO_SMALL journal.c",
        "NV_ERR_BUFFER_TOO_SMALL journal.c:1184",
    ],
)
def test_nv_err_buffer_too_small_journal_demoted_to_interesting(stream, obs_ctx, job_info, symptom):
    """Keep journal.c BUFFER_TOO_SMALL as context without treating it as a failure.

    Args:
        stream: Event stream fixture.
        obs_ctx: Observation context fixture.
        job_info: Job information fixture.
        symptom: journal.c BUFFER_TOO_SMALL symptom to demote.
    """
    timestamp = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": symptom},
        timestamp,
        100,
    )

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    interesting_id = stream.mgr.types.get_id("Attribution.node.interesting")

    bad_rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert bad_rows[0][0] == 0

    interesting_rows = stream.query_raw(
        "SELECT status, payload FROM events WHERE event_type=? AND line_num=100",
        [interesting_id],
    )
    assert interesting_rows[0][0] == 1
    assert symptom in interesting_rows[0][1]


@pytest.mark.parametrize(
    "symptom",
    [
        "NV_ERR_INSUFFICIENT_PERMISSIONS_FATAL kern_profiler_v2_ctrl.c",
        "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c",
        "NV_ERR_BUFFER_TOO_SMALL mem_multicast_fabric.c",
    ],
)
def test_nv_err_insufficient_permissions_demotion_preserves_other_nv_errs(stream, obs_ctx, job_info, symptom):
    """Keep unrelated NV_ERR symptoms classified as bad.

    Args:
        stream: Event stream fixture.
        obs_ctx: Observation context fixture.
        job_info: Job information fixture.
        symptom: NV_ERR symptom that must remain bad.
    """
    timestamp = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": symptom},
        timestamp,
        100,
    )

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    bad_rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert bad_rows[0][0] == 1


def test_nv_err_invalid_state_osapi_demoted_to_interesting(stream, obs_ctx, job_info):
    """Keep non-fatal osapi telemetry without treating it as a failure.

    Args:
        stream: Event stream fixture.
        obs_ctx: Observation context fixture.
        job_info: Job information fixture.
    """
    symptom = "NV_ERR_INVALID_STATE osapi.c"
    timestamp = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": symptom},
        timestamp,
        100,
    )

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    interesting_id = stream.mgr.types.get_id("Attribution.node.interesting")

    bad_rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert bad_rows[0][0] == 0

    interesting_rows = stream.query_raw(
        "SELECT status, payload FROM events WHERE event_type=? AND line_num=100",
        [interesting_id],
    )
    assert interesting_rows[0][0] == 1
    assert symptom in interesting_rows[0][1]


@pytest.mark.parametrize(
    "symptom",
    [
        "NV_ERR_INVALID_STATE nv_gpu_ops.c",
        "NV_ERR_INVALID_STATE kern_bus_ctrl.c",
        "NV_ERR_INVALID_STATE gpu_fabric_probe.c",
    ],
)
def test_nv_err_invalid_state_osapi_demotion_preserves_other_sources(stream, obs_ctx, job_info, symptom):
    """Keep other invalid-state source files classified as bad.

    Args:
        stream: Event stream fixture.
        obs_ctx: Observation context fixture.
        job_info: Job information fixture.
        symptom: Invalid-state symptom that must remain bad.
    """
    timestamp = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    create_event(
        stream,
        obs_ctx,
        "Attribution.node.bad",
        {"node": "node01", "symptom": symptom},
        timestamp,
        100,
    )

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    bad_rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert bad_rows[0][0] == 1


@pytest.mark.parametrize(
    "real_xid",
    [
        "XID-94 (Containment error)",
        "XID-108-NVLINK_REMOTE_TRANSLATION_ERROR-DETAIL",
        "XID-109-CTXSW_TIMEOUT_ERROR",
    ],
)
def test_clean_informational_xids_preserves_real_xids(stream, obs_ctx, job_info, real_xid):
    """Test that non-informational XIDs remain as bad events."""
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    payload = {"node": "node01", "symptom": real_xid}
    create_event(stream, obs_ctx, "Attribution.node.bad", payload, t1, 100)

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert rows[0][0] == 1, f"{real_xid} should remain active as bad"


def test_clean_informational_xids_ignores_interesting(stream, obs_ctx, job_info):
    """Test that already-interesting informational XIDs are left untouched."""
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    xid = "XID-63-INFOROM_DRAM_RETIREMENT_EVENT"
    payload = {"node": "node01", "symptom": f"(non-fatal) {xid}"}
    create_event(stream, obs_ctx, "Attribution.node.interesting", payload, t1, 100)

    filter_events(stream, job_info)
    stream.flush()

    int_id = stream.mgr.types.get_id("Attribution.node.interesting")
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [int_id])
    assert rows[0][0] == 1, "Already-interesting informational XID should remain untouched"


def test_clean_informational_xids_mixed(stream, obs_ctx, job_info):
    """Test that only informational XIDs are demoted when mixed with real XIDs."""
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))

    info_xid = "XID-63-INFOROM_DRAM_RETIREMENT_EVENT"
    real_xid = "XID-94 (Containment error)"

    create_event(stream, obs_ctx, "Attribution.node.bad", {"node": "node01", "symptom": info_xid}, t1, 100)
    create_event(stream, obs_ctx, "Attribution.node.bad", {"node": "node01", "symptom": real_xid}, t1, 200)

    filter_events(stream, job_info)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")
    int_id = stream.mgr.types.get_id("Attribution.node.interesting")

    # Informational XID should be demoted to interesting
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [bad_id])
    assert rows[0][0] == 0, "Informational XID bad event should be deactivated"
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=100", [int_id])
    assert rows[0][0] == 1, "Informational XID should have interesting replacement"

    # Real XID should remain bad
    rows = stream.query_raw("SELECT status FROM events WHERE event_type=? AND line_num=200", [bad_id])
    assert rows[0][0] == 1, "Real XID should remain active as bad"


def _make_job_info(ranks_per_node, nodes=None):
    """JobInfo with specific topology; other fields match the canonical default."""
    start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    return default_job_info(
        nodes=nodes,
        ranks_per_node=ranks_per_node,
        job_end_time=start + timedelta(hours=1),
    )


def test_fm_node_failure_all_gpus(stream, obs_ctx):
    """When distinct GPU count == ranks_per_node, individual symptoms are replaced by aggregated one."""
    ji = _make_job_info(ranks_per_node=4)
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))

    for i in range(4):
        # Tracking event emitted by the extractor
        create_event(
            stream,
            obs_ctx,
            "FM_parser.lost_gpu_connection",
            {"node": "node01", "gpu_guid": f"0x{i:016x}"},
            t1 + timedelta(seconds=i),
            100 + i,
        )
        # Individual bad event emitted by the extractor
        create_event(
            stream,
            obs_ctx,
            "Attribution.node.bad",
            {"node": "node01", "symptom": "Fabric manager lost connection to GPU"},
            t1 + timedelta(seconds=i),
            100 + i,
            rule="fm-lost-connection-to-gpu",
        )

    filter_events(stream, ji)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")

    # Individual events should be deactivated
    for i in range(4):
        rows = stream.query_raw(
            "SELECT status, payload FROM events WHERE event_type=? AND line_num=? AND payload LIKE '%lost connection to GPU%'",
            [bad_id, 100 + i],
        )
        assert rows[0][0] == 0, f"Individual FM event at line {100 + i} should be deactivated"

    # Aggregated event should exist
    rows = stream.query_raw(
        "SELECT status, payload FROM events WHERE event_type=? AND payload LIKE '%all GPUs%'",
        [bad_id],
    )
    assert len(rows) == 4, "one aggregated event per distinct GPU so every per-GPU line is retained"
    assert all(r[0] == 1 for r in rows), "Aggregated FM events should be active"


def test_fm_node_failure_retains_per_gpu_evidence(stream, obs_ctx):
    """All per-GPU 'Lost connection to GPU' lines survive as evidence under the aggregated symptom."""
    ji = _make_job_info(ranks_per_node=4)
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))

    # Real FM logs are syslog-sourced; evidence extraction keeps only SYSLOG attributions.
    syslog_ctx = ObservationContext(
        time_interval=TimeInterval(
            datetime(2024, 1, 1, 9, 0, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
        ),
        source=ObsSource.SYSLOG,
        format=ObsFormat.ENTRY_TIMESERIES,
        timezone=ZoneInfo("UTC"),
        id=Id(777),
    )
    stream.register_observation_context(syslog_ctx)

    gpu_lines = {}
    for i in range(4):
        guid = f"0x{i:016x}"
        line = f"nmxc-fm: Lost connection to GPU on nodeId 5, gpuGUID {guid}, GPU state Error."
        gpu_lines[guid] = line
        create_event(
            stream,
            syslog_ctx,
            "FM_parser.lost_gpu_connection",
            {"node": "node01", "gpu_guid": guid},
            t1 + timedelta(seconds=i),
            100 + i,
            datum=line,
        )
        create_event(
            stream,
            syslog_ctx,
            "Attribution.node.bad",
            {"node": "node01", "symptom": "Fabric manager lost connection to GPU"},
            t1 + timedelta(seconds=i),
            100 + i,
            rule="fm-lost-connection-to-gpu",
            datum=line,
        )

    attrs = JobNodeAttributions.from_event_stream(stream, ji)
    node_to_evidence: defaultdict[str, set[str]] = defaultdict(set)
    attrs.extract_evidence_and_xid(node_to_evidence, defaultdict(set))

    assert node_to_evidence["node01"] == set(gpu_lines.values()), (
        f"expected all 4 per-GPU lines retained, got {sorted(node_to_evidence['node01'])}"
    )

    # The evidence must come from the aggregation firing (one event per GPU), not from the
    # individual per-GPU events surviving - they carry the same datums, so without this the
    # test would pass even if FmNodeFailure never ran.
    # Prove the rule actually fired: the node's symptom must be the aggregate, not the
    # per-GPU one - otherwise the surviving individual events (same datums) would satisfy
    # the evidence check above even if FmNodeFailure never ran.
    symptoms = set(attrs["node01"].symptoms())
    assert "Fabric manager lost connection to all GPUs" in symptoms, "aggregation rule must have fired"
    assert "Fabric manager lost connection to GPU" not in symptoms, "per-GPU symptom must be replaced"


def test_fm_node_failure_below_threshold(stream, obs_ctx):
    """When distinct GPU count < ranks_per_node, individual symptoms are kept."""
    ji = _make_job_info(ranks_per_node=4)
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))

    for i in range(3):
        create_event(
            stream,
            obs_ctx,
            "FM_parser.lost_gpu_connection",
            {"node": "node01", "gpu_guid": f"0x{i:016x}"},
            t1 + timedelta(seconds=i),
            200 + i,
        )
        create_event(
            stream,
            obs_ctx,
            "Attribution.node.bad",
            {"node": "node01", "symptom": "Fabric manager lost connection to GPU"},
            t1 + timedelta(seconds=i),
            200 + i,
            rule="fm-lost-connection-to-gpu",
        )

    filter_events(stream, ji)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")

    # Individual events should remain active
    for i in range(3):
        rows = stream.query_raw(
            "SELECT status FROM events WHERE event_type=? AND line_num=?",
            [bad_id, 200 + i],
        )
        assert rows[0][0] == 1, f"Individual FM event at line {200 + i} should remain active"

    # No aggregated event should exist
    rows = stream.query_raw(
        "SELECT status FROM events WHERE event_type=? AND payload LIKE '%all GPUs%'",
        [bad_id],
    )
    assert len(rows) == 0


def test_fm_node_failure_duplicate_guids(stream, obs_ctx):
    """Duplicate GUIDs should not inflate count - only distinct GUIDs matter."""
    ji = _make_job_info(ranks_per_node=4)
    t1 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))

    # 8 log lines but only 2 distinct GUIDs
    for i in range(8):
        create_event(
            stream,
            obs_ctx,
            "FM_parser.lost_gpu_connection",
            {"node": "node01", "gpu_guid": f"0x{(i % 2):016x}"},
            t1 + timedelta(seconds=i),
            300 + i,
        )
        create_event(
            stream,
            obs_ctx,
            "Attribution.node.bad",
            {"node": "node01", "symptom": "Fabric manager lost connection to GPU"},
            t1 + timedelta(seconds=i),
            300 + i,
            rule="fm-lost-connection-to-gpu",
        )

    filter_events(stream, ji)
    stream.flush()

    bad_id = stream.mgr.types.get_id("Attribution.node.bad")

    # Individual events should remain active
    for i in range(8):
        rows = stream.query_raw(
            "SELECT status FROM events WHERE event_type=? AND line_num=?",
            [bad_id, 300 + i],
        )
        assert rows[0][0] == 1, f"Individual FM event at line {300 + i} should remain active"

    # No aggregated event
    rows = stream.query_raw(
        "SELECT status FROM events WHERE event_type=? AND payload LIKE '%all GPUs%'",
        [bad_id],
    )
    assert len(rows) == 0


def test_get_pytorch_timeout_minutes(stream, obs_ctx):
    # Add runtime args event
    t1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    payload = {"pytorch_timeout_minutes": 15.5}
    create_event(stream, obs_ctx, "Application.megatron-lm.runtime_args", payload, t1, 100)

    timeout_ss = get_pytorch_timeout_minutes(stream)
    assert timeout_ss is not None
    # get_pytorch_timeout_minutes returns a SourceSet, we need to get the value
    assert float(timeout_ss.value()) == 15.5


def test_get_pytorch_timeout_minutes_skips_payload_without_keys(stream, obs_ctx):
    # An earlier runtime_args payload that carries neither timeout key must not
    # short-circuit the scan; the value in a later payload should still be found.
    t1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    t2 = datetime(2024, 1, 1, 10, 5, 0, tzinfo=ZoneInfo("UTC"))
    create_event(stream, obs_ctx, "Application.megatron-lm.runtime_args", {"other_arg": 1}, t1, 100)
    create_event(
        stream, obs_ctx, "Application.megatron-lm.runtime_args", {"distributed_timeout_minutes": 20.0}, t2, 101
    )

    timeout_ss = get_pytorch_timeout_minutes(stream)
    assert timeout_ss is not None
    assert float(timeout_ss.value()) == 20.0


class TestNcclLateArrival:
    _NODES = ("node-000", "node-001", "node-002", "node-003")
    _GET_INTERESTING_NODE_QUERY = "SELECT payload FROM events WHERE event_type=? AND payload LIKE '%NCCL Late Arrival%'"

    @staticmethod
    def _inject_watchdog_report(stream, obs_ctx, ranks):
        """Inject an Application.NCCLWatchdog.report event carrying the given watchdog ranks."""
        source = Source(
            source_context=obs_ctx,
            line_num=100,
            datum="dummy line",
            time=datetime(2024, 1, 1, 10, 30, 0, tzinfo=ZoneInfo("UTC")),
            resources=AllJobResources,
            rule="NcclWatchdog",
        )
        data = {
            "source": source.to_dict(),
            "stacktrace_by_key": {},
            "timeouts_by_key": {
                "WorkNCCL(SeqNum=506, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)": {
                    "ranks": list(ranks),
                    "timestamp_from_rank": {str(r): "E222 10:30:00.000000000" for r in ranks},
                    "example": (
                        "WorkNCCL(SeqNum=506, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)"
                        " ran for 600001 milliseconds before timing out."
                    ),
                }
            },
        }
        stream.add_event("Application.NCCLWatchdog.report", {"data": data}, source)
        stream.flush()

    def test_global_straggler_detected(self, stream, obs_ctx):
        """Missing rank's node must appear as an interesting node after filter runs."""
        job_info = _make_job_info(ranks_per_node=1, nodes=self._NODES)
        # ranks 0, 1, 3 report; rank 2 (node-002) never reports
        self._inject_watchdog_report(stream, obs_ctx, (0, 1, 3))

        filter_events(stream, job_info)
        stream.flush()

        int_id = stream.mgr.types.get_id("Attribution.node.interesting")
        rows = stream.query_raw(self._GET_INTERESTING_NODE_QUERY, [int_id])
        assert len(rows) == 1
        assert "NCCL Late Arrival" in rows[0][0]
        assert "node-002" in rows[0][0]

    def test_no_straggler_when_all_global_ranks_reported(self, stream, obs_ctx):
        """Emitting a "straggler" does not make sense in the case where all ranks have reported the watchdog timeout."""
        job_info = _make_job_info(ranks_per_node=1, nodes=self._NODES)
        self._inject_watchdog_report(stream, obs_ctx, (0, 1, 2, 3))

        filter_events(stream, job_info)
        stream.flush()

        int_id = stream.mgr.types.get_id("Attribution.node.interesting")
        rows = stream.query_raw(self._GET_INTERESTING_NODE_QUERY, [int_id])
        assert len(rows) == 0

    def test_straggler_detected_in_multi_node_process_group(self, stream, obs_ctx):
        """Straggler detected when there is a stride in the process group."""
        # 4 nodes × 8 GPUs = world_size=32, process group: {0,8,16,24}
        job_info = _make_job_info(ranks_per_node=8, nodes=self._NODES)
        # ranks 0, 16, 24 report; rank 8 (node-001) never reports
        self._inject_watchdog_report(stream, obs_ctx, (0, 16, 24))

        filter_events(stream, job_info)
        stream.flush()

        int_id = stream.mgr.types.get_id("Attribution.node.interesting")
        rows = stream.query_raw(self._GET_INTERESTING_NODE_QUERY, [int_id])
        assert len(rows) == 1
        assert "node-001" in rows[0][0]

    def test_no_straggler_when_all_ranks_in_multi_node_process_group_report(self, stream, obs_ctx):
        """Emitting a "straggler" does not make sense in the case where all ranks in a process group have reported the watchdog timeout."""
        job_info = _make_job_info(ranks_per_node=8, nodes=self._NODES)
        # all ranks in the strided process group report
        self._inject_watchdog_report(stream, obs_ctx, (0, 8, 16, 24))

        filter_events(stream, job_info)
        stream.flush()

        int_id = stream.mgr.types.get_id("Attribution.node.interesting")
        rows = stream.query_raw(self._GET_INTERESTING_NODE_QUERY, [int_id])
        assert len(rows) == 0

    def test_no_straggler_when_all_ranks_in_single_node_process_group_report(self, stream, obs_ctx):
        """Emitting a "straggler" does not make sense in the case where all ranks in a process group have reported the watchdog timeout."""
        job_info = _make_job_info(ranks_per_node=8, nodes=self._NODES)
        # all 8 ranks of node-001 report (single-node group, no straggler)
        self._inject_watchdog_report(stream, obs_ctx, (8, 9, 10, 11, 12, 13, 14, 15))

        filter_events(stream, job_info)
        stream.flush()

        int_id = stream.mgr.types.get_id("Attribution.node.interesting")
        rows = stream.query_raw(self._GET_INTERESTING_NODE_QUERY, [int_id])
        assert len(rows) == 0

    def test_multiple_stragglers_detected(self, stream, obs_ctx):
        """Multiple stragglers detected in a large job."""
        nodes = [f"node-{i:03d}" for i in range(200)]
        job_info = _make_job_info(ranks_per_node=1, nodes=nodes)
        # ranks 0-199 report; ranks 25, 50 (node-025, node-050) never report (1% of nodes)
        reported_ranks = set(range(200))
        reported_ranks.remove(25)  # node-025
        reported_ranks.remove(50)  # node-050
        self._inject_watchdog_report(stream, obs_ctx, reported_ranks)

        filter_events(stream, job_info)
        stream.flush()

        int_id = stream.mgr.types.get_id("Attribution.node.interesting")
        rows = stream.query_raw(self._GET_INTERESTING_NODE_QUERY, [int_id])
        assert len(rows) == 2
        assert "node-025" in rows[0][0]
        assert "node-050" in rows[1][0]

    # Known limitation: False negative.
    def test_size_guard_misses_two_independent_dp_stragglers(self, stream, obs_ctx):
        """No stragglers detected when two independent DP-group stragglers break stride inference."""
        job_info = _make_job_info(ranks_per_node=8, nodes=self._NODES)
        # rank 8 missing from DP group {0,8,16,24}; rank 25 missing from {1,9,17,25}
        # The combined reporter set {0,1,9,16,17,24} has no coherent stride and falls back to global.
        self._inject_watchdog_report(stream, obs_ctx, {0, 16, 24, 1, 9, 17})

        filter_events(stream, job_info)
        stream.flush()

        int_id = stream.mgr.types.get_id("Attribution.node.interesting")
        rows = stream.query_raw(self._GET_INTERESTING_NODE_QUERY, [int_id])
        assert len(rows) == 0, "size guard should suppresses stragglers when stride inference collapses to global"

    # Known limitation: False positive.
    def test_false_positive_when_comm_smaller_than_stride_period(self, stream, obs_ctx):
        """False positive: a node is attributed as straggler when it was never in the communicator."""
        job_info = _make_job_info(ranks_per_node=8, nodes=self._NODES)
        # All 3 members of the actual communicator reported: no real straggler exists.
        self._inject_watchdog_report(stream, obs_ctx, {0, 8, 16})

        filter_events(stream, job_info)
        stream.flush()

        int_id = stream.mgr.types.get_id("Attribution.node.interesting")
        rows = stream.query_raw(self._GET_INTERESTING_NODE_QUERY, [int_id])
        # Known false positive: node-003 attributed as straggler despite never being in the comm.
        assert len(rows) == 1
        assert "node-003" in rows[0][0]
