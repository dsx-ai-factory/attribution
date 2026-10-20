# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for Slurm timeout XID-145 cascade handling."""

from datetime import UTC, datetime, timedelta

from attribution_lib.attributor import Attributor
from attribution_lib.job_info import Scheduler
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Observation as InboundObservation,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.resources import AllJobResources
from test_support.helpers import line_entries, make_applog_entry, make_syslog_entry


def test_terminal_rxpipe_after_raw_slurm_timeout_finishes_as_pass(
    request,
    event_manager,
    simple_job_factory,
    timezone,
):
    """Parse the Jira log shape and suppress only the resulting teardown XID."""
    job_start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone)
    job_end = job_start + timedelta(minutes=60)
    job_info = simple_job_factory(
        job_id="test-job-145",
        state="CANCELLED",
        start_timestamp=int(job_start.timestamp()),
        end_timestamp=int(job_end.timestamp()),
        nodes=["node01"],
        time_limit=60,
        exit_code_signal=0,
        scheduler=Scheduler.SLURM,
    )
    timeout_time = job_end - timedelta(seconds=97)
    xid_time = job_end - timedelta(seconds=34)
    xid_timestamp_ns = int(xid_time.timestamp() * 1_000_000_000)
    timeout_timestamp_ns = int(timeout_time.timestamp() * 1_000_000_000)
    interval = TimeInterval(job_info.job_start_time, job_info.job_end_time)

    syslog = InboundObservation(
        context=InboundContext(
            time_interval=interval,
            source=ObservationSource.SYSLOG,
            format=ObservationFormat.ENTRY_TIMESERIES,
            timezone=timezone,
        ),
        lines=[
            make_syslog_entry(
                body=(
                    "kernel: NVRM: Xid (PCI:0000:e4:00): 145, NVLINK_RLW_ERROR (0x00000006 0x00000008 0x0 0x0 0x0 0x0)"
                ),
                node="node01",
                timestamp_ns=xid_timestamp_ns,
            )
        ],
    )
    applog = InboundObservation(
        context=InboundContext(
            time_interval=interval,
            source=ObservationSource.APPLOG,
            format=ObservationFormat.ENTRY_TIMESERIES,
            timezone=timezone,
        ),
        lines=[
            make_applog_entry(
                body="0: error: *** STEP 424242.0 CANCELLED DUE TO TIME LIMIT ***",
                hostname="node01",
                timestamp_ns=timeout_timestamp_ns,
            )
        ],
    )

    stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(stream, job_info)
    attributor.ingest(syslog)
    attributor.ingest(applog)

    result = attributor.attribute()

    assert result.outcome_attr.value().outcome == "Pass"
    assert set(result.symptoms.symptoms()) == {"Cancelled by Timeout"}
    node_symptoms = result.bad_nodes_attr.my_bad_nodes["node01"].attributions
    assert "XID-145-RLW_RXPIPE-8" not in node_symptoms
    assert "XID-145-MultiCategory" not in node_symptoms
    assert node_symptoms._data["XID-145-RLW_RXPIPE-8"].has_interesting()
    assert node_symptoms._data["XID-145-MultiCategory"].has_interesting()


def test_timeout_cascades_and_unused_xid_leave_only_timeout_failure(
    request,
    event_manager,
    simple_job_factory,
    timezone,
):
    """Verify that cascaded and informational XIDs do not override a timeout."""
    job_start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    job_end = datetime(2025, 1, 1, 4, 0, 36, tzinfo=UTC)
    job_info = simple_job_factory(
        job_id="424242",
        cluster="sample-cluster",
        state="CANCELLED",
        start_timestamp=int(job_start.timestamp()),
        end_timestamp=int(job_end.timestamp()),
        nodes=["test-node-alpha", "test-node-bravo"],
        time_limit=None,
        exit_code_signal=0,
        scheduler=Scheduler.SLURM,
    )
    interval = TimeInterval(job_info.job_start_time, job_info.job_end_time)

    syslog = InboundObservation(
        context=InboundContext(
            time_interval=interval,
            source=ObservationSource.SYSLOG,
            format=ObservationFormat.ENTRY_TIMESERIES,
            timezone=timezone,
        ),
        lines=[
            make_syslog_entry(
                body=(
                    "NVRM: Xid (PCI:0000:01:00): 108, NVLink remote translation error: "
                    "faulted @ 0x0000_00001000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ"
                ),
                node="test-node-bravo",
                timestamp_ns=1735703990465323000,
            ),
            make_syslog_entry(
                body=("NVRM: Xid (PCI:0000:01:00): 94, pid=1234, name=worker, Contained: SM (0x1). RST: No, D-RST: No"),
                node="test-node-alpha",
                timestamp_ns=1735703990466455000,
            ),
            make_syslog_entry(
                body=(
                    "NVRM: Xid (PCI:0000:01:00): 145, pid=5678, name=python, "
                    "RLW_RXPIPE Nonfatal XC0 i0 Link 00 "
                    "(0x04080006 0x00000008 0x00000000 0x00000000 0x00000000 0x00000000)"
                ),
                node="test-node-alpha",
                timestamp_ns=1735703990536625000,
            ),
            make_syslog_entry(
                body=(
                    "NVRM: Xid (PCI:0000:01:00): 137, pid=5678, name=python, "
                    "RLW_RXPIPE interrupt hit on link 0 on GPU0: PRIV Error"
                ),
                node="test-node-alpha",
                timestamp_ns=1735703990536645000,
            ),
        ],
    )
    applog = InboundObservation(
        context=InboundContext(
            time_interval=interval,
            source=ObservationSource.APPLOG,
            format=ObservationFormat.ENTRY_TIMESERIES,
            timezone=timezone,
        ),
        lines=line_entries(
            [
                "0: [Jan 01 03:59:47.858661 9012 slurmstepd 0x000000001234] error: "
                "*** STEP 424242.0 ON test-node-alpha CANCELLED AT "
                "2025-01-01T03:59:47 DUE TO TIME LIMIT ***"
            ]
        ),
    )

    stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(stream, job_info)
    attributor.ingest(syslog)
    attributor.ingest(applog)

    result = attributor.attribute()

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == "Cancelled by Timeout"
    assert outcome.hw_cause == "Not Applicable"
    assert result.cancelled_by == "Timeout"
    assert set(result.symptoms.symptoms()) == {"Cancelled by Timeout"}

    timeout_sources = list(result.symptoms._data["Cancelled by Timeout"].all_sources())
    assert timeout_sources
    for symptom in (
        "XID-145-RLW_RXPIPE-8",
        "XID-145-MultiCategory",
        "XID-137-NVLINK_PRIV_ERR",
        "XID-94-CONTAINED_ERROR-SM",
    ):
        assert not result.symptoms._data[symptom].has_bad()
        assert result.symptoms._data[symptom].has_interesting()
    # Informational XIDs are filtered before job-level aggregation, but their
    # per-node source remains available as interesting evidence below.
    assert "XID-108-NVLINK_REMOTE_TRANSLATION_ERROR" not in result.symptoms._data

    cascade_node = result.bad_nodes_attr.my_bad_nodes["test-node-alpha"].attributions
    for symptom in (
        "XID-145-RLW_RXPIPE-8",
        "XID-145-MultiCategory",
        "XID-137-NVLINK_PRIV_ERR",
        "XID-94-CONTAINED_ERROR-SM",
    ):
        assert symptom not in cascade_node
        assert cascade_node._data[symptom].has_interesting()

    xid108_node = result.bad_nodes_attr.my_bad_nodes["test-node-bravo"].attributions
    assert "XID-108-NVLINK_REMOTE_TRANSLATION_ERROR" not in xid108_node
    assert xid108_node._data["XID-108-NVLINK_REMOTE_TRANSLATION_ERROR"].has_interesting()
