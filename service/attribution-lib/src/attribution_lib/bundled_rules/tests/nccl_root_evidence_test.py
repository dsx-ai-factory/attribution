# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the evidence for a selected NCCL outcome through symptom cleanup."""

from datetime import UTC, datetime, timedelta

import pytest

from attribution_lib.attributor import Attributor
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.job_info import Scheduler
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Observation as InboundObservation,
    Source as ObservationSource,
    TimeInterval,
)
from test_support.helpers import make_applog_entry, make_syslog_entry

_NCCL_ERROR = "0: ncclSystemError"
_FILE_ERROR = "1: FileNotFoundError: [Errno 2] No such file or directory: '/data/checkpoint.pt'"


@pytest.fixture
def attribute_logs(request, event_manager):
    """Attribute synthetic log evidence well before the job ends."""

    def _attribute(lines, *, state="FAILED", syslog_lines=()):
        job = default_job_info(
            nodes=["node01", "node02"],
            state=state,
            job_end_time=datetime(2024, 1, 1, 11, tzinfo=UTC),
            exit_code_signal=None,
            scheduler=Scheduler.SLURM,
        )
        timestamp_ns = int((job.job_start_time + timedelta(minutes=5)).timestamp() * 10**9)
        stream = event_manager.create_stream(request.node.name)
        attributor = Attributor.new(stream, job)
        for source, entries in (
            (ObservationSource.APPLOG, [make_applog_entry(line, timestamp_ns) for line in lines]),
            (ObservationSource.SYSLOG, [make_syslog_entry(line, "node02", timestamp_ns) for line in syslog_lines]),
        ):
            if entries:
                attributor.ingest(
                    InboundObservation(
                        context=InboundContext(
                            time_interval=TimeInterval(job.job_start_time, job.job_end_time),
                            source=source,
                            format=ObservationFormat.ENTRY_TIMESERIES,
                            timezone=job.timezone,
                        ),
                        lines=entries,
                    )
                )
        return attributor.attribute()

    return _attribute


@pytest.mark.parametrize("state", ["FAILED", "PREEMPTED"])
def test_selected_nccl_root_keeps_its_source_beside_a_file_error(attribute_logs, state):
    result = attribute_logs([_NCCL_ERROR, _FILE_ERROR], state=state)

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail MultiCategory"
    assert outcome.root_cause_symptom == "ncclSystemError"
    assert outcome.doc == "ncclsystemerror"
    assert set(result.symptoms.symptoms()) == {"ncclSystemError", "No such file or directory"}
    source = result.symptoms["ncclSystemError"].one_source()
    assert source.datum == _NCCL_ERROR
    assert source.rule == "alias else"
    assert source.line_num == 1
    assert source.time == datetime(2024, 1, 1, 10, 5, tzinfo=UTC)
    assert {resource.hostname for resource in source.resources} == {"node01"}
    assert result.outcome_attr.one_source() == source


def test_hardware_winner_still_suppresses_secondary_nccl_error(attribute_logs):
    result = attribute_logs(
        [_NCCL_ERROR],
        syslog_lines=["NVRM: Xid (PCI:0000:06:00): 48, pid=1234, name=worker, Uncorrectable memory error"],
    )

    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert result.outcome_attr.value().root_cause_symptom == "XID-48-GPU_ECC_DBE"
    assert "ncclSystemError" not in result.symptoms


@pytest.mark.parametrize(
    "line, expected_outcome, expected_root",
    [
        pytest.param(_NCCL_ERROR, "Fail MultiCategory", "ncclSystemError", id="nccl-only"),
        pytest.param(_FILE_ERROR, "Fail User", "No such file or directory", id="file-only"),
    ],
)
def test_single_error_outcome_stays_unchanged(attribute_logs, line, expected_outcome, expected_root):
    result = attribute_logs([line])

    assert result.outcome_attr.value().outcome == expected_outcome
    assert result.outcome_attr.value().root_cause_symptom == expected_root
    assert set(result.symptoms.symptoms()) == {expected_root}
