# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from attribution_lib.observation import Observation
from attribution_lib.observation.inbound import (
    Format as LibFormat,
    Source,
)
from attribution_lib.observation.processing import EntrySourceBase
from attribution_lib.rules_engine import body_fallbacks_for, process_observation_source

from app.util import TimeInterval
from app.v1.legacy_formats import _raw_loki_unpack, extract_body_and_node
from app.v1.observation import ObservationContextV1, ObservationFormat, ObservationV1
from tests.conftest import make_up_attributor_info, post_attributor

JOB_START = datetime(2025, 4, 15, 10, 0, tzinfo=timezone.utc)
JOB_END = datetime(2025, 4, 15, 11, 0, tzinfo=timezone.utc)
# A year after the job end, so a year-less timestamp proves it took its year from here, not from the job.
DECLARED_END = datetime(2026, 1, 10, tzinfo=timezone.utc)
TS_NS = 1744734437830498765
# The legacy float-seconds conversion rounded to the nearest microsecond.
TS_NS_LEGACY = 1744734437830499000


def _node(hostname: str) -> list[dict[str, str]]:
    return [{"kind": "node", "hostname": hostname}]


def _job_window(ji):
    return ji.job_start_time, ji.job_end_time


def _interest_window(ji):
    return ji.interest_start_time, ji.interest_end_time


@pytest.mark.parametrize(
    ("source", "format", "body", "expected_entries", "expected_window"),
    [
        (
            Source.APPLOG,
            ObservationFormat.RAW_LOG_LINES,
            "[E1231 23:57:09.5 ProcessGroupNCCL.cpp:1923] 3: watchdog timeout\nno timestamp",
            [
                {"body": "[E1231 23:57:09.5 ProcessGroupNCCL.cpp:1923] 3: watchdog timeout", "ts": 1767225429 * 10**9},
                {"body": "no timestamp", "ts": None},
            ],
            lambda ji: (None, None),
        ),
        (
            Source.APPLOG,
            ObservationFormat.LOKI_DICT_LIST,
            [{"timestamp": str(TS_NS), "line": json.dumps({"body": "0: line"})}],
            [{"body": "0: line", "ts": TS_NS_LEGACY}],
            lambda ji: (None, None),
        ),
        (
            Source.APPLOG,
            ObservationFormat.RAW_LOKI_STREAMS,
            [
                {
                    "stream": {"node_name": "node1", "pod": "worker-0"},
                    "values": [[str(TS_NS), json.dumps({"body": "0: line", "attributes": {"hostname": "node2"}})]],
                }
            ],
            [
                {
                    "body": "0: line",
                    "ts": TS_NS_LEGACY,
                    "resources": [{"kind": "node", "hostname": "node1"}, {"kind": "replica", "name": "worker-0"}],
                }
            ],
            lambda ji: (None, None),
        ),
        (
            Source.SYSLOG,
            ObservationFormat.LOKI_DICT_LIST,
            [{"timestamp": TS_NS, "node": "node2", "body": "kernel: Xid"}],
            [{"body": "kernel: Xid", "ts": TS_NS_LEGACY, "resources": _node("node2")}],
            _interest_window,
        ),
        (
            Source.HEALTH_CHECK,
            ObservationFormat.LOKI_DICT_LIST,
            [{"timestamp": 1744734437830, "node": "node3", "body": "check failed"}],
            [{"body": "check failed", "ts": 1744734437830000000, "resources": _node("node3")}],
            lambda ji: (ji.interest_start_time, ji.interest_end_time + timedelta(minutes=10)),
        ),
        (
            Source.HEALTH_CHECK,
            ObservationFormat.RAW_LOKI_STREAMS,
            [{"stream": {"pod": "ignored"}, "values": [[str(TS_NS), "Apr 15 10:00:00 node3 check failed"]]}],
            [{"body": "Apr 15 10:00:00 node3 check failed", "ts": TS_NS_LEGACY}],
            lambda ji: (ji.interest_start_time, ji.interest_end_time + timedelta(minutes=10)),
        ),
        (
            Source.UFMLOG,
            ObservationFormat.UFMLOG,
            [{"body": "(node4) node is Down", "node": "fabric-manager", "timestamp": TS_NS}],
            [{"body": "(node4) node is Down", "ts": TS_NS_LEGACY}],
            _job_window,
        ),
        (
            Source.NODE_METRICS,
            ObservationFormat.PROMETHEUS_METRICS,
            [{"timestamp": 1744734437, "value": "1", "metric": "m", "labels": {}, "node": "node5", "threshold": 5}],
            [
                {
                    "body": (
                        "time: 1744734437, metric_name: m, metric_value: 1, threshold: None, "
                        "cluster: None, node: node5, component: None"
                    ),
                    "ts": 1744734437 * 10**9,
                    "resources": _node("node5"),
                }
            ],
            _job_window,
        ),
        (
            Source.CSP_EVENT,
            ObservationFormat.CSP_EVENT_LIST,
            [
                {
                    "hostname": "node6",
                    "termination_timestamp_s": 1718452800.123456,
                    "category": "HOST_FAILURE",
                    "reason": "BAD_GPU",
                    "version": 1,
                    "source_event": {
                        "source": "s",
                        "event_id": "e",
                        "event_created_at": "t",
                        "event_region": "r",
                        "hostname": "node6",
                        "fault_id": "f",
                    },
                }
            ],
            [
                {
                    "body": (
                        "csp_event: category=HOST_FAILURE, reason=BAD_GPU, status=, source_event="
                        '{"event_created_at":"t","event_id":"e","event_region":"r","fault_id":"f",'
                        '"hostname":"node6","source":"s"}'
                    ),
                    "ts": 1718452800123456000,
                    "resources": _node("node6"),
                }
            ],
            lambda ji: (ji.job_start_time, None),
        ),
    ],
)
def test_legacy_body_converts_to_entry_timeseries_over_the_source_window(
    source, format, body, expected_entries, expected_window
):
    job_info = make_up_attributor_info("v1", job_start_time=JOB_START, job_end_time=JOB_END).to_lib_jobinfo()
    observation = ObservationV1(
        context=ObservationContextV1(
            time_interval=TimeInterval(start=JOB_START, end=DECLARED_END), source=source, format=format
        ),
        body=body if isinstance(body, str) else json.dumps(body),
    )

    lib_observation = observation.to_lib_observation(job_info)

    assert lib_observation.lines == [{"resources": None} | entry for entry in expected_entries]
    assert lib_observation.context.format is LibFormat.ENTRY_TIMESERIES
    lib_window = lib_observation.context.time_interval
    assert (lib_window.start, lib_window.end) == expected_window(job_info)


def _health_check_observation(format, records=None):
    """Encode synthetic timestamp/body pairs in either legacy health-check format."""
    if records is None:
        records = [(JOB_END + timedelta(minutes=1), "health check failed")]
    if format is ObservationFormat.LOKI_DICT_LIST:
        payload = [
            {"timestamp": int(moment.timestamp()) * 10**3, "node": "node0", "body": body} for moment, body in records
        ]
    else:
        payload = [
            {
                "stream": {"node_name": "node0"},
                "values": [[str(int(moment.timestamp()) * 10**9), body] for moment, body in records],
            }
        ]
    return ObservationV1(
        context=ObservationContextV1(
            time_interval=TimeInterval(start=JOB_START, end=DECLARED_END),
            source=Source.HEALTH_CHECK,
            format=format,
        ),
        body=json.dumps(payload),
    )


@pytest.mark.parametrize("format", [ObservationFormat.LOKI_DICT_LIST, ObservationFormat.RAW_LOKI_STREAMS])
@pytest.mark.parametrize(
    ("last_tflops_time", "expected_start"),
    [
        pytest.param(None, JOB_START, id="no-progress"),
        pytest.param(JOB_START - timedelta(seconds=1), JOB_START, id="before-window"),
        pytest.param(JOB_START, JOB_START, id="at-start"),
        pytest.param(JOB_START + timedelta(minutes=1), JOB_START + timedelta(minutes=1), id="inside-window"),
        pytest.param(JOB_END + timedelta(minutes=5), JOB_END + timedelta(minutes=5), id="inside-epilog-tail"),
        pytest.param(JOB_END + timedelta(minutes=10), JOB_END + timedelta(minutes=10), id="at-end"),
        pytest.param(JOB_END + timedelta(minutes=10, microseconds=1), JOB_START, id="just-after-window"),
        pytest.param(JOB_END + timedelta(hours=1), JOB_START, id="later-shared-log-progress"),
    ],
)
def test_legacy_health_check_progress_respects_window(format, last_tflops_time, expected_start):
    job_info = make_up_attributor_info(
        "v1", job_start_time=JOB_START, job_end_time=JOB_END, last_tflops_time=last_tflops_time
    ).to_lib_jobinfo()

    observation = _health_check_observation(format).to_lib_observation(job_info)

    assert observation.context.time_interval.start == expected_start
    assert observation.context.time_interval.end == JOB_END + timedelta(minutes=10)
    assert observation.lines == [
        {
            "body": "health check failed",
            "ts": int((JOB_END + timedelta(minutes=1)).timestamp()) * 10**9,
            "resources": _node("node0"),
        }
    ]


@pytest.mark.parametrize("format", [ObservationFormat.LOKI_DICT_LIST, ObservationFormat.RAW_LOKI_STREAMS])
@pytest.mark.parametrize("narrow_interest", [False, True], ids=["job-window", "narrow-interest-window"])
def test_legacy_health_check_evidence_reaches_rules_after_time_filtering(format, narrow_interest):
    """Keep the failure inside the interest window and exclude evidence outside it."""
    interest_start = JOB_START + timedelta(minutes=20) if narrow_interest else JOB_START
    interest_end = JOB_START + timedelta(minutes=30) if narrow_interest else JOB_END
    # In the narrow case, progress is inside the job but beyond the interest tail.
    progress = JOB_START + timedelta(minutes=50) if narrow_interest else JOB_END + timedelta(hours=1)
    job_info = make_up_attributor_info(
        "v1",
        job_start_time=JOB_START,
        job_end_time=JOB_END,
        interest_start_time=interest_start,
        interest_end_time=interest_end,
        last_tflops_time=progress,
    ).to_lib_jobinfo()
    failure_time = interest_end + timedelta(minutes=1)
    failure = "Health check failed: Not responding"
    wire = _health_check_observation(
        format,
        records=[
            (interest_start - timedelta(seconds=1), "before interest start"),
            (failure_time, failure),
            (interest_end + timedelta(minutes=10, seconds=1), "after epilog tail"),
        ],
    )
    observation = wire.to_lib_observation(job_info)
    assert observation.context.time_interval.start == interest_start
    assert observation.context.time_interval.end == interest_end + timedelta(minutes=10)

    engine = Mock()
    process_observation_source(
        event_stream=None,
        observation_source=EntrySourceBase.from_observation(
            Observation.from_inbound_observation(observation),
            body_fallbacks=body_fallbacks_for(Source.HEALTH_CHECK, job_info),
        ),
        job_info=job_info,
        rules_engine=engine,
        pre_filtered=True,
    )

    # The production time/node filters run; capture what reaches rule processing.
    processed = engine.observation_processor.return_value.process_line
    processed.assert_called_once()
    evidence = processed.call_args.args[0]
    assert evidence.datum == failure
    assert evidence.time == failure_time
    assert evidence.source_context.source is Source.HEALTH_CHECK


@pytest.mark.parametrize("format", [ObservationFormat.LOKI_DICT_LIST, ObservationFormat.RAW_LOKI_STREAMS])
def test_v1_accepts_health_check_with_progress_after_window(client, format):
    info = make_up_attributor_info(
        "v1", job_start_time=JOB_START, job_end_time=JOB_END, last_tflops_time=JOB_END + timedelta(hours=1)
    )
    att_id = post_attributor(client, info)

    response = client.post(
        f"/v1/attributor/{att_id}/observation", content=_health_check_observation(format).model_dump_json()
    )

    assert response.status_code == 201, response.text
    assert client.get(f"/v1/attributor/{att_id}/attribution").status_code == 200


class TestExtractBodyAndNode:
    """Tests for extract_body_and_node covering all three envelope formats."""

    def test_raw_envelope_body_and_attributes(self):
        """Raw log envelope format: top-level 'body' + attributes.hostname."""
        parsed = {"body": "some syslog line", "attributes": {"hostname": "node-01"}}
        body, node = extract_body_and_node(parsed, None, '{"body":"..."}')
        assert body == "some syslog line"
        assert node == "node-01"

    def test_lepton_message(self):
        """Lepton format: top-level 'message', no attributes."""
        parsed = {"message": "lepton log line"}
        body, node = extract_body_and_node(parsed, None, '{"message":"..."}')
        assert body == "lepton log line"
        assert node is None

    def test_cloudevents_fm_log(self):
        """CloudEvents envelope: data.message + data.hostname."""
        raw = '{"data":{"hostname":"nvswitch-01","message":"nmxc-fm: Lost connection to GPU gpuGUID 0xabc"}}'
        parsed = json.loads(raw)
        body, node = extract_body_and_node(parsed, None, raw)
        assert "Lost connection to GPU" in body
        assert "gpuGUID 0xabc" in body
        assert node == "nvswitch-01"

    def test_cloudevents_does_not_overwrite_existing_node(self):
        """When node is already known, data.hostname should not overwrite it."""
        parsed = {"data": {"hostname": "nvswitch-01", "message": "fm log"}}
        body, node = extract_body_and_node(parsed, "already-set", "{}")
        assert body == "fm log"
        assert node == "already-set"

    def test_attributes_does_not_overwrite_existing_node(self):
        """When node is already known, attributes.hostname should not overwrite it."""
        parsed = {"body": "log line", "attributes": {"hostname": "attr-node"}}
        body, node = extract_body_and_node(parsed, "stream-node", "{}")
        assert body == "log line"
        assert node == "stream-node"

    def test_unknown_format_falls_back_to_raw_payload(self):
        """Unknown JSON structure falls back to the raw payload string."""
        raw = '{"specversion":"1.0","type":"event","data":{"severity":"INFO"}}'
        parsed = json.loads(raw)
        body, node = extract_body_and_node(parsed, None, raw)
        assert body == raw
        assert node is None

    def test_data_not_dict_falls_back_to_raw(self):
        """data field is a string, not a dict - skip CloudEvents path, fall back to raw."""
        raw = '{"data":"just a string"}'
        parsed = json.loads(raw)
        body, node = extract_body_and_node(parsed, None, raw)
        assert body == raw

    def test_raw_loki_unpack_cloudevents(self):
        """Integration: raw Loki unpacking correctly unpacks a CloudEvents FM log."""
        raw = json.dumps(
            {
                "data": {
                    "hostname": "nvswitch-02",
                    "message": "nmxc-fm: Lost connection to GPU gpuGUID 0xdef",
                }
            }
        )
        result = _raw_loki_unpack(1000000000, raw, None, None)
        assert "Lost connection to GPU" in result["body"]
        assert result["node"] == "nvswitch-02"

    def test_raw_loki_unpack_preserves_existing_node(self):
        """Raw Loki unpacking should not clobber node from stream labels."""
        raw = json.dumps({"body": "some log", "attributes": {"hostname": "attr-node"}})
        result = _raw_loki_unpack(1000000000, raw, "stream-node", None)
        assert result["node"] == "stream-node"
