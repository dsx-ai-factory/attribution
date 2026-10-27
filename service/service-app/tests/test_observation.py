# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: parametrize these tests across v1/v2

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from attribution_lib.observation.inbound import Source as ObservationSource

from app.attributor import (
    AttributorMetadata,
    WorkloadTypeEnum,
)
from app.util import TimeInterval
from app.v1.models import (
    AttributionResponseV1,
    AttributorInfoV1,
    AttributorWorkloadInfoV1,
    GlobalAttribution,
    NodeAttribution,
)
from app.v1.observation import ObservationContextV1, ObservationFormat, ObservationV1
from tests.conftest import make_up_attributor_info, post_attributor

JOB_NODE = "node0"
UNBOUNDED = {"start": None, "end": None}


@pytest.fixture
def attributor_info() -> AttributorInfoV1:
    return AttributorInfoV1(
        workload=AttributorWorkloadInfoV1(
            id="120123",
            type=WorkloadTypeEnum.SLURM,
            cancelled_by=None,
            last_tflops_time=None,
            nranks=10,
            ranks_per_node=8,
            job_start_time=datetime.fromtimestamp(1741860016).astimezone(),
            job_end_time=datetime.fromtimestamp(1744734448).astimezone(),
            exit_code_signal=1,
            nodes="test-node-cpu-003-020-009,test-node-cpu-003-022-017,test-node-cpu-003-024-009,test-node-cpu-003-024-017,test-node-h100-001-006-003,test-node-h100-001-006-013,test-node-h100-001-006-023,test-node-h100-001-006-033,test-node-h100-001-007-033,test-node-h100-001-008-033,test-node-h100-001-009-003,test-node-h100-001-010-003,test-node-h100-001-010-013,test-node-h100-001-010-023,test-node-h100-001-011-013,test-node-h100-001-011-033,test-node-h100-001-042-026,test-node-h100-001-044-026,test-node-h100-001-047-012,test-node-h100-001-049-012,test-node-h100-001-049-026,test-node-h100-001-054-012,test-node-h100-001-054-026,test-node-h100-001-057-012,test-node-h100-001-058-012,test-node-h100-001-062-012,test-node-h100-001-063-012,test-node-h100-001-064-012,test-node-h100-001-066-026,test-node-h100-001-072-012,test-node-h100-001-077-012,test-node-h100-001-083-012,test-node-h100-001-084-012,test-node-h100-001-084-026,test-node-h100-001-086-012,test-node-h100-001-087-026,test-node-h100-001-089-026,test-node-h100-001-093-012,test-node-h100-001-095-012,test-node-h100-001-098-012,test-node-h100-001-098-026,test-node-h100-001-099-012,test-node-h100-001-122-026,test-node-h100-001-149-026,test-node-h100-001-152-012,test-node-h100-001-152-026,test-node-h100-001-155-012,test-node-h100-001-156-012,test-node-h100-001-156-026,test-node-h100-001-157-026,test-node-h100-001-158-012,test-node-h100-001-158-026,test-node-h100-001-162-026,test-node-h100-001-166-012,test-node-h100-001-168-012,test-node-h100-001-243-026,test-node-h100-001-249-012,test-node-h100-001-255-026,test-node-h100-001-259-026,test-node-h100-001-273-012,test-node-h100-001-274-012,test-node-h100-001-277-012,test-node-h100-001-277-026,test-node-h100-001-287-026,test-node-h100-001-297-026,test-node-h100-001-302-026,test-node-h100-001-306-012,test-node-h100-001-307-026,test-node-h100-001-318-026,test-node-h100-001-321-012,test-node-h100-001-328-012,test-node-h100-001-329-012,test-node-h100-001-339-026,test-node-h100-001-342-026,test-node-h100-001-346-026,test-node-h100-001-347-012,test-node-h100-001-353-026,test-node-h100-001-355-012,test-node-h100-001-355-026,test-node-h100-001-357-026,test-node-h100-001-359-012,test-node-h100-001-359-026,test-node-h100-001-368-026,test-node-h100-001-380-026,test-node-h100-002-043-012,test-node-h100-002-061-026,test-node-h100-002-063-012,test-node-h100-002-068-012,test-node-h100-002-083-026,test-node-h100-002-085-026,test-node-h100-002-104-012,test-node-h100-002-104-026,test-node-h100-002-134-026,test-node-h100-002-147-026,test-node-h100-002-149-026,test-node-h100-002-153-026,test-node-h100-002-154-012,test-node-h100-002-156-026,test-node-h100-002-176-012,test-node-h100-002-177-012,test-node-h100-002-177-026,test-node-h100-002-179-012,test-node-h100-002-179-026,test-node-h100-002-204-003,test-node-h100-002-205-033,test-node-h100-002-207-033,test-node-h100-002-211-033,test-node-h100-002-219-033,test-node-h100-002-221-033,test-node-h100-002-244-012,test-node-h100-002-250-012,test-node-h100-002-251-012,test-node-h100-002-253-012,test-node-h100-002-253-026,test-node-h100-002-254-012,test-node-h100-002-254-026,test-node-h100-002-264-012,test-node-h100-002-277-012,test-node-h100-002-284-012,test-node-h100-002-288-026,test-node-h100-002-289-026,test-node-h100-002-295-026,test-node-h100-002-297-026,test-node-h100-002-307-012,test-node-h100-002-313-012,test-node-h100-002-338-012,test-node-h100-002-339-026,test-node-h100-002-354-012,test-node-h100-002-362-012,test-node-h100-002-364-026,test-node-h100-002-368-026,test-node-h100-002-374-012,test-node-h100-002-374-026,test-node-h100-003-087-026,test-node-h100-003-088-012,test-node-h100-003-098-026,test-node-h100-003-109-012,test-node-h100-003-109-026,test-node-h100-003-126-012,test-node-h100-003-127-026,test-node-h100-003-129-026,test-node-h100-003-130-012,test-node-h100-003-131-026,test-node-h100-003-135-012,test-node-h100-003-139-012,test-node-h100-003-180-012,test-node-h100-003-268-026,test-node-h100-003-284-026,test-node-h100-003-299-012,test-node-h100-003-302-026,test-node-h100-003-303-012,test-node-h100-003-310-026,test-node-h100-003-323-012,test-node-h100-003-325-026,test-node-h100-003-326-012,test-node-h100-003-334-012,test-node-h100-003-348-012,test-node-h100-003-349-012,test-node-h100-003-369-012,test-node-h100-004-005-023,test-node-h100-004-007-023,test-node-h100-004-010-013,test-node-h100-004-011-013,test-node-h100-004-014-003,test-node-h100-004-039-012,test-node-h100-004-056-012,test-node-h100-004-059-012,test-node-h100-004-066-026,test-node-h100-004-080-012,test-node-h100-004-082-026,test-node-h100-004-084-026,test-node-h100-004-105-026,test-node-h100-004-141-012,test-node-h100-004-208-003,test-node-h100-004-209-023,test-node-h100-004-215-033,test-node-h100-004-259-026,test-node-h100-004-268-012,test-node-h100-004-300-012,test-node-h100-004-311-026,test-node-h100-004-315-012,test-node-h100-004-321-012,test-node-h100-004-325-012,test-node-h100-004-326-012,test-node-h100-004-327-012,test-node-h100-004-328-026,test-node-h100-004-330-012,test-node-h100-004-341-026,test-node-h100-004-342-012,test-node-h100-004-342-026,test-node-h100-004-349-026,test-node-h100-004-356-012,test-node-h100-004-356-026,test-node-h100-004-366-026".split(
                ","
            ),
            status="CANCELED",
            username="test-agent",
            name="test_job",
            framework="megatron",
        ),
        metadata=AttributorMetadata(
            cluster="test-cluster-a",
            agent="test",
            tenant="testing",
            ruleset="default",
        ),
    )


@pytest.fixture
def raw_loki_obs_applog(attributor_info) -> ObservationV1:
    logroot = Path(__file__).parent / "testdata"
    path = logroot / "applog.json"
    payload = path.read_bytes()

    # Create observation context
    time_interval = TimeInterval(
        start=attributor_info.workload.job_start_time, end=attributor_info.workload.job_end_time
    )
    context = ObservationContextV1(
        time_interval=time_interval,
        source=ObservationSource.APPLOG,
        format=ObservationFormat.RAW_LOKI_STREAMS,
    )

    return ObservationV1(context=context, body=payload)


def test_raw_loki_obs_applog(client, attributor_info, raw_loki_obs_applog):
    response = client.post("/v1/attributor", content=attributor_info.model_dump_json())
    assert response.status_code == 201
    att_id = response.json()["attributor_id"]

    response = client.post(f"/v1/attributor/{att_id}/observation", content=raw_loki_obs_applog.model_dump_json())
    assert response.status_code == 201
    obs_id = response.json()["observation_id"]

    response = client.get(f"/v1/attributor/{att_id}/attribution")
    assert response.status_code == 200
    attribution = AttributionResponseV1.model_validate(response.json())
    assert set([obs_id]) == set(attribution.observation_context)
    found = False
    for latt in attribution.attributions:
        match latt:
            case GlobalAttribution():
                assert any(("No such file or directory" in att.symptom for att in latt.attributions))
                found = True
                break
            case _:
                pass
    assert found


@pytest.fixture
def raw_loki_obs(attributor_info) -> ObservationV1:
    logroot = Path(__file__).parent / "testdata"
    path = logroot / "logbody-body.json"
    payload = path.read_bytes()

    # Create observation context
    time_interval = TimeInterval(
        start=attributor_info.workload.job_start_time, end=attributor_info.workload.job_end_time
    )
    context = ObservationContextV1(
        time_interval=time_interval,
        source=ObservationSource.SYSLOG,
        format=ObservationFormat.RAW_LOKI_STREAMS,
    )

    return ObservationV1(context=context, body=payload)


def test_raw_loki_obs(client, attributor_info, raw_loki_obs):
    response = client.post("/v1/attributor", content=attributor_info.model_dump_json())
    assert response.status_code == 201
    att_id = response.json()["attributor_id"]

    response = client.post(f"/v1/attributor/{att_id}/observation", content=raw_loki_obs.model_dump_json())
    assert response.status_code == 201
    obs_id = response.json()["observation_id"]

    response = client.get(f"/v1/attributor/{att_id}/attribution")
    assert response.status_code == 200
    attribution = AttributionResponseV1.model_validate(response.json())
    assert set([obs_id]) == set(attribution.observation_context)
    found = False
    for latt in attribution.attributions:
        match latt:
            case NodeAttribution():
                if latt.node == "test-node-h100-003-180-012":
                    assert any(("XID-95-UNCONTAINED_ERROR" in att.symptom for att in latt.attributions))
                    found = True
                    break
            case _:
                pass
    assert found


@pytest.fixture
def prometheus_metrics_obs(attributor_info):
    logroot = Path(__file__).parent / "testdata"
    path = logroot / "prometheus_metrics.json"
    payload = path.read_bytes()
    time_interval = TimeInterval(
        start=attributor_info.workload.job_start_time, end=attributor_info.workload.job_end_time
    )
    context = ObservationContextV1(
        time_interval=time_interval,
        source=ObservationSource.NODE_METRICS,
        format=ObservationFormat.PROMETHEUS_METRICS,
    )
    return ObservationV1(context=context, body=payload)


def test_prometheus_metrics_observation_ingests_successfully(client, attributor_info, prometheus_metrics_obs):
    response = client.post("/v1/attributor", content=attributor_info.model_dump_json())
    assert response.status_code == 201
    att_id = response.json()["attributor_id"]

    response = client.post(f"/v1/attributor/{att_id}/observation", content=prometheus_metrics_obs.model_dump_json())
    assert response.status_code == 201
    obs_id = response.json()["observation_id"]

    response = client.get(f"/v1/attributor/{att_id}/attribution")
    assert response.status_code == 200
    attribution = AttributionResponseV1.model_validate(response.json())
    assert set([obs_id]) == set(attribution.observation_context)


def _string_list_context(**overrides) -> ObservationContextV1:
    """Build a string-list observation context for tests.

    Args:
        **overrides: Observation context fields that replace the defaults.

    Returns:
        An observation context with the requested overrides.
    """
    now = datetime.now().astimezone()
    fields = dict(
        time_interval=TimeInterval(start=now, end=now),
        source=ObservationSource.JOB_KEYWORDS,
        format=ObservationFormat.STRING_LIST,
    )
    fields.update(overrides)
    return ObservationContextV1(**fields)


def test_context_with_is_prefiltered_true_marks_lib_context_pre_filtered():
    """Confirm that is_prefiltered maps to the library context."""
    context = _string_list_context(is_prefiltered=True)

    lib_context = context.to_lib_context(ZoneInfo("UTC"))

    assert lib_context.pre_filtered is True


def test_context_without_is_prefiltered_leaves_lib_context_not_pre_filtered():
    """Confirm that omitting is_prefiltered preserves the false default."""
    context = _string_list_context()

    lib_context = context.to_lib_context(ZoneInfo("UTC"))

    assert lib_context.pre_filtered is False


def test_prefiltered_flag_flows_through_to_lib_observation():
    """Confirm that ObservationV1 carries is_prefiltered to the library."""
    context = _string_list_context(is_prefiltered=True)
    observation = ObservationV1(context=context, body="first\nsecond")

    lib_observation = observation.to_lib_observation(make_up_attributor_info().to_lib_jobinfo())

    assert lib_observation.context.pre_filtered is True


def test_request_body_with_is_prefiltered_true_reaches_lib_context():
    """Confirm the flag survives JSON request parsing and conversion."""
    now = datetime.now().astimezone()
    body = json.dumps(
        {
            "context": {
                "time_interval": {"start": now.isoformat(), "end": now.isoformat()},
                "source": ObservationSource.JOB_KEYWORDS.value,
                "format": ObservationFormat.STRING_LIST.value,
                "is_prefiltered": True,
            },
            "body": "first\nsecond",
        }
    )

    observation = ObservationV1.model_validate_json(body)

    assert observation.to_lib_observation(make_up_attributor_info().to_lib_jobinfo()).context.pre_filtered is True


def test_a_well_formed_body_decodes_to_the_entries_the_library_receives():
    entries = [
        {"ts": 1744734437830498000, "resources": [{"kind": "node", "hostname": JOB_NODE}], "body": "first"},
        {"body": "only a body"},
        {"body": "third", "resources": "AllJobResources"},
        {"body": "explicit nulls", "ts": None, "resources": None},
        {"body": "empty resources", "resources": []},
        {
            "body": "node beside a replica",
            "resources": [{"kind": "node", "hostname": JOB_NODE}, {"kind": "replica", "name": "worker0"}],
        },
    ]
    observation = ObservationV1(
        context=ObservationContextV1(
            time_interval=TimeInterval(start=None, end=None),
            source=ObservationSource.SYSLOG,
            format=ObservationFormat.ENTRY_TIMESERIES,
        ),
        body=json.dumps(entries),
    )

    lines = observation.to_lib_observation(make_up_attributor_info().to_lib_jobinfo()).lines

    assert lines[0] == {
        "body": "first",
        "ts": 1744734437830498000,
        "resources": [{"kind": "node", "key": JOB_NODE, "hostname": JOB_NODE}],
    }
    # Absent optional keys decode as nulls rather than going missing.
    assert lines[1] == {"body": "only a body", "ts": None, "resources": None}
    assert lines[2]["resources"] == "AllJobResources"
    # An explicit null is indistinguishable from an absent key once decoded.
    assert lines[3] == lines[1] | {"body": "explicit nulls"}
    # An empty array stays empty; it does not collapse to null or to AllJobResources.
    assert lines[4]["resources"] == []
    # A replica rides along beside the node, each keyed by its own identifying field.
    assert lines[5]["resources"] == [
        {"kind": "node", "key": JOB_NODE, "hostname": JOB_NODE},
        {"kind": "replica", "key": "worker0", "name": "worker0"},
    ]


@pytest.mark.parametrize("ts,decoded", [("123", 123), (2.0, 2), (True, 1)])
def test_a_ts_the_validator_coerces_decodes_to_an_int(ts, decoded):
    observation = ObservationV1(
        context=ObservationContextV1(
            time_interval=TimeInterval(start=None, end=None),
            source=ObservationSource.SYSLOG,
            format=ObservationFormat.ENTRY_TIMESERIES,
        ),
        body=json.dumps([{"body": "x", "ts": ts}]),
    )

    (line,) = observation.to_lib_observation(make_up_attributor_info().to_lib_jobinfo()).lines

    assert line == {"body": "x", "ts": decoded, "resources": None}
    assert type(line["ts"]) is int


@pytest.mark.parametrize(
    "body,reason",
    [
        ("not json at all", "unparseable"),
        ('[{"body": "x", "resources": [{"kind": "planet", "name": "mars"}]}]', "unknown resource kind"),
        ('[{"body": "x", "timestamp": 1744734437830498000}]', "misspelled ts would be silently dropped"),
        ('[{"body": "x", "ts": 1744734437830498000000}]', "timestamp out of datetime range"),
        ('[{"body": "x", "ts": -1744734437830498000000}]', "timestamp below datetime range"),
        ('[{"body": "x", "ts": 2.5}]', "fractional timestamp"),
        ('[{"ts": 1744734437830498000}]', "missing body"),
        ('[{"body": 5}]', "non-string body"),
        ('{"body": "x"}', "an entry outside a list"),
    ],
)
def test_a_malformed_body_is_rejected_with_a_client_error(client, body, reason):
    """A malformed body fails here, with the other structured-body failures, not deep inside extraction."""
    att_id = post_attributor(client, make_up_attributor_info("v2"))

    response = client.post(
        f"/v2/attributor/{att_id}/observation",
        content=json.dumps(
            {
                "context": {"time_interval": UNBOUNDED, "source": ObservationSource.SYSLOG.value},
                "body": body,
            }
        ),
    )

    assert response.status_code == 400, reason
    assert response.json()["model_type"] == "InvalidObservationBodyError", reason
