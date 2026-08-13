# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import http
import json
from datetime import datetime
from unittest.mock import patch

import pytest
from attribution_lib.exceptions import UnprocessedObservationError

from app.exceptions import (
    DuplicateAttributorError,
    DuplicateObservationError,
    ObservationIngestionErrorResponse,
    ResponseModel,
    UnknownAttributorIdError,
)
from app.observation import (
    Observation,
    ObservationContext,
    ObservationFormat,
    ObservationSource,
)
from app.util import TimeInterval
from tests.conftest import (
    make_up_attributor_info,
    post_attributor,
)


def fake_applog_observation(body: str) -> Observation:
    start = datetime(2025, 3, 14, 9, 10, 00).astimezone()
    end = datetime(2025, 3, 14, 9, 11, 00).astimezone()
    context = ObservationContext(
        time_interval=TimeInterval(start=start, end=end),
        resources="AllJobResources",
        source=ObservationSource.APPLOG,
        format=ObservationFormat.RAW_LOG_LINES,
    )
    return Observation(context=context, body=body)


def fake_syslog_observation(body: str) -> Observation:
    start = datetime(2025, 3, 14, 9, 10, 00).astimezone()
    end = datetime(2025, 3, 14, 9, 11, 00).astimezone()
    context = ObservationContext(
        time_interval=TimeInterval(start=start, end=end),
        resources="AllJobResources",
        source=ObservationSource.SYSLOG,
        format=ObservationFormat.RAW_LOKI_STREAMS,
    )
    return Observation(context=context, body=body)


def test_observation_ingestion_errors_are_declared_in_openapi(app):
    schema = app.openapi()
    expected_ref = {"$ref": "#/components/schemas/ObservationIngestionErrorResponse"}
    expected_examples = {
        "UnknownObservationFormatError": {
            "summary": "Unsupported recognized observation format",
            "value": {
                "model_type": "UnknownObservationFormatError",
                "format": "future_format",
            },
        },
        "InvalidObservationBodyError": {
            "summary": "Malformed structured observation body",
            "value": {
                "model_type": "InvalidObservationBodyError",
                "format": "raw_loki_streams",
            },
        },
        "IncompatibleObservationSourceFormatError": {
            "summary": "Incompatible observation source and format",
            "value": {
                "model_type": "IncompatibleObservationSourceFormatError",
                "source": "applog",
                "format": "string_list",
            },
        },
    }

    # v1 error-envelope wiring is intentionally out of scope for this change.
    for version in ("v2", "latest"):
        response_schema = schema["paths"][f"/{version}/attributor/{{att_id}}/observation"]["post"]["responses"]["400"]
        response_content = response_schema["content"]["application/json"]
        assert response_content["schema"] == expected_ref
        assert response_content["examples"] == expected_examples
        for example in response_content["examples"].values():
            ObservationIngestionErrorResponse.model_validate(example["value"])

    error_schema = schema["components"]["schemas"]["ObservationIngestionErrorResponse"]
    assert {variant["$ref"] for variant in error_schema["oneOf"]} == {
        "#/components/schemas/IncompatibleObservationSourceFormatError",
        "#/components/schemas/InvalidObservationBodyError",
        "#/components/schemas/UnknownObservationFormatError",
    }
    assert error_schema["discriminator"]["propertyName"] == "model_type"


class TestEndpoints:
    """Endpoint behavior shared across API versions. Tests are parameterized by
    `version` and run once per supported API version."""

    def test_post_attributor_success(self, client, version):
        info = make_up_attributor_info(version)
        response = client.post(f"/{version}/attributor", content=info.model_dump_json())
        assert response.status_code == 201

        # different metadata gets a different ID
        att_id = response.json()["attributor_id"]
        info.metadata.cluster = "test-cluster-b"
        response = client.post(f"/{version}/attributor", content=info.model_dump_json())
        assert response.status_code == 201
        assert response.json()["attributor_id"] != att_id

    def test_post_attributor_duplicate_rejected(self, client, version):
        info = make_up_attributor_info(version)
        response = client.post(f"/{version}/attributor", content=info.model_dump_json())
        assert response.status_code == 201
        att_id = ResponseModel.from_dict(response.json()).attributor_id

        response = client.post(f"/{version}/attributor", content=info.model_dump_json())
        assert response.status_code == DuplicateAttributorError.code
        assert ResponseModel.from_dict(response.json()) == DuplicateAttributorError.new(att_id)

    def test_post_attributor_validation_error_response(self, client, version):
        # Test with invalid data that should trigger validation errors
        invalid_data = {
            "workload": {
                "id": "120123",
                "srun_cmd": "",
                "type": "INVALID_TYPE",  # Invalid type
                "nranks": -1,  # Invalid negative number
                "ranks_per_node": 0,  # Invalid zero value
                "start_time": "2023-01-01T00:00:00-07:00",
                "end_time": "2022-12-31T23:59:59-07:00",  # End time before start time
                "exit_code_signal": 1,
                "nodes": [],
                "status": "CANCELED",
                "username": "test-agent",
                "name": "test_job",
                "framework": "megatron",
            },
            "metadata": {
                "cluster": "test-cluster-a",
                "agent": "test",
                "tenant": "testing",
                "ruleset": "default",
            },
        }

        response = client.post(f"/{version}/attributor", json=invalid_data)
        assert response.status_code == 422

        # Verify the response structure
        response_data = response.json()
        assert "errors" in response_data
        assert "body" in response_data
        assert "message" in response_data
        assert response_data["message"] == "Validation failed. Please check the errors."

        # Verify we have multiple validation errors
        assert len(response_data["errors"]) > 1

        # Check that each error has the required fields
        for error in response_data["errors"]:
            assert "loc" in error
            assert "msg" in error
            assert "type" in error

    def test_get_attribution_unknown_id_rejected(self, client, version):
        badid = "fakefake"
        response = client.get(f"/{version}/attributor/{badid}/attribution")
        assert response.status_code == UnknownAttributorIdError.code
        assert ResponseModel.from_dict(response.json()) == UnknownAttributorIdError.new(badid)

    def test_get_attribution_unhandled_exception(self, client, version):
        with patch("attribution_lib.attributor.Attributor.attribute") as mock_attribute:
            mock_attribute.side_effect = UnprocessedObservationError()
            att_id = post_attributor(client, make_up_attributor_info(version))
            response = client.get(f"/{version}/attributor/{att_id}/attribution")
            assert response.status_code == 500

    def test_post_observation_success(self, client, version):
        att_id = post_attributor(client, make_up_attributor_info(version))
        obs = fake_applog_observation("Example\nLines")

        response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs.model_dump_json())

        assert response.status_code == 201

    def test_post_observation_duplicate_rejected(self, client, version):
        att_id = post_attributor(client, make_up_attributor_info(version))
        obs = fake_applog_observation("Example\nLines")

        response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs.model_dump_json())
        assert response.status_code == 201
        obs_id = response.json()["observation_id"]

        response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs.model_dump_json())
        assert response.status_code == DuplicateObservationError.code
        assert ResponseModel.from_dict(response.json()) == DuplicateObservationError.new(obs_id)

    def test_post_observation_invalid_format(self, client, version):
        att_id = post_attributor(client, make_up_attributor_info(version))
        obs = fake_applog_observation("Example\nLines")
        obs.context.format = "fooey"

        response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs.model_dump_json())

        assert response.status_code == 422

    @pytest.mark.parametrize(
        ("source", "format"),
        [
            (ObservationSource.UFMLOG, ObservationFormat.UFMLOG),
            (ObservationSource.NODE_METRICS, ObservationFormat.PROMETHEUS_METRICS),
            (ObservationSource.CSP_EVENT, ObservationFormat.CSP_EVENT_LIST),
        ],
    )
    def test_post_observation_supported_structured_format(self, client, version, source, format):
        att_id = post_attributor(client, make_up_attributor_info(version))
        obs = fake_applog_observation("[]")
        obs.context.source = source
        obs.context.format = format

        response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs.model_dump_json())

        assert response.status_code == 201

    def test_post_observation_incompatible_source_format_returns_400(self, client, version):
        att_id = post_attributor(client, make_up_attributor_info(version))
        obs = fake_applog_observation("Example\nLines")
        obs.context.format = ObservationFormat.STRING_LIST

        response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs.model_dump_json())

        assert response.status_code == 400
        assert response.json() == {
            "model_type": "IncompatibleObservationSourceFormatError",
            "source": ObservationSource.APPLOG.value,
            "format": ObservationFormat.STRING_LIST.value,
        }

    @pytest.mark.parametrize(
        ("source", "format"),
        [
            (ObservationSource.SYSLOG, ObservationFormat.LOKI_DICT_LIST),
            (ObservationSource.APPLOG, ObservationFormat.RAW_LOKI_STREAMS),
            (ObservationSource.UFMLOG, ObservationFormat.UFMLOG),
            (ObservationSource.NODE_METRICS, ObservationFormat.PROMETHEUS_METRICS),
            (ObservationSource.CSP_EVENT, ObservationFormat.CSP_EVENT_LIST),
        ],
    )
    def test_post_observation_malformed_structured_body_returns_400(self, client, version, source, format):
        att_id = post_attributor(client, make_up_attributor_info(version))
        obs = fake_applog_observation("not-json")
        obs.context.source = source
        obs.context.format = format

        response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs.model_dump_json())

        assert response.status_code == 400
        assert response.json() == {
            "model_type": "InvalidObservationBodyError",
            "format": format.value,
        }

    def test_delete_attributor_success(self, client, version):
        att_id = post_attributor(client, make_up_attributor_info(version))

        response = client.delete(f"/{version}/attributor/{att_id}")
        assert response.status_code == 204

    def test_delete_attributor_unknown_id_rejected(self, client, version):
        att_id = "fakefake"
        response = client.delete(f"/{version}/attributor/{att_id}")
        assert response.status_code == UnknownAttributorIdError.code
        assert ResponseModel.from_dict(response.json()) == UnknownAttributorIdError.new(att_id)

    def test_healthz_ok(self, client, version):
        response = client.get(f"/{version}/healthz")
        assert response.status_code == 200


class TestUserJourneys:
    """Test known user journeys.

    These include, but are not limited to:
    - Single and multi-source attribution
    - Freely interleaved observation ingestion and attribution calls
    """

    def test_single_source_attribution(self, client, version):
        """Create -> Ingest -> Attribute -> Delete"""
        # Create attributor
        att_id = post_attributor(client, make_up_attributor_info(version))

        # Ingest observation
        obs = fake_applog_observation("applog line 1\napplog line 2")
        obs_response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs.model_dump_json())
        assert obs_response.status_code == 201
        obs_id = obs_response.json()["observation_id"]

        # Attribute
        att_response = client.get(f"/{version}/attributor/{att_id}/attribution")
        assert att_response.status_code == 200
        assert set(att_response.json()["observation_context"]) == {obs_id}

        # Delete attributor
        delete_response = client.delete(f"/{version}/attributor/{att_id}")
        assert delete_response.status_code == 204

    def test_multi_source_ingest_attribution(self, client, version):
        """Create -> Ingest syslog -> Ingest applog -> Attribute"""
        # Create attributor
        att_id = post_attributor(client, make_up_attributor_info(version))

        # Ingest syslog
        syslog_obs = fake_syslog_observation("[]")
        obs_response = client.post(f"/{version}/attributor/{att_id}/observation", content=syslog_obs.model_dump_json())
        assert obs_response.status_code == 201
        syslog_obs_id = obs_response.json()["observation_id"]

        # Ingest applog
        applog_obs = fake_applog_observation("applog line 1\napplog line 2")
        obs_response = client.post(f"/{version}/attributor/{att_id}/observation", content=applog_obs.model_dump_json())
        assert obs_response.status_code == 201
        applog_obs_id = obs_response.json()["observation_id"]

        # Attribute: both observations should be referenced
        att_response = client.get(f"/{version}/attributor/{att_id}/attribution")
        assert att_response.status_code == 200
        assert set(att_response.json()["observation_context"]) == {syslog_obs_id, applog_obs_id}

    def test_multi_source_multi_ingest_attribution(self, client, version):
        """Create -> Ingest 1st applog -> Ingest syslog -> Ingest 2nd applog -> Attribute"""
        att_id = post_attributor(client, make_up_attributor_info(version))

        # Ingest 1st applog
        applog_obs_1 = fake_applog_observation("applog line 1\napplog line 2")
        response = client.post(f"/{version}/attributor/{att_id}/observation", content=applog_obs_1.model_dump_json())
        assert response.status_code == 201
        applog_obs_id_1 = response.json()["observation_id"]

        # Ingest syslog
        syslog_obs = fake_syslog_observation("[]")
        response = client.post(f"/{version}/attributor/{att_id}/observation", content=syslog_obs.model_dump_json())
        assert response.status_code == 201
        syslog_obs_id = response.json()["observation_id"]

        # Ingest 2nd applog with distinct content
        applog_obs_2 = fake_applog_observation("applog line 3\napplog line 4")
        response = client.post(f"/{version}/attributor/{att_id}/observation", content=applog_obs_2.model_dump_json())
        assert response.status_code == 201
        applog_obs_id_2 = response.json()["observation_id"]

        # Attribute: all three observations should be referenced
        att_response = client.get(f"/{version}/attributor/{att_id}/attribution")
        assert att_response.status_code == 200
        assert set(att_response.json()["observation_context"]) == {applog_obs_id_1, syslog_obs_id, applog_obs_id_2}

    def test_multi_phase_attribution(self, client, version):
        """Ingest -> Attribute -> Ingest -> Attribute -> Attribute (idempotent)"""
        att_id = post_attributor(client, make_up_attributor_info(version))

        # Phase 1: ingest + attribute
        obs1 = fake_applog_observation("phase 1 log data")
        response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs1.model_dump_json())
        assert response.status_code == 201
        obs_id_1 = response.json()["observation_id"]

        att_resp_1 = client.get(f"/{version}/attributor/{att_id}/attribution").json()
        assert set(att_resp_1["observation_context"]) == {obs_id_1}

        # Phase 2: ingest more + attribute again
        obs2 = fake_syslog_observation("[]")
        response = client.post(f"/{version}/attributor/{att_id}/observation", content=obs2.model_dump_json())
        assert response.status_code == 201
        obs_id_2 = response.json()["observation_id"]

        att_resp_2 = client.get(f"/{version}/attributor/{att_id}/attribution").json()
        assert set(att_resp_2["observation_context"]) == {obs_id_1, obs_id_2}

        # Phase 3: attribute again without new ingest - full body is stable (idempotent)
        att_resp_3 = client.get(f"/{version}/attributor/{att_id}/attribution").json()
        assert att_resp_3 == att_resp_2


class TestV2WorkloadValidation:
    """V2 workload payloads that fail validation surface as 422 (not 500), whether the failure is
    wire-level (a missing or invalid field) or lib-level (JobInfoError via its exception handler)."""

    def _assert_validation_error_shape(self, response):
        assert response.status_code == 422
        body = response.json()
        assert "errors" in body
        assert "body" in body
        assert "message" in body

    def test_missing_timezone_returns_422(self, client):
        """V2 requires an explicit IANA timezone; a payload without one is rejected at the wire
        level. V1 has no such field and infers the zone from job_start_time instead."""
        info_dict = json.loads(make_up_attributor_info("v2").model_dump_json())
        del info_dict["workload"]["timezone"]

        response = client.post("/v2/attributor", content=json.dumps(info_dict))
        self._assert_validation_error_shape(response)

    def test_empty_nodes_dict_is_accepted_as_never_ran(self, client):
        """Mirroring the lib, empty `nodes` mapping is a valid workload representing a job that
        never received an allocation."""
        info = make_up_attributor_info("v2", nodes={})
        response = client.post("/v2/attributor", content=info.model_dump_json())
        assert response.status_code == http.HTTPStatus.CREATED

    def test_ragged_node_to_ranks_returns_422(self, client):
        info = make_up_attributor_info("v2", nodes={"node0": [0, 1], "node1": [2]})
        response = client.post("/v2/attributor", content=info.model_dump_json())
        self._assert_validation_error_shape(response)


class TestV1NodeToRanksOverride:
    """V1 accepts an optional explicit node_to_ranks mapping in place of the positional
    nodes/nranks/ranks_per_node trio. If both are sent, the explicit mapping wins (the lib has a
    single node_to_ranks input, so there is no conflict to surface)."""

    _MAPPING = {f"node{s}": list(range(s * 8, (s + 1) * 8)) for s in range(10)}

    def test_explicit_node_to_ranks_only_succeeds(self, client):
        info = make_up_attributor_info(
            "v1",
            nodes=None,
            nranks=None,
            ranks_per_node=None,
            node_to_ranks=self._MAPPING,
        )

        response = client.post("/v1/attributor", content=info.model_dump_json())

        assert response.status_code == http.HTTPStatus.CREATED, response.text

    def test_positional_plus_node_to_ranks_succeeds(self, client):
        """Both shapes supplied no longer 422s; the explicit mapping is used."""
        info = make_up_attributor_info("v1", node_to_ranks=self._MAPPING)

        response = client.post("/v1/attributor", content=info.model_dump_json())

        assert response.status_code == http.HTTPStatus.CREATED, response.text

    def test_positional_plus_node_to_ranks_uses_explicit_mapping(self):
        """Distinct node names make the winner observable: to_lib_jobinfo ignores the positional
        nodes and builds the JobInfo from the explicit mapping."""
        info = make_up_attributor_info(
            "v1",
            nodes=["positional-only"],
            nranks=2,
            ranks_per_node=2,
            node_to_ranks={"explicit-a": [0, 1], "explicit-b": [2, 3]},
        )

        job_info = info.to_lib_jobinfo()

        assert job_info.nodes == ["explicit-a", "explicit-b"]

    def test_nodes_without_ranks_per_node_kept_as_empty_members(self):
        """nodes set but ranks_per_node absent: the node list is preserved as non-participating
        members (empty rank lists), not discarded so the job looks like it never ran."""
        info = make_up_attributor_info(
            "v1",
            nodes=["lonely-a", "lonely-b"],
            nranks=None,
            ranks_per_node=None,
            node_to_ranks=None,
        )

        job_info = info.to_lib_jobinfo()

        assert job_info.nodes == ["lonely-a", "lonely-b"]
        assert job_info.RANKS_PER_NODE == 0.0
        assert job_info.node_from_rank == {}


class TestV2AttributionShape:
    """Wire-level check that /v2/attribution returns the V2 response model (not V1)."""

    def test_attribution_returns_v2_shape(self, client):
        att_id = post_attributor(client, make_up_attributor_info("v2"))

        body = client.get(f"/v2/attributor/{att_id}/attribution").json()

        assert "verdict" in body
        assert "attributions" not in body

    def test_decision_graph_is_disabled_by_default(self, client):
        att_id = post_attributor(client, make_up_attributor_info("v2"))

        body = client.get(f"/v2/attributor/{att_id}/attribution").json()

        assert "decision_graph" not in body["verdict"]

    def test_decision_graph_is_included_when_requested(self, client):
        att_id = post_attributor(client, make_up_attributor_info("v2"))

        response = client.get(
            f"/v2/attributor/{att_id}/attribution",
            params={"include_decision_graph": True},
        )

        assert response.status_code == http.HTTPStatus.OK
        graph = response.json()["verdict"]["decision_graph"]
        assert len(graph["nodes"]) > 0
        assert any(node["node_type"] == "OUTCOME" for node in graph["nodes"])
        assert isinstance(graph["edges"], list)

    @pytest.mark.parametrize(
        "target",
        ["app.attributor.build_decision_graph", "app.v2.models.DecisionGraph.from_lib"],
    )
    @pytest.mark.parametrize(
        "error",
        [AttributeError, TypeError, KeyError, IndexError, ValueError],
    )
    def test_broken_decision_graph_does_not_fail_the_verdict(self, client, target, error):
        """The graph is a diagnostic add-on, so losing it must not cost the caller its verdict."""
        att_id = post_attributor(client, make_up_attributor_info("v2"))

        with patch(target, side_effect=error("node_symptoms_base")):
            response = client.get(
                f"/v2/attributor/{att_id}/attribution",
                params={"include_decision_graph": True},
            )

        assert response.status_code == http.HTTPStatus.OK
        body = response.json()
        # Requested but unavailable, which is reported as null rather than an absent field.
        assert body["verdict"]["decision_graph"] is None
        assert body["verdict"]["attribution_outcome"]

    def test_failed_recording_does_not_fail_the_attribution(self, client):
        """The lib records decision nodes mid-attribution without guarding those calls."""
        att_id = post_attributor(client, make_up_attributor_info("v2"))

        with patch(
            "attribution_lib.decision_trace_recorder.DecisionRecorder.record",
            side_effect=TypeError("unexpected keyword argument"),
        ):
            response = client.get(
                f"/v2/attributor/{att_id}/attribution",
                params={"include_decision_graph": True},
            )

        assert response.status_code == http.HTTPStatus.OK
        body = response.json()
        assert body["verdict"]["decision_graph"] is None
        assert body["verdict"]["attribution_outcome"]
