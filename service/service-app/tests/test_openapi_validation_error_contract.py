# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from http import HTTPStatus

import pytest
from attribution_lib import get_rules_engine
from attribution_lib.rules import CollectorRule, RuleLocation
from jsonschema import Draft202012Validator, ValidationError

from tests.conftest import make_up_attributor_info

_SCHEMA_NAME = "ValidationErrorResponse"
# v1 error-envelope wiring is intentionally out of scope for this change.
_AFFECTED_OPERATIONS = (
    ("/v2/attributor", "post"),
    ("/v2/attributor/{att_id}/observation", "post"),
    ("/v2/attributor/{att_id}/attribution", "get"),
    ("/v2/attributor/{att_id}", "delete"),
)


def _validation_error_schema(app) -> dict:
    return app.openapi()["components"]["schemas"][_SCHEMA_NAME]


def _assert_runtime_envelope(app, response) -> dict:
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    payload = response.json()
    assert set(payload) == {"errors", "body", "message"}
    Draft202012Validator(_validation_error_schema(app)).validate(payload)
    return payload


@pytest.fixture
def no_params_collector_rule(monkeypatch):
    name = "Test.ValidationEnvelopeNoParams"
    rule = CollectorRule(
        name=name,
        author="test",
        location=RuleLocation("test_openapi_validation_error_contract.py", 0),
        processor=lambda _es, _ji, _params: {},
        formatter=lambda raw: raw,
        params_model=None,
    )
    engine = get_rules_engine()
    get_rule = engine.get_rule
    monkeypatch.setattr(engine, "get_rule", lambda rule_id: rule if rule_id == name else get_rule(rule_id))
    yield name


def test_openapi_422_operations_use_runtime_envelope(app):
    schema = app.openapi()
    expected_ref = {"$ref": f"#/components/schemas/{_SCHEMA_NAME}"}

    for path, method in _AFFECTED_OPERATIONS:
        response_schema = schema["paths"][path][method]["responses"]["422"]["content"]["application/json"]["schema"]
        assert response_schema == expected_ref

    validation_schema = schema["components"]["schemas"][_SCHEMA_NAME]
    assert set(validation_schema["required"]) == {"errors", "body", "message"}
    assert "detail" not in validation_schema["properties"]


def test_validation_error_schema_accepts_runtime_variants(app):
    validator = Draft202012Validator(_validation_error_schema(app))
    variants = (
        {
            "errors": [{"loc": ["body", "workload"], "msg": "Field required", "type": "missing"}],
            "body": {"metadata": {}},
            "message": "Validation failed. Please check the errors.",
        },
        {
            "errors": "",
            "body": "JobInfoIncompleteEndTimeError()",
            "message": "Workload validation failed.",
        },
        {
            "errors": ["rule does not accept parameters"],
            "body": "RuleAcceptsNoParamsError()",
            "message": "Rule parameter validation failed.",
        },
    )

    for payload in variants:
        validator.validate(payload)


@pytest.mark.parametrize("invalid_errors", ({"unexpected": True}, 7, True, None))
def test_validation_error_schema_rejects_unknown_error_variants(app, invalid_errors):
    payload = {
        "errors": invalid_errors,
        "body": None,
        "message": "Validation failed.",
    }

    with pytest.raises(ValidationError):
        Draft202012Validator(_validation_error_schema(app)).validate(payload)


def test_request_validation_422_runtime_contract(app, client):
    response = client.post("/v2/attributor", json={"workload": {}})

    payload = _assert_runtime_envelope(app, response)
    assert isinstance(payload["errors"], list)
    assert payload["body"] == {"workload": {}}
    assert payload["message"] == "Validation failed. Please check the errors."


def test_job_validation_422_runtime_contract(app, client):
    info = make_up_attributor_info("v2", nodes={"node0": [0, 1], "node1": [2]})
    response = client.post("/v2/attributor", content=info.model_dump_json())

    payload = _assert_runtime_envelope(app, response)
    assert isinstance(payload["errors"], str)
    assert isinstance(payload["body"], str)
    assert payload["message"] == "Workload validation failed."


def test_rule_parameter_validation_422_runtime_contract(app, client, no_params_collector_rule):
    body = json.loads(make_up_attributor_info("v2").model_dump_json())
    body["parameters"] = {"rule_params": {no_params_collector_rule: {"params": {"unexpected": True}}}}
    response = client.post("/v2/attributor", json=body)

    payload = _assert_runtime_envelope(app, response)
    assert isinstance(payload["errors"], list)
    assert isinstance(payload["body"], str)
    assert payload["message"] == "Rule parameter validation failed."
