# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from datetime import datetime, timedelta
from http import HTTPStatus

import pytest
from attribution_lib import get_rules_engine
from attribution_lib.rules_engine import CollectorRule, RuleLocation, RuleParamsBase

from app.exceptions import AttributionHTTPException
from tests.conftest import make_up_attributor_info


class TestAttributorExpiration:
    def test_attributor_expired_after_configured_period(self, app):
        """Test that attributors are properly expired after the configured time period."""
        attributor_info = make_up_attributor_info(
            id="test-workload",
            status="COMPLETED",
            exit_code_signal=0,
            nranks=1,
            ranks_per_node=1,
            nodes=["node1"],
        )

        # Add the attributor
        attributor = app.state.aim.add_attributor(attributor_info)

        # Verify it exists
        assert attributor.id in app.state.aim.attributors

        # Simulate time passing by setting the last access time to be older than the expiration period
        attributor.last_access_time = datetime.now() - timedelta(hours=25)

        # Run cleanup
        app.state.aim.cleanup_expired_attributors()

        # Verify the attributor was removed
        assert attributor.id not in app.state.aim.attributors

        # Verify we can't get the attributor anymore
        with pytest.raises(AttributionHTTPException) as e:
            app.state.aim.get_attributor(attributor.id)
        assert e.value.status_code == 404

    def test_attributor_not_expired_before_configured_period(self, app):
        """Test that attributors are not expired before the configured time period."""
        attributor_info = make_up_attributor_info(
            id="test-workload",
            status="COMPLETED",
            exit_code_signal=0,
            nranks=1,
            ranks_per_node=1,
            nodes=["node1"],
        )

        # Add the attributor
        attributor = app.state.aim.add_attributor(attributor_info)

        # Verify it exists
        assert attributor.id in app.state.aim.attributors

        # Simulate time passing by setting the last access time to be just under the expiration period
        attributor.last_access_time = datetime.now() - timedelta(hours=23)

        # Run cleanup
        app.state.aim.cleanup_expired_attributors()

        # Verify the attributor still exists
        assert attributor.id in app.state.aim.attributors

        # Verify we can still get the attributor
        retrieved_attributor = app.state.aim.get_attributor(attributor.id)
        assert retrieved_attributor.id == attributor.id


class _PassthroughParams(RuleParamsBase):
    value: int = 0


@pytest.fixture
def passthrough_collector_rule():
    """Register a rule-agnostic test collector with a params_model and tear down after."""
    name = "Test.ParamsPassthrough"
    rule = CollectorRule(
        name=name,
        author="test",
        location=RuleLocation("test_attributor_manager.py", 0),
        processor=lambda _es, _ji, _params: {},
        formatter=lambda raw: raw,
        params_model=_PassthroughParams,
    )
    engine = get_rules_engine()
    engine.collector_rules[name] = rule
    try:
        yield name
    finally:
        engine.collector_rules.pop(name, None)


@pytest.fixture
def no_params_collector_rule():
    """Register a collector with no params_model and tear down after."""
    name = "Test.NoParams"
    rule = CollectorRule(
        name=name,
        author="test",
        location=RuleLocation("test_attributor_manager.py", 0),
        processor=lambda _es, _ji, _params: {},
        formatter=lambda raw: raw,
        params_model=None,
    )
    engine = get_rules_engine()
    engine.collector_rules[name] = rule
    try:
        yield name
    finally:
        engine.collector_rules.pop(name, None)


class TestRuleParamsValidation:
    """Test that the rule_params validator accepts a known rule's schema and rejects unknown ids."""

    def test_valid_params_accepted(self, client, passthrough_collector_rule):
        body = json.loads(make_up_attributor_info().model_dump_json())
        body["parameters"] = {"rule_params": {passthrough_collector_rule: {"params": {"value": 42}}}}
        response = client.post("/v1/attributor", content=json.dumps(body))
        assert response.status_code == HTTPStatus.CREATED

    def test_unknown_rule_id_rejected(self, client):
        body = json.loads(make_up_attributor_info().model_dump_json())
        body["parameters"] = {"rule_params": {"Bogus.NotARealRule": {"params": {}}}}
        response = client.post("/v1/attributor", content=json.dumps(body))
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    def test_params_for_no_params_rule_rejected(self, client, no_params_collector_rule):
        body = json.loads(make_up_attributor_info().model_dump_json())
        body["parameters"] = {"rule_params": {no_params_collector_rule: {"params": {"foo": 1}}}}
        response = client.post("/v1/attributor", content=json.dumps(body))
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
