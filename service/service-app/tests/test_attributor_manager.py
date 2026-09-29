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
import threading
from datetime import datetime, timedelta
from http import HTTPStatus

import pytest
from attribution_lib import get_rules_engine
from attribution_lib.rules import CollectorRule, RuleLocation, RuleParameters, RuleParamsBase

from app.attributor import AttributionParameters
from app.exceptions import AttributionHTTPException, UnknownAttributorIdError
from tests.conftest import make_up_attributor_info
from tests.test_basic_functionality import fake_applog_observation


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


def _failing_close():
    raise RuntimeError("boom")


def _expire(*wrappers):
    for wrapper in wrappers:
        wrapper.last_access_time = datetime.now() - timedelta(hours=25)


class TestAttributorCleanupClosesEventManager:
    """Deleting or expiring an attributor closes its DuckDB connection at once,
    instead of leaving it to the garbage collector (EventStream <-> EventManager
    is a reference cycle)."""

    def test_delete_attributor_closes_event_manager(self, app):
        wrapper = app.state.aim.add_attributor(make_up_attributor_info(id="close-delete"))
        event_manager = wrapper.attributor.event_stream.mgr

        app.state.aim.delete_attributor(wrapper.id)

        assert event_manager._closed is True

    def test_cleanup_expired_attributors_closes_event_manager(self, app):
        wrapper = app.state.aim.add_attributor(make_up_attributor_info(id="close-cleanup"))
        event_manager = wrapper.attributor.event_stream.mgr
        _expire(wrapper)

        app.state.aim.cleanup_expired_attributors()

        assert event_manager._closed is True

    def test_delete_attributor_removes_entry_even_if_close_raises(self, app, monkeypatch):
        wrapper = app.state.aim.add_attributor(make_up_attributor_info(id="close-raises"))
        monkeypatch.setattr(wrapper.attributor.event_stream.mgr, "close", _failing_close)

        with pytest.raises(RuntimeError):
            app.state.aim.delete_attributor(wrapper.id)

        assert wrapper.id not in app.state.aim.attributors

    def test_cleanup_drops_a_failing_attributor_and_the_next_pass_removes_the_rest(self, app, monkeypatch):
        broken = app.state.aim.add_attributor(make_up_attributor_info(id="cleanup-broken"))
        ok = app.state.aim.add_attributor(make_up_attributor_info(id="cleanup-ok"))
        ok_event_manager = ok.attributor.event_stream.mgr
        monkeypatch.setattr(broken.attributor.event_stream.mgr, "close", _failing_close)
        _expire(broken, ok)

        with pytest.raises(RuntimeError):
            app.state.aim.cleanup_expired_attributors()
        assert broken.id not in app.state.aim.attributors

        app.state.aim.cleanup_expired_attributors()

        assert ok.id not in app.state.aim.attributors
        assert ok_event_manager._closed is True


@pytest.mark.parametrize("close_raises", [False, True])
@pytest.mark.parametrize(
    "call",
    [
        lambda wrapper: wrapper.ingest(fake_applog_observation("Example\nLines")),
        lambda wrapper: wrapper.attribute(),
    ],
    ids=["ingest", "attribute"],
)
def test_request_on_deleted_attributor_raises_unknown_attributor(app, monkeypatch, call, close_raises):
    """A request that fetched the wrapper before a delete gets the unknown-id
    error, not the closed connection, even when closing the connection failed."""
    wrapper = app.state.aim.add_attributor(make_up_attributor_info(id="closed-in-flight"))
    if close_raises:
        monkeypatch.setattr(wrapper.attributor.event_stream.mgr, "close", _failing_close)
        with pytest.raises(RuntimeError):
            app.state.aim.delete_attributor(wrapper.id)
    else:
        app.state.aim.delete_attributor(wrapper.id)

    with pytest.raises(AttributionHTTPException) as e:
        call(wrapper)

    assert e.value.status_code == UnknownAttributorIdError.code
    assert e.value.detail["att_id"] == wrapper.id


class _RefreshingLock:
    """Stand-in for AttributorWrapper.lock that bumps last_access_time on entry.

    Simulates a request ingesting/attributing on this attributor in the gap
    between cleanup_expired_attributors() selecting it as expired and actually
    taking its lock.
    """

    def __init__(self, attributor):
        self._attributor = attributor
        self._real = threading.Lock()

    def __enter__(self):
        self._attributor.last_access_time = datetime.now()
        return self._real.__enter__()

    def __exit__(self, *exc_info):
        return self._real.__exit__(*exc_info)


class TestCleanupRechecksExpiryUnderLock:
    def test_wrapper_touched_during_cleanup_survives(self, app):
        wrapper = app.state.aim.add_attributor(make_up_attributor_info(id="cleanup-race"))
        event_manager = wrapper.attributor.event_stream.mgr
        wrapper.last_access_time = datetime.now() - timedelta(hours=25)
        wrapper.lock = _RefreshingLock(wrapper)

        app.state.aim.cleanup_expired_attributors()

        assert wrapper.id in app.state.aim.attributors
        assert event_manager._closed is False


class _PassthroughParams(RuleParamsBase):
    value: int = 0


@pytest.fixture
def passthrough_collector_rule(monkeypatch):
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
    get_rule = engine.get_rule
    monkeypatch.setattr(engine, "get_rule", lambda rule_id: rule if rule_id == name else get_rule(rule_id))
    yield name


@pytest.fixture
def no_params_collector_rule(monkeypatch):
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
    get_rule = engine.get_rule
    monkeypatch.setattr(engine, "get_rule", lambda rule_id: rule if rule_id == name else get_rule(rule_id))
    yield name


class TestRuleParamsValidation:
    """Test that the rule_params validator accepts a known rule's schema and rejects unknown ids."""

    def test_parameter_model_uses_registered_rule_schema(self, app, passthrough_collector_rule):
        parameters = AttributionParameters(
            rule_params={passthrough_collector_rule: RuleParameters(params={"value": 42})}
        )

        assert parameters.rule_params[passthrough_collector_rule].params == {"value": 42}

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
