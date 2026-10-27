# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def _failing_close(**kwargs):
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

    def test_cleanup_failure_does_not_block_other_expired_attributors(self, app, monkeypatch):
        broken = app.state.aim.add_attributor(make_up_attributor_info(id="cleanup-broken"))
        ok = app.state.aim.add_attributor(make_up_attributor_info(id="cleanup-ok"))
        ok_event_manager = ok.attributor.event_stream.mgr
        monkeypatch.setattr(broken.attributor.event_stream.mgr, "close", _failing_close)
        _expire(broken, ok)

        with pytest.raises(RuntimeError):
            app.state.aim.cleanup_expired_attributors()
        assert broken.id not in app.state.aim.attributors

        assert ok.id not in app.state.aim.attributors
        assert ok_event_manager._closed is True
        assert broken in app.state.aim._pending_cleanup.values()


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


@pytest.mark.parametrize("operation", ["delete", "expire", "shutdown"])
def test_retirement_retries_cleanup_without_flushing(app, monkeypatch, tmp_path, operation):
    import shutil
    from pathlib import Path
    from unittest.mock import Mock

    monkeypatch.setenv("ATTRIBUTOR_ALLOW_DISK_SPILL", "true")
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_TEMP_ROOT", str(tmp_path))
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "16MB")
    aim = app.state.aim
    info = make_up_attributor_info(id="retry-spill")
    original = aim.add_attributor(info)
    original._event_manager._conn.execute("CREATE TABLE big AS SELECT repeat('x', 1000000) FROM range(50)")
    assert list(Path(original._event_manager._spill_dir).iterdir())
    monkeypatch.setattr(original.attributor.event_stream, "flush", Mock(side_effect=AssertionError("must not flush")))
    if operation == "expire":
        _expire(original)
    action = {
        "delete": lambda: aim.delete_attributor(original.id),
        "expire": aim.cleanup_expired_attributors,
        "shutdown": aim.close,
    }[operation]
    with monkeypatch.context() as failing:
        failing.setattr(shutil, "rmtree", Mock(side_effect=PermissionError("injected")))
        with pytest.raises(PermissionError):
            action()
        (retired,) = aim._pending_cleanup.values()
        directory = Path(retired._event_manager._spill_dir)
        assert directory.exists() and retired._closed
        assert original.id not in aim.attributors
        with pytest.raises(PermissionError):
            aim.cleanup_expired_attributors()
        assert list(aim._pending_cleanup.values()) == [retired]
    live = aim.add_attributor(info)
    aim.cleanup_expired_attributors()
    assert not aim._pending_cleanup and not directory.exists()
    assert aim.get_attributor(live.id) is live
    assert live._event_manager._conn.execute("SELECT 1").fetchone() == (1,)
    aim.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["connect", "attributor", "checkpoint"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_setup_is_already_owned_and_released(app, monkeypatch, tmp_path, stage, error_type):
    from unittest.mock import patch

    monkeypatch.setenv("ATTRIBUTOR_ALLOW_DISK_SPILL", "true")
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_TEMP_ROOT", str(tmp_path))
    aim = app.state.aim
    retained = []

    def fail(*args, **kwargs):
        (wrapper,) = aim._pending_cleanup.values()
        retained.append(wrapper)
        assert not aim.attributors
        raise error_type("setup failed")

    target = {
        "connect": "duckdb.connect",
        "attributor": "app.attributor.Attributor.new",
        "checkpoint": "app.attributor.MemoryCheckpointBackend",
    }[stage]
    with patch(target, side_effect=fail), pytest.raises(error_type, match="setup failed"):
        aim.add_attributor(make_up_attributor_info())
    assert retained[0]._closed and retained[0]._event_manager._closed
    assert not aim._pending_cleanup and not aim.attributors
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["event_manager", "wrapper"])
@pytest.mark.parametrize("cleanup_failure", ["connection", "directory"])
def test_failed_construction_and_failed_cleanup_remain_owned(app, monkeypatch, tmp_path, stage, cleanup_failure):
    import gc
    import shutil
    import weakref
    from pathlib import Path

    from attribution_lib.events import EventManager

    from app.attributor import Attributor

    monkeypatch.setenv("ATTRIBUTOR_ALLOW_DISK_SPILL", "true")
    monkeypatch.setenv("ATTRIBUTOR_DUCKDB_TEMP_ROOT", str(tmp_path))
    aim = app.state.aim
    info = make_up_attributor_info(id="double-failure")
    original_new = Attributor.new
    original_close = EventManager.close
    original_rmtree = shutil.rmtree
    failed_identity = None
    failed_directory = None

    def fail_wrapper(*args):
        raise RuntimeError("initial setup failed")

    def fail_close(manager, **kwargs):
        nonlocal failed_identity
        if failed_identity is None:
            failed_identity = id(manager)
        if id(manager) == failed_identity:
            raise PermissionError("injected close failure")
        return original_close(manager, **kwargs)

    def fail_removal(path, *args, **kwargs):
        nonlocal failed_directory
        if failed_directory is None:
            failed_directory = path
        if path == failed_directory:
            raise PermissionError("injected directory failure")
        return original_rmtree(path, *args, **kwargs)

    if stage == "event_manager":
        monkeypatch.setenv("ATTRIBUTOR_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", "invalid-size")
    else:
        monkeypatch.setattr(Attributor, "new", fail_wrapper)
    if cleanup_failure == "connection":
        monkeypatch.setattr(EventManager, "close", fail_close)
    else:
        monkeypatch.setattr(shutil, "rmtree", fail_removal)

    from attribution_lib.exceptions import InvalidEventStorageSizeError

    with pytest.raises(InvalidEventStorageSizeError if stage == "event_manager" else RuntimeError) as caught:
        aim.add_attributor(info)
    assert "injected" not in str(caught.value)  # Cleanup must not replace the setup error.
    assert not aim.attributors
    assert len(aim._pending_cleanup) == 1
    resource = next(iter(aim._pending_cleanup.values()))._event_manager
    reference = weakref.ref(resource)
    directory = Path(resource._spill_dir)
    del resource, caught
    gc.collect()
    assert reference() is not None
    assert directory.exists()

    # A failed constructor is not an active handle. A same-ID replacement can
    # run, and unrelated expired objects must still close while retries fail.
    monkeypatch.delenv("ATTRIBUTOR_DUCKDB_MAX_TEMP_DIRECTORY_SIZE", raising=False)
    monkeypatch.setattr(Attributor, "new", original_new)
    replacement = aim.add_attributor(info)
    expired = aim.add_attributor(make_up_attributor_info(id="unrelated-expired"))
    _expire(expired)
    for _ in range(2):
        with pytest.raises(PermissionError, match="injected"):
            aim.cleanup_expired_attributors()
        assert len(aim._pending_cleanup) == 1
        assert aim.get_attributor(replacement.id) is replacement
        assert expired._event_manager._closed

    monkeypatch.setattr(EventManager, "close", original_close)
    monkeypatch.setattr(shutil, "rmtree", original_rmtree)
    aim.cleanup_expired_attributors()
    assert not aim._pending_cleanup
    assert not directory.exists()
    assert replacement._event_manager._conn.execute("SELECT 1").fetchone() == (1,)
    aim.close()
    assert not list(tmp_path.iterdir())


def test_duplicate_post_never_initializes_or_cleans_resources(app, client, monkeypatch):
    from unittest.mock import Mock

    from app.attributor import AttributorWrapper

    info = make_up_attributor_info()
    original = app.state.aim.add_attributor(info)
    initialize = Mock(side_effect=AssertionError("duplicate must not initialize"))
    close = Mock(side_effect=AssertionError("duplicate must not close"))
    monkeypatch.setattr(AttributorWrapper, "initialize", initialize)
    monkeypatch.setattr(AttributorWrapper, "close", close)
    assert client.post("/v1/attributor", content=info.model_dump_json()).status_code == 409
    initialize.assert_not_called()
    close.assert_not_called()
    assert app.state.aim.get_attributor(original.id) is original
    assert not app.state.aim._pending_cleanup
    original._event_manager.close(flush=False)


def test_standalone_wrapper_preserves_setup_error():
    from unittest.mock import patch

    from app.attributor import AttributorWrapper

    original = RuntimeError("setup failed")
    with (
        patch.object(AttributorWrapper, "initialize", side_effect=original),
        patch.object(AttributorWrapper, "close", side_effect=PermissionError("close failed")),
        pytest.raises(RuntimeError) as caught,
    ):
        AttributorWrapper.new(make_up_attributor_info())
    assert caught.value is original
