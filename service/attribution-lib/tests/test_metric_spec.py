# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import attribution_lib
from attribution_lib.core import get_bundled_rules_path
from attribution_lib.exceptions import RuleIngestionError, UninitializedRuleEngineError
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.prefilters import MetricSpec, register_metric_spec
from attribution_lib.rules_engine import RuleLoader


@pytest.fixture
def active_loader():
    with RuleLoader() as loader:
        yield loader


class TestMetricSpecRegistration:
    def test_register_and_get_metric_spec(self, active_loader, monkeypatch):
        spec = MetricSpec(thresholds={"cpu_temp": {"threshold": 90, "operator": "gt"}})
        register_metric_spec(ObservationSource.NODE_METRICS, spec)
        monkeypatch.setattr(attribution_lib, "_rules_engine", active_loader.build())
        result = attribution_lib.get_metric_spec(ObservationSource.NODE_METRICS)
        assert result is not None
        assert "cpu_temp" in result.thresholds
        assert result.thresholds["cpu_temp"]["threshold"] == 90

    def test_get_metric_spec_unregistered_raises(self, active_loader):
        with pytest.raises(UninitializedRuleEngineError):
            active_loader.build().get_metric_spec(ObservationSource.NVLINK_METRICS)

    def test_multiple_registrations_merge(self, active_loader):
        register_metric_spec(
            ObservationSource.NVLINK_METRICS,
            MetricSpec(thresholds={"m1": {"threshold": 1}}),
        )
        register_metric_spec(
            ObservationSource.NVLINK_METRICS,
            MetricSpec(thresholds={"m2": {"threshold": 2}}),
        )
        result = active_loader.build().get_metric_spec(ObservationSource.NVLINK_METRICS)
        assert result is not None
        assert "m1" in result.thresholds
        assert "m2" in result.thresholds

    def test_registration_copies_dict(self, active_loader):
        """Registration should copy, not alias, the input dict."""
        original = {"m1": {"threshold": 1}}
        register_metric_spec(
            ObservationSource.NODE_METRICS,
            MetricSpec(thresholds=original),
        )
        original["m2"] = {"threshold": 2}
        result = active_loader.build().get_metric_spec(ObservationSource.NODE_METRICS)
        assert "m2" not in result.thresholds

    def test_registration_requires_an_active_loader(self):
        with pytest.raises(RuleIngestionError, match="No active rule loader"):
            register_metric_spec(ObservationSource.NODE_METRICS, MetricSpec())

    def test_loaders_do_not_share_metric_specs(self):
        first = RuleLoader()
        second = RuleLoader()
        first.register_metric_spec(ObservationSource.NODE_METRICS, MetricSpec(thresholds={"m1": {}}))

        assert ObservationSource.NODE_METRICS in first.metric_specs
        assert second.metric_specs == {}


def test_repeated_bundled_loads_each_own_their_metric_specs():
    rules_path = get_bundled_rules_path()
    first = RuleLoader.load_all([rules_path])
    second = RuleLoader.load_all([rules_path])

    expected = {ObservationSource.NODE_METRICS}
    assert set(first.metric_specs) == expected
    assert set(second.metric_specs) == expected
