# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest

import attribution_lib
import attribution_lib.exceptions
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rules_engine import _METRIC_SPECS, MetricSpec, register_metric_spec


class TestMetricSpecRegistration:
    def setup_method(self):
        """Clear registration state between tests."""
        _METRIC_SPECS.clear()

    def test_register_and_get_metric_spec(self):
        spec = MetricSpec(thresholds={"cpu_temp": {"threshold": 90, "operator": "gt"}})
        register_metric_spec(ObservationSource.NODE_METRICS, spec)
        result = attribution_lib.get_metric_spec(ObservationSource.NODE_METRICS)
        assert result is not None
        assert "cpu_temp" in result.thresholds
        assert result.thresholds["cpu_temp"]["threshold"] == 90

    def test_get_metric_spec_unregistered_raises(self):
        with pytest.raises(attribution_lib.exceptions.UninitializedRuleEngineError):
            attribution_lib.get_metric_spec(ObservationSource.NVLINK_METRICS)

    def test_multiple_registrations_merge(self):
        register_metric_spec(
            ObservationSource.NVLINK_METRICS,
            MetricSpec(thresholds={"m1": {"threshold": 1}}),
        )
        register_metric_spec(
            ObservationSource.NVLINK_METRICS,
            MetricSpec(thresholds={"m2": {"threshold": 2}}),
        )
        result = attribution_lib.get_metric_spec(ObservationSource.NVLINK_METRICS)
        assert result is not None
        assert "m1" in result.thresholds
        assert "m2" in result.thresholds

    def test_registration_copies_dict(self):
        """Registration should copy, not alias, the input dict."""
        original = {"m1": {"threshold": 1}}
        register_metric_spec(
            ObservationSource.NODE_METRICS,
            MetricSpec(thresholds=original),
        )
        original["m2"] = {"threshold": 2}
        result = attribution_lib.get_metric_spec(ObservationSource.NODE_METRICS)
        assert "m2" not in result.thresholds
