# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-filter and metric specifications declared by Python rule files.

The specifications themselves are plain values. Registration delegates to the
active ``RuleLoader``; runtime lookup belongs to the built ``RuleEngine``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, Union

from attribution_lib.declarative.context import _require_rule_loader
from attribution_lib.observation.inbound import Source as ObservationSource

if TYPE_CHECKING:
    from attribution_lib.job_info import JobInfo
    from attribution_lib.observation.processing import EntryProtocol


PreFilterCallable: TypeAlias = Callable[["EntryProtocol", "JobInfo"], Union[bool, tuple[bool, list[str]]]]


@dataclass
class PreFilterSpec:
    """Pre-filter specification registered by rules for a given source type."""

    include_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    callables: list[PreFilterCallable] = field(default_factory=list)


@dataclass
class MetricSpec:
    """Metric collection specification registered by rules."""

    thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)


def register_pre_filter(source: ObservationSource, spec: PreFilterSpec) -> None:
    """Register a pre-filter specification on the active loader."""
    loader = _require_rule_loader("registering a pre-filter")
    loader.register_pre_filter(source, spec)


def register_metric_spec(source: ObservationSource, spec: MetricSpec) -> None:
    """Register a metric specification on the active loader."""
    loader = _require_rule_loader("registering a metric specification")
    loader.register_metric_spec(source, spec)
