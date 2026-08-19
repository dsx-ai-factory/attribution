# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Per-source pre-filter and metric specifications registered by rule files.

Deliberately a leaf: this module imports nothing from ``attribution_lib``
beyond ``exceptions`` and ``observation.inbound``, and never the package
itself. ``attribution_lib/__init__.py`` imports ``get_pre_filter`` and
``get_metric_spec`` from here, so anything this module pulls in is pulled in
before the package finishes initializing.

That constraint is what keeps the package acyclic. These lived in
``rules_engine`` and dragged the whole engine into every ``import
attribution_lib``, which in turn made ``utils`` (via ``import
attribution_lib``) a transitive importer of the engine, so ``declarative``
modules could not import their own dependencies at module scope.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, Union

from attribution_lib.exceptions import UninitializedRuleEngineError
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


_PRE_FILTERS: dict[ObservationSource, PreFilterSpec] = {}
_METRIC_SPECS: dict[ObservationSource, MetricSpec] = {}


def register_pre_filter(source: ObservationSource, spec: PreFilterSpec) -> None:
    """Register pre-filter spec for an observation source. Multiple calls merge."""
    if source in _PRE_FILTERS:
        existing = _PRE_FILTERS[source]
        existing.include_terms.extend(spec.include_terms)
        existing.exclude_terms.extend(spec.exclude_terms)
        existing.callables.extend(spec.callables)
    else:
        _PRE_FILTERS[source] = PreFilterSpec(
            include_terms=list(spec.include_terms),
            exclude_terms=list(spec.exclude_terms),
            callables=list(spec.callables),
        )


def register_metric_spec(source: ObservationSource, spec: MetricSpec) -> None:
    """Register metric spec for an observation source. Multiple calls merge."""
    if source in _METRIC_SPECS:
        _METRIC_SPECS[source].thresholds.update(spec.thresholds)
    else:
        _METRIC_SPECS[source] = MetricSpec(thresholds=dict(spec.thresholds))


def get_pre_filter(source: ObservationSource) -> PreFilterSpec:
    """Return the registered pre-filter spec for source, or an empty spec if it has none."""
    return _PRE_FILTERS.get(source, PreFilterSpec())


def get_metric_spec(source: ObservationSource) -> MetricSpec:
    """Return metric spec for source. Raises if not registered."""
    if source not in _METRIC_SPECS:
        raise UninitializedRuleEngineError(f"No metric spec registered for {source.name}. Was setup_rules() called?")
    return _METRIC_SPECS[source]
