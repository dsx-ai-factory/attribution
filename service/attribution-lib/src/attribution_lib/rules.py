# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public authoring surface for Python rule modules."""

from attribution_lib.declarative.context import const, expr_helper, rows, rule_processor
from attribution_lib.prefilters import MetricSpec, PreFilterSpec, register_metric_spec, register_pre_filter
from attribution_lib.rule_types.definitions import (
    CollectorRule,
    EventTriple,
    ExtractorAccumulator,
    FilterContext,
    FilterPriority,
    RefinerContext,
    RuleContext,
    RuleId,
    RuleLocation,
    RuleParameters,
    RuleParamsBase,
)
from attribution_lib.rule_types.registration import (
    collector_def,
    declare_event_types,
    declare_rule_names,
    extractor_accumulator_def,
    filter_def,
    refiner_def,
)
from attribution_lib.symptoms.family import SymptomFamily
from attribution_lib.symptoms.registration import (
    cascade_symptom,
    known_symptom,
    outcome_symptom,
    symptom_predicate,
)
from attribution_lib.symptoms.selector import SymptomSelector

__all__ = [
    "CollectorRule",
    "EventTriple",
    "ExtractorAccumulator",
    "FilterContext",
    "FilterPriority",
    "MetricSpec",
    "PreFilterSpec",
    "RefinerContext",
    "RuleContext",
    "RuleId",
    "RuleLocation",
    "RuleParameters",
    "RuleParamsBase",
    "SymptomFamily",
    "SymptomSelector",
    "cascade_symptom",
    "collector_def",
    "const",
    "declare_event_types",
    "declare_rule_names",
    "expr_helper",
    "extractor_accumulator_def",
    "filter_def",
    "known_symptom",
    "outcome_symptom",
    "refiner_def",
    "register_metric_spec",
    "register_pre_filter",
    "rows",
    "rule_processor",
    "symptom_predicate",
]
