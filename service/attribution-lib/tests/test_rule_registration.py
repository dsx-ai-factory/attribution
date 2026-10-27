# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from attribution_lib.declarative.context import _using_module_globals
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rule_types.definitions import (
    CollectorRule,
    EventTriple,
    ExtractorAccumulator,
    ExtractorAccumulatorRule,
    FilterPriority,
    FilterRule,
    RefinerRule,
)
from attribution_lib.rule_types.registration import (
    collector_def,
    extractor_accumulator_def,
    filter_def,
    refiner_def,
)
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.sources import Source


def test_rule_decorators_share_registration_metadata_and_return_the_decorated_object():
    with (
        _using_module_globals(
            {
                "rule_author": "Module Author",
                "rule_sources": {ObservationSource.APPLOG},
            }
        ),
        RuleLoader() as loader,
    ):

        def filter_processor(context: Any) -> None:
            pass

        registered_filter = filter_def(
            "Filter",
            author="Explicit Author",
            priority=FilterPriority.EARLY,
            derived_rules=["filter-derived"],
            derived_event_types=["filter-event"],
            tags=[" second ", "first", "first"],
        )(filter_processor)

        class Accumulator(ExtractorAccumulator):
            def process_line(self, source: Source) -> None:
                pass

            @property
            def is_consuming(self) -> bool:
                return False

            def finalize(self) -> list[EventTriple]:
                return []

        registered_accumulator = extractor_accumulator_def(
            "Accumulator",
            match_re="trigger",
            derived_rules=["accumulator-derived"],
            derived_event_types=["accumulator-event"],
            tags=["second", "first"],
        )(Accumulator)

        def refiner_processor(context: Any) -> None:
            pass

        registered_refiner = refiner_def(
            "Refiner",
            derived_rules=["refiner-derived"],
            derived_event_types=["refiner-event"],
            tags=["second", "first"],
        )(refiner_processor)

        def collector_processor(event_stream: Any, job_info: Any, params: Any) -> None:
            pass

        def formatter(raw: Any) -> dict[str, Any]:
            return {}

        registered_collector = collector_def(
            "Collector",
            formatter=formatter,
            derived_rules=["collector-derived"],
            derived_event_types=["collector-event"],
            tags=["second", "first"],
        )(collector_processor)

    assert registered_filter is filter_processor
    assert registered_accumulator is Accumulator
    assert registered_refiner is refiner_processor
    assert registered_collector is collector_processor

    assert isinstance(loader.rules["Filter"], FilterRule)
    assert isinstance(loader.rules["Accumulator"], ExtractorAccumulatorRule)
    assert isinstance(loader.rules["Refiner"], RefinerRule)
    assert isinstance(loader.rules["Collector"], CollectorRule)

    for name in ("Filter", "Accumulator", "Refiner", "Collector"):
        rule = loader.rules[name]
        expected_author = "Explicit Author" if name == "Filter" else "Module Author"
        assert rule.author == expected_author
        assert rule.tags == ("first", "second")
        assert rule.location.file == __file__
        assert rule.location.line > 0

    accumulator_rule = loader.rules["Accumulator"]
    assert isinstance(accumulator_rule, ExtractorAccumulatorRule)
    assert accumulator_rule.source_filters == frozenset({ObservationSource.APPLOG})

    assert loader.declared_rule_names == {
        "filter-derived",
        "accumulator-derived",
        "refiner-derived",
        "collector-derived",
    }
    assert loader.declared_event_types == {
        "filter-event",
        "accumulator-event",
        "refiner-event",
        "collector-event",
    }
