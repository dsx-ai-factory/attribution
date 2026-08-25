# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from typing import cast

import pytest

from attribution_lib.exceptions import UninitializedRuleEngineError
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rule_types.definitions import (
    EventTriple,
    ExtractorAccumulator,
    ExtractorAccumulatorRule,
    ExtractorRule,
    RuleContext,
    RuleLocation,
)
from attribution_lib.rule_types.extractor import ExtractorEngine
from attribution_lib.rule_types.observation import ObservationProcessor
from attribution_lib.sources import Source


def _source(datum: str) -> Source:
    return Source(
        source_context=cast(ObservationContext, None),
        line_num=None,
        datum=datum,
        time=None,
        resources=AllJobResources,
        rule=None,
    )


def test_active_accumulator_is_not_retriggered():
    class RecordingAccumulator(ExtractorAccumulator):
        def __init__(self) -> None:
            self.sources: list[Source] = []
            self._consuming = True

        def process_line(self, source: Source) -> None:
            self.sources.append(source)
            if source.datum == "stop":
                self._consuming = False

        @property
        def is_consuming(self) -> bool:
            return self._consuming

        def finalize(self) -> list[EventTriple]:
            return [
                EventTriple(
                    "test.accumulator",
                    {"lines": [source.datum for source in self.sources]},
                    self.sources[0],
                )
            ]

    created: list[RecordingAccumulator] = []

    def factory() -> RecordingAccumulator:
        accumulator = RecordingAccumulator()
        created.append(accumulator)
        return accumulator

    rule = ExtractorAccumulatorRule(
        name="single",
        author="test",
        location=RuleLocation(__file__, 0),
        match_re="start",
        factory=factory,
        source_filters=frozenset({ObservationSource.APPLOG}),
    )
    engine = ExtractorEngine(accumulator_rules={rule.name: rule})
    engine.build_matcher()
    emitted = []
    processor = ObservationProcessor(
        engine=engine,
        source_type=ObservationSource.APPLOG,
        context=RuleContext(),
        event_adder=lambda event_type, payload, source, *, emitter_rule_id=None: emitted.append(
            (event_type, payload, source, emitter_rule_id)
        ),
    )

    processor.process_line(_source("start one"))
    processor.process_line(_source("start two"))

    assert len(created) == 1
    assert set(processor._active) == {"single"}

    processor.process_line(_source("stop"))

    assert processor._active == {}
    assert [payload["lines"] for _, payload, _, _ in emitted] == [["start one", "start two", "stop"]]
    assert {emitter_rule_id for _, _, _, emitter_rule_id in emitted} == {"single"}


def test_plain_extractor_tags_emitted_source_with_stable_id():
    """A matched extractor ID follows its emitted event into provenance."""

    def emit_event(_groups, _hostname, source, _context):
        return [EventTriple("test.extractor", {}, source.derived(rule="credited_rule"))]

    rule = ExtractorRule(
        name="stable_extractor",
        author="test",
        location=RuleLocation(__file__, 0),
        match_re="matched",
        processor=emit_event,
        source_filters=frozenset({ObservationSource.APPLOG}),
    )
    engine = ExtractorEngine(rules={rule.name: rule})
    engine.build_matcher()
    emitted = []
    processor = ObservationProcessor(
        engine=engine,
        source_type=ObservationSource.APPLOG,
        context=RuleContext(),
        event_adder=lambda event_type, payload, source, *, emitter_rule_id=None: emitted.append(
            (event_type, payload, source, emitter_rule_id)
        ),
    )

    processor.process_line(_source("matched line"))

    assert len(emitted) == 1
    assert emitted[0][2].rule == "credited_rule"
    assert emitted[0][3] == "stable_extractor"


def test_processing_requires_an_initialized_matcher():
    processor = ObservationProcessor(
        engine=ExtractorEngine(),
        source_type=ObservationSource.APPLOG,
        context=RuleContext(),
        event_adder=lambda *_args: None,
    )

    with pytest.raises(UninitializedRuleEngineError, match="matcher is initialized"):
        processor.process_line(_source("line"))
