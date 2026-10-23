# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-observation execution for extractor rules."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional, Protocol

from loguru import logger

from attribution_lib.exceptions import UninitializedRuleEngineError, UnknownResourceError
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.resources import get_single_node
from attribution_lib.rule_types.definitions import (
    ExtractorAccumulator,
    ExtractorAccumulatorRule,
    ExtractorRule,
    RuleContext,
    RuleId,
    TwoStageSourceMatcher,
)
from attribution_lib.sources import Source


class ExtractorSource(Protocol):
    """What an ObservationProcessor needs from whatever supplies its rules.

    Structural rather than nominal so the processor does not care whether it
    is handed the coordinator or the extractor sub-engine.
    """

    @property
    def extractor_rules(self) -> dict[RuleId, ExtractorRule]: ...

    @property
    def extractor_accumulator_rules(self) -> dict[RuleId, ExtractorAccumulatorRule]: ...

    @property
    def matcher(self) -> Optional[TwoStageSourceMatcher]: ...


_SOFT_LINE_LIMIT = 100
_ACTIVE_COUNT_WARN = 10


@dataclass(slots=True, eq=False)
class _ActiveAccumulator:
    instance: ExtractorAccumulator
    lines_fed: int = 0
    warned_line_limit: bool = False


@dataclass(slots=True)
class ObservationProcessor:
    """Stateful per-observation processor created by RuleEngine.

    Owns extractor-accumulator state. Single public method ``process_line()`` does:
    1. Feed line to active extractor-accumulators + clean up finished ones
    2. Single Aho-Corasick pass → unified candidate loop dispatching to
       extractors OR spawning extractor-accumulators
    3. Emit events via the event adder

    Safeguards:
    - Soft line limit: warns once per extractor-accumulator exceeding threshold
    - Active count high-water mark: warns once if too many active
    """

    engine: ExtractorSource
    source_type: ObservationSource
    context: RuleContext
    event_adder: Callable
    _acc_rules: dict[RuleId, ExtractorAccumulatorRule] = field(init=False)
    _active: dict[RuleId, _ActiveAccumulator] = field(init=False, default_factory=dict)
    _warned_active_count: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._acc_rules = {
            r.name: r for r in self.engine.extractor_accumulator_rules.values() if self.source_type in r.source_filters
        }

    # -- Extractor-accumulator lifecycle helpers --

    def _emit_events(self, acc: ExtractorAccumulator) -> None:
        """Finalize an extractor-accumulator and emit its events."""
        for evt in acc.finalize():
            self.event_adder(evt.type, evt.payload, evt.source)

    def _deactivate(self, rid: RuleId) -> None:
        """Remove an active extractor-accumulator, clean up tracking state, and emit its events."""
        active = self._active.pop(rid)
        self._emit_events(active.instance)

    def _try_spawn(self, rule_id: RuleId, source: Source) -> None:
        """Spawn a new extractor-accumulator if the rule isn't already active.

        If the new instance is still consuming after start(), it becomes
        active and will be fed subsequent lines. Otherwise it is finalized
        immediately and discarded.
        """
        if rule_id in self._active:
            return
        new_acc = self._acc_rules[rule_id].process()
        new_acc.start(source)
        if new_acc.is_consuming is True:
            self._active[rule_id] = _ActiveAccumulator(new_acc)
            self._check_active_count()
        else:
            self._emit_events(new_acc)

    def _feed_active(self, source: Source) -> None:
        """Feed a line to every active extractor-accumulator, then finalize any that finished."""
        if len(self._active) == 0:
            return
        finished: list[RuleId] = []
        for rid, active in self._active.items():
            active.instance.process_line(source)
            active.lines_fed += 1
            if not active.warned_line_limit and active.lines_fed >= _SOFT_LINE_LIMIT:
                active.warned_line_limit = True
                logger.warning(
                    f"Extractor-accumulator {rid!r} has consumed {active.lines_fed} lines without finalizing"
                )
            if active.instance.is_consuming is False:
                finished.append(rid)
        for rid in finished:
            self._deactivate(rid)

    def _check_active_count(self) -> None:
        active_count = len(self._active)
        if self._warned_active_count is False and active_count >= _ACTIVE_COUNT_WARN:
            self._warned_active_count = True
            logger.warning(f"High extractor-accumulator count: {active_count} active")

    # -- Main entry points --

    def process_line(self, source: Source) -> None:
        """Process a single line through extractors and extractor-accumulators."""
        try:
            self.context.hostname = get_single_node(source.resources).hostname
        except UnknownResourceError:
            self.context.hostname = None

        self._feed_active(source)

        matcher = self.engine.matcher
        if matcher is None:
            raise UninitializedRuleEngineError("Cannot process observations before the rule matcher is initialized")
        for rule_id in matcher.get_candidates(self.source_type, source.datum):
            match = matcher.confirm_match(self.source_type, rule_id, source.datum)
            if match is None:
                continue
            if (ext_rule := self.engine.extractor_rules.get(rule_id)) is not None:
                for evt in ext_rule.process(match, source, self.context):
                    self.event_adder(evt.type, evt.payload, evt.source)
            elif rule_id in self._acc_rules:
                self._try_spawn(rule_id, source)

    def finalize(self) -> None:
        """End-of-stream: finalize all remaining active extractor-accumulators."""
        for rid in list(self._active):
            self._deactivate(rid)
