# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extractor rules: turn observation lines into events.

Two rule types share this pair. A plain extractor matches one line and emits;
an extractor-accumulator is spawned by a trigger match and then consumes
subsequent lines until it decides it is done. They share a matcher because
the Aho-Corasick pre-filter is rule-type-agnostic.
"""

import hashlib
from collections.abc import Mapping
from typing import Any, Optional

from attribution_lib.exceptions import DuplicateRuleError
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rule_types.bases import LoaderBase, SubEngineBase, rule_type
from attribution_lib.rule_types.definitions import (
    ExecutionContext,
    ExtractionContext,
    ExtractorAccumulatorRule,
    ExtractorRule,
    Rule,
    RuleContext,
    RuleId,
    TwoStageSourceMatcher,
)


class ExtractorEngine(SubEngineBase[ExtractorRule]):
    """Builds the per-source matcher and hands out ObservationProcessors."""

    def __init__(
        self,
        rules: Optional[dict[RuleId, ExtractorRule]] = None,
        accumulator_rules: Optional[dict[RuleId, ExtractorAccumulatorRule]] = None,
        declarative_emit: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(rules)
        self.extractor_accumulator_rules: dict[RuleId, ExtractorAccumulatorRule] = (
            dict(accumulator_rules) if accumulator_rules is not None else {}
        )
        self.declarative_emit: dict[str, str] = dict(declarative_emit) if declarative_emit is not None else {}
        self.matcher: Optional[TwoStageSourceMatcher] = None

    @property
    def extractor_rules(self) -> dict[RuleId, ExtractorRule]:
        """Alias for self.rules under the name ObservationProcessor expects."""
        return self.rules

    def build_matcher(self) -> None:
        """Build the two-stage matcher from both extractor rule types.

        Built lazily on first execute() rather than at construction: compiling
        the Aho-Corasick automaton is not free, and an engine built but never
        run should not pay for it.
        """
        patterns_by_source: dict[ObservationSource, dict[RuleId, str]] = {}
        for rule in self.rules.values():
            if rule.match_re == "":
                continue
            for source in rule.source_filters:
                patterns_by_source.setdefault(source, {})[rule.name] = rule.match_re
        for acc_rule in self.extractor_accumulator_rules.values():
            if acc_rule.match_re == "":
                continue
            for source in acc_rule.source_filters:
                patterns_by_source.setdefault(source, {})[acc_rule.name] = acc_rule.match_re
        self.matcher = TwoStageSourceMatcher(patterns_by_source)

    def handles(self, context: ExecutionContext) -> bool:
        return isinstance(context, ExtractionContext)

    def execute(self, context: ExecutionContext) -> None:
        """Create an ObservationProcessor for the context's source type."""
        if not isinstance(context, ExtractionContext):  # narrowing guard; mypy does not narrow `is False`
            return
        if self.matcher is None:
            self.build_matcher()
        assert context.event_stream is not None
        # Local import: observation imports this module for its engine type.
        from attribution_lib.rule_types.observation import ObservationProcessor

        context.result = ObservationProcessor(
            engine=self,
            source_type=context.source_type,
            context=RuleContext(job_info=context.job_info),
            event_adder=context.event_stream.event_adder(),
        )

    def extra_rules(self) -> Mapping[RuleId, Rule]:
        """Accumulator rules, for the coordinator's combined rules view."""
        return self.extractor_accumulator_rules

    def extra_rule_names(self) -> set:
        """Accumulator rule names, for all_rule_names."""
        return set(self.extractor_accumulator_rules.keys())

    def extractor_hash_contribution(self) -> str:
        """Identity of every extractor rule, for extraction cache keying."""
        h = hashlib.sha256()
        for name in sorted(self.rules):
            rule = self.rules[name]
            h.update(name.encode())
            h.update(rule.match_re.encode())
            h.update(str(sorted(sf.value for sf in rule.source_filters)).encode())
            h.update(self.declarative_emit.get(name, "").encode())
        for name in sorted(self.extractor_accumulator_rules):
            rule = self.extractor_accumulator_rules[name]
            h.update(name.encode())
            h.update(rule.match_re.encode())
            h.update(str(sorted(sf.value for sf in rule.source_filters)).encode())
        return h.hexdigest()


@rule_type(ExtractorRule, ExtractorAccumulatorRule)
class ExtractorLoader(LoaderBase[ExtractorRule]):
    """Accumulates both extractor rule types, keeping them in separate stores."""

    engine_class = ExtractorEngine

    def __init__(self) -> None:
        super().__init__()
        self.extractor_accumulator_rules: dict[RuleId, ExtractorAccumulatorRule] = {}
        self.declarative_emit: dict[str, str] = {}

    def register(self, rule: ExtractorRule | ExtractorAccumulatorRule) -> None:
        """Route to the right store based on rule type."""
        if isinstance(rule, ExtractorAccumulatorRule):
            self._register_accumulator(rule)
            return
        super().register(rule)

    def _register_accumulator(self, rule: ExtractorAccumulatorRule) -> None:
        existing = self.find_by_name(rule.name)
        if existing is not None:
            raise DuplicateRuleError(
                f"Duplicate rule '{rule.name}': "
                f"first at {existing.location.file}:{existing.location.line}, "
                f"duplicate at {rule.location.file}:{rule.location.line}"
            )
        self.extractor_accumulator_rules[rule.name] = rule

    def find_by_name(self, name: RuleId) -> Any:
        """Look in both stores; a name may not be reused across either."""
        rule = self.rules.get(name)
        return rule if rule is not None else self.extractor_accumulator_rules.get(name)

    def extra_rules(self) -> Mapping[RuleId, Rule]:
        """Accumulator rules, for the coordinator's combined rules view."""
        return self.extractor_accumulator_rules

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "extractor_accumulator_rules": dict(self.extractor_accumulator_rules),
            "declarative_emit": dict(self.declarative_emit),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        super().restore(snap)
        self.extractor_accumulator_rules = dict(snap["extractor_accumulator_rules"])
        self.declarative_emit = dict(snap["declarative_emit"])

    def declarative_emit_store(self) -> dict[str, str]:
        """Expose this loader's declarative_emit for the coordinator's proxy."""
        return self.declarative_emit

    def build(self) -> ExtractorEngine:
        return ExtractorEngine(
            rules=dict(self.rules),
            accumulator_rules=dict(self.extractor_accumulator_rules),
            declarative_emit=dict(self.declarative_emit),
        )
