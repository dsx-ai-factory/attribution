# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refiner rules: run on materialized symptoms and bad nodes, in priority order."""

import hashlib
from typing import Optional

from attribution_lib.decision_trace_recorder import NodeSymptomSnapshot, snapshot_bad_nodes
from attribution_lib.rule_types.bases import OrderedLoader, OrderedSubEngine, rule_type
from attribution_lib.rule_types.definitions import (
    ExecutionContext,
    RefinerContext,
    RefinerProcessor,
    RefinerRule,
)


class RefinerEngine(OrderedSubEngine[RefinerRule, RefinerProcessor]):
    """Runs refiner processors in priority order, recording per-rule diffs."""

    def handles(self, context: ExecutionContext) -> bool:
        return isinstance(context, RefinerContext)

    def execute(self, context: ExecutionContext) -> None:
        """Run refiner processors in priority order.

        ``context.phase_window`` supplies inclusive lower and exclusive upper
        priority bounds. A ``None`` bound is open-ended.
        """
        if not isinstance(context, RefinerContext):  # narrowing guard; mypy does not narrow `is False`
            return
        self._ensure_sorted()
        lower, upper = context.phase_window
        for priority in sorted(self._by_phase.keys()):
            if lower is not None and priority < lower:
                continue
            if upper is not None and priority >= upper:
                break
            self._run_phase(self._by_phase[priority], context)

    @staticmethod
    def _run_phase(phase_rules: list[tuple[str, RefinerProcessor]], context: RefinerContext) -> None:
        """Run one priority's processors, recording what each changed.

        Diffing is skipped entirely when no recorder is attached, which is the
        common path.
        """
        recorder = context.recorder
        symptoms = context.symptoms
        my_bad_nodes = context.my_bad_nodes
        cached_bad: Optional[NodeSymptomSnapshot] = None

        for name, processor in phase_rules:
            before: Optional[set[str]] = None
            if recorder is not None:
                before = set(symptoms.symptoms()) if symptoms is not None else set()
                if cached_bad is None:
                    cached_bad = snapshot_bad_nodes(my_bad_nodes)
            processor(context)
            if recorder is not None and before is not None:
                after = set(symptoms.symptoms()) if symptoms is not None else set()
                after_bad = snapshot_bad_nodes(my_bad_nodes)
                if after != before or after_bad != cached_bad:
                    recorder.record_refiner_diff(name, before, after, cached_bad or {}, after_bad)
                cached_bad = after_bad

    def attribution_hash_contribution(self) -> bytes:
        """Name and priority of every refiner rule, for attribution cache keying."""
        h = hashlib.sha256()
        for name in sorted(self.rules):
            h.update(name.encode())
            h.update(str(self.rules[name].priority).encode())
        return h.digest()


@rule_type(RefinerRule)
class RefinerLoader(OrderedLoader[RefinerRule, RefinerProcessor]):
    """Accumulates refiner rules and their registration order."""

    engine_class = RefinerEngine

    def register(self, rule: RefinerRule) -> None:
        super().register(rule)
        self._add_to_order(rule.priority, rule.name, rule.processor)

    def build(self) -> RefinerEngine:
        return RefinerEngine(rules=dict(self.rules), execution_order=list(self._execution_order))
