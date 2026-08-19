# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Filter rules: transform or prune the event stream, in priority order."""

import hashlib

from attribution_lib.rule_types.bases import OrderedLoader, OrderedSubEngine, rule_type
from attribution_lib.rule_types.definitions import (
    ExecutionContext,
    FilterContext,
    FilterProcessor,
    FilterRule,
)


class FilterEngine(OrderedSubEngine[FilterRule, FilterProcessor]):
    """Runs filter processors in priority order."""

    def handles(self, context: ExecutionContext) -> bool:
        return isinstance(context, FilterContext)

    def execute(self, context: ExecutionContext) -> None:
        """Run filter processors in priority order.

        With ``context.phase`` set, only processors registered at that exact
        priority run. With it None, every phase runs in ascending priority.
        """
        if not isinstance(context, FilterContext):  # narrowing guard; mypy does not narrow `is False`
            return
        self._ensure_sorted()
        phase = context.phase
        if phase is not None:
            for _name, processor in self._by_phase.get(phase, []):
                processor(context)
            return
        for priority in sorted(self._by_phase.keys()):
            for _name, processor in self._by_phase[priority]:
                processor(context)

    def attribution_hash_contribution(self) -> bytes:
        """Name and priority of every filter rule, for attribution cache keying."""
        h = hashlib.sha256()
        for name in sorted(self.rules):
            h.update(name.encode())
            h.update(str(self.rules[name].priority).encode())
        return h.digest()


@rule_type(FilterRule)
class FilterLoader(OrderedLoader[FilterRule, FilterProcessor]):
    """Accumulates filter rules and their registration order."""

    engine_class = FilterEngine

    def register(self, rule: FilterRule) -> None:
        super().register(rule)
        self._add_to_order(rule.priority, rule.name, rule.processor)

    def build(self) -> FilterEngine:
        return FilterEngine(rules=dict(self.rules), execution_order=list(self._execution_order))
