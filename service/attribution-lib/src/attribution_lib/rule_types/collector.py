# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector rules: query the finished event stream and format digest output."""

import hashlib
from typing import cast

from loguru import logger

from attribution_lib.events import EventStream
from attribution_lib.job_info import JobInfo
from attribution_lib.rule_types.bases import LoaderBase, SubEngineBase, rule_type
from attribution_lib.rule_types.definitions import (
    CollectorContext,
    CollectorRule,
    ExecutionContext,
)


class CollectorEngine(SubEngineBase[CollectorRule]):
    """Runs every collector and merges their digest output."""

    def handles(self, context: ExecutionContext) -> bool:
        return isinstance(context, CollectorContext)

    def execute(self, context: ExecutionContext) -> None:
        """Run every collector, accumulating into ``context.result``.

        A collector that raises is logged and skipped; one bad collector must
        not lose the whole digest. Duplicate keys across collectors are logged
        and the first value wins.
        """
        if not isinstance(context, CollectorContext):  # narrowing guard; mypy does not narrow `is False`
            return
        params_by_rule = context.rule_params if context.rule_params is not None else {}
        for rule_id, rule in self.rules.items():
            try:
                params = rule.parse_params(params_by_rule.get(rule_id))
                result = rule.process(cast(EventStream, context.event_stream), cast(JobInfo, context.job_info), params)
            except Exception:
                logger.opt(exception=True).error(f"Collector '{rule_id}' failed")
                continue
            for key, value in result.items():
                if key in context.result:
                    logger.error(
                        f"Duplicate digest key '{key}': collector '{rule_id}' "
                        f"would overwrite existing value (keeping first)"
                    )
                else:
                    context.result[key] = value

    def attribution_hash_contribution(self) -> bytes:
        """Collector rule names, for attribution cache keying."""
        h = hashlib.sha256()
        for name in sorted(self.rules):
            h.update(name.encode())
        return h.digest()


@rule_type(CollectorRule)
class CollectorLoader(LoaderBase[CollectorRule]):
    """Accumulates collector rules."""

    engine_class = CollectorEngine

    def build(self) -> CollectorEngine:
        return CollectorEngine(rules=dict(self.rules))
