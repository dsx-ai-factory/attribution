# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Collector rules: query the finished event stream and format digest output."""

import hashlib
from typing import Optional, cast

from loguru import logger
from pydantic import ValidationError

from attribution_lib.events import EventStream
from attribution_lib.exceptions import InvalidRuleParams, RuleAcceptsNoParamsError
from attribution_lib.job_info import JobInfo
from attribution_lib.rule_types.bases import LoaderBase, SubEngineBase, rule_type
from attribution_lib.rule_types.definitions import (
    CollectorContext,
    CollectorRule,
    ExecutionContext,
    RuleParameters,
    RuleParamsBase,
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
                params = self.parse_params(rule, params_by_rule.get(rule_id))
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

    @staticmethod
    def parse_params(rule: CollectorRule, raw: Optional[RuleParameters]) -> Optional[RuleParamsBase]:
        """Parse a ``RuleParameters`` envelope against the rule's params_model.

        Returns:
            ``None`` if the rule declares no ``params_model``; otherwise
            constructs the model (with defaults when ``raw`` is ``None``).
        Raises:
            ``RuleAcceptsNoParamsError``: If params are supplied for a rule with
                no ``params_model``.
            ``InvalidRuleParams``: If the payload fails ``params_model`` validation.
        """
        if rule.params_model is None:
            if raw is not None and raw.params != {}:
                raise RuleAcceptsNoParamsError(rule.name)
            return None

        params_data = raw.params if raw is not None else {}
        try:
            return rule.params_model(**params_data)
        except ValidationError as e:
            raise InvalidRuleParams(rule.name, e) from e

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
