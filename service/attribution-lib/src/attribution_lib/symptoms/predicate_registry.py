# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared type for conditional-discard predicates.

Cascade rules reference a predicate by name in ``conditional_discards``. The
predicate implementations are discrete (they live in bundled_rules) and register
onto the active :class:`~attribution_lib.rules_engine.RuleEngine` via the
``@symptom_predicate`` decorator when their rule file loads. The compiled
``GeneratedSymptomRegistry`` carries the resolved name-to-callable map that
``symptom_set_operator`` consults. This module only defines the shared callable
type; the registry itself lives on the engine (and its registry snapshot), not in
process-global state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from attribution_lib.job_info import JobInfo
    from attribution_lib.sources import SymptomSet

SymptomPredicate = Callable[["SymptomSet[Any]", "Optional[JobInfo]"], bool]
