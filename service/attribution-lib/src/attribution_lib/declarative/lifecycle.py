# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Lifecycle predicates + the {source, ctx} injection convention.

A grammar-reachable callable (an @expr_helper or a processor)
receives ``source`` and/or ``ctx`` injected as keyword arguments iff it declares
those parameter names. ``at_end_of_job()`` is the one engine-provided lifecycle
predicate: a nullary call from the author's view, implemented here as a
{source, ctx}-injecting built-in.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from attribution_lib.rule_types.definitions import RuleContext
    from attribution_lib.sources import Source

from attribution_lib.utils import is_event_at_end_of_job

# The only parameter names the engine injects into a grammar-reachable callable.
INJECTABLE_PARAMS = ("source", "ctx")


def injections_of(func: Callable[..., Any]) -> frozenset[str]:
    """The subset of {"source", "ctx"} that ``func`` declares as parameters."""
    params = inspect.signature(func).parameters
    return frozenset(p for p in INJECTABLE_PARAMS if p in params)


def _at_end_of_job(
    *,
    source: Source,
    ctx: RuleContext,
) -> bool:
    """True when this event's time is within the end-of-job window (utils: <=780s)."""
    if source.time is None or ctx.job_info is None:
        return False
    return is_event_at_end_of_job(source.time, ctx.job_info.job_end_time)


# Engine-provided lifecycle predicates, callable bare in a `when` (like a fixed
# function) and bound to the current {source, ctx} per match.
LIFECYCLE_PREDICATES = {"at_end_of_job": _at_end_of_job}
LIFECYCLE_INJECTIONS = {name: injections_of(func) for name, func in LIFECYCLE_PREDICATES.items()}
