# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Batch collection: collect evidence once for many execution contexts.

Per-context collection issues one telemetry query per context. Fleet-scale
runs discover hundreds of contexts whose windows overlap, so the same Loki or
PromQL range is fetched again and again. Batch collection fixes that: the
orchestrator groups contexts whose end times fall inside one window, hands the
whole group to every collector that implements `collect_batch`, and the
collector fetches the union once and slices the result per context.

Collectors opt in by exposing `collect_batch(contexts)`, which returns
observations keyed by context id; a missing id means nothing was found for that
context, and raising aborts the batch for that collector only. Collectors
without `collect_batch` keep working per context inside the same batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from attribution_blueprint.types import ExecutionContext, assert_aware

COLLECTION_MODES = ("context", "batch")
DEFAULT_BATCH_WINDOW_HOURS = 2.0


def supports_batch_collection(collector: object) -> bool:
    """True when `collector` exposes a callable `collect_batch`."""
    return callable(getattr(collector, "collect_batch", None))


def time_bounded_batches(
    contexts: Sequence[ExecutionContext],
    *,
    window: timedelta | None,
    max_contexts: int | None = None,
) -> list[list[ExecutionContext]]:
    """Group contexts so each batch's end times span at most `window`.

    Contexts are ordered by `end_time` (then `start_time`, then `id`) so one
    telemetry query per batch covers a bounded trailing window. `window=None`
    disables time bounding; `max_contexts` caps the batch size. With both
    unset every context lands in one batch. `build_config` validates the
    settings once; this function trusts its arguments.
    """
    if not contexts:
        return []
    for ctx in contexts:
        assert_aware(ctx["end_time"], "end_time")
        assert_aware(ctx["start_time"], "start_time")
    ordered = sorted(contexts, key=lambda ctx: (ctx["end_time"], ctx["start_time"], str(ctx.get("id", ""))))
    batches: list[list[ExecutionContext]] = []
    current: list[ExecutionContext] = []
    for ctx in ordered:
        if current:
            span_exceeded = window is not None and ctx["end_time"] - current[0]["end_time"] > window
            size_exceeded = max_contexts is not None and len(current) >= max_contexts
            if span_exceeded or size_exceeded:
                batches.append(current)
                current = []
        current.append(ctx)
    if current:
        batches.append(current)
    return batches


__all__ = [
    "COLLECTION_MODES",
    "DEFAULT_BATCH_WINDOW_HOURS",
    "supports_batch_collection",
    "time_bounded_batches",
]
