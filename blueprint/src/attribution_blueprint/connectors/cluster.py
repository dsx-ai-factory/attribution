# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose the cluster value a telemetry connector renders into its queries."""

from __future__ import annotations

from typing import Mapping

# A context cluster comes from scheduler or event data, not from operator
# config, and query templates place it inside a quoted literal. Reject
# characters that could end or escape that literal instead of guessing the
# template's quoting style.
_UNSAFE_CHARACTERS = frozenset("\"'`\\")


def query_cluster(context: Mapping, configured: str | None, *, from_context: bool) -> str | None:
    """Return the context's cluster when enabled and present, else the configured cluster.

    Args:
        context: Execution context being collected.
        configured: Cluster value from connector configuration.
        from_context: Whether a non-empty context ``cluster`` takes precedence.

    Returns:
        The cluster value to render into queries.

    Raises:
        ValueError: If the context cluster contains a quote, backslash, or
            control character.
    """
    if not from_context:
        return configured
    cluster = context.get("cluster")
    if not isinstance(cluster, str) or cluster == "":
        return configured
    if any(char in _UNSAFE_CHARACTERS or ord(char) < 32 or ord(char) == 127 for char in cluster):
        raise ValueError(f"context cluster {cluster!r} contains a quote, backslash, or control character")
    return cluster
