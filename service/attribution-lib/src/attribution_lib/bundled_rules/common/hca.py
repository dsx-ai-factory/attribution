# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared extraction helpers for local RDMA control-call failures."""

import re

from attribution_lib.network_identity import network_source_identity, normalize_network_hostname
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules import EventTriple, RuleContext
from attribution_lib.sources import Source
from attribution_lib.utils import NCCL_HOST_PID_TID_REGEX, _leading_explicit_applog_rank

_DEVICE_RE = re.compile(r"\b(?:mlx5_\d+|[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])\b")
_ERRNO_RE = re.compile(r"\berrno\s+(\d+)\b")
_LEADING_ERRNO_RE = re.compile(r"^\s*(?:error\s+)?(\d+)\b")


def parse_command_errno(error: str) -> int | None:
    """Read numeric errno first; accept known text-only NCCL wrapper variants."""
    match = _ERRNO_RE.search(error) or _LEADING_ERRNO_RE.match(error)
    if match:
        return int(match[1])
    if "Resource temporarily unavailable" in error:
        return 11
    if "Connection timed out" in error:
        return 110
    return None


def command_devices(datum: str) -> set[str]:
    """Extract explicit device identifiers without inferring PCI-to-mlx5 mappings."""
    return {match.group().lower() for match in _DEVICE_RE.finditer(datum)}


def command_source_node(node: str | None, source: Source, ctx: RuleContext) -> str | None:
    """Prefer logged/mapped identity and metadata; resolve explicit ranks last."""
    identity = network_source_identity(source.datum, prefix_hostname=node)
    declared = normalize_network_hostname(source.declared_node) if source.declared_node else None
    if identity.hostname or declared:
        return identity.hostname or declared
    rank = _leading_explicit_applog_rank(source.datum)
    ranked_node = ctx.get_node_from_rank(rank) if rank is not None else None
    return normalize_network_hostname(ranked_node) if ranked_node else None


def command_failure_event(
    verb: str, errno: int, node: str | None, source: Source, device: str | None = None
) -> EventTriple:
    """Keep the resolved emitter and source/process/device context."""
    devices = command_devices(device or source.datum)
    process = NCCL_HOST_PID_TID_REGEX.search(source.datum)
    return EventTriple(
        type="Application.hca.command_failure",
        payload={
            "node": node,
            "verb": verb,
            "errno": errno,
            "device": next(iter(devices)) if len(devices) == 1 else None,
            "process": f"{process['pid']}:{process['tid']}" if process else None,
        },
        source=source,
    )


def command_symptom(verb: str, errno: int) -> str:
    return f"Local HCA command failure ({verb} eno {errno})"


def command_failure_events(
    verb: str, errno: int, node: str | None, source: Source, device: str | None = None
) -> list[EventTriple]:
    """Send candidates through the standard progress-demotion filter."""
    event = command_failure_event(verb, errno, node, source, device)
    emitter = event.payload["node"]
    symptom = command_symptom(verb, errno)
    attribution = (
        _bad(emitter, symptom, source, demotion="LocalHcaCommandFailure")
        if emitter is not None
        else _interesting(None, symptom, source)
    )
    return [event, attribution]
