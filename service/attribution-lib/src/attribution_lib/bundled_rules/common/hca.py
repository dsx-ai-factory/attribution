# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract and correlate local RDMA control-call failures."""

import re
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import EventStream
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rules import EventTriple, RefinerContext
from attribution_lib.sources import Source, SourceSet
from attribution_lib.utils import (
    NCCL_HOST_PID_TID_REGEX,
    _leading_explicit_applog_rank,
    convert_hostname_to_short_format,
)

_DEVICE_RE = re.compile(r"\b(?:mlx5_\d+|[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])\b")
_ERRNO_RE = re.compile(r"\berrno\s+(\d+)\b|^\s*(\d+)\b")
_WINDOW = timedelta(minutes=5)
_NIC_SYSLOG = {"irisc not responding", "detected bad NIC state"}
# Provider connectors own instance identity, lifecycle checks and event mapping.
# An instance-status check alone does not establish a hardware fault.
_PROVIDER_FAULTS = {"CSP Event NIC Fault", "CSP Event EC2 System Check Failed"}
_COMPOUND = "Corroborated local HCA command failure"


def parse_command_errno(error: str) -> int | None:
    """Read numeric errno first; accept known text-only NCCL wrapper variants."""
    match = _ERRNO_RE.search(error)
    if match:
        return int(match[1] or match[2])
    if "Resource temporarily unavailable" in error:
        return 11
    if "Connection timed out" in error:
        return 110
    return None


def _devices(datum: str) -> set[str]:
    return {match.group().lower() for match in _DEVICE_RE.finditer(datum)}


def command_source_node(node: str | None, source: Source) -> str | None:
    """Accept a declared node, a logged emitter or an explicitly mapped rank."""
    if source.declared_node:
        return convert_hostname_to_short_format(source.declared_node).lower()
    logged = NCCL_HOST_PID_TID_REGEX.search(source.datum)
    if logged:
        return convert_hostname_to_short_format(logged["node"]).lower()
    if _leading_explicit_applog_rank(source.datum) is not None and node:
        return convert_hostname_to_short_format(node).lower()
    return None


def command_failure_event(
    verb: str, errno: int, node: str | None, source: Source, device: str | None = None
) -> EventTriple:
    """Keep exact source/process/device context without assigning a remote fault."""
    devices = _devices(device or source.datum)
    process = NCCL_HOST_PID_TID_REGEX.search(source.datum)
    return EventTriple(
        type="Application.hca.command_failure",
        payload={
            "node": command_source_node(node, source),
            "verb": verb,
            "errno": errno,
            "device": next(iter(devices)) if len(devices) == 1 else None,
            "process": f"{process['pid']}:{process['tid']}" if process else None,
        },
        source=source,
    )


@dataclass(frozen=True)
class _Command:
    node: str | None
    verb: str
    errno: int
    device: str | None
    process: str | None
    source: Source
    device_source: Source | None = None


def _near(first: datetime | None, second: datetime | None) -> bool:
    return first is not None and second is not None and abs(first - second) <= _WINDOW


def _source_near(sources: list[Source], time: datetime | None) -> Source | None:
    """Find one proof in a sorted index, avoiding quadratic evidence expansion."""
    if time is None:
        return None
    index = bisect_left(sources, time - _WINDOW, key=lambda source: source.time)
    return sources[index] if index < len(sources) and _near(sources[index].time, time) else None


def _commands(stream: EventStream) -> list[_Command]:
    commands = [
        _Command(**payload, source=source)
        for _, _, payload, source in query_events_with_source(
            stream, stream.mgr.types.get_id("Application.hca.command_failure")
        )
    ]
    anchors = defaultdict(lambda: defaultdict(list))
    for command in commands:
        if command.device and command.node and command.process and command.source.time is not None:
            anchors[command.node, command.process][command.device].append(command.source)
    for devices in anchors.values():
        for sources in devices.values():
            sources.sort(key=lambda source: source.time)
    resolved = []
    for command in commands:
        if command.device is None and command.process:
            nearby = [
                (device, source)
                for device, sources in anchors[command.node, command.process].items()
                if (source := _source_near(sources, command.source.time)) is not None
            ]
            if len(nearby) == 1:
                command = replace(command, device=nearby[0][0], device_source=nearby[0][1])
        resolved.append(command)
    return resolved


def _corroboration(command: _Command, group) -> list[tuple[str, Source]]:
    """Use materialized NIC diagnoses; never repeat their correlation logic."""
    matches = []
    for symptom in _NIC_SYSLOG | _PROVIDER_FAULTS:
        if symptom not in group:
            continue
        for source in group[symptom]:
            if not _near(source.time, command.source.time):
                continue
            if symptom in _NIC_SYSLOG:
                if source.source_context.source != ObservationSource.SYSLOG:
                    continue
                if command.device is None or command.device not in _devices(source.datum):
                    continue
            elif source.source_context.source != ObservationSource.CSP_EVENT:
                continue
            matches.append((symptom, source))
    return matches


def attribute_local_hca_failures(context: RefinerContext) -> None:
    """Apply local attribution, registration guards and corroborated NIC outcomes.

    Existing per-node NIC correlation has already run. Only that node's active
    diagnosis can corroborate a local call. Unavailable identities, ambiguous
    devices and missing timestamps cannot satisfy a multi-verb guard.
    """
    stream, nodes = context.event_stream, context.my_bad_nodes
    if stream is None or nodes is None:
        return
    commands = _commands(stream)
    if not commands:
        return
    # Successful progress makes earlier application failures contextual. Do not
    # use them as proofs for a later isolated registration failure.
    mfu = [
        source.time
        for _, _, _, source in query_events_with_source(stream, stream.mgr.types.get_id("Application.mfu"))
        if source.time is not None
    ]
    last_progress = max(mfu, default=None)
    active = [
        command
        for command in commands
        if not (last_progress is not None and command.source.time is not None and command.source.time < last_progress)
    ]
    active_set = set(active)
    by_device = defaultdict(lambda: defaultdict(list))
    for command in active:
        if command.node and command.device and command.source.time is not None:
            by_device[command.node, command.device][command.verb].append(command.source)
    for verbs in by_device.values():
        for sources in verbs.values():
            sources.sort(key=lambda source: source.time)
    corroborating_sources = []
    for command in commands:
        symptom = f"Local HCA command failure ({command.verb} eno {command.errno})"
        if command.node is None:
            unattributed = SourceSet(symptom)
            if command in active_set and not (command.verb == "ibv_reg_mr" and command.errno == 11):
                unattributed.add(command.source)
            else:
                unattributed.add_interesting(command.source)
            nodes.job_level.add(unattributed)
            continue
        proof = []
        if command.device:
            proof = [
                source
                for verb, sources in by_device[command.node, command.device].items()
                if verb != command.verb and (source := _source_near(sources, command.source.time)) is not None
            ]
        group = nodes[command.node].attributions if command.node in nodes else None
        corroboration = _corroboration(command, group) if group is not None else []
        guarded = command.verb == "ibv_reg_mr" and command.errno == 11
        is_active = command in active_set
        guard_proof = proof or [
            source for _, source in corroboration if source.source_context.source == ObservationSource.SYSLOG
        ]
        qualifies = is_active and (not guarded or guard_proof)
        sources = SourceSet(symptom)
        if qualifies:
            sources.add(command.source)
            if guarded:
                for source in guard_proof:
                    sources.add(source)
        else:
            sources.add_interesting(command.source)
        if command.device_source is not None:
            sources.add_interesting(command.device_source)
        nodes.add([command.node], sources)
        if qualifies and corroboration:
            combined = SourceSet(_COMPOUND).add(command.source)
            if command.device_source is not None:
                combined.add(command.device_source)
            for matched_symptom, source in corroboration:
                combined.add(source)
                corroborating_sources.append((command.node, matched_symptom, source))
            nodes.add([command.node], combined)
    # Demote only proofs consumed by the compound, after all commands have been
    # examined. Independent nodes, episodes and outcomes remain available.
    for node, symptom, source in corroborating_sources:
        nodes[node].attributions[symptom].discard_selected(SourceSet(symptom).add(source))
