# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Syslog rules ported from attribution_lib/syslog.py node_single().

Rules are ordered to match their appearance in the original node_single function.
XID burst post-processing (formerly handle_xid_events) is now a filter rule
in filter_rules.py (HandleXidBurstEvents, EARLY priority).
"""

import json
import math
import re
from datetime import datetime
from typing import Dict, List, Optional

from common.syslog_helpers import SyslogAttributionParams
from common.xid_errors import (
    format_xid_with_subcode,
    get_catalog,
    parse_nvlink5_error,
)

xid_catalog = get_catalog()
from loguru import logger

from attribution_lib.events import DemotionTypes
from attribution_lib.rule_helpers import _bad, _interesting, _other_event
from attribution_lib.rules import (
    EventTriple,
    RuleContext,
    collector_def,
    declare_event_types,
    declare_rule_names,
    expr_helper,
    rows,
    rule_processor,
)
from attribution_lib.sources import Source, SourceSet
from attribution_lib.symptoms.symptom_enums import RegisteredSymptoms
from attribution_lib.xid_analyzer.nvidia_xid_analyzer import get_resolution_for_xid154

rule_author = "NVIDIA Attribution team"

# Event types from _bad()/_interesting() helpers and _other_event() calls
declare_event_types(
    "Attribution.node.bad",
    "Attribution.node.interesting",
    "XID_parser.pci_mapping",
    "XID_parser.burst",
    "XID_parser.actions",
    "FM_parser.lost_gpu_connection",
)

# Demotion provenance rule name used via `DemotionTypes.AFTER_MFU_DEMOTION`
# below. Not a registered rule, just a name recorded on the replacement
# event; see `attribution_lib.bundled_rules.filter_rules._sql_demote_events`.
declare_rule_names("xid-outside-of-job-execution")


@rows("xid")
def _xid_records():
    """Expose the XID catalog as a record table for declarative catalog_lookup
    (SyslogSXid projects each record's nice_name)."""
    return xid_catalog


@expr_helper("extract_fn_prefix")
def _extract_function_prefix(line: str, message: str) -> str:
    """Extract function name prefix if present, otherwise return message as-is."""
    pattern = rf"(\w+): {re.escape(message)}"
    match = re.search(pattern, line)
    if match:
        return f"{match.group(1)}: {message}"
    return message


# OOM symptom constants still referenced by the KubeletContainerOomKilled processor.
# (SyslogOomKill's system-service / other symptoms moved to its declarative pick_one.)
OOM_KILL_OF_SYSTEM_POD_SYMPTOM = RegisteredSymptoms.OOM_KILL_OF_SYSTEM_POD.value
USER_TASK_OOM_KILL_SYMPTOM = RegisteredSymptoms.USER_TASK_OOM_KILL.value
_SYSTEM_SERVICE_PROCESS_NAMES = frozenset(
    {
        "munged",
        "slurmd",
        "slurmstepd",
        "supervisord",
    }
)
_SYSTEM_SERVICE_PROCESS_PREFIXES = ("sssd",)
_SYSTEM_POD_CONTAINER_NAMES = ("slurmd",)
_JSON_QUOTE_RE = r'(?:\\"|")'
_CONTAINER_STATUSES_RE = rf"(?:containerStatuses|{_JSON_QUOTE_RE}containerStatuses{_JSON_QUOTE_RE})\s*:\s*\["
_CONTAINER_OOMKILLED_RE = (
    rf"{_CONTAINER_STATUSES_RE}"
    rf"(?:(?!{_JSON_QUOTE_RE}reason{_JSON_QUOTE_RE}:{_JSON_QUOTE_RE}OOMKilled{_JSON_QUOTE_RE}).)*?"
    rf"{_JSON_QUOTE_RE}name{_JSON_QUOTE_RE}:{_JSON_QUOTE_RE}(?P<container>[^\"\\]+){_JSON_QUOTE_RE}"
    rf"(?:(?!{_JSON_QUOTE_RE}name{_JSON_QUOTE_RE}).)*?"
    rf"{_JSON_QUOTE_RE}reason{_JSON_QUOTE_RE}:{_JSON_QUOTE_RE}OOMKilled{_JSON_QUOTE_RE}"
    # Optionally capture the terminated block's ``finishedAt`` (the real OOM time).
    # Stop at the next container's ``name`` so we never grab a sibling's timestamp.
    rf"(?:(?:(?!{_JSON_QUOTE_RE}name{_JSON_QUOTE_RE}).)*?"
    rf"{_JSON_QUOTE_RE}finishedAt{_JSON_QUOTE_RE}:{_JSON_QUOTE_RE}(?P<finished_at>[^\"\\]+){_JSON_QUOTE_RE})?"
)


@expr_helper("is_system_service_process")
def _is_system_service_process(process_name: str) -> bool:
    is_named_service = process_name in _SYSTEM_SERVICE_PROCESS_NAMES
    is_prefixed_service = process_name.startswith(_SYSTEM_SERVICE_PROCESS_PREFIXES)
    return is_named_service or is_prefixed_service


def _is_system_pod_container(container_name: str) -> bool:
    return container_name.lower() in _SYSTEM_POD_CONTAINER_NAMES


def _is_stale_kubelet_oom(finished_at: Optional[str], ctx: RuleContext) -> bool:
    """Return True when a kubelet OOMKilled status is a stale re-publication.

    The kubelet re-emits a container's termination state on every status sync, so
    a single OOM keeps appearing in syslog for the whole life of the restarted
    container (days, spanning many unrelated jobs). Attributing on the re-publish
    time tags that one event onto every job that lands on the node afterwards.
    Gate on the embedded ``finishedAt`` (the actual OOM time) instead: if the OOM
    completed before this job started, it is not this job's failure.

    Args:
        finished_at: ``finishedAt`` ISO-8601 timestamp from the OOMKilled
            terminated block, or None when the payload omits it.
        ctx: Rule context; ``ctx.job_info.job_start_time`` bounds the job.

    Returns:
        True if the OOM finished before the job started. False when the timestamp
        is absent, unparseable, or there is no job window to compare against
        (in which case we cannot gate and fall back to attributing).
    """
    if finished_at is None or ctx.job_info is None:
        return False
    start = ctx.job_info.job_start_time
    if start is None:
        return False
    try:
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        # A timezone-naive finished_at parses fine but can't be compared to the
        # tz-aware start, raising TypeError here rather than at the parse above.
        return finished < start
    except (ValueError, TypeError):
        return False


@rule_processor("syslog:SyslogXid")
def syslog_xid(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    pci_id = groups["pci_id"]
    xid = int(groups["xid"])
    xid_key = format_xid_with_subcode(xid, source.datum)

    events: List[EventTriple] = []

    # Store hostname:PCI ID for GPU correlation
    events.append(
        _other_event(
            "XID_parser.pci_mapping",
            {xid_key: f"{node.lower()}:{pci_id}"},
            source,
            "Xid pci extraction",
        )
    )

    if 144 <= xid <= 150:
        events.append(_other_event("XID_parser.burst", {"node": node}, source, "known-xid-event-burst"))
        return events

    extra = groups.get("extra")
    recovery_action = None
    investigatory_action = None

    if xid not in xid_catalog:
        logger.error(f"Encountered unknown xid {xid} in {node}")
        return events

    catalog = xid_catalog[xid]
    if catalog["type"] == "SXID":
        logger.error(f"Encountered sxid {xid} in XID syslog rule for {node}")
        return events

    nice_name = catalog["nice_name"]
    recovery_action = catalog["recovery_action"]
    catalog_investigatory_action = catalog["investigatory_action"]
    if isinstance(catalog_investigatory_action, str) or not math.isnan(catalog_investigatory_action):
        investigatory_action = catalog_investigatory_action

    if catalog["decode_workflow"] == "WORKFLOW_NVLINK5_ERR":
        nvlink5_info = parse_nvlink5_error(xid, extra)
        if nvlink5_info:
            nice_name = nvlink5_info["nice_name"]
            src = source.derived(rule="WORKFLOW_NVLINK5_ERR")
            recovery_action = SourceSet(nvlink5_info["recovery_action"]).add(src)
            investigatory_action = SourceSet(nvlink5_info["investigatory_action"]).add(src)
    elif catalog["decode_workflow"] == "XID_154":
        xid_154_action = get_resolution_for_xid154(source.datum)
        if xid_154_action is not None:
            events.append(
                _other_event(
                    "XID_parser.actions",
                    {"recovery_action": xid_154_action},
                    source,
                    "XID 154; GPU recovery action changed",
                )
            )
            return events
    else:
        payload: dict = {}
        if recovery_action is not None and recovery_action != "":
            payload["recovery_action"] = recovery_action
        if investigatory_action is not None and investigatory_action != "":
            payload["investigatory_action"] = investigatory_action
        if len(payload) > 0:
            events.append(_other_event("XID_parser.actions", payload, source, "XPA xid_catalog"))

    events.append(
        _bad(
            node,
            nice_name,
            source.derived(rule="known-xid-event"),
            demotion={DemotionTypes.AFTER_MFU_DEMOTION: "xid-outside-of-job-execution"},
        )
    )
    events.append(_other_event("XID_parser.burst", {"node": node}, source, "known-xid-event-burst"))

    return events


# Kubelet pod-status patches embed a ``containerStatuses`` array. A terminated
# container with ``reason: OOMKilled`` identifies the victim by name; the
# tempered-greedy ``name`` guard pairs each OOMKilled with the nearest preceding
# ``name`` field so multi-container patches resolve correctly. Match both plain
# JSON quotes and escaped quotes because Loki payloads vary by envelope.
@rule_processor("syslog:KubeletContainerOomKilled")
def kubelet_container_oomkilled(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    """Classify kubelet OOMKilled container status patches.

    The matched ``containerStatuses`` payload names the container killed by the
    OOM killer. System pod containers are infrastructure failures; other
    containers are treated as user task OOMs.

    The kubelet re-publishes a container's termination state on every status sync,
    so a stale OOM keeps matching long after the container recovered. The status's
    ``finishedAt`` is the real OOM time: drop the event when it completed before
    this job started so one old OOM is not re-attributed to every later job.
    """
    if _is_stale_kubelet_oom(groups.get("finished_at"), ctx):
        return []
    if _is_system_pod_container(groups["container"]):
        return [
            _bad(
                node,
                OOM_KILL_OF_SYSTEM_POD_SYMPTOM,
                source.derived(rule="kubelet-container-oomkilled-system-pod"),
            )
        ]
    return [
        _bad(
            node,
            USER_TASK_OOM_KILL_SYMPTOM,
            source.derived(rule="kubelet-container-oomkilled-user-task"),
        )
    ]


# --- Collector rules ---


def _fmt_xid_pci_mapping(raw):
    return {"xid_pci_mappings": json.dumps(raw, default=str, separators=(",", ": "))}


@collector_def(name="Collect.XidPciMapping", formatter=_fmt_xid_pci_mapping)
def _collect_xid_pci_mapping(event_stream, job_info, _params):
    return SyslogAttributionParams.collect_xid_pci_mapping(event_stream)


def _fmt_xidparser_actions(raw):
    recovery_actions, investigatory_actions = raw
    recovery = "; ".join(str(x) for x in sorted(recovery_actions.symptoms()))
    investigatory = "; ".join(str(x) for x in sorted(investigatory_actions.symptoms()))
    return {"recovery_actions": recovery, "investigatory_actions": investigatory}


@collector_def(name="Collect.XidParserActions", formatter=_fmt_xidparser_actions)
def _collect_xidparser_actions(event_stream, job_info, _params):
    return SyslogAttributionParams.collect_xidparser_actions(event_stream)
