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
from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rule_helpers import _bad, _interesting, _other_event
from attribution_lib.rules_engine import EventTriple, RuleContext, collector_def, declare_event_types, extractor_def
from attribution_lib.sources import Source, SourceSet
from attribution_lib.symptoms.symptom_enums import RegisteredSymptoms

rule_author = "FACT Team"
rule_sources = {ObsSource.SYSLOG}

# Event types from _bad()/_interesting() helpers and _other_event() calls
declare_event_types(
    "Attribution.node.bad",
    "Attribution.node.interesting",
    "XID_parser.pci_mapping",
    "XID_parser.burst",
    "XID_parser.actions",
    "FM_parser.lost_gpu_connection",
)


def _extract_function_prefix(line: str, message: str) -> str:
    """Extract function name prefix if present, otherwise return message as-is."""
    pattern = rf"(\w+): {re.escape(message)}"
    match = re.search(pattern, line)
    if match:
        return f"{match.group(1)}: {message}"
    return message


SYSTEM_SERVICE_OOM_KILL_SYMPTOM = RegisteredSymptoms.SYSTEM_SERVICE_OOM_KILL.value
OOM_KILL_OF_SYSTEM_POD_SYMPTOM = RegisteredSymptoms.OOM_KILL_OF_SYSTEM_POD.value
USER_TASK_OOM_KILL_SYMPTOM = RegisteredSymptoms.USER_TASK_OOM_KILL.value
OOM_KILL_OTHER_SYMPTOM = RegisteredSymptoms.OOM_KILL_OTHER.value
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


@extractor_def(
    name="SyslogXid",
    match_re=r"NVRM:\s+Xid \(PCI:(?P<pci_id>[0-9a-fA-F.:]+)\): (?P<xid>\d+),\s*(?P<extra>[A-Z_]+)?",
    derived_rules=[
        "single-gpu-xid",
        "known-xid-event",
        "known-xid-event-burst",
        "Xid pci extraction",
        "XPA xid_catalog",
        "XID 154; GPU recovery action changed",
        "WORKFLOW_NVLINK5_ERR",
    ],
)
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
    nice_name = catalog["nice_name"]
    recovery_action = catalog["resolution_datacenter"]
    if isinstance(catalog["resolution_investigatory"], str) or not math.isnan(catalog["resolution_investigatory"]):
        investigatory_action = catalog["resolution_investigatory"]

    if recovery_action == "WORKFLOW_NVLINK5_ERR":
        nvlink5_info = parse_nvlink5_error(xid, extra)
        if nvlink5_info:
            nice_name = nvlink5_info["nice_name"]
            src = source.derived(rule="WORKFLOW_NVLINK5_ERR")
            recovery_action = SourceSet(nvlink5_info["sheet_resolution"]).add(src)
            investigatory_action = SourceSet(nvlink5_info["sheet_investigatory_action"]).add(src)
    elif xid == 154:
        # NVRM: Xid (PCI:0009:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x4 (Drain and Reset)
        pattern_154 = r".*GPU recovery action changed from \w+ \((.*)\) to \w+ \((.*)\)"
        if match_154 := re.match(pattern_154, source.datum):
            events.append(
                _other_event(
                    "XID_parser.actions",
                    {"recovery_action": match_154.group(2)},
                    source,
                    "XID 154; GPU recovery action changed",
                )
            )
            return events
    else:
        payload: dict = {}
        if recovery_action is not None:
            payload["recovery_action"] = recovery_action
        if investigatory_action is not None:
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


@extractor_def(name="SyslogXidUnknown", match_re=r"NVRM:\s+Xid ", derived_rules=["unknown-xid-event"])
def syslog_xid_unknown(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    # Only fire if the PCI format didn't match (the main SyslogXid rule handles PCI format)
    if re.search(r"NVRM:\s+Xid \(PCI:[0-9a-fA-F.:]+\): \d+", source.datum):
        return []
    return [_interesting(node, "XID-unknown", source.derived(rule="unknown-xid-event"))]


@extractor_def(
    name="SyslogSXid",
    match_re=r" SXid \(PCI:[0-9a-fA-F.:]+\): (?P<sxid>\d+)",
    derived_rules=["known-sxid-event", "sxid-id-missing-in-catalog"],
)
def syslog_sxid(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    sxid = int(groups["sxid"])
    if sxid in xid_catalog:
        nice_name = xid_catalog[sxid]["nice_name"]
        return [_interesting(node, nice_name, source.derived(rule="known-sxid-event"))]
    else:
        return [_interesting(node, f"SXID-{sxid}", source.derived(rule="sxid-id-missing-in-catalog"))]


@extractor_def(name="SyslogSXidUnknown", match_re=r" SXid ", derived_rules=["unknown-sxid-event"])
def syslog_sxid_unknown(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    # Only fire if the PCI format didn't match (the main SyslogSXid rule handles PCI format)
    if re.search(r" SXid \(PCI:[0-9a-fA-F.:]+\): \d+", source.datum):
        return []
    return [_interesting(node, "SXID-unknown", source.derived(rule="unknown-sxid-event"))]


@extractor_def(
    name="SyslogGspFailure",
    match_re=r" NVRM: rpcRmApiAlloc_GSP: GspRmAlloc failed",
    derived_rules=["gsp-failure"],
)
def syslog_gsp_failure(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "GspRmAlloc failed", source.derived(rule="gsp-failure"))]


@extractor_def(
    name="SyslogNumaMemoryExhaustion",
    match_re=re.escape(
        "NVRM: nvAssertFailedNoLog: Assertion failed: (status == NV_OK) || ((status == NV_ERR_NO_MEMORY)"
    ),
    derived_rules=["numa-node-memory-exhaustion"],
)
def syslog_numa_memory_exhaustion(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "NUMA node memory exhaustion", source.derived(rule="numa-node-memory-exhaustion"))]


@extractor_def(
    name="SyslogEapFailure",
    match_re=r" CTRL-EVENT-EAP-FAILURE EAP authentication failed",
    derived_rules=["eap-failure"],
)
def syslog_eap_failure(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "EAP authentication failed", source.derived(rule="eap-failure"))]


@extractor_def(
    name="SyslogEapSuccess",
    match_re=re.escape("CTRL-EVENT-EAP-SUCCESS EAP authentication completed successfully"),
    derived_rules=["eap-success"],
)
def syslog_eap_success(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "EAP authentication recovery", source.derived(rule="eap-success"))]


@extractor_def(
    name="SyslogSystemPowerDown", match_re=re.escape("System is powering down"), derived_rules=["system-powering-down"]
)
def syslog_system_power_down(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "System is powering down", source.derived(rule="system-powering-down"))]


@extractor_def(
    name="SyslogOomKill",
    match_re=r"(?P<cgroup>Memory cgroup )?[Oo]ut of memory: Killed process \d+ \((?P<process_name>[^)]+)\)(?:.*?UID:(?P<uid>\d+))?",
    derived_rules=["user-task-oom-kill", "system-service-oom-kill", "oom-kill-other"],
)
def syslog_oom_kill(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Classify kernel OOM-killer kill messages.

    Matches both global OOM (``Out of memory: Killed process ...``) and cgroup
    OOM (``Memory cgroup out of memory: Killed process ...``). UID is captured
    from the trailing ``UID:N`` field of the kill line when present.

    Decision matrix:
        - System service by name (``slurmd``, ``slurmstepd``, ``munged``,
          ``supervisord``, ``sssd*``): emit ``OOM kill of system service``
          regardless of variant. UID is not used here because a root-owned
          process being OOM-killed is not by itself a failure (could be any
          system daemon hitting a limit).
        - User-owned (``UID > 0``) victim under cgroup OOM: emit
          ``OOM kill of user task`` — the SLURM-enforced per-task memory
          limit was hit.
        - Everything else (global OOM of a non-listed process, cgroup OOM of
          an unknown root process, kill line without UID): record as
          interesting via ``OOM kill (other)`` for context; does not drive
          outcome attribution.

    Args:
        groups: Named regex groups: ``cgroup`` (optional, present for cgroup
            OOM), ``process_name``, ``uid`` (optional).
        node: Hostname the syslog entry came from.
        source: Source attribution for the matched line.
        ctx: Rule context (unused here).

    Returns:
        Single-element list with one EventTriple: ``_bad`` for system service
        and cgroup-user-task kills, ``_interesting`` for everything else.
    """
    if _is_system_service_process(groups["process_name"]):
        return [_bad(node, SYSTEM_SERVICE_OOM_KILL_SYMPTOM, source.derived(rule="system-service-oom-kill"))]
    uid_str = groups.get("uid")
    if groups.get("cgroup") is not None and uid_str is not None and int(uid_str) > 0:
        return [_bad(node, USER_TASK_OOM_KILL_SYMPTOM, source.derived(rule="user-task-oom-kill"))]
    return [_interesting(node, OOM_KILL_OTHER_SYMPTOM, source.derived(rule="oom-kill-other"))]


# Match only the leaf ``kubepods*-pod<UUID>.slice`` so we get one event per
# kill (systemd reports the same kill three times: at ``kubepods.slice``,
# ``kubepods-<qos>.slice``, and the leaf pod slice). The slice line only
# identifies a pod cgroup, not the container name, so treat it as a user pod OOM
# unless a kubelet status patch identifies the killed container as infra.
@extractor_def(
    name="SyslogPodSliceOomKill",
    match_re=r"kubepods[\w\-]*-pod[0-9a-f_]+\.slice: A process of this unit has been killed by the OOM killer",
    derived_rules=["pod-slice-oom-kill"],
)
def syslog_pod_slice_oom_kill(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Attribute leaf Kubernetes pod-slice OOM kills as user task OOMs.

    The systemd slice line identifies only the pod cgroup. A later kubelet
    ``containerStatuses`` patch can override this via cascade pruning when it
    identifies the killed container as a system pod container.
    """
    return [_bad(node, USER_TASK_OOM_KILL_SYMPTOM, source.derived(rule="pod-slice-oom-kill"))]


# Kubelet pod-status patches embed a ``containerStatuses`` array. A terminated
# container with ``reason: OOMKilled`` identifies the victim by name; the
# tempered-greedy ``name`` guard pairs each OOMKilled with the nearest preceding
# ``name`` field so multi-container patches resolve correctly. Match both plain
# JSON quotes and escaped quotes because Loki payloads vary by envelope.
@extractor_def(
    name="KubeletContainerOomKilled",
    match_re=_CONTAINER_OOMKILLED_RE,
    derived_rules=["kubelet-container-oomkilled-system-pod", "kubelet-container-oomkilled-user-task"],
)
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


@extractor_def(
    name="SyslogFabricManagerNotPresent",
    match_re=re.escape("NV_ERR_FABRIC_MANAGER_NOT_PRESENT"),
    derived_rules=["fabric-manager-not-present"],
)
def syslog_fabric_manager_not_present(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "NV_ERR_FABRIC_MANAGER_NOT_PRESENT", source.derived(rule="fabric-manager-not-present"))]


@extractor_def(
    name="SyslogNmiWatchdog",
    match_re=re.escape("NMI watchdog: Watchdog detected hard LOCKUP"),
    derived_rules=["nmi-watchdog-hard-lockup"],
)
def syslog_nmi_watchdog(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "NMI watchdog hard LOCKUP", source.derived(rule="nmi-watchdog-hard-lockup"))]


@extractor_def(
    name="SyslogGeneralProtectionFault",
    match_re=re.escape("general protection fault"),
    derived_rules=["general-protection-fault"],
)
def syslog_general_protection_fault(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "general protection fault", source.derived(rule="general-protection-fault"))]


@extractor_def(
    name="SyslogKernelStackFrame",
    match_re=r"kernel stack frame pointer at .* in (?P<process>\S+):",
    derived_rules=["kernel-stack-frame-pointer"],
)
def syslog_kernel_stack_frame(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    process_name = groups["process"]
    return [
        _bad(
            node,
            f"bad kernel stack frame pointer in {process_name}",
            source.derived(rule="kernel-stack-frame-pointer"),
        )
    ]


@extractor_def(
    name="SyslogLustreError",
    match_re=re.escape("LustreError"),
    derived_rules=["lustre-error", "lustre-client-evicted", "lustre-error-non-fatal"],
)
def syslog_lustre_error(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    events: List[EventTriple] = []

    # LustreError is always created as bad with demotion flags
    events.append(
        _bad(
            node,
            "LustreError",
            source.derived(rule="lustre-error"),
            demotion={
                DemotionTypes.NOT_NEAR_WATCHDOG_EVENT_DEMOTION: "lustre-error-non-fatal-watchdog",
                DemotionTypes.BEFORE_DATALOADER_DEMOTION: "lustre-error-non-fatal-dataloader-exception",
            },
        )
    )

    client_evicted_regex = r".*This client was evicted by.*in progress operations using this service will fail\."
    if re.match(client_evicted_regex, source.datum):
        # Client eviction is always bad (separate symptom, no demotion)
        events.append(_bad(node, "Lustre Client Evicted", source.derived(rule="lustre-client-evicted")))
    else:
        # For non-eviction, also create interesting symptom
        events.append(_interesting(node, "(non-fatal) LustreError", source.derived(rule="lustre-error-non-fatal")))

    return events


@extractor_def(
    name="SyslogIscsiTimeout",
    match_re=re.escape("connection2:0: ping timeout"),
    derived_rules=["iscsi-detected-conn-error"],
)
def syslog_iscsi_timeout(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "iSCSI detected conn error", source.derived(rule="iscsi-detected-conn-error"))]


@extractor_def(
    name="SyslogIscsiConnError", match_re=re.escape("detected conn error"), derived_rules=["iscsi-detected-conn-error"]
)
def syslog_iscsi_conn_error(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "iSCSI detected conn error", source.derived(rule="iscsi-detected-conn-error"))]


@extractor_def(
    name="SyslogImexShutdown",
    match_re=re.escape("Abrupt nvidia-imex daemon shutdown detected, robust channel recovery invoked!"),
    derived_rules=["imex-daemon-shutdown"],
)
def syslog_imex_shutdown(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    symptom = _extract_function_prefix(
        source.datum, "Abrupt nvidia-imex daemon shutdown detected, robust channel recovery invoked!"
    )
    return [_bad(node, symptom, source.derived(rule="imex-daemon-shutdown"))]


@extractor_def(
    name="SyslogNvErr",
    match_re=r"\[(?P<error_code>NV_ERR_[A-Z_]+)\](?:.*?@ (?P<file_line>[\w\.]+))?",
    derived_rules=["NV_ERR_"],
)
def syslog_nv_err(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    error_code = groups["error_code"]
    file_line = groups.get("file_line") or ""
    symptom = f"{error_code} {file_line}"
    return [_bad(node, symptom, source.derived(rule="NV_ERR_"))]


@extractor_def(
    name="SyslogNvlinkStatusFailure",
    match_re=re.escape("Failed to collect nvlink status info!"),
    derived_rules=["collect-nvlink-status-failure"],
)
def syslog_nvlink_status_failure(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "Failed to collect nvlink status info", source.derived(rule="collect-nvlink-status-failure"))]


@extractor_def(
    name="SyslogNfsNotResponding",
    match_re=re.escape("not responding, still trying"),
    derived_rules=["nfs-not-responding"],
)
def syslog_nfs_not_responding(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "nfs not responding", source.derived(rule="nfs-not-responding"))]


@extractor_def(
    name="SyslogControllerStop",
    match_re=re.escape("Node was stopped by CMDaemon"),
    derived_rules=["node-stopped-by-controller"],
)
def syslog_controller_stop(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "Node stopped by controller", source.derived(rule="node-stopped-by-controller"))]


@extractor_def(
    name="SyslogRxDetectLinkMask",
    match_re=re.escape("Failed to update Rx Detect Link mask!"),
    derived_rules=["update-rx-detect-link-mask-failure"],
)
def syslog_rx_detect_link_mask(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    symptom = _extract_function_prefix(source.datum, "Failed to update Rx Detect Link mask!")
    return [_bad(node, symptom, source.derived(rule="update-rx-detect-link-mask-failure"))]


@extractor_def(
    name="SyslogImexServiceStop",
    match_re=re.escape("Stopping nvidia-imex.service"),
    derived_rules=["imex-service-stoppage"],
)
def syslog_imex_service_stop(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "Stopping nvidia-imex.service", source.derived(rule="imex-service-stoppage"))]


@extractor_def(
    name="ImexLostConnection",
    match_re=r"Connection lost to node \d+: (?P<hostname>[\w-]+)",
    derived_rules=["imex-lost-connection"],
)
def imex_lost_connection(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    affected_node = groups["hostname"]
    return [_interesting(affected_node, "IMEX lost connection", source.derived(rule="imex-lost-connection"))]


@extractor_def(
    name="SyslogFmLostConnectionToGpu",
    match_re=re.escape("Lost connection to GPU"),
    derived_rules=["fm-lost-connection-to-gpu", "fm-lost-gpu-tracking"],
)
def syslog_fm_lost_connection_to_gpu(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    events: List[EventTriple] = []

    # Extract GPU GUID if present, for per-node aggregation in the filter phase.
    # Log line format: "Lost connection to GPU on nodeId 13, ..., gpuGUID 0xfb843fc93d454d89, ..."
    guid_match = re.search(r"gpuGUID\s+(0x[0-9a-fA-F]+)", source.datum)
    if guid_match:
        events.append(
            _other_event(
                "FM_parser.lost_gpu_connection",
                {"node": node, "gpu_guid": guid_match.group(1)},
                source,
                "fm-lost-gpu-tracking",
            )
        )

    events.append(_bad(node, "Fabric manager lost connection to GPU", source.derived(rule="fm-lost-connection-to-gpu")))
    return events


@extractor_def(
    name="SyslogKernelInvalidMemoryAccess",
    match_re=re.escape("Unable to handle kernel"),
    derived_rules=["kernel-invalid-memory-access"],
)
def syslog_kernel_invalid_memory_access(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "kernel invalid memory access", source.derived(rule="kernel-invalid-memory-access"))]


# Generic kernel die() banner. Matches the "Oops:" colon form so it fires whether
# or not the printk-uptime prefix survives log forwarding, while excluding
# slurmd "Internal error: Oops handler"
# debug strings (no colon). Covers BUG_ON/BRK/SError paths when the more-specific
# "Unable to handle kernel" line is absent.
@extractor_def(
    name="SyslogKernelOops",
    match_re=r"Internal error: Oops:",
    derived_rules=["kernel-oops"],
)
def syslog_kernel_oops(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "kernel oops", source.derived(rule="kernel-oops"))]


@extractor_def(
    name="SyslogNvWarn",
    match_re=r"\[(?P<warning_code>NV_WARN_[A-Z_]+)\](?:.*?@ (?P<file_line>[\w\.]+))?",
    derived_rules=["NV_WARN_"],
)
def syslog_nv_warn(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    warning_code = groups["warning_code"]
    file_line = groups.get("file_line") or ""
    symptom = f"{warning_code} {file_line}"
    return [_interesting(node, symptom, source.derived(rule="NV_WARN_"))]


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
