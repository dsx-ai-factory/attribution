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
Applog rules ported from attribution_lib/applog.py node_single function.

Rules are ordered to match their appearance in the original node_single function,
followed by rules from applog_to_symptom.rules in their original dictionary order.

This module directly loads cuda_errors.json to generate CUDA error matching rules,
eliminating the need for applog_to_symptom.py and the dynamic dictionary building in core.py.
"""

import json
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import common.applog_prefilter  # noqa: F401 — triggers pre-filter registration
from common.nccl_helpers import get_status_name
from common.nccl_watchdog import NcclWatchdogData, NcclWatchdogParseState
from common.rankpulse import RankpulseParseState
from common.rankpulse_all_common import RankpulseData, format_rankpulse_all_for_display
from loguru import logger as _logger

import attribution_lib
from attribution_lib.attr_types import BadPairInfo
from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import BAD_PAIR_EVENT_TYPES, DemotionTypes, sql_placeholders
from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules_engine import (
    EventTriple,
    ExtractorAccumulator,
    RuleContext,
    collector_def,
    declare_event_types,
    declare_rule_names,
    extractor_accumulator_def,
    extractor_def,
)
from attribution_lib.sources import Source, SourceSet
from attribution_lib.utils import get_hostname

# User should not overwrite these NCCL parameters
NCCL_PARAMS_OVERWRITE_FORBIDDEN = [
    "NCCL_IB_TC",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "NCCL_IB_SL",
    "NCCL_CROSS_NIC",
    "NCCL_IB_ADDR_RANGE",
    "NCCL_IB_ADDR_FAMILY",
    "NCCL_IB_ROCE_VERSION_NUM",
    "NCCL_IB_FIFO_TC",
]

# Load CUDA error catalog directly from JSON
_CUDA_CATALOG_PATH = Path(__file__).parent / "cuda_errors.json"
_CUDA_ERRORS: dict[str, dict[str, str]] = {}


def _load_cuda_errors():
    """Load CUDA errors from JSON file once at module import time."""
    global _CUDA_ERRORS
    if _CUDA_ERRORS:
        return _CUDA_ERRORS

    with open(_CUDA_CATALOG_PATH) as f:
        error_table = json.load(f)
        for entry in error_table:
            description = entry.get("description", "")
            driver_name = entry.get("driver_name", "") or ""
            runtime_name = entry.get("runtime_name", "") or ""
            runtime_mapsto = entry.get("runtime_mapsto", "") or ""
            cnp_name = entry.get("cnp_name", "") or ""

            # prefer runtime_name, as documented in CUDA error reference
            nice_name = runtime_name or driver_name or runtime_mapsto or cnp_name or description

            if description and description not in ["no error", "driver shutting down"]:
                _CUDA_ERRORS[description] = {
                    "nice_name": nice_name,
                    "driver_name": driver_name,
                    "runtime_name": runtime_name,
                }

    # Customize nice_name of "out of memory" to be compatible with FACT history
    if "out of memory" in _CUDA_ERRORS:
        _CUDA_ERRORS["out of memory"]["nice_name"] = "cudaError out of memory"

    # Sort by length descending for better matching (longer patterns first)
    sorted_errors = dict(sorted(_CUDA_ERRORS.items(), key=lambda x: len(x[0]), reverse=True))
    _CUDA_ERRORS = sorted_errors
    return _CUDA_ERRORS


# Load errors at module import time
_load_cuda_errors()

GRACEFUL_MULTICAST_SKIP_MESSAGE = "Gracefully skipping multicast initialization"


def get_cuda_nice_name(key: str) -> Optional[str]:
    for error_id, item in _CUDA_ERRORS.items():
        if error_id in key:
            return item.get("nice_name")
    return None


rule_author = "FACT Team"
rule_sources = {ObsSource.APPLOG}

# Shared derived rule names for loop-generated rules where co-location isn't practical.
declare_rule_names(
    "alias else",
    "DataLoader Exception",
    "the process completes",
)

# Event types emitted by helpers (_bad, _bad_peers) and shared across multiple extractors.
declare_event_types(
    "Attribution.node.bad",
    "Attribution.node.bad_peers",
    "Application.bad_pair.complete",
    "Application.bad_pair.partial",
    "Application.pytorch.nccl_watchdog",
    "Application.dataloader.exception",
    "Application.megatron-lm.runtime_args",
)


def _bad_peers(
    node: str, peer: str, symptom: str, source: Source, demotion: Optional[str] = None, rule_name: Optional[str] = None
) -> EventTriple:
    """Create a bad node pair event.

    Args:
        node: Node hostname
        peer: Peer node hostname
        symptom: Symptom description
        source: Source attribution (rule name will be set by rule engine unless overridden)
        demotion: Demotion classification for AFTER_MFU_DEMOTION (not the rule name)
        rule_name: If provided, override the rule name in source
    """
    payload = {"node": node, "peer": peer, "symptom": symptom}
    if demotion:
        payload[DemotionTypes.AFTER_MFU_DEMOTION] = demotion
    # Override rule name if specified, otherwise let rule engine set it
    event_source = source.derived(rule=rule_name) if rule_name else source
    return EventTriple(type="Attribution.node.bad_peers", payload=payload, source=event_source)


def _keyword_to_rule_name(keyword: str) -> str:
    """Convert a keyword string to a CamelCase rule name."""
    # Remove special chars, split on spaces/underscores, capitalize each word
    cleaned = re.sub(r"[^a-zA-Z0-9\s_]", "", keyword)
    words = re.split(r"[\s_]+", cleaned)
    return "".join(word.capitalize() for word in words if word)


def _format_ib_error_message(error_code: int, vendor_err: int) -> str:
    """Format an IB work-completion error as a FACT symptom.

    Args:
        error_code: Numeric ibv_wc_status value.
        vendor_err: Numeric vendor error from the completion.

    Returns:
        A normalized IB Error symptom string.
    """
    error_code_name = get_status_name(error_code)
    msg = f"IB Error-{error_code}"
    if vendor_err != 0:
        msg += f" Vendor-{vendor_err}"
    if error_code_name is not None:
        msg += f" ({error_code_name})"
    return msg


def _path_is_user_owned(path: str) -> bool:
    """Paths under /home/ or /lustre/ are user-managed; attribution should be Fail User for permission/space issues there.

    Matches the existing `pyxis_enroot_mount_failure_user_event` heuristic.
    """
    return path.startswith("/home/") or path.startswith("/lustre/")


# Matches pyxis ephemeral container names like pyxis_1620149.0, pyxis_1620149_1620149.0.
_EPHEMERAL_PYXIS_NAME_RE = re.compile(r"pyxis_[0-9]+(_[0-9]+)?(\.[0-9]+)?$")


def _path_is_ephemeral_pyxis_container(path: str) -> bool:
    last = path.rstrip("/").rsplit("/", 1)[-1]
    return bool(_EPHEMERAL_PYXIS_NAME_RE.match(last))


# =============================================================================
# Rules from node_single (in original order) - these have complex logic
# =============================================================================


@extractor_def(
    name="MFU",
    match_re=r"\[(?P<isotime>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)\]\s+iteration\s+\d+/\s*\d+.*TFLOP/s/GPU\):\s*\d+\.\d+",
    derived_event_types=["Application.mfu"],
)
def mfu_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Extract MFU iteration timestamp."""
    time = datetime.fromisoformat(groups["isotime"]).replace(tzinfo=source.source_context.timezone)
    return [EventTriple(type="Application.mfu", payload={"datetime": time}, source=source.derived(time=time))]


@extractor_def(
    name="socketStartConnect",
    match_re=r"socketStartConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)",
)
def socket_start_connect_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    peer_node = get_hostname(groups["ip"])
    return [_bad_peers(node, peer_node, groups["reason"], source, demotion="socketStartconnect")]


@extractor_def(
    name="socketPollConnect",
    match_re=r"socketPollConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> returned \d+\((?P<reason>[^)]+)\) errno",
)
def socket_poll_connect_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    peer_node = get_hostname(groups["ip"])
    return [_bad_peers(node, peer_node, groups["reason"], source, demotion="socketPollConnect")]


@extractor_def(
    name="socketProgressOpt",
    match_re=r"socketProgressOpt: Call to recv from (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)",
)
def socket_progress_opt_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    peer_node = get_hostname(groups["ip"])
    return [_bad(peer_node, groups["reason"], source, demotion="socketProgressOpt")] if node != peer_node else []


@extractor_def(
    name="IPv4CannotBeRetrieved",
    match_re=r"IPv4 network addresses of \((?P<hostname>[^,]+), \d+\) cannot be retrieved",
)
def ipv4_cannot_be_retrieved_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    peer_node = get_hostname(groups["hostname"])
    return [_bad(peer_node, "network address cannot be retrieved", source, demotion="cannot be retrieved")]


@extractor_def(
    name="ClientSocketTimeout",
    match_re=r"client socket has timed out after \d+ms while trying to connect to \((?P<hostname>[^,]+), \d+\)",
)
def client_socket_timeout_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    peer_node = get_hostname(groups["hostname"])
    return [_bad(peer_node, "client socket has timed out", source, demotion="client socket has timed out")]


@extractor_def(
    name="AttributeTypeError",
    match_re=r"(?:AttributeError|TypeError): (?:'[\w\d]+' object has no attribute '[\w\d]+'|object of type '[\w\d]+' has no [\w\d()]+)",
)
def attribute_type_error_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "Python bug", source, demotion="AttributeError/TypeError")]


@extractor_def(name="KeyValidationError", match_re=r"KeyValidationError: ListConfig indices must be .+, not .+")
def key_validation_error_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "Python Exception", source, demotion="KeyValidationError")]


@extractor_def(name="completed with error", match_re=r"OFI Request .* completed with error\. RC: (?P<error_code>\d+)")
def efa_completed_with_error_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, f"EFA Error-{int(groups['error_code'])}", source, demotion="completed with error")]


@extractor_def(
    name="Got completion from peer (new)",
    match_re=r"Got completion from peer (?P<peer_ip>\d+\.\d+\.\d+\.\d+).* with \w+[= ]\s*(?:\w+\()?(?P<error_code>\d+)\)?.*vendor[ _]err[ =](?P<vendor_err>\d+).*localGid (?P<local_gid>::ffff:\d+\.\d+\.\d+\.\d+).*remoteGids\s*(?P<remote_gid>::ffff:\d+\.\d+\.\d+\.\d+).*hca (?P<hca>\S+)",
    derived_rules=["Got completion from peer"],
)
def got_completion_from_peer_new_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    try:
        peer_node = get_hostname(groups["peer_ip"])
    except (socket.gaierror, socket.herror, socket.timeout, OSError) as e:
        _logger.warning(f"Failed to get hostname for {groups['peer_ip']}: {e}")
        peer_node = groups["peer_ip"]

    msg = _format_ib_error_message(int(groups["error_code"]), int(groups["vendor_err"]))

    events = [
        _bad_peers(
            node, peer_node, msg, source, demotion="Got completion from peer", rule_name="Got completion from peer"
        )
    ]

    # Try to create complete bad_pair event with all fields
    local_dev_ip = groups["local_gid"][7:] if groups["local_gid"].startswith("::ffff:") else None
    remote_dev_ip = groups["remote_gid"][7:] if groups["remote_gid"].startswith("::ffff:") else None

    if local_dev_ip and remote_dev_ip:
        bad_pair_payload = {
            "local_node": node,
            "remote_node": peer_node,
            "local_dev_ip": local_dev_ip,
            "local_device": groups["hca"],
            "remote_dev_ip": remote_dev_ip,
        }
        events.append(
            EventTriple(
                type="Application.bad_pair.complete",
                payload=bad_pair_payload,
                source=source.derived(rule="Got completion from peer"),
            )
        )
    else:
        bad_pair_payload = {
            "local_node": node,
            "remote_node": peer_node,
        }
        events.append(
            EventTriple(
                type="Application.bad_pair.partial",
                payload=bad_pair_payload,
                source=source.derived(rule="Got completion from peer"),
            )
        )
    return events


@extractor_def(
    name="Got completion from peer (old)",
    match_re=r"Got completion from peer (?P<peer_ip>\d+\.\d+\.\d+\.\d+).* with \w+[= ]\s*(?:\w+\()?(?P<error_code>\d+)\)?.*vendor[ _]err[ =](?P<vendor_err>\d+)",
    derived_rules=["Got completion from peer"],
)
def got_completion_from_peer_old_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    try:
        peer_node = get_hostname(groups["peer_ip"])
    except (socket.gaierror, socket.herror, socket.timeout, OSError) as e:
        _logger.warning(f"Failed to get hostname for {groups['peer_ip']}: {e}")
        peer_node = groups["peer_ip"]

    msg = _format_ib_error_message(int(groups["error_code"]), int(groups["vendor_err"]))

    events = [
        _bad_peers(
            node, peer_node, msg, source, demotion="Got completion from peer", rule_name="Got completion from peer"
        )
    ]

    # Old format doesn't have GID fields, create partial bad_pair
    bad_pair_payload = {
        "local_node": node,
        "remote_node": peer_node,
    }
    events.append(
        EventTriple(
            type="Application.bad_pair.partial",
            payload=bad_pair_payload,
            source=source.derived(rule="Got completion from peer"),
        )
    )
    return events


@extractor_def(
    name="Got completion with error",
    match_re=r"Got completion with error (?P<error_code>\d+), opcode \d+, len \d+, vendor err (?P<vendor_err>\d+)",
)
def got_completion_with_error_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    """Create a node-level IB symptom for old NCCL completion warnings.

    Args:
        groups: Regex capture groups for error and vendor status.
        node: Local node where the line was observed.
        source: Source metadata for the matched log line.
        ctx: Rule context for the current attribution run.

    Returns:
        A bad-node event with the normalized IB completion symptom.
    """
    return [
        _bad(
            node,
            _format_ib_error_message(int(groups["error_code"]), int(groups["vendor_err"])),
            source,
            demotion="Got completion with error",
        )
    ]


@extractor_def(
    name="Irrecoverable NET/IB communicator error",
    match_re=(
        r"(?:\[[^\]]+\]\s+(?P<log_node>[^:\s]+):\d+:\d+\s+\[\d+\]\s+)?"
        r".*?"
        r"Irrecoverable (?P<direction>send|receive) communicator error on device (?P<device>\d+)\. "
        r"Reason: (?P<reason>.*?)\. Peer: (?P<peer_ip>[^<,\s]+)<\d+>,.*"
        r"status[= ]\s*(?:[A-Z0-9_]+\()?(?P<error_code>\d+)\)?.*"
        r"vendor[ _]err[ =](?P<vendor_err>\d+).*"
        r"localGid[: ](?P<local_gid>[0-9a-fA-F:.]+) remoteGid[: ](?P<remote_gid>[0-9a-fA-F:.]+),.*"
        r"hca[= ](?P<hca>\S+)"
    ),
    derived_rules=["Irrecoverable NET/IB communicator error"],
)
def irrecoverable_net_ib_communicator_error_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    """Create peer and bad-pair symptoms for irrecoverable NET/IB warnings.

    Args:
        groups: Regex capture groups for peer, GID, HCA, and completion status fields.
        node: Parser-supplied local node fallback.
        source: Source metadata for the matched log line.
        ctx: Rule context for the current attribution run.

    Returns:
        A bad-peer event plus a complete or partial bad-pair event.
    """
    local_node = groups.get("log_node")
    if local_node is None or local_node == "":
        local_node = node

    try:
        peer_node = get_hostname(groups["peer_ip"])
    except (socket.gaierror, socket.herror, socket.timeout, OSError) as e:
        _logger.warning(f"Failed to get hostname for {groups['peer_ip']}: {e}")
        peer_node = groups["peer_ip"]
    msg = _format_ib_error_message(int(groups["error_code"]), int(groups["vendor_err"]))
    rule_name = "Irrecoverable NET/IB communicator error"

    events = [
        _bad_peers(
            local_node,
            peer_node,
            msg,
            source,
            demotion=rule_name,
            rule_name=rule_name,
        )
    ]

    local_dev_ip = groups["local_gid"][7:] if groups["local_gid"].startswith("::ffff:") else None
    remote_dev_ip = groups["remote_gid"][7:] if groups["remote_gid"].startswith("::ffff:") else None
    if local_dev_ip is not None and remote_dev_ip is not None:
        events.append(
            EventTriple(
                type="Application.bad_pair.complete",
                payload={
                    "local_node": local_node,
                    "remote_node": peer_node,
                    "local_dev_ip": local_dev_ip,
                    "local_device": groups["hca"],
                    "remote_dev_ip": remote_dev_ip,
                },
                source=source.derived(rule=rule_name),
            )
        )
    else:
        events.append(
            EventTriple(
                type="Application.bad_pair.partial",
                payload={
                    "local_node": local_node,
                    "remote_node": peer_node,
                },
                source=source.derived(rule=rule_name),
            )
        )
    return events


@extractor_def(
    name="Call to ibv_modify_qp failed",
    match_re=r"Call to ibv_modify_qp failed with (?P<error_code>\d+) .* on dev (?P<local_hca>[^,\s]+).*local GID index \d+, local GID (?P<local_gid>[^,\s]+), remote GID (?P<remote_gid>[^,\s\n]+)",
    derived_rules=["Call to ibv_modify_qp failed"],
)
def ibv_modify_qp_failed_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    error_code = int(groups["error_code"])
    local_gid, remote_gid = groups["local_gid"], groups["remote_gid"]
    remote_ip = remote_gid[7:] if remote_gid.startswith("::ffff:") else remote_gid

    try:
        peer_node = get_hostname(remote_ip)
    except (socket.gaierror, socket.herror, socket.timeout, OSError) as e:
        _logger.warning(f"Failed to get hostname for {remote_ip}: {e}")
        peer_node = remote_ip

    msg = f"Call to ibv_modify_qp failed: eno {error_code}"
    if error_code == 22:
        msg += " invalid argument"
        if not all("::ffff:" in gid for gid in [local_gid, remote_gid]):
            msg += "(Mismatched GID)"
    elif error_code == 110:
        msg += " timeout"
        # Check if job script contains forbidden NCCL parameters
        if ctx.job_script_content and any(param in ctx.job_script_content for param in NCCL_PARAMS_OVERWRITE_FORBIDDEN):
            msg += "(Overwrite NCCL parameters)"

    events = [_bad_peers(node, peer_node, msg, source, demotion="Call to ibv_modify_qp failed")]

    # Try to create complete bad_pair event with all fields
    local_dev_ip = local_gid[7:] if local_gid.startswith("::ffff:") else None
    remote_dev_ip = remote_gid[7:] if remote_gid.startswith("::ffff:") else None

    if local_dev_ip and remote_dev_ip:
        bad_pair_payload = {
            "local_node": node,
            "remote_node": peer_node,
            "local_dev_ip": local_dev_ip,
            "local_device": groups.get("local_hca"),
            "remote_dev_ip": remote_dev_ip,
        }
        events.append(
            EventTriple(
                type="Application.bad_pair.complete",
                payload=bad_pair_payload,
                source=source.derived(rule="Call to ibv_modify_qp failed"),
            )
        )
    else:
        bad_pair_payload = {
            "local_node": node,
            "remote_node": peer_node,
        }
        events.append(
            EventTriple(
                type="Application.bad_pair.partial",
                payload=bad_pair_payload,
                source=source.derived(rule="Call to ibv_modify_qp failed"),
            )
        )
    return events


@extractor_def(
    name="Connection closed by remote peer", match_re=r"Connection closed by remote peer (?P<hostname>\S+)[^<]*<"
)
def connection_closed_by_remote_peer_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    peer_node = get_hostname(groups["hostname"])
    short_peer = peer_node.split(".")[0]
    if attribution_lib.ignored_hostname_prefixes:
        if any(short_peer.startswith(p) for p in attribution_lib.ignored_hostname_prefixes):
            return []
    return [_bad(peer_node, "Connection closed", source, demotion="Connection closed by remote peer")]


@extractor_def(name="BusErrorSignal", match_re=r"Caught signal \d+ \(Bus error: nonexistent physical address\)")
def bus_error_signal_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "bus error", source, demotion="Bus error: nonexistent physical address")]


# Generic "bus error" keyword match (from applog_to_symptom.rules)
@extractor_def(name="BusErrorKeyword", match_re=r"bus error")
def bus_error_keyword_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    payload = {"node": node, "symptom": "bus error"}
    payload[DemotionTypes.AFTER_MFU_DEMOTION] = "alias else"
    return [EventTriple(type="Attribution.node.bad", payload=payload, source=source.derived(rule="alias else"))]


@extractor_def(name="AddressAlreadyInUse", match_re=r"Address already in use", derived_rules=["address-in-use"])
def address_in_use_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    if "ras/client_support.cc" in source.datum and "NCCL WARN Call to bind failed:" in source.datum:
        return []
    payload = {"node": node, "symptom": "Address already in use"}
    payload[DemotionTypes.AFTER_MFU_DEMOTION] = "address-in-use"
    return [EventTriple(type="Attribution.node.bad", payload=payload, source=source.derived(rule="address-in-use"))]


# =============================================================================
# Rules with special logic (CUDA catalog lookup, peer resolution, etc.)
# =============================================================================


@extractor_def(name="NCCLWarnCudaFailure", match_re=r"NCCL WARN Cuda failure")
def nccl_warn_cuda_failure_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    """Handle NCCL WARN Cuda failure - looks up CUDA error in catalog."""
    nice_name = get_cuda_nice_name(source.datum)
    if nice_name:
        # Determine rule name and demotion based on nice_name pattern
        if nice_name.startswith("cuda"):
            rule_name = nice_name
            demotion = nice_name
        else:
            rule_name = "alias else"
            demotion = "alias else"

        payload = {"node": node, "symptom": nice_name}
        payload[DemotionTypes.AFTER_MFU_DEMOTION] = demotion
        return [
            EventTriple(
                type="Attribution.node.bad",
                payload=payload,
                source=source.derived(rule=rule_name),
            )
        ]
    return []


WATCHDOG_REGEX = re.compile(
    r".*\s(\d+:\d+:\d+).*Watchdog caught collective operation timeout: WorkNCCL\(SeqNum=(\d+), OpType=(\w+), NumelIn=(\d+), NumelOut=(\d+), Timeout\(ms\)=(\d+)\) ran for (\d+) milliseconds before timing out."
)


def _pytorch_watchdog_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Handle PyTorch NCCL watchdog events with CUDA error lookup.

    Also checks if this matches the full watchdog timeout pattern and emits
    the Application.pytorch.nccl_watchdog event if so.
    """
    events = []

    # Check if this matches the full watchdog timeout pattern using WatchdogFoundValues.REGEX
    if match := WATCHDOG_REGEX.match(source.datum):
        # Extract watchdog metadata and emit the watchdog event
        timeout_seconds = int(match.group(7)) / 1000
        watchdog_event = EventTriple(
            type="Application.pytorch.nccl_watchdog",
            payload={"raw_error_time": match.group(1), "timeout_seconds": timeout_seconds},
            source=source,
        )
        events.append(watchdog_event)

    # Also emit the bad node event (with CUDA error lookup if present)
    nice_name = get_cuda_nice_name(source.datum)
    if nice_name:
        # Determine rule name and demotion based on nice_name pattern
        if nice_name.startswith("cuda"):
            rule_name = nice_name
            demotion = nice_name
        else:
            rule_name = "alias else"
            demotion = "alias else"

        payload = {"node": node, "symptom": nice_name}
        payload[DemotionTypes.AFTER_MFU_DEMOTION] = demotion
        events.append(
            EventTriple(
                type="Attribution.node.bad",
                payload=payload,
                source=source.derived(rule=rule_name),
            )
        )
    else:
        payload = {"node": node, "symptom": "PyTorch NCCL watchdog"}
        payload[DemotionTypes.AFTER_MFU_DEMOTION] = "alias else"
        events.append(
            EventTriple(
                type="Attribution.node.bad",
                payload=payload,
                source=source.derived(rule="alias else"),
            )
        )

    return events


@extractor_def(name="pushrecv", match_re=r"pushrecv")
def pushrecv_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Handle pushrecv stuck error with peer rank resolution."""
    for pattern in [r"(\d+) from (\d+)\] pushrecv:", r"pushrecv.*dst:\d+.*src:(\d+)"]:
        match = re.search(pattern, source.datum)
        if match:
            peer_rank = int(match.group(2) if "from" in pattern else match.group(1))
            peer_node = ctx.get_node_from_rank(peer_rank)
            if peer_node:
                return [_bad_peers(node, peer_node, "pushrecv", source, demotion="pushrecv")]
    return [_bad(node, "pushrecv", source, demotion="pushrecv")]


def _dataloader_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Helper to emit both dataloader exception event and bad node event."""
    payload = {"node": node, "symptom": "DataLoader Exception"}
    payload[DemotionTypes.AFTER_MFU_DEMOTION] = "DataLoader Exception"
    return [
        EventTriple(type="Application.dataloader.exception", payload={}, source=source),
        EventTriple(
            type="Attribution.node.bad",
            payload=payload,
            source=source.derived(rule="DataLoader Exception"),
        ),
    ]


def _process_completes_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "the process completes", source, demotion="the process completes")]


# Pyxis mount failure rules (need different symptoms based on path)
@extractor_def(
    name="PyxisEnrootMountFailureSystem",
    match_re=r"enroot-mount: failed to mount:\s*(?!/home/)(?!/lustre/)\S+\s+at\s+",
)
def pyxis_enroot_mount_failure_system_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if "No such file or directory" in source.datum:
        return [_bad(node, "pyxis: enroot mount path not found", source)]
    return [_bad(node, "pyxis: enroot mount failure from system directory", source)]


@extractor_def(
    name="PyxisEnrootMountFailureUser",
    match_re=r"enroot-mount: failed to mount:\s*(?P<path>(?:/home/|/lustre/)\S+)\s+at\s+",
)
def pyxis_enroot_mount_failure_user_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "pyxis: enroot mount failure from user directory", source)]


_DOCKER_HUB_FALLBACK_PATH_RE = re.compile(r"registry-1\.docker\.io/v2/([^/\s]+)/")


@extractor_def(
    name="EnrootRegistryHttpError",
    match_re=r"\[ERROR\]\s+URL\s+\S+\s+returned error code:\s*(?P<code>\d{3})\b",
)
def enroot_registry_http_error(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    code = groups.get("code", "")
    if code in ("401", "403"):
        # The edge case of hostname misparsed into the /v2/ repo path
        # is detected by a dotted hostname in the path segment.
        m = _DOCKER_HUB_FALLBACK_PATH_RE.search(source.datum)
        if m and "." in m.group(1):
            return [_bad(node, "enroot: missing credentials (docker hub fallback)", source)]
        return [_bad(node, "enroot: registry auth denied", source)]
    if code == "404":
        return [_bad(node, "enroot: registry image not found", source)]
    return []


# Path-aware "Not enough space left in <path>".
@extractor_def(
    name="EnrootNotEnoughSpace",
    match_re=r"\[ERROR\]\s+Not enough space left in\s+(?P<path>\S+)",
)
def enroot_not_enough_space(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    path = groups.get("path", "")
    if _path_is_user_owned(path):
        return [_bad(node, "enroot: not enough space (user path)", source)]
    return [_bad(node, "enroot: not enough space (system path)", source)]


# "File already exists" only emits a specific symptom for the ephemeral
# container-name variant where the user cannot have caused a collision.
@extractor_def(
    name="EnrootFileAlreadyExists",
    match_re=r"\[ERROR\]\s+File already exists:\s+(?P<path>\S+)",
)
def enroot_file_already_exists(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    path = groups.get("path", "")
    if _path_is_ephemeral_pyxis_container(path):
        return [_bad(node, "pyxis: file already exists (ephemeral)", source)]
    return []


# Path-aware mkdir permission denied (enroot init phase, before extraction).
@extractor_def(
    name="PyxisMkdirPermissionDenied",
    match_re=r"mkdir: cannot create directory '(?P<path>[^']+)': Permission denied",
)
def pyxis_mkdir_permission_denied(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    path = groups.get("path", "")
    if _path_is_user_owned(path):
        return [_bad(node, "pyxis: mkdir permission denied (user path)", source)]
    return [_bad(node, "pyxis: mkdir permission denied (system path)", source)]


# Path-aware unsquashfs dir_scan permission denied (extraction phase).
@extractor_def(
    name="UnsquashfsDirScanPermissionDenied",
    match_re=r"FATAL ERROR: dir_scan: failed to make directory (?P<path>\S+?), because Permission denied",
)
def unsquashfs_dir_scan_permission_denied(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    path = groups.get("path", "")
    if _path_is_user_owned(path):
        return [_bad(node, "unsquashfs: dir_scan permission denied (user path)", source)]
    return [_bad(node, "unsquashfs: dir_scan permission denied (system path)", source)]


# unsquashfs killed by OOM.
@extractor_def(
    name="UnsquashfsOomKilled",
    match_re=r"runtime\.sh: line \d+:\s+\d+\s+Killed\s+unsquashfs",
)
def unsquashfs_oom_killed(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "unsquashfs: killed by OOM", source)]


# mksquashfs disk full during import.
@extractor_def(
    name="MksquashfsDiskFull",
    match_re=r"FATAL ERROR: Probably out of space on output filesystem",
)
def mksquashfs_disk_full(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "mksquashfs: disk full (system path)", source)]


# Path-aware unsquashfs writer failure (disk full during extraction).
@extractor_def(
    name="UnsquashfsWriteFailed",
    match_re=r"FATAL ERROR:writer: failed to write file (?P<path>\S+)",
)
def unsquashfs_write_failed(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    path = groups.get("path", "")
    if _path_is_user_owned(path):
        return [_bad(node, "unsquashfs: write failed (user path)", source)]
    return [_bad(node, "unsquashfs: write failed (system path)", source)]


# Namespace creation failure (user/mount/network/IPC/PID/cgroup).
@extractor_def(
    name="PyxisNamespaceCreateFailed",
    match_re=r"failed to create (?:user|mount|network|IPC|PID|cgroup) namespace",
)
def pyxis_namespace_create_failed(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "pyxis: failed to create namespace", source)]


# Any enroot C binary (aufs2ovlfs/mount/nsenter/switchroot) failing.
@extractor_def(
    name="EnrootCapSetFailed",
    match_re=r"enroot-[a-z0-9]+: failed to (?:set|get) capabilities:",
)
def enroot_cap_set_failed(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "enroot: failed to set capabilities", source)]


# pyxis container start failure (typically superseded by a specific
# enroot error).
@extractor_def(
    name="PyxisContainerStartNonzero",
    match_re=r"pyxis: container start failed with error code:\s*\d+",
)
def pyxis_container_start_nonzero(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "pyxis: container start nonzero exit", source)]


# chdir failure for user-supplied --container-workdir pointing at a path that
# doesn't exist in the image. The negative lookahead ensures this doesn't match
# the `pyxis: couldn't chdir to container cwd <path>` line.
@extractor_def(
    name="PyxisCouldntChdirWorkdir",
    match_re=r"pyxis: couldn't chdir to (?!container cwd)",
)
def pyxis_couldnt_chdir_workdir(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "pyxis: couldn't chdir to container workdir", source)]


# Conflicting pyxis options (any `both --<flag> and --<other> were specified`).
@extractor_def(
    name="PyxisConflictingOptions",
    match_re=r"pyxis: both --container-\S+ and --(?:no-)?container-\S+ were specified",
)
def pyxis_conflicting_options(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    return [_bad(node, "pyxis: conflicting options", source)]


# srun-level "Invalid --container-<flag> argument".
# catch-all for any pyxis SPANK arg-validation failure.
@extractor_def(
    name="SrunInvalidPyxisArgument",
    match_re=r"srun: error: Invalid --(?:no-)?container-\S+ argument:",
)
def srun_invalid_pyxis_argument(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "srun: invalid pyxis argument", source)]


# Older-pyxis cancellation variant: `pyxis: could not wait for child <pid>:
# Interrupted system call` — EINTR on waitpid almost always means a cancellation
# signal arrived on older pyxis versions (which lack a dedicated SIGTERM handler).
@extractor_def(
    name="PyxisCouldNotWaitChildInterrupted",
    match_re=r"pyxis: could not wait for child \d+: Interrupted system call",
)
def pyxis_could_not_wait_child_interrupted(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_bad(node, "pyxis: cancelled by SIGTERM", source)]


# PyTorch NCCL watchdog rules
for keyword in [
    "NCCL watchdog thread terminated",
    "Exception raised from ncclCommWatchdog",
    "ProcessGroupNCCL's watchdog got stuck",
    "detected by watchdog at worker",
    "Watchdog caught collective operation",
]:
    extractor_def(name=_keyword_to_rule_name(keyword), match_re=re.escape(keyword), processor=_pytorch_watchdog_event)

# Additional watchdog pattern
extractor_def(
    name="ProcessGroupWatchdogTerminated",
    match_re=r"Process group watchdog thread terminated",
    processor=_pytorch_watchdog_event,
)


# Marker used for multicast NV_ERR suppression during refinement.
@extractor_def(
    name="GracefullySkippingMulticastInitialization",
    match_re=re.escape(GRACEFUL_MULTICAST_SKIP_MESSAGE),
)
def gracefully_skipping_multicast_initialization_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    _logger.info("Applog marker detected: Gracefully skipping multicast initialization")
    return []


# DataLoader exception rules
for keyword in ["Error: DataLoader worker", "Error in DataLoader worker", "DatasetGenerationError"]:
    extractor_def(name=_keyword_to_rule_name(keyword), match_re=re.escape(keyword), processor=_dataloader_event)

# "the process completes" rules
for keyword in [
    "exiting program at",
    "exiting program after",
    "Inference job is done",
    "Start calculating task metric",
    "Done waiting for other agents",
    "Writing temporary files back into original files",
    "Main process ended cleanly",
    "finish all lines",
    "end the job",
    "Validating solutions: 100%",
    "test results at the end of training",
]:
    extractor_def(
        name=_keyword_to_rule_name(keyword),
        match_re=re.escape(keyword),
        processor=lambda groups, node, source, ctx: [
            _bad(
                node,
                "the process completes",
                source.derived(rule="the process completes"),
                demotion="the process completes",
            )
        ],
    )

# =============================================================================
# CUDA Error Rules (auto-generated from cuda_errors.json)
# =============================================================================


def _cuda_error_processor(symptom: str):
    """Create a processor for a specific CUDA error symptom."""

    def processor(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
        # Determine rule name and demotion based on symptom name pattern
        if symptom.startswith("cuda"):
            rule_name = symptom
            demotion = symptom
        else:
            rule_name = "alias else"
            demotion = "alias else"

        payload = {"node": node, "symptom": symptom}
        payload[DemotionTypes.AFTER_MFU_DEMOTION] = demotion
        return [
            EventTriple(
                type="Attribution.node.bad",
                payload=payload,
                source=source.derived(rule=rule_name),
            )
        ]

    return processor


# Generate rules for all CUDA error patterns
for error_desc, error_info in _CUDA_ERRORS.items():
    nice_name = error_info["nice_name"]

    for idx, pattern_template in enumerate(
        [
            "Cuda failure '{}'",
            "CUDA error: {}",
            "Cuda Runtime ({})",
            "CUDA {}",
        ]
    ):
        pattern = pattern_template.format(error_desc)
        rule_name = f"CudaError_{nice_name}_{idx}"
        extractor_def(
            name=rule_name,
            match_re=re.escape(pattern),
            processor=_cuda_error_processor(nice_name),
        )

# cudaError-branch findings now credit their own nice_name; register so source.rule resolves.
_CUDA_NICE_NAMES = {info["nice_name"] for info in _CUDA_ERRORS.values()}
declare_rule_names(*sorted({n for n in _CUDA_NICE_NAMES if n.startswith("cuda")}))


# Generic keyword rule for "No such file or directory"
@extractor_def(name="No such file or directory", match_re=r"No such file or directory")
def no_such_file_or_directory_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if "error: pyxis:" in source.datum:
        return []
    payload = {"node": node, "symptom": "No such file or directory"}
    payload[DemotionTypes.AFTER_MFU_DEMOTION] = "alias else"
    return [EventTriple(type="Attribution.node.bad", payload=payload, source=source.derived(rule="alias else"))]


@extractor_def(name="Gloo connectFullMesh failed", match_re="Gloo connectFullMesh failed")
def gloo_connectfullmesh_failed(
    _groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    return [_interesting(node, "Gloo connectFullMesh failed", source)]


# =============================================================================
# Generic keyword rules from applog_to_symptom.py
# =============================================================================

# Additional generic keyword rules.
# These tags only *classify* which rows credit their own symptom (the finding)
# instead of a shared placeholder bucket — they are no longer credited as rule
# names themselves. "alias else" / "the process completes" rows are deferred
# (scrub-coupled) and keep crediting their bucket name.
_RETIRE_NOW_BUCKETS = {"cudaError", "io-error"}

_GENERIC_KEYWORD_RULES = [
    ("Unexpected error from cudaGetDeviceCount()", None, "cudaError"),
    ("local catastrophic error", None, "cudaError"),
    ("ibv_open_device failed", None, "cudaError"),
    ("Segmentation fault", None, "alias else"),
    ("terminate called", None, "alias else"),
    ("No CUDA runtime is found", None, "alias else"),
    ("CE deadlock DETECTED", None, "alias else"),
    ("Disk quota exceeded", None, "alias else"),
    ("ncclUnhandledCudaError", None, "alias else"),
    ("ncclSystemError", None, "alias else"),
    ("ncclInternalError", None, "alias else"),
    ("ncclInvalidArgument", None, "alias else"),
    ("ncclInvalidUsage", None, "alias else"),
    ("ncclRemoteError", None, "alias else"),
    ("ncclTimeout", None, "alias else"),
    ("Unconvertible NCCL type", None, "alias else"),
    ("BrokenPipeError", None, "alias else"),
    ("megatron.core.dist_checkpointing.core.CheckpointingException", None, "alias else"),
    ("wandb.Error", None, "alias else"),
    ("distributed.comm.core.CommClosedError", None, "alias else"),
    (
        "Exception: Failed to write dataset materials to the data cache directory",
        "Dataset cache write failure",
        "alias else",
    ),
    ("file write failed", None, "alias else"),
    ("failed to import docker image", None, "alias else"),
    ("pyxis: couldn't start container", None, "alias else"),
    ("error: pyxis:     [ERROR] No such file or directory", "pyxis: cannot find container file", "cudaError"),
    ("98-nvidia.sh exited with return code", "pyxis: nvidia hook failed", "alias else"),
    ("10-cgroups.sh exited with return code", "pyxis: cgroup hook failed", "alias else"),
    ("nvidia-container-cli: mount error", "pyxis: nvidia-container-cli mount error", "alias else"),
    ("nvidia-container-cli: initialization error", "pyxis: nvidia-container-cli init error", "alias else"),
    ("nvidia-container-cli: device error", "pyxis: nvidia-container-cli device error", "alias else"),
    ("Unknown NVIDIA driver capability", "pyxis: nvidia unknown driver capability", "alias else"),
    ("nvidia-container-cli: requirement error", "pyxis: nvidia-container-cli requirement error", "alias else"),
    ("nvidia-container-cli: ldcache error", "pyxis: nvidia-container-cli ldcache error", "alias else"),
    ("nvidia-container-cli: detection error", "pyxis: nvidia-container-cli detection error", "alias else"),
    ("[ERROR] Command not found: nvidia-container-cli", "pyxis: nvidia-container-cli not installed", "alias else"),
    ("99-mellanox.sh exited with return code", "pyxis: mellanox hook failed", "alias else"),
    ("[ERROR] Invalid image reference:", "enroot: invalid image reference", "alias else"),
    ("[ERROR] Could not retrieve digest from", "enroot: digest error", "alias else"),
    ("[ERROR] Could not parse digest information from", "enroot: digest error", "alias else"),
    ("curl: (56) CONNECT tunnel failed", "enroot: curl tunnel failure", "alias else"),
    ("curl: (28) Operation timed out", "enroot: curl transfer timeout", "alias else"),
    ("curl: (52) Empty reply from server", "enroot: curl empty reply from server", "alias else"),
    ("curl: (6) Could not resolve host:", "enroot: DNS resolution failed", "alias else"),
    ("[ERROR] Could not process JSON input", "enroot: JSON parse failed", "alias else"),
    ("/*stdin*\\", "enroot: corrupted cache layer", "alias else"),
    ("[ERROR] Could not acquire rootfs lock", "enroot: rootfs lock contention", "alias else"),
    ("[ERROR] Invalid image format:", "enroot: invalid image format", "alias else"),
    ("FATAL ERROR: File system corruption detected", "unsquashfs: file system corruption", "alias else"),
    ("enroot-switchroot: failed to execute", "enroot: switchroot exec failed", "alias else"),
    (
        "pyxis: --container-image specified multiple times",
        "pyxis: --container-image specified multiple times",
        "alias else",
    ),
    ("pyxis: mount source must be", "pyxis: mount source invalid", "alias else"),
    (
        '"create" flag was passed to --container-name but the container already exists',
        "pyxis: container name already exists",
        "alias else",
    ),
    ("pyxis: failed to create container filesystem", "pyxis: failed to create container filesystem", "alias else"),
    ("pyxis: received SIGTERM, forwarding to child", "pyxis: cancelled by SIGTERM", "alias else"),
    ("pyxis: container start error: Interrupted system call", "pyxis: cancelled by SIGTERM", "alias else"),
    ("container was not started successfully by another task", "pyxis: cascade from peer task", "alias else"),
    ("error: pyxis", None, "alias else"),
    ("RuntimeError: Numpy is not available", None, "alias else"),
    ("ValueError: Tasks not found", None, "alias else"),
    ("Stopping cuda profiler", None, "alias else"),
    ("JSONDecodeError", None, "alias else"),
    ("Stale file handle", None, "alias else"),
    (" port error", "link down(east-west)", "cudaError"),
    ("DUE TO TIME LIMIT", "Cancelled by Timeout", "alias else"),
    ("DUE TO TASK FAILURE", "Cancelled due to task failure", "alias else"),
    ("GPU is lost", "GPU is lost reported by slurm", "cudaError"),
    ("botocore.exceptions", "S3 Exception", "alias else"),
    ("s3_indexed_dataset.py", "S3 Exception", "alias else"),
    ("S3Error: Exception", "S3 Exception", "alias else"),
    ("device is incompatible with the local", "NCCL IB device incompatibility", "alias else"),
    ("NCCL Error 1: unhandled cuda error", "ncclUnhandledCudaError", "alias else"),
    ("NCCL Error 3: internal error", "ncclInternalError", "alias else"),
    ("torch.OutOfMemoryError", "cudaError out of memory", "cudaError"),
    ("torch.cuda.OutOfMemoryError", "cudaError out of memory", "cudaError"),
    ("CANCELLED AT", "Cancelled by user", "alias else"),
    ("found NaN in", "NaN reported by Megatron", "cudaError"),
    ("found NaN or inf in", "NaN reported by Megatron", "cudaError"),
    ("collective timeout seq", "pmix collective timeout", "alias else"),
    ("Unexpected result inf", "Bad Token", "alias else"),
    ("RESOURCE_EXHAUSTED: Out of memory", "cudaError out of memory", "cudaError"),
    ("wandb.errors.CommError: Run initialization has timed out", None, "alias else"),
    ("http.client.RemoteDisconnected", None, "alias else"),
    ("EOFError: Ran out of input", None, "alias else"),
    ("[Errno 5] Input/output error", "IO error", "io-error"),
    ("RuntimeError: Rank", "RuntimeError: Rank, node host", "alias else"),
    ("assert number_image_tags <= number_of_images", None, "alias else"),
    ("oom_kill event in StepId", "System-out of memory", "alias else"),
    ("oom_kill events in StepId", "System-out of memory", "alias else"),
    ("sigkill_handler] Killing subprocess", None, "alias else"),
    ("Call to ibv_set_ece failed", None, "cudaError"),
]

for keyword, symptom, rule_name in _GENERIC_KEYWORD_RULES:
    final_symptom = symptom if symptom else keyword
    # Retired buckets credit/demote the finding (symptom); other rows keep their name.
    credited = final_symptom if rule_name in _RETIRE_NOW_BUCKETS else rule_name
    extractor_def(
        name=_keyword_to_rule_name(keyword),
        match_re=re.escape(keyword),
        processor=lambda groups, node, source, ctx, s=final_symptom, c=credited: [
            _bad(node, s, source.derived(rule=c) if c else source, demotion=c)
        ],
    )

_KEYWORD_RETIRED_FINDINGS = {
    (symptom if symptom else keyword)
    for keyword, symptom, rule_name in _GENERIC_KEYWORD_RULES
    if rule_name in _RETIRE_NOW_BUCKETS
}
declare_rule_names(*sorted(_KEYWORD_RETIRED_FINDINGS))


@extractor_def(
    name="NcclWarnMessageTruncated",
    match_re=re.escape("NCCL WARN Message truncated"),
    derived_rules=["nccl-invalid-nic-selection"],
)
def nccl_warn_message_truncated(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    """Detect NCCL invalid NIC selection from 'Message truncated' warnings.

    Dedicated extractor so it gets rule name ``nccl-invalid-nic-selection``
    instead of ``alias else`` and avoids ``AFTER_MFU_DEMOTION``.
    """
    return [_bad(node, "NCCL invalid NIC selection", source.derived(rule="nccl-invalid-nic-selection"))]


# =============================================================================
# Metadata extractors (migrated from applog.py process_single)
# =============================================================================


@extractor_def(
    name="NCCLCudaVersion",
    match_re=r"NCCL version (?P<nccl_ver>\d+\.\d+\.\d+)(?:\S*\+cuda(?P<cuda_ver>\d+\.\d+))?",
    derived_rules=["extract nccl cuda"],
    derived_event_types=["Application.nccl.version"],
)
def nccl_cuda_version(groups: Dict[str, str], _node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Extract NCCL and CUDA version from log line."""
    payload = {}
    if groups.get("nccl_ver") is not None:
        payload["nccl_version"] = groups["nccl_ver"]
    if groups.get("cuda_ver") is not None:
        payload["cuda_version"] = groups["cuda_ver"]
    if len(payload) > 0:
        return [
            EventTriple(
                type="Application.nccl.version",
                payload=payload,
                source=source.derived(rule="extract nccl cuda"),
            )
        ]
    return []


@extractor_def(
    name="DistributedTimeoutMinutes",
    match_re=r"--distributed-timeout-minutes\s+(?P<minutes>\d+)",
    derived_rules=["distributed timeout minutes"],
)
def distributed_timeout_minutes(
    groups: Dict[str, str], _node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    """Extract distributed timeout minutes from command line args."""
    return [
        EventTriple(
            type="Application.megatron-lm.runtime_args",
            payload={"pytorch_timeout_minutes": groups["minutes"]},
            source=source.derived(rule="distributed timeout minutes"),
        )
    ]


@extractor_def(
    name="DistributedInitTimeout",
    match_re=r"Initialized distributed training.*with\s*timeout\s+(?P<seconds>\d+)",
    derived_rules=["distributed timeout (init line, normalized)"],
)
def distributed_init_timeout(groups: Dict[str, str], _node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Extract and normalize distributed init timeout from seconds to minutes."""
    try:
        seconds = int(groups["seconds"])
        minutes = str(int(round(seconds / 60)))
        return [
            EventTriple(
                type="Application.megatron-lm.runtime_args",
                payload={"pytorch_timeout_minutes": minutes},
                source=source.derived(rule="distributed timeout (init line, normalized)"),
            )
        ]
    except ValueError:
        return []


@extractor_def(
    name="MegatronRuntimeArgs",
    match_re=r"\s+(?P<key>\S+)\s+\.{3,}\s+(?P<value>\S+)",
    derived_rules=["megatron runtime args"],
)
def megatron_runtime_args(groups: Dict[str, str], _node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Extract Megatron-LM runtime args (key ... value format)."""
    if ctx.job_info and ctx.job_info.framework == "megatron-lm":
        return [
            EventTriple(
                type="Application.megatron-lm.runtime_args",
                payload={groups["key"]: groups["value"]},
                source=source.derived(rule="megatron runtime args"),
            )
        ]
    return []


# --- Extractor-accumulator rules ---


@extractor_accumulator_def(
    name="NcclWatchdog",
    match_re=r"\[rank\d+\].*Stack trace of the failed collective:|Watchdog caught collective operation timeout:",
    source_filters={ObsSource.APPLOG},
    derived_rules=["Initial line for NCCLWatchdog"],
    derived_event_types=["Application.NCCLWatchdog.report"],
)
class NcclWatchdogAccumulator(ExtractorAccumulator):
    """Accumulates generic PyTorch NCCL watchdog data (stack traces + watchdog timeouts).

    Once triggered, receives all subsequent lines until end-of-stream.
    The internal state machine (NcclWatchdogParseState) handles watchdog and
    stack trace lines; all other lines are ignored.
    """

    def __init__(self):
        self.state = NcclWatchdogParseState()
        self._consuming = False

    def start(self, source: Source) -> None:
        self._consuming = True
        self.process_line(source)

    @property
    def is_consuming(self) -> bool:
        return self._consuming

    def process_line(self, source: Source) -> None:
        self.state.process_line(source.datum, source)

    def finalize(self) -> list[EventTriple]:
        if data := self.state.finalize_and_extract():
            return [EventTriple("Application.NCCLWatchdog.report", {"data": data}, data.source)]
        return []


@extractor_accumulator_def(
    name="RankPulse",
    match_re=r"\[RANK \d+\] rankpulse - INFO:",
    source_filters={ObsSource.APPLOG},
    derived_rules=["Initial line for Rankpulse"],
    derived_event_types=["Application.RankPulse.report"],
)
class RankpulseAccumulator(ExtractorAccumulator):
    """Accumulates RankPulse callstack data.

    Once triggered, receives all subsequent lines until end-of-stream.
    The internal state machine (RankpulseParseState) handles rankpulse lines
    and ignores all others.
    """

    def __init__(self):
        self.state = RankpulseParseState()
        self._consuming = False

    def start(self, source: Source) -> None:
        self._consuming = True
        self.process_line(source)

    @property
    def is_consuming(self) -> bool:
        return self._consuming

    def process_line(self, source: Source) -> None:
        self.state.process_line(source.datum, source)

    def finalize(self) -> list[EventTriple]:
        if data := self.state.finalize_and_extract():
            return [EventTriple("Application.RankPulse.report", {"data": data}, data.source)]
        return []


# --- Collector rules ---
# Post-attribution digest collectors. Each queries the event stream and
# formats results for the report output.


def _fmt_nccl_cuda_version(raw):
    nccl = raw["nccl_version"].value() if "nccl_version" in raw else "NA"
    cuda = raw["cuda_version"].value() if "cuda_version" in raw else "NA"
    return {"nccl_version": nccl, "cuda_version": cuda}


@collector_def(name="Collect.NCCLCudaVersion", formatter=_fmt_nccl_cuda_version)
def collect_nccl_cuda_version(event_stream, _job_info, _params):
    typ_id = event_stream.mgr.types.get_id("Application.nccl.version")
    if typ_id is None:
        return {}

    res: dict[str, SourceSet[str]] = {}

    # We only need the first event, so use ORDER BY with a manual limit
    # Note: query_events_with_source returns an iterator, so we can just take the first item
    for eid, evt_type, payload, source in query_events_with_source(event_stream, typ_id, order_by="e.line_num ASC"):
        if source is not None:
            if (nccl_version := payload.get("nccl_version")) is not None:
                res["nccl_version"] = SourceSet(nccl_version).add_interesting(source)
            if (cuda_version := payload.get("cuda_version")) is not None:
                res["cuda_version"] = SourceSet(cuda_version).add_interesting(source)
        # Only process the first event
        break

    return res


def _fmt_watchdog(raw):
    result = {"watchdog_error_time": None, "watchdog_timeout_seconds": None}
    if raw is not None:
        watchdog = raw.value()
        if "timeout_seconds" in watchdog:
            result["watchdog_timeout_seconds"] = watchdog["timeout_seconds"]
        if "error_time" in watchdog:
            result["watchdog_error_time"] = watchdog["error_time"].isoformat()
    return result


@collector_def(
    name="Collect.WatchdogFoundValues",
    formatter=_fmt_watchdog,
    derived_rules=["jobinfo extract watchdog time", "mfu extract watchdog time"],
)
def collect_watchdog(event_stream, job_info, _params):
    ## Might be better to have emit a pseudo-mfu at job start time for this, but we'll just use jobinfo for now.
    mfu_typ = event_stream.mgr.types.get_id("Application.mfu")
    watchdog_typ = event_stream.mgr.types.get_id("Application.pytorch.nccl_watchdog")

    type_ids = [t for t in [mfu_typ, watchdog_typ] if t is not None]
    if not type_ids:
        return None

    last_time = job_info.source_value(
        job_info.interest_start_time, datum="<JobInfo>", rule="jobinfo extract watchdog time"
    )

    # Need to process events in order since we track last_time across MFU events
    for eid, evt_type, payload, source in query_events_with_source(event_stream, type_ids, order_by="e.line_num ASC"):
        if not payload:
            continue

        if evt_type == mfu_typ:
            if source is not None:
                # Parse datetime from ISO string stored in payload
                mfu_datetime = datetime.fromisoformat(payload["datetime"])
                last_time = SourceSet(mfu_datetime).add_interesting(source.derived(rule="mfu extract watchdog time"))
        elif evt_type == watchdog_typ:
            if last_time is not None and source is not None:
                error_time = datetime.strptime(payload["raw_error_time"], "%H:%M:%S").replace(
                    year=last_time.value().year,
                    month=last_time.value().month,
                    day=last_time.value().day,
                    tzinfo=last_time.value().tzinfo,
                )

                watchdog = {"error_time": error_time, "timeout_seconds": payload["timeout_seconds"]}
                last_time_source = last_time.value_mapped_copy(
                    lambda x: watchdog
                )  # these are some gymastics to enable us to merge the sources for the mfu/jobinfo time with the sources for the watchdog event
                result = SourceSet(watchdog).add_interesting(source).update(last_time_source)
                return result
    return None


def _fmt_runtime_args(raw):
    if raw:
        return {"runtime_args": json.dumps({k: v.value() for k, v in raw.items()})}
    return {"runtime_args": json.dumps({})}


@collector_def(name="Collect.RuntimeArgs", formatter=_fmt_runtime_args)
def collect_runtime_args(event_stream, _job_info, _params):
    typ_id = event_stream.mgr.types.get_id("Application.megatron-lm.runtime_args")
    if typ_id is None:
        return {}

    res = {}
    for _eid, _evt_type, payload, source in query_events_with_source(event_stream, typ_id):
        if not payload:
            continue
        key, val = list(payload.items())[0]
        res[key] = SourceSet(val).add(source)

    return res


def _format_bad_pair(pair):
    result = {"src_node": pair.local_node, "dst_node": pair.remote_node}
    if pair.local_dev_ip is not None:
        result["src_ip"] = pair.local_dev_ip
    if pair.local_device is not None:
        result["src_device"] = pair.local_device
    if pair.remote_dev_ip is not None:
        result["dst_ip"] = pair.remote_dev_ip
    if pair.remote_device is not None:
        result["dst_device"] = pair.remote_device
    return str(result)


def _fmt_bad_pairs(raw):
    formatted = sorted([_format_bad_pair(pair) for pair in raw])
    return {"bad_pairs": "; ".join(str(x) for x in formatted)}


@collector_def(name="Collect.BadPairs", formatter=_fmt_bad_pairs)
def _collect_bad_pairs(event_stream, _job_info, _params):
    type_ids = event_stream.mgr.types.get_ids(BAD_PAIR_EVENT_TYPES)
    if not type_ids:
        return []

    type_ph = sql_placeholders(len(type_ids))
    rows = event_stream.query_raw(
        f"SELECT line_num, payload FROM events WHERE context_id=? AND event_type IN ({type_ph}) AND status=1 ORDER BY line_num ASC",
        [event_stream.ctx_id] + type_ids,
    )

    bad_pairs: list[BadPairInfo] = []
    for _line_num, payload_json in rows:
        payload = json.loads(payload_json) if payload_json else {}
        bad_pair = BadPairInfo(**payload)
        found = False
        for pair in bad_pairs:  # this is inefficient, but it's the original logic and we'll keep it for now
            if pair == bad_pair:
                pair.update(bad_pair)
                found = True
                break
        if not found:
            bad_pairs.append(bad_pair)
    return bad_pairs


def _fmt_rankpulse_all(raw: dict[Literal["nccl", "rankpulse"], list[NcclWatchdogData] | list[RankpulseData]]):
    nccl_list: list[NcclWatchdogData] = raw["nccl"]
    rankpulse_list: list[RankpulseData] = raw["rankpulse"]
    if len(nccl_list) == 0 and len(rankpulse_list) == 0:
        return {}

    if len(nccl_list) > 1:
        _logger.error("Got more than one NCCL watchdog summary; only reporting the first one")
    if len(rankpulse_list) > 1:
        _logger.error("Got more than one rankpulse summary; only reporting the first one")

    nccl = nccl_list[0] if len(nccl_list) > 0 else None
    rankpulse = rankpulse_list[0] if len(rankpulse_list) > 0 else None
    formatted_text = format_rankpulse_all_for_display(
        nccl,
        rankpulse,
        max_bytes=32000,
    )
    if formatted_text:
        return {"rankpulse_all": formatted_text}
    return {}


@collector_def(name="Collect.RankpulseAll", formatter=_fmt_rankpulse_all)
def _collect_rankpulse_all(event_stream, _job_info, _params):
    return {
        "nccl": NcclWatchdogParseState.collect(event_stream),
        "rankpulse": RankpulseParseState.collect(event_stream),
    }
