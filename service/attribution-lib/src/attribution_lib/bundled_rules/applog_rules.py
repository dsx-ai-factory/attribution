# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Applog rules ported from attribution_lib/applog.py node_single function.

Rules are ordered to match their appearance in the original node_single function,
followed by rules from applog_to_symptom.rules in their original dictionary order.

This module directly loads cuda_errors.json to generate CUDA error matching rules,
eliminating the need for applog_to_symptom.py and the dynamic dictionary building in core.py.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

from common.nccl_helpers import get_status_name
from common.nccl_watchdog import NcclWatchdogData, NcclWatchdogParseState
from common.rankpulse import RankpulseParseState
from common.rankpulse_all_common import RankpulseData, format_rankpulse_all_for_display
from loguru import logger as _logger

import attribution_lib
from attribution_lib.attr_types import BadPairInfo
from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import BAD_PAIR_EVENT_TYPES, DemotionTypes, sql_placeholders
from attribution_lib.network_identity import NetworkSourceIdentity, network_source_identity, normalize_network_hostname
from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rule_helpers import _bad
from attribution_lib.rules import (
    EventTriple,
    ExtractorAccumulator,
    RuleContext,
    collector_def,
    declare_event_types,
    declare_rule_names,
    expr_helper,
    extractor_accumulator_def,
    rows,
    rule_processor,
)
from attribution_lib.sources import Source, SourceSet
from attribution_lib.utils import (
    _leading_explicit_applog_rank,
    convert_hostname_to_short_format,
    get_hostname,
    matches_ignored_hostname_prefix,
)

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

    # Customize nice_name of "out of memory" to be compatible with NVIDIA Attribution history
    if "out of memory" in _CUDA_ERRORS:
        _CUDA_ERRORS["out of memory"]["nice_name"] = "cudaError out of memory"

    # Both the CUDA family and NCCL processors use this lookup table.
    _CUDA_ERRORS["cudaErrorIllegalAddress"] = _CUDA_ERRORS["an illegal memory access was encountered"]

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


@rows("cuda_errors")
def _cuda_error_rows():
    """Expose the CUDA error catalog as a record table for the declarative
    ``CudaError`` family (bundled_rules/applog_cuda_family.toml). One record per
    error description; the credit is PRE-RESOLVED here (the family emit is a static
    template and cannot branch): a "cuda*" nice_name credits itself, everything
    else credits "alias else" (mirrors the former _cuda_error_processor)."""
    _load_cuda_errors()  # idempotent; ensures _CUDA_ERRORS is built
    out = {}
    for desc, info in _CUDA_ERRORS.items():
        nice = info["nice_name"]
        credit = nice if nice.startswith("cuda") else "alias else"
        out[desc] = {"error_desc": desc, "symptom": nice, "credit": credit}
    return out


rule_author = "NVIDIA Attribution team"

# Shared derived rule names for loop-generated rules where co-location isn't practical.
declare_rule_names(
    "alias else",
    "DataLoader Exception",
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


@expr_helper("format_ib_error")
def _format_ib_error_message(error_code: int, vendor_err: int) -> str:
    """Format an IB work-completion error as an NVIDIA Attribution symptom.

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


@expr_helper("is_user_path")
def _path_is_user_owned(path: str) -> bool:
    """Paths under /home/ or /lustre/ are user-managed; attribution should be Fail User for permission/space issues there.

    Matches the existing `pyxis_enroot_mount_failure_user_event` heuristic.
    """
    return path.startswith("/home/") or path.startswith("/lustre/")


# Matches pyxis ephemeral container names like pyxis_00001.0, pyxis_00001_1000001.0.
_EPHEMERAL_PYXIS_NAME_RE = re.compile(r"pyxis_[0-9]+(_[0-9]+)?(\.[0-9]+)?$")


@expr_helper("is_ephemeral_pyxis_container")
def _path_is_ephemeral_pyxis_container(path: str) -> bool:
    last = path.rstrip("/").rsplit("/", 1)[-1]
    return bool(_EPHEMERAL_PYXIS_NAME_RE.match(last))


# =============================================================================
# Rules from node_single (in original order) - these have complex logic
# =============================================================================


@rule_processor("applog:MFU")
def mfu_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Extract MFU iteration timestamp."""
    time = datetime.fromisoformat(groups["isotime"]).replace(tzinfo=source.source_context.timezone)
    return [EventTriple(type="Application.mfu", payload={"datetime": time}, source=source.derived(time=time))]


@rule_processor("applog:socketStartConnect")
def socket_start_connect_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    peer_node = get_hostname(groups["ip"])
    return [_bad_peers(node, peer_node, groups["reason"], source, demotion="socketStartconnect")]


@rule_processor("applog:socketPollConnect")
def socket_poll_connect_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    peer_node = get_hostname(groups["ip"])
    return [_bad_peers(node, peer_node, groups["reason"], source, demotion="socketPollConnect")]


@rule_processor("applog:socketProgressOpt")
def socket_progress_opt_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    peer_node = get_hostname(groups["ip"])
    # The verified node can be fully qualified while the mapped peer is short, so a short name
    # matches its qualified form. Two qualified names must match exactly, as in verify_node.
    same_node = node == peer_node or (
        ("." not in node or "." not in peer_node)
        and convert_hostname_to_short_format(node) == convert_hostname_to_short_format(peer_node)
    )
    if same_node:
        return []
    return [_bad(peer_node, groups["reason"], source, demotion="socketProgressOpt")]


@rule_processor("applog:IPv4CannotBeRetrieved")
def ipv4_cannot_be_retrieved_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    peer_node = get_hostname(groups["hostname"])
    return [_bad(peer_node, "network address cannot be retrieved", source, demotion="cannot be retrieved")]


@rule_processor("applog:ClientSocketTimeout")
def client_socket_timeout_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    peer_node = get_hostname(groups["hostname"])
    return [_bad(peer_node, "client socket has timed out", source, demotion="client socket has timed out")]


# AttributeTypeError, KeyValidationError, and "completed with error" ported to
# bundled_rules/applog_leaf_rules.toml.


def _completion_pair_node(identity: NetworkSourceIdentity, node: str | None, source: Source) -> str | None:
    """Preserve known source names without promoting IPs or implicit rank zero.

    Args:
        identity: Explicit log hostname and scoped GID mapping.
        node: Allocation-verified node, possibly inferred by the generic parser.
        source: Original line and verified producer-declared node, when present.
    Returns:
        An evidenced source hostname, or None when no hostname is known.
    """
    if identity.hostname is not None:
        return identity.hostname
    if source.declared_node is not None:
        return normalize_network_hostname(source.declared_node)
    if _leading_explicit_applog_rank(source.datum) is not None:
        return normalize_network_hostname(node) if node is not None else None
    return None


@rule_processor("applog:Got completion from peer (new)")
def got_completion_from_peer_new_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    peer_node = get_hostname(groups["peer_ip"])

    msg = _format_ib_error_message(int(groups["error_code"]), int(groups["vendor_err"]))

    events = [
        _bad_peers(
            node, peer_node, msg, source, demotion="Got completion from peer", rule_name="Got completion from peer"
        )
    ]

    # Try to create complete bad_pair event with all fields
    local_dev_ip = groups["local_gid"][7:] if groups["local_gid"].startswith("::ffff:") else None
    remote_dev_ip = groups["remote_gid"][7:] if groups["remote_gid"].startswith("::ffff:") else None

    # Pair hostnames describe the log and GID mapping independently of the
    # allocation-filtered node used for fault attribution above.
    identity = network_source_identity(source.datum, prefix_hostname=node)
    pair_node = _completion_pair_node(identity, node, source)
    if local_dev_ip and remote_dev_ip:
        bad_pair_payload = {
            "local_node": pair_node,
            "local_gid_node": identity.mapped_hostname,
            "remote_node": normalize_network_hostname(peer_node) or peer_node,
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
            "local_node": pair_node,
            "local_gid_node": identity.mapped_hostname,
            "remote_node": normalize_network_hostname(peer_node) or peer_node,
        }
        events.append(
            EventTriple(
                type="Application.bad_pair.partial",
                payload=bad_pair_payload,
                source=source.derived(rule="Got completion from peer"),
            )
        )
    return events


@rule_processor("applog:Got completion from peer (old)")
def got_completion_from_peer_old_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    peer_node = get_hostname(groups["peer_ip"])

    msg = _format_ib_error_message(int(groups["error_code"]), int(groups["vendor_err"]))

    events = [
        _bad_peers(
            node, peer_node, msg, source, demotion="Got completion from peer", rule_name="Got completion from peer"
        )
    ]

    # This matcher also handles GID-bearing lines. Both matchers must preserve
    # the same source names so their complete/partial events can be combined.
    # Without GIDs, an explicit hostname still outranks the filtered node.
    identity = network_source_identity(source.datum, prefix_hostname=node)
    pair_node = _completion_pair_node(identity, node, source)
    pair_peer = normalize_network_hostname(peer_node) or peer_node

    # Old format doesn't have GID fields, create partial bad_pair
    bad_pair_payload = {
        "local_node": pair_node,
        "local_gid_node": identity.mapped_hostname,
        "remote_node": pair_peer,
    }
    events.append(
        EventTriple(
            type="Application.bad_pair.partial",
            payload=bad_pair_payload,
            source=source.derived(rule="Got completion from peer"),
        )
    )
    return events


# "Got completion with error" (leaf, format_ib_error templated symptom) ported to
# bundled_rules/applog_leaf_rules.toml.


@rule_processor("applog:Irrecoverable NET/IB communicator error")
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

    peer_node = get_hostname(groups["peer_ip"])
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


@rule_processor("applog:Call to ibv_modify_qp failed")
def ibv_modify_qp_failed_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    error_code = int(groups["error_code"])
    local_gid, remote_gid = groups["local_gid"], groups["remote_gid"]
    remote_ip = remote_gid[7:] if remote_gid.startswith("::ffff:") else remote_gid

    peer_node = get_hostname(remote_ip)

    msg = f"Call to ibv_modify_qp failed: eno {error_code}"
    if error_code == 11:
        msg += " resource unavailable"
    elif error_code == 22:
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


@rule_processor("applog:Connection closed by remote peer")
def connection_closed_by_remote_peer_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    raw_peer = groups["hostname"]
    peer_node = get_hostname(raw_peer)
    # Match the resolved identity and the raw log peer. Do not split IPs on
    # dots: that is a hostname FQDN strip, and it turns 192.0.2.1 into "192".
    if matches_ignored_hostname_prefix(peer_node) or matches_ignored_hostname_prefix(raw_peer):
        return []
    return [_bad(peer_node, "Connection closed", source, demotion="Connection closed by remote peer")]


# BusErrorSignal, BusErrorKeyword, and AddressAlreadyInUse ported to
# bundled_rules/applog_leaf_rules.toml.


# =============================================================================
# Rules with special logic (CUDA catalog lookup, peer resolution, etc.)
# =============================================================================


@rule_processor("applog:NCCLWarnCudaFailure")
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


@rule_processor("applog:pytorch_watchdog")
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


@rule_processor("applog:pushrecv")
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


@rule_processor("applog:dataloader")
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


# PyxisEnrootMountFailureSystem (pick_one) and PyxisEnrootMountFailureUser ported
# to bundled_rules/applog_leaf_rules.toml.


_DOCKER_HUB_FALLBACK_PATH_RE = re.compile(r"registry-1\.docker\.io/v2/([^/\s]+)/")


@rule_processor("applog:EnrootRegistryHttpError")
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


# EnrootFileAlreadyExists (leaf with an is_ephemeral_pyxis_container `when` guard)
# ported to bundled_rules/applog_leaf_rules.toml.

# Path-aware mkdir permission denied (PyxisMkdirPermissionDenied), unsquashfs
# dir_scan permission denied (UnsquashfsDirScanPermissionDenied), and unsquashfs
# writer failure (UnsquashfsWriteFailed) are pick_one rules branching on
# is_user_path; ported to bundled_rules/applog_combinator_rules.toml.


# UnsquashfsOomKilled and MksquashfsDiskFull ported to
# bundled_rules/applog_leaf_rules.toml.


# PyxisNamespaceCreateFailed, EnrootCapSetFailed, PyxisContainerStartNonzero,
# PyxisCouldntChdirWorkdir, PyxisConflictingOptions, SrunInvalidPyxisArgument, and
# PyxisCouldNotWaitChildInterrupted ported to bundled_rules/applog_leaf_rules.toml.


# PyTorch NCCL watchdog rules ported to
# bundled_rules/applog_processor_rules.toml (6 stanzas -> one @rule_processor).


# Marker used for multicast NV_ERR suppression during refinement.
@rule_processor("applog:GracefullySkippingMulticastInitialization")
def gracefully_skipping_multicast_initialization_event(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    _logger.info("Applog marker detected: Gracefully skipping multicast initialization")
    return []


# DataLoader exception rules ported to
# bundled_rules/applog_processor_rules.toml (N stanzas -> one @rule_processor).

# "the process completes" rules ported to
# bundled_rules/applog_process_completes_family.toml (declarative [[family]]).


# =============================================================================
# CUDA Error Rules (auto-generated from cuda_errors.json)
# =============================================================================


# The CUDA error matrix (135 descriptions x 4 pattern templates = 540 rules) is
# now a declarative rows-backed [[family]] in bundled_rules/applog_cuda_family.toml,
# fed by the @rows("cuda_errors") provider above.

# cudaError-branch findings now credit their own nice_name; register so source.rule resolves.
_CUDA_NICE_NAMES = {info["nice_name"] for info in _CUDA_ERRORS.values()}
declare_rule_names(*sorted({n for n in _CUDA_NICE_NAMES if n.startswith("cuda")}))


# "No such file or directory" and "Gloo connectFullMesh failed" ported to
# bundled_rules/applog_leaf_rules.toml.


# =============================================================================
# Generic keyword rules from applog_to_symptom.py
# =============================================================================

# The generic-keyword family (~90 keyword rules) migrated to a TOML [[family]]
# in bundled_rules/generic_keyword_family.toml (pattern = the literal keyword,
# re.escape'd; symptom/credit/demotion resolved into explicit row columns).

# NcclWarnMessageTruncated ported to bundled_rules/applog_leaf_rules.toml.


# =============================================================================
# Metadata extractors (migrated from applog.py process_single)
# =============================================================================


@rule_processor("applog:NCCLCudaVersion")
def nccl_cuda_version(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
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


# DistributedTimeoutMinutes and DistributedInitTimeout ported to
# bundled_rules/applog_leaf_rules.toml.


@rule_processor("applog:MegatronRuntimeArgs")
def megatron_runtime_args(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
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
    if pair.local_gid_node is not None and pair.local_gid_node != pair.local_node:
        result["src_gid_node"] = pair.local_gid_node
    if pair.remote_gid_node is not None and pair.remote_gid_node != pair.remote_node:
        result["dst_gid_node"] = pair.remote_gid_node
    return str(result)


def _fmt_bad_pairs(raw):
    formatted = sorted([_format_bad_pair(pair) for pair in raw])
    return {"bad_pairs": "; ".join(str(x) for x in formatted)}


def _bad_pair_endpoints_agree(existing: BadPairInfo, incoming: BadPairInfo) -> bool:
    """Allow node-pair merging only when known IP/device/GID names agree.

    Args:
        existing: A previously collected pair.
        incoming: A pair whose direction may be reversed.
    Returns:
        Whether merging can fill missing fields without mixing interfaces.
    """
    if existing != incoming:
        return False
    current = (
        existing.local_dev_ip,
        existing.local_device,
        existing.local_gid_node,
        existing.remote_dev_ip,
        existing.remote_device,
        existing.remote_gid_node,
    )
    if existing.local_node == incoming.local_node:
        other = (
            incoming.local_dev_ip,
            incoming.local_device,
            incoming.local_gid_node,
            incoming.remote_dev_ip,
            incoming.remote_device,
            incoming.remote_gid_node,
        )
    else:
        other = (
            incoming.remote_dev_ip,
            incoming.remote_device,
            incoming.remote_gid_node,
            incoming.local_dev_ip,
            incoming.local_device,
            incoming.local_gid_node,
        )
    return all(left is None or right is None or left == right for left, right in zip(current, other))


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
            if _bad_pair_endpoints_agree(pair, bad_pair):
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
