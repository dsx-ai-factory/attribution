# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/nverror.h
# https://docs.nvidia.com/deploy/xid-errors/index.html
import re
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from typing import Any, Optional

from attribution_lib.job_info import JobInfo
from attribution_lib.xid_analyzer import nvidia_xid_analyzer
from attribution_lib.sources import Source

nvidia_xid_analyzer.setup_rules()

xpa_xid_catalog: dict[str, Any] = nvidia_xid_analyzer.XID_CATALOG
xpa_nvl5_xid_decoding_rules: dict[int, list[dict[str, Any]]] = nvidia_xid_analyzer.NVL5_XID_DECODING_RULES
xpa_sticky_xids: Any = nvidia_xid_analyzer.STICKY_XIDS
xpa_xid_policies: Any = nvidia_xid_analyzer.XID_POLICY

xid_catalog: dict[int, dict[str, Any]] = {}
nvlink5_catalog: dict[str, Any] = defaultdict(list)

# Error event decode fields for XID 144-150 processing
# v2 fields. Driver version 575 and above.
fields_v2: list[dict[str, Any]] = [
    {
        "field": "NV_MSETP_HDR_RMCMD_ERROR_EVENT_V2_NTR_INFO_BLOCK_ERROR_CODE",
        "shift": 0,
        "mask": 0x7F,
        "code": {
            0x00000000: "INVALID_0",
            0x00000001: "SAW_MVB",
            0x00000002: "SAW_MSEFW",
            0x00000003: "RLW_CTRL",
            0x00000004: "RLW_REMAP",
            0x00000005: "RLW_RSPCOL",
            0x00000006: "RLW_RXPIPE",
            0x00000007: "RLW_SRC_TRACK",
            0x00000008: "RLW_TAGSTATE",
            0x00000009: "TLW_CTRL",
            0x0000000A: "TLW_RX_PIPE0",
            0x0000000B: "TLW_RX_PIPE1",
            0x0000000C: "TLW_TX_PIPE0",
            0x0000000D: "TLW_TX_PIPE1",
            0x0000000E: "TREX",
            0x0000000F: "NVLPW",
            0x00000010: "NETIR_PMPE",
            0x00000011: "NETIR_LINK_DOWN",
            0x00000012: "NETIR_THERMAL_EVENT",
            0x00000013: "NETIR_BER_EVENT",
            0x00000014: "NETIR_MFDE_EVENT",
            0x00000015: "MVB_CTRL",
            0x00000016: "MVB_RX",
            0x00000017: "MVB_TX",
            0x00000018: "NETIR_INT",
        },
    },
    {
        "field": "NV_MSETP_HDR_RMCMD_ERROR_EVENT_V2_PORT_DOWN_REASON_CODE",
        "shift": 20,
        "mask": 0x3F,
        "code": {
            0x00000000: "NO_LINK_DOWN",
            0x00000001: "UNKNOWN",
            0x00000002: "HI_SER_BER",
            0x00000003: "BLOCK_LOCK_LOSS",
            0x00000004: "ALIGNMENT_LOSS",
            0x00000005: "FEC_SYNC_LOSS",
            0x00000006: "PLL_LOCK_LOSS",
            0x00000007: "FIFO_OVERFLOW",
            0x00000008: "FALSE_SKIP_CONDITION",
            0x00000009: "MINOR_ERR_THRESHOLD",
            0x0000000A: "PHY_LAYER_RETRANSMIT_TIMEOUT",
            0x0000000B: "HEARTBEAT_ERRORS",
            0x0000000C: "LINK_LAYER_CREDIT_MON_WD",
            0x0000000D: "LINK_LAYER_INTEGRITY_THRESHOLD",
            0x0000000E: "LINK_LAYER_BUFFER_OVERRUN",
            0x0000000F: "OOB_CMD_LINK_HEALTHY",
            0x00000010: "OOB_CMD_LINK_HI_BER",
            0x00000011: "INBAND_CMD_LINK_HEALTHY",
            0x00000012: "INBAND_CMD_LINK_HI_BER",
            0x00000013: "DOWN_BY_VERIFICATION_GW",
            0x00000014: "RECEIVED_REMOTE_FAULT",
            0x00000015: "RECEIEVED_TS1",
            0x00000016: "DOWN_BY_MGMT_CMD",
            0x00000017: "CABLE_UNPLUGGED",
            0x00000018: "CABLE_ACCESS_ISSUES",
            0x00000019: "THERMAL_SHUTDOWN",
            0x0000001A: "CURRENT_ISSUE",
            0x0000001B: "POWER_BUDGET",
            0x0000001C: "FAST_RECOVERY_RAW_BER",
            0x0000001D: "FAST_RECOVERY_EFFECTIVE_BER",
            0x0000001E: "FAST_RECOVERY_SYMBOL_BER",
            0x0000001F: "FAST_RECOVERY_CREDIT_WATCHDOG",
            0x00000020: "PEER_SLEEP",
            0x00000021: "PEER_DISABLE",
            0x00000022: "PEER_DISABLE_LOCK",
            0x00000023: "PEER_THERMAL_EVENT",
            0x00000024: "PEER_FORCE_EVENT",
            0x00000025: "PEER_RESET_EVENT",
            0x00000026: "RESET_NO_POWER_CYCLE",
            0x00000028: "DOWN_DUE_TO_HW_FORCE_EVENT",
            0x0000002A: "L1_FAILURE",  # LinkFatal, RESET_GPU
            0x0000002B: "TOO_MANY_LINK_ERROR_RECOVERIES",  # LinkFatal, INVESTIGATE_LINK_SI_AND_CABLES
            0x0000002C: "DOWN_DUE_TO_CONTAIN_MODE",  # LinkFatal, IGNORE
            0x0000002D: "BW_LOSS_THRESHOLD_EXCEEDED",  # NonFatal, RESET_GPU (This will show up in interesting_nodes, debug for performance issues)
            0x00000030: "DOWN_BY_NMX_ADMIN_STATE",  # Variable, REENABLE_LINK
        },
    },
]


"""
XID-145 Decoding

For subcode RLW_REMAP: 00100
==============
Error Status: Job Severity, HW/SW classifications details below:
0x00000001: AppCrash, SW  # Check for XC, Xid 45
0x00000002: AppCrash, SW  # Check for XC, Xid 45
0x00000004: AppCrash, SW  # Check for XC, Xid 45, likely trunk link failure
0x00000008: AppCrash, SW  # Check for XC, Xid 45, likely trunk link failure
0x00000010: AppCrash, SW  # Check for XC, Xid 45, likely trunk link failure
0x00000020: AppCrash, SW  # Check for XC, Xid 45, likely trunk link failure
0x00000040: Fatal, InfraHW  # Xid 45 and 48, RESET
0x00000080: Fatal, InfraHW  # Xid 45 and 48
0x00000100: JobNonFatal, HW  # Correctable HW error
0x00000200: JobNonFatal, HW  # Correctable HW error
0x80000000: JobNonFatal, SW


For subcode RLW_RSPCOL: 00101
==============
Error Status: Job Severity, HW/SW classifications details below:
0x00000001: JobNonFatal, HW
0x00000002: Fatal, InfraHW # Xid 45 and 48, RESET
0x80000000: JobNonFatal, SW


For subcode RLW_RXPIPE: 00110
==============
Error Status: Job Severity, HW/SW classifications details below:
0x00000001: RemotePacketLoss, InfraSW
0x00000002: RemotePacketLoss, InfraSW
0x00000004: LocalPacketLoss, InfraSW
0x00000008: PRIV_ERR, MultiCategory
0x00000010: JobNonFatal, ShouldNotOccur
0x00000020: JobNonFatal, ShouldNotOccur
0x00000040: JobNonFatal, ShouldNotOccur
0x80000000: JobNonFatal, SW


For subcode RLW_SRC_TRACK: 00111
==============
Error Status: Job Severity, HW/SW classifications details below:
0x00000001: Fatal, InfraHW
0x00000002: JobNonFatal, HW  # Correctable HW error
0x00000004: AppCrash, MultiCategory # Xid 45, Packet loss on read.
0x00000008: AppCrash, MultiCategory # Xid 45, Packet loss on write.
0x00000010: Fatal, InfraHW  # Xid 45 and 48, RESET
0x00000020: Fatal, InfraHW  # Xid 45 and 48, RESET
0x80000000: JobNonFatal, SW


For subcode RLW_TAGSTATE: 01000
==============
Error Status: Job Severity, HW/SW classifications details below:
0x00000001: JobNonFatal, HW  # Correctable HW error
0x00000002: Fatal, InfraHW  # Xid 45 and 48, RESET
0x00010000: JobNonFatal, HW  # Correctable HW error
0x00020000: Fatal, InfraHW  # Xid 45 and 48, RESET
0x00100000: Fatal, InfraHW  # Xid 45 and 48, RESET
0x80000000: JobNonFatal, SW
"""

# XID-145 Decoding Data Structure
# Maps (subcode, error_status) -> (job_severity, hw_sw_classification)
XID_145_DECODING_RULES = {
    "RLW_REMAP": {
        0x00000001: ("AppCrash", "SW"),
        0x00000002: ("AppCrash", "SW"),
        0x00000004: ("AppCrash", "SW"),
        0x00000008: ("AppCrash", "SW"),
        0x00000010: ("AppCrash", "SW"),
        0x00000020: ("AppCrash", "SW"),
        0x00000040: ("Fatal", "InfraHW"),
        0x00000080: ("Fatal", "InfraHW"),
        0x00000100: ("JobNonFatal", "HW"),
        0x00000200: ("JobNonFatal", "HW"),
        0x80000000: ("JobNonFatal", "SW"),
    },
    "RLW_RSPCOL": {
        0x00000001: ("JobNonFatal", "HW"),
        0x00000002: ("Fatal", "InfraHW"),
        0x80000000: ("JobNonFatal", "SW"),
    },
    "RLW_RXPIPE": {
        0x00000001: ("RemotePacketLoss", "InfraSW"),
        0x00000002: ("RemotePacketLoss", "InfraSW"),
        0x00000004: ("LocalPacketLoss", "InfraSW"),
        0x00000008: ("PRIV_ERR", "MultiCategory"),
        0x00000010: ("JobNonFatal", "ShouldNotOccur"),
        0x00000020: ("JobNonFatal", "ShouldNotOccur"),
        0x00000040: ("JobNonFatal", "ShouldNotOccur"),
        0x80000000: ("JobNonFatal", "SW"),
    },
    "RLW_SRC_TRACK": {
        0x00000001: ("Fatal", "InfraHW"),
        0x00000002: ("JobNonFatal", "HW"),
        0x00000004: ("AppCrash", "MultiCategory"),
        0x00000008: ("AppCrash", "MultiCategory"),
        0x00000010: ("Fatal", "InfraHW"),
        0x00000020: ("Fatal", "InfraHW"),
        0x80000000: ("JobNonFatal", "SW"),
    },
    "RLW_TAGSTATE": {
        0x00000001: ("JobNonFatal", "HW"),
        0x00000002: ("Fatal", "InfraHW"),
        0x00010000: ("JobNonFatal", "HW"),
        0x00020000: ("Fatal", "InfraHW"),
        0x00100000: ("Fatal", "InfraHW"),
        0x80000000: ("JobNonFatal", "SW"),
    },
}


def get_xid_145_classification(subcode: str, error_status: int) -> Optional[tuple[str, str]]:
    """
    Get JobSeverity and HW/SW Classification for XID-145 based on subcode and ErrorStatus
    Ignores JobNonFatal cases.

    Args:
        subcode: Subcode string like "RLW_REMAP", "RLW_RSPCOL", etc.
        error_status: ErrorStatus as integer (e.g., 0x00000001)

    Returns:
        tuple: (job_severity, hw_sw_classification) or None if JobNonFatal or not found
    """
    if subcode not in XID_145_DECODING_RULES:
        return None

    subcode_rules = XID_145_DECODING_RULES[subcode]
    if error_status not in subcode_rules:
        return None

    job_severity, hw_sw_classification = subcode_rules[error_status]

    # Ignore JobNonFatal cases
    if job_severity == "JobNonFatal":
        return None

    return (job_severity, hw_sw_classification)


def get_xid_145_attribution(subcode: str, error_status_decimal: str) -> tuple[Optional[str], Optional[str]]:
    """
    Get attribution info for XID-145 in NVIDIA Attribution format

    Args:
        subcode: Subcode string like "RLW_REMAP", "RLW_RSPCOL", etc.
        error_status_decimal: ErrorStatus as decimal string (e.g., "1")

    Returns:
        tuple: (job_disruption, failure_category) or (None, None) if JobNonFatal or not found
    """
    try:
        error_status_int = int(error_status_decimal)
    except (ValueError, TypeError):
        return (None, None)

    result = get_xid_145_classification(subcode, error_status_int)
    if result is None:
        return (None, None)

    job_severity, hw_sw_classification = result

    # Map job_severity to job_disruption (Yes/No)
    job_disruption = "Yes"  # All non-JobNonFatal cases cause job disruption

    # Use hw_sw_classification directly as failure_category
    failure_category = hw_sw_classification

    return (job_disruption, failure_category)


def decode_error_event(error_code_hex: str) -> dict:
    """
    Decode error event data using v2 decoding.

    Args:
        error_code_hex: Hex string like "0x004425c6"

    Returns:
        dictionary with subcode, port_down_reason_code and value
    """
    try:
        error_code = int(error_code_hex, 16)
    except ValueError:
        return {}

    result = {}

    for field in fields_v2:
        temp = (error_code >> field["shift"]) & field["mask"]
        field_name = field["field"]

        if field_name == "NV_MSETP_HDR_RMCMD_ERROR_EVENT_V2_NTR_INFO_BLOCK_ERROR_CODE":
            result["subcode"] = field["code"].get(temp, f"UNKNOWN_SUBCODE_{temp}")
        elif field_name == "NV_MSETP_HDR_RMCMD_ERROR_EVENT_V2_PORT_DOWN_REASON_CODE":
            result["port_down_reason_code"] = field["code"].get(temp, f"UNKNOWN_REASON_{temp}")
            result["port_down_reason_code_value"] = str(temp)

    return result


def decode_xid_error_from_syslog(xid: int, syslog_line: str) -> dict:
    """Decode XID error data from syslog line for NVLink XIDs (144-150).

    Returns:
        dict: Decoded error data containing code, subcode, port_down_reason_code, etc.
              Empty dict if decoding fails or not applicable.
    """
    # Only decode NVLink XIDs (144-150)
    if not (144 <= xid <= 150):
        return {}

    try:
        error_match = re.search(r"\((0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)", syslog_line)
        if error_match:
            error_code = error_match.group(1)
            return decode_error_event(error_code)
    except TypeError:
        pass

    return {}


def _format_nvlink_xid_identifier(xid: int, decoded_data: dict, error_status_decimal: str) -> str:
    """Format an NVLink XID identifier.

    Args:
        xid: XID number.
        decoded_data: Decoded error event data.
        error_status_decimal: ErrorStatus in decimal format.

    Returns:
        Identifier used for symptoms and PCI mapping keys.
    """
    subcode = decoded_data.get("subcode", "")
    port_down_reason_code = decoded_data.get("port_down_reason_code", "")
    port_down_reason_code_value = decoded_data.get("port_down_reason_code_value", "")

    if xid == 149 and port_down_reason_code != "" and port_down_reason_code_value != "":
        # For XID-149: print XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0
        return f"XID-{xid}.{port_down_reason_code_value}-{subcode}-{port_down_reason_code}-{error_status_decimal}"
    if xid == 150:
        return f"XID-{xid}-{subcode}"
    if 144 <= xid <= 149:
        # For XID-145 and others: print XID-145-RLW_REMAP-1
        return f"XID-{xid}-{subcode}-{error_status_decimal}"
    return f"XID-{xid}"


def format_xid_with_subcode(xid: int, syslog_line: str = "") -> str:
    """Format XID with subcode for consistent mapping keys

    For NVLink XIDs (144-150), uses proper decoding to extract meaningful subcodes.
    For other XIDs, extracts subcode from syslog line and uses enum mapping.
    """

    decoded_data = {}
    error_status_decimal = "0"

    if 144 <= xid <= 150 and syslog_line:
        decoded_data = decode_xid_error_from_syslog(xid, syslog_line)

        error_match = re.search(r"\((0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)", syslog_line)
        if error_match:
            error_status = error_match.group(2)
            try:
                error_status_decimal = str(int(error_status, 16))
            except ValueError:
                error_status_decimal = "0"

    return _format_nvlink_xid_identifier(xid, decoded_data, error_status_decimal)


def create_symptom_nvl(xid_number: int, decoded_data: dict, error_status_decimal: str = "0") -> str:
    """
    Create symptom_nvl format based on test_xid_error_decode.py logic

    Args:
        xid_number: XID number (e.g., 149, 145)
        decoded_data: Decoded error event data
        error_status_decimal: ErrorStatus in decimal format

    Returns:
        symptom_nvl formatted string
    """
    return _format_nvlink_xid_identifier(xid_number, decoded_data, error_status_decimal)


# fmt: off
class NVLINK5_CODE(Enum):
    NVLINK_SAW_ERROR        = 0b00001
    NVLINK_RLW_ERROR        = 0b00010
    NVLINK_TLW_ERROR        = 0b00011
    NVLINK_NVLPW_CTRL_ERROR = 0b00101 # XID-148, deprecated?
    NVLINK_TREX_ERROR       = 0b00100
    NVLINK_NETIR_ERROR	    = 0b00110
    MSE_WATCHDOG            = 0b00000
# fmt: on


# fmt: off
class NVLINK5_SUBCODE(Enum):
    SAW_MVB             = 0b00001
    RLW_CTRL            = 0b00011
    RLW_REMAP           = 0b00100
    RLW_RSPCOL          = 0b00101
    RLW_RXPIPE          = 0b00110
    RLW_SRC_TRACK       = 0b00111
    RLW_TAGSTATE        = 0b01000
    TLW_CTRL            = 0b01001
    TLW_RX_PIPE0        = 0b01010
    TLW_RX_PIPE1        = 0b01010
    TLW_TX_PIPE0        = 0b01011
    TLW_TX_PIPE1        = 0b01011
    TREX                = 0b01100
    NOT_APPLICABLE      = 0b00000
    NETIR_LINK_EVT      = 0b01110
    NETIR_BER_EVENT     = 0b10001
    NETIR_MFDE_EVENT    = 0b1001
# fmt: on


# fmt: off
class PORT_DOWN_REASON_CODE(Enum):
    No_link_down_indication                             = 0b000000
    Unknown_reason                                      = 0b000001
    Hi_SER_or_Hi_BER                                    = 0b000010
    Alignment_loss                                      = 0b000100
    Physical_layer_retransmission_timeout               = 0b001010
    Down_by_outband_command_with_healthy_link           = 0b001111
    Down_by_outband_command_for_link_with_hi_ber        = 0b010000
    Down_by_inband_command_with_healthy_link            = 0b010001
    Down_by_inband_command_for_link_with_hi_ber         = 0b010010
    Received_TS1                                        = 0b010101
    Down_by_management_command                          = 0b010110
    Peer_side_down_to_sleep_state                       = 0b100000
    Peer_side_down_to_disable_state                     = 0b100001
    Peer_side_down_to_disable_and_port_lock             = 0b100010
    Peer_side_down_due_to_thermal_event                 = 0b100011
    Peer_side_down_due_to_force_event                   = 0b100100
    Peer_side_down_due_to_reset_event                   = 0b100101
    Reset_no_power_cycle                                = 0b100110
    Down_due_to_HW_force_event                          = 0b101000
    Down_due_to_thermal_event                           = 0b101001
# fmt: on

"""
result = {
    'event': 'NETIR_LINK_EVT'
    'severity': 'Fatal'
    'cross_contain': 'XC0'
    'injected': 'i0'
    'link': '09'
    'intrInfo': '0x025525c6'
    'errorStatus': '0x00000000'
    'errorDebugData_0': '0x00000000'
    'errorDebugData_1': '0x00000000'
    'errorDebugData_2': '0x00000000'
    'errorDebugData_3': '0x00000000'
    'recovery_action': ''
    'investigatory_action': ''
}
"""


def parse_nvlink5_error(xid, extra) -> dict:
    if not nvlink5_catalog:
        for id, subcodes in xpa_nvl5_xid_decoding_rules.items():
            for subcode_info in subcodes:
                row = subcode_info.copy()

                subcode_name = str(row["sheet_subcode"][0])
                PortDownReason = str(row["sheet_comments"])
                errorStatus = int(row["sheet_error_status_hex"][0])

                row["nice_name"] = f"XID-{id}-{subcode_name}"
                description = str(row["sheet_comments"])
                if description:
                    row["nice_name"] += f"-{description}"

                unique_id = f"{id}-{subcode_name}-{PortDownReason}-{errorStatus}"

                nvlink5_catalog[unique_id] = row

    # NETIR_LINK_EVT  Fatal   XC0 i0 Link 08 (0x025505c6 0x00000000 0x00000000 0x00000000 0x00000000 0x00000000)
    # NETIR           Fatal   XC0 i0 Link -1 (0x000fe406 0x00000000 0x00000000 0x00000000 0x00000000 0x00000000)
    # <sub component> <fatal vs nonfatal> <Crosscontain> <injected> <link> (<intrInfo>, <errorStatus>, <errorDebugData[0]>, <errorDebugData[1]>, <errorDebugData[2]>, <errorDebugData[3]>, <errorDebugData[4]>)
    """
    IntrInfo bit mapping

    AAABCD	                        [31:26]: [A: 31:29] RSVD, [B: 28] V:Debug, [C: 27] OVF, [D: 26] First
        EEEEEE	                    [E: 25:20] Reason code
                FFGGGGG	            [19:13]: [F: 19:18] Post Status and [G: 17:13] Port Source
                    HIJ 	        [12:10]: [H: 12] EINJ, [I: 11] Cross Contain, [J: 10] Severity
                        KKKKK	    [K: 9:5] Subcode
                            LLLLL	[L: 4:0] Code
    [31:26]	RSVD
    [25:20]	Port Down ReasonCode
    [19:18]	Port Status
    [17:13]	Port Source
    [12]	Injected:	      i0, i1
    [11]	Cross Contain:  xC0, Xc1
    [10]	Severity:	      Fatal Non-Fatal
    [9:5]	Subcode:	      Plain text in Xid message
    [4:0]	Code
    """
    pattern = r"(?P<event>\w+)\s+(?P<severity>\w+)\s+(?P<cross_contain>\w+)\s+(?P<injected>\w+)\s+Link\s+(?P<link>-?\d+)\s+\((?P<intrInfo>\w+)\s+(?P<errorStatus>\w+)\s+(?P<errorDebugData_0>\w+)\s+(?P<errorDebugData_1>\w+)\s+(?P<errorDebugData_2>\w+)\s+(?P<errorDebugData_3>\w+)\)"
    match = re.search(pattern, extra)
    if not match:
        return {}

    info = match.groupdict()
    intrInfo = int(info["intrInfo"], 16)
    errorStatus = int(info["errorStatus"], 16)
    RSVD = (intrInfo >> 26) & 0x3F
    Port_Down_ReasonCode = PORT_DOWN_REASON_CODE((intrInfo >> 20) & 0x3F)
    Port_Status = (intrInfo >> 18) & 0x3
    Port_Source = (intrInfo >> 13) & 0x1F
    Injected = (intrInfo >> 12) & 0x1
    Cross_Contain = (intrInfo >> 11) & 0x1
    Severity = (intrInfo >> 10) & 0x1
    Subcode = NVLINK5_SUBCODE((intrInfo >> 5) & 0x1F)
    Code = NVLINK5_CODE(intrInfo & 0x1F)

    # xid alone is not unique, so we need to add subcode and port down reason code
    unique_id = f"{xid}-{Subcode.name}-{Port_Down_ReasonCode.name}-{errorStatus}"

    nvlink5_info = nvlink5_catalog.get(unique_id)
    if not nvlink5_info:
        return {}

    return nvlink5_info


def get_catalog():
    if not xid_catalog:
        for k, v in xpa_xid_catalog.items():
            row = v.copy()
            id = int(k)
            # shortern the mnemonic to make it more readable in dashboard
            row["mnemonic"] = (
                row["mnemonic"]
                .replace("FIFO_ERROR_", "")
                .replace("ROBUST_CHANNEL_", "")
                .replace("NVSWITCH_ERR_HW_", "")
            )

            row["nice_name"] = f"{row['type']}-{id}-{row['mnemonic']}"
            xid_catalog[id] = row

    return xid_catalog


def get_entry(xid):
    get_catalog()

    if xid in xid_catalog:
        return xid_catalog[xid]

    return None


def parse_syslog_events(
    syslog_records: Iterable[tuple[Source, str]],
    job_info: JobInfo,
    driver_version: str = "525.105.17",
    verbose: bool = True,
):
    fleet_data = nvidia_xid_analyzer.FleetData()

    nodes_with_interruption = []
    nvlink_xid_entries = []

    for record, machine in syslog_records:
        dt_event_time = record.time
        fmt_string = "%Y-%m-%d %H:%M:%S"
        time_str = dt_event_time.strftime(fmt_string) if dt_event_time is not None else ""
        xid_context = record.datum
        if "NVRM: Xid" in xid_context:
            # Extract XID number first to determine processing path
            xid_match = re.search(r":\s*(\d+),", xid_context)
            xid_number = int(xid_match.group(1)) if xid_match else None

            if xid_number and 144 <= xid_number <= 150:
                # Special handling for NVLink XIDs - don't use nvidia_xid_analyzer
                # Extract error code and error status
                error_match = re.search(r"\((0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)", xid_context)
                error_code = error_match.group(1) if error_match else None
                error_status = error_match.group(2) if error_match else None

                if error_code:
                    decoded_data = decode_error_event(error_code)
                    # Convert error_status to decimal for symptom
                    error_status_decimal = str(int(error_status, 16)) if error_status else "0"
                    symptom_nvl = create_symptom_nvl(xid_number, decoded_data, error_status_decimal)

                    subcode = decoded_data.get("subcode", "")

                    if xid_number == 145:
                        job_disruption, failure_category = get_xid_145_attribution(subcode, error_status_decimal)

                        if job_disruption is None:
                            job_disruption = "No"
                        if failure_category is None:
                            failure_category = "Not Applicable"
                        if failure_category == "SW":
                            # This can either be InfraSW or ApplicationSW. The XID catalog
                            # attributes this class to the platform software stack, so it is
                            # narrowed to InfraSW here.
                            failure_category = "InfraSW"
                        elif failure_category == "ShouldNotOccur":
                            failure_category = "Undetermined"

                    else:
                        # Check for "Nonfatal" in syslog message to determine job_disruption,
                        # Eventually, job_disruption needs to evolve to include user-space failures.
                        job_disruption = "No" if "Nonfatal" in xid_context else "Yes"
                        failure_category = "InfraHW" if job_disruption == "Yes" else "Not Applicable"

                    # For XID-145, create dynamic symptom for outcome calculation
                    dynamic_symptom = None
                    if xid_number == 145 and failure_category and failure_category != "Not Applicable":
                        dynamic_symptom = f"XID-{xid_number}-{failure_category}"

                    nvlink_xid_entries.append(
                        {
                            "node": machine,
                            "source": record,
                            "category": "NvLink XID",
                            "sub_category": symptom_nvl,
                            "job_disruption": job_disruption,
                            "failure_category": failure_category,
                            "xid_pattern": [str(xid_number)],
                            "xid_number": xid_number,
                            "code": decoded_data.get("code", ""),
                            "subcode": subcode,
                            "port_down_reason_code": decoded_data.get("port_down_reason_code", ""),
                            "dynamic_symptom": dynamic_symptom,  # XID-145-X or None
                        }
                    )
            else:
                # Normal processing for other XIDs
                xid_entry = nvidia_xid_analyzer.XIDEntry.from_message(
                    timestamp=time_str, message=xid_context, driver=driver_version, machine=machine
                )
                xid_entry.source = record
                fleet_data._add_xid_entry(xid_entry, drop_duplicates=True)

    has_xids = any(len(node.xid_entries) > 0 for node in fleet_data.nodes)
    if not has_xids:
        return nvlink_xid_entries

    fleet_data.find_resolutions()
    fleet_data.tag_xids()
    fleet_data.group_xids_into_bursts()
    fleet_data.match_bursts_to_patterns()

    for node in fleet_data.nodes:
        if node.interruptions_count <= 0 and "internal pointer error" not in xid_context:
            continue

        for burst in node.xid_bursts:
            nodes_with_interruption.append(
                {
                    "node": node.name,
                    "source": burst.sources[0],
                    "category": burst.category,
                    "sub_category": burst.sub_category,
                    "job_disruption": "Yes" if burst.is_interruption else "No",
                    "xid_pattern": sorted(list(burst.xids)),
                }
            )

    # Add NVLINK XIDs to the results
    nodes_with_interruption.extend(nvlink_xid_entries)

    return nodes_with_interruption


if __name__ == "__main__":
    for i in [13, 31, 95]:
        entry = get_entry(i)
        print(entry)
