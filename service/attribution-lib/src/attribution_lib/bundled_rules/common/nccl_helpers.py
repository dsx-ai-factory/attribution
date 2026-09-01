# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import Optional


# fmt: off
class IBV_WC_STATUS(IntEnum):
    IBV_WC_SUCCESS = 0                  # Operation completed successfully
    IBV_WC_LOC_LEN_ERR = 1             # Local Length Error
    IBV_WC_LOC_QP_OP_ERR = 2           # Local QP Operation Error
    IBV_WC_LOC_EEC_OP_ERR = 3          # Local EE Context Operation Error
    IBV_WC_LOC_PROT_ERR = 4            # Local Protection Error
    IBV_WC_WR_FLUSH_ERR = 5            # Work Request Flushed Error
    IBV_WC_MW_BIND_ERR = 6             # Memory Window Binding Error
    IBV_WC_BAD_RESP_ERR = 7            # Bad Response Error
    IBV_WC_LOC_ACCESS_ERR = 8          # Local Access Error
    IBV_WC_REM_INV_REQ_ERR = 9         # Remote Invalid Request Error
    IBV_WC_REM_ACCESS_ERR = 10         # Remote Access Error
    IBV_WC_REM_OP_ERR = 11             # Remote Operation Error
    IBV_WC_RETRY_EXC_ERR = 12          # Transport Retry Counter Exceeded
    IBV_WC_RNR_RETRY_EXC_ERR = 13      # RNR Retry Counter Exceeded
    IBV_WC_LOC_RDD_VIOL_ERR = 14       # Local RDD Violation Error
    IBV_WC_REM_INV_RD_REQ_ERR = 15     # Remote Invalid RD Request
    IBV_WC_REM_ABORT_ERR = 16          # Remote Aborted Error
    IBV_WC_INV_EECN_ERR = 17           # Invalid EE Context Number
    IBV_WC_INV_EEC_STATE_ERR = 18      # Invalid EE Context State Error
    IBV_WC_FATAL_ERR = 19              # Fatal Error
    IBV_WC_RESP_TIMEOUT_ERR = 20       # Response Timeout Error
    IBV_WC_GENERAL_ERR = 21            # General Error
# fmt: on

# Detailed descriptions decode map
WC_STATUS_DECODE: dict[IBV_WC_STATUS, str] = {
    IBV_WC_STATUS.IBV_WC_SUCCESS: "Operation completed successfully. Work Request ended and memory buffers are ready for reuse.",
    IBV_WC_STATUS.IBV_WC_LOC_LEN_ERR: "Local Length Error: Message size exceeds maximum supported size or atomic operation size != 8 bytes.",
    IBV_WC_STATUS.IBV_WC_LOC_QP_OP_ERR: "Local QP Operation Error: QP consistency error (e.g., UD QP with mismatched Protection Domain).",
    IBV_WC_STATUS.IBV_WC_LOC_EEC_OP_ERR: "Local EE Context Operation Error: Internal EE Context consistency error (unused).",
    IBV_WC_STATUS.IBV_WC_LOC_PROT_ERR: "Local Protection Error: Invalid Memory Region reference in scatter/gather list.",
    IBV_WC_STATUS.IBV_WC_WR_FLUSH_ERR: "Work Request Flushed Error: QP transitioned to Error State during processing.",
    IBV_WC_STATUS.IBV_WC_MW_BIND_ERR: "Memory Window Binding Error: Failed to bind Memory Window to Memory Region.",
    IBV_WC_STATUS.IBV_WC_BAD_RESP_ERR: "Bad Response Error: Unexpected transport layer opcode from responder (RC QPs).",
    IBV_WC_STATUS.IBV_WC_LOC_ACCESS_ERR: "Local Access Error: Protection error on local data buffer during RDMA Write with Immediate.",
    IBV_WC_STATUS.IBV_WC_REM_INV_REQ_ERR: "Remote Invalid Request Error: Invalid message detected (insufficient buffers, unsupported operation).",
    IBV_WC_STATUS.IBV_WC_REM_ACCESS_ERR: "Remote Access Error: Protection error on remote data buffer during RDMA or atomic operation.",
    IBV_WC_STATUS.IBV_WC_REM_OP_ERR: "Remote Operation Error: Operation failed at responder (QP error or malformed WQE).",
    IBV_WC_STATUS.IBV_WC_RETRY_EXC_ERR: "Transport Retry Counter Exceeded: No Ack/Nack received, possible connection issues.",
    IBV_WC_STATUS.IBV_WC_RNR_RETRY_EXC_ERR: "RNR Retry Counter Exceeded: Remote side likely has no posted Work Requests.",
    IBV_WC_STATUS.IBV_WC_LOC_RDD_VIOL_ERR: "Local RDD Violation Error: RDD mismatch between QP and EE Context (unused).",
    IBV_WC_STATUS.IBV_WC_REM_INV_RD_REQ_ERR: "Remote Invalid RD Request: Invalid RD message detected (Q_Key/RDD violation, unused).",
    IBV_WC_STATUS.IBV_WC_REM_ABORT_ERR: "Remote Aborted Error: Operation aborted by responder (UD/UC QPs with SRQ).",
    IBV_WC_STATUS.IBV_WC_INV_EECN_ERR: "Invalid EE Context Number: Invalid EE Context number detected (unused).",
    IBV_WC_STATUS.IBV_WC_INV_EEC_STATE_ERR: "Invalid EE Context State Error: Operation illegal for EE Context state (unused).",
    IBV_WC_STATUS.IBV_WC_FATAL_ERR: "Fatal Error: Unrecoverable error occurred.",
    IBV_WC_STATUS.IBV_WC_RESP_TIMEOUT_ERR: "Response Timeout Error: Operation timed out waiting for response.",
    IBV_WC_STATUS.IBV_WC_GENERAL_ERR: "General Error: Unspecified error not covered by other categories.",
}


def get_status_name(value: int) -> Optional[str]:
    # If it's already the correct type, return it
    if isinstance(value, IBV_WC_STATUS):
        return value.name

    # Handle integer input
    if isinstance(value, int):
        try:
            return IBV_WC_STATUS(value).name
        except ValueError:
            return None

    return None
