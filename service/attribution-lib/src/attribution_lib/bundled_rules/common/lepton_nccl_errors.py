# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from dataclasses import dataclass

from attribution_lib.attr_sub_class import AttributionSubClassEnum


# --------------------------------------------------------------------------- #
# Data structures                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ErrorCode:
    error_class: AttributionSubClassEnum
    suggestion: str


# --------------------------------------------------------------------------- #
# Canonical NCCL error map                                                    #
# --------------------------------------------------------------------------- #
# Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t
_NCCL_ERROR_MAP: dict[str, ErrorCode] = {
    "ncclUnhandledCudaError": ErrorCode(
        AttributionSubClassEnum.NCCL_UNHANDLED_CUDA_ERROR, "ncclUnhandledCudaError(1): Call to a CUDA function failed."
    ),
    "ncclSystemError": ErrorCode(
        AttributionSubClassEnum.NCCL_SYSTEM_ERROR,
        "ncclSystemError(2): System call (e.g. socket, malloc) or external library call failed or device error.",
    ),
    "ncclInternalError": ErrorCode(
        AttributionSubClassEnum.NCCL_INTERNAL_ERROR,
        "ncclInternalError(3): An internal check failed. This is due to either a bug in NCCL or memory corruption.",
    ),
    "ncclInvalidArgument": ErrorCode(
        AttributionSubClassEnum.NCCL_INVALID_ARGUMENT, "ncclInvalidArgument(4): Invalid value for an argument."
    ),
    "ncclInvalidUsage": ErrorCode(
        AttributionSubClassEnum.NCCL_INVALID_USAGE,
        "ncclInvalidUsage(5): This usually reflects invalid usage of the NCCL library.",
    ),
    "ncclRemoteError": ErrorCode(
        AttributionSubClassEnum.NCCL_REMOTE_ERROR,
        "ncclRemoteError(6): A call failed, possibly due to a network error or a remote process exiting prematurely.",
    ),
    "ncclInProgress": ErrorCode(
        AttributionSubClassEnum.NCCL_IN_PROGRESS,
        "ncclInProgress(7): A NCCL operation on the communicator is being enqueued and progressed in the background.",
    ),
}

# Catch-all class if we only see “NCCL WARN …”
NCCL_GENERIC = AttributionSubClassEnum.NCCL
_ERR_NCCL_SUGGESTION_TMPL = "{}"

# Members that should trigger user-facing alerts
_NCCL_ALERT_CLASSES = {ec.error_class for ec in _NCCL_ERROR_MAP.values()}.union({NCCL_GENERIC})


# --------------------------------------------------------------------------- #
# Helper sub-string detectors                                                 #
# --------------------------------------------------------------------------- #
def _contains_split_share(line: str) -> bool:
    return "splitShare" in line and "is not a valid value" in line


def _contains_nccl_cga_cluster_size(line: str) -> bool:
    return "NCCL_CGA_CLUSTER_SIZE value" in line and "is too big. Limiting value to " in line


def _contains_min_max_ctas_upper_limit(line: str) -> bool:
    return ("minCTAs" in line or "maxCTAs" in line) and "is larger than #channels upper limit" in line


def _contains_min_max_ctas_compare(line: str) -> bool:
    return "minCTAs" in line and "is larger than maxCTAs" in line


def _contains_user_asked_for_channels(line: str) -> bool:
    return "User asked for a minimum of " in line and "channels, limiting to " in line


# --------------------------------------------------------------------------- #
# Public classification routine                                               #
# --------------------------------------------------------------------------- #
def is_nccl_error(
    line: str,
) -> tuple[AttributionSubClassEnum, str, bool, bool]:
    """
    Parameters
    ----------
    line : str
        Raw log line (case preserved).

    Returns
    -------
    err_class : AttributionSubClassEnum
        * NCCL_*  if matched against a specific coded error
        * NCCL    if only the generic WARN match triggered
        * ""          if not a NCCL error
    suggestion : str
        User-facing hint (may include original line for context).
    precise : bool
        True  → error matched a concrete NCCL result code.
        False → only generic “WARN” match without explicit code.
    is_program_error : bool
        True  → failure originates inside NCCL/user code.
        False → line represents benign condition or infra problem.
    """

    # ------------------------------------------------------------------ #
    # Rule 1 – filter explicit `ERR_<code>` substrings                    #
    # ------------------------------------------------------------------ #
    for pattern, ec in _NCCL_ERROR_MAP.items():
        if pattern in line:
            # Internal error is considered infra (is_program_error=False).
            if pattern == "ncclInternalError":
                return ec.error_class, ec.suggestion, True, False
            return ec.error_class, ec.suggestion, True, True

    # ------------------------------------------------------------------ #
    # Rule 2 – match generic “NCCL WARN …” lines                          #
    # ------------------------------------------------------------------ #
    if "NCCL WARN" in line:
        # Skip benign WARN flavours copied from the Go ignore list.
        if any(
            checker(line)
            for checker in (
                lambda l: "Net : Unsupported address family type" in l,
                lambda l: "Ignoring SHM locality, must be 1 (sender side) or 2 (receiver side, default)" in l,
                lambda l: "[Proxy Progress] Failed to set CUDA device" in l,
                _contains_nccl_cga_cluster_size,
                _contains_split_share,
                _contains_min_max_ctas_upper_limit,
                _contains_min_max_ctas_compare,
                _contains_user_asked_for_channels,
            )
        ):
            # Benign; treat as no NCCL failure
            return AttributionSubClassEnum.UNDETERMINED, "", False, False

        # Otherwise surface as a generic NCCL error.
        return (
            NCCL_GENERIC,
            _ERR_NCCL_SUGGESTION_TMPL.format(line),
            True,
            True,
        )

    # ------------------------------------------------------------------ #
    # No NCCL pattern matched                                            #
    # ------------------------------------------------------------------ #
    return AttributionSubClassEnum.UNDETERMINED, "", False, False


# --------------------------------------------------------------------------- #
# Human-readable convenience wrapper                                          #
# --------------------------------------------------------------------------- #
def classify_nccl_line(line: str) -> str:
    """
    Convenience routine: return a formatted string when an error is found,
    or an empty string otherwise.
    """
    err_class, suggestion, precise, _ = is_nccl_error(line)
    if err_class:
        badge = "PRECISE" if precise else "GENERIC"
        return f"[{badge}] {err_class}: {suggestion}"
    return ""
