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
Utility for consistent logging when log lines are ignored during processing.

This module provides a centralized way to log when log lines are dropped for
various reasons including node filtering, time boundaries, content patterns,
format issues, and other filtering criteria.

Supported filtering scenarios:
- Node filtering: Lines from nodes not in the current job's node list
- Time filtering: Lines outside the job's time boundaries
- Content filtering: Lines matching exclusion patterns such as prolog entries
- Future: Format errors, size limits, rate limiting, business rules
"""

from typing import Optional

from loguru import logger

from attribution_lib.exceptions import (
    InvalidRankError,
    UnidentifiableNodeError,
    UnknownNodeError,
    UnknownResourceError,
)
from attribution_lib.job_info import JobInfo
from attribution_lib.observation.inbound import Source
from attribution_lib.observation.processing import EntryProtocol
from attribution_lib.utils import syslog_extract_node


def _log_ignored_node_line(
    exception: UnknownResourceError,
    entry: EntryProtocol,
    job_info: JobInfo,
    source: Source,
    log_level: str = "DEBUG",
    custom_context: Optional[str] = None,
) -> None:
    # Common logging context
    job_nodes_count = len(job_info.nodes)
    line_preview = entry.body()[:80]
    line_suffix = "..." if len(entry.body()) > 80 else ""

    # Specific reasons based on exception subclasses
    match exception:
        case UnidentifiableNodeError():
            reason = "no node or rank identifier found in log line"
            detail = "unable to extract node information from content"
        case InvalidRankError():
            rank = getattr(exception, "rank", "unknown")
            reason = f"rank {rank} not found in job info"
            detail = f"rank {rank} cannot be processed (job has {job_nodes_count} nodes, max rank: {int(job_nodes_count * job_info.RANKS_PER_NODE - 1)})"
        case UnknownNodeError():
            node = getattr(exception, "node", "unknown")
            reason = f"node '{node}' not in job nodelist"
            detail = f"node '{node}' not found in job's {job_nodes_count} allocated nodes"
        case _:
            extracted_node = getattr(exception, "node", None)

            if extracted_node is None:
                extracted_node = syslog_extract_node(entry.body())

            node_str = extracted_node if extracted_node else "unknown"
            reason = f"node '{node_str}' not in job nodes"
            detail = f"generic resource error for node '{node_str}'"

    # Log the filtered line with detailed context
    if custom_context:
        log_message = f"{source} ignored: {reason}, detail={detail}, {custom_context}"
    else:
        log_message = f"{source} ignored: {reason}, detail={detail}, line={line_preview}{line_suffix}"

    # Use the appropriate log level
    getattr(logger, log_level.lower())(log_message)


def _log_ignored_time_line(
    entry: EntryProtocol, job_info: JobInfo, reason: str, source: Source, log_level: str = "DEBUG"
) -> None:
    entry_time = entry.time()
    line_preview = entry.body()[:80]
    line_suffix = "..." if len(entry.body()) > 80 else ""

    log_message = (
        f"{source} ignored: time_{reason}, "
        f"entry_time={entry_time.isoformat() if entry_time else 'None'}, "
        f"job_start={job_info.interest_start_time.isoformat()}, "
        f"line={line_preview}{line_suffix}"
    )

    getattr(logger, log_level.lower())(log_message)


def _log_ignored_content_line(
    entry: EntryProtocol, pattern: str, reason: str, source: Source, log_level: str = "DEBUG"
) -> None:
    line_preview = entry.body()[:80]
    line_suffix = "..." if len(entry.body()) > 80 else ""

    log_message = (
        f"{source} ignored: content_filter, reason={reason}, pattern='{pattern}', line={line_preview}{line_suffix}"
    )

    getattr(logger, log_level.lower())(log_message)


# Public API functions with built-in error handling


def log_ignored_node_line(
    exception: UnknownResourceError,
    entry: EntryProtocol,
    job_info: JobInfo,
    source: Source,
    log_level: str = "DEBUG",
    custom_context: Optional[str] = None,
) -> None:
    """
    Log when a log line is ignored because its node is not in the current job.

    This function provides consistent, safe logging for filtered log lines across
    all attribution components. It includes built-in error handling to protect
    critical flows (like state machines) from logging failures.

    Error handling strategy:
    - Known issues (AttributeError, TypeError, KeyError) are logged and not re-raised to protect state machines
    - Unknown issues are logged and re-raised so they can be investigated and fixed

    Args:
        exception: The UnknownResourceError that was caught
        entry: The log entry that was being processed
        job_info: Job information containing the valid node list
        source: The observation source (Source.SYSLOG, Source.APPLOG, Source.HEALTH_CHECK)
        log_level: Log level to use ("DEBUG", "INFO", "WARNING")
        custom_context: Optional custom context to replace the "line=" part (e.g., "skipping straggler data with score 214.52")
    """
    try:
        _log_ignored_node_line(exception, entry, job_info, source, log_level, custom_context)
    except (AttributeError, TypeError, KeyError) as e:
        logger.error(f"Known error in node logline filtering for {source}: {type(e).__name__}: {e}")
    except Exception as e:
        logger.error(f"Unknown error in node logline filtering for {source}: {type(e).__name__}: {e}")
        raise


def log_ignored_time_line(
    entry: EntryProtocol, job_info: JobInfo, reason: str, source: Source, log_level: str = "DEBUG"
) -> None:
    """
    Log when a log line is ignored because it's outside the job's time boundaries.

    This function provides consistent logging with error handling.
    Safe to raise exceptions since it's not called from state machines.

    Error handling strategy:
    - Known issues (AttributeError, TypeError, KeyError) are logged and not re-raised
    - Unknown issues are logged and re-raised so they can be investigated and fixed

    Args:
        entry: The log entry that was being processed
        job_info: Job information containing time boundaries
        reason: Reason for time filtering ("before_job_start", "after_job_end")
        source: The observation source
        log_level: Log level to use
    """
    try:
        _log_ignored_time_line(entry, job_info, reason, source, log_level)
    except (AttributeError, TypeError, KeyError) as e:
        logger.error(f"Known error in time logline filtering for {source}: {type(e).__name__}: {e}")
    except Exception as e:
        logger.error(f"Unknown error in time logline filtering for {source}: {type(e).__name__}: {e}")
        raise


def log_ignored_content_line(
    entry: EntryProtocol, pattern: str, reason: str, source: Source, log_level: str = "DEBUG"
) -> None:
    """
    Log when a log line is ignored due to content filtering.

    This function provides consistent logging with error handling.
    Safe to raise exceptions since it's not called from state machines.

    Error handling strategy:
    - Known issues (AttributeError, TypeError, KeyError) are logged and not re-raised
    - Unknown issues are logged and re-raised so they can be investigated and fixed

    Args:
        entry: The log entry that was being processed
        pattern: The pattern that caused the filtering
        reason: Human-readable reason for filtering
        source: The observation source
        log_level: Log level to use
    """
    try:
        _log_ignored_content_line(entry, pattern, reason, source, log_level)
    except (AttributeError, TypeError, KeyError) as e:
        logger.error(f"Known error in content logline filtering for {source}: {type(e).__name__}: {e}")
    except Exception as e:
        logger.error(f"Unknown error in content logline filtering for {source}: {type(e).__name__}: {e}")
        raise
