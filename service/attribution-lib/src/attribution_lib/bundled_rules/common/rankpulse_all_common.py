# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RankPulse callstack processing and combined display formatting.

Handles parsing and grouping of RankPulse callstacks, and provides the
shared formatting function that combines NcclWatchdog and RankPulse data
for human-readable display.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Self

from attribution_lib.sources import Source

from .nccl_watchdog import NcclWatchdogData

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Callbacks:
    """Data for a single rankpulse callstack group."""

    ranks: list[int] = field(default_factory=list)
    timestamp: str = ""
    callstacks: list[str] = field(default_factory=list)
    in_collection: bool = False  # Internal flag for tracking collection state


@dataclass
class RankpulseData:
    """Combined RankPulse callstack data."""

    source: Source
    rankpulse_callstacks: dict[int, Callbacks] = field(default_factory=dict)  # rank -> data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Reconstruct RankpulseData from dict."""
        source = Source.from_dict(data["source"])

        # Reconstruct rankpulse_callstacks
        rankpulse_callstacks = {}
        for k, v in data.get("rankpulse_callstacks", {}).items():
            if isinstance(v, dict):
                rankpulse_callstacks[int(k)] = Callbacks(
                    ranks=v.get("ranks", []),
                    timestamp=v.get("timestamp", ""),
                    callstacks=v.get("callstacks", []),
                    in_collection=v.get("in_collection", False),
                )

        return cls(
            source=source,
            rankpulse_callstacks=rankpulse_callstacks,
        )


@dataclass
class GroupedCallbacks:
    """Grouped rankpulse callstacks for deduplication display."""

    ranks: list[int] = field(default_factory=list)
    timestamps: dict[int, str] = field(default_factory=dict)
    example: Optional[list[str]] = None  # Example callstack content


# ============================================================================
# Regular Expression Patterns
# ============================================================================

# Rankpulse callstack patterns
# Example: [RANK 2] rankpulse - INFO: CPU hang detected at [2025-10-14 03:01:37]: GPU 0% for 5s
# Example with prefix: 2: [RANK 2] rankpulse - INFO: CPU hang detected at [2025-10-14 07:12:06]: GPU 0% for 30s
RANKPULSE_HEADER_PATTERN = re.compile(r"\[RANK (\d+)\] rankpulse - INFO: CPU hang detected at \[([^\]]+)\]:")
# Example: [RANK 2] rankpulse - INFO:   File "/path/to/file.py", line 143 in function
RANKPULSE_LINE_PATTERN = re.compile(r"\[RANK (\d+)\] rankpulse - INFO: (.*)$")
# Example: [RANK 2] rankpulse - INFO: ----- Finished -----
RANKPULSE_END_PATTERN = re.compile(r"\[RANK (\d+)\] rankpulse - INFO: ----- Finished")


# ============================================================================
# Rankpulse Processing
# ============================================================================


def _format_rank_list(rank_list: list[int]) -> str:
    """Format rank list with ranges: [1,2,3,5] -> "1-3,5"."""
    rank_list = sorted(set(rank_list))
    if not rank_list:
        return ""

    result = []
    start = end = rank_list[0]

    for r in rank_list[1:]:
        if r == end + 1:
            end = r
        else:
            result.append(str(start) if start == end else f"{start}-{end}")
            start = end = r

    result.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(result)


def process_rankpulse_header(line: str, rankpulse_callstacks: dict[int, Callbacks]) -> None:
    """Process rankpulse callstack header line.

    Example: [RANK 2] rankpulse - INFO: CPU hang detected at [2025-10-14 03:01:37]: GPU 0% for 5s
    """
    rankpulse_header_match = RANKPULSE_HEADER_PATTERN.search(line)
    if rankpulse_header_match:
        rank_num = int(rankpulse_header_match.group(1))
        timestamp = rankpulse_header_match.group(2)
        # Always reset/update when we see a new header (new callstack collection)
        rankpulse_callstacks[rank_num] = Callbacks(timestamp=timestamp, in_collection=True)
        return


def process_rankpulse_line(line: str, rankpulse_callstacks: dict[int, Callbacks]) -> None:
    """Process rankpulse callstack content line.

    Handles content lines and end markers:
    - Content: [RANK 2] rankpulse - INFO:   File "/path/to/file.py", line 143 in function
    - End marker: [RANK 2] rankpulse - INFO: ----- Finished -----
    """
    # Check for end marker first: [RANK 2] rankpulse - INFO: ----- Finished -----
    end_match = RANKPULSE_END_PATTERN.search(line)
    if end_match:
        rank_num = int(end_match.group(1))
        if rank_num in rankpulse_callstacks:
            rankpulse_callstacks[rank_num].in_collection = False
        return

    # Process regular lines: [RANK 2] rankpulse - INFO: ...
    line_match = RANKPULSE_LINE_PATTERN.search(line)
    if line_match:
        rank_num = int(line_match.group(1))
        content = line_match.group(2)

        # Initialize if needed
        if rank_num not in rankpulse_callstacks:
            rankpulse_callstacks[rank_num] = Callbacks(timestamp="N/A")

        # Only collect if we're in collection mode
        if rankpulse_callstacks[rank_num].in_collection:
            # Skip debug lines and header separators (CPU callstack, CUDA debug info, etc.)
            if (
                not content.startswith("GPU utilization:") and not content.startswith("-----") and content.strip()
            ):  # Not empty
                rankpulse_callstacks[rank_num].callstacks.append(content)
        return


def normalize_rankpulse_callstack(callstacks: list[str]) -> str:
    """Normalize rankpulse callstack for deduplication.

    This function normalizes thread IDs, addresses, and timestamps to group similar callstacks
    together regardless of specific thread numbers, memory addresses, or timing.

    Normalizations:
    - Timestamps: "at [2025-10-14 03:01:37]" → "at [<TIMESTAMP>]"
    - Current thread: "Current thread <anything> (" → "Current thread <ADDR> ("
    - Thread: "Thread <anything> (" → "Thread <ADDR> ("

    Examples:
    - "CPU hang detected at [2025-10-14 03:01:37]:" → "CPU hang detected at [<TIMESTAMP>]:"
    - "Current thread 0x00007644ab7ff6c0 (most..." → "Current thread <ADDR> (most..."
    - "Current thread aaa (most..." → "Current thread <ADDR> (most..."
    - "Thread 0x00007644e5de9740 (most..." → "Thread <ADDR> (most..."
    - "Thread aabbbb (most..." → "Thread <ADDR> (most..."
    """
    normalized = []
    for line in callstacks:
        normalized_line = line
        # Normalize timestamps in brackets: [2025-10-14 03:01:37] → [<TIMESTAMP>]
        normalized_line = re.sub(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", "[<TIMESTAMP>]", normalized_line)
        # Normalize "Current thread <anything> (" to "Current thread <ADDR> ("
        normalized_line = re.sub(r"Current thread [^\s]+ \(", "Current thread <ADDR> (", normalized_line)
        # Normalize "Thread <anything> (" to "Thread <ADDR> ("
        normalized_line = re.sub(r"Thread [^\s]+ \(", "Thread <ADDR> (", normalized_line)
        normalized.append(normalized_line)
    return "\n".join(normalized)


def group_rankpulse_callstacks(rankpulse_callstacks: dict[int, Callbacks]) -> dict[str, GroupedCallbacks]:
    """Group rankpulse callstacks by normalized content (dedup)."""
    grouped: dict[str, GroupedCallbacks] = {}

    for rank, data in rankpulse_callstacks.items():
        callstacks = data.callstacks
        timestamp = data.timestamp
        normalized_key = normalize_rankpulse_callstack(callstacks)

        # Initialize if not exists
        if normalized_key not in grouped:
            grouped[normalized_key] = GroupedCallbacks()

        grouped[normalized_key].ranks.append(rank)
        grouped[normalized_key].timestamps[rank] = timestamp
        if grouped[normalized_key].example is None:
            grouped[normalized_key].example = callstacks

    return grouped


# ============================================================================
# Formatting Functions for Display (shared by tools and Dashboard)
# ============================================================================


def format_rankpulse_all_for_display(
    nccl_watchdog_data: Optional[NcclWatchdogData],
    rankpulse_data: Optional[RankpulseData],
    max_lines_per_trace: int = 1000,
    max_bytes: Optional[int] = None,
) -> str:
    """
    Format combined NCCL watchdog and rankpulse data for human-readable display.

    This function generates output similar to analyze_stack_traces.py tool,
    suitable for both CLI tools and Dashboard rendering.

    Args:
        nccl_watchdog_data: NcclWatchdogData containing pytorch_stacks and watchdog_timeouts
        rankpulse_data: RankpulseData containing rankpulse_callstacks
        max_lines_per_trace: Maximum number of lines to show per trace (default: 1000)
        max_bytes: Maximum bytes for output (default: None, no limit).
                   If set, output will be truncated to fit within this limit.
                   Useful for Elasticsearch field limit (32766 bytes).

    Returns:
        Formatted string ready for display
    """
    pytorch_stacks = nccl_watchdog_data.stacktrace_by_key if nccl_watchdog_data is not None else {}
    watchdog_timeouts = nccl_watchdog_data.timeouts_by_key if nccl_watchdog_data is not None else {}
    rankpulse_callstacks = rankpulse_data.rankpulse_callstacks if rankpulse_data is not None else {}

    if not pytorch_stacks and not watchdog_timeouts and not rankpulse_callstacks:
        return ""

    output = []

    # Count total ranks with PyTorch stack info
    if pytorch_stacks:
        total_ranks = sum(len(data.ranks) for data in pytorch_stacks.values())
        output.append(f"Found {total_ranks} ranks with stack trace info\n")
        output.append("=" * 80)
        output.append("")

        # Display PyTorch NCCL stack traces
        for trace, data in pytorch_stacks.items():
            rank_str = _format_rank_list(data.ranks)
            first_rank = sorted(data.ranks)[0]
            timestamp_str = data.timestamp_from_rank.get(first_rank, "N/A")
            output.append(f"=== Rank [{rank_str}] [{timestamp_str}] ===")

            trace_lines = trace.split("\n")
            for line in trace_lines[:max_lines_per_trace]:
                output.append(line)
            if len(trace_lines) > max_lines_per_trace:
                output.append(f"... ({len(trace_lines) - max_lines_per_trace} more lines)")
            output.append("")

    # Display watchdog timeouts
    if watchdog_timeouts:
        output.append("Watchdog NCCL collective timeout summary:")
        output.append("")

        for event, timeout_data in watchdog_timeouts.items():
            rank_str = _format_rank_list(timeout_data.ranks)
            first_rank = sorted(timeout_data.ranks)[0]
            timestamp_str = timeout_data.timestamp_from_rank.get(first_rank, "N/A")
            output.append(f"Rank [{rank_str}] [{timestamp_str}]:")
            output.append(f"  {timeout_data.example}")
            output.append("")

    # Display rankpulse callstacks
    if rankpulse_callstacks:
        # Filter out ranks with actual callstacks
        ranks_with_callstacks = {
            rank: data for rank, data in rankpulse_callstacks.items() if data.callstacks and len(data.callstacks) > 0
        }

        if ranks_with_callstacks:
            output.append("\n" + "=" * 80)
            output.append("\nRankpulse triggered callstack summary:")
            output.append("")

            # Group callstacks by normalized content
            grouped = group_rankpulse_callstacks(ranks_with_callstacks)

            for normalized_key, grouped_data in grouped.items():
                rank_str = _format_rank_list(grouped_data.ranks)
                first_rank = sorted(grouped_data.ranks)[0]
                timestamp_str = grouped_data.timestamps.get(first_rank, "N/A")

                output.append(f"=== Rank [{rank_str}] [{timestamp_str}] ===")
                callstacks = grouped_data.example
                if callstacks:
                    for line in callstacks[:max_lines_per_trace]:
                        output.append(line)
                    if len(callstacks) > max_lines_per_trace:
                        output.append(f"... ({len(callstacks) - max_lines_per_trace} more lines)")
                output.append("")

    result = "\n".join(output)

    # Check byte limit for Elasticsearch compatibility
    if max_bytes is not None:
        result_bytes = result.encode("utf-8")
        if len(result_bytes) > max_bytes:
            # Truncate to fit within byte limit, leaving room for warning message
            warning_msg = f"\n\n[TRUNCATED: Output exceeded {max_bytes} bytes limit (Elasticsearch field limit). Full data available in raw logs.]"
            warning_bytes = warning_msg.encode("utf-8")
            available_bytes = max_bytes - len(warning_bytes)

            if available_bytes > 0:
                # Truncate at character boundary (avoid splitting multi-byte UTF-8 chars)
                truncated = result_bytes[:available_bytes].decode("utf-8", errors="ignore")
                result = truncated + warning_msg
            else:
                # Warning message itself is too large, just truncate
                result = result_bytes[:max_bytes].decode("utf-8", errors="ignore")

    return result


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Data structures
    "Callbacks",
    "RankpulseData",
    "GroupedCallbacks",
    # Core API (used by applog.py)
    "process_rankpulse_header",
    "process_rankpulse_line",
    "group_rankpulse_callstacks",
    # Formatting API (used by tools and Dashboard)
    "format_rankpulse_all_for_display",
]
