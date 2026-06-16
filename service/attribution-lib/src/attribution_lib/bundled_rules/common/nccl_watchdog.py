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
NCCL watchdog processing utilities.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import reduce
from itertools import pairwise
from math import gcd
from typing import NamedTuple, Optional, Self, Sequence

from attribution_lib.events import EventStream
from attribution_lib.exceptions import ParseStateError
from attribution_lib.sources import Source

# Example: [rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] Stack trace of the failed collective:
STACK_TRACE_HEADER_PATTERN = re.compile(r"\[rank(\d+)\]:\[(E\d+ [\d:.]+)[^\]]*\] Stack trace of the failed collective:")
# Example: #0 at /opt/pytorch/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:603
STACK_TRACE_LINE_PATTERN = re.compile(r"#\d+\s+")

# Example: [rank0]:[E1014 22:57:41.268311790 ProcessGroupNCCL.cpp:685] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(...)
WATCHDOG_PATTERN = re.compile(
    r"\[rank(\d+)\]:\[(E\d+ [\d:.]+)[^\]]*\]\s+\[Rank\s+\d+\]\s+Watchdog caught collective operation timeout:\s+(WorkNCCL\([^)]+\)[^\n]*?before timing out\.)",
    re.MULTILINE,
)

# Default values for the stack trace and lookahead state machine.
_DEFAULT_MAX_STACK_TRACES = 100
_DEFAULT_MAX_LOOKAHEAD = 20

# Minimum coverage of the inferred stride pattern to be considered valid.
_MIN_STRIDE_PATTERN_COVERAGE = 0.5


@dataclass
class Stacktrace:
    """Data for a single PyTorch stack trace group."""

    ranks: list[int] = field(default_factory=list)
    timestamp_from_rank: dict[int, str] = field(default_factory=dict)


@dataclass
class Timeouts:
    """Data for a single watchdog timeout group."""

    ranks: list[int] = field(default_factory=list)
    timestamp_from_rank: dict[int, str] = field(default_factory=dict)
    example: str = ""  # Example WorkNCCL operation


def _trace_hash(trace_lines: list[str]) -> str:
    """Generate MD5 hash for trace lines."""
    combined = "\n".join(trace_lines)
    return hashlib.md5(combined.encode()).hexdigest()


def _canonical_trace_key(trace_lines: list[str]) -> str:
    """Create canonical key from trace lines."""
    return "\n".join(trace_lines)


def _generalize_nccl_timeout(line: str) -> str:
    """Keep WorkNCCL params but ignore 'ran for X milliseconds before timing out'."""
    if match := re.match(
        r"(WorkNCCL\(SeqNum=\d+, OpType=[^,]+, NumelIn=\d+, NumelOut=\d+, Timeout\(ms\)=\d+\)).*", line
    ):
        return match.group(1)
    return line


def _process_watchdog(line: str, timeouts_by_key: dict[str, Timeouts]) -> None:
    """Process watchdog timeout line."""
    watchdog_match = WATCHDOG_PATTERN.search(line)
    if watchdog_match is None:
        return

    rank = int(watchdog_match.group(1))
    timestamp = watchdog_match.group(2)
    workdesc = watchdog_match.group(3)
    gen_key = _generalize_nccl_timeout(workdesc)

    timeouts = timeouts_by_key.setdefault(gen_key, Timeouts())
    timeouts.timestamp_from_rank[rank] = timestamp
    timeouts.ranks.append(rank)
    if timeouts.example == "":
        timeouts.example = workdesc


@dataclass
class PyTorchStackState:
    """State machine for processing PyTorch NCCL stack traces line by line."""

    # Current trace state
    in_trace: bool = field(init=False, default=False)
    rank_num: Optional[int] = field(init=False, default=None)
    timestamp: Optional[str] = field(init=False, default=None)
    stack_lines: list = field(init=False, default_factory=list)
    lookahead_counter: int = field(init=False, default=0)

    # Storage for PyTorch stacks: hash -> Stacktrace
    stack_groups: dict[str, Stacktrace] = field(init=False, default_factory=dict)
    trace_to_key: dict[str, str] = field(init=False, default_factory=dict)

    # Configuration
    max_stack_traces: int = field(default=_DEFAULT_MAX_STACK_TRACES)
    max_lookahead: int = field(default=_DEFAULT_MAX_LOOKAHEAD)

    def start_trace(self, rank_num: int, timestamp: str) -> None:
        """Start collecting a new stack trace."""
        self.in_trace = True
        self.rank_num = rank_num
        self.timestamp = timestamp
        self.stack_lines = []
        self.lookahead_counter = 0

    def add_line(self, line: str) -> None:
        """Add a stack line to current trace."""
        self.stack_lines.append(line.strip())
        self.lookahead_counter = 0

    def increment_lookahead(self) -> None:
        """Increment the lookahead counter."""
        self.lookahead_counter += 1

    def should_end_trace(self) -> bool:
        """Check if should end trace (after max_lookahead lines)."""
        return self.lookahead_counter >= self.max_lookahead

    def finalize_trace(self) -> None:
        """Finalize and save the current stack trace.

        Raises:
            ParseStateError: If called without a prior start_trace.
        """
        if self.rank_num is None or self.timestamp is None:
            raise ParseStateError(
                "PyTorchStackState.finalize_trace() called without an active trace. Was start_trace() ever called?"
            )

        if not self.stack_lines:
            self._reset_state()
            return

        if len(self.stack_groups) >= self.max_stack_traces:
            self._reset_state()
            return

        h = _trace_hash(self.stack_lines)

        if h not in self.trace_to_key:
            self.trace_to_key[h] = _canonical_trace_key(self.stack_lines)

        # Initialize or update Stacktrace
        if h not in self.stack_groups:
            self.stack_groups[h] = Stacktrace()

        self.stack_groups[h].ranks.append(self.rank_num)
        self.stack_groups[h].timestamp_from_rank[self.rank_num] = self.timestamp

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset current trace state."""
        self.in_trace = False
        self.rank_num = None
        self.timestamp = None
        self.stack_lines = []
        self.lookahead_counter = 0

    def get_grouped_traces(self) -> dict[str, Stacktrace]:
        """Get grouped PyTorch stack traces: {trace_text: Stacktrace}."""
        result = {}
        for h, data in self.stack_groups.items():
            if h in self.trace_to_key:
                result[self.trace_to_key[h]] = data
        return result


@dataclass
class NcclWatchdogData:
    """Combined data for NCCL watchdog analysis: stack traces and watchdog timeouts."""

    source: Source
    stacktrace_by_key: dict[str, Stacktrace] = field(default_factory=dict)
    timeouts_by_key: dict[str, Timeouts] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Reconstruct NcclWatchdogData from dict."""
        source = Source.from_dict(data["source"])

        # Reconstruct stacktrace_by_key
        stacktrace_by_key = {}
        for k, v in data.get("stacktrace_by_key", {}).items():
            if isinstance(v, dict):
                timestamps = {int(rk): rv for rk, rv in v.get("timestamp_from_rank", {}).items()}
                stacktrace_by_key[k] = Stacktrace(ranks=v["ranks"], timestamp_from_rank=timestamps)

        # Reconstruct timeouts_by_key
        timeouts_by_key = {}
        for k, v in data.get("timeouts_by_key", {}).items():
            if isinstance(v, dict):
                timestamps = {int(rk): rv for rk, rv in v.get("timestamp_from_rank", {}).items()}
                timeouts_by_key[k] = Timeouts(ranks=v["ranks"], timestamp_from_rank=timestamps, example=v["example"])

        return cls(
            source=source,
            stacktrace_by_key=stacktrace_by_key,
            timeouts_by_key=timeouts_by_key,
        )


@dataclass
class NcclWatchdogParseState:
    """Container for NCCL watchdog parsing state (PyTorch stacks + watchdog timeouts)."""

    source: Optional[Source] = None
    # PyTorch stack trace state machine
    pytorch_stack_state: PyTorchStackState = field(default_factory=PyTorchStackState)
    # Temporary storage for watchdog timeout info during parsing
    timeouts_by_key: dict[str, Timeouts] = field(default_factory=dict)

    def process_line(self, line: str, source: Source) -> None:
        """Process a single line for watchdog and stack trace data."""
        if self.source is None:
            self.source = source.derived(rule="Initial line for NCCLWatchdog")

        # 1. Check if this is a new stack trace header
        if header_match := STACK_TRACE_HEADER_PATTERN.search(line):
            if self.pytorch_stack_state.in_trace:
                self.pytorch_stack_state.finalize_trace()
            rank_num = int(header_match.group(1))
            timestamp = header_match.group(2)
            self.pytorch_stack_state.start_trace(rank_num, timestamp)
            return

        # 2. If we're in a trace, check if this is a stack line
        if self.pytorch_stack_state.in_trace:
            if stack_match := STACK_TRACE_LINE_PATTERN.search(line):
                # Extract the stack line starting from the match (#0, #1, etc.)
                # This removes any line prefix like "5: " before the actual stack content
                stack_line = line[stack_match.start() :]
                self.pytorch_stack_state.add_line(stack_line)
                return
            else:
                self.pytorch_stack_state.increment_lookahead()
                if self.pytorch_stack_state.should_end_trace():
                    self.pytorch_stack_state.finalize_trace()

        # 3. Process watchdog timeouts
        _process_watchdog(line, self.timeouts_by_key)

    def finalize_and_extract(self) -> Optional[NcclWatchdogData]:
        """Finalize parsing and extract consolidated NCCL watchdog data.

        Returns:
            NcclWatchdogData with pytorch_stacks and watchdog_timeouts or None if no data collected.

        Raises:
            ParseStateError: If method is called before any lines were processed.
        """
        if self.source is None:
            raise ParseStateError(
                "NcclWatchdogParseState.finalize_and_extract() called with no source. Was process_line() ever called?"
            )

        # Finalize any in-progress stack trace
        if self.pytorch_stack_state.in_trace:
            self.pytorch_stack_state.finalize_trace()

        pytorch_stacks = self.pytorch_stack_state.get_grouped_traces()

        # Return None if no data collected
        if len(pytorch_stacks) == 0 and len(self.timeouts_by_key) == 0:
            return None

        return NcclWatchdogData(
            source=self.source,
            stacktrace_by_key=pytorch_stacks,
            timeouts_by_key=self.timeouts_by_key,
        )

    @classmethod
    def collect(cls, event_stream: EventStream) -> list[NcclWatchdogData]:
        """Collect NcclWatchdogData from Application.NCCLWatchdog.report events."""
        typ_id = event_stream.mgr.types.get_id("Application.NCCLWatchdog.report")

        rows = event_stream.query_raw(
            "SELECT payload FROM events WHERE context_id=? AND event_type=? AND status=1",
            [event_stream.ctx_id, typ_id],
        )

        results = []
        for (payload_json,) in rows:
            if payload_json is None:
                continue
            payload = json.loads(payload_json)
            if (data := payload.get("data")) is not None:
                results.append(NcclWatchdogData.from_dict(data))
        return results


class StridePattern(NamedTuple):
    """The inferred stride structure of an NCCL process group.

    stride: gap between consecutive group members in global rank space
        (e.g. 8 for a DP allreduce when TP=8)
    positions: which offsets within each stride period are occupied
        (e.g. [0] for a single-position group, [0,1,2] for a multi-position one)
    """

    stride: int
    positions: Sequence[int]


def _expand_pattern_members(world_size: int, pattern: StridePattern) -> set[int]:
    """Expand a stride+positions pattern into the full rank-member set."""
    return {
        base + p for p in pattern.positions for base in range(0, world_size, pattern.stride) if base + p < world_size
    }


def _evaluate_stride_candidate(
    reported_ranks: set[int], world_size: int, stride: int
) -> Optional[tuple[StridePattern, float]]:
    """Validate one stride candidate and return its pattern and coverage."""
    # Reject trivial candidates (stride < 2) and non-sensical candidates (stride >= world_size).
    if stride < 2 or stride >= world_size or world_size % stride != 0:
        return None

    positions = sorted({r % stride for r in reported_ranks})
    candidate = StridePattern(stride=stride, positions=positions)
    cand_set = _expand_pattern_members(world_size, candidate)
    if not reported_ranks.issubset(cand_set):
        return None

    coverage = len(reported_ranks) / len(cand_set)
    if coverage < _MIN_STRIDE_PATTERN_COVERAGE:
        return None
    return candidate, coverage


def detect_stride_pattern(reported_ranks: set[int], world_size: int) -> Optional[StridePattern]:
    """Infer the stride pattern of the NCCL process group from reporting ranks.

    Runs two methods and picks the best candidate (highest coverage, smallest stride on ties):
    * Method A — GCD of consecutive rank differences: works for single-position strides.
      (e.g. assuming 8 ranks/node, stride=8 with positions=[0] means ranks 0,8,16,24,...)
    * Method B — group-start spacing: works for multi-position strides
      (e.g. assuming 8 ranks/node, stride=8 with positions=[0,1,2] means ranks 0,1,2,8,9,10,...)

    A candidate is accepted only if it covers >= 50% of the inferred group (guards against
    false inferences when ranks are sparse or noisy) and evenly divides world_size.

    Args:
        reported_ranks: The set of ranks that have reported.
        world_size: The total number of ranks in the world.

    Returns:
        The inferred stride pattern, or None if no non-trivial pattern is found.
    """
    # Cannot measure "gaps" if there are fewer than 2 ranks or no ranks in the world.
    if world_size <= 0 or len(reported_ranks) < 2:
        return None

    sorted_ranks = sorted(reported_ranks)
    gaps = tuple(r2 - r1 for r1, r2 in pairwise(sorted_ranks))

    # Method A: GCD of consecutive rank differences.
    stride_gcd = reduce(gcd, gaps)

    # Method B: split ranks into tight "clusters", measure spacing between cluster starts.
    min_gap = min(gaps)
    groups = [[sorted_ranks[0]]]
    for i, gap in enumerate(gaps, start=1):
        rank = sorted_ranks[i]
        if gap > min_gap:
            groups.append([rank])
        else:
            groups[-1].append(rank)

    stride_grouped: Optional[int] = None
    # This grouping method needs at least two "clusters" to measure inter-cluster spacing.
    # A single cluster means all ranks are within the same tight block (stride indeterminate).
    if len(groups) >= 2:
        start_gaps = (right[0] - left[0] for left, right in pairwise(groups))
        stride_grouped = reduce(gcd, start_gaps)

    # Evaluate the two methods and pick the best candidate, preferring the one with the highest coverage.
    # Ties are broken by preferring the smaller stride.
    best_cov = 0.0
    best: Optional[StridePattern] = None

    for candidate in (stride_gcd, stride_grouped):
        if candidate is None:
            continue

        evaluated = _evaluate_stride_candidate(reported_ranks, world_size, candidate)
        if evaluated is None:
            continue

        pattern, cov = evaluated
        if cov > best_cov or (cov == best_cov and (best is None or candidate < best.stride)):
            best = pattern
            best_cov = cov

    return best


def infer_process_group(reported_ranks: set[int], world_size: int) -> set[int]:
    """Return the set of ranks expected in the same NCCL collective as the reporters.

    If a pattern is found, expands it to the full set of expected ranks.
    Falls back to range(world_size) when no non-trivial pattern is detected.
    """
    pattern = detect_stride_pattern(reported_ranks, world_size)
    if pattern is None:
        return set(range(world_size))
    return _expand_pattern_members(world_size, pattern)
