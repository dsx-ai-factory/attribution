# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State owned by one discovery call, including its rebalance callbacks."""

from dataclasses import dataclass, field

from attribution_blueprint.connectors.execution_contexts.kafka.events import KafkaExecutionContext

PartitionKey = tuple[str, int]


@dataclass
class PartitionProgress:
    """Offsets for records handled during the current ownership interval.

    first_offset is the earliest record not yet returned to the caller.
    next_offset is one past the last handled record, including intentional
    skips. Commit next_offset on success; rewind/bookmark first_offset on
    failure/revoke. Neither offset means attribution has completed.
    """

    first_offset: int
    next_offset: int


@dataclass
class DiscoveryBatch:
    """Mutable batch state shared only while discover() is active.

    partitions contains owned records awaiting a decision. contexts preserves
    poll order across partitions. Revoke/loss removes the affected records.
    committed identifies which contexts can be returned after partial failure.
    Successfully rewound partitions are removed, so cleanup cannot seek twice.
    EOF positions are fetch evidence, never offsets to commit.
    """

    partitions: dict[PartitionKey, PartitionProgress] = field(default_factory=dict)
    contexts: list[tuple[PartitionKey, KafkaExecutionContext]] = field(default_factory=list)
    committed: set[PartitionKey] = field(default_factory=set)
    commit_failures: set[PartitionKey] = field(default_factory=set)
    bookmark_failures: set[PartitionKey] = field(default_factory=set)
    end_offsets: dict[PartitionKey, int] = field(default_factory=dict)
    released_count: int = 0

    def observe(self, key: PartitionKey, offset: int) -> None:
        """Track a handled record before parsing, so skipped records advance too."""
        progress = self.partitions.setdefault(key, PartitionProgress(offset, offset + 1))
        progress.next_offset = offset + 1

    def release(self, keys: set[PartitionKey]) -> None:
        """Discard revoked/lost records; the next owner reads them again."""
        self.released_count += len(keys & self.partitions.keys())
        self.contexts = [(key, ctx) for key, ctx in self.contexts if key not in keys]
        for key in keys:
            self.partitions.pop(key, None)
            self.end_offsets.pop(key, None)


@dataclass
class PollWait:
    """Deadlines and retry count for one discovery loop.

    assignment_deadline bounds time without an assignment. fetch_deadline
    bounds empty polls while assigned partitions still contain unread records.
    A record resets both waits; an empty poll never proves a topic is drained.
    """

    assignment_deadline: float | None
    fetch_deadline: float | None = None
    transient_errors: int = 0
