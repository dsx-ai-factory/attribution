# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timedelta

import pytest

from attribution_lib.exceptions import (
    InvalidRankError,
    JobInfoDuplicateFieldError,
    JobInfoNodeNameCollisionError,
    JobInfoUnevenNodeToRanksError,
)
from attribution_lib.job_info import JobInfoBuilder, UnknownNodeError, _coerce_optional_int, _coerce_tags


class TestJobInfo:
    """Unit tests for the JobInfo type."""

    def test_job_info_nodes_are_lowercase(self, simple_job_factory):
        """Nodes are normalized to lowercase at build time."""
        job_info = simple_job_factory(nodes=["node-Name01", "node-Name02"])

        assert job_info.nodes == ["node-name01", "node-name02"]

    def test_verify_node_returns_normalized_for_existing_node(self, simple_job_factory):
        """verify_node returns normalized node name regardless of input case."""
        job_info = simple_job_factory(nodes=["node-Name01"])

        assert job_info.verify_node("node-Name01") == "node-name01"
        assert job_info.verify_node("NODE-NAME01") == "node-name01"

    def test_verify_node_raises_for_unknown_node(self, simple_job_factory):
        """verify_node raises UnknownNodeError when node is not in the job at all."""
        job_info = simple_job_factory(nodes=["node-Name01"])

        with pytest.raises(UnknownNodeError):
            job_info.verify_node("node-Name02")

    def test_applog_extract_node_prefers_nccl_hostname_over_out_of_range_rank(self, simple_job_factory):
        """NCCL hostname in this job wins even when the srun rank is out of range."""
        job_info = simple_job_factory(nodes=["node-01", "node-12"], ranks_per_node=4)
        line = (
            "814: [2026-01-02 03:04:05] node-12:1234567:1234568 [2] "
            "NCCL WARN NET/IB: Got completion from peer 192.0.2.10<63270>"
        )

        assert job_info.applog_extract_node(line) == "node-12"

    def test_applog_extract_node_falls_back_to_rank_for_foreign_nccl_hostname(self, simple_job_factory):
        """Hostnames that are not in this job still map via rank when the rank is valid."""
        job_info = simple_job_factory(nodes=["node-01", "node-12"], ranks_per_node=4)
        line = "0: other-node-99:1234567:1234568 [2] NCCL WARN NET/IB: Got completion from peer 192.0.2.10<63270>"

        assert job_info.applog_extract_node(line) == "node-01"

    def test_applog_extract_node_foreign_host_keeps_srun_rank_not_cuda_device(self, simple_job_factory):
        """srun rank 0 and NCCL ``[2]`` must not be confused when they live on different nodes."""
        job_info = simple_job_factory(nodes=["node-01", "node-02", "node-12"], ranks_per_node=1)
        line = "0: other-node-99:1234567:1234568 [2] NCCL WARN NET/IB: Got completion from peer 192.0.2.10<63270>"

        assert job_info.applog_extract_node(line) == "node-01"

    def test_applog_extract_node_timestamp_nccl_uses_hostname_not_rank_zero(self, simple_job_factory):
        """Known NCCL host wins; timestamp parsing must not substitute rank 0's node."""
        job_info = simple_job_factory(nodes=["node-01", "node-02", "node-12"], ranks_per_node=1)
        line = (
            "[2026-01-02 03:04:05] node-12:1234567:1234568 [2] "
            "NCCL WARN NET/IB: Got completion from peer 192.0.2.10<63270>"
        )

        assert job_info.applog_extract_node(line) == "node-12"

    def test_applog_extract_node_foreign_timestamp_nccl_does_not_use_rank_zero(self, simple_job_factory):
        """Foreign host + no srun rank must not map onto rank 0 (or cudaDev 2)."""
        job_info = simple_job_factory(nodes=["node-01", "node-02", "node-12"], ranks_per_node=1)
        line = (
            "[2026-01-02 03:04:05] other-node-99:1234567:1234568 [2] "
            "NCCL WARN NET/IB: Got completion from peer 192.0.2.10<63270>"
        )

        with pytest.raises(UnknownNodeError):
            job_info.applog_extract_node(line)

    def test_applog_extract_node_raises_for_foreign_hostname_and_invalid_rank(self, simple_job_factory):
        """Reject a foreign NCCL hostname when its explicit rank is also invalid."""
        job_info = simple_job_factory(nodes=["node-01", "node-12"], ranks_per_node=4)
        line = (
            "814: [2026-01-02 03:04:05] other-node-99:1234567:1234568 [2] "
            "NCCL WARN NET/IB: Got completion from peer 192.0.2.10<63270>"
        )

        with pytest.raises(InvalidRankError):
            job_info.applog_extract_node(line)

    def test_time_limit_uses_parent_job_interval_not_interest_interval(self, timezone):
        start = datetime.fromtimestamp(1700000000, tz=timezone)
        builder = _builder(timezone, node_to_ranks={"node-a": [0]})
        job_info = (
            builder.job_end_time(start + timedelta(minutes=240, seconds=12))
            .interest_start_time(start + timedelta(seconds=14))
            .interest_end_time(start + timedelta(minutes=240, seconds=12))
            .time_limit(240)
            .build()
        )

        assert job_info.interest_end_time - job_info.interest_start_time < timedelta(minutes=240)
        assert job_info.exceeded_time_limit() is True

    def test_missing_time_limit_does_not_exceed(self, simple_job_factory):
        job_info = simple_job_factory(time_limit=None)

        assert job_info.exceeded_time_limit() is False


def _builder(timezone, *, node_to_ranks, **overrides) -> JobInfoBuilder:
    kwargs = dict(
        id="job-1",
        type="training",
        state="RUNNING",
        cluster="test-cluster",
        job_start_time=datetime.fromtimestamp(1700000000, tz=timezone),
        framework="pytorch",
        source_name="test",
        timezone=timezone,
        node_to_ranks=node_to_ranks,
    )
    kwargs.update(overrides)
    return JobInfoBuilder(**kwargs)


class TestNodeToRanksBuilder:
    """Builder behavior for the node_to_ranks ctor kwarg."""

    def test_stores_mapping_and_derives_legacy_fields(self, simple_job_factory):
        mapping = {"node-a": [0, 1, 2, 3], "node-b": [4, 5, 6, 7]}

        job_info = simple_job_factory(node_to_ranks=mapping)

        assert job_info.node_from_rank == {
            0: "node-a",
            1: "node-a",
            2: "node-a",
            3: "node-a",
            4: "node-b",
            5: "node-b",
            6: "node-b",
            7: "node-b",
        }
        assert job_info.nodes == ["node-a", "node-b"]
        assert job_info.RANKS_PER_NODE == 4.0
        assert job_info.num_gpus == 8

    def test_empty_mapping_represents_never_ran(self, timezone):
        builder = _builder(timezone, node_to_ranks={})
        builder.job_end_time(datetime.fromtimestamp(1700001000, tz=timezone))
        job_info = builder.build()

        assert job_info.node_from_rank == {}
        assert job_info.nodes == []
        assert job_info.RANKS_PER_NODE == 0.0

    def test_uneven_per_node_lengths_raise(self, timezone):
        with pytest.raises(JobInfoUnevenNodeToRanksError):
            _builder(timezone, node_to_ranks={"node-a": [0, 1, 2, 3, 4, 5, 6, 7], "node-b": [8]})

    def test_case_collision_raises(self, timezone):
        with pytest.raises(JobInfoNodeNameCollisionError) as exc:
            _builder(timezone, node_to_ranks={"NodeA": [0, 1], "nodea": [2, 3]})
        assert exc.value.original_keys == ["NodeA", "nodea"]

    def test_non_participating_nodes_keep_empty_rank_lists(self, simple_job_factory):
        """Nodes mapping to [] are members that ran no collective ranks (e.g. dataflow stages):
        accepted alongside participating nodes, counted in `nodes`, but carry no ranks."""
        mapping = {"node-a": [0, 1, 2, 3], "node-b": [4, 5, 6, 7], "node-c": [], "node-d": []}

        job_info = simple_job_factory(node_to_ranks=mapping)

        # all four nodes are job members and attributable by hostname...
        assert job_info.nodes == ["node-a", "node-b", "node-c", "node-d"]
        assert job_info.verify_node("node-c") == "node-c"
        # ...but only the participating nodes carry ranks
        assert job_info.RANKS_PER_NODE == 4.0
        assert job_info.num_gpus == 8
        assert job_info.node_from_rank == {
            0: "node-a",
            1: "node-a",
            2: "node-a",
            3: "node-a",
            4: "node-b",
            5: "node-b",
            6: "node-b",
            7: "node-b",
        }

    def test_uneven_widths_ignore_empty_lists(self, timezone):
        """The uniform-width check only applies to participating nodes; empty lists don't trip it."""
        builder = _builder(timezone, node_to_ranks={"node-a": [0, 1, 2, 3], "node-b": []})
        builder.job_end_time(datetime.fromtimestamp(1700001000, tz=timezone))
        job_info = builder.build()

        assert job_info.nodes == ["node-a", "node-b"]
        assert job_info.RANKS_PER_NODE == 4.0

    def test_all_empty_rank_lists_yield_zero_width(self, timezone):
        """A non-empty mapping where every node is non-participating: width 0, no ranks, but the
        nodes are still present (distinct from the empty-mapping never-ran case)."""
        builder = _builder(timezone, node_to_ranks={"node-a": [], "node-b": []})
        builder.job_end_time(datetime.fromtimestamp(1700001000, tz=timezone))
        job_info = builder.build()

        assert job_info.nodes == ["node-a", "node-b"]
        assert job_info.RANKS_PER_NODE == 0.0
        assert job_info.num_gpus == 0
        assert job_info.node_from_rank == {}


class TestUniformNodeToRanks:
    """The JobInfoBuilder.uniform_node_to_ranks helper for callers that have
    a node list + per-node rank count instead of an explicit mapping."""

    def test_packs_ranks_contiguously_per_node(self):
        assert JobInfoBuilder.uniform_node_to_ranks(["node-a", "node-b"], 4) == {
            "node-a": [0, 1, 2, 3],
            "node-b": [4, 5, 6, 7],
        }

    def test_empty_node_list_yields_empty_mapping(self):
        assert JobInfoBuilder.uniform_node_to_ranks([], 8) == {}

    def test_single_node_with_single_rank(self):
        assert JobInfoBuilder.uniform_node_to_ranks(["solo"], 1) == {"solo": [0]}

    def test_starts_component_at_global_rank_offset(self):
        assert JobInfoBuilder.uniform_node_to_ranks(["component-node-a", "component-node-b"], 4, start_rank=8) == {
            "component-node-a": [8, 9, 10, 11],
            "component-node-b": [12, 13, 14, 15],
        }


class TestNodeToRanksLookup:
    """Lookup methods consult the explicit dict when present."""

    def test_get_name_from_rank_uses_explicit_mapping(self, simple_job_factory):
        # Non-contiguous ranks: positional heuristic would lie, but explicit mapping is exact.
        job_info = simple_job_factory(node_to_ranks={"node-a": [10, 11], "node-b": [20, 21]})

        assert job_info.get_name_from_rank(10) == "node-a"
        assert job_info.get_name_from_rank(21) == "node-b"

    def test_get_name_from_rank_raises_for_unknown_rank(self, simple_job_factory):
        job_info = simple_job_factory(node_to_ranks={"node-a": [0, 1], "node-b": [2, 3]})

        with pytest.raises(InvalidRankError):
            job_info.get_name_from_rank(99)

    def test_verify_node_round_trips_via_derived_nodes(self, simple_job_factory):
        job_info = simple_job_factory(node_to_ranks={"NODE-A": [0, 1], "node-b": [2, 3]})

        assert job_info.verify_node("Node-A") == "node-a"
        assert job_info.verify_node("node-b") == "node-b"


class TestJobInfoTags:
    """SlurmDB tag coercion and the builder duplicate-field guard."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (["nvrx"], frozenset({"nvrx"})),
            (["nvrx", "foo"], frozenset({"nvrx", "foo"})),
            (("nvrx",), frozenset({"nvrx"})),
            ({"nvrx"}, frozenset({"nvrx"})),
            ("nvrx", frozenset({"nvrx"})),  # a bare string must not split into characters
            (None, frozenset()),
            ("", frozenset()),
            ([], frozenset()),
        ],
    )
    def test_coerce_tags(self, raw, expected):
        assert _coerce_tags(raw) == expected

    def test_tags_setter_rejects_duplicate(self, timezone):
        builder = _builder(timezone, node_to_ranks={})
        builder.tags(frozenset({"nvrx"}))
        with pytest.raises(JobInfoDuplicateFieldError):
            builder.tags(frozenset({"other"}))

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("0", 0),  # exit code 0 must be preserved, not dropped as falsy
            (0, 0),
            ("1", 1),
            (143, 143),
            (None, None),
            ("", None),  # SlurmDB empty field must not reach int("") -> ValueError
        ],
    )
    def test_coerce_optional_int(self, raw, expected):
        assert _coerce_optional_int(raw) == expected
