# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for scrub-phase refiners and scrub helper utilities."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import attribution_lib
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.bundled_rules.common.scrub_helpers import (
    clean_bad_nodes_of_cascaded_symptoms,
    filter_xid_145_cascade_symptoms,
    get_clean_bad_nodes,
    is_non_node_attributable_symptom,
)
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.core import get_bundled_rules_path
from attribution_lib.events import EventManager
from attribution_lib.filter import run_scrub_phase_refiners
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import Format as ObsFormat, Id, Source as ObsSource, TimeInterval
from attribution_lib.resources import AllJobResources, Node, ResourceSet
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.sources import Source, SourceSet, SymptomSet


@pytest.fixture(autouse=True)
def rules_engine():
    original_engine = attribution_lib._rules_engine
    loader = RuleLoader()
    rules_dir = get_bundled_rules_path()
    loader.load(str(rules_dir / "filter_rules.py"))
    loader.load(str(rules_dir / "refiner_rules.py"))
    loader.load(str(rules_dir / "scrub_rules.py"))
    attribution_lib._rules_engine = loader.build()
    yield attribution_lib.get_rules_engine()
    attribution_lib._rules_engine = original_engine


@pytest.fixture
def obs_ctx():
    return ObservationContext(
        time_interval=TimeInterval(
            datetime(2024, 1, 1, 9, 0, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
        ),
        resources=ResourceSet([Node(hostname="node01")]),
        source=ObsSource.SYSLOG,
        format=ObsFormat.RAW_LOKI_STREAMS,
        timezone=ZoneInfo("UTC"),
        id=Id(999),
    )


@pytest.fixture
def job_info():
    start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    return default_job_info(job_end_time=start + timedelta(hours=1))


@pytest.fixture
def mgr(rules_engine):
    manager = EventManager(rules_engine=rules_engine)
    manager.preload_rule_names({"test"})
    yield manager
    manager.close()


@pytest.fixture
def stream(mgr):
    return mgr.create_stream("test_job")


def _make_source(obs_ctx, source_time=None, *, datum="test", line_num=1) -> Source:
    return Source(
        source_context=obs_ctx,
        line_num=line_num,
        datum=datum,
        time=source_time or datetime(2024, 1, 1, 10, 30, 0, tzinfo=ZoneInfo("UTC")),
        resources=AllJobResources,
        rule="test",
    )


def _make_bad_nodes(obs_ctx, node_symptoms: dict[str, list[str]]) -> JobNodeAttributions:
    bad_nodes = JobNodeAttributions()
    for node, syms in node_symptoms.items():
        for sym in syms:
            bad_nodes.add([node], SourceSet(sym).add(_make_source(obs_ctx)))
    return bad_nodes


def _make_symptoms(obs_ctx, *symptom_names: str) -> SymptomSet[str]:
    """Build a SymptomSet with one synthetic source-backed entry per symptom."""
    symptoms: SymptomSet[str] = SymptomSet()
    for symptom in symptom_names:
        symptoms.add(SourceSet(symptom).add(_make_source(obs_ctx)))
    return symptoms


def _bad_symptoms_by_node(bad_nodes: JobNodeAttributions) -> dict[str, set[str]]:
    """Map each node with bad attributions to its current bad symptom names."""
    return {node: set(group.symptoms()) for node, group in bad_nodes.items() if group.has_bad()}


def _run_scrub(stream, job_info, bad_nodes):
    run_scrub_phase_refiners(stream, job_info, my_bad_nodes=bad_nodes)


class TestLustreErrorCleanup:
    def test_lustre_error_sole_symptom_discarded(self, stream, obs_ctx, job_info):
        """LustreError as sole per-node symptom is removed."""
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["LustreError"]})
        _run_scrub(stream, job_info, bad_nodes)
        assert "LustreError" not in bad_nodes["node01"]

    def test_lustre_error_with_client_evicted_preserved(self, stream, obs_ctx, job_info):
        """LustreError is kept when Lustre Client Evicted is also present."""
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["LustreError", "Lustre Client Evicted"]})
        _run_scrub(stream, job_info, bad_nodes)
        assert "LustreError" in bad_nodes["node01"]

    def test_lustre_error_with_other_symptoms_preserved(self, stream, obs_ctx, job_info):
        """LustreError with other symptoms is not discarded here."""
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["LustreError", "IB Error-5 (IBV_WC_WR_FLUSH_ERR)"]})
        _run_scrub(stream, job_info, bad_nodes)
        assert "LustreError" in bad_nodes["node01"]

    def test_dataloader_exception_discarded_with_lustre_error(self, stream, obs_ctx, job_info):
        """DataLoader Exception is discarded from nodes that also have LustreError."""
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["LustreError", "DataLoader Exception"]})
        _run_scrub(stream, job_info, bad_nodes)
        assert "DataLoader Exception" not in bad_nodes["node01"]

    def test_dataloader_exception_preserved_without_lustre_error(self, stream, obs_ctx, job_info):
        """DataLoader Exception without LustreError is unaffected."""
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["DataLoader Exception"]})
        _run_scrub(stream, job_info, bad_nodes)
        assert "DataLoader Exception" in bad_nodes["node01"]


class TestIsNonNodeAttributableSymptom:
    @pytest.mark.parametrize(
        "symptom",
        [
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "IB Error-5 (IBV_WC_WR_FLUSH_ERR)",
            "IB Error",
            "local catastrophic error",
            "CE deadlock DETECTED",
            "PyTorch NCCL watchdog",
            "PyTorch NCCL watchdog cancelled",
            "pushrecv",
            "pushrecv timeout",
        ],
    )
    def test_markers_match(self, symptom):
        assert is_non_node_attributable_symptom(symptom)

    @pytest.mark.parametrize(
        "symptom",
        [
            "XID-79-GPU_OFF_BUS",
            "cudaErrorLaunchFailure",
            "Node failure reported by slurm",
            "DataLoader Exception",
            "",
        ],
    )
    def test_non_markers_do_not_match(self, symptom):
        assert not is_non_node_attributable_symptom(symptom)


class TestGetCleanBadNodesMarkerStripping:
    def test_ib_error_stripped_from_per_node(self, obs_ctx):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["IB Error-12 (IBV_WC_RETRY_EXC_ERR)"]})
        get_clean_bad_nodes(bad_nodes, cluster="test")
        assert "node01" not in bad_nodes.attributions or not bad_nodes["node01"].has_bad()

    def test_pytorch_nccl_watchdog_stripped(self, obs_ctx):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["PyTorch NCCL watchdog"]})
        get_clean_bad_nodes(bad_nodes, cluster="test")
        assert "node01" not in bad_nodes.attributions or "PyTorch NCCL watchdog" not in bad_nodes["node01"]

    def test_non_marker_symptom_preserved(self, obs_ctx):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["XID-79-GPU_OFF_BUS"]})
        get_clean_bad_nodes(bad_nodes, cluster="test")
        assert "XID-79-GPU_OFF_BUS" in bad_nodes["node01"]


class TestCascadeCleanupBadNodeFanout:
    def test_clean_bad_nodes_of_cascaded_symptoms_collapses_to_single_node_fault(self, obs_ctx):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", "hist15_high"],
                "node02": ["XID-149.0-NETIR_LINK_DOWN-TIMEOUT-0"],
                "node03": ["effective_ber_high"],
            },
        )

        clean_bad_nodes_of_cascaded_symptoms(
            bad_nodes,
            ["XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0"],
        )

        assert _bad_symptoms_by_node(bad_nodes) == {
            "node01": {"XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0"},
        }
        assert bad_nodes["node01"].attributions["hist15_high"].has_interesting()
        assert bad_nodes["node02"].attributions["XID-149.0-NETIR_LINK_DOWN-TIMEOUT-0"].has_interesting()
        assert bad_nodes["node03"].attributions["effective_ber_high"].has_interesting()

    def test_get_clean_bad_nodes_removes_multi_node_nccl_job_wide_fanout(self, obs_ctx):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["PyTorch NCCL watchdog", "IB Error-12 (IBV_WC_RETRY_EXC_ERR)"],
                "node02": ["local catastrophic error during NCCL"],
                "node03": ["pushrecv timeout on collective"],
            },
        )

        cleaned = get_clean_bad_nodes(bad_nodes, cluster="test")

        assert _bad_symptoms_by_node(cleaned) == {}
        assert cleaned["node01"].attributions["PyTorch NCCL watchdog"].has_interesting()
        assert cleaned["node01"].attributions["IB Error-12 (IBV_WC_RETRY_EXC_ERR)"].has_interesting()
        assert cleaned["node02"].attributions["local catastrophic error during NCCL"].has_interesting()
        assert cleaned["node03"].attributions["pushrecv timeout on collective"].has_interesting()

    def test_filter_xid_145_cascade_symptoms_keeps_rlw_rxpipe_primary(self, obs_ctx):
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-145-RLW_RXPIPE-8",
            "XID-145-RLW_SRC_TRACK-4",
            "XID-145-RLW_SRC_TRACK-8",
            "cudaErrorLaunchFailure",
            "cudaErrorNvlinkUncorrectable",
            "XID-45-PREEMPTIVE_REMOVAL",
            "NV_ERR_INVALID_STATE nv_gpu_ops.c",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["XID-145-RLW_RXPIPE-8", "cudaErrorLaunchFailure"],
                "node02": ["XID-145-RLW_SRC_TRACK-4", "XID-45-PREEMPTIVE_REMOVAL"],
                "node03": ["XID-145-RLW_SRC_TRACK-8", "NV_ERR_INVALID_STATE nv_gpu_ops.c"],
            },
        )

        filter_xid_145_cascade_symptoms(symptoms, bad_nodes)

        assert set(symptoms.symptoms()) == {"XID-145-RLW_RXPIPE-8"}
        assert _bad_symptoms_by_node(bad_nodes) == {
            "node01": {"XID-145-RLW_RXPIPE-8"},
        }
        assert bad_nodes["node01"].attributions["cudaErrorLaunchFailure"].has_interesting()
        assert bad_nodes["node02"].attributions["XID-145-RLW_SRC_TRACK-4"].has_interesting()
        assert bad_nodes["node02"].attributions["XID-45-PREEMPTIVE_REMOVAL"].has_interesting()
        assert bad_nodes["node03"].attributions["XID-145-RLW_SRC_TRACK-8"].has_interesting()
        assert bad_nodes["node03"].attributions["NV_ERR_INVALID_STATE nv_gpu_ops.c"].has_interesting()

    def test_filter_xid_145_cascade_symptoms_keeps_src_track_without_job_level_rxpipe(self, obs_ctx):
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-145-RLW_SRC_TRACK-4",
            "XID-145-RLW_SRC_TRACK-8",
            "cudaErrorLaunchFailure",
            "XID-45-PREEMPTIVE_REMOVAL",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["XID-145-RLW_RXPIPE-8", "cudaErrorLaunchFailure"],
                "node02": ["XID-145-RLW_SRC_TRACK-4", "XID-45-PREEMPTIVE_REMOVAL"],
                "node03": ["XID-145-RLW_SRC_TRACK-8"],
            },
        )

        filter_xid_145_cascade_symptoms(symptoms, bad_nodes)

        assert set(symptoms.symptoms()) == {
            "XID-145-RLW_SRC_TRACK-4",
            "XID-145-RLW_SRC_TRACK-8",
        }
        assert _bad_symptoms_by_node(bad_nodes) == {
            "node01": {"XID-145-RLW_RXPIPE-8"},
            "node02": {"XID-145-RLW_SRC_TRACK-4"},
            "node03": {"XID-145-RLW_SRC_TRACK-8"},
        }
        assert bad_nodes["node01"].attributions["cudaErrorLaunchFailure"].has_interesting()
        assert bad_nodes["node02"].attributions["XID-45-PREEMPTIVE_REMOVAL"].has_interesting()
