# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for refiner rules."""

from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import attribution_lib
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.core import (
    AttributionResult,
    CalcBadNodesAttributionResult,
    calculate_symptoms,
    get_bundled_rules_path,
)
from attribution_lib.decision_trace_recorder import Action, DecisionRecorder, NodeType
from attribution_lib.events import EventManager
from attribution_lib.failed_node_provenance import FailedNodeResolution
from attribution_lib.filter import run_post_outcome_refiners, run_pre_outcome_refiners, run_scrub_phase_refiners
from attribution_lib.hooks import singlehooks
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.job_info import JobInfo, JobInfoBuilder, Scheduler
from attribution_lib.node_not_responding_provenance import NodeNotRespondingProvenance
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import Format as ObsFormat, Id, Source as ObsSource, TimeInterval
from attribution_lib.outcome import OutcomeAttribution, calculate_outcome
from attribution_lib.resources import AllJobResources, Node, ResourceSet
from attribution_lib.rule_types.definitions import RefinerContext
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.sources import Source, SourceSet, SymptomSet
from attribution_lib.symptoms.symptom_enums import RegisteredSymptoms


@pytest.fixture(autouse=True)
def rules_engine():
    """Load filter and refiner rule modules."""
    original_engine = attribution_lib._rules_engine
    loader = RuleLoader()
    rules_dir = get_bundled_rules_path()
    loader.load(str(rules_dir / "filter_rules.py"))
    loader.load(str(rules_dir / "refiner_rules.py"))
    attribution_lib._rules_engine = loader.build()

    yield attribution_lib.get_rules_engine()

    attribution_lib._rules_engine = original_engine


@pytest.fixture
def mgr(rules_engine):
    m = EventManager(rules_engine=rules_engine)
    m.preload_rule_names({"test"})
    yield m
    m.close()


@pytest.fixture
def stream(mgr):
    return mgr.create_stream("test_job")


@pytest.fixture
def obs_ctx():
    return ObservationContext(
        time_interval=TimeInterval(
            datetime(2024, 1, 1, 9, 0, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
        ),
        source=ObsSource.APPLOG,
        format=ObsFormat.ENTRY_TIMESERIES,
        timezone=ZoneInfo("UTC"),
        id=Id(999),
    )


@pytest.fixture
def job_info():
    start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    return default_job_info(job_end_time=start + timedelta(hours=1))


@pytest.fixture(autouse=True)
def _register_ctx(stream, obs_ctx):
    stream.register_observation_context(obs_ctx)


def _make_source(obs_ctx, source_time=None, *, datum="test", line_num=1, rule="test") -> Source:
    """Create a minimal Source for building SymptomSets."""
    return Source(
        source_context=obs_ctx,
        line_num=line_num,
        datum=datum,
        time=source_time or datetime(2024, 1, 1, 10, 30, 0, tzinfo=ZoneInfo("UTC")),
        resources=AllJobResources,
        rule=rule,
    )


def _make_symptoms(obs_ctx, *symptom_names, source_time=None) -> SymptomSet:
    """Build a SymptomSet from symptom name strings."""
    symptoms = SymptomSet()
    for name in symptom_names:
        symptoms.add(SourceSet(name).add(_make_source(obs_ctx, source_time)))
    return symptoms


def _make_bad_nodes(obs_ctx, node_symptoms: dict[str, list[str]], source_time=None) -> JobNodeAttributions:
    """Build JobNodeAttributions from {node: [symptom, ...]}."""
    bad_nodes = JobNodeAttributions()
    for node, syms in node_symptoms.items():
        for sym in syms:
            bad_nodes.add([node], SourceSet(sym).add(_make_source(obs_ctx, source_time)))
    return bad_nodes


def _run(
    stream,
    job_info,
    symptoms,
    bad_nodes=None,
    raw_keywords=None,
    extract_job_metadata_from_logfile=False,
):
    """Run all refiner rules (DagCascadePruning handles re-aggregation internally)."""
    if bad_nodes is None:
        bad_nodes = JobNodeAttributions()
    run_pre_outcome_refiners(
        stream,
        job_info,
        symptoms=symptoms,
        my_bad_nodes=bad_nodes,
        raw_keywords=raw_keywords,
        extract_job_metadata_from_logfile=extract_job_metadata_from_logfile,
    )


def _run_ib_cleanup(stream, job_info, symptoms, bad_nodes):
    """Run the IBSymptomCleanup refiner rule."""
    run_pre_outcome_refiners(stream, job_info, symptoms=symptoms, my_bad_nodes=bad_nodes)


def _attribute_outcome(stream, job_info, obs_ctx) -> str:
    """Run the pre-outcome -> outcome -> post-outcome chain and return the outcome.

    Mirrors core.attribute for tests that need the final outcome for a job with no
    node/log evidence, e.g. a cancellation resolved purely from SLURM state.
    """
    symptoms = _make_symptoms(obs_ctx)
    bad_nodes = JobNodeAttributions()
    run_pre_outcome_refiners(stream, job_info, symptoms=symptoms, my_bad_nodes=bad_nodes)

    result = AttributionResult(event_stream=stream, bad_nodes_attr=CalcBadNodesAttributionResult())
    result.symptoms.update(symptoms)
    result.outcome_attr = calculate_outcome(result.symptoms)
    run_post_outcome_refiners(
        stream,
        job_info,
        symptoms=result.symptoms,
        my_bad_nodes=bad_nodes,
        outcome=result.outcome_attr,
        cancelled_by=job_info.cancelled_by,
        attribution_result=result,
    )
    return result.outcome_attr.value().outcome


_DEFAULT_START = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
_DEFAULT_END = _DEFAULT_START + timedelta(hours=1)


def _job_info_with(**overrides) -> JobInfo:
    """Build the canonical bundled-rules JobInfo with optional overrides."""
    return default_job_info(job_end_time=_DEFAULT_END, **overrides)


def _slurm_job_info(**overrides) -> JobInfo:
    """Build a JobInfo explicitly identified as Slurm."""
    return default_job_info(scheduler=Scheduler.SLURM, **overrides)


def _run_scrub(stream, job_info, bad_nodes, recorder=None):
    """Run only SCRUB-phase refiners on my_bad_nodes."""
    run_scrub_phase_refiners(stream, job_info, my_bad_nodes=bad_nodes, recorder=recorder)


class TestXid140Cascade:
    _XID140 = "XID-140-UNRECOVERABLE_ECC_ERROR_ESCAPE"
    _FATAL = "NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR"
    _XID94 = "XID-94-CONTAINED_ERROR"
    _RESET = "NV_ERR_RESET_REQUIRED kernel_gsp.c"

    @pytest.mark.parametrize("trigger", [_XID140, "XID-1400-OTHER", "SXID-140-OTHER", None])
    def test_only_active_xid140_triggers_cleanup(self, stream, obs_ctx, job_info, trigger):
        """Do not suppress a valid trio for absent, demoted, or similarly named XIDs.

        Args:
            stream: Registered event stream.
            obs_ctx: Evidence source context.
            job_info: Job metadata.
            trigger: Active signal, or None to use a demoted XID-140.

        Returns:
            None.
        """
        bad_nodes = _make_bad_nodes(
            obs_ctx, {"node01": [trigger or self._XID140, self._FATAL, self._XID94, self._RESET]}
        )
        if trigger is None:
            bad_nodes["node01"].discard(self._XID140)

        _run_scrub(stream, job_info, bad_nodes)

        assert ("poisoned gsp" in bad_nodes["node01"]) == (trigger != self._XID140)
        if trigger == self._XID140:
            assert self._RESET in bad_nodes["node01"]

    def test_cleanup_preserves_unrelated_signals_and_records_provenance(self, stream, obs_ctx, job_info):
        """Limit cleanup to exact fatal and XID-94 signals while retaining sources.

        Args:
            stream: Registered event stream.
            obs_ctx: Evidence source context.
            job_info: Job metadata.

        Returns:
            None.
        """
        unrelated = [self._FATAL + "_OTHER", "XID-940-OTHER", self._RESET]
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [self._XID140, self._FATAL, self._XID94, *unrelated]})
        recorder = DecisionRecorder(job_info.id)

        _run_scrub(stream, job_info, bad_nodes, recorder=recorder)

        assert set(bad_nodes["node01"].symptoms()) == {self._XID140, *unrelated}
        for symptom in [self._FATAL, self._XID94]:
            assert bad_nodes["node01"].attributions[symptom].has_interesting()
        demoted = [
            event for event in recorder.events if event.rule_name == "Xid140Cascade" and event.action is Action.REMOVED
        ]
        assert {(event.hostname, event.symptom) for event in demoted} == {
            ("node01", self._FATAL),
            ("node01", self._XID94),
        }


class TestPoisonedGspComposite:
    _FATAL = "NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR"
    _XID94 = "XID-94-CONTAINED_ERROR"
    _RESET = "NV_ERR_RESET_REQUIRED kernel_gsp.c"

    @pytest.mark.parametrize("reset", ["NV_ERR_RESET_REQUIRED", "NV_ERR_RESET_REQUIRED ", _RESET])
    def test_combines_all_evidence_and_records_provenance(self, stream, obs_ctx, job_info, reset):
        bad_nodes = JobNodeAttributions()
        for i, symptom in enumerate([self._FATAL, self._XID94, reset]):
            source = _make_source(obs_ctx, datum=symptom, line_num=i + 1)
            bad_nodes.add(["node01"], SourceSet.new_one(symptom, source))
        recorder = DecisionRecorder(job_info.id)

        _run_scrub(stream, job_info, bad_nodes, recorder=recorder)

        sources = list(bad_nodes["node01"].attributions["poisoned gsp"])
        assert {source.datum for source in sources} == {self._FATAL, self._XID94, reset}
        assert {source.line_num for source in sources} == {1, 2, 3}
        created = [
            event
            for event in recorder.events
            if event.rule_name == "PoisonedGspComposite"
            and event.action is Action.CREATED
            and event.symptom == "poisoned gsp"
        ]
        assert len(created) == 1
        assert created[0].hostname == "node01"

    @pytest.mark.parametrize("demoted", [_FATAL, _XID94, _RESET])
    def test_demoted_evidence_does_not_complete_combination(self, stream, obs_ctx, job_info, demoted):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [self._FATAL, self._XID94, self._RESET]})
        bad_nodes["node01"].discard(demoted)

        _run_scrub(stream, job_info, bad_nodes)

        assert "poisoned gsp" not in bad_nodes["node01"]

    def test_component_pruning_preserves_evidence_on_other_nodes(self, stream, obs_ctx, job_info):
        job_info = job_info._replace(nodes=["node01", "node02"])
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._FATAL, self._XID94, self._RESET], "node02": [self._XID94, self._RESET]},
        )

        symptoms = calculate_symptoms(event_stream=stream, job_info=job_info, my_bad_nodes=bad_nodes)

        assert "poisoned gsp" in symptoms
        assert "poisoned gsp" in bad_nodes["node01"]
        assert "poisoned gsp" not in bad_nodes["node02"]
        for component in [self._FATAL, self._XID94, self._RESET]:
            assert component not in bad_nodes["node01"]
            assert bad_nodes["node01"].attributions[component].has_interesting()
        assert self._XID94 in bad_nodes["node02"]
        assert self._RESET in bad_nodes["node02"]
        assert self._XID94 in symptoms
        assert self._RESET in symptoms


class TestDetectBadNicState:
    _COMPONENTS = ("mlx5 firmware in initialization state", "mlx5 device initialization failed")

    @pytest.mark.parametrize("missing", ["timestamp", "job_end", "pci", "active_component"])
    def test_incomplete_context_does_not_combine(self, stream, obs_ctx, job_info, missing):
        bad_nodes = JobNodeAttributions()
        for i, symptom in enumerate(self._COMPONENTS):
            source = _make_source(
                obs_ctx,
                source_time=job_info.job_end_time,
                datum=f"mlx5_core 0000:01:00.0: {symptom}",
                line_num=i + 1,
            )
            if missing == "timestamp" and i == 0:
                source = source._replace(time=None)
            if missing == "pci" and i == 0:
                source = source._replace(datum=symptom)
            bad_nodes.add(["node01"], SourceSet.new_one(symptom, source))
        if missing == "job_end":
            job_info = job_info._replace(job_end_time=None)
        if missing == "active_component":
            bad_nodes["node01"].discard(self._COMPONENTS[0])

        _run_scrub(stream, job_info, bad_nodes)

        assert "detected bad NIC state" not in bad_nodes["node01"]

    def test_provenance_and_stale_sources_are_preserved(self, stream, obs_ctx, job_info):
        bad_nodes = JobNodeAttributions()
        for i, symptom in enumerate(self._COMPONENTS):
            source = _make_source(
                obs_ctx,
                source_time=job_info.job_end_time,
                datum=f"mlx5_core 0000:01:00.0: {symptom}",
                line_num=i + 1,
            )
            bad_nodes.add(["node01"], SourceSet.new_one(symptom, source))
        stale = source._replace(time=job_info.job_end_time - timedelta(minutes=3), line_num=3)
        bad_nodes.add(["node01"], SourceSet.new_one(self._COMPONENTS[1], stale))
        recorder = DecisionRecorder(job_info.id)

        _run_scrub(stream, job_info, bad_nodes, recorder=recorder)

        symptoms = bad_nodes["node01"].attributions
        assert {source.line_num for source in symptoms["detected bad NIC state"]} == {1, 2}
        assert list(symptoms[self._COMPONENTS[1]]) == [stale]
        assert all(symptoms[component].has_interesting() for component in self._COMPONENTS)
        created = [
            event
            for event in recorder.events
            if event.rule_name == "DetectBadNicState" and event.action is Action.CREATED
        ]
        assert len(created) == 1
        assert created[0].symptom == "detected bad NIC state"
        assert created[0].hostname == "node01"

    @pytest.mark.parametrize(
        "offset,expected",
        [(-120, False), (-1, False), (0, True), (4, True), (26, True), (120, True), (121, False)],
    )
    def test_post_job_window(self, stream, obs_ctx, job_info, offset, expected):
        bad_nodes = JobNodeAttributions()
        for i, symptom in enumerate(self._COMPONENTS):
            source = _make_source(
                obs_ctx,
                source_time=job_info.job_end_time + timedelta(seconds=offset),
                datum=f"mlx5_core 0000:01:00.0: {symptom}",
                line_num=i + 1,
            )
            bad_nodes.add(["node01"], SourceSet.new_one(symptom, source))
        _run_scrub(stream, job_info, bad_nodes)
        assert ("detected bad NIC state" in bad_nodes["node01"]) is expected

    def test_missing_job_start_does_not_prevent_detection(self, stream, obs_ctx, job_info):
        bad_nodes = JobNodeAttributions()
        for i, component in enumerate(self._COMPONENTS):
            source = _make_source(
                obs_ctx,
                source_time=job_info.job_end_time,
                datum=f"mlx5_core 0000:01:00.0: {component}",
                line_num=i + 1,
            )
            bad_nodes.add(["node01"], SourceSet.new_one(component, source))
        _run_scrub(stream, job_info._replace(job_start_time=None), bad_nodes)
        assert "detected bad NIC state" in bad_nodes["node01"]


class TestScrubBadNodes:
    """Tests for the ScrubBadNodes refiner (SCRUB phase, pre-aggregation)."""

    _CORROBORATED_FAILURE = "Corroborated node failure"

    def test_scheduler_raw_not_responding_reason_emits_renamed_symptom(self, stream, obs_ctx, job_info):
        job_info = job_info._replace(slurm_node_infos={"node01": {"reason": "Health check: Not responding"}})
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["IB Error-12 (IBV_WC_RETRY_EXC_ERR)"]})

        _run_scrub(stream, job_info, bad_nodes)

        assert "Node not responding" in bad_nodes["node01"]
        assert "Not responding" not in bad_nodes["node01"]
        source = bad_nodes["node01"].attributions["Node not responding"].one_source()
        assert source.rule == NodeNotRespondingProvenance.SLURM_NODE_REASON.value

    def test_slurm_failure_with_epilog_non_response_emits_corroborated_node_failure(self, stream, obs_ctx):
        job_info = _slurm_job_info(state="NODE_FAIL")._replace(failed_node="node01")
        epilog_source = _make_source(
            obs_ctx,
            datum="Health check failed: Not responding",
            rule=NodeNotRespondingProvenance.EPILOG_HEALTH_CHECK.value,
        )
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(
            ["node01"],
            SourceSet.new_one(RegisteredSymptoms.NODE_NOT_RESPONDING.value, epilog_source),
        )
        recorder = DecisionRecorder(job_info.id)

        _run_scrub(stream, job_info, bad_nodes, recorder=recorder)

        corroborated_failure = self._CORROBORATED_FAILURE
        assert corroborated_failure in bad_nodes["node01"]
        source_rules = {source.rule for source in bad_nodes["node01"].attributions[corroborated_failure]}
        assert source_rules == {
            NodeNotRespondingProvenance.EPILOG_HEALTH_CHECK.value,
            FailedNodeResolution.SLURM_FAILED_NODE_FIELD.value,
        }
        created = [
            event
            for event in recorder.events
            if event.node_type is NodeType.SCRUB
            and event.rule_name == "CorroboratedNodeFailure"
            and event.action is Action.CREATED
            and event.symptom == corroborated_failure
        ]
        assert len(created) == 1
        assert created[0].hostname == "node01"
        assert created[0].evidence is not None
        assert {source["rule_name"] for source in created[0].evidence["sources"]} == source_rules

    def test_sole_allocated_node_can_corroborate_epilog_non_response(self, stream, obs_ctx):
        job_info = _job_info_with(nodes=["node01"], state="NODE_FAIL")._replace(failed_node=None)
        epilog_source = _make_source(
            obs_ctx,
            datum="Health check failed: Not responding",
            rule=NodeNotRespondingProvenance.EPILOG_HEALTH_CHECK.value,
        )
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(
            ["node01"],
            SourceSet.new_one(RegisteredSymptoms.NODE_NOT_RESPONDING.value, epilog_source),
        )

        _run_scrub(stream, job_info, bad_nodes)

        source_rules = {source.rule for source in bad_nodes["node01"].attributions[self._CORROBORATED_FAILURE]}
        assert source_rules == {
            NodeNotRespondingProvenance.EPILOG_HEALTH_CHECK.value,
            FailedNodeResolution.SINGLE_ALLOCATED_NODE_INFERENCE.value,
        }

    def test_corroborated_node_failure_is_the_pruned_infra_hw_root_cause(self, stream, obs_ctx):
        job_info = _job_info_with(state="NODE_FAIL")._replace(failed_node="node01")
        epilog_source = _make_source(
            obs_ctx,
            datum="Health check failed: Not responding",
            rule=NodeNotRespondingProvenance.EPILOG_HEALTH_CHECK.value,
        )
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(
            ["node01"],
            SourceSet.new_one(RegisteredSymptoms.NODE_NOT_RESPONDING.value, epilog_source),
        )

        symptoms = calculate_symptoms(event_stream=stream, job_info=job_info, my_bad_nodes=bad_nodes)
        outcome = calculate_outcome(symptoms).value()

        corroborated_failure = self._CORROBORATED_FAILURE
        assert corroborated_failure in symptoms
        assert RegisteredSymptoms.NODE_FAILURE_REPORTED_BY_SLURM.value not in symptoms
        assert RegisteredSymptoms.NODE_NOT_RESPONDING.value not in symptoms
        assert outcome.outcome == "Fail InfraHW"
        assert outcome.root_cause_symptom == corroborated_failure

    def test_slurm_derived_non_response_does_not_emit_corroborated_node_failure(self, stream, obs_ctx):
        job_info = _job_info_with(state="NODE_FAIL")._replace(failed_node="node01")
        slurm_reason_source = _make_source(
            obs_ctx,
            rule=NodeNotRespondingProvenance.SLURM_NODE_REASON.value,
        )
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(
            ["node01"],
            SourceSet.new_one(RegisteredSymptoms.NODE_NOT_RESPONDING.value, slurm_reason_source),
        )

        _run_scrub(stream, job_info, bad_nodes)

        assert self._CORROBORATED_FAILURE not in bad_nodes["node01"]

    def test_failure_and_epilog_non_response_on_different_nodes_do_not_combine(self, stream, obs_ctx):
        job_info = _job_info_with(nodes=["node01", "node02"], state="NODE_FAIL")._replace(failed_node="node01")
        epilog_source = _make_source(
            obs_ctx,
            rule=NodeNotRespondingProvenance.EPILOG_HEALTH_CHECK.value,
        )
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(
            ["node02"],
            SourceSet.new_one(RegisteredSymptoms.NODE_NOT_RESPONDING.value, epilog_source),
        )

        _run_scrub(stream, job_info, bad_nodes)

        corroborated_failure = self._CORROBORATED_FAILURE
        assert all(corroborated_failure not in bad_nodes[node] for node in bad_nodes)

    def test_specific_node_failure_evidence_prevents_hardware_composite(self, stream, obs_ctx):
        job_info = _job_info_with(state="NODE_FAIL")._replace(failed_node="node01")
        epilog_source = _make_source(
            obs_ctx,
            rule=NodeNotRespondingProvenance.EPILOG_HEALTH_CHECK.value,
        )
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["XID-94-CONTAINED_ERROR"]})
        bad_nodes.add(
            ["node01"],
            SourceSet.new_one(RegisteredSymptoms.NODE_NOT_RESPONDING.value, epilog_source),
        )

        _run_scrub(stream, job_info, bad_nodes)

        assert self._CORROBORATED_FAILURE not in bad_nodes["node01"]

    def test_node_not_responding_discards_node_rebooted(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["Node not responding", "Node unexpectedly rebooted"]})
        _run_scrub(stream, job_info, bad_nodes)
        assert "Node not responding" in bad_nodes["node01"]
        assert "Node unexpectedly rebooted" not in bad_nodes["node01"]

    def test_powering_down_discards_cascaded(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["System is powering down", "Node not responding", "Node unexpectedly rebooted"]},
        )
        _run_scrub(stream, job_info, bad_nodes)
        assert "System is powering down" in bad_nodes["node01"]
        assert "Node not responding" not in bad_nodes["node01"]
        assert "Node unexpectedly rebooted" not in bad_nodes["node01"]

    def test_xid_discards_node_not_responding_preserves_node_failure(self, stream, obs_ctx, job_info):
        """XID discards 'Node not responding' but preserves 'Node failure reported by slurm' for DAG."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["XID-94-CONTAINED_ERROR", "Node not responding", "Node failure reported by slurm"]},
        )
        _run_scrub(stream, job_info, bad_nodes)
        assert "XID-94-CONTAINED_ERROR" in bad_nodes["node01"]
        assert "Node not responding" not in bad_nodes["node01"]
        assert "Node failure reported by slurm" in bad_nodes["node01"]

    def test_bad_stack_frame_python_preserves_node_failure(self, stream, obs_ctx, job_info):
        """Python stack frame pointer should NOT discard 'Node failure reported by slurm' - DAG handles it."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "bad kernel stack frame pointer in python",
                    "general protection fault",
                    "Node failure reported by slurm",
                ]
            },
        )
        _run_scrub(stream, job_info, bad_nodes)
        assert "bad kernel stack frame pointer in python" in bad_nodes["node01"]
        assert "general protection fault" not in bad_nodes["node01"]
        assert "Node failure reported by slurm" in bad_nodes["node01"]

    def test_bad_stack_frame_generic_preserves_node_failure_for_dag(self, stream, obs_ctx, job_info):
        """Generic stack frame pointer preserves 'Node failure' at scrub level - DAG discards it later."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["bad kernel stack frame pointer", "Node failure reported by slurm"]},
        )
        _run_scrub(stream, job_info, bad_nodes)
        assert "bad kernel stack frame pointer" in bad_nodes["node01"]
        assert "Node failure reported by slurm" in bad_nodes["node01"]

    def test_system_service_oom_kill_discards_not_responding_and_node_failure(self, stream, obs_ctx, job_info):
        """OOM kill of a system service is the root cause for downstream node-silent symptoms."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["OOM kill of system service", "Node not responding", "Node failure reported by slurm"]},
        )
        _run_scrub(stream, job_info, bad_nodes)
        assert "OOM kill of system service" in bad_nodes["node01"]
        assert "Node not responding" not in bad_nodes["node01"]
        assert "Node failure reported by slurm" not in bad_nodes["node01"]

    def test_system_service_oom_kill_preserves_node_rebooted(self, stream, obs_ctx, job_info):
        """A reboot is independent evidence - OOM of a system service does not imply it."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["OOM kill of system service", "Node unexpectedly rebooted"]},
        )
        _run_scrub(stream, job_info, bad_nodes)
        assert "OOM kill of system service" in bad_nodes["node01"]
        assert "Node unexpectedly rebooted" in bad_nodes["node01"]

    def test_same_node_failure_discards_dataloader_exception(self, stream, obs_ctx, job_info):
        """Node failure is the root cause; DataLoader workers crashing is downstream."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["Node failure reported by slurm", "DataLoader Exception"]},
        )
        _run_scrub(stream, job_info, bad_nodes)
        assert "Node failure reported by slurm" in bad_nodes["node01"]
        assert "DataLoader Exception" not in bad_nodes["node01"]

    def test_cross_node_failure_discards_dataloader_exception(self, stream, obs_ctx, job_info):
        """In distributed training, the DataLoader usually crashes on a peer of
        the node that failed, not on the failing node itself. The scrub must
        be job-wide, not per-node."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["Node failure reported by slurm"],
                "node02": ["DataLoader Exception"],
            },
        )
        _run_scrub(stream, job_info, bad_nodes)
        assert "Node failure reported by slurm" in bad_nodes["node01"]
        assert "node02" not in bad_nodes or "DataLoader Exception" not in bad_nodes["node02"]

    def test_dataloader_exception_preserved_without_node_failure(self, stream, obs_ctx, job_info):
        """Without a node-failure root cause, DataLoader Exception stands on its own."""
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["DataLoader Exception"]})
        _run_scrub(stream, job_info, bad_nodes)
        assert "DataLoader Exception" in bad_nodes["node01"]

    def test_job_level_cascade_is_scrubbed_like_a_node(self, stream, obs_ctx, job_info):
        """An unlabelled APPLOG line of a multi-node job lands in the job-level set.

        The same cascade rules must apply there: a DataLoader crash under a node
        failure is discarded whether or not its line named a node.
        """
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["Node failure reported by slurm"]})
        bad_nodes.job_level.add(SourceSet("DataLoader Exception").add(_make_source(obs_ctx)))
        _run_scrub(stream, job_info, bad_nodes)
        assert "Node failure reported by slurm" in bad_nodes["node01"]
        assert "DataLoader Exception" not in bad_nodes.job_level

    def test_job_level_symptom_preserved_without_its_cascade_root(self, stream, obs_ctx, job_info):
        """A job-level symptom with no discarding root on any node survives the scrub."""
        bad_nodes = JobNodeAttributions()
        bad_nodes.job_level.add(SourceSet("DataLoader Exception").add(_make_source(obs_ctx)))
        _run_scrub(stream, job_info, bad_nodes)
        assert "DataLoader Exception" in bad_nodes.job_level

    def test_false_positive_xid64_discards_cascaded(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "FalsePositive-XID-64",
                    "XID-64-INFOROM_DRAM_RETIREMENT_FAILURE",
                    "XID-92-EXCESSIVE_SBE_INTERRUPTS",
                ]
            },
        )
        _run_scrub(stream, job_info, bad_nodes)
        assert "FalsePositive-XID-64" in bad_nodes["node01"]
        assert "XID-64-INFOROM_DRAM_RETIREMENT_FAILURE" not in bad_nodes["node01"]
        assert "XID-92-EXCESSIVE_SBE_INTERRUPTS" not in bad_nodes["node01"]

    def test_eap_recovery_purges_node(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["EAP authentication recovery", "EAP authentication failed"]},
        )
        _run_scrub(stream, job_info, bad_nodes)
        # Both symptoms are removed from "bad"; node may remain with only "interesting" entries
        assert "EAP authentication recovery" not in bad_nodes["node01"]
        assert "EAP authentication failed" not in bad_nodes["node01"]
        assert not bad_nodes["node01"].has_bad()

    def test_cuda_get_device_count_discarded_with_zero_gpus(self, stream, obs_ctx):
        ji = (
            JobInfoBuilder(
                id="123",
                type="batch",
                state="FAILED",
                cluster="test",
                framework="pytorch",
                source_name="test",
                timezone=ZoneInfo("UTC"),
                job_start_time=_DEFAULT_START,
                node_to_ranks={"node01": [0]},
            )
            .num_gpus(0)
            .job_end_time(_DEFAULT_END)
            .build()
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["Unexpected error from cudaGetDeviceCount()"]},
        )
        _run_scrub(stream, ji, bad_nodes)
        assert "Unexpected error from cudaGetDeviceCount()" not in bad_nodes["node01"]

    def test_cuda_803_discards_generic_device_count_on_same_node(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["Unexpected error from cudaGetDeviceCount()", "cudaErrorSystemDriverMismatch"],
            },
        )
        _run_scrub(stream, job_info, bad_nodes)

        assert "Unexpected error from cudaGetDeviceCount()" not in bad_nodes["node01"]
        assert "cudaErrorSystemDriverMismatch" in bad_nodes["node01"]

    def test_scrub_does_not_run_final_refiners(self, stream, obs_ctx, job_info):
        """SCRUB phase must not execute FINAL-phase refiners like CancellationResolution."""
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["terminate called", "Cancelled by user"]})
        _run_scrub(stream, job_info, bad_nodes)
        assert "terminate called" in bad_nodes["node01"]
        assert "Cancelled by user" in bad_nodes["node01"]


class TestIBSymptomCleanup:
    """Tests for the IBSymptomCleanup refiner rule.

    Refiner rules operate on materialized SymptomSet and JobNodeAttributions
    (event stream Attribution.node.* events are consumed before refiner runs).
    """

    @pytest.mark.parametrize("xid_symptom", ["XID-145-RLW_RXPIPE-8", "XID-137-NVLINK_PRIV_ERR"])
    def test_rxpipe_or_priv_err_skips_ib_cleanup(self, stream, obs_ctx, job_info, xid_symptom):
        """Preserve IB symptoms when RXPIPE-8 or XID-137 is present."""
        ib_symptoms = [
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "IB Error-5 (IBV_WC_WR_FLUSH_ERR)",
        ]
        symptoms = _make_symptoms(obs_ctx, xid_symptom, *ib_symptoms)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [xid_symptom, *ib_symptoms]})

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        for ib_symptom in ib_symptoms:
            assert ib_symptom in symptoms
            assert ib_symptom in bad_nodes["node01"].attributions

    def test_non_ib_root_cause_discards_ib_errors(self, stream, obs_ctx, job_info):
        """Per-node: XID root cause discards cascaded IB/EFA errors."""
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-94 (Containment error)",
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "EFA Error-107",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["XID-94 (Containment error)", "IB Error-12 (IBV_WC_RETRY_EXC_ERR)", "EFA Error-107"],
            },
        )

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "XID-94 (Containment error)" in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" not in symptoms
        assert "EFA Error-107" not in symptoms

    def test_non_ib_root_cause_python_exception(self, stream, obs_ctx, job_info):
        """Per-node: Python Exception discards cascaded IB errors."""
        symptoms = _make_symptoms(
            obs_ctx,
            "Python Exception",
            "IB Error-5 (IBV_WC_WR_FLUSH_ERR)",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["Python Exception", "IB Error-5 (IBV_WC_WR_FLUSH_ERR)"],
            },
        )

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "Python Exception" in symptoms
        assert "IB Error-5 (IBV_WC_WR_FLUSH_ERR)" not in symptoms

    @pytest.mark.parametrize("lifecycle", ["Node reboot initiated", "System is powering down"])
    def test_orderly_lifecycle_discards_cross_node_ib_aftermath(self, stream, obs_ctx, job_info, lifecycle):
        """Treat orderly lifecycle evidence as the root cause of IB aftermath.

        Args:
            stream: Event stream fixture used by the refiner.
            obs_ctx: Observation context fixture used to create symptoms.
            job_info: Job information fixture containing the participating nodes.
            lifecycle: Orderly lifecycle symptom under test.
        """
        job_info.nodes.append("node02")
        symptoms = _make_symptoms(
            obs_ctx,
            lifecycle,
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [lifecycle],
                "node02": ["IB Error-12 (IBV_WC_RETRY_EXC_ERR)"],
            },
        )

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert lifecycle in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" not in symptoms
        outcome = calculate_outcome(symptoms).value()
        assert outcome.outcome == "Fail InfraSW"
        assert outcome.root_cause_symptom == lifecycle

    def test_job_level_root_cause_discards_ib_errors(self, stream, obs_ctx, job_info):
        """A root cause from an unlabelled applog line (job-level set) still discards cascaded IB errors.

        Mirrors test_non_ib_root_cause_discards_ib_errors with the root cause in the
        job-level set instead of on a node: the cascade leaves the job-level symptoms
        and the root cause's own group, exactly as it does for a per-node root cause.
        """
        ib12 = "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)"
        symptoms = _make_symptoms(obs_ctx, "cudaErrorECCUncorrectable", ib12, "EFA Error-107")
        bad_nodes = JobNodeAttributions()
        for symptom in ("cudaErrorECCUncorrectable", ib12, "EFA Error-107"):
            bad_nodes.job_level.add(SourceSet(symptom).add(_make_source(obs_ctx)))

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "cudaErrorECCUncorrectable" in symptoms
        assert ib12 not in symptoms
        assert "EFA Error-107" not in symptoms
        assert "cudaErrorECCUncorrectable" in bad_nodes.job_level
        assert ib12 not in bad_nodes.job_level
        assert "EFA Error-107" not in bad_nodes.job_level

    def test_ibv_modify_qp_removes_network_cascade_from_job_level(self, stream, obs_ctx, job_info):
        """The network-cascade discard reaches job-level symptoms too."""
        trigger = "Call to ibv_modify_qp failed"
        symptoms = _make_symptoms(obs_ctx, trigger, "PyTorch NCCL watchdog")
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [trigger]})
        bad_nodes.job_level.add(SourceSet("PyTorch NCCL watchdog").add(_make_source(obs_ctx)))

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "PyTorch NCCL watchdog" not in symptoms
        assert "PyTorch NCCL watchdog" not in bad_nodes.job_level

    def test_non_ib_root_cause_cleans_all_nodes(self, stream, obs_ctx, job_info):
        """Per-node root cause cleanup iterates ALL nodes, not just the first match."""
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-94 (Containment error)",
            "EFA Error-107",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["XID-94 (Containment error)", "EFA Error-107"],
                "node02": ["Python Exception", "EFA Error-107"],
            },
        )

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        # EFA Error-107 should be cleaned from BOTH nodes
        assert "EFA Error-107" not in bad_nodes["node01"].attributions
        assert "EFA Error-107" not in bad_nodes["node02"].attributions

    def test_ib_hierarchy_vendor_129_discards_error_12_and_5(self, stream, obs_ctx, job_info):
        """IB hierarchy: Vendor-129 present discards Error-12 and Error-5."""
        symptoms = _make_symptoms(
            obs_ctx,
            "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)",
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "IB Error-5 (IBV_WC_WR_FLUSH_ERR)",
        )
        bad_nodes = _make_bad_nodes(obs_ctx, {})

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)" in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" not in symptoms
        assert "IB Error-5 (IBV_WC_WR_FLUSH_ERR)" not in symptoms

    def test_ib_hierarchy_error_12_discards_error_5(self, stream, obs_ctx, job_info):
        """IB hierarchy: Error-12 present discards Error-5."""
        symptoms = _make_symptoms(
            obs_ctx,
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "IB Error-5 (IBV_WC_WR_FLUSH_ERR)",
        )
        bad_nodes = _make_bad_nodes(obs_ctx, {})

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" in symptoms
        assert "IB Error-5 (IBV_WC_WR_FLUSH_ERR)" not in symptoms

    @pytest.mark.parametrize("nccl_symptom", ["ncclTimeout"])
    def test_link_down_discards_generic_nccl_symptoms(self, stream, obs_ctx, job_info, nccl_symptom):
        """IB link-down root cause discards generic NCCL symptoms."""
        symptoms = _make_symptoms(
            obs_ctx,
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "link down(east-west)",
            nccl_symptom,
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["IB Error-12 (IBV_WC_RETRY_EXC_ERR)", "link down(east-west)", nccl_symptom],
            },
        )

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "link down(east-west)" in symptoms
        assert nccl_symptom not in symptoms
        assert nccl_symptom not in bad_nodes["node01"].attributions

    @pytest.mark.parametrize(
        "trigger_symptom",
        [
            "Call to ibv_modify_qp failed",
            "Call to ibv_modify_qp failed: eno 11 resource unavailable",
            "Call to ibv_modify_qp failed: eno 22 invalid argument",
            "Call to ibv_modify_qp failed: eno 22 invalid argument(Mismatched GID)",
            "Call to ibv_modify_qp failed: eno 110 timeout",
        ],
    )
    def test_ibv_modify_qp_removes_network_cascade(self, stream, obs_ctx, job_info, trigger_symptom):
        """ibv_modify_qp/set_ece removes cascaded network symptoms."""
        symptoms = _make_symptoms(
            obs_ctx,
            trigger_symptom,
            "Connection reset by peer",
            "PyTorch NCCL watchdog",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [trigger_symptom, "Connection reset by peer"],
            },
        )

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert trigger_symptom in symptoms
        assert "Connection reset by peer" not in symptoms
        assert "PyTorch NCCL watchdog" not in symptoms
        assert "Connection reset by peer" not in bad_nodes["node01"].attributions

    def test_cancelled_by_timeout_discards_ib_when_over_limit(self, stream, obs_ctx):
        """Cancelled by Timeout discards IB/EFA when job exceeded time limit."""
        # Job ran for 2 hours but time_limit is 60 minutes
        ji = default_job_info(state="TIMEOUT", job_end_time=_DEFAULT_START + timedelta(hours=2), time_limit=60)
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
        )
        bad_nodes = _make_bad_nodes(obs_ctx, {})

        _run_ib_cleanup(stream, ji, symptoms, bad_nodes)

        assert "Cancelled by Timeout" in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" not in symptoms

    def test_cancelled_by_timeout_uses_parent_window_when_step_is_under_limit(self, stream, obs_ctx):
        """A short selected-step window does not hide a parent job timeout."""
        parent_end = _DEFAULT_START + timedelta(minutes=240, seconds=12)
        ji = default_job_info(state="TIMEOUT", job_end_time=parent_end, time_limit=240)._replace(
            interest_start_time=_DEFAULT_START + timedelta(seconds=14),
            interest_end_time=parent_end,
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
        )

        _run_ib_cleanup(stream, ji, symptoms, _make_bad_nodes(obs_ctx, {}))

        assert ji.interest_end_time - ji.interest_start_time < timedelta(minutes=240)
        assert "Cancelled by Timeout" in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" not in symptoms

    @pytest.mark.parametrize(
        ("job_id", "parent_elapsed_seconds", "step_elapsed_seconds"),
        [
            ("3548776", 14_412, 14_398),
            ("3497110", 14_405, 14_391),
        ],
    )
    def test_production_timeout_examples_become_timeout_only(
        self,
        stream,
        obs_ctx,
        job_id,
        parent_elapsed_seconds,
        step_elapsed_seconds,
    ):
        """Known false fabric attributions lose their timeout-cascade IB symptoms."""
        parent_end = _DEFAULT_START + timedelta(seconds=parent_elapsed_seconds)
        ji = default_job_info(state="CANCELLED", job_end_time=parent_end, time_limit=240)._replace(
            id=job_id,
            interest_start_time=parent_end - timedelta(seconds=step_elapsed_seconds),
            interest_end_time=parent_end,
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)",
            "IB Error-5 Vendor-244 (IBV_WC_WR_FLUSH_ERR)",
            "IB Error-5 Vendor-249 (IBV_WC_WR_FLUSH_ERR)",
        )

        _run_ib_cleanup(stream, ji, symptoms, _make_bad_nodes(obs_ctx, {}))

        assert ji.exceeded_time_limit() is True
        assert ji.interest_end_time - ji.interest_start_time < timedelta(minutes=240)
        assert set(symptoms.symptoms()) == {"Cancelled by Timeout"}

        bad_nodes_result = CalcBadNodesAttributionResult()
        result = AttributionResult(event_stream=stream, bad_nodes_attr=bad_nodes_result)
        result.symptoms.update(symptoms)
        result.outcome_attr = _make_outcome("Fail InfraHW")
        result.cancelled_by = "Timeout"
        _run_post_outcome(
            stream,
            ji,
            symptoms,
            outcome=result.outcome_attr,
            cancelled_by="Timeout",
            attribution_result=result,
        )

        assert result.outcome_attr.value().outcome == "Pass"

    def test_cancelled_by_timeout_keeps_ib_when_under_limit(self, stream, obs_ctx):
        """Cancelled by Timeout does NOT discard IB when job is under time limit."""
        # Job ran for 30 minutes, time_limit is 60 minutes
        ji = default_job_info(state="TIMEOUT", job_end_time=_DEFAULT_START + timedelta(minutes=30), time_limit=60)
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
        )
        bad_nodes = _make_bad_nodes(obs_ctx, {})

        _run_ib_cleanup(stream, ji, symptoms, bad_nodes)

        assert "Cancelled by Timeout" in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" in symptoms

    def test_eap_auth_discarded_under_2_hours(self, stream, obs_ctx, job_info):
        """EAP authentication failed discarded if job ran < 2 hours."""
        # Default job_info fixture: start + 1 hour end → elapsed < 2h
        symptoms = _make_symptoms(obs_ctx, "EAP authentication failed")
        bad_nodes = _make_bad_nodes(obs_ctx, {})

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "EAP authentication failed" not in symptoms

    def test_eap_auth_kept_over_2_hours(self, stream, obs_ctx):
        """EAP authentication failed kept if job ran >= 2 hours."""
        ji = default_job_info(job_end_time=_DEFAULT_START + timedelta(hours=3))
        symptoms = _make_symptoms(obs_ctx, "EAP authentication failed")
        bad_nodes = _make_bad_nodes(obs_ctx, {})

        _run_ib_cleanup(stream, ji, symptoms, bad_nodes)

        assert "EAP authentication failed" in symptoms

    def test_cuda_error_invalid_value_discards_efa_error_5(self, stream, obs_ctx, job_info):
        """cudaErrorInvalidValue discards EFA Error-5."""
        symptoms = _make_symptoms(
            obs_ctx,
            "cudaErrorInvalidValue",
            "EFA Error-5",
        )
        bad_nodes = _make_bad_nodes(obs_ctx, {})

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "cudaErrorInvalidValue" in symptoms
        assert "EFA Error-5" not in symptoms

    def test_ib_error_with_link_down_keeps_link_down(self, stream, obs_ctx, job_info):
        """CASE 1: IB error + link down = fabric issue. Keep link down, discard IB."""
        symptoms = _make_symptoms(
            obs_ctx,
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "link down(east-west)",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["IB Error-12 (IBV_WC_RETRY_EXC_ERR)", "link down(east-west)"],
            },
        )

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "link down(east-west)" in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" not in symptoms

    def test_no_ib_errors_no_change(self, stream, obs_ctx, job_info):
        """When no IB/EFA symptoms exist, nothing is discarded."""
        symptoms = _make_symptoms(
            obs_ctx,
            "Python Exception",
            "cudaErrorIllegalAddress",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["Python Exception"],
            },
        )

        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)

        assert "Python Exception" in symptoms
        assert "cudaErrorIllegalAddress" in symptoms

    def test_nccl_invalid_nic_discards_ib_errors_and_watchdog(self, stream, obs_ctx, job_info):
        """NCCL invalid NIC selection triggers discard of IB errors and PyTorch NCCL watchdog."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "NCCL invalid NIC selection",
                    "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
                    "PyTorch NCCL watchdog",
                    "Connection reset by peer",
                ]
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "PyTorch NCCL watchdog",
            "Connection reset by peer",
        )
        _run_ib_cleanup(stream, job_info, symptoms, bad_nodes)
        assert "NCCL invalid NIC selection" in bad_nodes["node01"]
        assert not any("IB Error-" in s for s in bad_nodes["node01"].symptoms())
        assert "PyTorch NCCL watchdog" not in bad_nodes["node01"]
        assert "Connection reset by peer" not in bad_nodes["node01"]
        assert not any("IB Error-" in s for s in symptoms.symptoms())
        assert "PyTorch NCCL watchdog" not in symptoms.symptoms()
        assert "Connection reset by peer" not in symptoms.symptoms()

    def test_nccl_invalid_nic_selection_in_calculate_symptoms_pipeline(self, stream, obs_ctx, job_info):
        """NCCL invalid NIC selection is preserved and IB errors discarded through the full calculate_symptoms pipeline."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "NCCL invalid NIC selection",
                    "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
                ]
            },
        )
        result = calculate_symptoms(
            event_stream=stream,
            job_info=job_info,
            my_bad_nodes=bad_nodes,
        )
        assert any("NCCL invalid NIC selection" in s for s in result.symptoms())
        assert not any("IB Error-" in s for s in result.symptoms())

    def test_none_symptoms_is_noop(self, stream, job_info):
        """Refiner rule is a no-op when symptoms or my_bad_nodes is None."""
        # Should not raise
        run_pre_outcome_refiners(
            stream,
            job_info,
            symptoms=None,
            my_bad_nodes=None,
        )


@pytest.mark.parametrize("xid_node", ["node01", "node02"])
@pytest.mark.parametrize(
    "classification",
    [None, "InfraHW", "InfraSW", "MultiCategory", "Undetermined"],
)
def test_kernel_panic_precedes_xid145(stream, obs_ctx, xid_node, classification):
    """Select kernel panic while preserving same-node or peer XID evidence.

    Args:
        stream: Event stream used by refiners.
        obs_ctx: Synthetic evidence context.
        xid_node: Node reporting the XID.
        classification: Optional XID-145 classification marker.
    """
    xid = "XID-145-RLW_RXPIPE-8"
    xid_symptoms = [xid]
    if classification is not None:
        xid_symptoms.append(f"XID-145-{classification}")
    node_symptoms = {"node01": ["kernel panic"]}
    node_symptoms.setdefault(xid_node, []).extend(xid_symptoms)
    symptoms = _make_symptoms(obs_ctx, "kernel panic", *xid_symptoms)
    bad_nodes = _make_bad_nodes(obs_ctx, node_symptoms)

    _run(stream, _job_info_with(nodes=["node01", "node02"]), symptoms, bad_nodes)
    outcome = calculate_outcome(symptoms).value()

    assert outcome.root_cause_symptom == "kernel panic"
    assert outcome.outcome == "Fail InfraSW"
    assert xid in symptoms
    assert xid in bad_nodes[xid_node].attributions


class TestKernelCrashIbCascadeCleanup:
    """Kernel crashes discard the IB cascade reported by peer nodes."""

    @pytest.fixture
    def two_node_job(self):
        return _job_info_with(nodes=["node01", "node02"])

    @pytest.mark.parametrize("crash_symptom", ["kernel panic", "kernel oops", "kernel invalid memory access"])
    def test_peer_kernel_crash_discards_ib_error_12(self, stream, obs_ctx, two_node_job, crash_symptom):
        """Every kernel crash type makes a peer's IB Error-12 a cascade."""
        ib12 = "IB Error-12 (IBV_WC_RETRY_EXC_ERR)"
        symptoms = _make_symptoms(obs_ctx, crash_symptom, ib12)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [crash_symptom], "node02": [ib12]})

        _run(stream, two_node_job, symptoms, bad_nodes)

        assert crash_symptom in symptoms
        assert ib12 not in symptoms
        assert ib12 not in bad_nodes["node02"].attributions

    @pytest.mark.parametrize(
        "ib12",
        [
            "IB Error-12",
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "IB Error-12 Vendor-1",
            "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)",
        ],
    )
    def test_all_peer_ib_error_12_forms_trigger(self, stream, obs_ctx, two_node_job, ib12):
        """Every emitted IB Error-12 spelling triggers the peer cleanup."""
        symptoms = _make_symptoms(obs_ctx, "kernel panic", ib12)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["kernel panic"], "node02": [ib12]})

        _run(stream, two_node_job, symptoms, bad_nodes)

        assert ib12 not in symptoms
        assert ib12 not in bad_nodes["node02"].attributions

    def test_ib_error_120_does_not_trigger_peer_cleanup(self, stream, obs_ctx, two_node_job):
        """Error-120 must not be mistaken for the Error-12 cascade trigger."""
        ib120 = "IB Error-120"
        symptoms = _make_symptoms(obs_ctx, "kernel panic", ib120)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["kernel panic"], "node02": [ib120]})

        _run(stream, two_node_job, symptoms, bad_nodes)

        assert ib120 in symptoms
        assert ib120 in bad_nodes["node02"].attributions

    def test_peer_error_12_discards_full_ib_cascade(self, stream, obs_ctx, two_node_job):
        """Error-12 triggers the shared full IB/EFA/NCCL cascade cleanup."""
        peer_cascade = [
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "IB Error-5 (IBV_WC_WR_FLUSH_ERR)",
            "EFA Error-107",
            "ncclTimeout",
        ]
        symptoms = _make_symptoms(obs_ctx, "kernel panic", *peer_cascade)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["kernel panic"], "node02": peer_cascade})

        _run(stream, two_node_job, symptoms, bad_nodes)

        assert "kernel panic" in symptoms
        assert all(symptom not in symptoms for symptom in peer_cascade)
        assert all(symptom not in bad_nodes["node02"].attributions for symptom in peer_cascade)

    def test_ib_error_12_on_crashed_node_is_kept(self, stream, obs_ctx, job_info):
        """IB evidence on the crashed node itself is not treated as peer cascade."""
        ib12 = "IB Error-12 (IBV_WC_RETRY_EXC_ERR)"
        symptoms = _make_symptoms(obs_ctx, "kernel panic", ib12)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["kernel panic", ib12]})

        _run(stream, job_info, symptoms, bad_nodes)

        assert ib12 in symptoms
        assert ib12 in bad_nodes["node01"].attributions

    def test_crashed_node_evidence_keeps_job_level_symptom(self, stream, obs_ctx, two_node_job):
        """Peer cleanup does not erase the same IB evidence retained by a crashed node."""
        ib12 = "IB Error-12 (IBV_WC_RETRY_EXC_ERR)"
        symptoms = _make_symptoms(obs_ctx, "kernel panic", ib12)
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["kernel panic", ib12],
                "node02": [ib12],
            },
        )

        _run(stream, two_node_job, symptoms, bad_nodes)

        assert ib12 in symptoms
        assert ib12 in bad_nodes["node01"].attributions
        assert ib12 not in bad_nodes["node02"].attributions

    def test_soft_lockup_does_not_trigger_peer_cleanup(self, stream, obs_ctx, two_node_job):
        """A soft lockup is not one of the kernel crash signals."""
        ib12 = "IB Error-12 (IBV_WC_RETRY_EXC_ERR)"
        symptoms = _make_symptoms(obs_ctx, "kernel soft lockup", ib12)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["kernel soft lockup"], "node02": [ib12]})

        _run(stream, two_node_job, symptoms, bad_nodes)

        assert ib12 in symptoms
        assert ib12 in bad_nodes["node02"].attributions

    def test_peer_without_error_12_is_unchanged(self, stream, obs_ctx, two_node_job):
        """A peer carrying only another IB error does not trigger cleanup."""
        ib5 = "IB Error-5 (IBV_WC_WR_FLUSH_ERR)"
        symptoms = _make_symptoms(obs_ctx, "kernel panic", ib5)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["kernel panic"], "node02": [ib5]})

        _run(stream, two_node_job, symptoms, bad_nodes)

        assert ib5 in symptoms
        assert ib5 in bad_nodes["node02"].attributions

    def test_job_level_ib_error_12_is_preserved_under_kernel_crash(self, stream, obs_ctx, two_node_job):
        """Unlabelled IB Error-12 (job-level set) has no known origin, so it is kept like crashed-node evidence.

        The labelled peer report is still discarded. The preserved job-level copy
        keeps the symptom in the job-level set, rather than leaving ``job_level``
        bad while the verdict drops it.
        """
        ib12 = "IB Error-12 (IBV_WC_RETRY_EXC_ERR)"
        symptoms = _make_symptoms(obs_ctx, "kernel panic", ib12)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["kernel panic"], "node02": [ib12]})
        bad_nodes.job_level.add(SourceSet(ib12).add(_make_source(obs_ctx)))

        _run(stream, two_node_job, symptoms, bad_nodes)

        assert "kernel panic" in symptoms
        assert ib12 not in bad_nodes["node02"].attributions
        assert ib12 in bad_nodes.job_level
        assert ib12 in symptoms


class TestSlurmStateSymptoms:
    def test_cancelled_state_adds_symptom(self, stream, obs_ctx):
        ji = _job_info_with(state="CANCELLED", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by user" in symptoms

    def test_requeued_state_adds_symptom(self, stream, obs_ctx):
        ji = _job_info_with(state="REQUEUED")
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Job requeued" in symptoms

    def test_cancelled_state_with_uid_suffix_adds_symptom(self, stream, obs_ctx):
        """SLURM state 'CANCELLED by <uid>' should be handled like 'CANCELLED'."""
        ji = _job_info_with(state="CANCELLED by 43638", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by user" in symptoms

    def test_scheduler_cancelled_state_adds_scheduler_symptom(self, stream, obs_ctx):
        ji = _job_info_with(state="CANCELLED", cancelled_by="site-automation-account", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by cluster operators" in symptoms
        assert "Cancelled by user" not in symptoms

    def test_scheduler_cancelled_with_uid_suffix(self, stream, obs_ctx):
        ji = _job_info_with(state="CANCELLED by 99999", cancelled_by="site-automation-account", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by cluster operators" in symptoms
        assert "Cancelled by user" not in symptoms

    @pytest.mark.parametrize(
        ("cancelled_by", "expected_symptom"),
        [
            ("site-automation-account", "Cancelled by cluster operators"),
            ("alice", "Cancelled by user"),
        ],
    )
    def test_uid0_cancel_respects_resolved_canceller(self, stream, obs_ctx, cancelled_by, expected_symptom):
        """Resolved identities override UID 0 throughout the pre-outcome rules.

        Args:
            stream: Event stream for the job.
            obs_ctx: Observation context for symptom sources.
            cancelled_by: Resolved non-root canceller identity.
            expected_symptom: Cancellation symptom expected for that identity.
        """
        ji = _job_info_with(state="CANCELLED by 0", cancelled_by=cancelled_by, exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert expected_symptom in symptoms
        assert "Cancelled by root" not in symptoms
        if cancelled_by == "site-automation-account":
            assert "Cancelled by user" not in symptoms
        else:
            assert "Cancelled by cluster operators" not in symptoms

    def test_requeued_state_with_no_nodes_remains_incomplete(self, stream, obs_ctx):
        ji = _job_info_with(state="REQUEUED", nodes=[], scheduler=Scheduler.KUEUE)
        symptoms = _make_symptoms(obs_ctx)

        _run(stream, ji, symptoms)

        assert "Job requeued" in symptoms
        assert "Job didn't start" not in symptoms
        assert calculate_outcome(symptoms).value().outcome == "Incomplete Info"

    def test_kueue_eviction_without_evidence_remains_incomplete(self, stream, obs_ctx):
        # The Kueue connector normalizes every eviction reason with no
        # dedicated service state (queue stopped, admission check failed,
        # pods-ready timeout, deactivated, ...) to the single EVICTED value;
        # rules never see the raw per-reason string.
        ji = _job_info_with(state="EVICTED", nodes=[], scheduler=Scheduler.KUEUE)
        symptoms = _make_symptoms(obs_ctx)

        _run(stream, ji, symptoms)

        assert not symptoms
        assert calculate_outcome(symptoms).value().outcome == "Incomplete Info"

    def test_kueue_eviction_preserves_real_failure_evidence(self, stream, obs_ctx):
        ji = _job_info_with(state="EVICTED", nodes=[], scheduler=Scheduler.KUEUE)
        symptoms = _make_symptoms(obs_ctx, "Python Exception")

        _run(stream, ji, symptoms)

        assert "Python Exception" in symptoms
        assert calculate_outcome(symptoms).value().outcome == "Fail User"

    def test_node_fail_without_specific_node_uses_scheduler_evidence(self, stream, obs_ctx):
        ji = _job_info_with(
            state="NODE_FAIL",
            nodes=["node01", "node02"],
            scheduler=Scheduler.KUEUE,
        )._replace(failed_node=None)
        symptoms = _make_symptoms(obs_ctx)

        _run(stream, ji, symptoms)

        assert "Node failure reported by slurm" in symptoms
        assert calculate_outcome(symptoms).value().outcome == "Fail InfraSW"

    def test_non_kueue_node_fail_without_specific_node_retains_legacy_fallback(
        self, stream, obs_ctx
    ):
        ji = _job_info_with(state="NODE_FAIL", nodes=["node01", "node02"])._replace(
            type="runai",
            failed_node=None,
        )
        symptoms = _make_symptoms(obs_ctx)

        _run(stream, ji, symptoms)

        assert "Node failure reported by slurm" not in symptoms
        assert "job failed" in symptoms

    def test_failed_state_no_cancelled_symptom(self, stream, obs_ctx):
        ji = _job_info_with(state="FAILED")
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by user" not in symptoms
        assert "Job requeued" not in symptoms

    def test_uid0_cancel_genuine_root_still_root(self, stream, obs_ctx):
        """A real admin scancel (uid 0, no automation canceller) stays 'Cancelled by root'."""
        ji = _job_info_with(state="CANCELLED by 0", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by root" in symptoms
        assert "Cancelled by user" not in symptoms

    def test_root_canceller_without_uid_decoration_is_root(self, stream, obs_ctx):
        """When the projection normalizes the state to a bare 'CANCELLED', root is
        still detected from the resolved canceller identity."""
        ji = _job_info_with(state="CANCELLED", cancelled_by="root", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by root" in symptoms
        assert "Cancelled by user" not in symptoms

    def test_uid0_cancel_with_timeout_canceller_not_root(self, stream, obs_ctx):
        """slurmctld cancels a job's steps as uid 0 when the job hits its time limit.
        When the resolved canceller is Timeout, the step is automation, not a human
        root scancel, so it must route to the timeout path, not 'Cancelled by root'."""
        ji = _job_info_with(state="CANCELLED by 0", cancelled_by="Timeout", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by root" not in symptoms
        assert "Cancelled by user" not in symptoms
        assert "Cancelled by Timeout" in symptoms

    def test_uid0_timeout_step_reaches_pass_end_to_end(self, stream, obs_ctx):
        """Full pipeline: a uid-0 step cancel whose job hit its time limit must reach
        Pass, matching a state=TIMEOUT job. The old 'CANCELLED by 0' -> 'Cancelled by
        root' mapping made it Fail User and blocked the timeout reclassification."""
        ji = _job_info_with(state="CANCELLED by 0", cancelled_by="Timeout", time_limit=3600, exit_code_signal=None)
        outcome = _attribute_outcome(stream, ji, obs_ctx)
        assert outcome == "Pass"

    def test_uid0_genuine_root_reaches_fail_user_end_to_end(self, stream, obs_ctx):
        """Full pipeline: a genuine admin scancel (uid 0, canceller root) stays Fail User."""
        ji = _job_info_with(state="CANCELLED by 0", cancelled_by="root", time_limit=3600, exit_code_signal=None)
        outcome = _attribute_outcome(stream, ji, obs_ctx)
        assert outcome == "Fail User"


class TestProcessCompletesCleanup:
    def test_discards_cascaded_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "the process completes", "Node not responding", "BrokenPipeError")
        _run(stream, job_info, symptoms)
        assert "the process completes" in symptoms
        assert "Node not responding" not in symptoms
        assert "BrokenPipeError" not in symptoms

    def test_discards_only_tenant_reported_csp_event(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "the process completes",
            "CSP Event Tenant Reported As Faulty",
            "CSP Event Terminated by CSP",
            "Python Exception",
        )

        _run(stream, job_info, symptoms)

        assert "the process completes" in symptoms
        assert "CSP Event Tenant Reported As Faulty" not in symptoms
        assert symptoms["CSP Event Tenant Reported As Faulty"].has_interesting()
        assert "CSP Event Terminated by CSP" in symptoms
        assert "Python Exception" in symptoms

    def test_discards_tenant_reported_from_bad_nodes(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "the process completes", "CSP Event Tenant Reported As Faulty")
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["CSP Event Tenant Reported As Faulty"]})

        _run(stream, job_info, symptoms, bad_nodes)

        assert "CSP Event Tenant Reported As Faulty" not in bad_nodes["node01"]
        assert bad_nodes["node01"].attributions["CSP Event Tenant Reported As Faulty"].has_interesting()

    def test_no_process_completes_keeps_tenant_reported(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "CSP Event Tenant Reported As Faulty")

        _run(stream, job_info, symptoms)

        assert "CSP Event Tenant Reported As Faulty" in symptoms

    def test_no_process_completes_keeps_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Node not responding")
        _run(stream, job_info, symptoms)
        assert "Node not responding" in symptoms


class TestCspEventPrecedenceAndCascades:
    _TENANT_REPORTED = "CSP Event Tenant Reported As Faulty"
    _TERMINATED = "CSP Event Terminated by CSP"

    @pytest.mark.parametrize(
        "ordered_csp_events",
        [
            (_TENANT_REPORTED, _TERMINATED),
            (_TERMINATED, _TENANT_REPORTED),
        ],
    )
    def test_tenant_reported_is_primary_regardless_of_input_order(
        self,
        stream,
        obs_ctx,
        job_info,
        ordered_csp_events,
    ):
        symptoms = _make_symptoms(obs_ctx, *ordered_csp_events)

        _run(stream, job_info, symptoms)

        assert self._TENANT_REPORTED in symptoms
        assert self._TERMINATED not in symptoms
        assert symptoms[self._TERMINATED].has_interesting()
        outcome = calculate_outcome(symptoms).value()
        assert outcome.root_cause_symptom == self._TENANT_REPORTED

    @pytest.mark.parametrize("csp_event", [_TENANT_REPORTED, _TERMINATED])
    def test_single_csp_event_is_unchanged(self, stream, obs_ctx, job_info, csp_event):
        symptoms = _make_symptoms(obs_ctx, csp_event)

        _run(stream, job_info, symptoms)

        assert csp_event in symptoms
        outcome = calculate_outcome(symptoms).value()
        assert outcome.root_cause_symptom == csp_event

    def test_same_node_terminated_event_becomes_supporting_evidence(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, self._TENANT_REPORTED, self._TERMINATED)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [self._TENANT_REPORTED, self._TERMINATED]})

        _run(stream, job_info, symptoms, bad_nodes)

        assert self._TENANT_REPORTED in bad_nodes["node01"]
        assert self._TERMINATED not in bad_nodes["node01"]
        assert bad_nodes["node01"].attributions[self._TERMINATED].has_interesting()

    def test_different_node_terminated_event_remains_bad_node_evidence(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, self._TENANT_REPORTED, self._TERMINATED)
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [self._TENANT_REPORTED],
                "node02": [self._TERMINATED],
            },
        )

        _run(stream, job_info, symptoms, bad_nodes)

        assert self._TENANT_REPORTED in symptoms
        assert self._TERMINATED not in symptoms
        assert self._TENANT_REPORTED in bad_nodes["node01"]
        assert self._TERMINATED in bad_nodes["node02"]

    def test_csp_event_discards_shutdown_and_pyxis_evidence(self, stream, obs_ctx, job_info):
        cascades = ["System is powering down", "error: pyxis"]
        symptoms = _make_symptoms(obs_ctx, self._TERMINATED, *cascades)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [self._TERMINATED, *cascades]})

        _run(stream, job_info, symptoms, bad_nodes)

        assert self._TERMINATED in symptoms
        assert self._TERMINATED in bad_nodes["node01"]
        for cascade in cascades:
            assert cascade not in symptoms
            assert symptoms[cascade].has_interesting()
            assert cascade not in bad_nodes["node01"]
            assert bad_nodes["node01"].attributions[cascade].has_interesting()
        outcome = calculate_outcome(symptoms).value()
        assert outcome.root_cause_symptom == self._TERMINATED

    @pytest.mark.parametrize(
        "pyxis_symptom",
        [
            "error: pyxis",
            "pyxis: couldn't start container",
            "srun: invalid pyxis argument",
        ],
    )
    def test_csp_event_discards_pyxis_family(self, stream, obs_ctx, job_info, pyxis_symptom):
        symptoms = _make_symptoms(obs_ctx, self._TERMINATED, pyxis_symptom)

        _run(stream, job_info, symptoms)

        assert self._TERMINATED in symptoms
        assert pyxis_symptom not in symptoms
        assert symptoms[pyxis_symptom].has_interesting()

    def test_csp_event_keeps_non_pyxis_enroot_symptom(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, self._TERMINATED, "enroot: registry auth denied")

        _run(stream, job_info, symptoms)

        assert self._TERMINATED in symptoms
        assert "enroot: registry auth denied" in symptoms

    def test_shutdown_and_pyxis_are_unchanged_without_csp_event(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "System is powering down", "error: pyxis")

        _run(stream, job_info, symptoms)

        assert "System is powering down" in symptoms
        assert "error: pyxis" in symptoms

    def test_completed_tenant_report_does_not_trigger_csp_cascades(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "the process completes",
            self._TENANT_REPORTED,
            "System is powering down",
            "error: pyxis",
        )

        _run(stream, job_info, symptoms)

        assert self._TENANT_REPORTED not in symptoms
        assert "System is powering down" in symptoms
        assert "error: pyxis" in symptoms

    def test_completed_process_keeps_termination_and_discards_csp_cascades(self, stream, obs_ctx, job_info):
        cascades = ["System is powering down", "error: pyxis"]
        symptoms = _make_symptoms(
            obs_ctx,
            "the process completes",
            self._TENANT_REPORTED,
            self._TERMINATED,
            *cascades,
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._TENANT_REPORTED, self._TERMINATED, *cascades]},
        )

        _run(stream, job_info, symptoms, bad_nodes)

        assert self._TENANT_REPORTED not in symptoms
        assert symptoms[self._TENANT_REPORTED].has_interesting()
        assert self._TERMINATED in symptoms
        assert self._TENANT_REPORTED not in bad_nodes["node01"]
        assert bad_nodes["node01"].attributions[self._TENANT_REPORTED].has_interesting()
        assert self._TERMINATED in bad_nodes["node01"]
        for cascade in cascades:
            assert cascade not in symptoms
            assert symptoms[cascade].has_interesting()
            assert cascade not in bad_nodes["node01"]
            assert bad_nodes["node01"].attributions[cascade].has_interesting()
        outcome = calculate_outcome(symptoms).value()
        assert outcome.root_cause_symptom == self._TERMINATED


class TestCancellationResolution:
    def test_terminate_discards_cancelled(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "terminate called", "Cancelled by user")
        _run(stream, job_info, symptoms)
        assert "Cancelled by user" not in symptoms

    def test_timeout_adds_cancelled_by_timeout(self, stream, obs_ctx):
        ji = _job_info_with(state="TIMEOUT")
        symptoms = _make_symptoms(obs_ctx, "BrokenPipeError")
        _run(stream, ji, symptoms)
        assert "Cancelled by Timeout" in symptoms
        assert "BrokenPipeError" not in symptoms

    def test_non_timeout_keeps_broken_pipe(self, stream, obs_ctx):
        """Retain BrokenPipeError as a failure symptom for non-timeout jobs.

        Args:
            stream: Event stream fixture used by the refiner.
            obs_ctx: Observation context fixture used to create symptoms.
        """
        ji = _job_info_with(state="FAILED")
        symptoms = _make_symptoms(obs_ctx, "BrokenPipeError")
        _run(stream, ji, symptoms)
        assert "BrokenPipeError" in symptoms

    def test_cancelled_by_timeout_discards_cancelled_by_user(self, stream, obs_ctx):
        ji = _job_info_with(state="TIMEOUT")
        symptoms = _make_symptoms(obs_ctx, "Cancelled by user")
        _run(stream, ji, symptoms)
        assert "Cancelled by user" not in symptoms
        assert "Cancelled by Timeout" in symptoms

    def test_scheduler_cancellation_discards_cancelled_by_user(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Cancelled by cluster operators", "Cancelled by user")
        _run(stream, job_info, symptoms)
        assert "Cancelled by user" not in symptoms
        assert "Cancelled by cluster operators" in symptoms

    def test_task_failure_discards_cancelled_by_user(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Cancelled due to task failure", "Cancelled by user")
        _run(stream, job_info, symptoms)
        assert "Cancelled by user" not in symptoms
        assert "Cancelled due to task failure" in symptoms

    def test_task_failure_alone_preserved(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Cancelled due to task failure")
        _run(stream, job_info, symptoms)
        assert "Cancelled due to task failure" in symptoms


class TestXid145RxpipePrivErrCascade:
    _RXPIPE = "XID-145-RLW_RXPIPE-8"
    _XID137 = "XID-137-NVLINK_PRIV_ERR"
    _XID94 = "XID-94-CONTAINED_ERROR"

    def test_same_node_companions_become_interesting_without_timeout(self, stream, obs_ctx, job_info):
        """Regression case is an RXPIPE cascade and does not depend on Slurm timeout."""
        rxpipe_time = _DEFAULT_START + timedelta(minutes=30)
        rxpipe_source = _make_source(obs_ctx, rxpipe_time, datum="RXPIPE", line_num=1)
        xid94_source = _make_source(
            obs_ctx,
            rxpipe_time - timedelta(milliseconds=75),
            datum="contained error",
            line_num=2,
        )
        xid137_source = _make_source(
            obs_ctx,
            rxpipe_time + timedelta(microseconds=30),
            datum="NVLink privilege error",
            line_num=3,
        )
        symptoms = SymptomSet()
        symptoms.add(SourceSet(self._RXPIPE).add(rxpipe_source))
        symptoms.add(SourceSet(self._XID94).add(xid94_source))
        symptoms.add(SourceSet(self._XID137).add(xid137_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes.add(["node01"], SourceSet(self._XID94).add(xid94_source))
        bad_nodes.add(["node01"], SourceSet(self._XID137).add(xid137_source))

        _run(stream, job_info, symptoms, bad_nodes)

        assert self._RXPIPE in symptoms
        assert self._XID94 not in symptoms
        assert self._XID137 not in symptoms
        for companion in (self._XID94, self._XID137):
            assert symptoms._data[companion].has_interesting()
            assert companion not in bad_nodes["node01"].attributions
            assert bad_nodes["node01"].attributions._data[companion].has_interesting()

    def test_cascade_is_node_time_and_source_scoped(self, stream, obs_ctx, job_info):
        """Independent companion evidence must remain eligible for attribution."""
        rxpipe_time = _DEFAULT_START + timedelta(minutes=30)
        rxpipe_source = _make_source(obs_ctx, rxpipe_time, datum="RXPIPE", line_num=1)
        symptoms = SymptomSet().add(SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))

        expected_bad_sources: dict[str, set[Source]] = {}
        correlated_sources: dict[str, Source] = {}
        for offset, companion in enumerate((self._XID94, self._XID137), start=2):
            correlated = _make_source(
                obs_ctx,
                rxpipe_time + timedelta(milliseconds=offset),
                datum=f"correlated {companion}",
                line_num=offset,
            )
            earlier_same_node = _make_source(
                obs_ctx,
                rxpipe_time - timedelta(minutes=10),
                datum=f"earlier {companion}",
                line_num=offset + 10,
            )
            other_node = _make_source(
                obs_ctx,
                rxpipe_time,
                datum=f"other-node {companion}",
                line_num=offset + 20,
            )
            symptoms.add(SourceSet(companion).add(correlated).add(earlier_same_node).add(other_node))
            bad_nodes.add(["node01"], SourceSet(companion).add(correlated).add(earlier_same_node))
            bad_nodes.add(["node02"], SourceSet(companion).add(other_node))
            expected_bad_sources[companion] = {earlier_same_node, other_node}
            correlated_sources[companion] = correlated

        _run(stream, job_info, symptoms, bad_nodes)

        for companion in (self._XID94, self._XID137):
            assert set(symptoms[companion]) == expected_bad_sources[companion]
            assert set(bad_nodes["node01"].attributions[companion]) == {
                source for source in expected_bad_sources[companion] if source.datum.startswith("earlier")
            }
            assert set(bad_nodes["node02"].attributions[companion]) == {
                source for source in expected_bad_sources[companion] if source.datum.startswith("other-node")
            }
            assert correlated_sources[companion] in symptoms[companion]._interesting
            assert correlated_sources[companion] in bad_nodes["node01"].attributions[companion]._interesting

    def test_companions_remain_bad_without_rxpipe(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, self._XID94, self._XID137)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [self._XID94, self._XID137]})

        _run(stream, job_info, symptoms, bad_nodes)

        assert self._XID94 in symptoms
        assert self._XID137 in symptoms
        assert self._XID94 in bad_nodes["node01"].attributions
        assert self._XID137 in bad_nodes["node01"].attributions

    def test_same_node_companion_on_different_gpu_remains_bad(self, stream, obs_ctx, job_info):
        """Concurrent XIDs from a different PCI device are not RXPIPE companions."""
        source_time = _DEFAULT_START + timedelta(minutes=30)
        rxpipe_source = _make_source(
            obs_ctx,
            source_time,
            datum="NVRM: Xid (PCI:0008:06:00): 145, RLW_RXPIPE",
            line_num=1,
        )
        companion_source = _make_source(
            obs_ctx,
            source_time,
            datum="NVRM: Xid (PCI:0009:06:00): 137, RLW_RXPIPE interrupt: PRIV Error",
            line_num=2,
        )
        symptoms = SymptomSet()
        symptoms.add(SourceSet(self._RXPIPE).add(rxpipe_source))
        symptoms.add(SourceSet(self._XID137).add(companion_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes.add(["node01"], SourceSet(self._XID137).add(companion_source))

        _run(stream, job_info, symptoms, bad_nodes)

        assert self._XID137 in symptoms
        assert self._XID137 in bad_nodes["node01"].attributions

    def test_equivalent_pci_formats_identify_same_gpu(self, stream, obs_ctx, job_info):
        """Syslog PCI-domain variants normalize to the same physical device."""
        source_time = _DEFAULT_START + timedelta(minutes=30)
        rxpipe_source = _make_source(
            obs_ctx,
            source_time,
            datum="NVRM: Xid (PCI:0000:c2:00): 145, RLW_RXPIPE",
            line_num=1,
        )
        companion_source = _make_source(
            obs_ctx,
            source_time,
            datum="NVRM: Xid (PCI:00000000:C2:00.0): 137, RLW_RXPIPE interrupt: PRIV Error",
            line_num=2,
        )
        symptoms = SymptomSet()
        symptoms.add(SourceSet(self._RXPIPE).add(rxpipe_source))
        symptoms.add(SourceSet(self._XID137).add(companion_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes.add(["node01"], SourceSet(self._XID137).add(companion_source))

        _run(stream, job_info, symptoms, bad_nodes)

        assert self._XID137 not in symptoms
        assert symptoms._data[self._XID137].has_interesting()
        assert self._XID137 not in bad_nodes["node01"].attributions
        assert bad_nodes["node01"].attributions._data[self._XID137].has_interesting()

    @pytest.mark.parametrize(
        ("rxpipe_datum", "companion_datum"),
        [
            (
                "NVRM: Xid (PCI:0008:06:00): 145, RLW_RXPIPE",
                "XID-137 RLW_RXPIPE interrupt: PRIV Error",
            ),
            (
                "XID-145 RLW_RXPIPE",
                "NVRM: Xid (PCI:0008:06:00): 137, RLW_RXPIPE interrupt: PRIV Error",
            ),
        ],
    )
    def test_one_sided_pci_device_does_not_establish_same_gpu(
        self,
        stream,
        obs_ctx,
        job_info,
        rxpipe_datum,
        companion_datum,
    ):
        """A device-specific record cannot be paired with an unidentified GPU."""
        source_time = _DEFAULT_START + timedelta(minutes=30)
        rxpipe_source = _make_source(obs_ctx, source_time, datum=rxpipe_datum, line_num=1)
        companion_source = _make_source(obs_ctx, source_time, datum=companion_datum, line_num=2)
        symptoms = SymptomSet()
        symptoms.add(SourceSet(self._RXPIPE).add(rxpipe_source))
        symptoms.add(SourceSet(self._XID137).add(companion_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes.add(["node01"], SourceSet(self._XID137).add(companion_source))

        _run(stream, job_info, symptoms, bad_nodes)

        assert self._XID137 in symptoms
        assert self._XID137 in bad_nodes["node01"].attributions

    def test_companions_demoted_before_node_offline_removes_rxpipe(self, stream, obs_ctx, job_info):
        """Node-offline cleanup must not hide the RXPIPE companion relationship."""
        source_time = _DEFAULT_START + timedelta(minutes=30)
        symptoms = _make_symptoms(
            obs_ctx,
            RegisteredSymptoms.NODE_OFFLINE.value,
            self._RXPIPE,
            self._XID94,
            self._XID137,
            source_time=source_time,
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [RegisteredSymptoms.NODE_OFFLINE.value, self._RXPIPE, self._XID94, self._XID137]},
            source_time=source_time,
        )

        _run(stream, job_info, symptoms, bad_nodes)

        for symptom in (self._RXPIPE, self._XID94, self._XID137):
            assert symptom not in symptoms
            assert symptoms._data[symptom].has_interesting()
            assert symptom not in bad_nodes["node01"].attributions
            assert bad_nodes["node01"].attributions._data[symptom].has_interesting()

    @pytest.mark.parametrize("companion", [_XID94, _XID137])
    @pytest.mark.parametrize("offset", [-timedelta(seconds=1), timedelta(seconds=1)])
    def test_companion_at_episode_window_boundary_becomes_interesting(
        self,
        stream,
        obs_ctx,
        job_info,
        companion,
        offset,
    ):
        rxpipe_time = _DEFAULT_START + timedelta(minutes=30)
        symptoms = _make_symptoms(obs_ctx, self._RXPIPE, source_time=rxpipe_time)
        companion_source = _make_source(obs_ctx, rxpipe_time + offset, datum=companion, line_num=2)
        symptoms.add(SourceSet(companion).add(companion_source))
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [self._RXPIPE]}, source_time=rxpipe_time)
        bad_nodes.add(["node01"], SourceSet(companion).add(companion_source))

        _run(stream, job_info, symptoms, bad_nodes)

        assert companion not in symptoms
        assert symptoms._data[companion].has_interesting()
        assert companion not in bad_nodes["node01"].attributions
        assert bad_nodes["node01"].attributions._data[companion].has_interesting()

    @pytest.mark.parametrize("companion", [_XID94, _XID137])
    @pytest.mark.parametrize(
        "offset",
        [
            -timedelta(seconds=1, microseconds=1),
            timedelta(seconds=1, microseconds=1),
        ],
    )
    def test_companion_just_outside_episode_window_remains_bad(
        self,
        stream,
        obs_ctx,
        job_info,
        companion,
        offset,
    ):
        rxpipe_time = _DEFAULT_START + timedelta(minutes=30)
        symptoms = _make_symptoms(obs_ctx, self._RXPIPE, source_time=rxpipe_time)
        companion_source = _make_source(obs_ctx, rxpipe_time + offset, datum=companion, line_num=2)
        symptoms.add(SourceSet(companion).add(companion_source))
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [self._RXPIPE]}, source_time=rxpipe_time)
        bad_nodes.add(["node01"], SourceSet(companion).add(companion_source))

        _run(stream, job_info, symptoms, bad_nodes)

        assert companion in symptoms
        assert companion in bad_nodes["node01"].attributions

    def test_xid_dependent_cleanup_runs_before_companion_demotion(self, stream, obs_ctx, job_info):
        """Demoting the XIDs must not strand their generic cascade symptoms."""
        symptoms = _make_symptoms(
            obs_ctx,
            self._RXPIPE,
            self._XID94,
            self._XID137,
            "Connection closed",
            "error running nvidia-smi",
            "UserBuffer timeout",
            "cudaErrorECCUncorrectable",
            "ncclUnhandledCudaError",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._RXPIPE, self._XID94, self._XID137, "Connection closed"]},
        )

        _run(stream, job_info, symptoms, bad_nodes)

        for generic_cascade in (
            "Connection closed",
            "error running nvidia-smi",
            "UserBuffer timeout",
            "cudaErrorECCUncorrectable",
            "ncclUnhandledCudaError",
        ):
            assert generic_cascade not in symptoms
        assert "Connection closed" not in bad_nodes["node01"].attributions

    def test_only_exact_companion_names_are_demoted(self, stream, obs_ctx, job_info):
        other_xid94 = "XID-94-CONTAINED_ERROR-DETAIL"
        other_xid137 = "XID-137-NVLINK_FLA_PRIV_ERR"
        symptoms = _make_symptoms(obs_ctx, self._RXPIPE, other_xid94, other_xid137)
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": [self._RXPIPE, other_xid94, other_xid137]})

        _run(stream, job_info, symptoms, bad_nodes)

        assert other_xid94 in symptoms
        assert other_xid137 in symptoms


class TestSlurmTimeoutXid145Cascade:
    _RXPIPE = "XID-145-RLW_RXPIPE-8"
    _DYNAMIC = "XID-145-MultiCategory"

    def test_terminal_rxpipe_discarded_for_applog_timeout(self, stream, obs_ctx):
        """CANCELLED jobs can get timeout evidence from the applog rather than Slurm state."""
        job_end = _DEFAULT_START + timedelta(minutes=60, seconds=5)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=60)
        source_time = job_end - timedelta(seconds=1)
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            self._RXPIPE,
            self._DYNAMIC,
            source_time=source_time,
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._RXPIPE, self._DYNAMIC]},
            source_time=source_time,
        )

        _run(stream, ji, symptoms, bad_nodes)

        assert set(symptoms.symptoms()) == {"Cancelled by Timeout"}
        assert self._RXPIPE not in bad_nodes["node01"].attributions
        assert self._DYNAMIC not in bad_nodes["node01"].attributions
        assert bad_nodes["node01"].attributions._data[self._RXPIPE].has_interesting()
        assert bad_nodes["node01"].attributions._data[self._DYNAMIC].has_interesting()

    def test_kueue_timeout_does_not_apply_slurm_teardown_cascade(self, stream, obs_ctx):
        job_end = _DEFAULT_START + timedelta(minutes=60, seconds=5)
        ji = default_job_info(
            state="TIMEOUT",
            job_end_time=job_end,
            time_limit=60,
            scheduler=Scheduler.KUEUE,
        )
        source_time = job_end - timedelta(seconds=1)
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            self._RXPIPE,
            self._DYNAMIC,
            source_time=source_time,
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._RXPIPE, self._DYNAMIC]},
            source_time=source_time,
        )

        _run(stream, ji, symptoms, bad_nodes)

        assert self._RXPIPE in symptoms
        assert self._DYNAMIC in symptoms
        assert self._RXPIPE in bad_nodes["node01"].attributions
        assert self._DYNAMIC in bad_nodes["node01"].attributions

    def test_non_kueue_timeout_retains_existing_teardown_cascade(self, stream, obs_ctx):
        job_end = _DEFAULT_START + timedelta(minutes=60, seconds=5)
        ji = default_job_info(state="TIMEOUT", job_end_time=job_end, time_limit=60)._replace(type="runai")
        source_time = job_end - timedelta(seconds=1)
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            self._RXPIPE,
            self._DYNAMIC,
            source_time=source_time,
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._RXPIPE, self._DYNAMIC]},
            source_time=source_time,
        )

        _run(stream, ji, symptoms, bad_nodes)

        assert set(symptoms.symptoms()) == {"Cancelled by Timeout"}
        assert self._RXPIPE not in bad_nodes["node01"].attributions
        assert self._DYNAMIC not in bad_nodes["node01"].attributions

    def test_terminal_rxpipe_cleans_its_xid145_cascades_before_demotion(self, stream, obs_ctx):
        """Demoting RXPIPE must not strand symptoms that depended on it for cleanup."""
        job_end = _DEFAULT_START + timedelta(minutes=60)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=60)
        source_time = job_end - timedelta(seconds=1)
        cascades = [
            "cudaErrorLaunchFailure",
            "cudaErrorNvlinkUncorrectable",
            "XID-45-PREEMPTIVE_REMOVAL",
            "NV_ERR_INVALID_STATE nv_gpu_ops.c",
            "XID-145-RLW_SRC_TRACK-4",
            "XID-145-RLW_SRC_TRACK-8",
        ]
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            self._RXPIPE,
            self._DYNAMIC,
            *cascades,
            source_time=source_time,
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._RXPIPE, self._DYNAMIC, *cascades]},
            source_time=source_time,
        )

        _run(stream, ji, symptoms, bad_nodes)

        assert set(symptoms.symptoms()) == {"Cancelled by Timeout"}
        for symptom in (self._RXPIPE, self._DYNAMIC, *cascades):
            assert symptoms._data[symptom].has_interesting()
            assert symptom not in bad_nodes["node01"].attributions
            assert bad_nodes["node01"].attributions._data[symptom].has_interesting()

    def test_timeout_state_synthesizes_cancellation_before_discard(self, stream, obs_ctx):
        """TIMEOUT state alone should establish the normalized timeout symptom."""
        job_end = _DEFAULT_START + timedelta(minutes=60, seconds=5)
        ji = _slurm_job_info(state="TIMEOUT", job_end_time=job_end, time_limit=60)
        source_time = job_end - timedelta(seconds=1)
        symptoms = _make_symptoms(obs_ctx, self._RXPIPE, self._DYNAMIC, source_time=source_time)
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._RXPIPE, self._DYNAMIC]},
            source_time=source_time,
        )

        _run(stream, ji, symptoms, bad_nodes)

        assert set(symptoms.symptoms()) == {"Cancelled by Timeout"}

    def test_real_job_timeout_and_rxpipe_offsets_are_correlated(self, stream, obs_ctx):
        """Cover the terminal timing observed for cloud test job."""
        job_end = _DEFAULT_START + timedelta(minutes=60)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=60)
        timeout_source = _make_source(
            obs_ctx,
            job_end - timedelta(seconds=97),
            datum="DUE TO TIME LIMIT",
            line_num=1,
        )
        rxpipe_source = _make_source(
            obs_ctx,
            job_end - timedelta(seconds=34),
            datum="terminal RXPIPE",
            line_num=2,
        )
        symptoms = SymptomSet()
        symptoms.add(SourceSet("Cancelled by Timeout").add(timeout_source))
        symptoms.add(SourceSet(self._RXPIPE).add(rxpipe_source))
        symptoms.add(SourceSet(self._DYNAMIC).add(rxpipe_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes.add(["node01"], SourceSet(self._DYNAMIC).add(rxpipe_source))

        _run(stream, ji, symptoms, bad_nodes)

        assert set(symptoms.symptoms()) == {"Cancelled by Timeout"}
        assert self._RXPIPE not in bad_nodes["node01"].attributions
        assert self._DYNAMIC not in bad_nodes["node01"].attributions

    @pytest.mark.parametrize(
        ("include_timeout", "event_offset"),
        [
            (False, timedelta(seconds=-1)),
            (True, timedelta(seconds=-91)),
            (True, timedelta(seconds=91)),
        ],
    )
    def test_rxpipe_kept_without_all_timeout_cascade_evidence(
        self,
        stream,
        obs_ctx,
        include_timeout,
        event_offset,
    ):
        """The rule requires normalized timeout evidence and a bounded terminal XID."""
        job_end = _DEFAULT_START + timedelta(minutes=60, seconds=5)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=60)
        source_time = job_end + event_offset
        names = (["Cancelled by Timeout"] if include_timeout else []) + [self._RXPIPE, self._DYNAMIC]
        symptoms = _make_symptoms(obs_ctx, *names, source_time=source_time)
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._RXPIPE, self._DYNAMIC]},
            source_time=source_time,
        )

        _run(stream, ji, symptoms, bad_nodes)

        assert self._RXPIPE in symptoms
        assert self._DYNAMIC in symptoms
        assert self._RXPIPE in bad_nodes["node01"].attributions
        assert self._DYNAMIC in bad_nodes["node01"].attributions

    def test_explicit_timeout_does_not_require_time_limit_metadata(self, stream, obs_ctx):
        """The applog's DUE TO TIME LIMIT marker is authoritative timeout evidence."""
        job_end = _DEFAULT_START + timedelta(minutes=30)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=None)
        source_time = job_end - timedelta(seconds=1)
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            self._RXPIPE,
            self._DYNAMIC,
            source_time=source_time,
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [self._RXPIPE, self._DYNAMIC]},
            source_time=source_time,
        )

        _run(stream, ji, symptoms, bad_nodes)

        assert set(symptoms.symptoms()) == {"Cancelled by Timeout"}

    def test_untimed_timeout_source_uses_job_end_correlation(self, stream, obs_ctx):
        """Raw Slurm lines can identify a timeout without yielding a source timestamp."""
        job_end = _DEFAULT_START + timedelta(minutes=60)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=None)
        timeout_source = Source(
            source_context=obs_ctx,
            line_num=1,
            datum="CANCELLED DUE TO TIME LIMIT",
            time=None,
            resources=AllJobResources,
            rule="test",
        )
        rxpipe_source = _make_source(obs_ctx, job_end - timedelta(seconds=34), datum="RXPIPE", line_num=2)
        symptoms = SymptomSet()
        symptoms.add(SourceSet("Cancelled by Timeout").add(timeout_source))
        symptoms.add(SourceSet(self._RXPIPE).add(rxpipe_source))
        symptoms.add(SourceSet(self._DYNAMIC).add(rxpipe_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes.add(["node01"], SourceSet(self._DYNAMIC).add(rxpipe_source))

        _run(stream, ji, symptoms, bad_nodes)

        assert set(symptoms.symptoms()) == {"Cancelled by Timeout"}
        assert self._RXPIPE not in bad_nodes["node01"].attributions

    def test_terminal_rxpipe_must_correlate_with_timed_timeout_source(self, stream, obs_ctx):
        """A timestamped timeout elsewhere in the job must not suppress the XID."""
        job_end = _DEFAULT_START + timedelta(minutes=60)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=60)
        timeout_source = _make_source(
            obs_ctx,
            job_end - timedelta(minutes=5),
            datum="earlier timeout marker",
            line_num=1,
        )
        rxpipe_source = _make_source(
            obs_ctx,
            job_end - timedelta(seconds=1),
            datum="terminal RXPIPE",
            line_num=2,
        )
        symptoms = SymptomSet()
        symptoms.add(SourceSet("Cancelled by Timeout").add(timeout_source))
        symptoms.add(SourceSet(self._RXPIPE).add(rxpipe_source))
        symptoms.add(SourceSet(self._DYNAMIC).add(rxpipe_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes.add(["node01"], SourceSet(self._DYNAMIC).add(rxpipe_source))

        _run(stream, ji, symptoms, bad_nodes)

        assert self._RXPIPE in symptoms
        assert self._DYNAMIC in symptoms
        assert self._RXPIPE in bad_nodes["node01"].attributions
        assert self._DYNAMIC in bad_nodes["node01"].attributions

    @pytest.mark.parametrize(
        ("timeout_from_job_end", "xid_from_timeout", "should_demote"),
        [
            (-timedelta(seconds=30), -timedelta(seconds=1), True),
            (-timedelta(seconds=30), -timedelta(seconds=1, microseconds=1), False),
            (-timedelta(seconds=90), timedelta(seconds=90), True),
            (-timedelta(seconds=90), timedelta(seconds=90, microseconds=1), False),
        ],
    )
    def test_timed_timeout_correlation_boundaries(
        self,
        stream,
        obs_ctx,
        timeout_from_job_end,
        xid_from_timeout,
        should_demote,
    ):
        """Timed markers allow 1 s clock skew before and 90 s teardown after."""
        job_end = _DEFAULT_START + timedelta(minutes=60)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=60)
        timeout_time = job_end + timeout_from_job_end
        xid_time = timeout_time + xid_from_timeout
        timeout_source = _make_source(obs_ctx, timeout_time, datum="DUE TO TIME LIMIT", line_num=1)
        rxpipe_source = _make_source(obs_ctx, xid_time, datum="terminal RXPIPE", line_num=2)
        symptoms = SymptomSet()
        symptoms.add(SourceSet("Cancelled by Timeout").add(timeout_source))
        symptoms.add(SourceSet(self._RXPIPE).add(rxpipe_source))
        symptoms.add(SourceSet(self._DYNAMIC).add(rxpipe_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes.add(["node01"], SourceSet(self._DYNAMIC).add(rxpipe_source))

        _run(stream, ji, symptoms, bad_nodes)

        if should_demote:
            assert self._RXPIPE not in symptoms
            assert self._DYNAMIC not in symptoms
            assert symptoms._data[self._RXPIPE].has_interesting()
            assert symptoms._data[self._DYNAMIC].has_interesting()
            assert self._RXPIPE not in bad_nodes["node01"].attributions
            assert self._DYNAMIC not in bad_nodes["node01"].attributions
            assert bad_nodes["node01"].attributions._data[self._RXPIPE].has_interesting()
            assert bad_nodes["node01"].attributions._data[self._DYNAMIC].has_interesting()
        else:
            assert set(symptoms[self._RXPIPE]) == {rxpipe_source}
            assert set(symptoms[self._DYNAMIC]) == {rxpipe_source}
            assert set(bad_nodes["node01"].attributions[self._RXPIPE]) == {rxpipe_source}
            assert set(bad_nodes["node01"].attributions[self._DYNAMIC]) == {rxpipe_source}

    def test_terminal_source_is_demoted_without_hiding_earlier_failure(self, stream, obs_ctx):
        """A teardown occurrence must not erase a genuine earlier RXPIPE failure."""
        job_end = _DEFAULT_START + timedelta(minutes=60)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=60)
        early_source = _make_source(
            obs_ctx,
            job_end - timedelta(minutes=10),
            datum="early RXPIPE",
            line_num=1,
        )
        terminal_source = _make_source(
            obs_ctx,
            job_end - timedelta(seconds=1),
            datum="terminal RXPIPE",
            line_num=2,
        )
        timeout_source = _make_source(
            obs_ctx,
            job_end,
            datum="DUE TO TIME LIMIT",
            line_num=3,
        )
        symptoms = SymptomSet()
        symptoms.add(SourceSet("Cancelled by Timeout").add(timeout_source))
        symptoms.add(SourceSet(self._RXPIPE).add(early_source).add(terminal_source))
        symptoms.add(SourceSet(self._DYNAMIC).add(early_source).add(terminal_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(early_source))
        bad_nodes.add(["node01"], SourceSet(self._DYNAMIC).add(early_source))
        bad_nodes.add(["node02"], SourceSet(self._RXPIPE).add(terminal_source))
        bad_nodes.add(["node02"], SourceSet(self._DYNAMIC).add(terminal_source))

        _run(stream, ji, symptoms, bad_nodes)

        assert set(symptoms[self._RXPIPE]) == {early_source}
        assert set(symptoms[self._DYNAMIC]) == {early_source}
        assert self._RXPIPE in bad_nodes["node01"].attributions
        assert self._DYNAMIC in bad_nodes["node01"].attributions
        assert self._RXPIPE not in bad_nodes["node02"].attributions
        assert self._DYNAMIC not in bad_nodes["node02"].attributions

    def test_other_multicategory_source_is_preserved(self, stream, obs_ctx):
        """Only the dynamic classification paired to RXPIPE is a timeout cascade."""
        job_end = _DEFAULT_START + timedelta(minutes=60)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=60)
        rxpipe_source = _make_source(
            obs_ctx,
            job_end - timedelta(seconds=1),
            datum="RXPIPE",
            line_num=1,
        )
        other_xid_source = _make_source(
            obs_ctx,
            job_end - timedelta(seconds=1),
            datum="SRC_TRACK",
            line_num=2,
        )
        timeout_source = _make_source(obs_ctx, job_end, datum="DUE TO TIME LIMIT", line_num=3)
        symptoms = SymptomSet()
        symptoms.add(SourceSet("Cancelled by Timeout").add(timeout_source))
        symptoms.add(SourceSet(self._RXPIPE).add(rxpipe_source))
        symptoms.add(SourceSet(self._DYNAMIC).add(rxpipe_source).add(other_xid_source))
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node01"], SourceSet(self._RXPIPE).add(rxpipe_source))
        bad_nodes.add(["node01"], SourceSet(self._DYNAMIC).add(rxpipe_source).add(other_xid_source))

        _run(stream, ji, symptoms, bad_nodes)

        assert self._RXPIPE not in symptoms
        assert set(symptoms[self._DYNAMIC]) == {other_xid_source}
        assert set(bad_nodes["node01"].attributions[self._DYNAMIC]) == {other_xid_source}

    def test_other_xid145_variant_kept_for_terminal_timeout(self, stream, obs_ctx):
        """The exception is exact and must not broaden to other XID-145 variants."""
        job_end = _DEFAULT_START + timedelta(minutes=60, seconds=5)
        ji = _slurm_job_info(state="CANCELLED", job_end_time=job_end, time_limit=60)
        source_time = job_end - timedelta(seconds=1)
        symptoms = _make_symptoms(
            obs_ctx,
            "Cancelled by Timeout",
            "XID-145-RLW_SRC_TRACK-8",
            self._DYNAMIC,
            source_time=source_time,
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["XID-145-RLW_SRC_TRACK-8", self._DYNAMIC]},
            source_time=source_time,
        )

        _run(stream, ji, symptoms, bad_nodes)

        assert "XID-145-RLW_SRC_TRACK-8" in symptoms
        assert self._DYNAMIC in symptoms
        assert "XID-145-RLW_SRC_TRACK-8" in bad_nodes["node01"].attributions
        assert self._DYNAMIC in bad_nodes["node01"].attributions


class TestSimpleSymptomCascades:
    def test_ibv_set_ece_discards_ibv_modify_qp(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Call to ibv_set_ece failed", "Call to ibv_modify_qp failed")
        _run(stream, job_info, symptoms)
        assert "Call to ibv_set_ece failed" in symptoms
        assert "Call to ibv_modify_qp failed" not in symptoms

    def test_pushrecv_renamed_to_userbuffer_timeout(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "pushrecv")
        _run(stream, job_info, symptoms)
        assert "pushrecv" not in symptoms
        assert "UserBuffer timeout" in symptoms

    def test_s3_discards_dataloader(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "S3 Exception", "DataLoader Exception")
        _run(stream, job_info, symptoms)
        assert "S3 Exception" in symptoms
        assert "DataLoader Exception" not in symptoms

    def test_pyxis_specific_discards_generic(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "pyxis: cannot find container file", "error: pyxis")
        _run(stream, job_info, symptoms)
        assert "pyxis: cannot find container file" in symptoms
        assert "error: pyxis" not in symptoms

    def test_pyxis_mount_path_not_found_discards_generic(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: enroot mount path not found",
            "No such file or directory",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: enroot mount path not found" in symptoms
        assert "No such file or directory" not in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_nvidia_hook_discards_generic_and_cli(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: nvidia hook failed",
            "pyxis: nvidia-container-cli mount error",
            "pyxis: nvidia-container-cli init error",
            "error: pyxis",
            "pyxis: couldn't start container",
            "failed to import docker image",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: nvidia hook failed" in symptoms
        assert "pyxis: nvidia-container-cli mount error" not in symptoms
        assert "pyxis: nvidia-container-cli init error" not in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms
        assert "failed to import docker image" not in symptoms

    def test_cgroup_hook_discards_generic(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: cgroup hook failed",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: cgroup hook failed" in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_cli_mount_error_without_hook_survives(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: nvidia-container-cli mount error",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: nvidia-container-cli mount error" in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_device_error_supersedes_hook_failed(self, stream, obs_ctx, job_info):
        # NVIDIA_VISIBLE_DEVICES=42 repro: device error is user-caused root; hook failure is cascade
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: nvidia-container-cli device error",
            "pyxis: nvidia hook failed",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: nvidia-container-cli device error" in symptoms
        assert "pyxis: nvidia hook failed" not in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_unknown_driver_capability_supersedes_hook_failed(self, stream, obs_ctx, job_info):
        # NVIDIA_DRIVER_CAPABILITIES=comput repro: typo is user-caused; hook failure is cascade
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: nvidia unknown driver capability",
            "pyxis: nvidia hook failed",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: nvidia unknown driver capability" in symptoms
        assert "pyxis: nvidia hook failed" not in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_cli_not_installed_supersedes_hook_failed(self, stream, obs_ctx, job_info):
        # nvidia-container-toolkit not installed on the node: the hook fails
        # because the cli binary is missing. The "cli not installed" symptom is
        # the specific root cause; the hook-failed umbrella is the cascade.
        # Both map to Fail InfraSW, but the more specific one is more useful.
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: nvidia-container-cli not installed",
            "pyxis: nvidia hook failed",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: nvidia-container-cli not installed" in symptoms
        assert "pyxis: nvidia hook failed" not in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_requirement_error_supersedes_hook_failed(self, stream, obs_ctx, job_info):
        # NVIDIA_REQUIRE_CUDA=cuda>=99.0 repro: unsatisfiable requirement is user-caused
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: nvidia-container-cli requirement error",
            "pyxis: nvidia hook failed",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: nvidia-container-cli requirement error" in symptoms
        assert "pyxis: nvidia hook failed" not in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_ldcache_and_detection_errors_discarded_by_hook(self, stream, obs_ctx, job_info):
        # ldcache/detection errors are evidence-level signals for the generic
        # nvidia hook failure; when the hook symptom is present, the cli-level
        # ones are noise.
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: nvidia hook failed",
            "pyxis: nvidia-container-cli ldcache error",
            "pyxis: nvidia-container-cli detection error",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: nvidia hook failed" in symptoms
        assert "pyxis: nvidia-container-cli ldcache error" not in symptoms
        assert "pyxis: nvidia-container-cli detection error" not in symptoms

    def test_mellanox_hook_supersedes_mount_path_not_found(self, stream, obs_ctx, job_info):
        # Fixes the attribution bug where the mellanox hook's inner enroot-mount
        # error (tmpfs at /raid/.../infiniband_mad) gets matched as `pyxis: enroot
        # mount path not found` (→ Fail User), but the real root cause is the
        # missing IB kernel module (→ Fail InfraSW via the mellanox hook).
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: mellanox hook failed",
            "pyxis: enroot mount path not found",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: mellanox hook failed" in symptoms
        assert "pyxis: enroot mount path not found" not in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_nvidia_hook_supersedes_mount_failure_noise(self, stream, obs_ctx, job_info):
        # On hook failure, pyxis dumps the entire enroot log including
        # best-effort bind mounts that fail when the host lacks the path
        # (e.g. /var/run/nvhcd → matched as `enroot mount failure from system
        # directory`). The hook is the real root cause; the mount line is
        # replay noise.
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: nvidia hook failed",
            "pyxis: enroot mount failure from system directory",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: nvidia hook failed" in symptoms
        assert "pyxis: enroot mount failure from system directory" not in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_registry_error_supersedes_mount_failure_noise(self, stream, obs_ctx, job_info):
        # Registry pull fails before any container mount runs, so any mount-
        # failure symptom present alongside a registry error came from the
        # enroot-log dump, not the actual killer.
        symptoms = _make_symptoms(
            obs_ctx,
            "enroot: registry image not found",
            "pyxis: enroot mount failure from user directory",
            "failed to import docker image",
        )
        _run(stream, job_info, symptoms)
        assert "enroot: registry image not found" in symptoms
        assert "pyxis: enroot mount failure from user directory" not in symptoms
        assert "failed to import docker image" not in symptoms

    def test_mount_failure_alone_survives(self, stream, obs_ctx, job_info):
        # When mount failure is the only specific signal, it must survive -
        # otherwise we lose attribution for genuine bind-mount-source-missing
        # cases.
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: enroot mount failure from user directory",
            "error: pyxis",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: enroot mount failure from user directory" in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_registry_error_supersedes_import_wrapper(self, stream, obs_ctx, job_info):
        # Any specific registry/pull signal supersedes the generic `failed to
        # import docker image` wrapper that always fires alongside it.
        symptoms = _make_symptoms(
            obs_ctx,
            "enroot: registry image not found",
            "failed to import docker image",
            "pyxis: couldn't start container",
        )
        _run(stream, job_info, symptoms)
        assert "enroot: registry image not found" in symptoms
        assert "failed to import docker image" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms

    def test_missing_credentials_supersedes_generic_auth(self, stream, obs_ctx, job_info):
        # The older-enroot docker-hub fallback variant is a specific version of
        # the generic 401/403 auth-denied signal. Both can get emitted by the
        # same regex; the refiner keeps only the more diagnostic symptom.
        symptoms = _make_symptoms(
            obs_ctx,
            "enroot: missing credentials (docker hub fallback)",
            "enroot: registry auth denied",
        )
        _run(stream, job_info, symptoms)
        assert "enroot: missing credentials (docker hub fallback)" in symptoms
        assert "enroot: registry auth denied" not in symptoms

    def test_peer_task_cascade_discarded_when_other_pyxis_symptom_present(self, stream, obs_ctx, job_info):
        # The peer-task cascade is noise when a concrete root-cause symptom is
        # already present.
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: nvidia-container-cli mount error",
            "pyxis: cascade from peer task",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: nvidia-container-cli mount error" in symptoms
        assert "pyxis: cascade from peer task" not in symptoms

    def test_peer_task_cascade_preserved_when_alone(self, stream, obs_ctx, job_info):
        # When the peer-task cascade is the only pyxis signal, keep it - the
        # primary task's log was lost and MultiCategory is the honest answer.
        symptoms = _make_symptoms(obs_ctx, "pyxis: cascade from peer task")
        _run(stream, job_info, symptoms)
        assert "pyxis: cascade from peer task" in symptoms

    def test_cancellation_sigterm_suppresses_pyxis_wrappers(self, stream, obs_ctx, job_info):
        # pyxis was killed mid-operation by an external SIGTERM. The follow-on
        # `failed to import`/`couldn't start container`/etc. are cascades of the
        # cancellation, not the root cause.
        symptoms = _make_symptoms(
            obs_ctx,
            "pyxis: cancelled by SIGTERM",
            "failed to import docker image",
            "pyxis: couldn't start container",
            "error: pyxis",
            "pyxis: container start nonzero exit",
            "pyxis: failed to create container filesystem",
        )
        _run(stream, job_info, symptoms)
        assert "pyxis: cancelled by SIGTERM" in symptoms
        assert "failed to import docker image" not in symptoms
        assert "pyxis: couldn't start container" not in symptoms
        assert "error: pyxis" not in symptoms
        assert "pyxis: container start nonzero exit" not in symptoms
        assert "pyxis: failed to create container filesystem" not in symptoms

    def test_file_error_discards_watchdog(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "No such file or directory", "PyTorch NCCL watchdog")
        _run(stream, job_info, symptoms)
        assert "No such file or directory" in symptoms
        assert "PyTorch NCCL watchdog" not in symptoms

    def test_numa_discards_no_cuda_runtime(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "NUMA node memory exhaustion", "No CUDA runtime is found")
        _run(stream, job_info, symptoms)
        assert "NUMA node memory exhaustion" in symptoms
        assert "No CUDA runtime is found" not in symptoms


class TestMulticastNvErrSuppression:
    def test_discards_multicast_nv_err_when_applog_marker_present(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c")
        raw_keywords = _make_symptoms(obs_ctx, "Gracefully skipping multicast initialization")
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c"]})

        _run(stream, job_info, symptoms, bad_nodes, raw_keywords=raw_keywords)

        assert "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c" not in symptoms
        if "node01" in bad_nodes:
            assert "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c" not in bad_nodes["node01"]

    def test_keeps_multicast_nv_err_without_applog_marker(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c")
        raw_keywords = _make_symptoms(obs_ctx, "Some other marker")
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c"]})

        _run(stream, job_info, symptoms, bad_nodes, raw_keywords=raw_keywords)

        assert "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c" in symptoms
        assert "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c" in bad_nodes["node01"]


class TestWatchdogTerminateResolution:
    """Tests for resolving watchdog termination symptoms and cascades."""

    def test_watchdog_terminate_aps(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "PyTorch NCCL watchdog", "terminate called")
        with patch.object(
            singlehooks.WatchdogCancellationOverride,
            "current_hook",
            lambda job_info: ("Cancelled by automation", "test rule"),
        ):
            _run(stream, job_info, symptoms)
        assert "Cancelled by automation" in symptoms
        assert "terminate called" not in symptoms

    def test_watchdog_terminate_non_aps(self, stream, obs_ctx):
        """Resolve a watchdog termination without an automation override."""
        ji = _job_info_with(cancelled_by=None)
        symptoms = _make_symptoms(
            obs_ctx,
            "PyTorch NCCL watchdog",
            "terminate called",
            "Cancelled due to task failure",
        )

        _run(stream, ji, symptoms)

        assert "Cancelled by PyTorch NCCL watchdog" in symptoms
        assert "PyTorch NCCL watchdog" not in symptoms
        assert "Cancelled due to task failure" not in symptoms
        assert "terminate called" not in symptoms

    def test_gpu_symptoms_preserved_without_watchdog_cancellation(self, stream, obs_ctx, job_info):
        """Preserve GPU symptoms when no watchdog cancellation is present."""
        symptoms = _make_symptoms(obs_ctx, "XID-145-RLW_RXPIPE-8", "cudaErrorContained")

        _run(stream, job_info, symptoms)

        assert "XID-145-RLW_RXPIPE-8" in symptoms
        assert "cudaErrorContained" in symptoms

    def test_watchdog_cancellation_discards_gpu_cascade_symptoms(self, stream, obs_ctx):
        """Discard GPU cascade symptoms after resolving a watchdog cancellation."""
        job_info = _job_info_with(cancelled_by=None)
        cascades = (
            "XID-145-RLW_RXPIPE-8",
            "cudaErrorContained",
            "XID-137-NVLINK_PRIV_ERR",
            "XID-94-CONTAINED_ERROR",
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "PyTorch NCCL watchdog",
            "terminate called",
            "Cancelled due to task failure",
            *cascades,
        )
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": list(cascades)})

        _run(stream, job_info, symptoms, bad_nodes)

        assert "Cancelled by PyTorch NCCL watchdog" in symptoms
        for symptom in cascades:
            assert symptom not in symptoms
            assert symptom not in bad_nodes["node01"].attributions

    def test_watchdog_discards_pyxis(self, stream, obs_ctx):
        ji = _job_info_with(cancelled_by=None)
        symptoms = _make_symptoms(obs_ctx, "PyTorch NCCL watchdog", "terminate called", "error: pyxis")
        _run(stream, ji, symptoms)
        assert "error: pyxis" not in symptoms


class TestCudaXidCascades:
    def test_xid31_discards_cuda_illegal_address(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-31-MMU_ERR_FLT", "cudaErrorIllegalAddress")
        _run(stream, job_info, symptoms)
        assert "XID-31-MMU_ERR_FLT" in symptoms
        assert "cudaErrorIllegalAddress" not in symptoms

    def test_cuda_invalid_device_discarded_when_not_alone(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "cudaErrorInvalidDevice", "XID-94 (Containment error)")
        _run(stream, job_info, symptoms)
        assert "cudaErrorInvalidDevice" not in symptoms

    def test_cuda_oom_discards_numa_and_devices_unavailable(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx, "cudaError out of memory", "NUMA node memory exhaustion", "cudaErrorDevicesUnavailable"
        )
        _run(stream, job_info, symptoms)
        assert "cudaError out of memory" in symptoms
        assert "NUMA node memory exhaustion" not in symptoms
        assert "cudaErrorDevicesUnavailable" not in symptoms

    def test_dataloader_discards_devices_unavailable(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "DataLoader Exception", "cudaErrorDevicesUnavailable")
        _run(stream, job_info, symptoms)
        assert "DataLoader Exception" in symptoms
        assert "cudaErrorDevicesUnavailable" not in symptoms


# Dedicated coverage for ConnectionClosedCascade trigger handling.
class TestConnectionClosedCascade:
    """Tests for the dedicated ConnectionClosedCascade refiner rule."""

    def test_cuda_invalid_resource_handle_discards_connection_closed_in_bad_nodes(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "cudaErrorInvalidResourceHandle", "Connection closed")
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["cudaErrorInvalidResourceHandle", "Connection closed"]},
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert "Connection closed" not in bad_nodes["node01"].attributions

    def test_cuda_invalid_resource_handle_discards_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "cudaErrorInvalidResourceHandle", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "cudaErrorInvalidResourceHandle" in symptoms
        assert "Connection closed" not in symptoms

    def test_cuda_nvlink_uncorrectable_discards_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "cudaErrorNvlinkUncorrectable", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "cudaErrorNvlinkUncorrectable" in symptoms
        assert "Connection closed" not in symptoms

    def test_xid137_nvlink_priv_err_discards_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-137-NVLINK_PRIV_ERR", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "XID-137-NVLINK_PRIV_ERR" in symptoms
        assert "Connection closed" not in symptoms

    def test_xid145_src_track_8_discards_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-145-RLW_SRC_TRACK-8", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "XID-145-RLW_SRC_TRACK-8" in symptoms
        assert "Connection closed" not in symptoms

    def test_xid145_src_track_4_discards_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-145-RLW_SRC_TRACK-4", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "XID-145-RLW_SRC_TRACK-4" in symptoms
        assert "Connection closed" not in symptoms

    def test_node_failure_reported_by_slurm_discards_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Node failure reported by slurm", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "Node failure reported by slurm" in symptoms
        assert "Connection closed" not in symptoms

    def test_nan_reported_by_megatron_discards_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "NaN reported by Megatron", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "NaN reported by Megatron" in symptoms
        assert "Connection closed" not in symptoms

    def test_cuda_out_of_memory_discards_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "cudaError out of memory", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "cudaError out of memory" in symptoms
        assert "Connection closed" not in symptoms

    def test_nccl_invalid_nic_selection_keeps_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "NCCL invalid NIC selection", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "NCCL invalid NIC selection" in symptoms
        assert "Connection closed" in symptoms

    def test_cuda_error_illegal_state_keeps_connection_closed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "cudaErrorIllegalState", "Connection closed")
        _run(stream, job_info, symptoms)
        assert "cudaErrorIllegalState" in symptoms
        assert "Connection closed" in symptoms


class TestXidGenericCleanup:
    def test_xid_present_discards_generic_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx, "XID-94 (Containment error)", "error running nvidia-smi", "UserBuffer timeout"
        )
        _run(stream, job_info, symptoms)
        assert "XID-94 (Containment error)" in symptoms
        assert "error running nvidia-smi" not in symptoms
        assert "UserBuffer timeout" not in symptoms

    def test_no_xid_keeps_generic(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "error running nvidia-smi")
        _run(stream, job_info, symptoms)
        assert "error running nvidia-smi" in symptoms


class TestPmixCleanup:
    def test_pmix_discarded_when_other_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "pmix collective timeout", "XID-94 (Containment error)")
        _run(stream, job_info, symptoms)
        assert "pmix collective timeout" not in symptoms

    def test_pmix_kept_when_alone(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "pmix collective timeout")
        _run(stream, job_info, symptoms)
        assert "pmix collective timeout" in symptoms


class TestNSConnectionLost:
    def test_multiple_connection_symptoms_become_ns_lost(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["Lustre Client Evicted", "iSCSI detected conn error"],
            },
        )
        symptoms = _make_symptoms(obs_ctx, "Lustre Client Evicted", "iSCSI detected conn error")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "NS network connection lost" in symptoms
        assert "Lustre Client Evicted" not in symptoms
        assert "iSCSI detected conn error" not in symptoms

    def test_single_connection_symptom_no_ns_lost(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["Lustre Client Evicted"]})
        symptoms = _make_symptoms(obs_ctx, "Lustre Client Evicted")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "NS network connection lost" not in symptoms

    def test_prometheus_lost_connection_discarded(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["prometheus lost connection"]})
        symptoms = _make_symptoms(obs_ctx, "prometheus lost connection")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "prometheus lost connection" not in symptoms

    def test_io_error_cascade_discarded_by_dag(self, stream, obs_ctx, job_info):
        """IO error is cascade-discarded via symptom_metadata when NS connection lost is synthesized."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["Lustre Client Evicted", "iSCSI detected conn error", "IO error"]},
        )
        symptoms = _make_symptoms(obs_ctx, "Lustre Client Evicted", "iSCSI detected conn error", "IO error")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "NS network connection lost" in symptoms
        assert "IO error" not in symptoms


class TestNodeOfflineComposite:
    def test_composite_signals_create_node_offline(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "iSCSI detected conn error",
                    "link down(east-west)",
                    "Fabric manager lost connection to all GPUs",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "iSCSI detected conn error",
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value in symptoms
        assert RegisteredSymptoms.NODE_OFFLINE.value in bad_nodes["node01"].attributions

    def test_fm_lost_single_gpu_triggers_node_offline(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "link down(east-west)",
                    "Fabric manager lost connection to GPU",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "link down(east-west)",
            "Fabric manager lost connection to GPU",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value in symptoms
        assert RegisteredSymptoms.NODE_OFFLINE.value in bad_nodes["node01"].attributions
        assert "Fabric manager lost connection to GPU" not in symptoms
        assert "Fabric manager lost connection to GPU" not in bad_nodes["node01"].attributions

    def test_requires_potential_ns_connection_symptom(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "link down(east-west)",
                    "Fabric manager lost connection to all GPUs",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value not in symptoms

    def test_requires_signals_on_same_node(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                ],
                "node02": [
                    "link down(east-west)",
                    "Fabric manager lost connection to all GPUs",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value not in symptoms

    def test_single_ns_symptom_sufficient(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "link down(east-west)",
                    "Fabric manager lost connection to all GPUs",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value in symptoms
        assert RegisteredSymptoms.NODE_OFFLINE.value in bad_nodes["node01"].attributions

    def test_requires_link_down(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "Fabric manager lost connection to all GPUs",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "Fabric manager lost connection to all GPUs",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value not in symptoms

    def test_requires_fm_lost_all_gpus(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "link down(east-west)",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "link down(east-west)",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value not in symptoms

    def test_each_ns_symptom_triggers(self, stream, obs_ctx, job_info):
        for ns_sym in [
            "Lustre Client Evicted",
            "iSCSI detected conn error",
            "prometheus lost connection",
            "Node failure reported by slurm",
        ]:
            bad_nodes = _make_bad_nodes(
                obs_ctx,
                {
                    "node01": [
                        ns_sym,
                        "link down(east-west)",
                        "Fabric manager lost connection to all GPUs",
                    ],
                },
            )
            symptoms = _make_symptoms(
                obs_ctx,
                ns_sym,
                "link down(east-west)",
                "Fabric manager lost connection to all GPUs",
            )
            _run(stream, job_info, symptoms, bad_nodes)
            assert RegisteredSymptoms.NODE_OFFLINE.value in symptoms, f"Expected node offline for {ns_sym}"

    def test_only_matching_node_gets_node_offline(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "link down(east-west)",
                    "Fabric manager lost connection to all GPUs",
                ],
                "node02": [
                    "Lustre Client Evicted",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value in symptoms
        assert RegisteredSymptoms.NODE_OFFLINE.value in bad_nodes["node01"].attributions
        assert RegisteredSymptoms.NODE_OFFLINE.value not in bad_nodes["node02"].attributions

    def test_composite_discards_component_symptoms(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "iSCSI detected conn error",
                    "link down(east-west)",
                    "Fabric manager lost connection to all GPUs",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "iSCSI detected conn error",
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value in symptoms
        node_errors = bad_nodes["node01"].attributions
        for component in [
            "Lustre Client Evicted",
            "iSCSI detected conn error",
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
        ]:
            assert component not in symptoms, f"{component} should be discarded from symptoms"
            assert component not in node_errors, f"{component} should be discarded from node errors"

    def test_composite_discards_only_on_matching_node(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "link down(east-west)",
                    "Fabric manager lost connection to all GPUs",
                ],
                "node02": [
                    "Lustre Client Evicted",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value in bad_nodes["node01"].attributions
        assert "Lustre Client Evicted" not in bad_nodes["node01"].attributions
        assert "Lustre Client Evicted" in bad_nodes["node02"].attributions


class TestNodeOfflineXid145Cascade:
    def test_node_offline_discards_xid145(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "link down(east-west)",
                    "Fabric manager lost connection to all GPUs",
                    "XID-145-InfraHW",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
            "XID-145-InfraHW",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value in symptoms
        assert "XID-145-InfraHW" not in symptoms
        assert "XID-145-InfraHW" not in bad_nodes["node01"].attributions

    def test_node_offline_discards_all_xid145_variants(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "Lustre Client Evicted",
                    "link down(east-west)",
                    "Fabric manager lost connection to all GPUs",
                    "XID-145-InfraHW",
                    "XID-145-InfraSW",
                    "XID-145-Undetermined",
                ],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx,
            "Lustre Client Evicted",
            "link down(east-west)",
            "Fabric manager lost connection to all GPUs",
            "XID-145-InfraHW",
            "XID-145-InfraSW",
            "XID-145-Undetermined",
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert RegisteredSymptoms.NODE_OFFLINE.value in symptoms
        for variant in ["XID-145-InfraHW", "XID-145-InfraSW", "XID-145-Undetermined"]:
            assert variant not in symptoms, f"{variant} should be discarded"
            assert variant not in bad_nodes["node01"].attributions, f"{variant} should be discarded from node"

    def test_xid145_kept_without_node_offline(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["XID-145-InfraHW"]})
        symptoms = _make_symptoms(obs_ctx, "XID-145-InfraHW")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-145-InfraHW" in symptoms


class TestPythonBugXidCleanup:
    def test_python_bug_discards_ib_errors(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Python bug", "IB Error-12 (IBV_WC_RETRY_EXC_ERR)")
        _run(stream, job_info, symptoms)
        assert "Python bug" in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" not in symptoms

    def test_xid74_discards_python_bug(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["XID-74 (NVLink error)", "Python bug"]})
        symptoms = _make_symptoms(obs_ctx, "XID-74 (NVLink error)", "Python bug")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-74 (NVLink error)" in symptoms
        assert "Python bug" not in symptoms
        assert "Python bug" not in bad_nodes["node01"].attributions


class TestNodeFailureNfsCleanup:
    def test_node_failure_discards_imex(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Node failure reported by slurm", "Stopping nvidia-imex.service")
        _run(stream, job_info, symptoms)
        assert "Stopping nvidia-imex.service" not in symptoms

    def test_nfs_with_node_failure_becomes_nfs_server_failure(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["nfs not responding"]})
        symptoms = _make_symptoms(obs_ctx, "Node failure reported by slurm", "nfs not responding")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "nfs server failure" in symptoms
        assert "nfs not responding" not in symptoms
        assert "Node failure reported by slurm" not in symptoms

    def test_nfs_alone_discarded(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "nfs not responding")
        _run(stream, job_info, symptoms)
        assert "nfs not responding" not in symptoms


class TestLastResortSymptoms:
    def test_empty_symptoms_completed_job(self, stream, obs_ctx):
        ji = _job_info_with(state="COMPLETED", scheduler=Scheduler.KUEUE)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "the process completes" in symptoms
        assert symptoms["the process completes"].one_source().rule == "last resort; slurm status"

    def test_empty_symptoms_cancelled_job(self, stream, obs_ctx):
        ji = _job_info_with(state="CANCELLED", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by user" in symptoms

    def test_empty_symptoms_cancelled_by_root(self, stream, obs_ctx):
        """SLURM state 'CANCELLED by 0' should produce 'Cancelled by root', not 'job failed'."""
        ji = _job_info_with(state="CANCELLED by 0")
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "Cancelled by root" in symptoms
        assert "Cancelled by user" not in symptoms
        assert "job failed" not in symptoms

    def test_empty_symptoms_unknown_state(self, stream, obs_ctx):
        ji = _job_info_with(state="OUT_OF_MEMORY", scheduler=Scheduler.KUEUE)
        symptoms = _make_symptoms(obs_ctx)
        _run(stream, ji, symptoms)
        assert "job failed" in symptoms
        assert symptoms["job failed"].one_source().rule == "last resort; slurm status"

    def test_non_empty_symptoms_no_last_resort(self, stream, obs_ctx):
        ji = _job_info_with(state="COMPLETED")
        symptoms = _make_symptoms(obs_ctx, "XID-94 (Containment error)")
        _run(stream, ji, symptoms)
        assert "the process completes" not in symptoms


class TestJobFailedState:
    def test_failed_adds_job_failed(self, stream, obs_ctx):
        ji = _job_info_with(state="FAILED")
        symptoms = _make_symptoms(obs_ctx, "XID-94 (Containment error)")
        _run(stream, ji, symptoms)
        assert "job failed" in symptoms

    def test_failed_with_terminate_no_job_failed(self, stream, obs_ctx):
        ji = _job_info_with(state="FAILED")
        symptoms = _make_symptoms(obs_ctx, "terminate called")
        _run(stream, ji, symptoms)
        assert "job failed" not in symptoms


class TestFabricManagerCascade:
    def test_fabric_manager_removes_cascaded_network(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["NV_ERR_FABRIC_MANAGER_NOT_PRESENT", "Connection reset by peer"],
            },
        )
        symptoms = _make_symptoms(obs_ctx, "NV_ERR_FABRIC_MANAGER_NOT_PRESENT", "Connection reset by peer")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "NV_ERR_FABRIC_MANAGER_NOT_PRESENT" in symptoms


class TestXid171Cascade:
    def test_xid171_discards_cascaded(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["XID-171-UNCORRECTABLE_DRAM_ERROR", "XID-48-ECC_DBE", "XID-94-CONTAINED"],
            },
        )
        symptoms = _make_symptoms(obs_ctx, "XID-171-UNCORRECTABLE_DRAM_ERROR", "XID-48-ECC_DBE", "XID-94-CONTAINED")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-171-UNCORRECTABLE_DRAM_ERROR" in symptoms
        assert "XID-48-ECC_DBE" not in symptoms
        assert "XID-94-CONTAINED" not in symptoms
        assert "XID-48-ECC_DBE" not in bad_nodes["node01"].attributions

    def test_no_xid171_keeps_xid48(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-48-ECC_DBE")
        _run(stream, job_info, symptoms)
        assert "XID-48-ECC_DBE" in symptoms


class TestXid172Cascade:
    def test_xid172_discards_cascaded(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["XID-172-UNCORRECTABLE_SRAM_ERROR", "XID-13-GR_EXCEPTION", "XID-94-CONTAINED"],
            },
        )
        symptoms = _make_symptoms(
            obs_ctx, "XID-172-UNCORRECTABLE_SRAM_ERROR", "XID-13-GR_EXCEPTION", "XID-94-CONTAINED"
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-172-UNCORRECTABLE_SRAM_ERROR" in symptoms
        assert "XID-13-GR_EXCEPTION" not in symptoms
        assert "XID-94-CONTAINED" not in symptoms
        assert "XID-13-GR_EXCEPTION" not in bad_nodes["node01"].attributions


# Raw-keyword promotion rules


def _make_raw_keywords(obs_ctx, *keyword_names) -> SymptomSet:
    """Build a raw_keywords SymptomSet from keyword name strings."""
    return _make_symptoms(obs_ctx, *keyword_names)


class TestMissingKeyInCheckpoint:
    def test_promotes_keyword_to_symptom(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "Missing key in checkpoint")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "Missing key in checkpoint" in symptoms

    def test_no_keyword_no_symptom(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx)
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "Missing key in checkpoint" not in symptoms

    def test_none_raw_keywords_safe(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        _run(stream, job_info, symptoms, raw_keywords=None)
        assert "Missing key in checkpoint" not in symptoms


class TestTrainingValidationCompletion:
    def test_both_present_produces_process_completes(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "after training is done", "validation loss at iteration")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "the process completes" in symptoms
        assert "after training is done" not in symptoms
        assert "validation loss at iteration" not in symptoms

    def test_only_training_done_no_change(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "after training is done")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "the process completes" not in symptoms

    def test_only_validation_no_change(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "validation loss at iteration")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "the process completes" not in symptoms


class TestPythonExceptionFromFile:
    def test_promotes_to_python_exception(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, 'File "')
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "Python Exception" in symptoms

    def test_suppressed_by_watchdog(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "PyTorch NCCL watchdog")
        raw_kw = _make_raw_keywords(obs_ctx, 'File "')
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "Python Exception" not in symptoms

    def test_suppressed_by_s3_exception(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "S3 Exception")
        raw_kw = _make_raw_keywords(obs_ctx, 'File "')
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "Python Exception" not in symptoms

    def test_suppressed_by_disk_quota(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Disk quota exceeded")
        raw_kw = _make_raw_keywords(obs_ctx, 'File "')
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "Python Exception" not in symptoms

    def test_no_file_keyword_no_exception(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx)
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "Python Exception" not in symptoms


class TestSigkillHandlerPromotion:
    def test_promotes_and_discards_ib_errors(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)",
            "IB Error-5 (IBV_WC_WR_FLUSH_ERR)",
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
        )
        raw_kw = _make_raw_keywords(obs_ctx, "sigkill_handler] Killing subprocess")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "si] Killing subprocess" in symptoms
        assert "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)" not in symptoms
        assert "IB Error-5 (IBV_WC_WR_FLUSH_ERR)" not in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" not in symptoms

    def test_no_keyword_preserves_ib(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "IB Error-12 (IBV_WC_RETRY_EXC_ERR)")
        raw_kw = _make_raw_keywords(obs_ctx)
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "si] Killing subprocess" not in symptoms
        assert "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" in symptoms


class TestEmptySymptomsFallback:
    def test_log_unrecognized_batch(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "log file is not recognized from slurm cmdline")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "log unrecognized" in symptoms

    def test_log_unrecognized_interactive(self, stream, obs_ctx):
        ji = (
            JobInfoBuilder(
                id="123",
                type="interactive",
                state="FAILED",
                cluster="test",
                framework="pytorch",
                source_name="test",
                timezone=ZoneInfo("UTC"),
                job_start_time=_DEFAULT_START,
                node_to_ranks={"node01": [0]},
            )
            .job_end_time(_DEFAULT_END)
            .build()
        )
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "log file is not recognized from slurm cmdline")
        _run(stream, ji, symptoms, raw_keywords=raw_kw)
        assert "interactive job" in symptoms

    def test_file_not_found(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "FileNotFoundError")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "log file not found" in symptoms

    def test_permission_error(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "PermissionError")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "log permission error" in symptoms

    def test_log_file_too_large(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "log file too large")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "log file too large" in symptoms

    def test_avg_bus_bandwidth(self, stream, obs_ctx, job_info):
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "Avg bus bandwidth")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "the process completes" in symptoms

    def test_not_triggered_when_real_symptoms_exist(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED")
        raw_kw = _make_raw_keywords(obs_ctx, "FileNotFoundError")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "log file not found" not in symptoms
        assert "XID-94-CONTAINED" in symptoms

    def test_triggered_when_only_cancelled_by_user(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Cancelled by user")
        raw_kw = _make_raw_keywords(obs_ctx, "FileNotFoundError")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "log file not found" in symptoms

    def test_elif_priority_log_unrecognized_wins(self, stream, obs_ctx, job_info):
        """When multiple fallback keywords present, log unrecognized takes priority (elif chain)."""
        symptoms = SymptomSet()
        raw_kw = _make_raw_keywords(obs_ctx, "log file is not recognized from slurm cmdline", "FileNotFoundError")
        _run(stream, job_info, symptoms, raw_keywords=raw_kw)
        assert "log unrecognized" in symptoms
        assert "log file not found" not in symptoms


class TestExtractJobMetadataFallback:
    def test_cancelled_with_logfile_metadata(self, stream, obs_ctx):
        ji = _job_info_with(state="CANCELLED", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx, "Cancelled by user")
        _run(stream, ji, symptoms, extract_job_metadata_from_logfile=True)
        assert "Cancelled by user" not in symptoms
        assert "job failed" in symptoms

    def test_cancelled_with_exit_code_signal(self, stream, obs_ctx):
        ji = _job_info_with(state="CANCELLED", exit_code_signal=9)
        symptoms = _make_symptoms(obs_ctx, "Cancelled by user")
        _run(stream, ji, symptoms, extract_job_metadata_from_logfile=False)
        assert "Cancelled by user" not in symptoms
        assert "exit code signal 9" in symptoms

    def test_cancelled_with_logfile_and_exit_code(self, stream, obs_ctx):
        """exit_code_signal > 0 takes precedence over extract_job_metadata_from_logfile."""
        ji = _job_info_with(state="CANCELLED", exit_code_signal=11)
        symptoms = _make_symptoms(obs_ctx, "Cancelled by user")
        _run(stream, ji, symptoms, extract_job_metadata_from_logfile=True)
        assert "Cancelled by user" not in symptoms
        assert "exit code signal 11" in symptoms

    def test_not_triggered_with_other_symptoms(self, stream, obs_ctx):
        ji = _job_info_with(state="CANCELLED", exit_code_signal=9)
        symptoms = _make_symptoms(obs_ctx, "Cancelled by user", "Python Exception")
        _run(stream, ji, symptoms, extract_job_metadata_from_logfile=True)
        assert "Cancelled by user" in symptoms

    def test_no_exit_code_no_logfile(self, stream, obs_ctx):
        ji = _job_info_with(state="CANCELLED", exit_code_signal=None)
        symptoms = _make_symptoms(obs_ctx, "Cancelled by user")
        _run(stream, ji, symptoms, extract_job_metadata_from_logfile=False)
        assert "Cancelled by user" in symptoms


class TestXid149AndNvlinkFiltering:
    def test_xid149_4_discards_others(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
            "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0",
            "XID-149.0-GENERIC-0",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
                    "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0",
                    "effective_ber_high",
                ],
            },
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0" in symptoms
        assert "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0" not in symptoms
        assert "XID-149.0-GENERIC-0" not in symptoms

    def test_xid149_10_discards_others_when_no_4(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0",
            "XID-149.0-GENERIC-0",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0", "effective_ber_high"]},
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0" in symptoms
        assert "XID-149.0-GENERIC-0" not in symptoms

    def test_xid149_0_discarded_when_other_xid149_present(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-149.0-GENERIC-0",
            "XID-149.7-SOME_OTHER-0",
        )
        _run(stream, job_info, symptoms)
        assert "XID-149.0-GENERIC-0" not in symptoms
        assert "XID-149.7-SOME_OTHER-0" in symptoms

    def test_xid149_0_kept_when_alone(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-149.0-GENERIC-0")
        _run(stream, job_info, symptoms)
        assert "XID-149.0-GENERIC-0" in symptoms

    def test_clean_bad_nodes_on_xid149_4(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
            "effective_ber_high",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
                    "effective_ber_high",
                    "hist15_high",
                ],
            },
        )
        _run(stream, job_info, symptoms, bad_nodes)
        # clean_bad_nodes_of_cascaded_symptoms keeps XID-149.4 and effective_ber_high;
        # then nvlink cleanup loop discards _SYMPTOMS_TO_CHECK from bad_nodes
        assert "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0" in bad_nodes["node01"].attributions
        assert "effective_ber_high" not in bad_nodes["node01"].attributions
        assert "hist15_high" not in bad_nodes["node01"].attributions

    def test_no_xid149_discards_nvlink_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "effective_ber_high", "plr_tx_bw_loss_high")
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["effective_ber_high", "plr_tx_bw_loss_high"]},
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert "effective_ber_high" not in symptoms
        assert "plr_tx_bw_loss_high" not in symptoms
        assert "effective_ber_high" not in bad_nodes["node01"].attributions


class TestGb200PriorityRootCause:
    def test_xid149_keeps_only_root_cause(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
            "Python Exception",
            "PyTorch NCCL watchdog",
        )
        _run(stream, job_info, symptoms)
        assert "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0" in symptoms
        assert "Python Exception" not in symptoms
        assert "PyTorch NCCL watchdog" not in symptoms

    def test_imex_service_keeps_only_root_cause(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "Stopping nvidia-imex.service",
            "Connection closed",
        )
        _run(stream, job_info, symptoms)
        assert "Stopping nvidia-imex.service" in symptoms
        assert "Connection closed" not in symptoms

    def test_fabric_manager_keeps_only_root_cause(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(
            obs_ctx,
            "NV_ERR_FABRIC_MANAGER_NOT_PRESENT",
            "PyTorch NCCL watchdog",
        )
        _run(stream, job_info, symptoms)
        assert "NV_ERR_FABRIC_MANAGER_NOT_PRESENT" in symptoms
        assert "PyTorch NCCL watchdog" not in symptoms

    def test_no_root_cause_keeps_all(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Python Exception", "PyTorch NCCL watchdog")
        _run(stream, job_info, symptoms)
        assert "Python Exception" in symptoms
        assert "PyTorch NCCL watchdog" in symptoms


class TestDagCascadePruning:
    def test_xid171_prunes_xid48_and_xid94(self, stream, obs_ctx, job_info):
        """XID-171 root cause prunes cascaded XID-48 and XID-94."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["XID-171-UNCORRECTABLE_DRAM_ERROR", "XID-48-GPU_ECC_DBE", "XID-94-CONTAINED_ERROR"]},
        )
        symptoms = _make_symptoms(
            obs_ctx, "XID-171-UNCORRECTABLE_DRAM_ERROR", "XID-48-GPU_ECC_DBE", "XID-94-CONTAINED_ERROR"
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-171-UNCORRECTABLE_DRAM_ERROR" in symptoms
        assert "XID-48-GPU_ECC_DBE" not in symptoms
        assert "XID-94-CONTAINED_ERROR" not in symptoms

    def test_xid172_prunes_xid48(self, stream, obs_ctx, job_info):
        """XID-172 root cause prunes cascaded XID-48."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["XID-172-UNCORRECTABLE_SRAM_ERROR", "XID-48-GPU_ECC_DBE"]},
        )
        symptoms = _make_symptoms(obs_ctx, "XID-172-UNCORRECTABLE_SRAM_ERROR", "XID-48-GPU_ECC_DBE")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-172-UNCORRECTABLE_SRAM_ERROR" in symptoms
        assert "XID-48-GPU_ECC_DBE" not in symptoms

    def test_preserves_refiner_added_symptoms(self, stream, obs_ctx, job_info):
        """Symptoms added by earlier refiners (not from my_bad_nodes) survive DAG pruning."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["XID-171-UNCORRECTABLE_DRAM_ERROR", "XID-48-GPU_ECC_DBE"]},
        )
        symptoms = _make_symptoms(
            obs_ctx, "XID-171-UNCORRECTABLE_DRAM_ERROR", "XID-48-GPU_ECC_DBE", "LIKELY-NVSWITCH-REBOOT"
        )
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-171-UNCORRECTABLE_DRAM_ERROR" in symptoms
        assert "XID-48-GPU_ECC_DBE" not in symptoms
        assert "LIKELY-NVSWITCH-REBOOT" in symptoms

    def test_no_pruning_without_cascade_parent(self, stream, obs_ctx, job_info):
        """XID-48 alone is kept when no cascade parent is present."""
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["XID-48-GPU_ECC_DBE"]})
        symptoms = _make_symptoms(obs_ctx, "XID-48-GPU_ECC_DBE")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-48-GPU_ECC_DBE" in symptoms

    def test_system_powering_down_prunes_node_not_responding(self, stream, obs_ctx, job_info):
        """System powering down prunes generic node-loss symptoms via DAG."""
        aftermath = ["Node not responding", "node offline"]
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["System is powering down", *aftermath]},
        )
        symptoms = _make_symptoms(obs_ctx, "System is powering down", *aftermath)
        _run(stream, job_info, symptoms, bad_nodes)
        assert "System is powering down" in symptoms
        for symptom in aftermath:
            assert symptom not in symptoms
        outcome = calculate_outcome(symptoms).value()
        assert outcome.outcome == "Fail InfraSW"
        assert outcome.hw_cause == "Not Applicable"

    @pytest.mark.parametrize(
        ("lifecycle", "hardware"),
        [
            ("System is powering down", "gpu unavailable"),
            ("Node reboot initiated", "XID-48-GPU_ECC_DBE"),
        ],
    )
    def test_hardware_root_outranks_orderly_lifecycle(self, stream, obs_ctx, job_info, lifecycle, hardware):
        """Prefer hardware evidence over orderly lifecycle evidence.

        Args:
            stream: Event stream fixture used by the refiner.
            obs_ctx: Observation context fixture used to create symptoms.
            job_info: Job information fixture used by the refiner.
            lifecycle: Orderly lifecycle symptom under test.
            hardware: Hardware root-cause symptom that must outrank it.
        """
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": [lifecycle, hardware]},
        )
        symptoms = _make_symptoms(obs_ctx, lifecycle, hardware)

        _run(stream, job_info, symptoms, bad_nodes)

        outcome = calculate_outcome(symptoms).value()
        assert outcome.outcome == "Fail InfraHW"
        assert outcome.root_cause_symptom == hardware

    def test_reboot_initiation_prunes_generic_aftermath(self, stream, obs_ctx, job_info):
        """Prune generic reboot aftermath after an initiated reboot.

        Args:
            stream: Event stream fixture used by the refiner.
            obs_ctx: Observation context fixture used to create symptoms.
            job_info: Job information fixture used by the refiner.
        """
        aftermath = [
            "Node unexpectedly rebooted",
            "Node failure reported by slurm",
            "Node not responding",
            "node offline",
            "System is powering down",
            "Fabric manager lost connection to all GPUs",
            "Stopping nvidia-imex.service",
        ]
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["Node reboot initiated", *aftermath]})
        symptoms = _make_symptoms(obs_ctx, "Node reboot initiated", *aftermath)

        _run(stream, job_info, symptoms, bad_nodes)

        assert "Node reboot initiated" in symptoms
        for symptom in aftermath:
            assert symptom not in symptoms
        assert calculate_outcome(symptoms).value().outcome == "Fail InfraSW"

    def test_unexpected_reboot_on_other_node_outranks_initiated_reboot(self, stream, obs_ctx, job_info):
        """Keep an unexpected reboot when it occurred on another node.

        Args:
            stream: Event stream fixture used by the refiner.
            obs_ctx: Observation context fixture used to create symptoms.
            job_info: Job information fixture containing the participating nodes.
        """
        job_info.nodes.append("node02")
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["Node reboot initiated"],
                "node02": ["Node unexpectedly rebooted"],
            },
        )
        symptoms = _make_symptoms(obs_ctx, "Node reboot initiated", "Node unexpectedly rebooted")

        _run(stream, job_info, symptoms, bad_nodes)

        assert "Node reboot initiated" in symptoms
        assert "Node unexpectedly rebooted" in symptoms
        outcome = calculate_outcome(symptoms).value()
        assert outcome.outcome == "Fail InfraHW"
        assert outcome.root_cause_symptom == "Node unexpectedly rebooted"

    def test_multi_node_partial_pruning(self, stream, obs_ctx, job_info):
        """Xid171Cascade refiner removes XID-48 globally; DAG then confirms per-node."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": ["XID-171-UNCORRECTABLE_DRAM_ERROR", "XID-48-GPU_ECC_DBE"],
                "node02": ["XID-48-GPU_ECC_DBE"],
            },
        )
        symptoms = _make_symptoms(obs_ctx, "XID-171-UNCORRECTABLE_DRAM_ERROR", "XID-48-GPU_ECC_DBE")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-171-UNCORRECTABLE_DRAM_ERROR" in symptoms
        # Xid171Cascade refiner runs before DAG and removes XID-48 globally
        assert "XID-48-GPU_ECC_DBE" not in symptoms

    def test_non_fatal_self_discard(self, stream, obs_ctx, job_info):
        """non-fatal symptoms are self-discarded by the DAG."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["XID-94-CONTAINED_ERROR", "non-fatal"]},
        )
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "XID-94-CONTAINED_ERROR" in symptoms
        assert "non-fatal" not in symptoms

    def test_unregistered_symptom_not_discarded(self, stream, obs_ctx, job_info):
        """Unregistered symptoms in my_bad_nodes must not be discarded by the DAG refiner."""
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {"node01": ["CustomAppCrash: unexpected state", "XID-94-CONTAINED_ERROR"]},
        )
        symptoms = _make_symptoms(obs_ctx, "CustomAppCrash: unexpected state", "XID-94-CONTAINED_ERROR")
        _run(stream, job_info, symptoms, bad_nodes)
        assert "unknown-symptom-CustomAppCrash: unexpected state" in symptoms
        assert "XID-94-CONTAINED_ERROR" in symptoms


def _make_outcome(outcome_str="Fail User", hw_cause="Not Applicable", root_cause="test"):
    return SourceSet(OutcomeAttribution(outcome=outcome_str, hw_cause=hw_cause, root_cause_symptom=root_cause))


def _run_post_outcome(
    stream, job_info, symptoms, outcome=None, cancelled_by=None, attribution_result=None, my_bad_nodes=None
):
    """Run only POST_OUTCOME refiners."""
    if outcome is None:
        outcome = _make_outcome()
    if my_bad_nodes is None:
        my_bad_nodes = JobNodeAttributions()
    return run_post_outcome_refiners(
        stream,
        job_info,
        symptoms=symptoms,
        my_bad_nodes=my_bad_nodes,
        outcome=outcome,
        cancelled_by=cancelled_by,
        attribution_result=attribution_result,
    )


class TestDemoteBadNodesOnIncompleteInfo:
    """When the outcome is Incomplete Info, the bad-node symptoms are demoted to
    interesting (the node stays for reporting but no longer counts as bad). This is the
    behavior moved out of core.attribute. The end-to-end Incomplete-Info-with-bad case
    is nearly unreachable through the full pipeline, so this exercises the refiner."""

    def test_incomplete_info_demotes_bad_nodes(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["XID-94-CONTAINED_ERROR"]})
        assert bad_nodes["node01"].has_bad()

        _run_post_outcome(
            stream, job_info, _make_symptoms(obs_ctx), outcome=_make_outcome("Incomplete Info"), my_bad_nodes=bad_nodes
        )

        assert not bad_nodes["node01"].has_bad()
        # Demoted, not removed: the node key remains.
        assert "node01" in bad_nodes.attributions

    def test_real_outcome_preserves_bad_nodes(self, stream, obs_ctx, job_info):
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["XID-94-CONTAINED_ERROR"]})

        _run_post_outcome(
            stream, job_info, _make_symptoms(obs_ctx), outcome=_make_outcome("Fail InfraHW"), my_bad_nodes=bad_nodes
        )

        assert bad_nodes["node01"].has_bad()


class TestPostProcessSymptoms:
    def test_infra_hw_discards_noise(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", "No such file or directory", "Broken pipe")
        outcome = _make_outcome("Fail InfraHW")
        _run_post_outcome(stream, job_info, symptoms, outcome=outcome)
        assert "No such file or directory" not in symptoms
        assert "Broken pipe" not in symptoms
        assert "XID-94-CONTAINED_ERROR" in symptoms

    def test_cancelled_by_user_discarded_when_other_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", "Cancelled by user")
        _run_post_outcome(stream, job_info, symptoms)
        assert "Cancelled by user" not in symptoms
        assert "XID-94-CONTAINED_ERROR" in symptoms

    def test_cancelled_by_timeout_extracts_cancelled_by(self, stream, obs_ctx, job_info):
        result = AttributionResult(event_stream=stream, bad_nodes_attr=CalcBadNodesAttributionResult())
        symptoms = _make_symptoms(obs_ctx, "Cancelled by Timeout")
        _run_post_outcome(stream, job_info, symptoms, attribution_result=result)
        assert result.cancelled_by == "Timeout"
        assert "Cancelled by Timeout" in symptoms

    def test_cancelled_by_nccl_watchdog(self, stream, obs_ctx, job_info):
        result = AttributionResult(event_stream=stream, bad_nodes_attr=CalcBadNodesAttributionResult())
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", "Cancelled by PyTorch NCCL watchdog")
        _run_post_outcome(stream, job_info, symptoms, attribution_result=result)
        assert result.cancelled_by == "PyTorch NCCL watchdog"
        assert "Cancelled by PyTorch NCCL watchdog" not in symptoms

    def test_python_exception_discarded_with_cancellation(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Cancelled by user", "Python Exception")
        _run_post_outcome(stream, job_info, symptoms)
        assert "Python Exception" not in symptoms
        assert "Cancelled by user" in symptoms

    def test_task_failure_replaced_with_job_failed(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "Cancelled due to task failure")
        _run_post_outcome(stream, job_info, symptoms)
        assert "Cancelled due to task failure" not in symptoms
        assert "job failed" in symptoms

    def test_task_failure_discarded_when_other_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", "Cancelled due to task failure")
        _run_post_outcome(stream, job_info, symptoms)
        assert "Cancelled due to task failure" not in symptoms
        assert "XID-94-CONTAINED_ERROR" in symptoms

    @pytest.mark.parametrize("nccl_symptom", ["ncclUnhandledCudaError", "ncclTimeout"])
    def test_nccl_errors_discarded_when_other_symptoms(self, stream, obs_ctx, job_info, nccl_symptom):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", nccl_symptom)
        _run_post_outcome(stream, job_info, symptoms)
        assert nccl_symptom not in symptoms
        assert "XID-94-CONTAINED_ERROR" in symptoms

    def test_segfault_discarded_when_other_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", "Segmentation fault")
        _run_post_outcome(stream, job_info, symptoms)
        assert "Segmentation fault" not in symptoms

    def test_userbuffer_timeout_discarded_with_ib_link_down(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "IB Link Down", "UserBuffer timeout")
        _run_post_outcome(stream, job_info, symptoms)
        assert "UserBuffer timeout" not in symptoms
        assert "IB Link Down" in symptoms

    def test_job_failed_discarded_when_other_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", "job failed")
        _run_post_outcome(stream, job_info, symptoms)
        assert "job failed" not in symptoms

    def test_terminate_called_discarded_when_other_symptoms(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", "terminate called")
        _run_post_outcome(stream, job_info, symptoms)
        assert "terminate called" not in symptoms

    def test_xid145_cascade_filters_job_level_symptoms(self, stream, obs_ctx, job_info):
        """XID-145 cascade symptoms should be removed from job-level symptoms."""
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-145-RLW_SRC_TRACK-8",
            "XID-45-PREEMPTIVE_REMOVAL",
            "cudaErrorLaunchFailure",
        )
        _run_post_outcome(stream, job_info, symptoms)
        assert "XID-145-RLW_SRC_TRACK-8" in symptoms
        assert "XID-45-PREEMPTIVE_REMOVAL" not in symptoms
        assert "cudaErrorLaunchFailure" not in symptoms

    def test_xid145_cascade_filters_my_bad_nodes(self, stream, obs_ctx, job_info):
        """XID-145 cascade symptoms should be removed from per-node attributions."""
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-145-RLW_SRC_TRACK-8",
            "XID-45-PREEMPTIVE_REMOVAL",
            "NV_ERR_INVALID_STATE nv_gpu_ops.c",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "XID-145-RLW_SRC_TRACK-8",
                    "XID-45-PREEMPTIVE_REMOVAL",
                    "NV_ERR_INVALID_STATE nv_gpu_ops.c",
                ],
            },
        )
        _run_post_outcome(stream, job_info, symptoms, my_bad_nodes=bad_nodes)

        assert "XID-145-RLW_SRC_TRACK-8" in symptoms
        assert "XID-45-PREEMPTIVE_REMOVAL" not in symptoms
        assert "NV_ERR_INVALID_STATE nv_gpu_ops.c" not in symptoms

        node_errors = bad_nodes["node01"].attributions
        assert "XID-145-RLW_SRC_TRACK-8" in node_errors
        assert "XID-45-PREEMPTIVE_REMOVAL" not in node_errors
        assert "NV_ERR_INVALID_STATE nv_gpu_ops.c" not in node_errors

    def test_xid145_rxpipe8_discards_src_track_variants(self, stream, obs_ctx, job_info):
        """RXPIPE-8 should discard SRC_TRACK cascade variants."""
        symptoms = _make_symptoms(
            obs_ctx,
            "XID-145-RLW_RXPIPE-8",
            "XID-145-RLW_SRC_TRACK-4",
            "XID-145-RLW_SRC_TRACK-8",
        )
        bad_nodes = _make_bad_nodes(
            obs_ctx,
            {
                "node01": [
                    "XID-145-RLW_RXPIPE-8",
                    "XID-145-RLW_SRC_TRACK-4",
                    "XID-145-RLW_SRC_TRACK-8",
                ],
            },
        )
        _run_post_outcome(stream, job_info, symptoms, my_bad_nodes=bad_nodes)

        assert "XID-145-RLW_RXPIPE-8" in symptoms
        assert "XID-145-RLW_SRC_TRACK-4" not in symptoms
        assert "XID-145-RLW_SRC_TRACK-8" not in symptoms

        node_errors = bad_nodes["node01"].attributions
        assert "XID-145-RLW_RXPIPE-8" in node_errors
        assert "XID-145-RLW_SRC_TRACK-4" not in node_errors
        assert "XID-145-RLW_SRC_TRACK-8" not in node_errors

    def test_xid145_src_track_variants_kept_without_rxpipe8(self, stream, obs_ctx, job_info):
        """Without RXPIPE-8, SRC_TRACK variants should not be filtered."""
        symptoms = _make_symptoms(obs_ctx, "XID-145-RLW_SRC_TRACK-4", "XID-145-RLW_SRC_TRACK-8")
        _run_post_outcome(stream, job_info, symptoms)

        assert "XID-145-RLW_SRC_TRACK-4" in symptoms
        assert "XID-145-RLW_SRC_TRACK-8" in symptoms

    def test_xid145_cascade_no_op_without_xid145(self, stream, obs_ctx, job_info):
        """When XID-145 is not present, cascade symptoms should not be filtered."""
        symptoms = _make_symptoms(obs_ctx, "XID-45-PREEMPTIVE_REMOVAL")
        _run_post_outcome(stream, job_info, symptoms)
        assert "XID-45-PREEMPTIVE_REMOVAL" in symptoms


class TestSuppressNonFatalJobSymptoms:
    """SuppressNonFatalJobSymptoms drops "(non-fatal)" symptoms from the job-level set
    but leaves my_bad_nodes untouched. This logic moved out of core.calculate_symptoms:
    the core aggregation loop used to skip "(non-fatal)" symptoms; now they aggregate
    and this pre-outcome refiner (running first) removes them from the job level."""

    def _run_refiner(self, stream, job_info, symptoms, my_bad_nodes):
        context = RefinerContext(job_info=job_info, event_stream=stream, symptoms=symptoms, my_bad_nodes=my_bad_nodes)
        attribution_lib.get_rules_engine().rules["SuppressNonFatalJobSymptoms"].process(context)

    def test_non_fatal_dropped_real_symptom_kept(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", "(non-fatal) LustreError")
        self._run_refiner(stream, job_info, symptoms, JobNodeAttributions())
        assert "XID-94-CONTAINED_ERROR" in symptoms
        assert "(non-fatal) LustreError" not in symptoms

    def test_matches_non_fatal_substring_anywhere(self, stream, obs_ctx, job_info):
        # The core check was substring "non-fatal", not a leading prefix - preserve that.
        symptoms = _make_symptoms(obs_ctx, "weird non-fatal thing", "XID-94-CONTAINED_ERROR")
        self._run_refiner(stream, job_info, symptoms, JobNodeAttributions())
        assert "weird non-fatal thing" not in symptoms
        assert "XID-94-CONTAINED_ERROR" in symptoms

    def test_my_bad_nodes_left_untouched(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "(non-fatal) LustreError")
        bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["(non-fatal) LustreError", "XID-94-CONTAINED_ERROR"]})
        self._run_refiner(stream, job_info, symptoms, bad_nodes)
        # Only the job-level set is filtered; per-node attributions are preserved.
        assert "(non-fatal) LustreError" not in symptoms
        assert "(non-fatal) LustreError" in bad_nodes["node01"]
        assert "XID-94-CONTAINED_ERROR" in bad_nodes["node01"]


class TestCleanupXid145DynamicSymptoms:
    """Tests for post-outcome cleanup of synthesized XID-145 categories."""

    _DYNAMIC = ["XID-145-InfraHW", "XID-145-InfraSW", "XID-145-MultiCategory", "XID-145-Undetermined"]

    @pytest.mark.parametrize("dynamic", _DYNAMIC)
    def test_dynamic_variant_discarded(self, stream, obs_ctx, job_info, dynamic):
        symptoms = _make_symptoms(obs_ctx, dynamic, "XID-94-CONTAINED_ERROR")
        _run_post_outcome(stream, job_info, symptoms)
        assert dynamic not in symptoms
        assert "XID-94-CONTAINED_ERROR" in symptoms

    def test_all_dynamic_variants_discarded_together(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, *self._DYNAMIC, "XID-94-CONTAINED_ERROR")
        _run_post_outcome(stream, job_info, symptoms)
        assert all(dynamic not in symptoms for dynamic in self._DYNAMIC)
        assert "XID-94-CONTAINED_ERROR" in symptoms

    def test_membership_is_exact_not_prefix(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-145-InfraHW", "XID-145-RLW_SRC_TRACK-8")
        _run_post_outcome(stream, job_info, symptoms)
        assert "XID-145-InfraHW" not in symptoms
        assert "XID-145-RLW_SRC_TRACK-8" in symptoms

    def test_no_op_without_dynamic_variants(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR")
        _run_post_outcome(stream, job_info, symptoms)
        assert "XID-94-CONTAINED_ERROR" in symptoms


class TestFilterFatalXidNoise:
    def test_xid79_keeps_only_itself(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-79-FATAL", "XID-94-CONTAINED_ERROR", "Python Exception")
        _run_post_outcome(stream, job_info, symptoms)
        assert "XID-79-FATAL" in symptoms
        assert "XID-94-CONTAINED_ERROR" not in symptoms
        assert "Python Exception" not in symptoms

    def test_no_xid79_keeps_all(self, stream, obs_ctx, job_info):
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR", "Python Exception")
        _run_post_outcome(stream, job_info, symptoms)
        assert "XID-94-CONTAINED_ERROR" in symptoms


class TestOutcomeReclassification:
    def test_preempted_reclassified_to_pass(self, stream, obs_ctx):
        ji = _job_info_with(state="PREEMPTED", scheduler=Scheduler.KUEUE)
        bad_nodes_result = CalcBadNodesAttributionResult()
        result = AttributionResult(event_stream=stream, bad_nodes_attr=bad_nodes_result)
        result.outcome_attr = _make_outcome("Fail User")

        # Non-empty and all "job failed": exercises the actual all(symptom ==
        # "job failed" ...) check. An empty symptom set would make this test
        # pass vacuously regardless of whether that check is implemented
        # correctly.
        symptoms = _make_symptoms(obs_ctx, "job failed")
        _run_post_outcome(stream, ji, symptoms, outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Pass"
        assert result.outcome_attr.value().root_cause_symptom == "SLURM preempted"

    def test_preempted_does_not_override_failure_evidence(self, stream, obs_ctx):
        ji = _job_info_with(state="PREEMPTED", scheduler=Scheduler.KUEUE)
        result = AttributionResult(event_stream=stream, bad_nodes_attr=CalcBadNodesAttributionResult())
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR")
        result.symptoms.update(symptoms)
        result.outcome_attr = _make_outcome("Fail InfraHW")

        _run_post_outcome(stream, ji, symptoms, outcome=result.outcome_attr, attribution_result=result)

        assert result.outcome_attr.value().outcome == "Fail InfraHW"

    def test_non_kueue_preempted_retains_existing_reclassification(self, stream, obs_ctx):
        ji = _job_info_with(state="PREEMPTED")._replace(type="runai")
        result = AttributionResult(event_stream=stream, bad_nodes_attr=CalcBadNodesAttributionResult())
        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR")
        result.symptoms.update(symptoms)
        result.outcome_attr = _make_outcome("Fail InfraHW")

        _run_post_outcome(
            stream,
            ji,
            symptoms,
            outcome=result.outcome_attr,
            attribution_result=result,
        )

        assert result.outcome_attr.value().outcome == "Pass"

    def test_timeout_only_reclassified_to_pass(self, stream, obs_ctx):
        ji = _job_info_with(
            state="TIMEOUT",
            cancelled_by="Timeout",
            time_limit=3600,
            scheduler=Scheduler.KUEUE,
        )
        bad_nodes_result = CalcBadNodesAttributionResult()
        result = AttributionResult(event_stream=stream, bad_nodes_attr=bad_nodes_result)
        symptoms = _make_symptoms(obs_ctx, "Cancelled by Timeout")
        result.symptoms.update(symptoms)
        result.outcome_attr = _make_outcome("Fail User")
        result.cancelled_by = "Timeout"

        _run_post_outcome(stream, ji, symptoms, outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Pass"
        assert "SLURM timeout" in result.outcome_attr.value().root_cause_symptom

    # ---- NVRx ft_launcher exit-code reclassification (Regression case) ----

    def _nvrx_result(self, stream, outcome_str, symptoms=None):
        result = AttributionResult(event_stream=stream, bad_nodes_attr=CalcBadNodesAttributionResult())
        result.outcome_attr = _make_outcome(outcome_str)
        if symptoms is not None:
            result.symptoms.update(symptoms)
        return result

    def test_nvrx_success_resolves_ambiguous_to_pass(self, stream, obs_ctx):
        # NVRx job, job-level exit 0:0, ambiguous outcome, no clear failure -> Pass.
        ji = _job_info_with(
            state="COMPLETED", tags=frozenset({"nvrx"}), exit_code=0, exit_code_signal=0, time_limit=None
        )
        result = self._nvrx_result(stream, "Fail Undetermined")
        _run_post_outcome(stream, ji, _make_symptoms(obs_ctx), outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Pass"

    def test_nvrx_success_with_preemption_becomes_pass(self, stream, obs_ctx):
        # PREEMPTED + NVRx 0:0 -> Pass (existing PREEMPTED branch, not blocked).
        ji = _job_info_with(state="PREEMPTED", tags=frozenset({"nvrx"}), exit_code=0, exit_code_signal=0)
        result = self._nvrx_result(stream, "Fail User")
        _run_post_outcome(stream, ji, _make_symptoms(obs_ctx), outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Pass"

    def test_nvrx_success_does_not_override_clear_infra_hw(self, stream, obs_ctx):
        # A clean 0:0 must NOT override a clear hardware failure.
        ji = _job_info_with(
            state="COMPLETED", tags=frozenset({"nvrx"}), exit_code=0, exit_code_signal=0, time_limit=None
        )
        result = self._nvrx_result(stream, "Fail InfraHW")
        _run_post_outcome(
            stream,
            ji,
            _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR"),
            outcome=result.outcome_attr,
            attribution_result=result,
        )
        assert result.outcome_attr.value().outcome == "Fail InfraHW"

    def test_nvrx_detected_failure_with_inherited_success_not_pass(self, stream, obs_ctx):
        # A failed cycle inherits the recovered job's 0:0; NVRx detected failure must stay a failure.
        ji = _job_info_with(state="FAILED", tags=frozenset({"nvrx"}), exit_code=0, exit_code_signal=0, time_limit=None)
        symptoms = _make_symptoms(obs_ctx, "NVRx detected failure")
        result = self._nvrx_result(stream, calculate_outcome(symptoms).value().outcome, symptoms)
        _run_post_outcome(stream, ji, symptoms, outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Fail MultiCategory"

    def test_nvrx_recovered_cycle_still_passes(self, stream, obs_ctx):
        # Job-level 0:0 -> Pass even if an intermediate cycle failed (DerivedExitCode is not used).
        ji = _job_info_with(
            state="COMPLETED", tags=frozenset({"nvrx"}), exit_code=0, exit_code_signal=0, time_limit=None
        )
        result = self._nvrx_result(stream, "Incomplete Info")
        _run_post_outcome(stream, ji, _make_symptoms(obs_ctx), outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Pass"

    def test_nvrx_nonzero_signal_with_preemption_not_pass(self, stream, obs_ctx):
        # exit 0:9 (killed by signal) is a failure; PREEMPTED must NOT flip it to Pass.
        ji = _job_info_with(state="PREEMPTED", tags=frozenset({"nvrx"}), exit_code=0, exit_code_signal=9)
        result = self._nvrx_result(stream, "Fail User")
        _run_post_outcome(stream, ji, _make_symptoms(obs_ctx), outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome != "Pass"

    def test_nvrx_nonzero_code_with_timeout_not_pass(self, stream, obs_ctx):
        # nonzero code + timeout-only must NOT flip to Pass.
        ji = _job_info_with(
            state="TIMEOUT",
            cancelled_by="Timeout",
            time_limit=3600,
            tags=frozenset({"nvrx"}),
            exit_code=1,
            exit_code_signal=0,
        )
        symptoms = _make_symptoms(obs_ctx, "Cancelled by Timeout")
        result = self._nvrx_result(stream, "Fail User", symptoms=symptoms)
        result.cancelled_by = "Timeout"
        _run_post_outcome(stream, ji, symptoms, outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome != "Pass"

    def test_nvrx_missing_exit_code_unchanged(self, stream, obs_ctx):
        # NVRx job with no exit code -> outcome unchanged.
        ji = _job_info_with(
            state="COMPLETED", tags=frozenset({"nvrx"}), exit_code=None, exit_code_signal=None, time_limit=None
        )
        result = self._nvrx_result(stream, "Fail Undetermined")
        _run_post_outcome(stream, ji, _make_symptoms(obs_ctx), outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Fail Undetermined"

    def test_non_nvrx_exit_zero_unchanged(self, stream, obs_ctx):
        # exit 0:0 but not NVRx -> no reclassification.
        ji = _job_info_with(state="COMPLETED", tags=frozenset(), exit_code=0, exit_code_signal=0, time_limit=None)
        result = self._nvrx_result(stream, "Fail Undetermined")
        _run_post_outcome(stream, ji, _make_symptoms(obs_ctx), outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Fail Undetermined"

    def test_nvrx_success_does_not_override_multicategory(self, stream, obs_ctx):
        # A clean 0:0 must NOT override a multi-category failure - it can bundle a real
        # hardware/infra component we must not mask (allowlist, not denylist).
        ji = _job_info_with(
            state="COMPLETED", tags=frozenset({"nvrx"}), exit_code=0, exit_code_signal=0, time_limit=None
        )
        result = self._nvrx_result(stream, "Fail MultiCategory")
        _run_post_outcome(stream, ji, _make_symptoms(obs_ctx), outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Fail MultiCategory"

    def test_user_cancellation_reclassified(self, stream, obs_ctx):
        ji = _job_info_with(cancelled_by="testuser")
        bad_nodes_result = CalcBadNodesAttributionResult()
        result = AttributionResult(event_stream=stream, bad_nodes_attr=bad_nodes_result)
        result.outcome_attr = _make_outcome("Fail User")

        symptoms = _make_symptoms(obs_ctx, "Python Exception")
        _run_post_outcome(stream, ji, symptoms, outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Cancelled by User"

    def test_system_account_not_reclassified(self, stream, obs_ctx):
        ji = _job_info_with(cancelled_by="root")
        bad_nodes_result = CalcBadNodesAttributionResult()
        result = AttributionResult(event_stream=stream, bad_nodes_attr=bad_nodes_result)
        result.outcome_attr = _make_outcome("Fail User")

        symptoms = _make_symptoms(obs_ctx, "Python Exception")
        _run_post_outcome(stream, ji, symptoms, outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Fail User"

    def test_infra_hw_not_reclassified_by_user_cancel(self, stream, obs_ctx):
        ji = _job_info_with(cancelled_by="testuser")
        bad_nodes_result = CalcBadNodesAttributionResult()
        result = AttributionResult(event_stream=stream, bad_nodes_attr=bad_nodes_result)
        result.outcome_attr = _make_outcome("Fail InfraHW")

        symptoms = _make_symptoms(obs_ctx, "XID-94-CONTAINED_ERROR")
        _run_post_outcome(stream, ji, symptoms, outcome=result.outcome_attr, attribution_result=result)
        assert result.outcome_attr.value().outcome == "Fail InfraHW"


class TestSingleNodeInfraBackfill:
    """SingleNodeInfraBackfill refiner: blame the sole node of a single-node job when
    the outcome is Fail InfraHW and no node-scoped evidence named it. Fail InfraSW is
    excluded (often a remote-end issue), multi-node jobs are not inferred (non-goal),
    and existing bad-node evidence is not overridden.
    """

    def _infra_result(self, stream, outcome_str="Fail InfraHW", root_cause="XID-79-GPU_OFF_BUS"):
        result = AttributionResult(event_stream=stream, bad_nodes_attr=CalcBadNodesAttributionResult())
        result.outcome_attr = _make_outcome(outcome_str, hw_cause="GPU", root_cause=root_cause)
        return result

    def test_single_node_infra_hw_backfills_sole_node(self, stream, obs_ctx, job_info):
        result = self._infra_result(stream)
        my_bad_nodes = JobNodeAttributions()
        _run_post_outcome(
            stream,
            job_info,
            _make_symptoms(obs_ctx),
            outcome=result.outcome_attr,
            attribution_result=result,
            my_bad_nodes=my_bad_nodes,
        )
        assert "node01" in my_bad_nodes.attributions
        assert "XID-79-GPU_OFF_BUS" in my_bad_nodes["node01"].symptoms()

    def test_single_node_infra_sw_is_not_backfilled(self, stream, obs_ctx, job_info):
        # Fail InfraSW single-node failures (e.g. enroot/curl) are often remote-end
        # issues, so the sole node must not be blamed - only Fail InfraHW backfills.
        result = self._infra_result(stream, outcome_str="Fail InfraSW")
        my_bad_nodes = JobNodeAttributions()
        _run_post_outcome(
            stream,
            job_info,
            _make_symptoms(obs_ctx),
            outcome=result.outcome_attr,
            attribution_result=result,
            my_bad_nodes=my_bad_nodes,
        )
        assert len(my_bad_nodes.attributions) == 0

    def test_multi_node_infra_failure_does_not_infer(self, stream, obs_ctx):
        ji = default_job_info(
            nodes=["node01", "node02"], job_end_time=datetime(2024, 1, 1, 11, 0, 0, tzinfo=ZoneInfo("UTC"))
        )
        result = self._infra_result(stream)
        my_bad_nodes = JobNodeAttributions()
        _run_post_outcome(
            stream,
            ji,
            _make_symptoms(obs_ctx),
            outcome=result.outcome_attr,
            attribution_result=result,
            my_bad_nodes=my_bad_nodes,
        )
        assert len(my_bad_nodes.attributions) == 0

    def test_non_node_attributable_symptom_is_not_backfilled(self, stream, obs_ctx, job_info):
        # Fabric/collective/catastrophic symptoms are not a single node's fault, so
        # the refiner must not pin them to the sole node (mirrors get_clean_bad_nodes).
        result = self._infra_result(stream, root_cause="PyTorch NCCL watchdog")
        my_bad_nodes = JobNodeAttributions()
        _run_post_outcome(
            stream,
            job_info,
            _make_symptoms(obs_ctx),
            outcome=result.outcome_attr,
            attribution_result=result,
            my_bad_nodes=my_bad_nodes,
        )
        assert len(my_bad_nodes.attributions) == 0

    def test_existing_bad_node_evidence_not_overridden(self, stream, obs_ctx, job_info):
        # Single-node job whose sole node already has node-scoped evidence: the
        # has_bad() guard must short-circuit, leaving the original attribution intact.
        my_bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["XID-94-CONTAINED_ERROR"]})
        result = self._infra_result(stream)
        _run_post_outcome(
            stream,
            job_info,
            _make_symptoms(obs_ctx),
            outcome=result.outcome_attr,
            attribution_result=result,
            my_bad_nodes=my_bad_nodes,
        )
        # The pre-existing attribution is untouched and the refiner did not add its symptom.
        assert "XID-94-CONTAINED_ERROR" in my_bad_nodes["node01"].symptoms()
        assert "XID-79-GPU_OFF_BUS" not in my_bad_nodes["node01"].symptoms()
        assert list(my_bad_nodes.attributions) == ["node01"]
