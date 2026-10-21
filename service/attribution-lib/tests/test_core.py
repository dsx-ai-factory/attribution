# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from common.refiner_helpers import _get_earliest_watchdog_source

import attribution_lib
import attribution_lib.core as core
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.bundled_rules.common.outcome_helpers import apply_outcome_reclassification_rules
from attribution_lib.bundled_rules.common.scrub_helpers import get_clean_bad_nodes
from attribution_lib.core import (
    AttributionResult,
    CalcBadNodesAttributionResult,
    calculate_bad_nodes,
    calculate_symptoms,
)
from attribution_lib.job_info import JobInfoBuilder, Scheduler
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import Id
from attribution_lib.observation.processing import _extract_timestamp_from_log_line
from attribution_lib.outcome import calculate_outcome
from attribution_lib.resources import AllJobResources
from attribution_lib.resources.resource_set import ResourceSet
from attribution_lib.sources import Source, SourceSet, SymptomSet
from attribution_lib.symptoms import (
    project_node_symptoms_to_bad_nodes,
    translate_and_prune_node_attributions,
)
from attribution_lib.symptoms.registry_generation import GeneratedSymptomRegistry
from attribution_lib.utils import get_hostname
from test_support.helpers import create_source_set, make_node_to_ranks


def test_attribute_rejects_kai_until_supported(simple_job_factory):
    """KAI is a recognized identity, but attribution has no KAI rules yet."""
    with pytest.raises(NotImplementedError, match="kai"):
        core.attribute(
            event_stream=None,  # type: ignore[arg-type]
            job_info=simple_job_factory(scheduler=Scheduler.KAI),
            bad_node_result=None,  # type: ignore[arg-type]
            extract_job_metadata_from_logfile=False,
        )


def test_setup_rules_without_mapping_clears_previous_mapping(monkeypatch):
    """A forced reconfiguration must not retain inventory from an earlier run."""
    for attribute in ("initialized", "node_domain_resolver", "_rules_engine", "_symptom_registry"):
        monkeypatch.setattr(attribution_lib, attribute, getattr(attribution_lib, attribute))
    monkeypatch.setattr(attribution_lib, "ip_to_nodenames", {"192.0.2.1": "stale-node"})

    core.setup_rules(rules_paths=[], force=True)

    assert attribution_lib.ip_to_nodenames == {}
    assert get_hostname("192.0.2.1") == "192.0.2.1"


# Shared job info fixture for multiple test classes
@pytest.fixture
def core_test_job_info(simple_job_factory):
    """Standard job info for core tests - FAILED state with one node."""
    return simple_job_factory(state="FAILED", exit_code_signal=1)


@pytest.fixture
def source_set_factory(dummy_ctx):
    """Factory fixture for creating SourceSets in tests.

    Creates a SourceSet with a single Source, useful for building
    test scenarios that require symptom attribution sources.

    Usage:
        def test_example(source_set_factory):
            symptom = source_set_factory("XID-95-UNCONTAINED_ERROR")
            # symptom is a SourceSet with one Source
    """

    def _create(symptom_name, datum="TestDatum", line_num=1, time=None, rule="test_rule"):
        return create_source_set(dummy_ctx, symptom_name, datum, line_num, time, rule)

    return _create


@pytest.fixture
def configured_node_domain_resolver(monkeypatch):
    """Configure deterministic domain membership for domain-pruning tests.

    Args:
        monkeypatch: Pytest fixture used to restore process-global configuration.
    """
    node_domains = {
        "rack-a-node-01": "domain-a",
        "rack-a-node-02": "domain-a",
        "rack-a-node-03": "domain-a",
        "rack-a-node-04": "domain-a",
        "rack-b-node-01": "domain-b",
        "node133-t01": "domain-133",
        "node133-t02": "domain-133",
        "example-domain-u32-controller-01": "domain-u32",
        "example-domain-u32-controller-02": "domain-u32",
        "example-domain-u36-controller-01": "domain-u36",
    }
    monkeypatch.setattr(attribution_lib, "node_domain_resolver", node_domains.get)


class DummyInbound:
    def __init__(self):
        self.time_interval = 0
        self.source = None
        self.format = None
        self.timezone = ZoneInfo("UTC")


@pytest.fixture
def dummy_inbound():
    """Fixture providing a dummy inbound object."""
    return DummyInbound()


@pytest.fixture
def symptoms():
    """Fixture providing a fresh SymptomSet for each test."""
    return SymptomSet()


@pytest.fixture
def default_context():
    """Fixture providing default test context."""
    return {
        "node_name": "test_node_name",
        "rule": "test_rule",
        "job_id": "test_job_id",
        "discard_reason": "test_reason",
    }


class TestSymptomSetLogging:
    def test_discard_logging_with_context(self, caplog, symptoms, source_set_factory, default_context):
        source_set = source_set_factory("TestSymptom")
        symptoms.add(source_set)
        with caplog.at_level("DEBUG"):
            symptoms.discard("TestSymptom", context=default_context)
        log_lines = [r for r in caplog.records if "Symptoms Discarded" in r.message]
        assert any("Node Name: test_node_name" in r.message for r in log_lines)
        assert any("Rule: test_rule" in r.message for r in log_lines)
        assert any("Job ID: test_job_id" in r.message for r in log_lines)
        assert any("Discard Reason: test_reason" in r.message for r in log_lines)

    def test_discard_logging_no_diff(self, caplog, symptoms, source_set_factory, default_context):
        source_set = source_set_factory("TestSymptom")
        symptoms.add(source_set)
        with caplog.at_level("DEBUG"):
            symptoms.discard("NonExistentSymptom", context=default_context)
        log_lines = [r for r in caplog.records if "Symptoms Discarded" in r.message]
        assert not log_lines  # No log should be produced if nothing was removed

    def test_discard_seq_logging(self, caplog, symptoms, source_set_factory, default_context):
        source_set1 = source_set_factory("Symptom1", datum="TestDatum1", line_num=1)
        source_set2 = source_set_factory("Symptom2", datum="TestDatum2", line_num=2)
        symptoms.add(source_set1)
        symptoms.add(source_set2)
        with caplog.at_level("DEBUG"):
            symptoms.discard_seq(["Symptom1", "Symptom2"], context=default_context)
        log_lines = [r for r in caplog.records if "Symptoms Discarded" in r.message]
        assert any("Symptom1" in r.message and "Symptom2" in r.message for r in log_lines)
        assert any("Node Name: test_node_name" in r.message for r in log_lines)

    def test_discard_logging_partial_context(self, caplog, symptoms, source_set_factory):
        source_set = source_set_factory("TestSymptom")
        symptoms.add(source_set)
        context = {"node_name": "test_node_name"}
        with caplog.at_level("DEBUG"):
            symptoms.discard("TestSymptom", context=context)
        log_lines = [r for r in caplog.records if "Symptoms Discarded" in r.message]
        assert any("Node Name: test_node_name" in r.message for r in log_lines)
        assert not any("Rule:" in r.message for r in log_lines)
        assert not any("Job ID:" in r.message for r in log_lines)
        assert not any("Discard Reason:" in r.message for r in log_lines)


class TestNeverRanJob:
    """Tests for pre-start jobs with no allocated nodes."""

    def test_never_ran_job_emits_didnt_start_symptom(self, event_stream, simple_job_factory):
        job_info = simple_job_factory(node_to_ranks={}, state="PENDING")

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=JobNodeAttributions(),
        )

        assert "Job didn't start" in result

    def test_never_ran_job_suppresses_other_symptoms(self, event_stream, simple_job_factory, source_set_factory):
        """A pending job with no nodes suppresses stray per-node attribution."""
        job_info = simple_job_factory(node_to_ranks={}, state="PENDING")
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("XID-95-UNCONTAINED_ERROR"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=bad_nodes,
        )

        assert list(result.symptoms()) == ["Job didn't start"]

    def test_kueue_terminal_failed_job_without_node_evidence_reports_failure(
        self, event_stream, simple_job_factory
    ):
        job_info = simple_job_factory(
            node_to_ranks={},
            state="FAILED",
            scheduler=Scheduler.KUEUE,
        )

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=JobNodeAttributions(),
        )

        assert "job failed" in result
        assert "Job didn't start" not in result

    def test_kueue_type_without_scheduler_retains_didnt_start_fallback(
        self, event_stream, simple_job_factory
    ):
        job_info = simple_job_factory(node_to_ranks={}, state="FAILED", job_type="kueue")

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=JobNodeAttributions(),
        )

        assert list(result.symptoms()) == ["Job didn't start"]

    def test_non_kueue_terminal_failed_job_without_nodes_retains_didnt_start_fallback(
        self, event_stream, simple_job_factory
    ):
        job_info = simple_job_factory(
            node_to_ranks={},
            state="FAILED",
            job_type="runai",
        )

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=JobNodeAttributions(),
        )

        assert list(result.symptoms()) == ["Job didn't start"]


class TestRunningJob:
    """Characterize the RUNNING-state fallback in calculate_symptoms."""

    def test_running_job_emits_running_without_error(self, event_stream, simple_job_factory):
        """A RUNNING job with no evidence reports 'running without error'."""
        job_info = simple_job_factory(state="RUNNING")

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=JobNodeAttributions(),
        )

        assert "running without error" in result

    def test_kueue_running_job_preserves_node_symptoms(
        self, event_stream, simple_job_factory, source_set_factory
    ):
        """A running Kueue workload must not hide collected failure evidence."""
        job_info = simple_job_factory(state="RUNNING", scheduler=Scheduler.KUEUE)
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("XID-95-UNCONTAINED_ERROR"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=bad_nodes,
        )

        assert "XID-95-UNCONTAINED_ERROR" in result
        assert "running without error" not in result

    def test_non_kueue_running_job_retains_suppressing_fallback(
        self, event_stream, simple_job_factory, source_set_factory
    ):
        job_info = simple_job_factory(state="RUNNING", job_type="runai")
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("XID-95-UNCONTAINED_ERROR"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=bad_nodes,
        )

        assert list(result.symptoms()) == ["running without error"]


class TestJobLevelSymptoms:
    """Tests for failure symptoms that intentionally do not blame a node."""

    def test_job_level_symptom_drives_outcome_without_bad_node(
        self, event_stream, simple_job_factory, source_set_factory
    ):
        """An NVRx reporter event can fail the run while leaving bad nodes empty."""
        job_info = simple_job_factory(state="FAILED")
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["reporter-node"], source_set_factory("NVRx detected failure"))

        symptoms = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=bad_nodes,
        )
        outcome = calculate_outcome(symptoms)
        get_clean_bad_nodes(bad_nodes, job_info.cluster)

        assert "NVRx detected failure" in symptoms
        assert not any(group.has_bad() for group in bad_nodes.attributions.values())
        assert outcome.value().outcome == "Fail Undetermined"


class TestMissingSrunCommand:
    """Missing command metadata must not suppress available attribution evidence."""

    def test_missing_srun_cmd_preserves_node_symptoms(self, event_stream, simple_job_factory, source_set_factory):
        """A missing command must not hide node symptoms collected from telemetry."""
        job_info = simple_job_factory(srun_cmd="", state="FAILED")
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("XID-95-UNCONTAINED_ERROR"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=bad_nodes,
        )

        assert "XID-95-UNCONTAINED_ERROR" in result
        assert "srun didn't happen" not in result

    def test_missing_srun_cmd_without_evidence_uses_state_fallback(self, event_stream, simple_job_factory):
        """No telemetry falls through to the ordinary Slurm-state fallback."""
        job_info = simple_job_factory(srun_cmd="", state="FAILED")

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=job_info,
            my_bad_nodes=JobNodeAttributions(),
        )

        assert "job failed" in result
        assert "srun didn't happen" not in result


class TestXIDCascadeFiltering:
    """Tests for XID-171/172 cascaded symptom filtering in calculate_symptoms()."""

    def test_xid_171_discards_cascaded_symptoms(self, event_stream, core_test_job_info, source_set_factory):
        """Test that XID-171 causes XID-48, XID-63, XID-94, and cudaErrorContained to be discarded."""
        # Create symptoms with XID-171 and cascaded symptoms
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("XID-171-UNCORRECTABLE_DRAM_ERROR"))
        symptoms.add(source_set_factory("XID-48-GPU_ECC_DBE"))
        symptoms.add(source_set_factory("XID-63-INFOROM_DRAM_RETIREMENT_EVENT"))
        symptoms.add(source_set_factory("XID-94-CONTAINED_ERROR"))
        symptoms.add(source_set_factory("cudaErrorContained"))

        # Create bad_nodes with same symptoms
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("XID-171-UNCORRECTABLE_DRAM_ERROR"))
        bad_nodes.add("test-node-001", source_set_factory("XID-48-GPU_ECC_DBE"))
        bad_nodes.add("test-node-001", source_set_factory("XID-63-INFOROM_DRAM_RETIREMENT_EVENT"))
        bad_nodes.add("test-node-001", source_set_factory("XID-94-CONTAINED_ERROR"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=core_test_job_info,
            my_bad_nodes=bad_nodes,
        )

        # Verify XID-171 is present
        assert any("XID-171" in s for s in result.symptoms()), f"XID-171 not found in {list(result.symptoms())}"

        # Verify cascaded symptoms are discarded
        assert not any("XID-48" in s for s in result.symptoms()), (
            f"XID-48 should be discarded: {list(result.symptoms())}"
        )
        assert not any("XID-63" in s for s in result.symptoms()), (
            f"XID-63 should be discarded: {list(result.symptoms())}"
        )
        assert not any("XID-94" in s for s in result.symptoms()), (
            f"XID-94 should be discarded: {list(result.symptoms())}"
        )
        assert not any("cudaErrorContained" in s for s in result.symptoms()), (
            f"cudaErrorContained should be discarded: {list(result.symptoms())}"
        )

    def test_xid_172_discards_cascaded_symptoms_including_xid_13(
        self, event_stream, core_test_job_info, source_set_factory
    ):
        """Test that XID-172 causes XID-13, XID-48, XID-94, and cudaErrorContained to be discarded."""
        # Create symptoms with XID-172 and cascaded symptoms (including XID-13)
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("XID-172-UNCORRECTABLE_SRAM_ERROR"))
        symptoms.add(source_set_factory("XID-13-GR_EXCEPTION"))
        symptoms.add(source_set_factory("XID-48-GPU_ECC_DBE"))
        symptoms.add(source_set_factory("XID-94-CONTAINED_ERROR"))
        symptoms.add(source_set_factory("cudaErrorContained"))

        # Create bad_nodes with same symptoms
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("XID-172-UNCORRECTABLE_SRAM_ERROR"))
        bad_nodes.add("test-node-001", source_set_factory("XID-13-GR_EXCEPTION"))
        bad_nodes.add("test-node-001", source_set_factory("XID-48-GPU_ECC_DBE"))
        bad_nodes.add("test-node-001", source_set_factory("XID-94-CONTAINED_ERROR"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=core_test_job_info,
            my_bad_nodes=bad_nodes,
        )

        # Verify XID-172 is present
        assert any("XID-172" in s for s in result.symptoms()), f"XID-172 not found in {list(result.symptoms())}"

        # Verify cascaded symptoms are discarded (including XID-13)
        assert not any("XID-13" in s for s in result.symptoms()), (
            f"XID-13 should be discarded: {list(result.symptoms())}"
        )
        assert not any("XID-48" in s for s in result.symptoms()), (
            f"XID-48 should be discarded: {list(result.symptoms())}"
        )
        assert not any("XID-94" in s for s in result.symptoms()), (
            f"XID-94 should be discarded: {list(result.symptoms())}"
        )
        assert not any("cudaErrorContained" in s for s in result.symptoms()), (
            f"cudaErrorContained should be discarded: {list(result.symptoms())}"
        )

    def test_xid_172_does_not_discard_xid_13(self, event_stream, core_test_job_info, source_set_factory):
        """Test that XID-13 is only discarded when XID-172 is present, not by itself."""
        # Create symptoms with only XID-13 (no XID-172)
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("XID-13-GR_EXCEPTION"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=core_test_job_info,
            my_bad_nodes=bad_nodes,
        )

        # Verify XID-13 is preserved when XID-172 is absent
        assert any("XID-13" in s for s in result.symptoms()), (
            f"XID-13 should be preserved without XID-172: {list(result.symptoms())}"
        )

    def test_cascaded_symptoms_preserved_without_xid_171_172(
        self, event_stream, core_test_job_info, source_set_factory
    ):
        """Test that cascaded symptoms are preserved when XID-171/172 are absent."""
        # Create symptoms without XID-171/172
        # Note: XID-63 is always discarded by scrub_bad_nodes(), so we don't test it here
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("XID-48-GPU_ECC_DBE"))
        bad_nodes.add("test-node-001", source_set_factory("XID-94-CONTAINED_ERROR"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=core_test_job_info,
            my_bad_nodes=bad_nodes,
        )

        # Verify symptoms are preserved when XID-171/172 are absent
        assert any("XID-48" in s for s in result.symptoms()), f"XID-48 should be preserved: {list(result.symptoms())}"
        assert any("XID-94" in s for s in result.symptoms()), f"XID-94 should be preserved: {list(result.symptoms())}"


class TestNSNetworkConnectionLostDiscard:
    """Tests for NS network connection lost discarding Node failure reported by slurm in scrub_bad_nodes()."""

    def test_ns_network_connection_lost_discards_node_failure_reported_by_slurm(
        self, core_test_job_info, source_set_factory
    ):
        """'NS network connection lost' should discard 'Node failure reported by slurm' via DAG pruning."""
        # Create bad_nodes with both symptoms
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("NS network connection lost"))
        bad_nodes.add(["test-node-001"], source_set_factory("Node failure reported by slurm"))

        # Verify both symptoms are present before DAG translation/pruning
        assert "NS network connection lost" in bad_nodes["test-node-001"].symptoms()
        assert "Node failure reported by slurm" in bad_nodes["test-node-001"].symptoms()

        # Run translation + DAG pruning + projection back to JobNodeAttributions
        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        # Verify 'NS network connection lost' is preserved as a bad symptom
        assert "NS network connection lost" in bad_nodes["test-node-001"].symptoms(), (
            f"'NS network connection lost' should be preserved: {list(bad_nodes['test-node-001'].symptoms())}"
        )

        # Verify 'Node failure reported by slurm' is no longer a bad symptom
        assert "Node failure reported by slurm" not in bad_nodes["test-node-001"].symptoms(), (
            f"'Node failure reported by slurm' should be discarded as bad: "
            f"{list(bad_nodes['test-node-001'].symptoms())}"
        )

    def test_node_failure_reported_by_slurm_preserved_without_ns_network_connection_lost(
        self, core_test_job_info, source_set_factory
    ):
        """'Node failure reported by slurm' should be preserved when NS network connection lost is absent."""
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("Node failure reported by slurm"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "Node failure reported by slurm" in bad_nodes["test-node-001"].symptoms(), (
            f"'Node failure reported by slurm' should be preserved without "
            f"'NS network connection lost': {list(bad_nodes['test-node-001'].symptoms())}"
        )

    def test_ns_network_connection_lost_discards_io_error(self, core_test_job_info, source_set_factory):
        """'NS network connection lost' should discard 'IO error' via DAG pruning."""
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("NS network connection lost"))
        bad_nodes.add(["test-node-001"], source_set_factory("IO error"))

        assert "NS network connection lost" in bad_nodes["test-node-001"].symptoms()
        assert "IO error" in bad_nodes["test-node-001"].symptoms()

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "NS network connection lost" in bad_nodes["test-node-001"].symptoms(), (
            f"'NS network connection lost' should be preserved: {list(bad_nodes['test-node-001'].symptoms())}"
        )
        assert "IO error" not in bad_nodes["test-node-001"].symptoms(), (
            f"'IO error' should be discarded as bad: {list(bad_nodes['test-node-001'].symptoms())}"
        )

    def test_io_error_preserved_without_ns_network_connection_lost(self, core_test_job_info, source_set_factory):
        """'IO error' should be preserved when NS network connection lost is absent."""
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("IO error"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "IO error" in bad_nodes["test-node-001"].symptoms(), (
            f"'IO error' should be preserved without "
            f"'NS network connection lost': {list(bad_nodes['test-node-001'].symptoms())}"
        )


class TestProjectionDagDiscardBehavior:
    """Characterize project_node_symptoms_to_bad_nodes per-symptom DAG-discard behavior.

    When the DAG discards a bad symptom, the projection normally demotes it (bad ->
    interesting). Three symptoms deviate: 'EAP authentication recovery' is purged
    entirely, and the two cudaError symptoms stay bad. An empty node_symptoms_base
    means the DAG kept nothing, so every symptom hits its discard branch.

    (In extract_bad_interesting_epilog, a bad symptom lands in BOTH the bad and
    interesting sets; a demoted symptom is interesting-only; a purged one is in
    neither.)
    """

    def test_dag_discard_purge_keep_and_demote(self, source_set_factory):
        node = "test-node-001"
        bad_nodes = JobNodeAttributions()
        for sym in (
            "EAP authentication recovery",
            "cudaErrorECCUncorrectable",
            "cudaErrorIllegalAddress",
            "IO error",
        ):
            bad_nodes.add([node], source_set_factory(sym))

        # Empty base => the DAG kept nothing, so all four are treated as discarded.
        project_node_symptoms_to_bad_nodes(bad_nodes, {})

        bad: defaultdict[str, set[str]] = defaultdict(set)
        interesting: defaultdict[str, set[str]] = defaultdict(set)
        epilog: defaultdict[str, set[str]] = defaultdict(set)
        bad_nodes.extract_bad_interesting_epilog(bad, interesting, epilog)

        # EAP recovery is purged entirely: not bad, not interesting.
        assert "EAP authentication recovery" not in bad[node]
        assert "EAP authentication recovery" not in interesting[node]

        # The cudaError symptoms stay bad.
        assert "cudaErrorECCUncorrectable" in bad[node]
        assert "cudaErrorIllegalAddress" in bad[node]

        # The default: demoted from bad to interesting.
        assert "IO error" not in bad[node]
        assert "IO error" in interesting[node]


class TestSystemPoweringDownDiscard:
    """Tests for System is powering down discarding Node failure reported by slurm in scrub_bad_nodes()."""

    def test_system_powering_down_discards_node_failure_reported_by_slurm(self, core_test_job_info, source_set_factory):
        """'System is powering down' should discard 'Node failure reported by slurm' via DAG pruning."""
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("System is powering down"))
        bad_nodes.add(["test-node-001"], source_set_factory("Node failure reported by slurm"))

        assert "System is powering down" in bad_nodes["test-node-001"].symptoms()
        assert "Node failure reported by slurm" in bad_nodes["test-node-001"].symptoms()

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "System is powering down" in bad_nodes["test-node-001"].symptoms(), (
            f"'System is powering down' should be preserved: {list(bad_nodes['test-node-001'].symptoms())}"
        )
        assert "Node failure reported by slurm" not in bad_nodes["test-node-001"].symptoms(), (
            f"'Node failure reported by slurm' should be discarded as bad: "
            f"{list(bad_nodes['test-node-001'].symptoms())}"
        )

    def test_node_failure_reported_by_slurm_preserved_without_system_powering_down(
        self, core_test_job_info, source_set_factory
    ):
        """'Node failure reported by slurm' should be preserved when System is powering down is absent."""
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("Node failure reported by slurm"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "Node failure reported by slurm" in bad_nodes["test-node-001"].symptoms(), (
            f"'Node failure reported by slurm' should be preserved without "
            f"'System is powering down': {list(bad_nodes['test-node-001'].symptoms())}"
        )

    def test_system_powering_down_discards_fabric_manager_lost_connection_to_all_gpus(
        self, core_test_job_info, source_set_factory
    ):
        """'System is powering down' should discard 'Fabric manager lost connection to all GPUs' via DAG pruning."""
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("System is powering down"))
        bad_nodes.add(["test-node-001"], source_set_factory("Fabric manager lost connection to all GPUs"))

        assert "System is powering down" in bad_nodes["test-node-001"].symptoms()
        assert "Fabric manager lost connection to all GPUs" in bad_nodes["test-node-001"].symptoms()

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "System is powering down" in bad_nodes["test-node-001"].symptoms(), (
            f"'System is powering down' should be preserved: {list(bad_nodes['test-node-001'].symptoms())}"
        )
        assert "Fabric manager lost connection to all GPUs" not in bad_nodes["test-node-001"].symptoms(), (
            f"'Fabric manager lost connection to all GPUs' should be discarded as bad: "
            f"{list(bad_nodes['test-node-001'].symptoms())}"
        )

    def test_fabric_manager_lost_connection_to_all_gpus_preserved_without_system_powering_down(
        self, core_test_job_info, source_set_factory
    ):
        """'Fabric manager lost connection to all GPUs' should be preserved when System is powering down is absent."""
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("Fabric manager lost connection to all GPUs"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "Fabric manager lost connection to all GPUs" in bad_nodes["test-node-001"].symptoms(), (
            f"'Fabric manager lost connection to all GPUs' should be preserved without "
            f"'System is powering down': {list(bad_nodes['test-node-001'].symptoms())}"
        )


class TestNodeFailureReportedBySlurmDiscard:
    """Tests that specific symptoms are not pruned same-node by Node failure reported by slurm."""

    @pytest.mark.parametrize(
        "cascade_symptom",
        [
            "XID-145-RLW_SRC_TRACK-8",
            "XID-44-GR_FAULT_DURING_CTXSW",
            "XID-145-MultiCategory",
            "XID-45-PREEMPTIVE_REMOVAL",
            "NV_ERR_INVALID_STATE",
            "cudaErrorLaunchFailure",
        ],
    )
    def test_node_failure_reported_by_slurm_preserves_specific_symptom_on_standalone_node(
        self, core_test_job_info, source_set_factory, cascade_symptom
    ):
        """A standalone (non-domain) node keeps its more-specific symptom.

        These targets are pruned only when a different node in the same domain
        reports a primary, so a single non-domain node retains them even
        alongside 'Node failure reported by slurm'.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
            cascade_symptom: Symptom expected to be preserved.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("Node failure reported by slurm"))
        bad_nodes.add(["test-node-001"], source_set_factory(cascade_symptom))

        assert "Node failure reported by slurm" in bad_nodes["test-node-001"].symptoms()
        assert cascade_symptom in bad_nodes["test-node-001"].symptoms()

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "Node failure reported by slurm" in bad_nodes["test-node-001"].symptoms(), (
            f"'Node failure reported by slurm' should be preserved: {list(bad_nodes['test-node-001'].symptoms())}"
        )
        assert cascade_symptom in bad_nodes["test-node-001"].symptoms(), (
            f"'{cascade_symptom}' should be preserved on a standalone node: "
            f"{list(bad_nodes['test-node-001'].symptoms())}"
        )

    @pytest.mark.parametrize(
        "cascade_symptom",
        [
            "XID-145-RLW_SRC_TRACK-8",
            "XID-44-GR_FAULT_DURING_CTXSW",
            "XID-145-MultiCategory",
            "XID-45-PREEMPTIVE_REMOVAL",
            "NV_ERR_INVALID_STATE",
            "cudaErrorLaunchFailure",
        ],
    )
    def test_cascade_symptom_preserved_without_node_failure_reported_by_slurm(
        self, core_test_job_info, source_set_factory, cascade_symptom
    ):
        """Selected cascade symptoms should be preserved when Node failure reported by slurm is absent.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
            cascade_symptom: Symptom expected to remain.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory(cascade_symptom))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert cascade_symptom in bad_nodes["test-node-001"].symptoms(), (
            f"'{cascade_symptom}' should be preserved without "
            f"'Node failure reported by slurm': {list(bad_nodes['test-node-001'].symptoms())}"
        )


class TestNvErrResetRequired:
    """NV_ERR_RESET_REQUIRED means a prior GPU fault left the GPU awaiting reset."""

    def test_maps_to_fail_infrahw(self, source_set_factory):
        """A GPU awaiting reset is a node hardware fault, not a driver bug."""
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("NV_ERR_RESET_REQUIRED kern_bus_gp100.c"))

        outcome = calculate_outcome(symptoms).value()

        assert outcome.outcome == "Fail InfraHW"
        assert outcome.hw_cause == "Known HW Cause - Node"

    @pytest.mark.parametrize(
        "other_symptom",
        ["NV_ERR_GENERIC rs_server.c", "NV_ERR_INVALID_STATE nv_gpu_ops.c", "NV_ERR_TIMEOUT nv_gpu_ops.c"],
    )
    def test_outranks_other_nv_err_codes(self, source_set_factory, other_symptom):
        """Other driver calls against a GPU awaiting reset fail too, so their codes must not win."""
        reset = "NV_ERR_RESET_REQUIRED gpu_fabric_probe.c"
        symptoms = SymptomSet()
        symptoms.add(source_set_factory(reset))
        symptoms.add(source_set_factory(other_symptom))

        outcome = calculate_outcome(symptoms).value()

        assert outcome.root_cause_symptom == reset
        assert outcome.outcome == "Fail InfraHW"

    def test_generic_xid_still_wins(self, source_set_factory):
        """The XID that caused the reset state stays the root cause."""
        xid = "XID-999-UNKNOWN"
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("NV_ERR_RESET_REQUIRED gpu_fabric_probe.c"))
        symptoms.add(source_set_factory(xid))

        assert calculate_outcome(symptoms).value().root_cause_symptom == xid

    @pytest.mark.parametrize(
        "cascade_symptom",
        [
            "Failed to collect nvlink status info",
            "knvlinkUpdatePostRxDetectLinkMask_IMPL: Failed to update Rx Detect Link mask!",
        ],
    )
    def test_discards_nvlink_cascade_symptom(self, core_test_job_info, source_set_factory, cascade_symptom):
        """NVLink queries against a GPU awaiting reset fail as a consequence of the reset state.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
            cascade_symptom: Symptom expected to be pruned.
        """
        reset = "NV_ERR_RESET_REQUIRED gpu_fabric_probe.c"
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory(reset))
        bad_nodes.add(["test-node-001"], source_set_factory(cascade_symptom))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert reset in bad_nodes["test-node-001"].symptoms()
        assert cascade_symptom not in bad_nodes["test-node-001"].symptoms()


class TestFabricManagerLostConnectionToAllGpusDiscard:
    """Tests for Fabric manager lost connection to all GPUs discarding cascade symptoms."""

    @pytest.mark.parametrize(
        "cascade_symptom",
        [
            "Node failure reported by slurm",
            "Stopping nvidia-imex.service",
        ],
    )
    def test_fabric_manager_lost_connection_to_all_gpus_discards_cascade_symptom(
        self, core_test_job_info, source_set_factory, cascade_symptom
    ):
        """'Fabric manager lost connection to all GPUs' should discard same-node cascade symptoms.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
            cascade_symptom: Symptom expected to be pruned.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("Fabric manager lost connection to all GPUs"))
        bad_nodes.add(["test-node-001"], source_set_factory(cascade_symptom))

        assert "Fabric manager lost connection to all GPUs" in bad_nodes["test-node-001"].symptoms()
        assert cascade_symptom in bad_nodes["test-node-001"].symptoms()

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "Fabric manager lost connection to all GPUs" in bad_nodes["test-node-001"].symptoms(), (
            f"'Fabric manager lost connection to all GPUs' should be preserved: "
            f"{list(bad_nodes['test-node-001'].symptoms())}"
        )
        assert cascade_symptom not in bad_nodes["test-node-001"].symptoms(), (
            f"'{cascade_symptom}' should be discarded as bad: {list(bad_nodes['test-node-001'].symptoms())}"
        )

    @pytest.mark.parametrize(
        "cascade_symptom",
        [
            "XID-145-RLW_SRC_TRACK-8",
            "XID-44-GR_FAULT_DURING_CTXSW",
            "XID-145-MultiCategory",
            "XID-45-PREEMPTIVE_REMOVAL",
            "NV_ERR_INVALID_STATE",
            "cudaErrorLaunchFailure",
        ],
    )
    def test_fabric_manager_lost_connection_to_all_gpus_preserves_specific_symptom_on_standalone_node(
        self, core_test_job_info, source_set_factory, cascade_symptom
    ):
        """A standalone (non-domain) node keeps its more-specific symptom.

        These targets are pruned only when a different node in the same domain
        reports a primary, so a single non-domain node retains them even
        alongside 'Fabric manager lost connection to all GPUs'.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
            cascade_symptom: Symptom expected to be preserved.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory("Fabric manager lost connection to all GPUs"))
        bad_nodes.add(["test-node-001"], source_set_factory(cascade_symptom))

        assert "Fabric manager lost connection to all GPUs" in bad_nodes["test-node-001"].symptoms()
        assert cascade_symptom in bad_nodes["test-node-001"].symptoms()

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "Fabric manager lost connection to all GPUs" in bad_nodes["test-node-001"].symptoms(), (
            f"'Fabric manager lost connection to all GPUs' should be preserved: "
            f"{list(bad_nodes['test-node-001'].symptoms())}"
        )
        assert cascade_symptom in bad_nodes["test-node-001"].symptoms(), (
            f"'{cascade_symptom}' should be preserved on a standalone node: "
            f"{list(bad_nodes['test-node-001'].symptoms())}"
        )

    @pytest.mark.parametrize(
        "cascade_symptom",
        [
            "Node failure reported by slurm",
            "Stopping nvidia-imex.service",
            "XID-145-RLW_SRC_TRACK-8",
            "XID-44-GR_FAULT_DURING_CTXSW",
            "XID-145-MultiCategory",
            "XID-45-PREEMPTIVE_REMOVAL",
            "NV_ERR_INVALID_STATE",
            "cudaErrorLaunchFailure",
        ],
    )
    def test_cascade_symptom_preserved_without_fabric_manager_lost_connection_to_all_gpus(
        self, core_test_job_info, source_set_factory, cascade_symptom
    ):
        """Selected cascade symptoms should be preserved when Fabric manager loss is absent.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
            cascade_symptom: Symptom expected to remain.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["test-node-001"], source_set_factory(cascade_symptom))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert cascade_symptom in bad_nodes["test-node-001"].symptoms(), (
            f"'{cascade_symptom}' should be preserved without 'Fabric manager lost connection to all GPUs': "
            f"{list(bad_nodes['test-node-001'].symptoms())}"
        )


@pytest.mark.usefixtures("configured_node_domain_resolver")
class TestDomainAwareCascadePruning:
    """Tests for domain-aware cascade pruning across sibling nodes."""

    @pytest.mark.parametrize(
        "primary_symptom",
        [
            "Node failure reported by slurm",
            "Fabric manager lost connection to all GPUs",
        ],
    )
    @pytest.mark.parametrize(
        "cascade_symptom",
        [
            "XID-145-RLW_SRC_TRACK-8",
            "XID-145-MultiCategory",
            "XID-145-RLW_SRC_TRACK-4",
            "XID-44-GR_FAULT_DURING_CTXSW",
            "XID-45-PREEMPTIVE_REMOVAL",
            "NV_ERR_INVALID_STATE",
            "cudaErrorLaunchFailure",
        ],
    )
    def test_primary_discards_specific_symptoms_in_same_node_domain(
        self, core_test_job_info, source_set_factory, primary_symptom, cascade_symptom
    ):
        """Domain primary should discard specific targets on sibling accelerator nodes.

        The primary lives on a different node than the targets, satisfying the
        'another node in the same domain' requirement.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
            primary_symptom: Primary symptom that triggers domain pruning.
            cascade_symptom: Specific symptom expected to be pruned in-domain.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["RACK-A-NODE-02"], source_set_factory(primary_symptom))
        bad_nodes.add(["RACK-A-NODE-01"], source_set_factory(cascade_symptom))
        bad_nodes.add(["RACK-A-NODE-04"], source_set_factory(cascade_symptom))
        bad_nodes.add(["RACK-B-NODE-01"], source_set_factory(cascade_symptom))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert primary_symptom in bad_nodes["rack-a-node-02"].symptoms()
        assert cascade_symptom not in bad_nodes["rack-a-node-01"].symptoms()
        assert cascade_symptom not in bad_nodes["rack-a-node-04"].symptoms()
        assert cascade_symptom in bad_nodes["rack-b-node-01"].symptoms()

    def test_lone_domain_node_keeps_target_with_own_primary(self, core_test_job_info, source_set_factory):
        """A single domain node carrying both a primary and a target keeps the target.

        With no *other* node in the domain reporting a primary, the more-specific
        symptom must survive.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["RACK-A-NODE-02"], source_set_factory("Node failure reported by slurm"))
        bad_nodes.add(["RACK-A-NODE-02"], source_set_factory("XID-44-GR_FAULT_DURING_CTXSW"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "Node failure reported by slurm" in bad_nodes["rack-a-node-02"].symptoms()
        assert "XID-44-GR_FAULT_DURING_CTXSW" in bad_nodes["rack-a-node-02"].symptoms(), (
            f"target should survive when no other domain node has a primary: "
            f"{list(bad_nodes['rack-a-node-02'].symptoms())}"
        )

    def test_two_primary_nodes_in_domain_prune_each_others_targets(self, core_test_job_info, source_set_factory):
        """Two domain nodes each with a primary + target prune each other's targets.

        Each node sees a primary on a *different* node, so both targets are pruned.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["RACK-A-NODE-02"], source_set_factory("Node failure reported by slurm"))
        bad_nodes.add(["RACK-A-NODE-02"], source_set_factory("XID-44-GR_FAULT_DURING_CTXSW"))
        bad_nodes.add(["RACK-A-NODE-03"], source_set_factory("Node failure reported by slurm"))
        bad_nodes.add(["RACK-A-NODE-03"], source_set_factory("XID-45-PREEMPTIVE_REMOVAL"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "XID-44-GR_FAULT_DURING_CTXSW" not in bad_nodes["rack-a-node-02"].symptoms()
        assert "XID-45-PREEMPTIVE_REMOVAL" not in bad_nodes["rack-a-node-03"].symptoms()

    def test_configured_resolver_groups_node_without_d(self, core_test_job_info, source_set_factory):
        """Configured domain membership should group accelerator names without a 'd'.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["node133-T02"], source_set_factory("Node failure reported by slurm"))
        bad_nodes.add(["node133-T01"], source_set_factory("XID-45-PREEMPTIVE_REMOVAL"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "Node failure reported by slurm" in bad_nodes["node133-t02"].symptoms()
        assert "XID-45-PREEMPTIVE_REMOVAL" not in bad_nodes["node133-t01"].symptoms()

    def test_configured_resolver_groups_mapped_domain(self, core_test_job_info, source_set_factory):
        """Configured domain membership should group mapped controllers.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(
            ["example-domain-u32-controller-01"], source_set_factory("Fabric manager lost connection to all GPUs")
        )
        bad_nodes.add(["example-domain-u32-controller-02"], source_set_factory("XID-145-RLW_SRC_TRACK-8"))
        bad_nodes.add(["example-domain-u36-controller-01"], source_set_factory("XID-145-RLW_SRC_TRACK-8"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "Fabric manager lost connection to all GPUs" in bad_nodes["example-domain-u32-controller-01"].symptoms()
        assert "XID-145-RLW_SRC_TRACK-8" not in bad_nodes["example-domain-u32-controller-02"].symptoms()
        assert "XID-145-RLW_SRC_TRACK-8" in bad_nodes["example-domain-u36-controller-01"].symptoms()

    def test_domain_pruning_is_disabled_without_resolver(self, core_test_job_info, source_set_factory, monkeypatch):
        """Domain-aware targets should remain when no resolver is configured.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
            monkeypatch: Pytest fixture used to override process-global configuration.
        """
        monkeypatch.setattr(attribution_lib, "node_domain_resolver", None)
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["RACK-A-NODE-02"], source_set_factory("Node failure reported by slurm"))
        bad_nodes.add(["RACK-A-NODE-01"], source_set_factory("XID-45-PREEMPTIVE_REMOVAL"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "XID-45-PREEMPTIVE_REMOVAL" in bad_nodes["rack-a-node-01"].symptoms()

    def test_intentionally_unknown_symptom_skips_scoped_lookup(self, core_test_job_info, source_set_factory):
        """Unknown placeholders should not be treated as scoped cascade primaries.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["RACK-A-NODE-02"], source_set_factory("Node failure reported by slurm"))
        bad_nodes.add(["RACK-A-NODE-01"], source_set_factory("unregistered test symptom"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)

        assert any(
            str(symptom).startswith("unknown-symptom-") for symptom in node_symptoms_base["rack-a-node-01"].symptoms()
        )

    def test_xid_symptoms_preserved_without_domain_primary(self, core_test_job_info, source_set_factory):
        """XID targets should stay bad when no primary exists in the domain.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add(["RACK-A-NODE-01"], source_set_factory("XID-145-RLW_SRC_TRACK-8"))
        bad_nodes.add(["RACK-A-NODE-04"], source_set_factory("XID-44-GR_FAULT_DURING_CTXSW"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "XID-145-RLW_SRC_TRACK-8" in bad_nodes["rack-a-node-01"].symptoms()
        assert "XID-44-GR_FAULT_DURING_CTXSW" in bad_nodes["rack-a-node-04"].symptoms()

    @pytest.mark.parametrize(
        "primary_node,cascade_node",
        [
            ("example-pool-00613", "example-pool-00614"),
            ("example-batch-block-02950", "example-batch-block-02951"),
            ("example-gpu-0133", "example-gpu-0134"),
            ("example-slurm-gpu-32", "example-slurm-gpu-33"),
        ],
    )
    def test_non_domain_formats_do_not_cross_prune(
        self, core_test_job_info, source_set_factory, primary_node, cascade_node
    ):
        """Broad cluster hostname prefixes should not create pruning domains.

        Args:
            core_test_job_info: Standard job info fixture.
            source_set_factory: Factory fixture for SourceSet instances.
            primary_node: Node carrying a primary symptom.
            cascade_node: Node carrying a candidate cascade symptom.
        """
        bad_nodes = JobNodeAttributions()
        bad_nodes.add([primary_node], source_set_factory("Node failure reported by slurm"))
        bad_nodes.add([cascade_node], source_set_factory("XID-45-PREEMPTIVE_REMOVAL"))

        node_symptoms_base, _ = translate_and_prune_node_attributions(bad_nodes, core_test_job_info)
        project_node_symptoms_to_bad_nodes(bad_nodes, node_symptoms_base)

        assert "XID-45-PREEMPTIVE_REMOVAL" in bad_nodes[cascade_node].symptoms()


class TestPyxisSymptomFiltering:
    """Tests for pyxis-related symptom filtering in calculate_symptoms()."""

    def test_pyxis_enroot_mount_failure_discards_related_symptoms(
        self, event_stream, core_test_job_info, source_set_factory
    ):
        """Test that pyxis enroot mount failure discards generic error symptoms."""
        # Create symptoms with pyxis enroot mount failure and related symptoms
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("pyxis: enroot mount failure from user directory"))
        bad_nodes.add("test-node-001", source_set_factory("No such file or directory"))
        bad_nodes.add("test-node-001", source_set_factory("error: pyxis"))
        bad_nodes.add("test-node-001", source_set_factory("pyxis: couldn't start container"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=core_test_job_info,
            my_bad_nodes=bad_nodes,
        )

        # Verify pyxis enroot mount failure is present
        assert any("pyxis: enroot mount failure from user directory" in s for s in result.symptoms()), (
            f"pyxis enroot mount failure not found in {list(result.symptoms())}"
        )

        # Verify generic symptoms are discarded
        assert not any("No such file or directory" in s for s in result.symptoms()), (
            f"'No such file or directory' should be discarded: {list(result.symptoms())}"
        )
        assert not any("error: pyxis" in s for s in result.symptoms()), (
            f"'error: pyxis' should be discarded: {list(result.symptoms())}"
        )
        assert not any("pyxis: couldn't start container" in s for s in result.symptoms()), (
            f"'pyxis: couldn't start container' should be discarded: {list(result.symptoms())}"
        )

    def test_pyxis_cannot_find_container_file_discards_related_symptoms(
        self, event_stream, core_test_job_info, source_set_factory
    ):
        """Test that pyxis cannot find container file discards generic error symptoms."""
        # Create symptoms with pyxis cannot find container file and related symptoms
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("pyxis: cannot find container file"))
        bad_nodes.add("test-node-001", source_set_factory("No such file or directory"))
        bad_nodes.add("test-node-001", source_set_factory("error: pyxis"))
        bad_nodes.add("test-node-001", source_set_factory("pyxis: couldn't start container"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=core_test_job_info,
            my_bad_nodes=bad_nodes,
        )

        # Verify pyxis cannot find container file is present
        assert any("pyxis: cannot find container file" in s for s in result.symptoms()), (
            f"pyxis cannot find container file not found in {list(result.symptoms())}"
        )

        # Verify generic symptoms are discarded
        assert not any("No such file or directory" in s for s in result.symptoms()), (
            f"'No such file or directory' should be discarded: {list(result.symptoms())}"
        )
        assert not any("error: pyxis" in s for s in result.symptoms()), (
            f"'error: pyxis' should be discarded: {list(result.symptoms())}"
        )
        assert not any("pyxis: couldn't start container" in s for s in result.symptoms()), (
            f"'pyxis: couldn't start container' should be discarded: {list(result.symptoms())}"
        )

    def test_generic_symptoms_preserved_without_pyxis_specific_errors(
        self, event_stream, core_test_job_info, source_set_factory
    ):
        """Test that generic symptoms are preserved when specific pyxis errors are absent."""
        # Create symptoms without specific pyxis errors
        bad_nodes = JobNodeAttributions()
        bad_nodes.add("test-node-001", source_set_factory("No such file or directory"))
        bad_nodes.add("test-node-001", source_set_factory("error: pyxis"))
        bad_nodes.add("test-node-001", source_set_factory("pyxis: couldn't start container"))

        result = calculate_symptoms(
            event_stream=event_stream("test-job"),
            job_info=core_test_job_info,
            my_bad_nodes=bad_nodes,
        )

        # Verify symptoms are preserved when specific pyxis errors are absent
        assert any("No such file or directory" in s for s in result.symptoms()), (
            f"'No such file or directory' should be preserved: {list(result.symptoms())}"
        )
        assert any("error: pyxis" in s for s in result.symptoms()), (
            f"'error: pyxis' should be preserved: {list(result.symptoms())}"
        )
        assert any("pyxis: couldn't start container" in s for s in result.symptoms()), (
            f"'pyxis: couldn't start container' should be preserved: {list(result.symptoms())}"
        )


class TestGetEarliestWatchdogSource:
    """Tests for the _get_earliest_watchdog_source function in core.py.

    This function finds the source with the earliest timestamp from
    PyTorch NCCL watchdog symptoms.
    """

    def test_get_earliest_watchdog_source_returns_earliest(self, source_set_factory, dummy_inbound):
        """Test that the function returns the source with the earliest timestamp.

        Setup: 3 PyTorch NCCL watchdog sources with different timestamps
        - Source 1: 06:58:25 (middle)
        - Source 2: 06:57:30 (earliest) <- should be returned
        - Source 3: 06:59:00 (latest)

        Example log format:
        2148: [rank2148]:[E1214 06:58:25.128361329 ProcessGroupNCCL.cpp:655] [Rank 36]
        Watchdog caught collective operation timeout: WorkNCCL(SeqNum=59598, OpType=COALESCED,
        NumelIn=12845056, NumelOut=616562688, Timeout(ms)=600000) ran for 600098 milliseconds
        before timing out.

        This test also verifies the body timestamp parsing handles the PyTorch log format.
        """

        # Log lines in PyTorch format
        datum_middle = "2148: [rank2148]:[E1214 06:58:25.128361329 ProcessGroupNCCL.cpp:655] [Rank 36] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=59598, OpType=COALESCED, NumelIn=12845056, NumelOut=616562688, Timeout(ms)=600000) ran for 600098 milliseconds before timing out."
        datum_earliest = "1024: [rank1024]:[E1214 06:57:30.128361329 ProcessGroupNCCL.cpp:655] [Rank 18] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=59590, OpType=COALESCED, NumelIn=12845056, NumelOut=616562688, Timeout(ms)=600000) ran for 600098 milliseconds before timing out."
        datum_latest = "3072: [rank3072]:[E1214 06:59:00.128361329 ProcessGroupNCCL.cpp:655] [Rank 54] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=59610, OpType=COALESCED, NumelIn=12845056, NumelOut=616562688, Timeout(ms)=600000) ran for 600098 milliseconds before timing out."

        context = ObservationContext(**vars(dummy_inbound), id=Id(12345))

        time_middle = _extract_timestamp_from_log_line(datum_middle, context.timezone)
        time_earliest = _extract_timestamp_from_log_line(datum_earliest, context.timezone)
        time_latest = _extract_timestamp_from_log_line(datum_latest, context.timezone)

        assert time_middle is not None
        assert time_earliest is not None
        assert time_latest is not None

        # Create 3 watchdog sources with extracted timestamps
        src_middle = Source(
            source_context=context,
            line_num=1,
            datum=datum_middle,
            time=time_middle,
            resources=ResourceSet(),
            rule="test_rule",
        )

        src_earliest = Source(
            source_context=context,
            line_num=2,
            datum=datum_earliest,
            time=time_earliest,
            resources=ResourceSet(),
            rule="test_rule",
        )

        src_latest = Source(
            source_context=context,
            line_num=3,
            datum=datum_latest,
            time=time_latest,
            resources=ResourceSet(),
            rule="test_rule",
        )

        # Create the "PyTorch NCCL watchdog" SourceSet with all 3 sources
        watchdog_source_set = SourceSet("PyTorch NCCL watchdog")
        watchdog_source_set.add(src_middle)
        watchdog_source_set.add(src_earliest)
        watchdog_source_set.add(src_latest)

        # Build the symptoms set
        symptoms = SymptomSet()
        symptoms.add(watchdog_source_set)

        # Verify initial state: 3 watchdog sources
        assert "PyTorch NCCL watchdog" in symptoms
        assert len(symptoms["PyTorch NCCL watchdog"]) == 3

        # Call the function
        result = _get_earliest_watchdog_source(symptoms)

        # Verify it returns the earliest source (06:57:30)
        assert result is not None
        assert result.time == time_earliest
        assert "rank1024" in result.datum
        # Verify the time ordering is correct
        assert time_earliest < time_middle < time_latest

    def test_get_earliest_watchdog_source_single_source(self, dummy_inbound):
        """Test with only one watchdog source - should return that source."""

        datum = "2148: [rank2148]:[E1214 06:58:25.128361329 ProcessGroupNCCL.cpp:655] [Rank 36] Watchdog caught collective operation timeout"

        context = ObservationContext(**vars(dummy_inbound), id=Id(12345))

        extracted_time = _extract_timestamp_from_log_line(datum, context.timezone)
        assert extracted_time is not None

        src = Source(
            source_context=context,
            line_num=1,
            datum=datum,
            time=extracted_time,
            resources=ResourceSet(),
            rule="test_rule",
        )

        watchdog_source_set = SourceSet("PyTorch NCCL watchdog")
        watchdog_source_set.add(src)

        symptoms = SymptomSet()
        symptoms.add(watchdog_source_set)

        result = _get_earliest_watchdog_source(symptoms)

        assert result is not None
        assert result.time == extracted_time

    def test_get_earliest_watchdog_source_with_none_timestamp(self, dummy_inbound):
        """Test that sources with None timestamp are skipped."""

        context = ObservationContext(**vars(dummy_inbound), id=Id(12345))
        # Source with valid timestamp
        src_valid = Source(
            source_context=context,
            line_num=1,
            datum="2148: [rank2148]: Watchdog caught collective operation timeout",
            time=datetime(2024, 12, 14, 6, 58, 25, tzinfo=context.timezone),
            resources=AllJobResources,
            rule="test_rule",
        )

        # Source with None timestamp
        src_no_time = Source(
            source_context=context,
            line_num=2,
            datum="1024: [rank1024]: Watchdog caught collective operation timeout",
            time=None,  # No timestamp
            resources=AllJobResources,
            rule="test_rule",
        )

        watchdog_source_set = SourceSet("PyTorch NCCL watchdog")
        watchdog_source_set.add(src_valid)
        watchdog_source_set.add(src_no_time)

        symptoms = SymptomSet()
        symptoms.add(watchdog_source_set)

        result = _get_earliest_watchdog_source(symptoms)

        # Should return the source with valid timestamp
        assert result is not None
        assert result.time == datetime(2024, 12, 14, 6, 58, 25, tzinfo=context.timezone)

    def test_get_earliest_watchdog_source_all_none_timestamps(self, dummy_inbound):
        """Test that None is returned when all sources have None timestamps."""
        src1 = Source(
            source_context=ObservationContext(**vars(dummy_inbound), id=Id(12345)),
            line_num=1,
            datum="source1",
            time=None,
            resources=ResourceSet(),
            rule="test_rule",
        )
        src2 = Source(
            source_context=ObservationContext(**vars(dummy_inbound), id=Id(12000)),
            line_num=2,
            datum="source2",
            time=None,
            resources=ResourceSet(),
            rule="test_rule",
        )

        watchdog_source_set = SourceSet("PyTorch NCCL watchdog")
        watchdog_source_set.add(src1)
        watchdog_source_set.add(src2)

        symptoms = SymptomSet()
        symptoms.add(watchdog_source_set)

        result = _get_earliest_watchdog_source(symptoms)

        # Should return None when no valid timestamps exist
        assert result is None


class TestTimeoutOutcome:
    """Tests for TIMEOUT outcome classification via reclassification rules."""

    @pytest.fixture
    def timeout_job_info(self, timezone):
        """Create a job info with TIMEOUT state for testing."""
        return (
            JobInfoBuilder(
                id="12345",
                cluster="test-cluster",
                state="TIMEOUT",
                type="training",
                framework="pytorch",
                node_to_ranks=make_node_to_ranks(["test-node-001"], 8),
                job_start_time=datetime.fromtimestamp(1700000000, tz=timezone),
                source_name="test",
                timezone=timezone,
            )
            .job_end_time(datetime.fromtimestamp(1700010000, tz=timezone))
            .srun_cmd("srun test.py")
            .exit_code_signal(0)
            .time_limit(6000)
            .build()
        )

    def test_timeout_only_reclassified_to_pass(self, event_stream, timeout_job_info, source_set_factory):
        """Test that timeout-only jobs get reclassified to Pass.

        When TIMEOUT is the only symptom (no other errors found), the outcome should be
        reclassified from 'Fail User' to 'Pass' via apply_outcome_reclassification_rules.
        """
        # Create symptoms with only Cancelled by Timeout
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("Cancelled by Timeout"))

        # Calculate initial outcome - should be Fail User
        outcome_result = calculate_outcome(symptoms)
        assert outcome_result.value().outcome == "Fail User", "Initial mapping should be Fail User"

        # Create AttributionResult for reclassification
        stream = event_stream("test-job")
        bad_nodes_result = CalcBadNodesAttributionResult()
        attribution_result = AttributionResult(
            bad_nodes_attr=bad_nodes_result,
            symptoms=symptoms,
            outcome_attr=outcome_result,
            cancelled_by="Timeout",
            event_stream=stream,
        )

        # Apply reclassification rules
        apply_outcome_reclassification_rules(attribution_result, stream, timeout_job_info)

        # Verify outcome is reclassified to Pass
        outcome = attribution_result.outcome_attr.value()
        assert outcome.outcome == "Pass", f"Expected 'Pass' after reclassification, got '{outcome.outcome}'"

    def test_timeout_with_other_symptoms_not_reclassified(self, event_stream, timeout_job_info, source_set_factory):
        """Test that timeout jobs with other symptoms are NOT reclassified to Pass.

        This addresses the reviewer's concern: if there's a Python Exception (or any other
        error symptom) along with a timeout, the job should NOT be marked as Pass.
        """
        # Create symptoms with Cancelled by Timeout AND Python Exception
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("Cancelled by Timeout"))
        symptoms.add(source_set_factory("Python Exception"))

        # Calculate initial outcome
        outcome_result = calculate_outcome(symptoms)

        # Create AttributionResult for reclassification
        stream = event_stream("test-job")
        bad_nodes_result = CalcBadNodesAttributionResult()
        attribution_result = AttributionResult(
            bad_nodes_attr=bad_nodes_result,
            symptoms=symptoms,
            outcome_attr=outcome_result,
            cancelled_by="Timeout",
            event_stream=stream,
        )

        # Apply reclassification rules
        apply_outcome_reclassification_rules(attribution_result, stream, timeout_job_info)

        # Verify outcome is NOT reclassified to Pass (should remain Fail User)
        outcome = attribution_result.outcome_attr.value()
        assert outcome.outcome == "Fail User", f"Expected 'Fail User' (not reclassified), got '{outcome.outcome}'"


class TestPLROutcomeIsolation:
    def test_plr_only_does_not_drive_failure_outcome(self, source_set_factory):
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("plr_tx_bw_loss_high_5m_1pct"))

        outcome_result = calculate_outcome(symptoms)

        assert outcome_result.value().outcome == "Incomplete Info"
        assert "plr_tx_bw_loss_high_5m_1pct" in list(symptoms.symptoms())

    def test_interesting_plr_rx_does_not_override_real_root_cause(self, source_set_factory):
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("plr_rx_bw_loss_high_10m_0p5pct"))
        symptoms.add(source_set_factory("XID-79-GPU_HAS_FALLEN_OFF_THE_BUS"))

        outcome_result = calculate_outcome(symptoms)

        assert outcome_result.value().outcome == "Fail InfraHW"
        assert outcome_result.value().root_cause_symptom.startswith("XID-79")
        assert "plr_rx_bw_loss_high_10m_0p5pct" in list(symptoms.symptoms())

    def test_plr_tx_ignore_prefix_blocks_outcome_even_if_mapped(self, source_set_factory, monkeypatch):
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("plr_tx_bw_loss_high_5m_1pct"))

        # Guardrail: even if a mapping is added later, ignored prefixes must not
        # drive root-cause outcome selection.
        mapped_registry = GeneratedSymptomRegistry(
            registered_symptoms=frozenset({"plr_tx_bw_loss_high"}),
            outcome_rules=({"plr_tx_bw_loss_high": ("Fail InfraHW", "Known HW Cause - Node")},),
            metadata={},
        )
        monkeypatch.setattr(attribution_lib, "_symptom_registry", mapped_registry)

        outcome_result = calculate_outcome(symptoms)

        assert outcome_result.value().outcome == "Incomplete Info"
        assert outcome_result.value().root_cause_symptom == ""


def test_xid_38_maps_to_fail_infrasw(source_set_factory):
    """XID-38 (FECS watchdog timeout, a driver firmware error) maps to 'Fail InfraSW'."""
    symptoms = SymptomSet()
    symptoms.add(source_set_factory("XID-38-FECS_ERR_WATCHDOG_TIMEOUT"))

    assert calculate_outcome(symptoms).value().outcome == "Fail InfraSW"


class TestCancelledByUserOutcome:
    """Tests for 'Cancelled by user' symptom mapping to 'Cancelled by User' outcome.

    User-cancelled jobs (with no other errors) should get 'Cancelled by User' outcome,
    which is distinct from 'Fail User' since a cancellation is not a failure.
    """

    def test_cancelled_by_user_maps_to_cancelled_by_user_outcome(self, source_set_factory):
        """Test that 'Cancelled by user' symptom maps to 'Cancelled by User' outcome.

        When a job is cancelled by the user and has no other error symptoms,
        the outcome should be 'Cancelled by User' (not 'Fail User').
        """
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("Cancelled by user"))

        outcome_result = calculate_outcome(symptoms)

        assert outcome_result.value().outcome == "Cancelled by User", (
            f"Expected 'Cancelled by User' outcome, got '{outcome_result.value().outcome}'"
        )

    def test_cancelled_by_user_with_other_symptoms_uses_other_symptom(self, source_set_factory):
        """Test that when other symptoms exist, they take priority over 'Cancelled by user'.

        The 'Cancelled by user' symptom is low priority and should be overridden
        by more specific error symptoms like XID errors.
        """
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("Cancelled by user"))
        symptoms.add(source_set_factory("XID-79-"))  # Higher priority HW error

        outcome_result = calculate_outcome(symptoms)

        # XID-79 should take priority, resulting in Fail InfraHW
        assert outcome_result.value().outcome == "Fail InfraHW", (
            f"Expected 'Fail InfraHW' (XID-79 takes priority), got '{outcome_result.value().outcome}'"
        )

    def test_cancelled_by_scheduler_maps_to_cancelled_by_scheduler_outcome(self, source_set_factory):
        """'Cancelled by cluster operators' symptom maps to 'Cancelled by cluster operators' outcome."""
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("Cancelled by cluster operators"))

        outcome_result = calculate_outcome(symptoms)

        assert outcome_result.value().outcome == "Cancelled by cluster operators", (
            f"Expected 'Cancelled by cluster operators' outcome, got '{outcome_result.value().outcome}'"
        )

    def test_cancelled_by_scheduler_with_other_symptoms_uses_other_symptom(self, source_set_factory):
        """Higher-priority error symptoms take precedence over 'Cancelled by cluster operators'."""
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("Cancelled by cluster operators"))
        symptoms.add(source_set_factory("XID-79-"))

        outcome_result = calculate_outcome(symptoms)

        assert outcome_result.value().outcome == "Fail InfraHW", (
            f"Expected 'Fail InfraHW' (XID-79 takes priority), got '{outcome_result.value().outcome}'"
        )

    @pytest.fixture
    def user_cancelled_job_info(self, timezone):
        """Create a job info with cancelled_by set to a regular user."""
        return (
            JobInfoBuilder(
                id="12345",
                cluster="test-cluster",
                state="CANCELLED",
                type="training",
                framework="pytorch",
                node_to_ranks=make_node_to_ranks(["test-node-001"], 8),
                job_start_time=datetime.fromtimestamp(1700000000, tz=timezone),
                source_name="test",
                timezone=timezone,
            )
            .job_end_time(datetime.fromtimestamp(1700010000, tz=timezone))
            .srun_cmd("srun test.py")
            .exit_code_signal(0)
            .cancelled_by("dnarayanan")  # Regular user
            .build()
        )

    def test_job_cancelled_by_user_reclassified_to_cancelled_by_user(
        self, event_stream, user_cancelled_job_info, source_set_factory
    ):
        """Test that jobs with cancelled_by set to a regular user get reclassified.

        When slurmdb_cancelled_by is a regular user (not a system account) AND
        the current outcome is 'Fail User', the outcome should be reclassified
        to 'Cancelled by User'.
        """
        # Create symptoms with Python Exception (would normally be Fail User)
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("Python Exception"))

        # Calculate initial outcome - should be Fail User
        outcome_result = calculate_outcome(symptoms)
        assert outcome_result.value().outcome == "Fail User", "Initial mapping should be Fail User"

        # Create AttributionResult for reclassification
        stream = event_stream("test-job")
        bad_nodes_result = CalcBadNodesAttributionResult()
        attribution_result = AttributionResult(
            bad_nodes_attr=bad_nodes_result,
            symptoms=symptoms,
            outcome_attr=outcome_result,
            cancelled_by="dnarayanan",
            event_stream=stream,
        )

        # Apply reclassification rules
        apply_outcome_reclassification_rules(attribution_result, stream, user_cancelled_job_info)

        # Verify outcome is reclassified to "Cancelled by User"
        outcome = attribution_result.outcome_attr.value()
        assert outcome.outcome == "Cancelled by User", (
            f"Expected 'Cancelled by User' after reclassification, got '{outcome.outcome}'"
        )
        assert "dnarayanan" in outcome.root_cause_symptom, (
            f"Expected username in root_cause_symptom, got '{outcome.root_cause_symptom}'"
        )

    def test_job_cancelled_by_system_not_reclassified(self, event_stream, source_set_factory, timezone):
        """Test that jobs cancelled by system accounts are NOT reclassified.

        System accounts like 'root', 'Timeout', or 'svc-*' should not trigger
        the 'Cancelled by User' reclassification.
        """
        # Create job info with system account
        system_job_info = (
            JobInfoBuilder(
                id="12345",
                cluster="test-cluster",
                state="CANCELLED",
                type="training",
                framework="pytorch",
                node_to_ranks=make_node_to_ranks(["test-node-001"], 8),
                job_start_time=datetime.fromtimestamp(1700000000, tz=timezone),
                source_name="test",
                timezone=timezone,
            )
            .job_end_time(datetime.fromtimestamp(1700010000, tz=timezone))
            .srun_cmd("srun test.py")
            .exit_code_signal(0)
            .cancelled_by("svc-example")  # Service account (svc- prefix marks non-user accounts)
            .build()
        )

        # Create symptoms with Python Exception
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("Python Exception"))

        # Calculate initial outcome - should be Fail User
        outcome_result = calculate_outcome(symptoms)
        assert outcome_result.value().outcome == "Fail User", "Initial mapping should be Fail User"

        # Create AttributionResult for reclassification
        stream = event_stream("test-job")
        bad_nodes_result = CalcBadNodesAttributionResult()
        attribution_result = AttributionResult(
            bad_nodes_attr=bad_nodes_result,
            symptoms=symptoms,
            outcome_attr=outcome_result,
            cancelled_by="svc-example",
            event_stream=stream,
        )

        # Apply reclassification rules
        apply_outcome_reclassification_rules(attribution_result, stream, system_job_info)

        # Verify outcome is NOT reclassified (should remain Fail User)
        outcome = attribution_result.outcome_attr.value()
        assert outcome.outcome == "Fail User", (
            f"Expected 'Fail User' (not reclassified for system account), got '{outcome.outcome}'"
        )

    def test_infra_hw_error_not_reclassified_even_when_cancelled_by_user(
        self, event_stream, user_cancelled_job_info, source_set_factory
    ):
        """Test that infrastructure errors are NOT reclassified to 'Cancelled by User'.

        When a job has a real hardware error (e.g., XID-48) AND was cancelled by a user,
        the hardware error should take priority. The user likely cancelled because
        they observed the hardware failure.
        """
        # Create symptoms with XID-48 (hardware error)
        symptoms = SymptomSet()
        symptoms.add(source_set_factory("XID-48"))

        # Calculate initial outcome - should be Fail InfraHW
        outcome_result = calculate_outcome(symptoms)
        assert outcome_result.value().outcome == "Fail InfraHW", "Initial mapping should be Fail InfraHW"

        # Create AttributionResult for reclassification
        stream = event_stream("test-job")
        bad_nodes_result = CalcBadNodesAttributionResult()
        attribution_result = AttributionResult(
            bad_nodes_attr=bad_nodes_result,
            symptoms=symptoms,
            outcome_attr=outcome_result,
            cancelled_by="dnarayanan",  # Regular user, but HW error takes priority
            event_stream=stream,
        )

        # Apply reclassification rules
        apply_outcome_reclassification_rules(attribution_result, stream, user_cancelled_job_info)

        # Verify outcome is NOT reclassified (HW error takes priority)
        outcome = attribution_result.outcome_attr.value()
        assert outcome.outcome == "Fail InfraHW", (
            f"Expected 'Fail InfraHW' (HW error takes priority over user cancellation), got '{outcome.outcome}'"
        )

    def test_cancelled_by_root_not_reclassified_to_cancelled_by_user(self, event_stream, source_set_factory, timezone):
        """Test that jobs cancelled by root keep 'Fail User' outcome and are not reclassified.

        When cancelled_by is 'root' (a system account) and the only symptom-derived
        outcome is 'Fail User' (e.g. from Python Exception that gets cleaned up in
        post_process_symptoms), the outcome should NOT be reclassified to
        'Cancelled by User'.
        """
        job_info = (
            JobInfoBuilder(
                id="466314",
                cluster="test-cluster",
                state="CANCELLED by 0",
                type="training",
                framework="megatron-lm",
                node_to_ranks=make_node_to_ranks(["test-node-001"], 8),
                job_start_time=datetime.fromtimestamp(1700000000, tz=timezone),
                source_name="test",
                timezone=timezone,
            )
            .job_end_time(datetime.fromtimestamp(1700010000, tz=timezone))
            .srun_cmd("srun test.py")
            .exit_code_signal(0)
            .cancelled_by("root")
            .build()
        )

        symptoms = SymptomSet()
        symptoms.add(source_set_factory("Python Exception"))

        outcome_result = calculate_outcome(symptoms)
        assert outcome_result.value().outcome == "Fail User"

        stream = event_stream("test-job")
        bad_nodes_result = CalcBadNodesAttributionResult()
        attribution_result = AttributionResult(
            bad_nodes_attr=bad_nodes_result,
            symptoms=symptoms,
            outcome_attr=outcome_result,
            cancelled_by="root",
            event_stream=stream,
        )

        apply_outcome_reclassification_rules(attribution_result, stream, job_info)

        assert attribution_result.outcome_attr.value().outcome == "Fail User", (
            f"Expected 'Fail User' (root is a system account, should not reclassify), "
            f"got '{attribution_result.outcome_attr.value().outcome}'"
        )


def test_system_service_oom_kill_maps_to_infrasw(source_set_factory):
    symptoms = SymptomSet()
    symptoms.add(source_set_factory("OOM kill of system service"))

    outcome_result = calculate_outcome(symptoms)

    outcome = outcome_result.value()
    assert outcome.outcome == "Fail InfraSW"
    assert outcome.root_cause_symptom == "OOM kill of system service"


def test_system_pod_oom_kill_maps_to_infrasw(source_set_factory):
    symptoms = SymptomSet()
    symptoms.add(source_set_factory("OOM kill of a system pod"))

    outcome_result = calculate_outcome(symptoms)

    outcome = outcome_result.value()
    assert outcome.outcome == "Fail InfraSW"
    assert outcome.root_cause_symptom == "OOM kill of a system pod"


def test_user_task_oom_kill_maps_to_fail_user(source_set_factory):
    symptoms = SymptomSet()
    symptoms.add(source_set_factory("OOM kill of user task"))

    outcome_result = calculate_outcome(symptoms)

    outcome = outcome_result.value()
    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == "OOM kill of user task"


def test_system_out_of_memory_outranks_ib_error_12(source_set_factory):
    symptoms = SymptomSet()
    symptoms.add(source_set_factory("System-out of memory"))
    symptoms.add(source_set_factory("IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)"))

    outcome_result = calculate_outcome(symptoms)

    outcome = outcome_result.value()
    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == "System-out of memory"


class TestSingleNodeNodeFail:
    """Tests for single-node NODE_FAIL node-level attribution.

    When SLURM reports NODE_FAIL but does not name a failed node, a single-node
    job implicates its sole node, so the bad-node fields are populated. Multi-node
    jobs are not inferred (non-goal). The attribution is performed by the
    SchedulerNodeFailure SCRUB-phase refiner, so these drive the full
    calculate_bad_nodes -> calculate_symptoms flow, which mutates my_bad_nodes.
    """

    _NODE_FAILURE_SYMPTOM = "Node failure reported by slurm"

    def _attributed_bad_nodes(self, event_stream, job_info):
        stream = event_stream("test-job")
        result = calculate_bad_nodes(event_stream=stream, job_info=job_info)
        calculate_symptoms(event_stream=stream, job_info=job_info, my_bad_nodes=result.my_bad_nodes)
        return result.my_bad_nodes

    def test_single_node_nodefail_no_failed_node_attributes_sole_node(self, event_stream, simple_job_factory):
        job_info = simple_job_factory(
            state="NODE_FAIL",
            nodes=["example-batch-node-00985"],
            job_type="slurm",
            scheduler=Scheduler.SLURM,
        )
        assert job_info.failed_node is None or job_info.failed_node == ""

        my_bad_nodes = self._attributed_bad_nodes(event_stream, job_info)

        assert "example-batch-node-00985" in my_bad_nodes.attributions
        assert self._NODE_FAILURE_SYMPTOM in my_bad_nodes["example-batch-node-00985"].symptoms()
        assert "" not in my_bad_nodes.attributions

        source = next(iter(my_bad_nodes["example-batch-node-00985"].attributions[self._NODE_FAILURE_SYMPTOM]))
        assert source.rule == "single_allocated_node_inference"

    def test_nodefail_with_failed_node_records_scheduler_field_provenance(self, event_stream, simple_job_factory):
        job_info = simple_job_factory(
            state="NODE_FAIL",
            nodes=["node-a", "node-b"],
            failed_node="node-b",
            job_type="slurm",
            scheduler=Scheduler.SLURM,
        )

        my_bad_nodes = self._attributed_bad_nodes(event_stream, job_info)

        source = next(iter(my_bad_nodes["node-b"].attributions[self._NODE_FAILURE_SYMPTOM]))
        assert source.rule == "slurm_failed_node_field"

    def test_multi_node_nodefail_no_failed_node_does_not_infer(self, event_stream, simple_job_factory):
        job_info = simple_job_factory(
            state="NODE_FAIL",
            nodes=["node-a", "node-b"],
            job_type="slurm",
            scheduler=Scheduler.SLURM,
        )
        assert job_info.failed_node is None or job_info.failed_node == ""

        my_bad_nodes = self._attributed_bad_nodes(event_stream, job_info)

        assert all(
            self._NODE_FAILURE_SYMPTOM not in my_bad_nodes[node].symptoms() for node in my_bad_nodes.attributions
        )
        assert "" not in my_bad_nodes.attributions

    def test_kueue_nodefail_uses_slurm_symptom_and_provenance(
        self, event_stream, simple_job_factory
    ):
        """Kueue uses the same "slurm" symptom/provenance as every other
        scheduler: a scheduler-specific string here would need to be added
        to every existing cascade_discards/pattern list that already
        references "Node failure reported by slurm", which is easy to miss.
        """
        job_info = simple_job_factory(
            state="NODE_FAIL",
            nodes=["node-a", "node-b"],
            failed_node="node-b",
            scheduler=Scheduler.KUEUE,
        )

        my_bad_nodes = self._attributed_bad_nodes(event_stream, job_info)

        source = next(iter(my_bad_nodes["node-b"].attributions[self._NODE_FAILURE_SYMPTOM]))
        assert source.rule == "slurm_failed_node_field"

    def test_non_kueue_nodefail_retains_slurm_symptom_and_provenance(
        self, event_stream, simple_job_factory
    ):
        job_info = simple_job_factory(
            state="NODE_FAIL",
            nodes=["node-a", "node-b"],
            failed_node="node-b",
            job_type="runai",
        )

        my_bad_nodes = self._attributed_bad_nodes(event_stream, job_info)

        source = next(iter(my_bad_nodes["node-b"].attributions[self._NODE_FAILURE_SYMPTOM]))
        assert source.rule == "slurm_failed_node_field"
