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

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.job_info import JobInfo, JobInfoBuilder
from attribution_lib.observation import Observation
from attribution_lib.observation.inbound import (
    Context as ObservationContext,
    Format as ObservationFormat,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.outcome import calculate_outcome
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.sources import Source, SourceSet, SymptomSet
from attribution_lib.symptoms import (
    BaseSymptom,
    RegisteredSymptoms,
    translate_and_prune_node_attributions,
    translate_node_attributions_to_symptoms,
)
from attribution_lib.symptoms.event_to_symptom import UNKNOWN_SYMPTOM_PREFIX
from test_support.helpers import make_node_to_ranks


def _make_source_for_node(node_name: str) -> Source:
    now = datetime.now(ZoneInfo("UTC"))
    ctx = ObservationContext(
        time_interval=TimeInterval(start=now, end=now),
        resources=AllJobResources,
        source=ObservationSource.APPLOG,
        format=ObservationFormat.RAW_LOG_LINES,
        timezone=ZoneInfo("UTC"),
    )
    return Source(
        source_context=Observation.from_inbound(ctx, ["Example", "lines"]).context,
        line_num=1,
        datum=f"test line for {node_name}",
        time=now,
        resources=AllJobResources,
        rule="unit-test",
    )


def _build_job_info_for_test(
    *,
    start: datetime,
    end: datetime,
    time_limit_minutes: int | None = None,
    state: str = "RUNNING",
) -> JobInfo:
    builder = JobInfoBuilder(
        id="job-1",
        type="batch",
        state=state,
        cluster="cluster",
        job_start_time=start,
        node_to_ranks=make_node_to_ranks(["node-0"], 1),
        framework="framework",
        source_name="test",
        timezone=ZoneInfo("UTC"),
    )
    builder.slurm_node_infos({})
    builder.job_end_time(end)
    builder.interest_start_time(start)
    builder.interest_end_time(end)
    if time_limit_minutes is not None:
        builder.time_limit(time_limit_minutes)
    return builder.build()


def test_translate_uses_enum_backed_path():
    node_name = "node0"
    src = _make_source_for_node(node_name)

    # Build the legacy per-node string attributions directly
    job_attributions = JobNodeAttributions()
    symptom_id = RegisteredSymptoms.ECC_ERROR.value
    src_set = SourceSet(symptom_id).add(src)
    job_attributions.add([node_name], src_set)

    # Now translate into per-node SymptomSet[BaseSymptom]
    node_symptoms, skipped = translate_node_attributions_to_symptoms(job_attributions)

    assert skipped == set()
    assert node_name in node_symptoms
    per_node = node_symptoms[node_name]
    assert isinstance(per_node, SymptomSet)

    # We expect exactly one BaseSymptom with the ECC_ERROR id
    symptoms = list(per_node.symptoms())
    assert len(symptoms) == 1

    symptom = symptoms[0]
    assert isinstance(symptom, BaseSymptom)
    assert symptom.symptom_id == RegisteredSymptoms.ECC_ERROR.value


def test_translate_resolves_non_fatal_substring_to_non_fatal():
    """Any symptom id containing 'non-fatal' resolves to the NON_FATAL symptom: it keeps
    its original id but inherits NON_FATAL's cascade metadata (so the DAG treats all
    non-fatal variants uniformly)."""
    node_name = "node0"
    src = _make_source_for_node(node_name)

    job_attributions = JobNodeAttributions()
    job_attributions.add([node_name], SourceSet("(non-fatal) LustreError").add(src))

    node_symptoms, _ = translate_node_attributions_to_symptoms(job_attributions)

    symptoms = list(node_symptoms[node_name].symptoms())
    assert len(symptoms) == 1
    symptom = symptoms[0]
    # Original id preserved...
    assert symptom.symptom_id == "(non-fatal) LustreError"
    # ...but resolved to NON_FATAL, inheriting its self-discard cascade.
    assert list(symptom.cascade_discards) == ["non-fatal"]


def test_translate_tags_id_absent_from_registry_as_unknown():
    node_name = "node1"
    src = _make_source_for_node(node_name)

    # An id that is not declared in the active generated registry vocabulary
    # (and matches no declared prefix) must be tagged as unknown.
    undeclared_id = "LEGACY_ONLY_SYMPTOM"

    job_attributions = JobNodeAttributions()
    src_set = SourceSet(undeclared_id).add(src)
    job_attributions.add([node_name], src_set)

    node_symptoms, skipped = translate_node_attributions_to_symptoms(job_attributions)
    assert skipped == set()
    assert node_name in node_symptoms
    symptoms = list(node_symptoms[node_name].symptoms())
    assert len(symptoms) == 1
    assert symptoms[0].symptom_id == f"{UNKNOWN_SYMPTOM_PREFIX}{undeclared_id}"


def test_translate_tags_unregistered_symptom_as_unknown():
    node_name = "node2"
    src = _make_source_for_node(node_name)

    job_attributions = JobNodeAttributions()
    unknown_id = "UNREGISTERED_SYMPTOM"
    src_set = SourceSet(unknown_id).add(src)
    job_attributions.add([node_name], src_set)

    node_symptoms, skipped = translate_node_attributions_to_symptoms(job_attributions)
    assert skipped == set()
    assert node_name in node_symptoms
    symptoms = list(node_symptoms[node_name].symptoms())
    assert len(symptoms) == 1
    assert symptoms[0].symptom_id == f"{UNKNOWN_SYMPTOM_PREFIX}{unknown_id}"


def test_translate_and_prune_discards_eap_for_short_job():
    node_name = "node-short"
    src = _make_source_for_node(node_name)

    # Single-node attribution with EAP_AUTHENTICATION_FAILED
    job_attributions = JobNodeAttributions()
    symptom_id = RegisteredSymptoms.EAP_AUTHENTICATION_FAILED.value
    src_set = SourceSet(symptom_id).add(src)
    job_attributions.add([node_name], src_set)

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    # Elapsed < 2 hours → predicate should fire and discard the symptom
    job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=1.5))

    pruned, skipped = translate_and_prune_node_attributions(job_attributions, job_info)
    # Entire node entry should disappear because its only symptom was discarded.
    assert node_name not in pruned
    assert skipped == set()


def test_translate_and_prune_keeps_eap_for_long_job():
    node_name = "node-long"
    src = _make_source_for_node(node_name)

    job_attributions = JobNodeAttributions()
    symptom_id = RegisteredSymptoms.EAP_AUTHENTICATION_FAILED.value
    src_set = SourceSet(symptom_id).add(src)
    job_attributions.add([node_name], src_set)

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    # Elapsed > 2 hours → predicate should NOT discard the symptom
    job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=3))

    pruned, skipped = translate_and_prune_node_attributions(job_attributions, job_info)
    assert node_name in pruned

    per_node = pruned[node_name]
    remaining_ids = {s.symptom_id for s in per_node.symptoms()}
    assert RegisteredSymptoms.EAP_AUTHENTICATION_FAILED.value in remaining_ids
    assert skipped == set()


def test_translate_and_prune_discards_network_cascade_from_qp_failure():
    node_name = "node-net"
    src = _make_source_for_node(node_name)

    job_attributions = JobNodeAttributions()

    # Primary symptom and downstream network symptoms that should be pruned
    qp_fail = RegisteredSymptoms.CALL_TO_IBV_MODIFY_QP_FAILED.value
    conn_reset = RegisteredSymptoms.CONNECTION_RESET_BY_PEER.value
    conn_closed = RegisteredSymptoms.CONNECTION_CLOSED.value

    for symptom_id in (qp_fail, conn_reset, conn_closed):
        src_set = SourceSet(symptom_id).add(src)
        job_attributions.add([node_name], src_set)

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=1))

    pruned, skipped = translate_and_prune_node_attributions(job_attributions, job_info)
    assert node_name in pruned

    per_node = pruned[node_name]
    remaining_ids = {s.symptom_id for s in per_node.symptoms()}

    # Primary should remain
    assert qp_fail in remaining_ids
    # Cascaded network symptoms should be removed
    assert conn_reset not in remaining_ids
    assert conn_closed not in remaining_ids
    assert skipped == set()


def test_translate_and_prune_keeps_slurm_node_failure_root_cause():
    node_name = "node-failure"
    src = _make_source_for_node(node_name)

    job_attributions = JobNodeAttributions()

    # Primary SLURM node failure symptom and its cascade target.
    node_failure = RegisteredSymptoms.NODE_FAILURE_REPORTED_BY_SLURM.value
    imex_stopping = RegisteredSymptoms.STOPPING_NVIDIA_IMEX_SERVICE.value

    for symptom_id in (node_failure, imex_stopping):
        src_set = SourceSet(symptom_id).add(src)
        job_attributions.add([node_name], src_set)

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=1))

    pruned, skipped = translate_and_prune_node_attributions(job_attributions, job_info)
    assert node_name in pruned

    per_node = pruned[node_name]
    remaining_ids = {s.symptom_id for s in per_node.symptoms()}

    # Root-cause SLURM node failure should remain
    assert node_failure in remaining_ids
    # Its cascade-only child should be discarded
    assert imex_stopping not in remaining_ids
    assert skipped == set()


@pytest.mark.parametrize(
    "root_cause",
    [
        RegisteredSymptoms.SYSTEM_SERVICE_OOM_KILL.value,
        RegisteredSymptoms.OOM_KILL_OF_SYSTEM_POD.value,
    ],
)
def test_translate_and_prune_system_oom_discards_slurm_noise(root_cause):
    node_name = "node-system-oom"
    src = _make_source_for_node(node_name)

    job_attributions = JobNodeAttributions()
    cascades = (
        RegisteredSymptoms.NODE_NOT_RESPONDING.value,
        RegisteredSymptoms.NODE_FAILURE_REPORTED_BY_SLURM.value,
        RegisteredSymptoms.NODE_UNEXPECTEDLY_REBOOTED.value,
        RegisteredSymptoms.NODE_OFFLINE.value,
    )
    if root_cause == RegisteredSymptoms.OOM_KILL_OF_SYSTEM_POD.value:
        cascades = (*cascades, RegisteredSymptoms.USER_TASK_OOM_KILL.value)

    for symptom_id in (root_cause, *cascades):
        src_set = SourceSet(symptom_id).add(src)
        job_attributions.add([node_name], src_set)

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=1))

    pruned, skipped = translate_and_prune_node_attributions(job_attributions, job_info)
    assert node_name in pruned

    remaining_ids = {s.symptom_id for s in pruned[node_name].symptoms()}
    assert remaining_ids == {root_cause}
    assert skipped == set()


class TestUnregisteredXidSubSymptomPreservation:
    """Dynamically-generated XID sub-symptoms (e.g. XID-149.21-...) that are
    not explicitly in RegisteredSymptoms must still flow through the translation
    pipeline with their original symptom_id preserved."""

    def test_unregistered_xid149_translated_not_skipped(self):
        node_name = "node-xid149"
        src = _make_source_for_node(node_name)
        symptom_id = "XID-149.21-NETIR_LINK_DOWN-RECEIEVED_TS1-0"

        job_attributions = JobNodeAttributions()
        job_attributions.add([node_name], SourceSet(symptom_id).add(src))

        node_symptoms, skipped = translate_node_attributions_to_symptoms(job_attributions)

        assert skipped == set()
        assert node_name in node_symptoms
        symptoms = list(node_symptoms[node_name].symptoms())
        assert len(symptoms) == 1
        assert symptoms[0].symptom_id == symptom_id

    def test_registered_xid149_still_exact_match(self):
        node_name = "node-xid149-reg"
        src = _make_source_for_node(node_name)
        symptom_id = RegisteredSymptoms.XID_149_4_NETIR_LINK_DOWN_ALIGNMENT_LOSS_0.value

        job_attributions = JobNodeAttributions()
        job_attributions.add([node_name], SourceSet(symptom_id).add(src))

        node_symptoms, skipped = translate_node_attributions_to_symptoms(job_attributions)

        assert skipped == set()
        symptoms = list(node_symptoms[node_name].symptoms())
        assert symptoms[0].symptom_id == symptom_id

    def test_registered_and_unregistered_xid149_coexist(self):
        node_name = "node-xid149-both"
        src = _make_source_for_node(node_name)
        registered_id = RegisteredSymptoms.XID_149_4_NETIR_LINK_DOWN_ALIGNMENT_LOSS_0.value
        unregistered_id = "XID-149.21-NETIR_LINK_DOWN-RECEIEVED_TS1-0"

        job_attributions = JobNodeAttributions()
        for sid in (registered_id, unregistered_id):
            job_attributions.add([node_name], SourceSet(sid).add(src))

        node_symptoms, skipped = translate_node_attributions_to_symptoms(job_attributions)

        assert skipped == set()
        symptom_ids = {s.symptom_id for s in node_symptoms[node_name].symptoms()}
        assert registered_id in symptom_ids
        assert unregistered_id in symptom_ids

    def test_unregistered_xid149_survives_dag_pruning(self):
        node_name = "node-xid149-prune"
        src = _make_source_for_node(node_name)
        symptom_id = "XID-149.21-NETIR_LINK_DOWN-RECEIEVED_TS1-0"

        job_attributions = JobNodeAttributions()
        job_attributions.add([node_name], SourceSet(symptom_id).add(src))

        start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=1))

        pruned, skipped = translate_and_prune_node_attributions(job_attributions, job_info)

        assert skipped == set()
        assert node_name in pruned
        remaining_ids = {s.symptom_id for s in pruned[node_name].symptoms()}
        assert symptom_id in remaining_ids


class TestUnknownSymptomTagging:
    """Completely unresolvable symptoms (no enum match, no prefix match)
    are tagged with 'unknown-symptom-' prefix and passed through."""

    def test_unknown_symptom_tagged_and_passed_through(self):
        node_name = "node-unknown"
        src = _make_source_for_node(node_name)
        unknown_id = "COMPLETELY_NOVEL_ERROR"

        job_attributions = JobNodeAttributions()
        job_attributions.add([node_name], SourceSet(unknown_id).add(src))

        node_symptoms, skipped = translate_node_attributions_to_symptoms(job_attributions)

        assert skipped == set()
        assert node_name in node_symptoms
        symptoms = list(node_symptoms[node_name].symptoms())
        assert len(symptoms) == 1
        assert symptoms[0].symptom_id == f"{UNKNOWN_SYMPTOM_PREFIX}{unknown_id}"
        assert symptoms[0].cascade_discards == []

    def test_unknown_symptom_survives_dag_pruning(self):
        node_name = "node-unknown-prune"
        src = _make_source_for_node(node_name)
        unknown_id = "COMPLETELY_NOVEL_ERROR"

        job_attributions = JobNodeAttributions()
        job_attributions.add([node_name], SourceSet(unknown_id).add(src))

        start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=1))

        pruned, skipped = translate_and_prune_node_attributions(job_attributions, job_info)

        assert skipped == set()
        assert node_name in pruned
        remaining_ids = {s.symptom_id for s in pruned[node_name].symptoms()}
        assert f"{UNKNOWN_SYMPTOM_PREFIX}{unknown_id}" in remaining_ids

    def test_unknown_coexists_with_known_symptoms(self):
        node_name = "node-mixed"
        src = _make_source_for_node(node_name)
        known_id = RegisteredSymptoms.ECC_ERROR.value
        unknown_id = "COMPLETELY_NOVEL_ERROR"

        job_attributions = JobNodeAttributions()
        for sid in (known_id, unknown_id):
            job_attributions.add([node_name], SourceSet(sid).add(src))

        node_symptoms, skipped = translate_node_attributions_to_symptoms(job_attributions)

        assert skipped == set()
        symptom_ids = {s.symptom_id for s in node_symptoms[node_name].symptoms()}
        assert known_id in symptom_ids
        assert f"{UNKNOWN_SYMPTOM_PREFIX}{unknown_id}" in symptom_ids

    def test_unknown_symptom_maps_to_fail_undetermined(self):
        src = _make_source_for_node("node-outcome")
        tagged_id = f"{UNKNOWN_SYMPTOM_PREFIX}COMPLETELY_NOVEL_ERROR"

        symptoms: SymptomSet[str] = SymptomSet()
        symptoms.add(SourceSet(tagged_id).add(src))

        outcome_result = calculate_outcome(symptoms)
        assert outcome_result.value().outcome == "Fail Undetermined"
        assert outcome_result.value().root_cause_symptom == tagged_id
