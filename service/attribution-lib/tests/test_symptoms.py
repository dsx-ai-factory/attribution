# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import attribution_lib
from attribution_lib.job_info import JobInfo, JobInfoBuilder
from attribution_lib.observation import Context
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.sources import Source, SourceSet, SymptomSet
from attribution_lib.symptoms import symptom_set_operator as op
from attribution_lib.symptoms.base import BaseSymptom
from attribution_lib.symptoms.declarations import SymptomCategory
from attribution_lib.symptoms.factory import SymptomFactory
from attribution_lib.symptoms.registry_generation import (
    GeneratedSymptomMetadata,
    GeneratedSymptomRegistry,
)
from attribution_lib.symptoms.symptom_enums import RegisteredSymptoms, resolve_registered_symptom
from test_support.helpers import make_node_to_ranks


def _registry_from_metadata(metadata: dict[str, GeneratedSymptomMetadata]) -> GeneratedSymptomRegistry:
    """Build a metadata-only registry double for cascade/factory unit tests."""
    return GeneratedSymptomRegistry(
        registered_symptoms=frozenset(metadata),
        outcome_rules=(),
        metadata=metadata,
    )


@contextmanager
def _use_registry(registry: GeneratedSymptomRegistry):
    """Temporarily install a registry double as the active symptom registry."""
    previous = attribution_lib._symptom_registry
    attribution_lib._symptom_registry = registry
    try:
        yield
    finally:
        attribution_lib._symptom_registry = previous


def test_basesymptom_string_equality_and_hash():
    """BaseSymptom equality/hash behave consistently with string ids."""
    symptom_xid_145_first = BaseSymptom(symptom_id="XID-145")
    symptom_xid_145_second = BaseSymptom(symptom_id="XID-145")
    symptom_xid_43 = BaseSymptom(symptom_id="XID-43")

    assert symptom_xid_145_first == symptom_xid_145_second
    assert symptom_xid_145_first != symptom_xid_43
    # string compare
    assert symptom_xid_145_first == "XID-145"
    # set semantics use string-based hash
    symptom_set = {symptom_xid_145_first, symptom_xid_43}
    found_xid_145 = False
    for item in symptom_set:
        if str(item) == "XID-145":
            found_xid_145 = True
            break
    assert found_xid_145


def test_factory_reads_registry_metadata():
    """SymptomFactory pulls id/cascade list from registry for enums."""
    symptom = SymptomFactory.from_enum(RegisteredSymptoms.SYSTEM_POWERING_DOWN, sources=None, attributes=None)
    assert symptom.symptom_id
    # cascade_discards present for recognized entries
    assert isinstance(symptom.cascade_discards, list)


def test_factory_reads_properties_from_injected_registry():
    """Factory properties reflect an injected registry (Support cluster variant, hotfix rollbacks etc)."""
    target_enum = RegisteredSymptoms.SYSTEM_POWERING_DOWN
    mock_md = GeneratedSymptomMetadata(
        id=getattr(target_enum, "value", str(target_enum)),
        cascade_discards=("MOCK-DISCARD-ONE", "MOCK-DISCARD-TWO"),
    )
    mock_registry = _registry_from_metadata({mock_md.id: mock_md})

    with _use_registry(mock_registry):
        symptom = SymptomFactory.from_enum(target_enum, sources=None, attributes=None)

    assert symptom.symptom_id == mock_md.id
    assert set(symptom.cascade_discards) == {"MOCK-DISCARD-ONE", "MOCK-DISCARD-TWO"}


def test_factory_from_string_rejects_unknown():
    with pytest.raises(ValueError):
        SymptomFactory.from_string("UNKNOWN-SYMPTOM", sources=None, attributes=None)


def test_factory_from_string_accepts_registered():
    # Use a known registered id from enums/registry
    known_id = RegisteredSymptoms.SYSTEM_POWERING_DOWN.value
    s = SymptomFactory.from_string(known_id, sources=None, attributes=None)
    assert s.symptom_id == known_id


def _one_source_set(context: Context, value: str) -> SourceSet[str]:
    src = Source(context, None, None, datetime.now(ZoneInfo("UTC")), AllJobResources, None)
    return SourceSet.new_one(value, src)


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


@pytest.fixture
def mock_registry_linear_dag():
    """Mock a small DAG: ROOT -> MID -> LEAF and a substring-only target."""
    return _registry_from_metadata(
        {
            "ROOT": GeneratedSymptomMetadata(id="ROOT", cascade_discards=("MID",)),
            "MID": GeneratedSymptomMetadata(id="MID", cascade_discards=("LEAF", "SUBSTRING-ONLY")),
            "LEAF": GeneratedSymptomMetadata(id="LEAF", cascade_discards=()),
        }
    )


@pytest.fixture
def mock_registry_with_cycle():
    """Mock DAG containing a downstream cycle to ensure traversal halts correctly."""
    return _registry_from_metadata(
        {
            "ENTRY": GeneratedSymptomMetadata(id="ENTRY", cascade_discards=("CYCLE_A",)),
            "CYCLE_A": GeneratedSymptomMetadata(id="CYCLE_A", cascade_discards=("CYCLE_B",)),
            "CYCLE_B": GeneratedSymptomMetadata(id="CYCLE_B", cascade_discards=("CYCLE_A",)),
        }
    )


@pytest.fixture
def mock_registry_conditional_chain():
    """Mock DAG with both static and conditional edges that have their own subtrees."""
    return _registry_from_metadata(
        {
            "PRIMARY": GeneratedSymptomMetadata(
                id="PRIMARY",
                cascade_discards=("STATIC_CHILD",),
                conditional_discards=(("COND_CHILD", "test_conditional_hook"),),
            ),
            "STATIC_CHILD": GeneratedSymptomMetadata(id="STATIC_CHILD", cascade_discards=("STATIC_LEAF",)),
            "STATIC_LEAF": GeneratedSymptomMetadata(id="STATIC_LEAF", cascade_discards=()),
            "COND_CHILD": GeneratedSymptomMetadata(id="COND_CHILD", cascade_discards=("COND_LEAF",)),
            "COND_LEAF": GeneratedSymptomMetadata(id="COND_LEAF", cascade_discards=()),
        }
    )


def test_cascade_prune_dag_transitive(mock_registry_linear_dag, dummy_ctx):
    """Full DFS: pruning removes transitive nodes and substring-only matches."""

    @dataclass(frozen=True)
    class MockEnum:
        value: str

    root_enum = MockEnum("ROOT")

    symptoms = SymptomSet[str]()
    symptoms.add(_one_source_set(dummy_ctx, "ROOT"))
    symptoms.add(_one_source_set(dummy_ctx, "MID"))
    symptoms.add(_one_source_set(dummy_ctx, "LEAF"))
    symptoms.add(_one_source_set(dummy_ctx, "contains SUBSTRING-ONLY marker"))
    symptoms.add(_one_source_set(dummy_ctx, "other"))

    with _use_registry(mock_registry_linear_dag):
        op.cascade_prune(symptoms, root_enum.value)

    remaining = set(symptoms.symptoms())
    assert "other" in remaining
    assert "ROOT" in remaining
    assert "MID" not in remaining
    assert "LEAF" not in remaining
    for sym in remaining:
        assert "SUBSTRING-ONLY" not in str(sym)


def test_cascade_prune_handles_downstream_cycle(mock_registry_with_cycle, dummy_ctx):
    """Traversal should not loop indefinitely when downstream nodes reference each other."""

    @dataclass(frozen=True)
    class MockEnum:
        value: str

    root_enum = MockEnum("ENTRY")

    symptoms = SymptomSet[str]()
    for value in ("ENTRY", "CYCLE_A", "CYCLE_B", "OTHER"):
        symptoms.add(_one_source_set(dummy_ctx, value))

    with _use_registry(mock_registry_with_cycle):
        op.cascade_prune(symptoms, root_enum.value)

    remaining = set(symptoms.symptoms())
    assert "ENTRY" in remaining
    assert "OTHER" in remaining
    assert "CYCLE_A" not in remaining
    assert "CYCLE_B" not in remaining


def test_conditional_discard_traverses_child_subtrees(mock_registry_conditional_chain, dummy_ctx):
    """Conditional targets should prune their own descendants when predicate evaluates True."""

    @dataclass(frozen=True)
    class MockEnum:
        value: str

    def predicate(symptom_set, job):
        return job is not None and job.state == "TRIGGER"

    registry = replace(mock_registry_conditional_chain, predicates={"test_conditional_hook": predicate})

    symptoms = SymptomSet[str]()
    for value in ("PRIMARY", "STATIC_CHILD", "STATIC_LEAF", "COND_CHILD", "COND_LEAF"):
        symptoms.add(_one_source_set(dummy_ctx, value))

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=1), state="TRIGGER")
    with _use_registry(registry):
        op.cascade_prune(symptoms, "PRIMARY", job_info=job_info)

    remaining = set(symptoms.symptoms())
    assert remaining == {"PRIMARY"}


def test_conditional_discard_ignored_when_predicate_missing(dummy_ctx):
    """Missing predicate names should be treated as False, leaving those symptoms untouched."""

    @dataclass(frozen=True)
    class MockEnum:
        value: str

    # Replace registry with one pointing to a nonexistent predicate.
    broken_registry = _registry_from_metadata(
        {
            "PRIMARY": GeneratedSymptomMetadata(
                id="PRIMARY",
                cascade_discards=("STATIC_CHILD",),
                conditional_discards=(("COND_CHILD", "nonexistent_predicate"),),
            ),
            "STATIC_CHILD": GeneratedSymptomMetadata(id="STATIC_CHILD", cascade_discards=()),
            "COND_CHILD": GeneratedSymptomMetadata(id="COND_CHILD", cascade_discards=()),
        }
    )

    symptoms = SymptomSet[str]()
    for value in ("PRIMARY", "STATIC_CHILD", "COND_CHILD"):
        symptoms.add(_one_source_set(dummy_ctx, value))

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=1))
    with _use_registry(broken_registry):
        op.cascade_prune(symptoms, "PRIMARY", job_info=job_info)

    remaining = set(symptoms.symptoms())
    assert "STATIC_CHILD" not in remaining  # static edge still prunes
    assert "COND_CHILD" in remaining  # conditional edge skipped


def test_conditional_eap_failed_short_job(dummy_ctx):
    symptoms = SymptomSet[str]()
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.EAP_AUTHENTICATION_FAILED.value))

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=1.5))
    op.cascade_prune(symptoms, RegisteredSymptoms.EAP_AUTHENTICATION_FAILED.value, job_info=job_info)

    remaining = set(symptoms.symptoms())
    assert RegisteredSymptoms.EAP_AUTHENTICATION_FAILED.value not in remaining


def test_conditional_eap_failed_long_job(dummy_ctx):
    symptoms = SymptomSet[str]()
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.EAP_AUTHENTICATION_FAILED.value))

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    job_info = _build_job_info_for_test(start=start, end=start + timedelta(hours=3))
    op.cascade_prune(symptoms, RegisteredSymptoms.EAP_AUTHENTICATION_FAILED.value, job_info=job_info)

    remaining = set(symptoms.symptoms())
    assert RegisteredSymptoms.EAP_AUTHENTICATION_FAILED.value in remaining


def test_conditional_cancelled_timeout_exceeded_discards_network(dummy_ctx):
    symptoms = SymptomSet[str]()
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.CANCELLED_BY_TIMEOUT.value))
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.IB_ERROR_5_WR_FLUSH_ERR.value))
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.EFA_ERROR_CONNECTION_RESET.value))

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    job_info = _build_job_info_for_test(
        start=start,
        end=start + timedelta(hours=3),
        time_limit_minutes=60,
    )
    op.cascade_prune(symptoms, RegisteredSymptoms.CANCELLED_BY_TIMEOUT.value, job_info=job_info)

    remaining = list(symptoms.symptoms())
    assert RegisteredSymptoms.IB_ERROR_5_WR_FLUSH_ERR.value not in remaining
    assert all("EFA Error-" not in s for s in remaining)


def test_conditional_cancelled_timeout_not_exceeded_keeps_network(dummy_ctx):
    symptoms = SymptomSet[str]()
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.CANCELLED_BY_TIMEOUT.value))
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.IB_ERROR_5_WR_FLUSH_ERR.value))
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.EFA_ERROR_CONNECTION_RESET.value))

    start = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
    job_info = _build_job_info_for_test(
        start=start,
        end=start + timedelta(minutes=30),
        time_limit_minutes=240,
    )
    op.cascade_prune(symptoms, RegisteredSymptoms.CANCELLED_BY_TIMEOUT.value, job_info=job_info)

    remaining = list(symptoms.symptoms())
    assert RegisteredSymptoms.IB_ERROR_5_WR_FLUSH_ERR.value in remaining
    assert any("EFA Error-" in s for s in remaining)


def test_set_ece_failed_discards_modify_qp_variants_via_conditional(dummy_ctx):
    symptoms = SymptomSet[str]()
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.CALL_TO_IBV_SET_ECE_FAILED.value))
    symptoms.add(_one_source_set(dummy_ctx, "Call to ibv_modify_qp failed: eno 110 timeout"))

    op.cascade_prune(symptoms, RegisteredSymptoms.CALL_TO_IBV_SET_ECE_FAILED.value)

    remaining = list(symptoms.symptoms())
    assert all("Call to ibv_modify_qp failed" not in s for s in remaining)


def test_route_unhealthy_discards_xid_145_routing_variants(dummy_ctx):
    """Verify that Route Unhealthy symptom correctly prunes specific XID-145 routing symptoms."""
    symptoms = SymptomSet[str]()
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.ROUTE_UNHEALTHY.value))
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.XID_145_RLW_SRC_TRACK_4.value))
    symptoms.add(_one_source_set(dummy_ctx, RegisteredSymptoms.XID_145_RLW_SRC_TRACK_8.value))
    symptoms.add(_one_source_set(dummy_ctx, "OTHER-SYMPTOM"))

    op.cascade_prune(symptoms, RegisteredSymptoms.ROUTE_UNHEALTHY.value)

    remaining = set(symptoms.symptoms())
    assert RegisteredSymptoms.ROUTE_UNHEALTHY.value in remaining
    assert "OTHER-SYMPTOM" in remaining
    assert RegisteredSymptoms.XID_145_RLW_SRC_TRACK_4.value not in remaining
    assert RegisteredSymptoms.XID_145_RLW_SRC_TRACK_8.value not in remaining


def test_csp_event_discards_slurm_node_failure_and_node_not_responding(dummy_ctx):
    """CSP blame symptoms prune redundant node-failure / responsiveness signals.

    CSP symptoms are read from the generated registry by category, so new ones
    need no edit here; interesting-only HEALTH_EVENT signals have no outcome
    and are excluded.
    """
    node_failure = "Node failure reported by slurm"
    node_not_responding = "Node not responding"
    registry = attribution_lib.get_symptom_registry()

    csp_blame_symptoms = sorted(
        symptom_id
        for symptom_id in registry.symptoms_in_category(SymptomCategory.CSP_EVENT)
        if registry.outcome_for([symptom_id]) is not None
    )
    assert csp_blame_symptoms, "expected CSP blame symptoms in the active registry"

    for symptom_id in csp_blame_symptoms:
        symptoms = SymptomSet[str]()
        symptoms.add(_one_source_set(dummy_ctx, symptom_id))
        symptoms.add(_one_source_set(dummy_ctx, node_failure))
        symptoms.add(_one_source_set(dummy_ctx, node_not_responding))
        symptoms.add(_one_source_set(dummy_ctx, "UNRELATED-SYMPTOM"))

        op.cascade_prune(symptoms, symptom_id)

        remaining = set(symptoms.symptoms())
        assert symptom_id in remaining, f"{symptom_id}: variant pruned itself"
        assert "UNRELATED-SYMPTOM" in remaining, f"{symptom_id}: unrelated symptom pruned"
        assert node_failure not in remaining, f"{symptom_id}: failed to prune {node_failure!r}"
        assert node_not_responding not in remaining, f"{symptom_id}: failed to prune {node_not_responding!r}"


def test_factory_creates_route_unhealthy():
    """Verify that SymptomFactory can create the new Route Unhealthy symptom."""
    symptom = SymptomFactory.from_enum(RegisteredSymptoms.ROUTE_UNHEALTHY, sources=None, attributes=None)
    assert symptom.symptom_id == "Route Unhealthy"
    assert RegisteredSymptoms.XID_145_RLW_SRC_TRACK_4.value in symptom.cascade_discards
    assert RegisteredSymptoms.XID_145_RLW_SRC_TRACK_8.value in symptom.cascade_discards


def test_not_responding_enum_member_aliases_renamed_symptom():
    """Keep the former enum member available while exposing the new symptom name."""
    assert RegisteredSymptoms.NOT_RESPONDING is RegisteredSymptoms.NODE_NOT_RESPONDING
    assert RegisteredSymptoms.NOT_RESPONDING.value == "Node not responding"


class TestResolveRegisteredSymptomPrefixFallback:
    """Verify XID prefix fallback in resolve_registered_symptom."""

    def test_exact_match_takes_priority(self):
        result = resolve_registered_symptom("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0")
        assert result == RegisteredSymptoms.XID_149_4_NETIR_LINK_DOWN_ALIGNMENT_LOSS_0

    def test_generic_xid149_exact_match(self):
        result = resolve_registered_symptom("XID-149")
        assert result == RegisteredSymptoms.XID_149

    def test_unregistered_xid149_falls_back_to_xid_generic(self):
        result = resolve_registered_symptom("XID-149.21-NETIR_LINK_DOWN-RECEIEVED_TS1-0")
        assert result == RegisteredSymptoms.XID_GENERIC

    def test_arbitrary_xid_subcode_falls_back_to_xid_generic(self):
        result = resolve_registered_symptom("XID-149.99-UNKNOWN_SUBCODE-UNKNOWN_REASON-0")
        assert result == RegisteredSymptoms.XID_GENERIC

    def test_unrelated_unknown_still_raises(self):
        with pytest.raises(ValueError):
            resolve_registered_symptom("TOTALLY_UNKNOWN_SYMPTOM")
