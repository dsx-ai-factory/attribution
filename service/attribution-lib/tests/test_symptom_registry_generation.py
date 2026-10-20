# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest

import attribution_lib
from attribution_lib.core import get_bundled_rules_path
from attribution_lib.exceptions import (
    DuplicateSymptomDeclarationError,
    DuplicateSymptomMetadataError,
    SymptomValidationError,
    UnknownSymptomError,
)
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.symptoms.base import BaseSymptom
from attribution_lib.symptoms.declarations import (
    CascadeScope,
    OutcomePriority,
    SymptomCategory,
    SymptomDeclaration,
)
from attribution_lib.symptoms.family import SymptomFamily
from attribution_lib.symptoms.registry_generation import (
    OUTCOME_PRIORITIES,
    GeneratedOutcomeMatch,
    GeneratedSymptomMetadata,
    known_prefixes_from_declarations,
    metadata_from_declarations,
    outcome_rules_from_declarations,
    registry_from_declarations,
    substring_aliases_from_declarations,
    symptom_docs_from_declarations,
)
from attribution_lib.symptoms.selector import SymptomSelector


def _load_committed_symptom_defaults() -> list[SymptomDeclaration]:
    loader = RuleLoader()
    loader.load(str(get_bundled_rules_path() / "symptom_defaults.py"))
    return list(loader.symptom_declarations)


def _load_committed_registry():
    return registry_from_declarations(_load_committed_symptom_defaults())


@pytest.mark.parametrize(
    ("symptoms", "expected"),
    [
        (["kernel panic", "ECC Error"], "ECC Error"),
        (["kernel panic", "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS"], "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS"),
        (["XID-145-RLW_RXPIPE-8", "XID-145-MultiCategory"], "XID-145-MultiCategory"),
        (
            [
                "kernel panic",
                "XID-145-RLW_RXPIPE-8",
                "XID-145-MultiCategory",
                "Call to ibv_modify_qp failed: eno 22 invalid argument(Mismatched GID)",
            ],
            "kernel panic",
        ),
    ],
)
def test_kernel_panic_priority_preserves_hardware_precedence(symptoms, expected):
    """Preserve hardware precedence and select panic over XID-145 and error 22.

    Args:
        symptoms: Synthetic symptoms to classify.
        expected: Root cause that should retain precedence.
    """
    match = _load_committed_registry().outcome_for(symptoms)

    assert match is not None
    assert match.symptom == expected


def test_known_prefixes_from_committed_declarations():
    declarations = _load_committed_symptom_defaults()

    prefixes = known_prefixes_from_declarations(declarations)

    assert "bus error" in prefixes
    assert "Python bug" in prefixes
    assert "XID-" in prefixes


def test_xid94_poison_victim_members_preserve_multicategory_outcome():
    """Declare poison-victim identity without assigning outcome policy."""

    registry = _load_committed_registry()
    expected_members = {
        "XID-94-CONTAINED_ERROR-CE": "CE",
        "XID-94-CONTAINED_ERROR-SM": "SM",
        "XID-94-CONTAINED_ERROR-GSP": "GSP",
    }

    for symptom_id, poison_victim in expected_members.items():
        properties = {
            "number": 94,
            "mnemonic": "CONTAINED_ERROR",
            "poison_victim": poison_victim,
        }
        assert registry.family_for(symptom_id) == SymptomFamily(name="xid", properties=properties)
        assert registry.selector_for(symptom_id) == SymptomSelector(
            family_name="xid",
            family_properties=properties,
        )

        outcome = registry.outcome_for([symptom_id])
        assert outcome is not None
        assert outcome.outcome == "Fail MultiCategory"


def test_cuda_not_ready_and_not_supported_classify_as_multicategory():
    """cudaErrorNotReady / cudaErrorNotSupported must resolve to Fail
    MultiCategory (consistent with cudaErrorDevicesUnavailable), not fall
    through to the generic cudaError -> Fail User default."""
    registry = _load_committed_registry()

    for symptom in ("cudaErrorNotReady", "cudaErrorNotSupported"):
        match = registry.outcome_for([symptom])
        assert match is not None, f"{symptom} produced no outcome match"
        assert match.outcome == "Fail MultiCategory", f"{symptom} -> {match.outcome}"


def test_nv_err_busy_retry_maps_to_infrasw():
    """Classify NV_ERR_BUSY_RETRY driver failures as infrastructure software."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["NV_ERR_BUSY_RETRY mem_multicast_fabric.c"])

    assert match is not None
    assert match.matched_prefix == "NV_ERR_BUSY_RETRY"
    assert match.outcome == "Fail InfraSW"


def test_nv_err_insufficient_resources_maps_to_multicategory():
    """Allow for users requesting more resources than are available."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["NV_ERR_INSUFFICIENT_RESOURCES mem_alloc.c"])

    assert match is not None
    assert match.matched_prefix == "NV_ERR_INSUFFICIENT_RESOURCES"
    assert match.outcome == "Fail MultiCategory"


def test_slurm_node_failure_maps_to_infrasw():
    """Treat the scheduler's NODE_FAIL state as infrastructure software evidence."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["Node failure reported by slurm"])

    assert match is not None
    assert match.matched_prefix == "Node failure reported by slurm"
    assert match.outcome == "Fail InfraSW"
    assert match.hw_cause == "Not Applicable"
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.MEDIUM)


def test_slurm_node_failure_takes_precedence_over_xid31_in_any_order():
    """Prefer the specific scheduler failure over ambiguous XID-31 evidence."""
    registry = _load_committed_registry()
    node_failure = "Node failure reported by slurm"
    xid31 = "XID-31-MMU_ERR_FLT"

    for symptoms in ([node_failure, xid31], [xid31, node_failure]):
        match = registry.outcome_for(symptoms)

        assert match is not None
        assert match.matched_prefix == node_failure
        assert match.outcome == "Fail InfraSW"
        assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.MEDIUM)


def test_node_not_responding_preserves_hardware_policy():
    """Rename the symptom without changing its hardware classification."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["Node not responding"])

    assert match is not None
    assert match.matched_prefix == "Node not responding"
    assert match.outcome == "Fail InfraHW"
    assert match.hw_cause == "Known HW Cause - Node"
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.HIGH)
    assert not registry.contains("Not responding")


def test_corroborated_node_failure_maps_to_infra_hw_and_discards_component_signals():
    """Classify the composite as hardware and keep its inputs as supporting evidence."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["Corroborated node failure"])

    assert match is not None
    assert match.matched_prefix == "Corroborated node failure"
    assert match.outcome == "Fail InfraHW"
    assert match.hw_cause == "Known HW Cause - Node"
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.HIGH_HW)
    assert registry.cascade_discards_for("Corroborated node failure") == [
        "Node failure reported by slurm",
        "Node not responding",
    ]


@pytest.mark.parametrize(
    "stronger_symptom",
    (
        "NMI watchdog hard LOCKUP",
        "XID-",
        "general protection fault",
        "Fabric manager lost connection to all GPUs",
    ),
)
def test_specific_node_failures_discard_corroborated_node_failure(stronger_symptom):
    """Keep the composite subordinate to independently identified causes."""
    registry = _load_committed_registry()

    assert "Corroborated node failure" in registry.cascade_discards_for(stronger_symptom)


def test_slurm_node_failure_yields_to_node_not_responding_in_any_order():
    """Prefer node non-response hardware evidence over the scheduler state."""
    registry = _load_committed_registry()
    node_failure = "Node failure reported by slurm"
    node_not_responding = "Node not responding"

    for symptoms in ([node_failure, node_not_responding], [node_not_responding, node_failure]):
        match = registry.outcome_for(symptoms)

        assert match is not None
        assert match.matched_prefix == node_not_responding
        assert match.outcome == "Fail InfraHW"
        assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.HIGH)


def test_wandb_error_maps_to_low_priority_fail_user():
    """Classify a generic W&B application exception as a low-priority user failure."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["wandb.Error"])

    assert match is not None
    assert match.matched_prefix == "wandb.Error"
    assert match.outcome == "Fail User"
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.LOW_USER)


def test_wandb_error_prefix_matches_suffixed_message():
    """Classify a W&B exception that includes its error message."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["wandb.Error: API request failed"])

    assert match is not None
    assert match.matched_prefix == "wandb.Error"
    assert match.outcome == "Fail User"


def test_wandb_error_yields_to_stronger_infrastructure_symptom_in_any_order():
    """Prefer infrastructure evidence over a generic W&B exception."""
    registry = _load_committed_registry()
    wandb_error = "wandb.Error: API request failed"
    infra_error = "NV_ERR_BUSY_RETRY mem_multicast_fabric.c"

    for symptoms in ([wandb_error, infra_error], [infra_error, wandb_error]):
        match = registry.outcome_for(symptoms)

        assert match is not None
        assert match.matched_prefix == "NV_ERR_BUSY_RETRY"
        assert match.outcome == "Fail InfraSW"


@pytest.mark.parametrize(
    ("code", "symptom"),
    [
        ("NV_ERR_FABRIC_STATE_OUT_OF_SYNC", "NV_ERR_FABRIC_STATE_OUT_OF_SYNC mem_multicast_fabric.c"),
        ("NV_ERR_BUFFER_TOO_SMALL", "NV_ERR_BUFFER_TOO_SMALL journal.c"),
        ("NV_ERR_INVALID_OBJECT", "NV_ERR_INVALID_OBJECT mem_multicast_fabric.c"),
        ("NV_ERR_TIMEOUT", "NV_ERR_TIMEOUT nv_gpu_ops.c"),
        ("NV_ERR_INVALID_STATE", "NV_ERR_INVALID_STATE nv_gpu_ops.c"),
        ("NV_ERR_OBJECT_NOT_FOUND", "NV_ERR_OBJECT_NOT_FOUND mem_multicast_fabric.c"),
        ("NV_ERR_GENERIC", "NV_ERR_GENERIC mem_multicast_fabric.c"),
        ("NV_ERR_INVALID_ARGUMENT", "NV_ERR_INVALID_ARGUMENT mem_multicast_fabric.c"),
        ("NV_ERR_OVERLAPPING_UVM_COMMIT", "NV_ERR_OVERLAPPING_UVM_COMMIT kernel_channel.c"),
    ],
)
def test_observed_nv_err_symptoms_map_to_infrasw(code: str, symptom: str):
    """Classify observed Resource Manager errors as infrastructure software.

    Args:
        code: Exact NV_ERR status code that must receive a specific mapping.
        symptom: Representative normalized symptom recorded by Attribution.
    """
    registry = _load_committed_registry()

    match = registry.outcome_for([symptom])

    assert match is not None
    assert match.matched_prefix == code
    assert match.outcome == "Fail InfraSW"


def test_nv_err_insufficient_permissions_has_no_specific_outcome_mapping():
    """Fall back to generic NV_ERR handling when filters are bypassed."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["NV_ERR_INSUFFICIENT_PERMISSIONS kern_profiler_v2_ctrl.c"])

    assert match is not None
    assert match.matched_prefix == "NV_ERR_"
    assert match.outcome == "Fail Undetermined"


@pytest.mark.parametrize(
    "symptom",
    [
        "NV_ERR_NO_MEMORY mem_desc.c",
        "NV_ERR_NO_MEMORY nv_gpu_ops.c",
        "NV_ERR_NO_MEMORY system_mem.c",
    ],
)
def test_non_multicast_nv_err_no_memory_maps_to_multicategory(symptom: str):
    """Keep non-multicast Resource Manager memory failures ambiguous.

    Args:
        symptom: Exact FACT DB symptom to classify.
    """
    registry = _load_committed_registry()

    match = registry.outcome_for([symptom])

    assert match is not None
    assert match.matched_prefix == "NV_ERR_NO_MEMORY"
    assert match.outcome == "Fail MultiCategory"
    assert match.hw_cause == "Not Applicable"


def test_multicast_nv_err_no_memory_maps_to_infrasw():
    """Classify multicast Resource Manager memory failure as infrastructure software."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["NV_ERR_NO_MEMORY mem_multicast_fabric.c"])

    assert match is not None
    assert match.matched_prefix == "NV_ERR_NO_MEMORY mem_multicast_fabric.c"
    assert match.outcome == "Fail InfraSW"
    assert match.hw_cause == "Driver bug"


@pytest.mark.parametrize(
    "non_multicast",
    [
        "NV_ERR_NO_MEMORY mem_desc.c",
        "NV_ERR_NO_MEMORY nv_gpu_ops.c",
        "NV_ERR_NO_MEMORY system_mem.c",
    ],
)
def test_mixed_nv_err_no_memory_maps_to_multicategory_in_any_order(non_multicast: str):
    """Keep mixed multicast and non-multicast memory failures ambiguous.

    Args:
        non_multicast: Non-multicast memory symptom paired with multicast evidence.
    """
    registry = _load_committed_registry()
    multicast = "NV_ERR_NO_MEMORY mem_multicast_fabric.c"

    for symptoms in ([non_multicast, multicast], [multicast, non_multicast]):
        match = registry.outcome_for(symptoms)

        assert match is not None
        assert match.matched_prefix == "NV_ERR_NO_MEMORY"
        assert match.outcome == "Fail MultiCategory"
        assert match.hw_cause == "Not Applicable"


def test_mixed_nv_err_no_memory_does_not_override_earlier_same_tier_match():
    """Preserve an unrelated same-tier match that sorts before no-memory evidence."""
    registry = _load_committed_registry()

    match = registry.outcome_for(
        [
            "Call to ibv_modify_qp failed: eno 110 timeout",
            "NV_ERR_NO_MEMORY mem_multicast_fabric.c",
            "NV_ERR_NO_MEMORY nv_gpu_ops.c",
        ]
    )

    assert match is not None
    assert match.matched_prefix == "Call to ibv_modify_qp failed: eno 110 timeout"
    assert match.outcome == "Fail InfraHW"


def test_bare_nv_err_no_memory_maps_to_multicategory():
    """Classify the bare memory status through the generic fallback."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["NV_ERR_NO_MEMORY"])

    assert match is not None
    assert match.matched_prefix == "NV_ERR_NO_MEMORY"
    assert match.outcome == "Fail MultiCategory"
    assert match.hw_cause == "Not Applicable"


def test_nvrx_detected_failure_maps_to_fail_undetermined():
    registry = _load_committed_registry()

    match = registry.outcome_for(["NVRx detected failure"])

    assert match is not None
    assert match.outcome == "Fail Undetermined"
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.LOW_GENERIC)


def test_nvrx_unhealthy_node_maps_to_fail_multicategory():
    """Classify an unhealthy NVRx node without asserting its root cause."""
    registry = _load_committed_registry()

    match = registry.outcome_for(["NVRx unhealthy node"])

    assert match is not None
    assert match.outcome == "Fail MultiCategory"
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.LOW_GENERIC)


@pytest.mark.parametrize(
    ("symptom", "declared"),
    [
        ("NVRx node health check failed: example_check", "NVRx node health check failed"),
        ("NVRx GPU recovery action: Operating at reduced capacity", "NVRx GPU recovery action"),
        ("NVRx step section timeout", "NVRx step section timeout"),
        ("NVRx no progress across restarts", "NVRx no progress across restarts"),
    ],
)
def test_nvrx_launcher_symptoms_map_to_fail_multicategory(symptom, declared):
    """Classify NVRx launcher symptoms like an unhealthy NVRx node, without asserting a root cause."""
    registry = _load_committed_registry()

    assert registry.resolve_symptom(symptom) == declared
    match = registry.outcome_for([symptom])
    assert match is not None
    assert match.matched_prefix == declared
    assert match.outcome == "Fail MultiCategory"
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.LOW_GENERIC)


def test_outcome_generation_preserves_within_tier_insertion_order():
    declarations = [
        SymptomDeclaration(
            prefix="cudaErrorDevicesUnavailable",
            outcome="Fail MultiCategory",
            priority=OutcomePriority.LOW_GENERIC,
        ),
        SymptomDeclaration(
            prefix="cudaError",
            outcome="Fail User",
            priority=OutcomePriority.LOW_GENERIC,
        ),
    ]

    generated_rules = outcome_rules_from_declarations(declarations)

    tier = generated_rules[OUTCOME_PRIORITIES.index(OutcomePriority.LOW_GENERIC)]
    assert list(tier.keys()) == ["cudaErrorDevicesUnavailable", "cudaError"]


def test_outcome_rules_from_declarations_rejects_duplicate_prefixes():
    declarations = [
        SymptomDeclaration(prefix="same", outcome="Fail User", priority=OutcomePriority.LOW_USER),
        SymptomDeclaration(prefix="same", outcome="Fail InfraSW", priority=OutcomePriority.LOW),
    ]

    with pytest.raises(DuplicateSymptomDeclarationError, match="Duplicate outcome declaration"):
        outcome_rules_from_declarations(declarations)


def test_metadata_from_declarations_rejects_duplicate_prefixes():
    declarations = [
        SymptomDeclaration(prefix="same", cascade_discards=("a",)),
        SymptomDeclaration(prefix="same", cascade_discards=("b",)),
    ]

    with pytest.raises(DuplicateSymptomMetadataError, match="Duplicate symptom metadata"):
        metadata_from_declarations(declarations)


def test_substring_aliases_reject_conflicting_substring():
    declarations = [
        SymptomDeclaration(prefix="non-fatal", resolve_contains="non-fatal"),
        SymptomDeclaration(prefix="other", resolve_contains="non-fatal"),
    ]

    with pytest.raises(DuplicateSymptomDeclarationError, match="Conflicting resolve_contains"):
        substring_aliases_from_declarations(declarations)


def test_substring_aliases_reject_nested_substring():
    # "fatal" is nested in "non-fatal", so a symptom containing "non-fatal" would
    # match both aliases; resolve_symptom returns the first hit, so the winner would
    # depend on load order. Reject the nesting at build time instead.
    declarations = [
        SymptomDeclaration(prefix="non-fatal", resolve_contains="non-fatal"),
        SymptomDeclaration(prefix="fatal-hw", resolve_contains="fatal"),
    ]

    with pytest.raises(DuplicateSymptomDeclarationError, match="Overlapping resolve_contains"):
        substring_aliases_from_declarations(declarations)


def test_substring_aliases_reject_empty_substring():
    # An empty needle is contained in every symptom and is checked before exact/prefix
    # matching, so it would hijack the whole vocabulary. Reject it at build time.
    declarations = [SymptomDeclaration(prefix="non-fatal", resolve_contains="")]

    with pytest.raises(SymptomValidationError, match="Empty resolve_contains"):
        substring_aliases_from_declarations(declarations)


def test_resolution_precedence_preserves_contains_exact_and_longest_fallback_order():
    """Lock the compatibility resolution order for the selector migration."""
    registry = registry_from_declarations(
        [
            SymptomDeclaration(prefix="alias", resolve_contains="[alias]"),
            SymptomDeclaration(prefix="exact [alias]"),
            SymptomDeclaration(prefix="family-", fallback_prefix=True),
            SymptomDeclaration(prefix="family-special-", fallback_prefix=True),
            SymptomDeclaration(prefix="family-exact"),
        ]
    )

    # Contains aliases intentionally run before both exact and fallback lookup.
    assert registry.resolve_symptom("exact [alias]") == "alias"
    assert registry.resolve_symptom("family-dynamic [alias]") == "alias"
    # Without an alias, exact lookup precedes fallback lookup.
    assert registry.resolve_symptom("family-exact") == "family-exact"
    # Overlapping fallbacks choose the longest, most-specific prefix.
    assert registry.resolve_symptom("family-special-dynamic") == "family-special-"


@pytest.mark.parametrize(
    ("prefixes", "expected_prefix"),
    [
        (("fault", "fault-specific"), "fault"),
        (("fault-specific", "fault"), "fault-specific"),
    ],
)
def test_outcome_selection_preserves_declaration_order_within_a_tier(prefixes, expected_prefix):
    """Lock first-declared prefix selection when one symptom matches both."""
    declarations = [
        SymptomDeclaration(
            prefix=prefix,
            outcome=f"outcome:{prefix}",
            priority=OutcomePriority.LOW,
        )
        for prefix in prefixes
    ]
    registry = registry_from_declarations(declarations)

    match = registry.outcome_for(["fault-specific detail"])

    assert match is not None
    assert match.matched_prefix == expected_prefix
    assert match.outcome == f"outcome:{expected_prefix}"


@pytest.mark.parametrize(
    "symptoms",
    [
        ("fault-specific", "fault-z"),
        ("fault-z", "fault-specific"),
    ],
)
def test_outcome_selection_preserves_broader_same_tier_override(symptoms):
    """Lock the broader-prefix override across distinct symptoms."""
    registry = registry_from_declarations(
        [
            SymptomDeclaration(
                prefix="fault-specific",
                outcome="specific outcome",
                priority=OutcomePriority.LOW,
            ),
            SymptomDeclaration(
                prefix="fault-",
                outcome="family outcome",
                priority=OutcomePriority.LOW,
            ),
        ]
    )

    match = registry.outcome_for(symptoms)

    assert match is not None
    assert match.symptom == "fault-z"
    assert match.matched_prefix == "fault-"
    assert match.outcome == "family outcome"


def test_outcome_generation_rejects_missing_hw_cause():
    declarations = [
        SymptomDeclaration(
            prefix="missing hw cause",
            outcome="Fail User",
            hw_cause=cast(str, None),
            priority=OutcomePriority.LOW_USER,
        )
    ]

    with pytest.raises(SymptomValidationError, match="missing hw_cause"):
        outcome_rules_from_declarations(declarations)


def test_generated_registry_facade_resolves_symptoms():
    registry = _load_committed_registry()

    assert registry.contains("XID-149")
    # Dynamic variants resolve only through generic fallback prefixes (XID-,
    # NV_ERR_, ...).
    assert registry.resolve_symptom("XID-149.21-NETIR_LINK_DOWN-RECEIEVED_TS1-0") == "XID-"
    assert registry.resolve_symptom("NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c") == "NV_ERR_"

    # A dash-suffixed prefix that is NOT a generic fallback (e.g. "DriverBug-")
    # must not resolve dynamic variants.
    with pytest.raises(UnknownSymptomError, match="known symptom"):
        registry.resolve_symptom("DriverBug-Bug1234")

    with pytest.raises(UnknownSymptomError, match="known symptom"):
        registry.resolve_symptom("not a registered symptom")


def test_generated_registry_resolves_all_declared_fallback_families():
    registry = _load_committed_registry()

    dynamic_to_prefix = {
        "XID-149.21-NETIR_LINK_DOWN-RECEIEVED_TS1-0": "XID-",
        "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c": "NV_ERR_",
        "NV_ERR_RESET_REQUIRED p2p_api.c": "NV_ERR_RESET_REQUIRED",
        "cudaErrorSomethingDynamic": "cudaError",
        "RuntimeError: dynamic message": "RuntimeError",
        "IB Error-99 something": "IB Error-",
        "EFA Error-7 something": "EFA Error-",
        "bad kernel stack frame pointer in some_dynamic_tool": "bad kernel stack frame pointer",
        "Call to ibv_modify_qp failed: eno 9999": "Call to ibv_modify_qp failed",
        "NVRx node health check failed: example_gpu_check": "NVRx node health check failed",
        "NVRx GPU recovery action: Requires a node reboot to recover": "NVRx GPU recovery action",
    }
    # Guard against drift: the cases above must exercise every declared fallback
    # family, so a newly declared fallback_prefix forces a new case here.
    assert set(dynamic_to_prefix.values()) == set(registry.fallback_prefixes)
    for symptom, prefix in dynamic_to_prefix.items():
        assert registry.resolve_symptom(symptom) == prefix, symptom


def test_ibv_modify_qp_eno_11_is_declared_and_unmapped_errnos_fall_back():
    registry = _load_committed_registry()

    # eno 11 is a declared symptom in its own right.
    eno_11 = "Call to ibv_modify_qp failed: eno 11 resource unavailable"
    assert registry.resolve_symptom(eno_11) == eno_11

    # Errnos without a dedicated declaration resolve to the generic prefix.
    # 9999 is not a real errno, so this case can never gain its own declaration.
    assert registry.resolve_symptom("Call to ibv_modify_qp failed: eno 9999") == "Call to ibv_modify_qp failed"

    # Outcome routing picks the specific eno 11 declaration over the generic.
    match = registry.outcome_for([eno_11])
    assert match is not None
    assert match.matched_prefix == eno_11
    assert match.outcome == "Fail InfraSW"

    # Outcome routing is startswith-based in declaration order, so the eno 11
    # declaration must not shadow the eno 110 variants.
    match_110 = registry.outcome_for(["Call to ibv_modify_qp failed: eno 110 timeout"])
    assert match_110 is not None
    assert match_110.matched_prefix == "Call to ibv_modify_qp failed: eno 110 timeout"


def test_generated_registry_facade_exposes_cascade_metadata():
    registry = _load_committed_registry()

    metadata = registry.metadata_for("Call to ibv_set_ece failed")

    assert metadata is not None
    assert "Connection reset by peer" in metadata.cascade_discards
    for strerror in ("Connection timed out", "Broken pipe"):
        assert registry.resolve_symptom(strerror) in metadata.cascade_discards
    assert registry.cascade_discards_for("Call to ibv_set_ece failed") == list(metadata.cascade_discards)


def test_domain_cascade_policy_is_generated_from_committed_declarations():
    registry = _load_committed_registry()
    expected_targets = {
        "XID-44-GR_FAULT_DURING_CTXSW",
        "XID-45-PREEMPTIVE_REMOVAL",
        "XID-145*",
        "NV_ERR_INVALID_STATE*",
        "cudaErrorLaunchFailure",
    }

    for primary in ("Node failure reported by slurm", "Fabric manager lost connection to all GPUs"):
        assert set(registry.cascade_discards_for(primary, scope=CascadeScope.NVL_DOMAIN)) == expected_targets


def test_csp_category_is_generated_from_committed_declarations():
    registry = _load_committed_registry()

    csp_symptoms = registry.symptoms_in_category(SymptomCategory.CSP_EVENT)

    assert "CSP Event GPU Fault" in csp_symptoms
    assert "CSP Event Classification Pending" in csp_symptoms
    assert "running without error" not in csp_symptoms


def test_batch_size_assertion_is_registered_above_generic_nvrx_failure():
    registry = _load_committed_registry()
    symptom = "Python Exception: AssertionError: batch size not divisible"

    assert registry.resolve_symptom(symptom) == symptom
    match = registry.outcome_for(["NVRx detected failure", "Python Exception", symptom])

    assert match is not None
    assert match.symptom == symptom
    assert match.matched_prefix == symptom
    assert match.outcome == "Fail User"
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.LOW)
    assert match.doc == "python-exception"


def test_category_membership_deduplicates_scoped_declarations():
    registry = registry_from_declarations(
        [
            SymptomDeclaration(prefix="primary", category=SymptomCategory.CSP_EVENT),
            SymptomDeclaration(
                prefix="primary",
                category=SymptomCategory.CSP_EVENT,
                cascade_discards=("target",),
                scope=CascadeScope.NVL_DOMAIN,
            ),
        ]
    )

    assert registry.symptoms_in_category(SymptomCategory.CSP_EVENT) == frozenset({"primary"})


def test_registry_compiles_immutable_family_metadata():
    xid_schema = SymptomFamily(name="xid", property_types={"code": int})
    xid_149 = SymptomFamily(name="xid", properties={"code": 149})
    registry = registry_from_declarations(
        [
            SymptomDeclaration(prefix="XID-149", family=xid_149),
            SymptomDeclaration(prefix="XID-", fallback_prefix=True, family=xid_schema),
        ]
    )

    assert registry.family_for("XID-149") == xid_149
    assert registry.family_for("XID-43-SW_NOTIFY") == xid_schema
    assert registry.family_for("not declared") is None
    with pytest.raises(TypeError):
        registry.families["XID-31"] = SymptomFamily(name="xid", properties={"code": 31})


def test_registry_rejects_conflicting_families_for_one_prefix():
    declarations = [
        SymptomDeclaration(prefix="bus error", family=SymptomFamily(name="network")),
        SymptomDeclaration(prefix="bus error", family=SymptomFamily(name="pcie")),
    ]

    with pytest.raises(SymptomValidationError, match="Conflicting families for 'bus error'"):
        registry_from_declarations(declarations)


@pytest.mark.parametrize(
    ("declarations", "error"),
    [
        (
            [SymptomDeclaration(prefix="bad", family=cast(SymptomFamily, "not-a-family"))],
            "expected SymptomFamily, got str",
        ),
        (
            [
                SymptomDeclaration(prefix="XID-", family=SymptomFamily(name="xid", property_types={"code": int})),
                SymptomDeclaration(prefix="XID_", family=SymptomFamily(name="xid", property_types={"code": str})),
            ],
            "Conflicting schemas for family 'xid'",
        ),
        (
            [SymptomDeclaration(prefix="XID-149", family=SymptomFamily(name="xid", properties={"code": 149}))],
            "declares properties without a schema",
        ),
        (
            [
                SymptomDeclaration(prefix="XID-", family=SymptomFamily(name="xid", property_types={"code": int})),
                SymptomDeclaration(
                    prefix="XID-149",
                    family=SymptomFamily(name="xid", properties={"severity": "fatal"}),
                ),
            ],
            "Unknown property 'severity'",
        ),
        (
            [
                SymptomDeclaration(prefix="XID-", family=SymptomFamily(name="xid", property_types={"code": int})),
                SymptomDeclaration(prefix="XID-149", family=SymptomFamily(name="xid", properties={"code": "149"})),
            ],
            "Invalid value for family 'xid' property 'code'",
        ),
    ],
)
def test_registry_rejects_invalid_family_contracts(
    declarations: list[SymptomDeclaration],
    error: str,
):
    """Reject malformed family metadata before publishing the registry."""
    with pytest.raises(SymptomValidationError, match=error):
        registry_from_declarations(declarations)


def test_registry_builds_selectors_from_resolved_declarations():
    xid_schema = SymptomFamily(name="xid", property_types={"code": int})
    xid_149 = SymptomFamily(name="xid", properties={"code": 149})
    registry = registry_from_declarations(
        [
            SymptomDeclaration(prefix="bus error"),
            SymptomDeclaration(prefix="RuntimeError", fallback_prefix=True),
            SymptomDeclaration(prefix="XID-", fallback_prefix=True, family=xid_schema),
            SymptomDeclaration(prefix="XID-149", family=xid_149),
        ]
    )

    assert registry.selector_for("bus error") == SymptomSelector(symptom_id="bus error")
    runtime_error = BaseSymptom(symptom_id="RuntimeError: dynamic detail")
    assert registry.selector_for(runtime_error.symptom_id).member(runtime_error)
    assert registry.selector_for("XID-43-SW_NOTIFY") == SymptomSelector(family_name="xid")
    xid_149_selector = registry.selector_for("XID-149")
    assert xid_149_selector == SymptomSelector(family_name="xid", family_properties={"code": 149})
    assert xid_149_selector.member(BaseSymptom(symptom_id="raw", family=xid_149))


def test_registry_rejects_selector_lookup_for_unknown_symptom():
    registry = registry_from_declarations([])

    with pytest.raises(UnknownSymptomError, match="not a known symptom id"):
        registry.selector_for("not declared")


def test_scoped_cascade_lookup_rejects_unknown_symptom():
    registry = _load_committed_registry()

    with pytest.raises(UnknownSymptomError, match="known symptom"):
        registry.cascade_discards_for("not a registered symptom", scope=CascadeScope.NVL_DOMAIN)


def test_non_node_scope_rejects_conditional_discards():
    declarations = [
        SymptomDeclaration(
            prefix="primary",
            conditional_discards=(("target", "predicate"),),
            scope=CascadeScope.NVL_DOMAIN,
        )
    ]

    with pytest.raises(SymptomValidationError, match="Conditional discards are not supported"):
        registry_from_declarations(declarations)


def test_generated_registry_facade_selects_outcome_by_priority():
    registry = _load_committed_registry()

    match = registry.outcome_for(["cudaError", "XID-149"])

    assert match is not None
    assert match.symptom == "XID-149"
    assert match.matched_prefix == "XID-149"
    assert match.outcome == "Fail InfraHW"
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.CRITICAL)


def test_generated_outcome_for_ignores_plr_tx_prefix():
    registry = _load_committed_registry()

    assert registry.outcome_for(["plr_tx_bw_loss_high_5m"]) is None


def test_symptom_docs_from_declarations_includes_only_doc_bearing_outcomes():
    declarations = [
        # Doc-bearing outcome declaration is carried.
        SymptomDeclaration(
            prefix="XID-149",
            outcome="Fail InfraHW",
            priority=OutcomePriority.CRITICAL,
            doc="xid-149",
        ),
        # Cascade-only pruning edge: not a symptom, excluded even with a doc.
        SymptomDeclaration(prefix="noop-only", cascade_noop=True, doc="noop-doc"),
        # Fallback prefixes retain an explicitly declared doc.
        SymptomDeclaration(
            prefix="XID-",
            outcome="Fail InfraHW",
            priority=OutcomePriority.MEDIUM_BROAD,
            doc="xid",
            fallback_prefix=True,
        ),
        # No outcome: not a user-facing symptom, excluded.
        SymptomDeclaration(prefix="cascade-edge", doc="edge-doc", cascade_discards=("x",)),
        # Empty doc slug: excluded (an empty slug would produce a dead link).
        SymptomDeclaration(
            prefix="cudaError",
            outcome="Fail User",
            priority=OutcomePriority.LOW_GENERIC,
            doc="",
        ),
        # Whitespace-only doc slug: excluded.
        SymptomDeclaration(
            prefix="blank-doc",
            outcome="Fail InfraSW",
            priority=OutcomePriority.LOW,
            doc=" \t\n",
        ),
        # Missing doc slug: excluded.
        SymptomDeclaration(
            prefix="RuntimeError",
            outcome="Fail InfraSW",
            priority=OutcomePriority.LOW,
        ),
    ]

    docs = symptom_docs_from_declarations(declarations)

    assert docs == {"XID-149": "xid-149", "XID-": "xid"}


def test_outcome_for_carries_doc_for_exact_and_fallback_prefixes():
    declarations = [
        SymptomDeclaration(
            prefix="XID-149",
            outcome="Fail InfraHW",
            priority=OutcomePriority.CRITICAL,
            doc="xid-149",
            hw_cause="Known HW Cause - Node",
        ),
        SymptomDeclaration(
            prefix="XID-",
            outcome="Fail InfraHW",
            priority=OutcomePriority.MEDIUM_BROAD,
            doc="xid",
            hw_cause="Known HW Cause - Node",
            fallback_prefix=True,
        ),
    ]
    registry = registry_from_declarations(declarations)

    doc_bearing = registry.outcome_for(["XID-149"])
    assert isinstance(doc_bearing, GeneratedOutcomeMatch)
    assert doc_bearing.matched_prefix == "XID-149"
    assert doc_bearing.doc == "xid-149"

    # A dynamic id inherits the documentation declared by its fallback prefix.
    fallback_match = registry.outcome_for(["XID-777-unmapped-variant"])
    assert isinstance(fallback_match, GeneratedOutcomeMatch)
    assert fallback_match.matched_prefix == "XID-"
    assert fallback_match.doc == "xid"


def test_generated_registry_uses_immutable_containers():
    registry = _load_committed_registry()

    with pytest.raises(AttributeError):
        registry.registered_symptoms.add("new-prefix-")
    with pytest.raises(TypeError):
        registry.outcome_rules[0]["new-prefix"] = ("Fail User", "Not Applicable")
    with pytest.raises(TypeError):
        registry.metadata["new-prefix"] = GeneratedSymptomMetadata(id="new-prefix")
    with pytest.raises(AttributeError):
        registry.symptoms_in_category(SymptomCategory.CSP_EVENT).add("new-prefix")

    assert registry.resolve_symptom("XID-149.21-NETIR_LINK_DOWN-RECEIEVED_TS1-0") == "XID-"


def test_cascade_only_noop_declaration_is_valid_registry_metadata(tmp_path):
    defaults_file = tmp_path / "noop_defaults.py"
    defaults_file.write_text(
        """
from dataclasses import replace

from attribution_lib.declarative.context import _require_rule_loader
from attribution_lib.rule_types.registration import _get_caller_location
from attribution_lib.symptoms.declarations import SymptomDeclaration


def _register_prefix(declaration: SymptomDeclaration) -> None:
    loader = _require_rule_loader("declaring prefix symptom")
    loader.register_symptom_declaration(replace(declaration, location=_get_caller_location()))


_register_prefix(SymptomDeclaration(prefix="noop-only", cascade_noop=True))
""",
        encoding="utf-8",
    )
    loader = RuleLoader()
    loader.load(str(defaults_file))

    registry = registry_from_declarations(loader.symptom_declarations)

    assert registry.metadata_for("noop-only") is not None
    assert registry.cascade_discards_for("noop-only") == []


def test_active_runtime_registry_matches_committed_defaults():
    """The registry built by setup_rules() must match the committed declarations."""
    runtime_registry = attribution_lib.get_symptom_registry()
    committed_registry = _load_committed_registry()

    # The active runtime loads all bundled rule files; the committed defaults are
    # a subset, so every committed structure must be present at runtime.
    assert committed_registry.registered_symptoms - runtime_registry.registered_symptoms == set()
    assert set(committed_registry.fallback_prefixes) - set(runtime_registry.fallback_prefixes) == set()

    # Outcome routing is the highest-value thing to regression-guard: every
    # committed outcome rule must be present in the matching runtime tier, and
    # every committed cascade metadata entry must be present at runtime.
    assert len(committed_registry.outcome_rules) == len(runtime_registry.outcome_rules)
    for tier_index, committed_tier in enumerate(committed_registry.outcome_rules):
        runtime_tier = runtime_registry.outcome_rules[tier_index]
        assert committed_tier.items() <= runtime_tier.items()
    assert committed_registry.metadata.items() <= runtime_registry.metadata.items()
    for scope, committed_discards in committed_registry.scoped_cascade_discards.items():
        assert committed_discards.items() <= runtime_registry.scoped_cascade_discards[scope].items()
    for category, committed_symptoms in committed_registry.symptoms_by_category.items():
        assert committed_symptoms <= runtime_registry.symptoms_in_category(category)


def test_symptoms_package_exports_declaration_types():
    from attribution_lib.symptoms import (
        CascadeScope as ExportedCascadeScope,
        OutcomePriority as ExportedOutcomePriority,
        SymptomCategory as ExportedSymptomCategory,
        SymptomDeclaration as ExportedSymptomDeclaration,
    )

    assert ExportedCascadeScope is CascadeScope
    assert ExportedOutcomePriority is OutcomePriority
    assert ExportedSymptomCategory is SymptomCategory
    assert ExportedSymptomDeclaration is SymptomDeclaration


@pytest.mark.parametrize(
    ("secondary", "baseline_outcome"),
    [
        ("Call to ibv_modify_qp failed", "Fail InfraHW"),
        ("Call to ibv_modify_qp failed: eno 110 timeout", "Fail InfraHW"),
        ("Call to ibv_modify_qp failed: eno 22 invalid argument(Mismatched GID)", "Fail InfraHW"),
        ("Call to ibv_modify_qp failed: eno 22 invalid argument", "Fail InfraSW"),
        ("Call to ibv_modify_qp failed: eno 11 resource unavailable", "Fail InfraSW"),
        ("Call to ibv_modify_qp failed: eno 5 input/output error", "Fail InfraHW"),
        ("ibv_open_device failed", "Fail InfraHW"),
    ],
)
def test_bad_nic_state_supersedes_only_secondary_outcomes(secondary, baseline_outcome):
    registry = _load_committed_registry()
    baseline = registry.outcome_for([secondary])
    assert baseline is not None
    assert baseline.outcome == baseline_outcome
    for symptoms in ([secondary, "detected bad NIC state"], ["detected bad NIC state", secondary]):
        match = registry.outcome_for(symptoms)
        assert match is not None
        assert match.outcome == "Fail InfraSW"
        assert match.symptom == "detected bad NIC state"


@pytest.mark.parametrize(
    ("independent", "expected_outcome"),
    [
        ("Node unexpectedly rebooted", "Fail InfraHW"),
        ("Page Retirement", "Fail InfraHW"),
        ("RDMA fabric flap", "Fail InfraHW"),
        ("Call to ibv_modify_qp failed: eno 110 timeout(Overwrite NCCL parameters)", "Fail User"),
    ],
)
@pytest.mark.parametrize("secondary", [None, "Call to ibv_modify_qp failed", "ibv_open_device failed"])
def test_bad_nic_state_preserves_independent_outcomes(independent, expected_outcome, secondary):
    symptoms = ["detected bad NIC state", independent]
    if secondary is not None:
        symptoms.append(secondary)
    match = _load_committed_registry().outcome_for(symptoms)
    assert match is not None
    assert match.outcome == expected_outcome
    assert match.symptom == independent


def test_bad_nic_state_outcome_metadata():
    match = _load_committed_registry().outcome_for(["detected bad NIC state"])
    assert match is not None
    assert match.doc == "detected-bad-nic-state"
    assert match.hw_cause == "NIC firmware initialization failure"
@pytest.mark.parametrize("symptom", ["NV_ERR_RESET_REQUIRED", "NV_ERR_RESET_REQUIRED kernel_channel.c"])
def test_gpu_reset_outcome_retains_documentation(symptom: str) -> None:
    """Check that GPU-reset variants retain their declared documentation.

    Args:
        symptom: Exact or file-suffixed GPU-reset symptom.
    """
    registry = _load_committed_registry()
    match = registry.outcome_for([symptom])

    assert isinstance(match, GeneratedOutcomeMatch)
    assert match.matched_prefix == "NV_ERR_RESET_REQUIRED"
    assert match.doc == "gpu-reset-required"
