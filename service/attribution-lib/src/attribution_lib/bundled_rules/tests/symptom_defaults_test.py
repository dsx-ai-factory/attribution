# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bundled symptom defaults."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.core import get_bundled_rules_path
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import Format as ObsFormat, Id, Source as ObsSource, TimeInterval
from attribution_lib.resources import AllJobResources
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.sources import Source, SourceSet
from attribution_lib.symptoms import translate_and_prune_node_attributions
from attribution_lib.symptoms.declarations import DagDiscardBehavior
from attribution_lib.symptoms.registry_generation import registry_from_declarations


def test_row_remap_does_not_drive_outcome():
    """Keep Row Remap in the vocabulary without mapping it to a failure."""
    loader = RuleLoader()
    loader.load(str(get_bundled_rules_path() / "symptom_defaults.py"))
    registry = registry_from_declarations(loader.symptom_declarations)

    assert registry.contains("Row Remap")
    assert registry.outcome_for(["Row Remap"]) is None


@pytest.mark.parametrize(
    ("other_checks", "expected_remaining"),
    [
        ((), set()),
        (("NVRx node health check failed: example_check",), {"NVRx node health check failed: example_check"}),
    ],
)
def test_nvrx_remapped_rows_health_check_is_pruned_with_unhealthy_node(other_checks, expected_remaining):
    """Prune the pending row-remap health check and the unhealthy-node symptom it causes.

    Args:
        other_checks: Other failed health-check symptoms on the same node.
        expected_remaining: Symptom ids that must survive pruning.
    """
    start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    obs_ctx = ObservationContext(
        time_interval=TimeInterval(start, start),
        source=ObsSource.APPLOG,
        format=ObsFormat.ENTRY_TIMESERIES,
        timezone=ZoneInfo("UTC"),
        id=Id(999),
    )
    source = Source(
        source_context=obs_ctx, line_num=1, datum="test", time=start, resources=AllJobResources, rule="test"
    )
    job_attributions = JobNodeAttributions()
    for symptom_id in (
        "NVRx node health check failed: hw_drv_gpu_ecc_remapped_rows",
        "NVRx unhealthy node",
        *other_checks,
    ):
        job_attributions.add(["node01"], SourceSet(symptom_id).add(source))

    pruned, skipped = translate_and_prune_node_attributions(job_attributions, default_job_info())

    remaining_ids = {s.symptom_id for s in pruned["node01"].symptoms()} if "node01" in pruned else set()
    assert remaining_ids == expected_remaining
    assert skipped == set()


def test_corroborated_node_failure_has_dedicated_documentation():
    """Keep the independently emitted composite on its canonical page."""
    loader = RuleLoader()
    loader.load(str(get_bundled_rules_path() / "symptom_defaults.py"))
    registry = registry_from_declarations(loader.symptom_declarations)

    assert registry.symptom_docs["Corroborated node failure"] == "corroborated-node-failure"


def test_xid_family_represents_declared_identifier_shapes() -> None:
    """Represent broad, classic, and decoded NVLink XID identities."""
    loader = RuleLoader()
    loader.load(str(get_bundled_rules_path() / "symptom_defaults.py"))
    registry = registry_from_declarations(loader.symptom_declarations)

    assert registry.families["XID-"].property_types == {
        "number": int,
        "mnemonic": str,
        "subcode": str,
        "port_down_reason_code": int,
        "port_down_reason": str,
        "error_status": int,
        "poison_victim": str,
    }
    assert registry.families["XID-94-"].properties == {"number": 94}
    assert registry.families["XID-94-CONTAINED_ERROR"].properties == {
        "number": 94,
        "mnemonic": "CONTAINED_ERROR",
    }
    for poison_victim in ("CE", "GSP", "SM"):
        assert registry.families[f"XID-94-CONTAINED_ERROR-{poison_victim}"].properties == {
            "number": 94,
            "mnemonic": "CONTAINED_ERROR",
            "poison_victim": poison_victim,
        }
    assert registry.families["XID-140-"].properties == {"number": 140}
    assert registry.family_for("XID-140-UNRECOVERABLE_ECC_ERROR_ESCAPE").properties == {
        "number": 140,
        "mnemonic": "UNRECOVERABLE_ECC_ERROR_ESCAPE",
    }
    assert registry.families["XID-13-GR_EXCEPTION /"].properties == {
        "number": 13,
        "mnemonic": "GR_EXCEPTION",
    }
    assert registry.families["XID-145-RLW_SRC_TRACK-8"].properties == {
        "number": 145,
        "subcode": "RLW_SRC_TRACK",
        "error_status": 8,
    }
    assert registry.families["XID-145-RLW_RXPIPE-8"].properties == {
        "number": 145,
        "subcode": "RLW_RXPIPE",
        "error_status": 8,
    }
    assert registry.families["XID-137-NVLINK_PRIV_ERR"].properties == {
        "number": 137,
        "mnemonic": "NVLINK_PRIV_ERR",
    }

    for symptom, reason_value, subcode, reason in (
        ("XID-149.0-NETIR_INT-NO_LINK_DOWN-0", 0, "NETIR_INT", "NO_LINK_DOWN"),
        (
            "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0",
            10,
            "NETIR_LINK_DOWN",
            "PHY_LAYER_RETRANSMIT_TIMEOUT",
        ),
        ("XID-149.37-NETIR_LINK_DOWN-PEER_RESET_EVENT-0", 37, "NETIR_LINK_DOWN", "PEER_RESET_EVENT"),
        ("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", 4, "NETIR_LINK_DOWN", "ALIGNMENT_LOSS"),
    ):
        assert registry.families[symptom].properties == {
            "number": 149,
            "subcode": subcode,
            "port_down_reason_code": reason_value,
            "port_down_reason": reason,
            "error_status": 0,
        }

    assert registry.family_for("XID-150-SAW_MSEFW") == registry.families["XID-"]
    assert registry.family_for("FalsePositive-XID-64") is None
    assert registry.family_for("DriverBug-XID-64") is None


def test_cuda_illegal_address_resolves_its_own_guidance():
    """Preserve CUDA outcome selection and KEEP behavior, including mixed NCCL evidence."""
    loader = RuleLoader()
    loader.load(str(get_bundled_rules_path() / "symptom_defaults.py"))
    registry = registry_from_declarations(loader.symptom_declarations)

    assert registry.dag_discard_for("cudaErrorIllegalAddress") == DagDiscardBehavior.KEEP
    for symptoms, expected_symptom, expected_outcome, expected_doc in (
        (["cudaErrorIllegalAddress"], "cudaErrorIllegalAddress", "Fail User", "cudaerrorillegaladdress"),
        (["ncclUnhandledCudaError"], "ncclUnhandledCudaError", "Fail MultiCategory", "ncclunhandledcudaerror"),
        (
            ["cudaErrorIllegalAddress", "ncclUnhandledCudaError"],
            "cudaErrorIllegalAddress",
            "Fail User",
            "cudaerrorillegaladdress",
        ),
        (
            ["ncclUnhandledCudaError", "cudaErrorIllegalAddress"],
            "cudaErrorIllegalAddress",
            "Fail User",
            "cudaerrorillegaladdress",
        ),
    ):
        match = registry.outcome_for(symptoms)
        assert match is not None
        assert match.symptom == expected_symptom
        assert match.outcome == expected_outcome
        assert match.doc == expected_doc


def test_ext4_corruption_outranks_the_application_errors_it_causes():
    """Corrupted node-local files break the application; the kernel corruption is the root cause."""
    loader = RuleLoader()
    loader.load(str(get_bundled_rules_path() / "symptom_defaults.py"))
    registry = registry_from_declarations(loader.symptom_declarations)

    match = registry.outcome_for(
        ["cudaErrorInvalidPtx", "No such file or directory", "IO error", "ext4 filesystem corruption"]
    )
    assert match is not None
    assert match.symptom == "ext4 filesystem corruption"
    assert match.outcome == "Fail InfraSW"
    assert match.doc == "ext4-filesystem-corruption"
