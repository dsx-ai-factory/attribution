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
Epilog rules ported from attribution_lib/epilog.py and epilog_to_symptom.py.
"""

import re
from typing import Dict, List

from common.xid_errors import get_catalog
from loguru import logger as _logger

from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules_engine import (
    EventTriple,
    PreFilterSpec,
    RuleContext,
    declare_event_types,
    declare_rule_names,
    extractor_def,
    register_pre_filter,
)
from attribution_lib.sources import Source

rule_author = "FACT Team"
rule_sources = {ObsSource.HEALTH_CHECK}


def _epilog_job_id_filter(entry, job_info):
    """Skip epilog entries for other jobs. Dynamic check - can't be a keyword."""
    line = entry.body()
    if "epilog" in line and f"epilog_{job_info.id}" not in line:
        return True  # skip
    return False


register_pre_filter(
    ObsSource.HEALTH_CHECK,
    PreFilterSpec(
        exclude_terms=["[FACT]", "laso", "prolog"],
        callables=[_epilog_job_id_filter],
    ),
)

# DCA check_name to (symptom, is_bad_node)
_DCA_CHECK_NAME_RULES: dict[str, tuple[str, bool]] = {
    "hw_drv_gpu_nvlink_active": ("NVLink inactive", True),
    "hw_drv_gpu_nvlink_error_counters": ("NVLink error counters", True),
    "hw_drv_net_ib_devinfo_cx7": ("IB device settings mismatch (CX7)", True),
    "hw_drv_net_ib_fw_cx7": ("IB firmware outdated", False),
    "hw_sys_cpu_pci_bus_exist": ("PCIe device missing", True),
    "hw_sys_cpu_pci_bus_speed": ("PCIe bus degraded", False),
    "hw_sys_cpu_pci_bus_width": ("PCIe bus degraded", False),
    "sw_sys_config_services": ("Required service not running", False),
    "sw_sys_fs_lustre_network_state": ("Lustre network connectivity error", True),
    "sw_sys_logs_network_conn_error": ("Lustre network connectivity error", True),
}

# Shared derived rule name for all keyword-based epilog extractors below.
# These are loop-generated rules so declare_rule_names is the cleanest way.
declare_rule_names("Epilog keyword")
declare_event_types("Attribution.node.bad")


def _ecc_error(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """ECC Error rules must check that the error is uncorrected."""
    if "uncorrected=" not in source.datum and "uncorrected_total=" not in source.datum:
        return []
    return [_bad(node, "ECC Error", source.derived(rule="Epilog keyword"))]


# ECC Error rules
for _keyword in [
    "gpu_ecc_error: gpu",
    "gpu-ecc-errors GPU",
    "Some of the ECC errors found",
]:
    extractor_def(
        name=f"Epilog_{re.sub(r'[^a-zA-Z0-9]', '_', _keyword)}",
        match_re=re.escape(_keyword),
        processor=_ecc_error,
    )

# Row Remap rules
for _keyword in [
    "gpu_remapped_rows: pci",
    "hw_drv_gpu_ecc_remapped_rows: pci",
    "gpu_remapped_rows: gpu",
    "gpu-remapped-rows Remap",
    "Some remapped rows exist",
]:
    extractor_def(
        name=f"Epilog_{re.sub(r'[^a-zA-Z0-9]', '_', _keyword)}",
        match_re=re.escape(_keyword),
        processor=lambda groups, node, source, ctx: [_bad(node, "Row Remap", source.derived(rule="Epilog keyword"))],
    )

# Page Retirement rules
for _keyword in [
    "gpu_retired_pages: pci",
    "gpu_retired_pages: gpu",
    "[FAILED] Retired pages found",
]:
    extractor_def(
        name=f"Epilog_{re.sub(r'[^a-zA-Z0-9]', '_', _keyword)}",
        match_re=re.escape(_keyword),
        processor=lambda groups, node, source, ctx: [
            _bad(node, "Page Retirement", source.derived(rule="Epilog keyword"))
        ],
    )

# Rules with aliased symptoms
for _keyword, _symptom in [
    ("Unable to query nvidia-smi", "error running nvidia-smi"),
    ("Some Infiniband devices down", "Mellanox links are down"),
    ("Missing some Infiniband devices", "Mellanox links are down"),
    ("sel reports Power Supply", "Power Supply Unit failure"),
    ("Route Unhealthy: True", "Route Unhealthy"),
    ("gpu_idle: gpu", "gpu_idle"),
    ("Mellanox links are down", "Mellanox links are down"),
]:
    extractor_def(
        name=f"Epilog_{re.sub(r'[^a-zA-Z0-9]', '_', _keyword)}",
        match_re=re.escape(_keyword),
        processor=lambda groups, node, source, ctx, s=_symptom: [_bad(node, s, source.derived(rule="Epilog keyword"))],
    )

# Rules with empty alias — keyword becomes the symptom
for _keyword in [
    "error running nvidia-smi",
    "ERR! found in nvidia-smi",
    "Not responding",
    "Node unexpectedly rebooted",
    "ec2 instance retirement",
    "CMDaemon: not ready for scheduling",
]:
    extractor_def(
        name=f"Epilog_{re.sub(r'[^a-zA-Z0-9]', '_', _keyword)}",
        match_re=re.escape(_keyword),
        processor=lambda groups, node, source, ctx, s=_keyword: [_bad(node, s, source.derived(rule="Epilog keyword"))],
    )

# Register dynamic epilog rules for XID catalog entries, e.g.
# gpu_xid: gpu @9e XID 63=1 cont/corr gpu @9e XID 94=2 cont/corr

for id, item in get_catalog().items():
    xid_type = item["type"]
    nice_name = item["nice_name"]
    keyword = f" {xid_type} {id}="
    extractor_def(
        name=f"Epilog_XID_{xid_type}_{id}",
        match_re=re.escape(keyword),
        processor=lambda groups, node, source, ctx, s=nice_name: [_bad(node, s, source.derived(rule="Epilog keyword"))],
    )


@extractor_def(
    name="Epilog_DCA_HealthCheck",
    # Matches: [HC] ACTION (hash) check_name: details
    #          [HC] <JobID:NNN> ACTION check_name: details
    # Example: [HC] RUN_FIELD_DIAGS (abc123) hw_drv_gpu_nvlink_active: NVLink link 0 is inactive
    match_re=r"\[HC\]\s*(?:<JobID:\d+>\s*)?(?P<action>\w+)\s+(?:\(\w+\)\s*)?(?P<remainder>.*)",
    derived_rules=[f"Epilog_DCA_HealthCheck/{name}" for name in _DCA_CHECK_NAME_RULES],
)
def epilog_dca_health_check(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    """Parse structured DCA health-check lines and map known check_names to symptoms."""

    remainder = groups["remainder"]
    check_name = remainder.split(": ", 1)[0].strip() if ": " in remainder else remainder[:60].strip()

    entry = _DCA_CHECK_NAME_RULES.get(check_name)
    if entry is None:
        _logger.debug(
            f"DCA check_name not in catalog: {check_name!r} action={groups['action']} body={source.datum[:120]!r}"
        )
        return []

    symptom, is_bad = entry
    emit = _bad if is_bad else _interesting
    return [emit(node, symptom, source.derived(rule=f"Epilog_DCA_HealthCheck/{check_name}"))]
