# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative symptom defaults.

The runtime source of truth for symptom vocabulary, outcome priority tiers, and
cascade metadata. The generated symptom registry consumed at runtime is built
from these declarations (see ``registry_from_declarations``). Future edits should
move declarations next to the extractor rules that emit them. doc anchors are
derived placeholders pending the attribution documentation manifest.
"""

from attribution_lib.rules import SymptomFamily, cascade_symptom, known_symptom, outcome_symptom
from attribution_lib.symptoms.declarations import (
    CascadeScope,
    DagDiscardBehavior,
    OutcomePriority,
    SymptomCategory,
)


NVL_DOMAIN_CASCADE_TARGETS = (
    "XID-44-GR_FAULT_DURING_CTXSW",
    "XID-45-PREEMPTIVE_REMOVAL",
    "XID-145*",
    "NV_ERR_INVALID_STATE*",
    "cudaErrorLaunchFailure",
)

NODE_FAILURE_CASCADE_TARGETS = (
    "Node failure reported by slurm",
    "Node not responding",
    "Corroborated node failure",
)

XID_FAMILY_SCHEMA = SymptomFamily(
    name="xid",
    property_types={
        "number": int,
        "mnemonic": str,
        "subcode": str,
        "port_down_reason_code": int,
        "port_down_reason": str,
        "error_status": int,
        "poison_victim": str,
    },
)


def _xid_member(
    number: int,
    *,
    mnemonic: str | None = None,
    subcode: str | None = None,
    port_down_reason_code: int | None = None,
    port_down_reason: str | None = None,
    error_status: int | None = None,
    poison_victim: str | None = None,
) -> SymptomFamily:
    """Build an XID family member from explicitly decoded identity fields.

    Args:
        number: NVIDIA XID number.
        mnemonic: Optional catalog mnemonic for a classic XID.
        subcode: Optional decoded NVLink subcode.
        port_down_reason_code: Optional numeric NVLink port-down reason.
        port_down_reason: Optional symbolic NVLink port-down reason.
        error_status: Optional decoded NVLink error status.
        poison_victim: Optional XID 94 poison-victim component.

    Returns:
        Structured XID family identity containing the supplied fields.
    """
    properties: dict[str, object] = {"number": number}
    for name, value in (
        ("mnemonic", mnemonic),
        ("subcode", subcode),
        ("port_down_reason_code", port_down_reason_code),
        ("port_down_reason", port_down_reason),
        ("error_status", error_status),
        ("poison_victim", poison_victim),
    ):
        if value is not None:
            properties[name] = value
    return SymptomFamily(name="xid", properties=properties)


outcome_symptom(
    "unknown-symptom-",
    outcome="Fail Undetermined",
    priority=OutcomePriority.SENTINEL,
    doc="unknown-symptom",
)

outcome_symptom(
    "NVRx synthetic fault injection",
    outcome="Artificial",
    priority=OutcomePriority.CRITICAL,
    doc="nvrx-synthetic-fault-injection",
)

outcome_symptom(
    "XID-149",
    family=_xid_member(149),
    outcome="Fail InfraHW",
    priority=OutcomePriority.CRITICAL,
    doc="xid-149",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "CSP Event GPU Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-gpu-fault",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event NVSwitch Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-nvswitch-fault",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event NIC Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-nic-fault",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event NVLink Trunk Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-nvlink-trunk-fault",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Power Loss",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-power-loss",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event ARM System Error",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-arm-system-error",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Hardware Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-hardware-fault",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event CPU Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-cpu-fault",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Memory Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-memory-fault",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Storage Device Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-storage-device-fault",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event IPU Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-ipu-fault",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Urgent Repair",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-urgent-repair",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

# A tenant report is the initiating repair request: the tenant marks the host
# faulty and asks the CSP to remove/repair it. When a correlated provider
# termination is also present, this initiating cause is the primary symptom.
outcome_symptom(
    "CSP Event Tenant Reported As Faulty",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-tenant-reported-as-faulty",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

# Provider host-maintenance termination is the resulting CSP action. It is
# primary by itself, but supporting evidence when paired with a tenant report
# for the same job/node (resolved by CspEventPrecedenceAndCascades).
outcome_symptom(
    "CSP Event Terminated by CSP",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-terminated-by-csp",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

# Provider interruption-report-specific hardware faults (FAILURE_TOR_SWITCH /
# FAILURE_HOST_HARDWARE_TEMPERATURE / SILENT_DATA_CORRUPTION) with no prior CSP symptom.
outcome_symptom(
    "CSP Event ToR Switch Fault",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-tor-switch-fault",
    hw_cause="Known HW Cause - Fabric",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Overtemperature",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-overtemperature",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Silent Data Corruption",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-silent-data-corruption",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Emergency Maintenance",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-emergency-maintenance",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Guest OS Panic",
    outcome="Fail InfraSW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.CSP,
    doc="csp-event-guest-os-panic",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "running without error",
    outcome="Pass",
    priority=OutcomePriority.HIGH,
    doc="running-without-error",
)

outcome_symptom(
    "the process completes",
    outcome="Pass",
    priority=OutcomePriority.HIGH,
    doc="the-process-completes",
    cascade_discards=("No such file or directory", "Node not responding"),
)

outcome_symptom(
    "Node not responding",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH,
    doc="not-responding",
    hw_cause="Known HW Cause - Node",
    cascade_discards=("Stopping nvidia-imex.service", "Node unexpectedly rebooted"),
)

outcome_symptom(
    "Corroborated node failure",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="corroborated-node-failure",
    hw_cause="Known HW Cause - Node",
    cascade_discards=("Node failure reported by slurm", "Node not responding"),
)

outcome_symptom(
    "Disk quota exceeded",
    outcome="Fail User",
    priority=OutcomePriority.HIGH,
    doc="disk-quota-exceeded",
)

outcome_symptom(
    "assert number_image_tags <= number_of_images",
    outcome="Fail User",
    priority=OutcomePriority.HIGH,
    doc="assert-number-image-tags-number-of-images",
)

outcome_symptom(
    "wandb.errors.CommError: Run initialization has timed out",
    outcome="Fail User",
    priority=OutcomePriority.HIGH,
    doc="wandb-errors-commerror-run-initialization-has-timed-out",
)

outcome_symptom(
    "cudaErrorSystemDriverMismatch",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.HIGH,
    doc="cudaerrorsystemdrivermismatch",
    hw_cause="CUDA error 803: system has unsupported display driver / cuda driver combination",
)

outcome_symptom(
    "Unexpected error from cudaGetDeviceCount()",
    outcome="Fail InfraSW",
    priority=OutcomePriority.HIGH,
    doc="unexpected-error-from-cudagetdevicecount",
    hw_cause="driver config",
)

outcome_symptom(
    "network address cannot be retrieved",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH,
    doc="network-address-cannot-be-retrieved",
    hw_cause="Likely caused by Eth flap",
)

outcome_symptom(
    "OOM kill of system service",
    outcome="Fail InfraSW",
    priority=OutcomePriority.HIGH,
    doc="oom-kill-of-system-service",
    cascade_discards=(
        "Node not responding",
        "Node failure reported by slurm",
        "Corroborated node failure",
        "Node unexpectedly rebooted",
        "node offline",
    ),
)

outcome_symptom(
    "OOM kill of a system pod",
    outcome="Fail InfraSW",
    priority=OutcomePriority.HIGH,
    doc="oom-kill-of-a-system-pod",
    cascade_discards=(
        "Node not responding",
        "Node failure reported by slurm",
        "Corroborated node failure",
        "Node unexpectedly rebooted",
        "node offline",
        "OOM kill of user task",
    ),
)

outcome_symptom(
    "OOM kill of user task",
    outcome="Fail User",
    priority=OutcomePriority.HIGH,
    doc="oom-kill-of-user-task",
)

outcome_symptom(
    "System-out of memory",
    outcome="Fail User",
    priority=OutcomePriority.HIGH,
    doc="system-out-of-memory",
)

outcome_symptom(
    "GspRmAlloc failed",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH,
    doc="gsprmalloc-failed",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "Stopping cuda profiler",
    outcome="Pass",
    priority=OutcomePriority.HIGH,
    doc="stopping-cuda-profiler",
)

outcome_symptom(
    "NaN reported by Megatron",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.HIGH,
    doc="nan-reported-by-megatron",
    hw_cause="Could be Silent Data Corruption if high occurrence",
)

outcome_symptom(
    "LustreError",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH,
    doc="lustreerror",
)

outcome_symptom(
    "Lustre network connectivity error",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.HIGH,
    doc="lustre-network-connectivity-error",
)

outcome_symptom(
    "Stale file handle",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.HIGH,
    doc="stale-file-handle",
)

outcome_symptom(
    "NCCL IB device incompatibility",
    outcome="Fail User",
    priority=OutcomePriority.HIGH,
    doc="nccl-ib-device-incompatibility",
)

outcome_symptom(
    "NMI watchdog hard LOCKUP",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.HIGH,
    doc="nmi-watchdog-hard-lockup",
    cascade_discards=("Node failure reported by slurm", "Corroborated node failure"),
)

outcome_symptom(
    "Stopping nvidia-imex.service",
    outcome="Fail InfraSW",
    priority=OutcomePriority.HIGH,
    doc="stopping-nvidia-imex-service",
)

outcome_symptom(
    "XID-74-",
    family=_xid_member(74),
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="xid-74",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "XID-79-",
    family=_xid_member(79),
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="xid-79",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "ECC Error",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="ecc-error",
    hw_cause="Known HW Cause - Node",
    cascade_discards=(
        "error running nvidia-smi",
        "UserBuffer timeout",
        "cudaErrorECCUncorrectable",
        "ncclUnhandledCudaError",
    ),
)

outcome_symptom(
    "XID-171-",
    family=_xid_member(171),
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="xid-171",
    hw_cause="Known HW Cause - Node",
    cascade_discards=("XID-48-", "XID-63-", "XID-94-", "cudaErrorContained"),
)

outcome_symptom(
    "XID-172-",
    family=_xid_member(172),
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="xid-172",
    hw_cause="Known HW Cause - Node",
    cascade_discards=("XID-48-", "XID-94-", "cudaErrorContained"),
)

outcome_symptom(
    "XID-48-",
    family=_xid_member(48),
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="xid-48",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "XID-95-",
    family=_xid_member(95),
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="xid-95",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "XID-140-",
    family=_xid_member(140),
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="xid-140",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "poisoned gsp",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="ecc-error",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "XID-120-",
    family=_xid_member(120),
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="xid-120",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "link down(east-west)",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="link-down-east-west",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "node down reported by ufm",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="node-down-reported-by-ufm",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "IB Error-12 Vendor-129",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="ib-error-12-vendor-129",
    hw_cause="Known HW Cause - Fabric",
    cascade_discards=("IB Error-12 (IBV_WC_RETRY_EXC_ERR)", "IB Error-5 (IBV_WC_WR_FLUSH_ERR)"),
)

outcome_symptom(
    "IB Error-12",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="ib-error-12",
    hw_cause="Known HW Cause - Fabric",
    cascade_discards=("IB Error-5 (IBV_WC_WR_FLUSH_ERR)",),
)

outcome_symptom(
    "Power Supply Unit failure",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="power-supply-unit-failure",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "System is powering down",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW,
    doc="system-is-powering-down",
    cascade_discards=(
        "Node not responding",
        "Node unexpectedly rebooted",
        "Node failure reported by slurm",
        "Corroborated node failure",
        "node offline",
        "Fabric manager lost connection to all GPUs",
        "Stopping nvidia-imex.service",
    ),
)

outcome_symptom(
    "Node reboot initiated",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW,
    doc="node-unexpectedly-rebooted",
    cascade_discards=(
        "Node unexpectedly rebooted",
        "Node failure reported by slurm",
        "Node not responding",
        "Corroborated node failure",
        "node offline",
        "System is powering down",
        "Fabric manager lost connection to all GPUs",
        "Stopping nvidia-imex.service",
    ),
)

outcome_symptom(
    "Node failure reported by slurm",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM,
    doc="node-failure-reported-by-slurm",
    cascade_discards=("Stopping nvidia-imex.service",),
)

cascade_symptom(
    "Node failure reported by slurm",
    cascade_discards=NVL_DOMAIN_CASCADE_TARGETS,
    scope=CascadeScope.NVL_DOMAIN,
)

outcome_symptom(
    "IMEX lost connection",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="imex-lost-connection",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "node offline",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="node-offline",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "Node stopped by controller",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="node-stopped-by-controller",
    cascade_discards=(
        "Node failure reported by slurm",
        "Node not responding",
        "Corroborated node failure",
        "Node unexpectedly rebooted",
        "node offline",
        "System is powering down",
    ),
)

outcome_symptom(
    "PCIe bus degraded",
    outcome="Fail Undetermined",
    priority=OutcomePriority.HIGH_HW,
    doc="pcie-bus-degraded",
)

outcome_symptom(
    "PCIe device missing",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="pcie-device-missing",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "XID-145-InfraHW",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_XID145,
    doc="xid-145-infrahw",
)

outcome_symptom(
    "XID-145-InfraSW",
    outcome="Fail InfraSW",
    priority=OutcomePriority.HIGH_XID145,
    doc="xid-145-infrasw",
)

outcome_symptom(
    "XID-145-MultiCategory",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.HIGH_XID145,
    doc="xid-145-multicategory",
)

outcome_symptom(
    "XID-145-Undetermined",
    outcome="Fail Undetermined",
    priority=OutcomePriority.HIGH_XID145,
    doc="xid-145-undetermined",
)

outcome_symptom(
    "NVLink inactive",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="nvlink-inactive",
)

outcome_symptom(
    "NVLink error counters",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="nvlink-error-counters",
)

outcome_symptom(
    "IB device settings mismatch (CX7)",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="ib-device-settings-mismatch-cx7",
)

outcome_symptom(
    "IB device settings mismatch",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="ib-device-settings-mismatch",
)

outcome_symptom(
    "XID-13-",
    family=_xid_member(13),
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="xid-13",
)

outcome_symptom(
    "XID-31-",
    family=_xid_member(31),
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="xid-31",
)

outcome_symptom(
    "XID-32-",
    family=_xid_member(32),
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="xid-32",
)

outcome_symptom(
    "XID-38-",
    family=_xid_member(38),
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM,
    doc="xid-38",
)

outcome_symptom(
    "XID-92-",
    family=_xid_member(92),
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="xid-92",
)

outcome_symptom(
    "XID-94-",
    family=_xid_member(94),
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="xid-94",
)

outcome_symptom(
    "XID-109-",
    family=_xid_member(109),
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="xid-109",
)

outcome_symptom(
    "XID-137-",
    family=_xid_member(137),
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM,
    doc="xid-137",
)

outcome_symptom(
    "XID-43-",
    family=_xid_member(43),
    outcome="Fail User",
    priority=OutcomePriority.MEDIUM_XID,
    doc="xid-43",
)

outcome_symptom(
    "XID-44-",
    family=_xid_member(44),
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM_XID,
    doc="xid-44",
)

outcome_symptom(
    "XID-45-",
    family=_xid_member(45),
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM_XID,
    doc="xid-45",
)

outcome_symptom(
    "XID-",
    family=XID_FAMILY_SCHEMA,
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_BROAD,
    doc=None,
    hw_cause="Known HW Cause - Node",
    fallback_prefix=True,
    cascade_discards=(
        "UserBuffer timeout",
        "cudaErrorECCUncorrectable",
        "error running nvidia-smi",
        "ncclUnhandledCudaError",
        "Node failure reported by slurm",
        "Corroborated node failure",
    ),
)

# Resource exhaustion can come from infrastructure pressure or users requesting more resources than are available.
outcome_symptom(
    "NV_ERR_INSUFFICIENT_RESOURCES",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-insufficient-resources",
)

outcome_symptom(
    "NV_ERR_BUSY_RETRY",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-busy-retry",
    hw_cause="Driver bug",
)

outcome_symptom(
    "NV_ERR_NO_MEMORY mem_multicast_fabric.c",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-no-memory",
    hw_cause="Driver bug",
)

outcome_symptom(
    "NV_ERR_NO_MEMORY",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-no-memory",
)

outcome_symptom(
    "NV_ERR_FABRIC_STATE_OUT_OF_SYNC",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-fabric-state-out-of-sync",
    hw_cause="Driver bug",
)

# The driver returns this after a prior GPU fault left the GPU awaiting reset, so
# later NVLink queries against that GPU fail as a consequence. Fallback prefix so
# file-suffixed variants (e.g. "NV_ERR_RESET_REQUIRED p2p_api.c") resolve here for cascades.
# Own tier so other NV_ERR_* codes from the same GPU cannot win the sorted tie in MEDIUM_SW.
outcome_symptom(
    "NV_ERR_RESET_REQUIRED",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_RESET_REQUIRED,
    doc="gpu-reset-required",
    hw_cause="Known HW Cause - Node",
    cascade_discards=("Failed to collect nvlink status info", "Failed to update Rx Detect Link mask!"),
    fallback_prefix=True,
)

outcome_symptom(
    "NV_ERR_BUFFER_TOO_SMALL",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-buffer-too-small",
    hw_cause="Driver bug",
)

outcome_symptom(
    "NV_ERR_INVALID_OBJECT",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-invalid-object",
    hw_cause="Driver bug",
)

outcome_symptom(
    "NV_ERR_TIMEOUT",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-timeout",
    hw_cause="Driver bug",
)

outcome_symptom(
    "NV_ERR_INVALID_STATE",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-invalid-state",
    hw_cause="Driver bug",
)

outcome_symptom(
    "NV_ERR_OBJECT_NOT_FOUND",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-object-not-found",
    hw_cause="Driver bug",
)

outcome_symptom(
    "NV_ERR_GENERIC",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-generic",
    hw_cause="Driver bug",
)

outcome_symptom(
    "NV_ERR_INVALID_ARGUMENT",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-invalid-argument",
    hw_cause="Driver bug",
)

outcome_symptom(
    "NV_ERR_OVERLAPPING_UVM_COMMIT",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-overlapping-uvm-commit",
    hw_cause="Driver bug",
)

outcome_symptom(
    "kernel invalid memory access",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="kernel-invalid-memory-access",
    # Same crash also prints the generic "Internal error: Oops"; keep this specific symptom.
    cascade_discards=(*NODE_FAILURE_CASCADE_TARGETS, "kernel oops"),
)

outcome_symptom(
    "kernel oops",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="kernel-oops",
    hw_cause="Kernel oops",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "kernel soft lockup",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="kernel-soft-lockup",
    hw_cause="Kernel soft lockup",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "kernel panic",
    outcome="Fail InfraSW",
    priority=OutcomePriority.HIGH_KERNEL_PANIC,
    doc="kernel-panic",
    hw_cause="Kernel panic",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

# Corrupted node-local files surface in the application as unrelated errors
# (e.g. JIT compilation failing on a truncated header); the kernel line is the root cause.
outcome_symptom(
    "ext4 filesystem corruption",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="ext4-filesystem-corruption",
    hw_cause="ext4 filesystem corruption",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "Failed to collect nvlink status info",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="failed-to-collect-nvlink-status-info",
    cascade_discards=(*NODE_FAILURE_CASCADE_TARGETS, "Node unexpectedly rebooted"),
)

outcome_symptom(
    "NV_ERR_FABRIC_MANAGER_NOT_PRESENT",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nv-err-fabric-manager-not-present",
    hw_cause="Fabric Manager Not Running",
    cascade_discards=(
        "Connection reset by peer",
        "Connection timed out",
        "Broken pipe",
        "Connection closed",
        "Call to ibv_modify_qp failed",
        "Call to ibv_set_ece failed",
    ),
)

outcome_symptom(
    "Silent Data Corruption",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="silent-data-corruption",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "error running nvidia-smi",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="error-running-nvidia-smi",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "RDMA fabric flap",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="rdma-fabric-flap",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "ibv_open_device failed",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="ibv-open-device-failed",
    hw_cause="Known HW Cause - Node",
)

known_symptom("Row Remap")

outcome_symptom(
    "Page Retirement",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="page-retirement",
    hw_cause="Known HW Cause - Node",
    cascade_discards=(
        "error running nvidia-smi",
        "UserBuffer timeout",
        "cudaErrorECCUncorrectable",
        "ncclUnhandledCudaError",
    ),
)

outcome_symptom(
    "ec2 instance retirement",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="ec2-instance-retirement",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "CSP Event Hypervisor Failure",
    outcome="Fail InfraHW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.MEDIUM_SW,
    doc="csp-event-hypervisor-failure",
    hw_cause="Known HW Cause - Node",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Software Fault",
    outcome="Fail InfraSW",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.MEDIUM_SW,
    doc="csp-event-software-fault",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "gpu unavailable",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="gpu-unavailable",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "Node unexpectedly rebooted",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="node-unexpectedly-rebooted",
    hw_cause="Known HW Cause - Node",
    cascade_discards=("Stopping nvidia-imex.service",),
)

outcome_symptom(
    "gpu_idle",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="gpu-idle",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "No route to host",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="no-route-to-host",
    hw_cause="Possibly IPoIB issue because N/S was not stable",
)

outcome_symptom(
    "Call to ibv_modify_qp failed: eno 110 timeout(Overwrite NCCL parameters)",
    outcome="Fail User",
    priority=OutcomePriority.MEDIUM_SW,
    doc="call-to-ibv-modify-qp-failed-eno-110-timeout-overwrite-nccl-parameters",
    hw_cause="User overwrite NCCL parameters",
)

outcome_symptom(
    "Call to ibv_modify_qp failed: eno 110 timeout",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="call-to-ibv-modify-qp-failed-eno-110-timeout",
    hw_cause="Intermittent Network Failure : Node Draining recommended",
)

outcome_symptom(
    "Call to ibv_modify_qp failed: eno 22 invalid argument(Mismatched GID)",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="call-to-ibv-modify-qp-failed-eno-22-invalid-argument-mismatched-gid",
    hw_cause="Known NCCL Cause - Check missing RoCE IP addresses or GIDs",
)

outcome_symptom(
    "Call to ibv_modify_qp failed: eno 22 invalid argument",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="call-to-ibv-modify-qp-failed-eno-22-invalid-argument",
    hw_cause="Likely Wrong GID Index picked, Remediation : FW fix needed or automatic GID index choosing with upgraded NCCL",
)

outcome_symptom(
    "Call to ibv_modify_qp failed: eno 11 resource unavailable",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="call-to-ibv-modify-qp-failed-eno-11-resource-unavailable",
    hw_cause="Resource temporarily unavailable (EAGAIN) on QP transition",
)

# Keep this generic declaration after the specific "eno N" variants: outcome
# routing matches by prefix in declaration order within a tier. fallback_prefix
# lets errnos without a dedicated declaration resolve here instead of
# surfacing as unknown-symptom.
outcome_symptom(
    "Call to ibv_modify_qp failed",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="call-to-ibv-modify-qp-failed",
    hw_cause="Known NCCL Cause - No retry on modify QP failure",
    fallback_prefix=True,
    cascade_discards=(
        "Connection reset by peer",
        "Connection timed out",
        "Broken pipe",
        "Connection closed",
        "Software caused connection abort",
    ),
)

outcome_symptom(
    "Call to ibv_set_ece failed",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="call-to-ibv-set-ece-failed",
    hw_cause="Known NCCL Cause - unstable feature or bug in a NCCL version prior 2.24",
    cascade_discards=(
        "Connection reset by peer",
        "Connection timed out",
        "Broken pipe",
        "Connection closed",
        "Software caused connection abort",
        "Call to ibv_modify_qp failed",
    ),
    conditional_discards=(("Call to ibv_modify_qp failed", "contains_ibv_modify_qp_failure"),),
)

outcome_symptom(
    "NCCL invalid NIC selection",
    outcome="Fail InfraSW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="nccl-invalid-nic-selection",
    hw_cause="Known NCCL Cause - Invalid NIC selection",
)

known_symptom("NCCL WARN Message truncated")

outcome_symptom(
    "EAP authentication failed",
    outcome="Fail InfraHW",
    priority=OutcomePriority.MEDIUM_SW,
    doc="eap-authentication-failed",
    hw_cause="Known Provider Network issue of Network Authentication Failure",
    conditional_discards=(("EAP authentication failed", "job_elapsed_below_eap_observation_window"),),
)

outcome_symptom(
    "Lustre Client Evicted",
    outcome="Fail Undetermined",
    priority=OutcomePriority.MEDIUM_SW,
    doc="lustre-client-evicted",
    cascade_discards=("LustreError",),
)

outcome_symptom(
    "client socket has timed out",
    outcome="Fail User",
    priority=OutcomePriority.MEDIUM_SW,
    doc="client-socket-has-timed-out",
    hw_cause="Possibly TCPStore scale related - especially on AWS",
)

outcome_symptom(
    "iSCSI detected conn error",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="iscsi-detected-conn-error",
)

outcome_symptom(
    "bad kernel stack frame pointer in nsys",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="bad-kernel-stack-frame-pointer-in-nsys",
    cascade_discards=(
        "NMI watchdog hard LOCKUP",
        "general protection fault",
        "Node not responding",
        "Node failure reported by slurm",
        "Corroborated node failure",
        "Node unexpectedly rebooted",
    ),
)

outcome_symptom(
    "bad kernel stack frame pointer in CudaGpuInfoDump",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="bad-kernel-stack-frame-pointer-in-cudagpuinfodump",
    cascade_discards=(
        "NMI watchdog hard LOCKUP",
        "general protection fault",
        "Node not responding",
        "Node failure reported by slurm",
        "Corroborated node failure",
        "Node unexpectedly rebooted",
    ),
)

outcome_symptom(
    "NUMA node memory exhaustion",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW,
    doc="numa-node-memory-exhaustion",
    cascade_discards=("No CUDA runtime is found",),
)

outcome_symptom(
    "nfs server failure",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW,
    doc="nfs-server-failure",
)

outcome_symptom(
    "bad kernel stack frame pointer",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="bad-kernel-stack-frame-pointer",
    fallback_prefix=True,
    cascade_discards=(
        "NMI watchdog hard LOCKUP",
        "general protection fault",
        "Node not responding",
        "Node failure reported by slurm",
        "Corroborated node failure",
        "Node unexpectedly rebooted",
    ),
)

outcome_symptom(
    "general protection fault",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="general-protection-fault",
    cascade_discards=("Node failure reported by slurm", "Corroborated node failure"),
)

outcome_symptom(
    "DriverBug-",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW,
    doc="driverbug",
)

outcome_symptom(
    "S3 Exception",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="s3-exception",
    cascade_discards=("DataLoader Exception",),
)

outcome_symptom(
    "Connection closed",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="connection-closed",
    hw_cause="Unknown HW Cause",
)

outcome_symptom(
    "CE deadlock DETECTED",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW,
    doc="ce-deadlock-detected",
)

outcome_symptom(
    "UserBuffer timeout",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="userbuffer-timeout",
)

outcome_symptom(
    "Segmentation fault",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="segmentation-fault",
)

outcome_symptom(
    "JSONDecodeError",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="jsondecodeerror",
)

outcome_symptom(
    "CMDaemon: not ready for scheduling",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW,
    doc="cmdaemon-not-ready-for-scheduling",
)

outcome_symptom(
    "failed to import docker image",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="failed-to-import-docker-image",
)

outcome_symptom(
    "megatron.core.dist_checkpointing.core.CheckpointingException",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="megatron-core-dist-checkpointing-core-checkpointingexception",
)

# A concrete Megatron configuration error outranks generic NVRx failure context.
# Keep configuration values in source evidence, not in the symptom label.
outcome_symptom(
    "Python Exception: AssertionError: batch size not divisible",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="python-exception",
)

outcome_symptom(
    "Missing key in checkpoint",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="missing-key-in-checkpoint",
)

outcome_symptom(
    "cudaErrorECCUncorrectable",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW,
    doc="cudaerroreccuncorrectable",
    hw_cause="Sticky ECC status in CUDA driver",
    dag_discard=DagDiscardBehavior.KEEP,
)

outcome_symptom(
    "cudaErrorUnknown",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="cudaerrorunknown",
)

outcome_symptom(
    "cudaErrorNvlinkUncorrectable",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="cudaerrornvlinkuncorrectable",
    hw_cause="NVLink5 uncorrectable error",
)

outcome_symptom(
    "cudaErrorLaunchFailure",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="cudaerrorlaunchfailure",
)

outcome_symptom(
    "cudaErrorInvalidValue",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW,
    doc="cudaerrorinvalidvalue",
)

outcome_symptom(
    "cudaError out of memory",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="cudaerror-out-of-memory",
    cascade_discards=("NUMA node memory exhaustion", "cudaErrorDevicesUnavailable"),
)

outcome_symptom(
    "ncclInternalError",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="ncclinternalerror",
)

outcome_symptom(
    "Address already in use",
    outcome="Fail User",
    priority=OutcomePriority.LOW,
    doc="address-already-in-use",
)

outcome_symptom(
    "FalsePositive-XID-64",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="falsepositive-xid-64",
    cascade_discards=("XID-64-INFOROM_DRAM_RETIREMENT_FAILURE", "XID-92-EXCESSIVE_SBE_INTERRUPTS"),
)

outcome_symptom(
    "GPU is lost reported by slurm",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="gpu-is-lost-reported-by-slurm",
    hw_cause="Known HW Cause - Node",
)

outcome_symptom(
    "IO error",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="io-error",
    cascade_noop=True,
)

outcome_symptom(
    "cudaErrorDevicesUnavailable",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="cudaerrordevicesunavailable",
)

outcome_symptom(
    "NV_ERR_",
    outcome="Fail Undetermined",
    priority=OutcomePriority.LOW_GENERIC,
    doc=None,
    hw_cause="Driver bug",
    fallback_prefix=True,
)

outcome_symptom(
    "imex-daemon-shutdown",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_GENERIC,
    doc="imex-daemon-shutdown",
    hw_cause="Imex bug",
)

outcome_symptom(
    "collect-nvlink-status-failure",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="collect-nvlink-status-failure",
    hw_cause="Driver bug",
)

outcome_symptom(
    "update-rx-detect-link-mask-failure",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="update-rx-detect-link-mask-failure",
    hw_cause="Driver bug",
)

outcome_symptom(
    "cudaErrorNotPermitted",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="cudaerrornotpermitted",
)

# cudaErrorNotReady (CUDA 600 'device not ready') and cudaErrorNotSupported
# (CUDA 801 'operation not supported') fire at NCCL transport init (NVLS/P2P)
# when the NVLink/MNNVL fabric is not yet ready. Without a specific rule they
# fall through to the generic "cudaError -> Fail User" default below, which
# mislabels an infra/device-readiness issue as user error. Classify as
# MultiCategory, consistent with the sibling cudaErrorDevicesUnavailable.
outcome_symptom(
    "cudaErrorNotReady",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="cudaerrornotready",
)

outcome_symptom(
    "cudaErrorNotSupported",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="cudaerrornotsupported",
)

outcome_symptom(
    "cudaErrorIllegalAddress",
    outcome="Fail User",
    priority=OutcomePriority.LOW_GENERIC,
    doc="cudaerrorillegaladdress",
    dag_discard=DagDiscardBehavior.KEEP,
)

outcome_symptom(
    "cudaError",
    outcome="Fail User",
    priority=OutcomePriority.LOW_GENERIC,
    doc=None,
    fallback_prefix=True,
)

outcome_symptom(
    "Unconvertible NCCL type",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_GENERIC,
    doc="unconvertible-nccl-type",
)

outcome_symptom(
    "NCCL WARN Cuda failure",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_GENERIC,
    doc="nccl-warn-cuda-failure",
)

outcome_symptom(
    "ncclUnhandledCudaError",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="ncclunhandledcudaerror",
)

outcome_symptom(
    "ncclSystemError",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="ncclsystemerror",
)

outcome_symptom(
    "ncclInvalidArgument",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="ncclinvalidargument",
)

outcome_symptom(
    "ncclInvalidUsage",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="ncclinvalidusage",
)

outcome_symptom(
    "ncclRemoteError",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="ncclremoteerror",
)

outcome_symptom(
    "ncclTimeout",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="nccltimeout",
)

outcome_symptom(
    "IB Error-",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_GENERIC,
    doc=None,
    fallback_prefix=True,
)

outcome_symptom(
    "EFA Error-",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc=None,
    fallback_prefix=True,
)

outcome_symptom(
    "Connection reset by peer",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="connection-reset-by-peer",
)

outcome_symptom(
    "Connection timed out",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="connection-timed-out",
)

outcome_symptom(
    "Broken pipe",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="broken-pipe",
)

outcome_symptom(
    "NCCL Late Arrival",
    outcome="Fail Undetermined",
    priority=OutcomePriority.LOW_GENERIC,
    doc="nccl-late-arrival",
)

outcome_symptom(
    "NVRx detected failure",
    outcome="Fail Undetermined",
    priority=OutcomePriority.LOW_GENERIC,
    doc="nvrx-detected-failure",
    hw_cause="Underlying failure reason unavailable from the collected evidence.",
)

outcome_symptom(
    "NVRx unhealthy node",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="nvrx-unhealthy-node",
)

# The check or action name follows these two prefixes, e.g.
# "NVRx node health check failed: example_check".
outcome_symptom(
    "NVRx node health check failed",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc=None,
    fallback_prefix=True,
)

outcome_symptom(
    "NVRx GPU recovery action",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc=None,
    fallback_prefix=True,
)

outcome_symptom(
    "NVRx step section timeout",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="nvrx-step-section-timeout",
)

outcome_symptom(
    "NVRx no progress across restarts",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="nvrx-no-progress-across-restarts",
)

outcome_symptom(
    "PyTorch NCCL watchdog",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_GENERIC,
    doc="pytorch-nccl-watchdog",
    cascade_noop=True,
)

outcome_symptom(
    "DataLoader Exception",
    outcome="Fail User",
    priority=OutcomePriority.LOW_GENERIC,
    doc="dataloader-exception",
    cascade_discards=("cudaErrorDevicesUnavailable",),
)

outcome_symptom(
    "local catastrophic error",
    outcome="Fail InfraHW",
    priority=OutcomePriority.LOW_GENERIC,
    doc="local-catastrophic-error",
)

outcome_symptom(
    "NS network connection lost",
    outcome="Fail InfraHW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="ns-network-connection-lost",
    hw_cause="possibly node crash or network issue",
    cascade_discards=(
        "Node failure reported by slurm",
        "Node not responding",
        "Corroborated node failure",
        "Lustre Client Evicted",
        "iSCSI detected conn error",
        "prometheus lost connection",
        "IO error",
    ),
)

outcome_symptom(
    "pyxis: enroot mount failure from system directory",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-enroot-mount-failure-from-system-directory",
)

outcome_symptom(
    "pyxis: nvidia hook failed",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-nvidia-hook-failed",
)

outcome_symptom(
    "pyxis: cgroup hook failed",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-cgroup-hook-failed",
)

outcome_symptom(
    "pyxis: nvidia-container-cli mount error",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-nvidia-container-cli-mount-error",
)

outcome_symptom(
    "pyxis: nvidia-container-cli init error",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-nvidia-container-cli-init-error",
)

outcome_symptom(
    "pyxis: nvidia-container-cli ldcache error",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-nvidia-container-cli-ldcache-error",
)

outcome_symptom(
    "pyxis: nvidia-container-cli detection error",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-nvidia-container-cli-detection-error",
)

outcome_symptom(
    "pyxis: nvidia-container-cli not installed",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-nvidia-container-cli-not-installed",
)

outcome_symptom(
    "pyxis: mellanox hook failed",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-mellanox-hook-failed",
)

outcome_symptom(
    "enroot: digest error",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="enroot-digest-error",
)

outcome_symptom(
    "enroot: curl empty reply from server",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="enroot-curl-empty-reply-from-server",
)

outcome_symptom(
    "enroot: JSON parse failed",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="enroot-json-parse-failed",
)

outcome_symptom(
    "enroot: rootfs lock contention",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="enroot-rootfs-lock-contention",
)

outcome_symptom(
    "enroot: not enough space (system path)",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="enroot-not-enough-space-system-path",
)

outcome_symptom(
    "pyxis: file already exists (ephemeral)",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-file-already-exists-ephemeral",
)

outcome_symptom(
    "pyxis: mkdir permission denied (system path)",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-mkdir-permission-denied-system-path",
)

outcome_symptom(
    "unsquashfs: dir_scan permission denied (system path)",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="unsquashfs-dir-scan-permission-denied-system-path",
)

outcome_symptom(
    "mksquashfs: disk full (system path)",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="mksquashfs-disk-full-system-path",
)

outcome_symptom(
    "unsquashfs: write failed (system path)",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="unsquashfs-write-failed-system-path",
)

outcome_symptom(
    "pyxis: failed to create namespace",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="pyxis-failed-to-create-namespace",
)

outcome_symptom(
    "enroot: failed to set capabilities",
    outcome="Fail InfraSW",
    priority=OutcomePriority.LOW_NETWORK,
    doc="enroot-failed-to-set-capabilities",
)

outcome_symptom(
    "prometheus lost connection",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_NETWORK,
    doc="prometheus-lost-connection",
    hw_cause="prometheus lost connection to node",
)

outcome_symptom(
    "RuntimeError: Numpy is not available",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="runtimeerror-numpy-is-not-available",
)

outcome_symptom(
    "Bad Token",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="bad-token",
    hw_cause="Bad input tokens causing numerical instability",
)

outcome_symptom(
    "pyxis: enroot mount failure from user directory",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-enroot-mount-failure-from-user-directory",
)

outcome_symptom(
    "pyxis: enroot mount path not found",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-enroot-mount-path-not-found",
)

outcome_symptom(
    "pyxis: cannot find container file",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-cannot-find-container-file",
)

outcome_symptom(
    "pyxis: nvidia-container-cli device error",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-nvidia-container-cli-device-error",
)

outcome_symptom(
    "pyxis: nvidia unknown driver capability",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-nvidia-unknown-driver-capability",
)

outcome_symptom(
    "pyxis: nvidia-container-cli requirement error",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-nvidia-container-cli-requirement-error",
)

outcome_symptom(
    "enroot: registry auth denied",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-registry-auth-denied",
)

outcome_symptom(
    "enroot: registry image not found",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-registry-image-not-found",
)

outcome_symptom(
    "enroot: invalid image reference",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-invalid-image-reference",
)

outcome_symptom(
    "enroot: missing credentials (docker hub fallback)",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-missing-credentials-docker-hub-fallback",
)

outcome_symptom(
    "enroot: not enough space (user path)",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-not-enough-space-user-path",
)

outcome_symptom(
    "pyxis: mkdir permission denied (user path)",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-mkdir-permission-denied-user-path",
)

outcome_symptom(
    "unsquashfs: dir_scan permission denied (user path)",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="unsquashfs-dir-scan-permission-denied-user-path",
)

outcome_symptom(
    "mksquashfs: disk full (user path)",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="mksquashfs-disk-full-user-path",
)

outcome_symptom(
    "unsquashfs: write failed (user path)",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="unsquashfs-write-failed-user-path",
)

outcome_symptom(
    "enroot: switchroot exec failed",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-switchroot-exec-failed",
)

outcome_symptom(
    "pyxis: couldn't chdir to container workdir",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-couldn-t-chdir-to-container-workdir",
)

outcome_symptom(
    "pyxis: --container-image specified multiple times",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-container-image-specified-multiple-times",
)

outcome_symptom(
    "pyxis: mount source invalid",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-mount-source-invalid",
)

outcome_symptom(
    "pyxis: conflicting options",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-conflicting-options",
)

outcome_symptom(
    "pyxis: container name already exists",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-container-name-already-exists",
)

outcome_symptom(
    "srun: invalid pyxis argument",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="srun-invalid-pyxis-argument",
)

outcome_symptom(
    "pyxis: couldn't start container",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-couldn-t-start-container",
)

outcome_symptom(
    "enroot: curl tunnel failure",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-curl-tunnel-failure",
)

outcome_symptom(
    "enroot: curl transfer timeout",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-curl-transfer-timeout",
)

outcome_symptom(
    "enroot: DNS resolution failed",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-dns-resolution-failed",
)

outcome_symptom(
    "enroot: corrupted cache layer",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-corrupted-cache-layer",
)

outcome_symptom(
    "enroot: invalid image format",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="enroot-invalid-image-format",
)

outcome_symptom(
    "unsquashfs: file system corruption",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="unsquashfs-file-system-corruption",
)

outcome_symptom(
    "unsquashfs: killed by OOM",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="unsquashfs-killed-by-oom",
)

outcome_symptom(
    "pyxis: container start nonzero exit",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-container-start-nonzero-exit",
)

outcome_symptom(
    "pyxis: failed to create container filesystem",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-failed-to-create-container-filesystem",
)

outcome_symptom(
    "pyxis: cascade from peer task",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="pyxis-cascade-from-peer-task",
)

outcome_symptom(
    "nfs not responding",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="nfs-not-responding",
)

outcome_symptom(
    "pmix collective timeout",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="pmix-collective-timeout",
)

outcome_symptom(
    "bus error",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="bus-error",
)

outcome_symptom(
    "Cancelled by Timeout",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="cancelled-by-timeout",
    conditional_discards=(("IB Error-", "job_exceeded_time_limit"), ("EFA Error-", "job_exceeded_time_limit")),
)

outcome_symptom(
    "Cancelled due to task failure",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="cancelled-due-to-task-failure",
)

outcome_symptom(
    "No CUDA runtime is found",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="no-cuda-runtime-is-found",
    cascade_discards=("No such file or directory",),
)

outcome_symptom(
    "Job didn't start",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="job-didn-t-start",
)

outcome_symptom(
    "srun didn't happen",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="srun-didn-t-happen",
)

outcome_symptom(
    "terminate called",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="terminate-called",
    cascade_noop=True,
)

outcome_symptom(
    "fatal",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="fatal",
)

outcome_symptom(
    "No such file or directory",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="no-such-file-or-directory",
    cascade_noop=True,
)

outcome_symptom(
    "file write failed",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="file-write-failed",
)

outcome_symptom(
    "Python Exception",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="python-exception",
)

outcome_symptom(
    "wandb.Error",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="wandb-error",
)

outcome_symptom(
    "ValueError: Tasks not found",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="valueerror-tasks-not-found",
)

outcome_symptom(
    "EOFError: Ran out of input",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="eoferror-ran-out-of-input",
)

outcome_symptom(
    "http.client.RemoteDisconnected",
    outcome="Fail User",
    priority=OutcomePriority.LOW_USER,
    doc="http-client-remotedisconnected",
)

outcome_symptom(
    "Gloo connectFullMesh failed",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.LOW_USER,
    doc="gloo-connectfullmesh-failed",
)

outcome_symptom(
    "error: pyxis",
    outcome="Fail User",
    priority=OutcomePriority.INCOMPLETE,
    doc="error-pyxis",
)

outcome_symptom(
    "Dataset cache write failure",
    outcome="Fail User",
    priority=OutcomePriority.INCOMPLETE,
    doc="dataset-cache-write-failure",
)

outcome_symptom(
    "Cancelled by PyTorch NCCL watchdog",
    outcome="Fail MultiCategory",
    priority=OutcomePriority.INCOMPLETE,
    doc="cancelled-by-pytorch-nccl-watchdog",
)

outcome_symptom(
    "Cancelled by site automation",
    outcome="Fail User",
    priority=OutcomePriority.INCOMPLETE,
    doc="cancelled-by-site-automation",
)

outcome_symptom(
    "log unrecognized",
    outcome="Incomplete Info",
    priority=OutcomePriority.INCOMPLETE,
    doc="log-unrecognized",
)

outcome_symptom(
    "log unrecognized from interactive job",
    outcome="Incomplete Info",
    priority=OutcomePriority.INCOMPLETE,
    doc="log-unrecognized-from-interactive-job",
)

outcome_symptom(
    "log file not found",
    outcome="Incomplete Info",
    priority=OutcomePriority.INCOMPLETE,
    doc="log-file-not-found",
)

outcome_symptom(
    "log permission error",
    outcome="Incomplete Info",
    priority=OutcomePriority.INCOMPLETE,
    doc="log-permission-error",
)

outcome_symptom(
    "log file too large",
    outcome="Incomplete Info",
    priority=OutcomePriority.INCOMPLETE,
    doc="log-file-too-large",
)

outcome_symptom(
    "CSP Event Unrecognized Cause",
    outcome="Incomplete Info",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.FALLBACK,
    doc="csp-event-unrecognized-cause",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "CSP Event Classification Pending",
    outcome="Incomplete Info",
    category=SymptomCategory.CSP_EVENT,
    priority=OutcomePriority.FALLBACK,
    doc="csp-event-classification-pending",
    cascade_discards=NODE_FAILURE_CASCADE_TARGETS,
)

outcome_symptom(
    "exit code signal",
    outcome="Fail User",
    priority=OutcomePriority.FALLBACK,
    doc="exit-code-signal",
)

outcome_symptom(
    "Cancelled by cluster operators",
    outcome="Cancelled by cluster operators",
    priority=OutcomePriority.FALLBACK,
    doc="cancelled-by-cluster-operators",
)

outcome_symptom(
    "Cancelled by user",
    outcome="Cancelled by User",
    priority=OutcomePriority.FALLBACK,
    doc="cancelled-by-user",
)

outcome_symptom(
    "Cancelled by root",
    outcome="Fail User",
    priority=OutcomePriority.FALLBACK,
    doc="cancelled-by-root",
)

outcome_symptom(
    "job failed",
    outcome="Fail User",
    priority=OutcomePriority.FALLBACK,
    doc="job-failed",
)

outcome_symptom(
    "Job requeued",
    outcome="Incomplete Info",
    priority=OutcomePriority.FALLBACK,
    doc="job-requeued",
)

outcome_symptom(
    "sigkill_handler] Killing subprocess",
    outcome="Fail User",
    priority=OutcomePriority.FALLBACK,
    doc="sigkill-handler-killing-subprocess",
    cascade_discards=(
        "IB Error-12",
        "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
        "IB Error-12 Vendor-129",
        "IB Error-5 (IBV_WC_WR_FLUSH_ERR)",
    ),
)

outcome_symptom(
    "placeholder",
    outcome="Fail User",
    priority=OutcomePriority.FALLBACK,
    doc="placeholder",
)

cascade_symptom(
    "Mellanox links are down",
    cascade_discards=(
        "ncclRemoteError",
        "ncclTimeout",
        "ncclInternalError",
        "Connection reset by peer",
        "Connection timed out",
        "Broken pipe",
        "Connection closed",
        "Call to ibv_modify_qp failed",
        "Call to ibv_set_ece failed",
    ),
)

# dag_discard: how these symptoms project back when the DAG discards them
# (default DEMOTE). Formerly hardcoded in symptoms/event_to_symptom.py's projection.
cascade_symptom(
    "EAP authentication recovery",
    cascade_discards=("EAP authentication failed", "EAP authentication recovery"),
    dag_discard=DagDiscardBehavior.PURGE,
)

cascade_symptom(
    "XID-145",
    family=_xid_member(145),
    cascade_discards=(
        "cudaErrorLaunchFailure",
        "cudaErrorNvlinkUncorrectable",
        "XID-45-PREEMPTIVE_REMOVAL",
        "NV_ERR_INVALID_STATE",
    ),
)

cascade_symptom(
    "Route Unhealthy",
    cascade_discards=("XID-145-RLW_SRC_TRACK-4", "XID-145-RLW_SRC_TRACK-8"),
)

cascade_symptom(
    "DriverBug-XID-64",
    cascade_discards=(
        "FalsePositive-XID-64",
        "XID-64-INFOROM_DRAM_RETIREMENT_FAILURE",
        "XID-92-EXCESSIVE_SBE_INTERRUPTS",
    ),
)

cascade_symptom(
    "FalsePositive-XID-63",
    cascade_discards=("XID-63-INFOROM_DRAM_RETIREMENT_EVENT",),
)

cascade_symptom(
    "XID-63-INFOROM_DRAM_RETIREMENT_EVENT",
    family=_xid_member(63, mnemonic="INFOROM_DRAM_RETIREMENT_EVENT"),
    cascade_discards=("XID-63-INFOROM_DRAM_RETIREMENT_EVENT",),
)

cascade_symptom(
    "XID-31-MMU_ERR_FLT",
    family=_xid_member(31, mnemonic="MMU_ERR_FLT"),
    cascade_discards=("cudaErrorIllegalAddress",),
)

cascade_symptom(
    "XID-171-UNCORRECTABLE_DRAM_ERROR",
    family=_xid_member(171, mnemonic="UNCORRECTABLE_DRAM_ERROR"),
    cascade_discards=("XID-48-", "XID-63-", "XID-94-", "cudaErrorContained"),
)

cascade_symptom(
    "XID-172-UNCORRECTABLE_SRAM_ERROR",
    family=_xid_member(172, mnemonic="UNCORRECTABLE_SRAM_ERROR"),
    cascade_discards=("XID-48-", "XID-94-", "cudaErrorContained"),
)

cascade_symptom(
    "cudaErrorInvalidResourceHandle",
    cascade_discards=("Connection closed",),
)

cascade_symptom(
    "Python bug",
    cascade_discards=("IB Error-12", "IB Error-12 (IBV_WC_RETRY_EXC_ERR)", "IB Error-5 (IBV_WC_WR_FLUSH_ERR)"),
)

cascade_symptom(
    "non-fatal",
    cascade_discards=("non-fatal",),
    # Any raw symptom id containing "non-fatal" (e.g. "(non-fatal) LustreError")
    # resolves to this prefix. Formerly a hardcoded check in event_to_symptom.py.
    resolve_contains="non-fatal",
)

cascade_symptom(
    "after training is done",
    cascade_discards=("No such file or directory", "validation loss at iteration", "Node not responding"),
)

cascade_symptom(
    "validation loss at iteration",
    cascade_discards=("after training is done",),
)

known_symptom("BrokenPipeError")

known_symptom("EFA Error-107")

known_symptom("EFA Error-connection-reset")

known_symptom("Fabric manager lost connection to GPU")

# The fatal GSP marker needs XID-94 and NV_ERR_RESET_REQUIRED to diagnose ECC.
known_symptom("NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR")

outcome_symptom(
    "Fabric manager lost connection to all GPUs",
    outcome="Fail InfraHW",
    priority=OutcomePriority.HIGH_HW,
    doc="fabric-manager-lost-connection-to-all-gpus",
    hw_cause="Known HW Cause - Node",
    cascade_discards=(
        "Node failure reported by slurm",
        "Corroborated node failure",
        "Stopping nvidia-imex.service",
    ),
)

cascade_symptom(
    "Fabric manager lost connection to all GPUs",
    cascade_discards=NVL_DOMAIN_CASCADE_TARGETS,
    scope=CascadeScope.NVL_DOMAIN,
)

known_symptom("IB Error-12 (IBV_WC_RETRY_EXC_ERR)")

known_symptom("IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)")

known_symptom("IB Error-5 (IBV_WC_WR_FLUSH_ERR)")

known_symptom("IB firmware outdated")

known_symptom("NV_ERR_INVALID_STATE nv_gpu_ops.c")

known_symptom("NVRx GPU memory reclamation timeout")

known_symptom("OOM kill (other)")

known_symptom("Required service not running")

known_symptom("RuntimeError", fallback_prefix=True)

known_symptom("Software caused connection abort")

known_symptom("XID-109-CTXSW_TIMEOUT_ERROR", family=_xid_member(109, mnemonic="CTXSW_TIMEOUT_ERROR"))

known_symptom("XID-119-GSP_RPC_TIMEOUT", family=_xid_member(119, mnemonic="GSP_RPC_TIMEOUT"))

known_symptom("XID-120-GSP_ERROR", family=_xid_member(120, mnemonic="GSP_ERROR"))

known_symptom("XID-13-GR_EXCEPTION", family=_xid_member(13, mnemonic="GR_EXCEPTION"))

known_symptom("XID-13-GR_EXCEPTION /", family=_xid_member(13, mnemonic="GR_EXCEPTION"))

known_symptom(
    "XID-13-GR_EXCEPTION /\nGR_ERROR_SW_NOTIFY",
    family=_xid_member(13, mnemonic="GR_EXCEPTION"),
)

known_symptom("XID-137-NVLINK_PRIV_ERR", family=_xid_member(137, mnemonic="NVLINK_PRIV_ERR"))

known_symptom(
    "XID-140-UNRECOVERABLE_ECC_ERROR_ESCAPE",
    family=_xid_member(140, mnemonic="UNRECOVERABLE_ECC_ERROR_ESCAPE"),
)

known_symptom("XID-145-RLW_REMAP-4", family=_xid_member(145, subcode="RLW_REMAP", error_status=4))

known_symptom("XID-145-RLW_REMAP-8", family=_xid_member(145, subcode="RLW_REMAP", error_status=8))

known_symptom("XID-145-RLW_RXPIPE-8", family=_xid_member(145, subcode="RLW_RXPIPE", error_status=8))

known_symptom("XID-145-RLW_SRC_TRACK-4", family=_xid_member(145, subcode="RLW_SRC_TRACK", error_status=4))

known_symptom("XID-145-RLW_SRC_TRACK-8", family=_xid_member(145, subcode="RLW_SRC_TRACK", error_status=8))

known_symptom(
    "XID-149.0-NETIR_INT-NO_LINK_DOWN-0",
    family=_xid_member(
        149,
        subcode="NETIR_INT",
        port_down_reason_code=0,
        port_down_reason="NO_LINK_DOWN",
        error_status=0,
    ),
)

known_symptom(
    "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0",
    family=_xid_member(
        149,
        subcode="NETIR_LINK_DOWN",
        port_down_reason_code=10,
        port_down_reason="PHY_LAYER_RETRANSMIT_TIMEOUT",
        error_status=0,
    ),
)

known_symptom(
    "XID-149.37-NETIR_LINK_DOWN-PEER_RESET_EVENT-0",
    family=_xid_member(
        149,
        subcode="NETIR_LINK_DOWN",
        port_down_reason_code=37,
        port_down_reason="PEER_RESET_EVENT",
        error_status=0,
    ),
)

known_symptom(
    "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
    family=_xid_member(
        149,
        subcode="NETIR_LINK_DOWN",
        port_down_reason_code=4,
        port_down_reason="ALIGNMENT_LOSS",
        error_status=0,
    ),
)

known_symptom("XID-162-PSHC_REENGAGED", family=_xid_member(162, mnemonic="PSHC_REENGAGED"))

known_symptom("XID-163-PSHC_DISENGAGED", family=_xid_member(163, mnemonic="PSHC_DISENGAGED"))

known_symptom("XID-43-RESETCHANNEL_VERIF_ERROR", family=_xid_member(43, mnemonic="RESETCHANNEL_VERIF_ERROR"))

known_symptom("XID-44-GR_FAULT_DURING_CTXSW", family=_xid_member(44, mnemonic="GR_FAULT_DURING_CTXSW"))

known_symptom("XID-45-PREEMPTIVE_REMOVAL", family=_xid_member(45, mnemonic="PREEMPTIVE_REMOVAL"))

known_symptom("XID-48-GPU_ECC_DBE", family=_xid_member(48, mnemonic="GPU_ECC_DBE"))

known_symptom("XID-62-", family=_xid_member(62))

known_symptom("XID-62-PMU_HALT_ERROR", family=_xid_member(62, mnemonic="PMU_HALT_ERROR"))

known_symptom("XID-63-", family=_xid_member(63))

known_symptom(
    "XID-64-INFOROM_DRAM_RETIREMENT_FAILURE",
    family=_xid_member(64, mnemonic="INFOROM_DRAM_RETIREMENT_FAILURE"),
)

known_symptom("XID-74-NVLINK_ERROR", family=_xid_member(74, mnemonic="NVLINK_ERROR"))

known_symptom(
    "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS",
    family=_xid_member(79, mnemonic="GPU_HAS_FALLEN_OFF_THE_BUS"),
)

known_symptom("XID-92-EXCESSIVE_SBE_INTERRUPTS", family=_xid_member(92, mnemonic="EXCESSIVE_SBE_INTERRUPTS"))

known_symptom("XID-94-CONTAINED_ERROR", family=_xid_member(94, mnemonic="CONTAINED_ERROR"))

known_symptom(
    "XID-94-CONTAINED_ERROR-CE",
    family=_xid_member(94, mnemonic="CONTAINED_ERROR", poison_victim="CE"),
)

known_symptom(
    "XID-94-CONTAINED_ERROR-SM",
    family=_xid_member(94, mnemonic="CONTAINED_ERROR", poison_victim="SM"),
)

known_symptom(
    "XID-94-CONTAINED_ERROR-GSP",
    family=_xid_member(94, mnemonic="CONTAINED_ERROR", poison_victim="GSP"),
)

known_symptom("XID-95-UNCONTAINED_ERROR", family=_xid_member(95, mnemonic="UNCONTAINED_ERROR"))

known_symptom("alias else")

known_symptom("bad kernel stack frame pointer in python")

known_symptom("cudaErrorContained")

known_symptom("effective_ber_high")

known_symptom("hist15_high")

known_symptom("knvlinkUpdatePostRxDetectLinkMask_IMPL: Failed to update Rx Detect Link mask!")

known_symptom("link down(mlx5 port)")

known_symptom("link down(nvlink physical layer)")

known_symptom("link down(nvlink)")

known_symptom("performance_phy_effective_errors_high")

known_symptom("performance_port_xmit_wait_bw_loss_host_1m_0p5pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_1m_10pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_1m_1pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_1m_50pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_1m_5pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_1m_gt0")

known_symptom("performance_port_xmit_wait_bw_loss_host_5m_0p5pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_5m_10pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_5m_1pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_5m_50pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_5m_5pct")

known_symptom("performance_port_xmit_wait_bw_loss_host_5m_gt0")

known_symptom("plr_rx_bw_loss_high_10m_0p5pct")

known_symptom("plr_rx_bw_loss_high_1m_5pct")

known_symptom("plr_rx_bw_loss_high_5m_1pct")

known_symptom("plr_tx_bw_loss_high_10m_0p5pct")

known_symptom("plr_tx_bw_loss_high_1m_5pct")

known_symptom("plr_tx_bw_loss_high_5m_1pct")

known_symptom("pyxis: cancelled by SIGTERM")

known_symptom("mlx5 firmware in initialization state")
known_symptom("mlx5 device initialization failed")

outcome_symptom(
    "detected bad NIC state",
    outcome="Fail InfraSW",
    # Independent MEDIUM_SW faults retain precedence. Outcome selection
    # handles the QP and device-open failures superseded by this symptom.
    priority=OutcomePriority.LOW,
    doc="detected-bad-nic-state",
    hw_cause="NIC firmware initialization failure",
)


outcome_symptom(
    "irisc not responding",
    outcome="Fail InfraSW",
    # Independent MEDIUM_SW faults retain precedence; outcome selection
    # supersedes only the QP and device-open mappings for this diagnosis.
    priority=OutcomePriority.LOW,
    doc="irisc-not-responding",
    hw_cause="irisc not responding",
)
