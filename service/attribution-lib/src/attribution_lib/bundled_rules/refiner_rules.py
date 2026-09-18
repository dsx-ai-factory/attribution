# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


# Refiner rules run on materialized symptoms and my_bad_nodes (post–event-stream).
# Execution order is defined by _REFINER_ORDER; insert new rules into the list
# at the desired position. Priorities are computed automatically.

import re
from datetime import timedelta
from typing import Optional

from common.refiner_helpers import (
    _add_symptom_from_source,
    _get_earliest_watchdog_source,
)
from loguru import logger

from attribution_lib.events import BAD_PAIR_EVENT_TYPES, EventStream, sql_placeholders
from attribution_lib.failed_node_provenance import FailedNodeResolution
from common.megatron import get_pytorch_timeout_minutes
from attribution_lib.hooks import multihooks, singlehooks
from attribution_lib.job_info import NVRX_TAG, JobInfo
from attribution_lib.node_not_responding_provenance import NodeNotRespondingProvenance
from attribution_lib.bundled_rules.common.outcome_helpers import apply_user_cancellation_reclassification
from attribution_lib.outcome import OutcomeAttribution
from attribution_lib.rules import (
    FilterPriority,
    RefinerContext,
    declare_event_types,
    declare_rule_names,
    refiner_def,
)
from attribution_lib.bundled_rules.common.scrub_helpers import (
    clean_bad_nodes_of_cascaded_symptoms,
    filter_cascade_symptoms_symptomset,
    filter_xid_145_cascade_symptoms,
    is_non_node_attributable_symptom,
    remove_cascaded_network_symptoms,
)
from attribution_lib.sources import Source, SourceSet, SymptomSet
from attribution_lib.symptoms import (
    project_node_symptoms_to_bad_nodes,
    translate_and_prune_node_attributions,
)
from attribution_lib.symptoms.symptom_enums import RegisteredSymptoms
from attribution_lib.utils import normalize_pci_id

declare_event_types(
    *BAD_PAIR_EVENT_TYPES,
    "Application.megatron-lm.runtime_args",
    "Node failure reported by slurm",
)

# Shared derived rule names used in core.py/bundled_rules/ (library code outside rule files).
declare_rule_names(
    "raw keywords",
    "remap SingleGPU_XID->XID",
    NodeNotRespondingProvenance.SLURM_NODE_REASON.value,
    FailedNodeResolution.SLURM_FAILED_NODE_FIELD.value,
    FailedNodeResolution.SINGLE_ALLOCATED_NODE_INFERENCE.value,
)

_REFINER_BASE = FilterPriority.FINAL
_STEP = 10

# Per-node scrubbing phase — runs before aggregation into job-level symptoms.
# Customers can add their own scrubbers here; aggregation happens once after all finish.
_SCRUB_ORDER = [
    # Scheduler-state attributions run first: they contribute the slurm node-failure
    # and "Node not responding" symptoms that ScrubBadNodes reads (see _NODE_FAILURE_SYMPTOMS)
    # to scrub peer DataLoader attributions. They must land before ScrubBadNodes.
    "SchedulerNodeFailure",
    "SchedulerNotResponding",
    "ScrubBadNodes",
    # Correlate the GSP evidence per node before aggregation and XID cascades.
    "PoisonedGspComposite",
    # Combine only the scheduler failure and surviving independent epilog
    # non-response evidence after more specific per-node causes are scrubbed.
    "CorroboratedNodeFailure",
]

_SCRUB_BASE = FilterPriority.SCRUB
_REFINER_BASE = FilterPriority.FINAL

# Single source of truth for pre-outcome refiner execution order.
# To insert a new rule, add its name at the desired position in this list.
_REFINER_ORDER = [
    # Runs first: drop "(non-fatal)" informational symptoms from the job-level set
    # before any other pre-outcome refiner inspects it (they stay on my_bad_nodes).
    "SuppressNonFatalJobSymptoms",
    # Raw-keyword promotion rules (ported from calculate_symptoms)
    "MissingKeyInCheckpoint",
    "TrainingValidationCompletion",
    "PythonExceptionFromFile",
    "SigkillHandlerPromotion",
    "EmptySymptomsFallback",
    # Symptom cleanup and resolution rules
    "SlurmStateSymptoms",
    "ProcessCompletesCleanup",
    "CspEventPrecedenceAndCascades",
    "CancellationResolution",
    "SimpleSymptomCascades",
    "MulticastNvErrSuppression",
    "WatchdogTerminateResolution",
    "NodeOfflineComposite",
    "IBSymptomCleanup",
    # Run after IBSymptomCleanup so the IB hierarchy is already normalized.
    "KernelCrashIbCascadeCleanup",
    "WatchdogGpuCascadeCleanup",
    "CudaXidCascades",
    "ConnectionClosedCascade",
    "XidGenericCleanup",
    "Xid145RxpipePrivErrCascade",
    "SlurmTimeoutXid145Cascade",
    "NodeOfflineXid145Cascade",
    "PmixCleanup",
    "NSConnectionLost",
    "PythonBugXidCleanup",
    "NodeFailureNfsCleanup",
    "LastResortSymptoms",
    "JobFailedState",
    "FabricManagerCascade",
    "Xid171Cascade",
    "Xid172Cascade",
    "ExtractJobMetadataFallback",
    "_reserved_final_1",
    "Xid149AndNvlinkFiltering",
    "Gb200PriorityRootCause",
    "_reserved_final_2",
    # DAG cascade pruning operates on the refined set, not the original
    "DagCascadePruning",
    # Job-lifecycle fallbacks run dead last: for a job with no nodes or a RUNNING job
    # they supersede everything, so nothing may run after them. No-op for normal jobs.
    "JobStateFallback",
]

GRACEFUL_MULTICAST_SKIP_MESSAGE = "Gracefully skipping multicast initialization"
MULTICAST_RESOURCE_SYMPTOM = "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c"
_CSP_EVENT_PREFIX = "CSP Event "
_CSP_TENANT_REPORTED = "CSP Event Tenant Reported As Faulty"
_CSP_TERMINATED = "CSP Event Terminated by CSP"
_SYSTEM_POWERING_DOWN = "System is powering down"


_POST_OUTCOME_ORDER = [
    # Runs first, reading the pre-reclassification outcome: when no root cause was
    # found (Incomplete Info), demote the bad-node symptoms to interesting. Moved out
    # of core.attribute; must precede OutcomeReclassification to see the same outcome
    # the old inline check saw.
    "DemoteBadNodesOnIncompleteInfo",
    # Runs so the XID-145 dynamic sub-symptoms are gone before any other post-outcome
    # refiner inspects the set (matching the old inline cleanup that ran at the end of
    # calculate_outcome, before run_post_outcome_refiners).
    "CleanupXid145DynamicSymptoms",
    "PostProcessSymptoms",
    "FilterFatalXidNoise",
    "OutcomeReclassification",
    # Runs after OutcomeReclassification so it sees the final (reclassified) outcome.
    "SingleNodeInfraBackfill",
]

_POST_OUTCOME_BASE = FilterPriority.POST_OUTCOME

_PRIORITY_MAP: dict[str, int] = {}
_PRIORITY_MAP.update({name: _SCRUB_BASE + i * _STEP for i, name in enumerate(_SCRUB_ORDER)})
_PRIORITY_MAP.update({name: _REFINER_BASE + i * _STEP for i, name in enumerate(_REFINER_ORDER)})
_PRIORITY_MAP.update({name: _POST_OUTCOME_BASE + i * _STEP for i, name in enumerate(_POST_OUTCOME_ORDER)})


def _priority(name: str) -> int:
    """Compute priority from position in _REFINER_ORDER or _POST_OUTCOME_ORDER."""
    return _PRIORITY_MAP[name]


# Pyxis / enroot cascade symptom sets. Used in `simple_symptom_cascades`
# to drive the cascades; kept 1:1 with the emitters in applog_rules.py /
# symptom_enums.py — when a new pyxis/enroot rule is added, include it here.
_PYXIS_MOUNT_FAILURE_SYMPTOMS = frozenset(
    {
        "pyxis: cannot find container file",
        "pyxis: enroot mount failure from user directory",
        "pyxis: enroot mount failure from system directory",
        "pyxis: enroot mount path not found",
    }
)

_PYXIS_REGISTRY_SYMPTOMS = frozenset(
    {
        "enroot: registry auth denied",
        "enroot: registry image not found",
        "enroot: invalid image reference",
        "enroot: digest error",
        "enroot: curl tunnel failure",
        "enroot: missing credentials (docker hub fallback)",
        "enroot: corrupted cache layer",
        "enroot: curl transfer timeout",
        "enroot: curl empty reply from server",
        "enroot: DNS resolution failed",
        "enroot: JSON parse failed",
    }
)

_PYXIS_HOOK_AND_CLI_SYMPTOMS = frozenset(
    {
        "pyxis: nvidia hook failed",
        "pyxis: cgroup hook failed",
        "pyxis: nvidia-container-cli mount error",
        "pyxis: nvidia-container-cli init error",
        "pyxis: nvidia-container-cli device error",
        "pyxis: nvidia unknown driver capability",
        "pyxis: nvidia-container-cli requirement error",
        "pyxis: nvidia-container-cli ldcache error",
        "pyxis: nvidia-container-cli detection error",
        "pyxis: nvidia-container-cli not installed",
        "pyxis: mellanox hook failed",
    }
)

_PYXIS_FILESYSTEM_SYMPTOMS = frozenset(
    {
        "enroot: rootfs lock contention",
        "enroot: not enough space (user path)",
        "enroot: not enough space (system path)",
        "enroot: invalid image format",
        "pyxis: file already exists (ephemeral)",
        "pyxis: mkdir permission denied (user path)",
        "pyxis: mkdir permission denied (system path)",
        "unsquashfs: file system corruption",
        "unsquashfs: dir_scan permission denied (user path)",
        "unsquashfs: dir_scan permission denied (system path)",
        "unsquashfs: killed by OOM",
        "mksquashfs: disk full (system path)",
        "mksquashfs: disk full (user path)",
        "unsquashfs: write failed (user path)",
        "unsquashfs: write failed (system path)",
    }
)

_PYXIS_KERNEL_AND_DIRECT_SYMPTOMS = frozenset(
    {
        "pyxis: failed to create namespace",
        "enroot: failed to set capabilities",
        "enroot: switchroot exec failed",
        "pyxis: couldn't chdir to container workdir",
        "pyxis: --container-image specified multiple times",
        "pyxis: mount source invalid",
        "pyxis: conflicting options",
        "pyxis: container name already exists",
    }
)

# Union of all specific pyxis/enroot symptoms — anything matching this is a
# concrete root-cause and supersedes the generic wrappers.
_PYXIS_SPECIFIC_SYMPTOMS = (
    _PYXIS_MOUNT_FAILURE_SYMPTOMS
    | _PYXIS_REGISTRY_SYMPTOMS
    | _PYXIS_HOOK_AND_CLI_SYMPTOMS
    | _PYXIS_FILESYSTEM_SYMPTOMS
    | _PYXIS_KERNEL_AND_DIRECT_SYMPTOMS
)

# Specific root causes that, when present, mean the mount-failure symptoms
# are evidence-replay from pyxis's enroot-log dump rather than the killer.
_PYXIS_NON_MOUNT_SPECIFIC = _PYXIS_SPECIFIC_SYMPTOMS - _PYXIS_MOUNT_FAILURE_SYMPTOMS

# NVIDIA cli sub-cases that supersede the generic `pyxis: nvidia hook failed`
# umbrella — both user-caused (device/capability/requirement) and infra-caused
# (cli-binary-missing) variants.
_PYXIS_NVIDIA_SPECIFIC_CLI_SYMPTOMS = frozenset(
    {
        "pyxis: nvidia-container-cli device error",
        "pyxis: nvidia unknown driver capability",
        "pyxis: nvidia-container-cli requirement error",
        "pyxis: nvidia-container-cli not installed",
    }
)


# Symptoms that mean the node became unusable mid-job. Used by cascade-scrub
# rules whose root cause is "the node went away."
_CORROBORATED_NODE_FAILURE = "Corroborated node failure"
_NODE_FAILURE_SYMPTOMS = frozenset(
    {
        "Node failure reported by slurm",
        "Node not responding",
        _CORROBORATED_NODE_FAILURE,
        "Node unexpectedly rebooted",
    }
)


POTENTIAL_NS_CONNECTION_SYMPTOMS = [
    "Lustre Client Evicted",
    "iSCSI detected conn error",
    "prometheus lost connection",
    "Node failure reported by slurm",
]


# Constants for IBSymptomCleanup

_IB_GENERAL_ERRORS = frozenset(
    {
        "Mellanox links are down",
        "ncclRemoteError",
        "ncclTimeout",
        "ncclInternalError",
        "Connection reset by peer",
        "Connection closed",
        "PyTorch NCCL watchdog",
        "UserBuffer timeout",
        "Call to ibv_modify_qp failed",
        "Call to ibv_set_ece failed",
    }
)

_NETWORK_CASCADE_SYMPTOMS = frozenset(
    {
        "Connection reset by peer",
        "Connection closed",
        "PyTorch NCCL watchdog",
        "Software caused connection abort",
        "cudaErrorDevicesUnavailable",
    }
)

_WATCHDOG_SYMPTOMS = [
    "Cancelled by PyTorch NCCL watchdog",
    "Cancelled by site automation",
    "PyTorch NCCL watchdog",
]

_IB_ERROR_TRIGGERS = [
    "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
    "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)",
    "IB Error-5 (IBV_WC_WR_FLUSH_ERR)",
]

_NON_IB_ROOT_CAUSE_PATTERNS = [
    "%XID-%",
    "%ECC%",
    "%Row Remap%",
    "%Node not responding%",
    "%NaN reported by Megatron%",
    "%megatron.core.dist_checkpointing.core.CheckpointingException%",
    "%Page Retirement%",
    "%Segmentation fault%",
    "%Node unexpectedly rebooted%",
    "%Node reboot initiated%",
    "%Disk quota exceeded%",
    "%error running nvidia-smi%",
    "%CE deadlock DETECTED%",
    "%JSONDecodeError%",
    "%Node failure reported by slurm%",
    "%Corroborated node failure%",
    "%Power Supply Unit failure%",
    "%System is powering down%",
    "%EOFError: Ran out of input%",
    "%http.client.RemoteDisconnected%",
    "%DataLoader Exception%",
    "%S3 Exception%",
    "%cudaError out of memory%",
    "%System-out of memory%",
    "%LustreError%",
    "%assert number_image_tags <= number_of_images%",
    "%wandb.errors.CommError: Run initialization has timed out%",
    "%Stale file handle%",
    "%Python Exception%",
    "%NCCL invalid NIC selection%",
]

_IBV_MODIFY_QP_SYMPTOMS = [
    "Call to ibv_modify_qp failed",
    "Call to ibv_set_ece failed",
    "Call to ibv_modify_qp failed: eno 22 invalid argument",
    "Call to ibv_modify_qp failed: eno 110 timeout",
    "Call to ibv_modify_qp failed: eno 22 invalid argument(Mismatched GID)",
    "Call to ibv_modify_qp failed: eno 11 resource unavailable",
]

_CONNECTION_CLOSED_TRIGGERS = [
    "cudaErrorInvalidResourceHandle",
    "cudaErrorNvlinkUncorrectable",
    "XID-137-NVLINK_PRIV_ERR",
    "XID-145-RLW_SRC_TRACK-8",
    "XID-145-RLW_SRC_TRACK-4",
    "Node failure reported by slurm",
    "NaN reported by Megatron",
]

_XID145_RXPIPE_SYMPTOM = "XID-145-RLW_RXPIPE-8"
_XID145_MULTICATEGORY_SYMPTOM = "XID-145-MultiCategory"
# In the regression fixture, RXPIPE was 33.5 s before job_end and 62.7 s after
# the timeout marker. A 90 s bound adds about 27 s of scheduler
# and log-delivery headroom beyond the larger observed separation.
_SLURM_TIMEOUT_XID145_WINDOW = timedelta(seconds=90)
# Timeout and XID timestamps come from separate log streams. Allow 1 s for
# timestamp precision and clock skew, rather than making the full 90 s window
# symmetric and potentially suppressing a genuine pre-timeout XID.
_SLURM_TIMEOUT_XID145_CLOCK_SKEW = timedelta(seconds=1)
_XID145_RXPIPE_COMPANIONS = frozenset(
    {
        "XID-137-NVLINK_PRIV_ERR",
        "XID-94-CONTAINED_ERROR",
    }
)
# In the regression fixture, XID-94 preceded RXPIPE by 70 ms, and XID-137
# followed it by 20 us. A 1 s bound tolerates kernel-log scheduling and
# interleaving while restricting correlation to the immediate episode.
_XID145_RXPIPE_COMPANION_WINDOW = timedelta(seconds=1)
_XID_PCI_ID_RE = re.compile(r"NVRM:\s+Xid \(PCI:(?P<pci_id>[0-9a-fA-F.:]+)\):")
_IB_CLEANUP_SKIP_SYMPTOMS = frozenset({_XID145_RXPIPE_SYMPTOM, "XID-137-NVLINK_PRIV_ERR"})
_KERNEL_CRASH_SYMPTOMS = frozenset(
    {
        "kernel panic",
        "kernel oops",
        "kernel invalid memory access",
    }
)
_IB_ERROR_12_PREFIX = "IB Error-12"


# IBSymptomCleanup helpers


def _is_ib_error_12(symptom: str) -> bool:
    """Match Error-12 as a complete code, including its formatted variants."""
    return symptom == _IB_ERROR_12_PREFIX or symptom.startswith(f"{_IB_ERROR_12_PREFIX} ")


def _deactivate_bad_pairs(event_stream: EventStream, rule_name: str) -> None:
    """Deactivate all bad pair events."""
    type_ids = event_stream.mgr.types.get_ids(BAD_PAIR_EVENT_TYPES)
    if not type_ids:
        return
    rule_id = event_stream.mgr.rules.get_id(rule_name)
    type_ph = sql_placeholders(len(type_ids))
    event_stream.execute(
        f"UPDATE events SET status=0, source_rule=? WHERE context_id=? AND event_type IN ({type_ph}) AND status=1",
        [rule_id, event_stream.ctx_id] + type_ids,
    )


def _discard_ib_cascade(symptom_list, ib_only: bool = False) -> None:
    """Discard IB/EFA cascaded symptoms from a SymptomSet or NodeAttributionGroup."""
    ib_efa_prefixes = ["IB Error-", "EFA Error-"]
    for symptom in list(symptom_list.symptoms()):
        for prefix in ib_efa_prefixes:
            if prefix in symptom:
                symptom_list.discard(symptom)

    if ib_only:
        return

    for error in _IB_GENERAL_ERRORS:
        symptom_list.discard(error)


def _remove_network_cascade(symptoms, my_bad_nodes) -> None:
    """Discard cascaded network symptoms from both symptoms and per-node attributions."""
    for symptom in _NETWORK_CASCADE_SYMPTOMS:
        symptoms.discard(symptom)
    for _node, errors in my_bad_nodes.items():
        for symptom in _NETWORK_CASCADE_SYMPTOMS:
            errors.discard(symptom)


def _remove_connection_closed_cascade(symptoms, my_bad_nodes) -> None:
    """Discard Connection closed from both symptoms and per-node attributions."""
    symptoms.discard("Connection closed")
    if my_bad_nodes is None:
        return
    for _node, errors in my_bad_nodes.items():
        errors.discard("Connection closed")


def _handle_link_down_time_proximity(symptoms, my_bad_nodes, event_stream: EventStream) -> None:
    """Handle link down(east-west) without IB errors using watchdog time-proximity."""
    link_down = "link down(east-west)"

    watchdog_cancellation_timestamp = None
    for watchdog_symptom in _WATCHDOG_SYMPTOMS:
        if watchdog_symptom in symptoms:
            watchdog_source = symptoms[watchdog_symptom].one_source()
            watchdog_cancellation_timestamp = watchdog_source.time if watchdog_source else None
            break

    if watchdog_cancellation_timestamp:
        timeout_seconds = 600
        buffer_seconds = 60
        if (pytorch_timeout := get_pytorch_timeout_minutes(event_stream)) is not None:
            timeout_seconds = int(pytorch_timeout.value() * 60)
        threshold = timeout_seconds + buffer_seconds

        keep_link_down_in_symptoms = False
        for _node, node_group in my_bad_nodes.items():
            errors = node_group.attributions
            if link_down in errors:
                node_link_down_set = errors[link_down]
                to_discard = []
                for s in node_link_down_set:
                    if not s.time:
                        to_discard.append(s)
                    else:
                        time_diff = (watchdog_cancellation_timestamp - s.time).total_seconds()
                        if time_diff < 0 or time_diff > threshold:
                            to_discard.append(s)

                node_link_down_set.discard_selected(to_discard)

                if not node_link_down_set.has_bad():
                    errors.discard(link_down)
                else:
                    keep_link_down_in_symptoms = True

        if not keep_link_down_in_symptoms:
            symptoms.discard(link_down)
        else:
            _discard_ib_cascade(symptoms)
    else:
        symptoms.discard(link_down)
        for _node, node_group in my_bad_nodes.items():
            errors = node_group.attributions
            if link_down in errors:
                errors.discard(link_down)


@refiner_def(name="SchedulerNodeFailure", author="NVIDIA Attribution team", priority=_priority("SchedulerNodeFailure"))
def scheduler_node_failure_refiner(context: RefinerContext):
    """Attribute a slurm NODE_FAIL to the node the scheduler blamed.

    When slurm reports a job as NODE_FAIL it names the failed node (or, on a
    single-node job, the sole allocated node necessarily is the one that failed).
    Emits "Node failure reported by slurm", a member of _NODE_FAILURE_SYMPTOMS that
    ScrubBadNodes reads, so this runs earlier in the SCRUB phase.
    """
    job_info = context.job_info
    my_bad_nodes = context.my_bad_nodes
    if job_info is None or my_bad_nodes is None:
        return
    if job_info.state != "NODE_FAIL":
        return

    failed_node = job_info.failed_node
    resolution = FailedNodeResolution.SLURM_FAILED_NODE_FIELD
    if (failed_node is None or failed_node == "") and len(job_info.nodes) == 1:
        # slurmdb did not record which node failed; on a single-node job the
        # sole allocated node is necessarily the one that failed.
        failed_node = job_info.nodes[0]
        resolution = FailedNodeResolution.SINGLE_ALLOCATED_NODE_INFERENCE
    # explicit checks so an empty string can't attribute to a node-less key
    if failed_node is not None and failed_node != "":
        source = job_info.source(job_info.state + failed_node, resolution.value)
        attr = SourceSet.new_one("Node failure reported by slurm", source)
        my_bad_nodes.add([failed_node], attr)


@refiner_def(
    name="SchedulerNotResponding", author="NVIDIA Attribution team", priority=_priority("SchedulerNotResponding")
)
def scheduler_not_responding_refiner(context: RefinerContext):
    """Back an "IB Error" with a raw scheduler "Not responding" reason.

    "IB Error" can be caused by a node going unresponsive, but that message may be
    missing from telemetry; slurmdb's node reason fills the gap. Emits the canonical
    "Node not responding" symptom, a member of _NODE_FAILURE_SYMPTOMS read
    by ScrubBadNodes, so this runs earlier in the SCRUB phase.
    """
    job_info = context.job_info
    my_bad_nodes = context.my_bad_nodes
    if job_info is None or my_bad_nodes is None:
        return

    for node in list(my_bad_nodes.attributions.keys()):
        errors = my_bad_nodes.attributions[node]

        # A heuristic to check if "IB Error" is caused by a raw "Not responding"
        # reason, using pyslurm to fetch this data from slurmdb because the
        # message might be missing in the telemetry.
        node_has_ib_error = any("IB Error" in error for error in errors.symptoms())
        if node_has_ib_error is False:
            continue

        if (node_info := job_info.slurm_node_infos.get(node)) is not None:
            node_reason = node_info.get("reason")
            # TODO: support other scheduler node-reason checks
            if node_reason is not None and "Not responding" in node_reason:
                my_bad_nodes[node].add(
                    SourceSet.new_one(
                        "Node not responding",
                        job_info.source(None, NodeNotRespondingProvenance.SLURM_NODE_REASON.value),
                    )
                )


@refiner_def(name="ScrubBadNodes", author="NVIDIA Attribution team", priority=_priority("ScrubBadNodes"))
def scrub_bad_nodes_refiner(context: RefinerContext):
    """Per-node cascade cleanup: discard secondary symptoms when a root cause is present.

    Runs in the SCRUB phase before aggregation into job-level symptoms.
    Additional scrubbers can be added to _SCRUB_ORDER; aggregation happens
    once in core.py after all scrubbers finish.
    """
    my_bad_nodes = context.my_bad_nodes
    job_info = context.job_info

    # Cross-node precompute: a node going down mid-job crashes DataLoader workers
    # on its peers, so the dataloader attribution typically lands on a DIFFERENT
    # node than the one that failed. Scrub it everywhere when any node on the
    # job carries a node-failure symptom.
    node_failure_present = any(s in my_bad_nodes[n] for n in my_bad_nodes for s in _NODE_FAILURE_SYMPTOMS)

    for k in list(my_bad_nodes):
        if "Node not responding" in my_bad_nodes[k]:
            my_bad_nodes[k].discard("Node unexpectedly rebooted")

        if "System is powering down" in my_bad_nodes[k]:
            my_bad_nodes[k].discard("Node not responding")
            my_bad_nodes[k].discard("Node unexpectedly rebooted")
            my_bad_nodes[k].discard("Stopping nvidia-imex.service")

        if (
            any("XID-" in error for error in my_bad_nodes[k].symptoms())
            or "NMI watchdog hard LOCKUP" in my_bad_nodes[k]
            or "general protection fault" in my_bad_nodes[k]
            or "kernel invalid memory access" in my_bad_nodes[k]
            or "Failed to collect nvlink status info" in my_bad_nodes[k]
        ):
            my_bad_nodes[k].discard("Node not responding")
            my_bad_nodes[k].discard("Node unexpectedly rebooted")

        if any("bad kernel stack frame pointer" in error for error in my_bad_nodes[k].symptoms()):
            my_bad_nodes[k].discard("NMI watchdog hard LOCKUP")
            my_bad_nodes[k].discard("general protection fault")
            my_bad_nodes[k].discard("Node not responding")
            my_bad_nodes[k].discard("Node unexpectedly rebooted")

        # An OOM kill of a system service (slurmd, slurmstepd, munged, etc.) is
        # the root cause; the node going silent or SLURM declaring NODE_FAIL
        # are downstream consequences.
        if "OOM kill of system service" in my_bad_nodes[k]:
            my_bad_nodes[k].discard("Node not responding")
            my_bad_nodes[k].discard("Node failure reported by slurm")

        # See cross-node precompute above — discard from every node, not just
        # the one where dataloader crashed.
        if node_failure_present:
            my_bad_nodes[k].discard("DataLoader Exception")

        if "FalsePositive-XID-64" in my_bad_nodes[k] or "DriverBug-XID-64" in my_bad_nodes[k]:
            my_bad_nodes[k].discard("XID-64-INFOROM_DRAM_RETIREMENT_FAILURE")
            my_bad_nodes[k].discard("XID-92-EXCESSIVE_SBE_INTERRUPTS")
            my_bad_nodes[k].discard("PyTorch NCCL watchdog")

        if "DriverBug-XID-64" in my_bad_nodes[k]:
            my_bad_nodes[k].discard("FalsePositive-XID-64")

        if "FalsePositive-XID-63" in my_bad_nodes[k]:
            my_bad_nodes[k].discard("XID-63-INFOROM_DRAM_RETIREMENT_EVENT")

        # Both an authentication failure and a subsequent success means
        # normal behavior — not a job interruption event.
        if "EAP authentication recovery" in my_bad_nodes[k]:
            my_bad_nodes[k].discard("EAP authentication failed")
            my_bad_nodes.purge_from_node(k, "EAP authentication recovery")
            if k not in my_bad_nodes:
                continue

        if "Unexpected error from cudaGetDeviceCount()" in my_bad_nodes[k] and job_info.num_gpus == 0:
            my_bad_nodes[k].discard("Unexpected error from cudaGetDeviceCount()")

        # Error 803 is the specific cause behind the generic device-count failure.
        if "cudaErrorSystemDriverMismatch" in my_bad_nodes[k]:
            my_bad_nodes[k].discard("Unexpected error from cudaGetDeviceCount()")


@refiner_def(
    name="CorroboratedNodeFailure",
    author="NVIDIA Attribution team",
    priority=_priority("CorroboratedNodeFailure"),
)
def corroborated_node_failure_refiner(context: RefinerContext):
    """Combine independent scheduler and epilog failure evidence on one node.

    A SLURM ``NODE_FAIL`` alone is scheduler evidence and remains InfraSW. When
    the epilog health check independently reports that the same node did not
    respond, emit the stronger ``Corroborated node failure`` symptom. A
    scheduler-derived node reason is not independent evidence and must not
    trigger this composite.
    """
    my_bad_nodes = context.my_bad_nodes
    if my_bad_nodes is None:
        return

    slurm_failure = RegisteredSymptoms.NODE_FAILURE_REPORTED_BY_SLURM.value
    node_not_responding = RegisteredSymptoms.NODE_NOT_RESPONDING.value
    epilog_rule = NodeNotRespondingProvenance.EPILOG_HEALTH_CHECK.value

    for _node, node_group in my_bad_nodes.items():
        node_attributions = node_group.attributions
        if slurm_failure not in node_attributions or node_not_responding not in node_attributions:
            continue

        epilog_sources = [source for source in node_attributions[node_not_responding] if source.rule == epilog_rule]
        if not epilog_sources:
            continue

        combined_sources = SourceSet(_CORROBORATED_NODE_FAILURE)
        for source in node_attributions[slurm_failure]:
            combined_sources.add(source)
        for source in epilog_sources:
            combined_sources.add(source)
        node_group.add(combined_sources)


@refiner_def(
    name="PoisonedGspComposite",
    author="NVIDIA Attribution team",
    priority=_priority("PoisonedGspComposite"),
)
def poisoned_gsp_composite(context: RefinerContext):
    """Recognize ECC from fatal GSP, contained-error, and reset evidence on one node.

    Correlate active evidence within the node's job evidence window, without an
    additional time-separation limit between the three signals.
    """
    if context.my_bad_nodes is None:
        return

    fatal_error = "NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR"
    reset_required = "NV_ERR_RESET_REQUIRED"
    for _node, node_group in context.my_bad_nodes.items():
        node_symptoms = node_group.attributions
        if fatal_error not in node_symptoms:
            continue

        xid94 = [symptom for symptom in node_symptoms.symptoms() if symptom.startswith("XID-94-")]
        # SyslogNvErr appends a space and, when available, the originating file.
        resets = [
            symptom
            for symptom in node_symptoms.symptoms()
            if symptom == reset_required or symptom.startswith(reset_required + " ")
        ]
        if not xid94 or not resets:
            continue

        components = [fatal_error, *xid94, *resets]
        combined_sources = SourceSet("poisoned gsp")
        for symptom in components:
            for source in node_symptoms[symptom]:
                combined_sources.add(source)
        node_group.add(combined_sources)

        # Demote only the exact components used above, retaining their evidence.
        # Declarative cascade discards use substring matching, which would also
        # hide unrelated codes such as NV_ERR_RESET_REQUIRED_OTHER.
        for symptom in components:
            node_group.discard(symptom)


# Raw-keyword promotion rules


@refiner_def(name="JobStateFallback", author="NVIDIA Attribution team", priority=_priority("JobStateFallback"))
def job_state_fallback(context: RefinerContext):
    """Job-lifecycle fallbacks moved out of core.calculate_symptoms.

    A job with no nodes never started; a RUNNING job has no verdict yet. Both cases
    supersede everything else, so this runs LAST (nothing may add symptoms after it) and
    clears the aggregated symptom set before emitting its single finding. This preserves
    the old core early-returns, which short-circuited the whole pipeline (RUNNING
    intentionally suppresses in-progress per-node symptoms).
    """
    symptoms = context.symptoms
    job_info = context.job_info
    if symptoms is None or job_info is None:
        return
    if len(job_info.nodes) == 0:
        symptoms.discard_all()
        symptoms.add(job_info.source_value("Job didn't start", None, "none assigned"))
        return
    if job_info.state == "RUNNING":
        symptoms.discard_all()
        symptoms.add(job_info.source_value("running without error", None, "no symptoms; status says running"))


@refiner_def(
    name="SuppressNonFatalJobSymptoms",
    author="NVIDIA Attribution team",
    priority=_priority("SuppressNonFatalJobSymptoms"),
)
def suppress_non_fatal_job_symptoms(context: RefinerContext):
    """Keep "(non-fatal)" symptoms per-node but out of the job-level symptom set.

    Rules tag informational findings with a "(non-fatal)" prefix (see
    filter_rules.py). Those attributions stay on my_bad_nodes for per-node reporting
    but must not reach the job-level symptom set that drives outcome selection.
    Runs first in the pre-outcome phase, right after aggregation, so the job-level
    set matches what the core aggregation loop produced before this moved out of core.
    """
    symptoms = context.symptoms
    if symptoms is None:
        return
    for symptom in list(symptoms.symptoms()):
        if "non-fatal" in symptom:
            symptoms.discard(symptom)


@refiner_def(
    name="MissingKeyInCheckpoint", author="NVIDIA Attribution team", priority=_priority("MissingKeyInCheckpoint")
)
def missing_key_in_checkpoint(context: RefinerContext):
    """Promote 'Missing key in checkpoint' from raw_keywords to a symptom."""
    raw_keywords = context.raw_keywords
    symptoms = context.symptoms
    if raw_keywords is None or symptoms is None:
        return
    if "Missing key in checkpoint" in raw_keywords.symptoms():
        sources = raw_keywords["Missing key in checkpoint"]
        symptoms.add(SourceSet.new_one("Missing key in checkpoint", sources.one_source()))


@refiner_def(
    name="TrainingValidationCompletion",
    author="NVIDIA Attribution team",
    priority=_priority("TrainingValidationCompletion"),
)
def training_validation_completion(context: RefinerContext):
    """Combine training and validation completion keywords into 'the process completes'."""
    raw_keywords = context.raw_keywords
    symptoms = context.symptoms
    if raw_keywords is None or symptoms is None:
        return
    # after training is done: the training phase is complete
    # validation loss at iteration: the validation phase is complete
    # there are corner cases where the training phase is complete but the validation phase fails, so we check both
    if "after training is done" in raw_keywords and "validation loss at iteration" in raw_keywords:
        symptoms.discard("after training is done")
        symptoms.discard("validation loss at iteration")
        sources = raw_keywords["after training is done"]
        symptoms.add(SourceSet.new_one("the process completes", sources.one_source()))


@refiner_def(
    name="PythonExceptionFromFile", author="NVIDIA Attribution team", priority=_priority("PythonExceptionFromFile")
)
def python_exception_from_file(context: RefinerContext):
    """Promote File " raw keyword to Python Exception unless higher-priority symptoms exist.

    Covers exception patterns like:
        raise EOFError
        raise OSError
        raise StatisticsError
        raise CheckpointingException
        raise CommClosedError
        raise RuntimeError
        raise ValueError
    """
    raw_keywords = context.raw_keywords
    symptoms = context.symptoms
    if raw_keywords is None or symptoms is None:
        return
    if 'File "' in raw_keywords:
        if (
            "PyTorch NCCL watchdog" not in symptoms
            and "S3 Exception" not in symptoms
            and "Disk quota exceeded" not in symptoms
        ):
            sources = raw_keywords['File "']
            symptoms.add(SourceSet.new_one("Python Exception", sources.one_source()))


@refiner_def(
    name="SigkillHandlerPromotion", author="NVIDIA Attribution team", priority=_priority("SigkillHandlerPromotion")
)
def sigkill_handler_promotion(context: RefinerContext):
    """Promote sigkill_handler to a symptom and discard cascaded IB errors."""
    raw_keywords = context.raw_keywords
    symptoms = context.symptoms
    if raw_keywords is None or symptoms is None:
        return
    if "sigkill_handler] Killing subprocess" in raw_keywords:
        sources = raw_keywords["sigkill_handler] Killing subprocess"]
        symptoms.add(SourceSet.new_one("si] Killing subprocess", sources.one_source()))
        symptoms.discard("IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)")
        symptoms.discard("IB Error-5 (IBV_WC_WR_FLUSH_ERR)")
        symptoms.discard("IB Error-12 (IBV_WC_RETRY_EXC_ERR)")


@refiner_def(
    name="EmptySymptomsFallback", author="NVIDIA Attribution team", priority=_priority("EmptySymptomsFallback")
)
def empty_symptoms_fallback(context: RefinerContext):
    """If the outcome is still empty, try to explain why from raw_keywords."""
    raw_keywords = context.raw_keywords
    symptoms = context.symptoms
    job_info = context.job_info
    if raw_keywords is None or symptoms is None or job_info is None:
        return
    if not symptoms or (len(symptoms) == 1 and "Cancelled by user" in symptoms):
        if "log file is not recognized from slurm cmdline" in raw_keywords:
            sources = raw_keywords["log file is not recognized from slurm cmdline"]
            if job_info.type == "interactive":
                symptoms.add(SourceSet.new_one("interactive job", sources.one_source()))
            else:
                symptoms.add(SourceSet.new_one("log unrecognized", sources.one_source()))
        elif "FileNotFoundError" in raw_keywords:
            sources = raw_keywords["FileNotFoundError"]
            symptoms.add(SourceSet.new_one("log file not found", sources.one_source()))
        elif "PermissionError" in raw_keywords:
            sources = raw_keywords["PermissionError"]
            symptoms.add(SourceSet.new_one("log permission error", sources.one_source()))
        elif "log file too large" in raw_keywords:
            sources = raw_keywords["log file too large"]
            symptoms.add(SourceSet.new_one("log file too large", sources.one_source()))
        elif "Avg bus bandwidth" in raw_keywords:
            sources = raw_keywords["Avg bus bandwidth"]
            symptoms.add(SourceSet.new_one("the process completes", sources.one_source()))


# Symptom cleanup and resolution rules


def _is_root_cancellation(job_info: JobInfo) -> bool:
    """Whether a cancellation is a genuine human root scancel, not automation.

    slurmctld cancels a job's steps as uid 0 (root) when the job hits its time
    limit, so a raw ``CANCELLED by 0`` step state is ambiguous. Treat it as a
    root scancel only when the resolved canceller is root; when the job-level
    canceller is ``Timeout`` the cancellation is Slurm enforcing the limit and
    must flow to the timeout handling instead. Keys on the resolved
    ``cancelled_by`` identity so it stays correct once the projection normalizes
    the state to a bare ``CANCELLED``; the raw-string check remains a fallback
    for feeds that emit the decoration without resolving the canceller.
    """
    if job_info.cancelled_by is not None:
        return job_info.cancelled_by == "root"
    return job_info.state == "CANCELLED by 0"


@refiner_def(name="SlurmStateSymptoms", author="NVIDIA Attribution team", priority=_priority("SlurmStateSymptoms"))
def slurm_state_symptoms(context: RefinerContext):
    """Add symptoms based on SLURM job state (CANCELLED, REQUEUED)."""
    symptoms = context.symptoms
    job_info = context.job_info
    if symptoms is None or job_info is None:
        return

    if _is_root_cancellation(job_info):
        symptoms.add(job_info.source_value("Cancelled by root", job_info.state, "state cancelled by root"))
    elif job_info.state.startswith("CANCELLED"):
        if job_info.cancelled_by == "site-automation-account":
            symptoms.add(job_info.source_value("Cancelled by cluster operators", job_info.state, "state cancelled"))
        elif job_info.cancelled_by == "Timeout":
            symptoms.add(job_info.source_value("Cancelled by Timeout", job_info.state, "state cancelled by timeout"))
        else:
            symptoms.add(job_info.source_value("Cancelled by user", job_info.state, "state cancelled"))
    elif job_info.state == "REQUEUED":
        symptoms.add(job_info.source_value("Job requeued", job_info.state, "state requeued"))


@refiner_def(
    name="ProcessCompletesCleanup", author="NVIDIA Attribution team", priority=_priority("ProcessCompletesCleanup")
)
def process_completes_cleanup(context: RefinerContext):
    """When 'the process completes', discard cascaded symptoms that are noise.

    A completed workload disproves a tenant report that the host made progress
    impossible. Other CSP events remain authoritative and continue through the
    normal precedence and cascade rules.
    """
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None:
        return

    # TODO: revisit later
    if "the process completes" in symptoms:
        symptoms.discard("Node not responding")
        symptoms.discard("BrokenPipeError")
        symptoms.discard("No such file or directory")
        symptoms.discard(_CSP_TENANT_REPORTED)
        # Completion is a job-lifecycle signal and is stripped from node
        # attributions, so same-node DAG cascade pruning cannot demote this CSP
        # symptom there.
        if my_bad_nodes is not None:
            for _node, node_group in my_bad_nodes.items():
                node_group.discard(_CSP_TENANT_REPORTED)


@refiner_def(
    name="CspEventPrecedenceAndCascades",
    author="NVIDIA Attribution team",
    priority=_priority("CspEventPrecedenceAndCascades"),
)
def csp_event_precedence_and_cascades(context: RefinerContext):
    """Resolve CSP event precedence and demote symptoms caused by CSP action.

    ``Tenant Reported As Faulty`` records the initiating tenant repair request,
    while ``Terminated by CSP`` records the provider action that follows when
    both describe the same interruption. The initiating report is therefore the
    primary job and same-node symptom; the termination remains supporting
    evidence. A termination on a different node remains bad-node evidence.

    Any CSP event that remains after precedence explains generic shutdown and
    Pyxis-family symptoms, so those are demoted from both job and node causes.
    This rule intentionally runs after ``ProcessCompletesCleanup``: completion
    first removes only the tenant report, allowing a co-occurring provider
    termination to remain authoritative.

    Args:
        context: Refiner state containing job symptoms and node attributions.
    """
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None:
        return

    if _CSP_TENANT_REPORTED in symptoms and _CSP_TERMINATED in symptoms:
        symptoms.discard(_CSP_TERMINATED)
        if my_bad_nodes is not None:
            for _node, node_group in my_bad_nodes.items():
                if _CSP_TENANT_REPORTED in node_group and _CSP_TERMINATED in node_group:
                    node_group.discard(_CSP_TERMINATED)

    if not any(symptom.startswith(_CSP_EVENT_PREFIX) for symptom in symptoms.symptoms()):
        return

    cascade_targets = [
        symptom for symptom in symptoms.symptoms() if symptom == _SYSTEM_POWERING_DOWN or "pyxis" in symptom.casefold()
    ]
    symptoms.discard_seq(cascade_targets)

    if my_bad_nodes is not None:
        for _node, node_group in my_bad_nodes.items():
            node_cascade_targets = [
                symptom
                for symptom in node_group.symptoms()
                if symptom == _SYSTEM_POWERING_DOWN or "pyxis" in symptom.casefold()
            ]
            node_group.attributions.discard_seq(node_cascade_targets)


@singlehooks.CancelledByCleanup.set_default
def cancelled_by_cleanup_default(job_info: JobInfo, symptoms: SymptomSet[str]) -> bool:
    return False


@refiner_def(
    name="CancellationResolution", author="NVIDIA Attribution team", priority=_priority("CancellationResolution")
)
def cancellation_resolution(context: RefinerContext):
    """Resolve cancellation source: terminate called, cleanup hook, or timeout."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    job_info = context.job_info
    if symptoms is None or job_info is None:
        return

    if "terminate called" in symptoms:
        symptoms.discard("Cancelled by user")
    elif "Cancelled by root" in symptoms:
        symptoms.discard("Cancelled by user")
    elif "Cancelled by cluster operators" in symptoms:
        symptoms.discard("Cancelled by user")
    elif singlehooks.CancelledByCleanup(job_info, symptoms):
        symptoms.discard("Cancelled by user")
    elif job_info.state == "TIMEOUT":
        symptoms.add(job_info.source_value("Cancelled by Timeout", job_info.state, "Timeout"))

    if "Cancelled by Timeout" in symptoms:
        symptoms.discard("Cancelled by user")
        symptoms.discard("BrokenPipeError")

    if "Cancelled due to task failure" in symptoms:
        symptoms.discard("Cancelled by user")

    # Purge both "Cancelled by user" and the transient "Cancelled due to task failure"
    # from per-node attributions — both fire on the same SLURM log line
    if my_bad_nodes is not None and "Cancelled due to task failure" in symptoms:
        for node in my_bad_nodes:
            if "Cancelled by user" in my_bad_nodes[node].symptoms():
                my_bad_nodes[node].discard("Cancelled by user", purge=True)
            if "Cancelled due to task failure" in my_bad_nodes[node].symptoms():
                my_bad_nodes[node].discard("Cancelled due to task failure", purge=True)


def _same_xid_device(left: Source, right: Source) -> bool:
    """Correlate matching PCI devices, with node/time fallback only if both are absent."""
    left_match = _XID_PCI_ID_RE.search(left.datum)
    right_match = _XID_PCI_ID_RE.search(right.datum)
    if left_match is None and right_match is None:
        return True
    if left_match is None or right_match is None:
        return False
    return normalize_pci_id(left_match.group("pci_id")) == normalize_pci_id(right_match.group("pci_id"))


@refiner_def(
    name="Xid145RxpipePrivErrCascade",
    author="NVIDIA Attribution team",
    priority=_priority("Xid145RxpipePrivErrCascade"),
)
def xid145_rxpipe_priv_err_cascade(context: RefinerContext):
    """Demote XID-137 and XID-94 emitted by the same RXPIPE-8 episode.

    RXPIPE-8 can produce an adjacent NVLink privilege error and contained
    error on the same node. Keep those companion records as interesting
    evidence, while leaving unrelated occurrences on other nodes or outside
    the immediate kernel-log episode eligible for attribution.
    """
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None or my_bad_nodes is None:
        return

    demoted_by_symptom: dict[str, set[Source]] = {symptom: set() for symptom in _XID145_RXPIPE_COMPANIONS}

    for _node, node_group in my_bad_nodes.items():
        node_symptoms = node_group.attributions
        if _XID145_RXPIPE_SYMPTOM not in node_symptoms:
            continue

        rxpipe_sources = [source for source in node_symptoms[_XID145_RXPIPE_SYMPTOM] if source.time is not None]
        if len(rxpipe_sources) == 0:
            continue

        for companion in _XID145_RXPIPE_COMPANIONS:
            if companion not in node_symptoms:
                continue
            correlated_sources = {
                source
                for source in node_symptoms[companion]
                if source.time is not None
                and any(
                    abs(source.time - rxpipe_source.time) <= _XID145_RXPIPE_COMPANION_WINDOW
                    and _same_xid_device(source, rxpipe_source)
                    for rxpipe_source in rxpipe_sources
                )
            }
            demoted_by_symptom[companion].update(node_symptoms[companion].discard_selected(correlated_sources))

    # Job-level symptoms were aggregated before refiners ran. Mirror only the
    # exact per-node source demotions, unless that same source remains bad on
    # another node.
    for companion, demoted_sources in demoted_by_symptom.items():
        if len(demoted_sources) == 0 or companion not in symptoms:
            continue
        still_bad_sources = {
            source
            for _node, node_group in my_bad_nodes.items()
            if companion in node_group.attributions
            for source in node_group.attributions[companion]
        }
        symptoms[companion].discard_selected(demoted_sources.difference(still_bad_sources))


@refiner_def(
    name="SlurmTimeoutXid145Cascade",
    author="NVIDIA Attribution team",
    priority=_priority("SlurmTimeoutXid145Cascade"),
)
def slurm_timeout_xid145_cascade(context: RefinerContext):
    """Discard terminal RXPIPE-8 errors caused by Slurm time-limit teardown.

    Slurm terminates every task when an allocation exceeds its time limit. A
    peer task can then report RXPIPE-8 while its shared memory is being torn
    down. In that narrow case, both the detailed symptom and its temporary
    outcome-classification symptom are cascades rather than root causes.
    """
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    job_info = context.job_info

    if (
        symptoms is None
        or my_bad_nodes is None
        or job_info is None
        or job_info.job_end_time is None
        or "Cancelled by Timeout" not in symptoms
        or _XID145_RXPIPE_SYMPTOM not in symptoms
    ):
        return

    timeout_times = {source.time for source in symptoms["Cancelled by Timeout"] if source.time is not None}

    def is_timeout_teardown(source: Source) -> bool:
        if source.time is None:
            return False
        if abs(job_info.job_end_time - source.time) > _SLURM_TIMEOUT_XID145_WINDOW:
            return False
        return len(timeout_times) == 0 or any(
            timeout_time - _SLURM_TIMEOUT_XID145_CLOCK_SKEW
            <= source.time
            <= timeout_time + _SLURM_TIMEOUT_XID145_WINDOW
            for timeout_time in timeout_times
        )

    terminal_rxpipe_sources = {source for source in symptoms[_XID145_RXPIPE_SYMPTOM] if is_timeout_teardown(source)}
    if len(terminal_rxpipe_sources) == 0:
        return

    # Preserve the existing XID-145 cascade cleanup while RXPIPE is still bad;
    # post-outcome filtering cannot detect the trigger after it is demoted here.
    filter_xid_145_cascade_symptoms(symptoms, my_bad_nodes)
    symptoms[_XID145_RXPIPE_SYMPTOM].discard_selected(terminal_rxpipe_sources)

    if _XID145_MULTICATEGORY_SYMPTOM in symptoms:
        dynamic_sources = symptoms[_XID145_MULTICATEGORY_SYMPTOM]
        paired_sources = terminal_rxpipe_sources.intersection(dynamic_sources)
        dynamic_sources.discard_selected(paired_sources)

    for _node, node_group in my_bad_nodes.items():
        node_symptoms = node_group.attributions
        if _XID145_RXPIPE_SYMPTOM not in node_symptoms:
            continue
        node_terminal_sources = {
            source for source in node_symptoms[_XID145_RXPIPE_SYMPTOM] if is_timeout_teardown(source)
        }
        node_symptoms[_XID145_RXPIPE_SYMPTOM].discard_selected(node_terminal_sources)
        if _XID145_MULTICATEGORY_SYMPTOM in node_symptoms:
            dynamic_sources = node_symptoms[_XID145_MULTICATEGORY_SYMPTOM]
            paired_sources = node_terminal_sources.intersection(dynamic_sources)
            dynamic_sources.discard_selected(paired_sources)


@refiner_def(
    name="SimpleSymptomCascades", author="NVIDIA Attribution team", priority=_priority("SimpleSymptomCascades")
)
def simple_symptom_cascades(context: RefinerContext):
    """Handle simple symptom cascade/rename rules."""
    symptoms = context.symptoms
    if symptoms is None:
        return

    # ibv_set_ece supersedes ibv_modify_qp
    if "Call to ibv_set_ece failed" in symptoms:
        symptoms.discard("Call to ibv_modify_qp failed")

    # Rename pushrecv to UserBuffer timeout
    if "pushrecv" in symptoms:
        symptoms.add(SourceSet.new_one("UserBuffer timeout", symptoms["pushrecv"].one_source()))
        symptoms.discard("pushrecv")

    # S3 root cause supersedes DataLoader
    if "S3 Exception" in symptoms and "DataLoader Exception" in symptoms:
        symptoms.discard("DataLoader Exception")

    # Specific pyxis/enroot symptoms supersede generic wrappers. Symptom sets
    # are defined at module level — see _PYXIS_*_SYMPTOMS above.
    if any(_sym in symptoms for _sym in _PYXIS_SPECIFIC_SYMPTOMS):
        symptoms.discard("No such file or directory")
        symptoms.discard("error: pyxis")
        symptoms.discard("pyxis: couldn't start container")
        symptoms.discard("failed to import docker image")
        symptoms.discard("pyxis: failed to create container filesystem")
        symptoms.discard("pyxis: container start nonzero exit")

    # Specific NVIDIA cli root causes supersede the hook-failed umbrella.
    # Includes both user-caused (device/capability/requirement) and
    # infra-caused (cli-binary-missing) sub-cases — the more specific
    # diagnostic always wins regardless of outcome bucket.
    if any(_sym in symptoms for _sym in _PYXIS_NVIDIA_SPECIFIC_CLI_SYMPTOMS):
        symptoms.discard("pyxis: nvidia hook failed")

    # NVIDIA hook failed supersedes its evidence-level cli errors.
    if "pyxis: nvidia hook failed" in symptoms:
        symptoms.discard("pyxis: nvidia-container-cli mount error")
        symptoms.discard("pyxis: nvidia-container-cli init error")
        symptoms.discard("pyxis: nvidia-container-cli ldcache error")
        symptoms.discard("pyxis: nvidia-container-cli detection error")

    # Non-mount root causes supersede mount-failure symptoms. pyxis prints
    # the full enroot log on container-start failure, including every bind
    # mount enroot attempted. Many config-driven bind mounts are best-effort
    # (/var/run/nvhcd is missing on most hosts; mellanox' infiniband_mad
    # tmpfs only exists when the kernel module is loaded) — they fail
    # individually but don't kill the container. When any concrete non-mount
    # pyxis/enroot root-cause is present (hook failure, registry error,
    # namespace failure, etc.), the mount-failure symptoms are evidence
    # replay, not the killer. Subsumes the older mellanox-specific cascade.
    if any(_sym in symptoms for _sym in _PYXIS_NON_MOUNT_SPECIFIC):
        for _ms in _PYXIS_MOUNT_FAILURE_SYMPTOMS:
            symptoms.discard(_ms)

    # Registry/curl errors supersede the `failed to import docker image`
    # wrapper line that always fires alongside them.
    if any(_sym in symptoms for _sym in _PYXIS_REGISTRY_SYMPTOMS):
        symptoms.discard("failed to import docker image")
        symptoms.discard("pyxis: container start nonzero exit")

    # The missing-credentials fallback symptom is a more specific diagnostic
    # variant of `enroot: registry auth denied` (both emitted by the same
    # regex, on a docker-hub URL with a dotted hostname in the /v2/ path).
    # Discard the generic when the specific is present.
    if "enroot: missing credentials (docker hub fallback)" in symptoms:
        symptoms.discard("enroot: registry auth denied")

    # The peer-task cascade symptom is noise when any other concrete
    # pyxis/enroot symptom is present on the job.
    if "pyxis: cascade from peer task" in symptoms:
        if any(_sym in symptoms for _sym in _PYXIS_SPECIFIC_SYMPTOMS):
            symptoms.discard("pyxis: cascade from peer task")

    # Cancellation marker: when pyxis was killed mid-operation by an external
    # SIGTERM, the job was cancelled — not a pyxis failure. Suppress all
    # Fail-family pyxis wrappers so the job attributes via the cancellation
    # taxonomy (Cancelled by User / Timeout / Watchdog / site automation / ...).
    if "pyxis: cancelled by SIGTERM" in symptoms:
        symptoms.discard("failed to import docker image")
        symptoms.discard("pyxis: failed to create container filesystem")
        symptoms.discard("pyxis: couldn't start container")
        symptoms.discard("error: pyxis")
        symptoms.discard("pyxis: container start nonzero exit")

    # File/wandb/IO errors are root cause; NCCL watchdog is cascaded
    if any(
        _smpt in symptoms
        for _smpt in [
            "No such file or directory",
            "wandb.Error",
            "IO error",
        ]
    ):
        symptoms.discard("PyTorch NCCL watchdog")

    # NUMA memory exhaustion supersedes missing CUDA runtime
    if "NUMA node memory exhaustion" in symptoms:
        symptoms.discard("No CUDA runtime is found")


@refiner_def(
    name="MulticastNvErrSuppression", author="NVIDIA Attribution team", priority=_priority("MulticastNvErrSuppression")
)
def multicast_nv_err_suppression(context: RefinerContext):
    """Suppress multicast NV_ERR when applog reports graceful multicast skip."""
    symptoms = context.symptoms
    raw_keywords = context.raw_keywords
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None or raw_keywords is None:
        return

    if GRACEFUL_MULTICAST_SKIP_MESSAGE not in raw_keywords:
        return
    if MULTICAST_RESOURCE_SYMPTOM not in symptoms:
        return

    logger.info(f"Discarding syslog symptom due to applog multicast skip marker: {MULTICAST_RESOURCE_SYMPTOM}")
    symptoms.discard(MULTICAST_RESOURCE_SYMPTOM)
    if my_bad_nodes is not None:
        for _node, node_group in my_bad_nodes.items():
            node_group.attributions.discard(MULTICAST_RESOURCE_SYMPTOM)


@singlehooks.WatchdogCancellationOverride.set_default
def _watchdog_cancellation_override_default(job_info: JobInfo) -> Optional[tuple[str, str]]:
    """Return (symptom_name, rule_description) override for watchdog+terminate cancellations, or None for the default."""
    return None


@refiner_def(
    name="WatchdogTerminateResolution",
    author="NVIDIA Attribution team",
    priority=_priority("WatchdogTerminateResolution"),
    derived_rules=["Canceled by automation-watchdog/terminate", "Canceled by watchdog (not automation)"],
)
def watchdog_terminate_resolution(context: RefinerContext):
    """Resolve NCCL watchdog + terminate into site automation or watchdog cancellation."""
    symptoms = context.symptoms
    job_info = context.job_info
    if symptoms is None or job_info is None:
        return

    if "PyTorch NCCL watchdog" in symptoms and "terminate called" in symptoms:
        symptoms.discard("terminate called")
        earliest_watchdog_source = _get_earliest_watchdog_source(symptoms)
        override = singlehooks.WatchdogCancellationOverride(job_info)
        if override is not None:
            override_symptom, override_rule = override
            # Create new symptom with preserved timestamp from watchdog
            _add_symptom_from_source(
                symptoms,
                earliest_watchdog_source,
                job_info,
                override_symptom,
                override_rule,
            )
        else:
            symptoms.discard("PyTorch NCCL watchdog")
            # Create new symptom with preserved timestamp from watchdog
            _add_symptom_from_source(
                symptoms,
                earliest_watchdog_source,
                job_info,
                "Cancelled by PyTorch NCCL watchdog",
                "Canceled by watchdog",
            )

    if "Cancelled by PyTorch NCCL watchdog" in symptoms:
        symptoms.discard("Cancelled due to task failure")

    # Discard pyxis errors when NCCL watchdog is present — cascaded effects
    if "Cancelled by PyTorch NCCL watchdog" in symptoms or "Cancelled by site automation" in symptoms:
        symptoms.discard("error: pyxis")
        symptoms.discard("pyxis: couldn't start container")
        symptoms.discard("pyxis: container start nonzero exit")
        symptoms.discard("failed to import docker image")
        symptoms.discard("pyxis: failed to create container filesystem")


# Node offline composite detection


@refiner_def(
    name="NodeOfflineComposite",
    author="NVIDIA Attribution team",
    priority=_priority("NodeOfflineComposite"),
    derived_rules=["NS connection + link down + FM lost all GPUs"],
)
def node_offline_composite(context: RefinerContext):
    """Detect node offline from NS connection symptom + link down + FM lost all GPUs on same node."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    job_info = context.job_info
    if symptoms is None or my_bad_nodes is None or job_info is None:
        return

    link_down = "link down(east-west)"
    fm_lost_symptoms = [
        "Fabric manager lost connection to all GPUs",
        "Fabric manager lost connection to GPU",
    ]

    if not any(_sym in symptoms for _sym in POTENTIAL_NS_CONNECTION_SYMPTOMS):
        return

    for node, errors in my_bad_nodes.attributions.items():
        has_ns = any(s in errors for s in POTENTIAL_NS_CONNECTION_SYMPTOMS)
        matched_fm = next((fm for fm in fm_lost_symptoms if fm in errors), None)
        if has_ns and link_down in errors and matched_fm is not None:
            source = job_info.source(None, "NS connection + link down + FM lost GPUs")
            my_bad_nodes[node].add(SourceSet.new_one(RegisteredSymptoms.NODE_OFFLINE.value, source))
            symptoms.add(SourceSet.new_one(RegisteredSymptoms.NODE_OFFLINE.value, source))
            for _sym in POTENTIAL_NS_CONNECTION_SYMPTOMS:
                errors.discard(_sym)
                symptoms.discard(_sym)
            errors.discard(link_down)
            symptoms.discard(link_down)
            errors.discard(matched_fm)
            symptoms.discard(matched_fm)


@refiner_def(
    name="NodeOfflineXid145Cascade", author="NVIDIA Attribution team", priority=_priority("NodeOfflineXid145Cascade")
)
def node_offline_xid145_cascade(context: RefinerContext):
    """Node offline → discard XID-145 variants (cascaded from the node going down)."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None:
        return

    if RegisteredSymptoms.NODE_OFFLINE.value in symptoms:
        _discard_xid_cascade_from_symptoms_and_nodes(symptoms, my_bad_nodes, ["XID-145"])


# IB symptom cleanup


@refiner_def(
    name="IBSymptomCleanup",
    author="NVIDIA Attribution team",
    priority=_priority("IBSymptomCleanup"),
    derived_rules=["IBSymptomCleanup", "Found IB error"],
)
def ib_symptom_cleanup(context: RefinerContext):
    """Clean up IB/EFA cascaded symptoms based on root cause analysis."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    es = context.event_stream
    job_info = context.job_info

    if symptoms is None or my_bad_nodes is None:
        return

    if any(symptom in symptoms for symptom in _IB_CLEANUP_SKIP_SYMPTOMS):
        return

    # CASE 1: IB Errors + link down = fabric issue
    has_ib_error = any(s in symptoms for s in _IB_ERROR_TRIGGERS)

    if has_ib_error:
        if "link down(east-west)" in symptoms:
            for _node, node_group in my_bad_nodes.items():
                errors = node_group.attributions
                if "link down(east-west)" in errors:
                    symptoms.add(errors["link down(east-west)"])
                    errors.add(errors["link down(east-west)"])
                    _discard_ib_cascade(symptoms)
                    _discard_ib_cascade(errors)
                    _deactivate_bad_pairs(es, "IBSymptomCleanup")
                    break
    elif "link down(east-west)" in symptoms:
        _handle_link_down_time_proximity(symptoms, my_bad_nodes, es)

    # Per-node: non-IB root cause → discard cascaded IB/EFA on that node
    substrings = [p.strip("%") for p in _NON_IB_ROOT_CAUSE_PATTERNS]
    for _node, node_group in my_bad_nodes.items():
        errors = node_group.attributions
        found_root_cause = False
        for err in list(errors.symptoms()):
            for substr in substrings:
                if substr in err:
                    found_root_cause = True
                    break
            if found_root_cause:
                break
        if found_root_cause:
            _discard_ib_cascade(symptoms)
            _discard_ib_cascade(errors)

    # IB hierarchy: Vendor-129 > Error-12 > Error-5
    if "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)" in symptoms:
        symptoms.discard("IB Error-12 (IBV_WC_RETRY_EXC_ERR)")
        symptoms.discard("IB Error-5 (IBV_WC_WR_FLUSH_ERR)")
    elif "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" in symptoms:
        symptoms.discard("IB Error-5 (IBV_WC_WR_FLUSH_ERR)")

    if any(s in symptoms for s in _IBV_MODIFY_QP_SYMPTOMS):
        _remove_network_cascade(symptoms, my_bad_nodes)

    if "Cancelled by Timeout" in symptoms and job_info is not None:
        if job_info.exceeded_time_limit():
            _discard_ib_cascade(symptoms, ib_only=True)

    if "EAP authentication failed" in symptoms and job_info is not None:
        if job_info.job_end_time is not None:
            elapsed_hours = (job_info.job_end_time - job_info.job_start_time).total_seconds() / 3600
            if elapsed_hours < 2:
                symptoms.discard("EAP authentication failed")

    if "cudaErrorInvalidValue" in symptoms:
        symptoms.discard("EFA Error-5")


@refiner_def(
    name="KernelCrashIbCascadeCleanup",
    author="NVIDIA Attribution team",
    priority=_priority("KernelCrashIbCascadeCleanup"),
)
def kernel_crash_ib_cascade_cleanup(context: RefinerContext):
    """Discard peer IB cascades caused by a node's kernel crash.

    A crashed node stops servicing its queue pairs, so peers can report
    retry-exceeded IB Error-12 against a healthy fabric. IB evidence on the
    crashed node itself is preserved.
    """
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None or my_bad_nodes is None:
        return

    crashed_nodes = {
        node
        for node, node_group in my_bad_nodes.items()
        if any(symptom in node_group.attributions for symptom in _KERNEL_CRASH_SYMPTOMS)
    }
    if len(crashed_nodes) == 0:
        return

    discarded: set[str] = set()
    for node, node_group in my_bad_nodes.items():
        if node in crashed_nodes:
            continue
        errors = node_group.attributions
        if not any(_is_ib_error_12(symptom) for symptom in errors.symptoms()):
            continue
        before = set(errors.symptoms())
        _discard_ib_cascade(errors)
        discarded.update(before - set(errors.symptoms()))

    if len(discarded) == 0:
        return

    surviving = {
        symptom for _node, node_group in my_bad_nodes.items() for symptom in node_group.attributions.symptoms()
    }
    for symptom in discarded - surviving:
        symptoms.discard(symptom)


@refiner_def(
    name="WatchdogGpuCascadeCleanup",
    author="NVIDIA Attribution team",
    priority=_priority("WatchdogGpuCascadeCleanup"),
)
def watchdog_gpu_cascade_cleanup(context: RefinerContext):
    """Discard watchdog GPU cascade symptoms after IB cleanup has evaluated them."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None or "Cancelled by PyTorch NCCL watchdog" not in symptoms:
        return

    watchdog_cascade_symptoms = (
        _XID145_RXPIPE_SYMPTOM,
        "cudaErrorContained",
        *_XID145_RXPIPE_COMPANIONS,
    )
    for symptom in watchdog_cascade_symptoms:
        symptoms.discard(symptom)
    if my_bad_nodes is not None:
        for _node, node_group in my_bad_nodes.items():
            for symptom in watchdog_cascade_symptoms:
                node_group.attributions.discard(symptom)


# Post-IB cascading rules


@refiner_def(name="CudaXidCascades", author="NVIDIA Attribution team", priority=_priority("CudaXidCascades"))
def cuda_xid_cascades(context: RefinerContext):
    """Handle CUDA/XID cascading: specific errors supersede generic ones."""
    symptoms = context.symptoms
    if symptoms is None:
        return

    # XID-31 memory fault → cudaErrorIllegalAddress is cascaded
    if "XID-31-MMU_ERR_FLT" in symptoms:
        symptoms.discard("cudaErrorIllegalAddress")

    # cudaErrorInvalidDevice is low-priority when other symptoms present
    if "cudaErrorInvalidDevice" in symptoms and len(symptoms) > 1:
        symptoms.discard("cudaErrorInvalidDevice")

    # No CUDA runtime supersedes No such file
    if "No such file or directory" in symptoms:
        if "No CUDA runtime is found" in symptoms:
            symptoms.discard("No such file or directory")

    # cudaError OOM is root cause; NUMA and DevicesUnavailable are cascaded
    if "cudaError out of memory" in symptoms:
        symptoms.discard("NUMA node memory exhaustion")
        symptoms.discard("cudaErrorDevicesUnavailable")

    # DataLoader root cause → cudaErrorDevicesUnavailable is cascaded
    if "DataLoader Exception" in symptoms:
        symptoms.discard("cudaErrorDevicesUnavailable")


@refiner_def(
    name="ConnectionClosedCascade", author="NVIDIA Attribution team", priority=_priority("ConnectionClosedCascade")
)
def connection_closed_cascade(context: RefinerContext):
    """Discard Connection closed when a stronger root-cause symptom is present."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None:
        return

    if any(root_cause in symptoms for root_cause in _CONNECTION_CLOSED_TRIGGERS):
        _remove_connection_closed_cascade(symptoms, my_bad_nodes)


@refiner_def(name="XidGenericCleanup", author="NVIDIA Attribution team", priority=_priority("XidGenericCleanup"))
def xid_generic_cleanup(context: RefinerContext):
    """When specific XID/ECC/Row Remap is present, discard generic symptoms."""
    symptoms = context.symptoms
    if symptoms is None:
        return

    for sym in symptoms.symptoms():
        if sym.startswith("XID-") or "Row Remap" in sym or "Page Retirement" in sym:
            # discard more general symptom
            symptoms.discard("error running nvidia-smi")
            symptoms.discard("UserBuffer timeout")
            symptoms.discard("cudaErrorECCUncorrectable")
            symptoms.discard("ncclUnhandledCudaError")
            break


# Post-IB composite rules


@refiner_def(name="PmixCleanup", author="NVIDIA Attribution team", priority=_priority("PmixCleanup"))
def pmix_cleanup(context: RefinerContext):
    """pmix collective timeout is low-priority; discard when other symptoms exist."""
    symptoms = context.symptoms
    if symptoms is None:
        return

    # TODO: add other pmix symptoms
    if "pmix collective timeout" in symptoms and len(symptoms) > 1:
        if "Cancelled by user" in symptoms:
            symptoms.discard("Cancelled by user")

        if len(symptoms) > 1:
            symptoms.discard("pmix collective timeout")


@refiner_def(
    name="NSConnectionLost",
    author="NVIDIA Attribution team",
    priority=_priority("NSConnectionLost"),
    derived_rules=["multiple connection issues found"],
)
def ns_connection_lost(context: RefinerContext):
    """Detect north-south network connection lost from multiple connection symptoms."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    job_info = context.job_info
    if symptoms is None or my_bad_nodes is None or job_info is None:
        return

    if any(_sym in symptoms for _sym in POTENTIAL_NS_CONNECTION_SYMPTOMS):
        # Check per-node: 2+ connection symptoms indicate NS connection lost
        for node, errors in my_bad_nodes.attributions.items():
            if sum(1 for s in POTENTIAL_NS_CONNECTION_SYMPTOMS if s in errors) >= 2:
                my_bad_nodes[node].add(
                    SourceSet.new_one(
                        "NS network connection lost",
                        job_info.source(None, "multiple connection issues found"),
                    )
                )
                symptoms.add(
                    SourceSet.new_one(
                        "NS network connection lost",
                        job_info.source(None, "multiple connection issues found"),
                    )
                )
                for _sym in POTENTIAL_NS_CONNECTION_SYMPTOMS:
                    errors.discard(_sym)
                    symptoms.discard(_sym)

        # prometheus lost connection is a signal, not a failure symptom
        if "prometheus lost connection" in symptoms:
            symptoms.discard("prometheus lost connection")


# Python bug / XID-74 cleanup


@refiner_def(name="PythonBugXidCleanup", author="NVIDIA Attribution team", priority=_priority("PythonBugXidCleanup"))
def python_bug_xid_cleanup(context: RefinerContext):
    """Python bug vs IB errors / XID-74 mutual exclusion."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None:
        return

    # Python issues make IB error attribution unreliable
    if "Python bug" in symptoms:
        for symptom in list(symptoms.symptoms()):
            if "IB Error-12" in symptom or "IB Error-5" in symptom:
                symptoms.discard(symptom)

    # XID-74 (NVLink HW error) makes Python bug attribution unreliable
    if any(symptom.startswith("XID-74") for symptom in symptoms.symptoms()):
        symptoms.discard("Python bug")
        # Also clean up "Python bug" from bad_nodes
        if my_bad_nodes is not None:
            for node in my_bad_nodes.attributions:
                my_bad_nodes.attributions[node].attributions.discard("Python bug")


# Node failure / NFS cleanup


@refiner_def(
    name="NodeFailureNfsCleanup", author="NVIDIA Attribution team", priority=_priority("NodeFailureNfsCleanup")
)
def node_failure_nfs_cleanup(context: RefinerContext):
    """Handle node failure, imex restart, and NFS-related symptom cleanup."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None:
        return

    # If a node failed, the imex service will be restarted
    if any(
        symptom in symptoms
        for symptom in [
            "Node failure reported by slurm",
            "Node not responding",
            "Node unexpectedly rebooted",
            "System is powering down",
        ]
    ):
        symptoms.discard("Stopping nvidia-imex.service")

    # An NFS not responding failure is recoverable, but if we also see a SLURM node failure it's
    # likely the NFS server was the root cause of the node failure.
    if "Node failure reported by slurm" in symptoms and my_bad_nodes is not None:
        for node, node_attrs in my_bad_nodes.attributions.items():
            if "nfs not responding" in node_attrs.attributions:
                nfs_source = node_attrs.attributions["nfs not responding"].one_source()
                symptoms.add(SourceSet.new_one("nfs server failure", nfs_source))
                symptoms.discard("nfs not responding")
                symptoms.discard("Node failure reported by slurm")

    # If there's no node failure, just register this symptom in interesting
    if "nfs not responding" in symptoms:
        symptoms.discard("nfs not responding")


# Last resort / job failed


@refiner_def(name="LastResortSymptoms", author="NVIDIA Attribution team", priority=_priority("LastResortSymptoms"))
def last_resort_symptoms(context: RefinerContext):
    """When no meaningful symptoms remain, fall back to SLURM job state."""
    symptoms = context.symptoms
    job_info = context.job_info
    if symptoms is None or job_info is None:
        return

    # LAST RESORTS
    #
    # TODO: too messy
    if not symptoms or (len(symptoms) == 1 and ("Cancelled by user" in symptoms or "FalsePositive-XID-64" in symptoms)):
        if job_info.state == "COMPLETED":
            symptoms.add(job_info.source_value("the process completes", job_info.state, "last resort; slurm status"))
        elif _is_root_cancellation(job_info):
            symptoms.add(job_info.source_value("Cancelled by root", job_info.state, "last resort; slurm status"))
        elif job_info.state.startswith("CANCELLED"):
            symptoms.add(job_info.source_value("Cancelled by user", job_info.state, "last resort; slurm status"))
        elif job_info.state == "REQUEUED":
            symptoms.add(job_info.source_value("Job requeued", job_info.state, "last resort; slurm status"))
        elif job_info.state == "RUNNING":
            symptoms.add(job_info.source_value("Job running", job_info.state, "last resort; running but amiss"))
        else:
            symptoms.add(job_info.source_value("job failed", job_info.state, "last resort; slurm status"))


@refiner_def(name="JobFailedState", author="NVIDIA Attribution team", priority=_priority("JobFailedState"))
def job_failed_state(context: RefinerContext):
    """Ensure FAILED jobs have a 'job failed' symptom when no cancellation is present."""
    symptoms = context.symptoms
    job_info = context.job_info
    if symptoms is None or job_info is None:
        return

    if (
        job_info.state == "FAILED"
        and "terminate called" not in symptoms
        and "Cancelled by root" not in symptoms
        and "Cancelled by user" not in symptoms
        and "Cancelled by cluster operators" not in symptoms
        and "Cancelled by site automation" not in symptoms
        and "Cancelled by PyTorch NCCL watchdog" not in symptoms
    ):
        symptoms.add(job_info.source_value("job failed", job_info.state, "last report; searching for symptom"))


# Fabric manager cascade


@refiner_def(name="FabricManagerCascade", author="NVIDIA Attribution team", priority=_priority("FabricManagerCascade"))
def fabric_manager_cascade(context: RefinerContext):
    """NV_ERR_FABRIC_MANAGER_NOT_PRESENT → discard cascaded network symptoms."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None or my_bad_nodes is None:
        return

    if "NV_ERR_FABRIC_MANAGER_NOT_PRESENT" in symptoms:
        remove_cascaded_network_symptoms(symptoms, my_bad_nodes)


# XID-171/172 filtering logic: Discard cascaded symptoms when root cause is present
# Unlike XID-149 which filters between sub-symptoms, XID-171/172 filter out entirely different symptoms


def _discard_xid_cascade_from_symptoms_and_nodes(symptoms, my_bad_nodes, patterns: list[str]) -> None:
    """Discard symptoms matching any of the given prefixes from both symptoms and bad_nodes."""
    for symptom in list(symptoms.symptoms()):
        if any(symptom.startswith(pattern) for pattern in patterns):
            symptoms.discard(symptom)

    if my_bad_nodes is not None:
        for node in my_bad_nodes.attributions:
            for symptom in list(my_bad_nodes.attributions[node].attributions.symptoms()):
                if any(symptom.startswith(pattern) for pattern in patterns):
                    my_bad_nodes.attributions[node].attributions.discard(symptom)


@refiner_def(name="Xid171Cascade", author="NVIDIA Attribution team", priority=_priority("Xid171Cascade"))
def xid_171_cascade(context: RefinerContext):
    """XID-171 (UNCORRECTABLE_DRAM_ERROR) → discard less specific cascaded symptoms."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None:
        return

    if any(sym.startswith("XID-171") for sym in symptoms.symptoms()):
        _discard_xid_cascade_from_symptoms_and_nodes(
            symptoms,
            my_bad_nodes,
            [
                "XID-48-",  # Generic HBM error — XID-171 tells us it's specifically DRAM
                "XID-63-",  # Row remap is a consequence of DRAM error
                "XID-94-",  # Generic contained error — XID-171 is more specific
                "cudaErrorContained",  # CUDA's generic error — XID-171 gives us the actual HW cause
            ],
        )


@refiner_def(name="Xid172Cascade", author="NVIDIA Attribution team", priority=_priority("Xid172Cascade"))
def xid_172_cascade(context: RefinerContext):
    """XID-172 (UNCORRECTABLE_SRAM_ERROR) → discard less specific cascaded symptoms."""
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None:
        return

    if any(sym.startswith("XID-172") for sym in symptoms.symptoms()):
        _discard_xid_cascade_from_symptoms_and_nodes(
            symptoms,
            my_bad_nodes,
            [
                "XID-13-",  # GR_EXCEPTION/GR_ERROR_SW_NOTIFY — cascaded from SRAM error
                "XID-48-",  # Generic HBM error — XID-172 tells us it's specifically SRAM
                "XID-94-",  # Generic contained error — XID-172 is more specific
                "cudaErrorContained",  # CUDA's generic error — XID-172 gives us the actual HW cause
            ],
        )


@refiner_def(
    name="ExtractJobMetadataFallback",
    author="NVIDIA Attribution team",
    priority=_priority("ExtractJobMetadataFallback"),
    derived_rules=["last resort; metadata from logfile", "last report; exit code signal"],
)
def extract_job_metadata_fallback(context: RefinerContext):
    """Last-resort reclassification when the only symptom is 'Cancelled by user'."""
    symptoms = context.symptoms
    job_info = context.job_info
    if symptoms is None:
        return

    if len(symptoms) == 1 and "Cancelled by user" in symptoms:
        source = symptoms["Cancelled by user"].one_source()
        if context.extract_job_metadata_from_logfile:
            symptoms.discard("Cancelled by user")
            symptoms.add(SourceSet("job failed").add(source.derived(rule="last resort; metadata from logfile")))
        exit_code_signal = job_info.exit_code_signal
        if exit_code_signal is not None and exit_code_signal > 0:
            symptoms.discard("Cancelled by user")
            symptoms.add(
                SourceSet(f"exit code signal {exit_code_signal}").add(
                    source.derived(rule="last report; exit code signal")
                )
            )


_SYMPTOMS_TO_CHECK = [
    "effective_ber_high",
    "plr_tx_bw_loss_high",
    "hist15_high",
    "plr_tx_bw_loss_switch_high",
    "hist15_switch_high",
    "effective_ber_magnitude_switch_high",
    "hist15_and_plr_tx_bw_loss",
]


@refiner_def(
    name="Xid149AndNvlinkFiltering", author="NVIDIA Attribution team", priority=_priority("Xid149AndNvlinkFiltering")
)
def xid149_and_nvlink_filtering(context: RefinerContext):
    """XID-149 priority filtering and nvlink symptom cleanup.

    Priority: XID-149.4 > XID-149.10 > other XID-149.x > XID-149.0
    """
    symptoms = context.symptoms
    my_bad_nodes = context.my_bad_nodes
    if symptoms is None:
        return

    xid_149_symptoms = [sym for sym in symptoms.symptoms() if sym.startswith("XID-149")]

    # Allowlist for GB200
    if xid_149_symptoms:
        has_xid_149_4 = any("XID-149.4-" in sym for sym in xid_149_symptoms)
        has_xid_149_10 = any("XID-149.10-" in sym for sym in xid_149_symptoms)
        has_xid_149_0 = any("XID-149.0-" in sym for sym in xid_149_symptoms)

        if has_xid_149_4:
            # If XID-149.4 is seen, discard all other XID-149 symptoms
            for sym in xid_149_symptoms:
                if "XID-149.4-" not in sym:
                    symptoms.discard(sym)
        elif has_xid_149_10:
            # If XID-149.10 is seen, discard all other XID-149 symptoms
            for sym in xid_149_symptoms:
                if "XID-149.10-" not in sym:
                    symptoms.discard(sym)
        elif has_xid_149_0 and len(xid_149_symptoms) > 1:
            # If XID-149.0 is present with any other XID-149, discard XID-149.0
            for sym in xid_149_symptoms:
                if "XID-149.0-" in sym:
                    symptoms.discard(sym)

        if has_xid_149_4 or has_xid_149_10:
            clean_bad_nodes_of_cascaded_symptoms(
                my_bad_nodes,
                [
                    "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
                    "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0",
                    "effective_ber_high",
                    "plr_tx_bw_loss_high",
                    "hist15_high",
                    "plr_tx_bw_loss_switch_high",
                    "hist15_switch_high",
                    "effective_ber_magnitude_switch_high",
                ],
            )

        # Find nodes that have XID-149.4 or XID-149.10, check if they are on the same node with nvlink symptoms
        # These symptoms are for research purposes right now.

        xid_149_nodes = set()
        for node, node_attrs in my_bad_nodes.attributions.items():
            if (
                "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0" in node_attrs.attributions
                or "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0" in node_attrs.attributions
            ):
                xid_149_nodes.add(node)

        for k in my_bad_nodes:
            for symptom in my_bad_nodes[k].symptoms():
                if symptom in _SYMPTOMS_TO_CHECK:
                    # They are too noisy to be appeared in symptoms so comment out for now.
                    # if k in xid_149_nodes:
                    #     allowed_symptoms.append(symptom)
                    # else:
                    my_bad_nodes[k].discard(symptom)

    else:
        for symptom in _SYMPTOMS_TO_CHECK:
            if symptom in symptoms:
                symptoms.discard(symptom)
            for k in my_bad_nodes:
                for symptom in my_bad_nodes[k].symptoms():
                    if symptom in _SYMPTOMS_TO_CHECK:
                        my_bad_nodes[k].discard(symptom)


@refiner_def(
    name="Gb200PriorityRootCause", author="NVIDIA Attribution team", priority=_priority("Gb200PriorityRootCause")
)
def gb200_priority_root_cause(context: RefinerContext):
    """Priority-based root cause symptom filtering for GB200."""
    symptoms = context.symptoms
    if symptoms is None:
        return

    root_cause_symptoms_by_priority = [
        ["XID-149"],  # Priority 1 (highest)
        ["Stopping nvidia-imex.service"],  # Priority 2
        ["NV_ERR_FABRIC_MANAGER_NOT_PRESENT"],  # Priority 3 (lowest)
    ]

    all_symptoms = list(symptoms.symptoms())

    all_root_cause_patterns = [
        pattern for priority_group in root_cause_symptoms_by_priority for pattern in priority_group
    ]
    matching_root_cause_symptoms = []
    for symptom in all_symptoms:
        if any(rc in symptom for rc in all_root_cause_patterns):
            matching_root_cause_symptoms.append(symptom)

    # If we found any root cause symptoms, keep only those
    if matching_root_cause_symptoms:
        for sym in all_symptoms:
            if sym not in matching_root_cause_symptoms:
                symptoms.discard(sym)


# DAG cascade pruning — runs last, on the already-refined symptoms and my_bad_nodes


@refiner_def(name="DagCascadePruning", author="NVIDIA Attribution team", priority=_priority("DagCascadePruning"))
def dag_cascade_pruning(context: RefinerContext):
    """Apply symptom DAG cascade pruning on the refined set.

    Translates the full my_bad_nodes to BaseSymptom, applies DAG cascade
    rules, replaces raw symptom names in the job-level symptoms with the
    translated+pruned result, and projects pruning decisions back to
    my_bad_nodes.
    """
    my_bad_nodes = context.my_bad_nodes
    symptoms = context.symptoms
    job_info = context.job_info
    if my_bad_nodes is None or symptoms is None or job_info is None:
        return

    # Snapshot which my_bad_nodes symptoms were discarded by earlier refiners
    all_from_bad_nodes: set[str] = set()
    for _node, node_group in my_bad_nodes.items():
        for symptom_str in node_group.symptoms():
            if "non-fatal" not in symptom_str:
                all_from_bad_nodes.add(symptom_str)

    discarded_by_refiners = all_from_bad_nodes - set(symptoms.symptoms())

    node_symptoms_base, skipped = translate_and_prune_node_attributions(
        my_bad_nodes, job_info, recorder=context.recorder
    )

    context.node_symptoms_base = node_symptoms_base
    context.skipped_symptom_ids = skipped

    for s in all_from_bad_nodes & set(symptoms.symptoms()):
        symptoms.discard(s, purge=True)
    for per_node_set in node_symptoms_base.values():
        for bs, src_set in per_node_set.items():
            symptoms.add(src_set.value_mapped_copy(lambda s: str(s)))

    # Re-apply earlier refiners' discards so their work is not undone
    for s in discarded_by_refiners:
        symptoms.discard(s)

    # Project pruning decisions back to my_bad_nodes (demote pruned → interesting)
    project_node_symptoms_to_bad_nodes(my_bad_nodes, node_symptoms_base)


@refiner_def(name="PostProcessSymptoms", author="NVIDIA Attribution team", priority=_priority("PostProcessSymptoms"))
def post_process_symptoms_refiner(context: RefinerContext):
    """Post-outcome symptom cleanup: extract cancelled_by, discard low-priority symptoms."""
    symptoms = context.symptoms
    job_info = context.job_info
    outcome = context.outcome
    cancelled_by = context.cancelled_by
    if symptoms is None or job_info is None:
        return

    if outcome is not None and outcome.value().outcome in ["Fail InfraHW", "Fail InfraSW"]:
        symptoms.discard("No such file or directory")
        symptoms.discard("Broken pipe")

    hook_params = {"cancelled_by": cancelled_by}
    multihooks.PostProcessSymptoms(symptoms, hook_params)
    cancelled_by = hook_params["cancelled_by"]

    # When only "Cancelled by user" and "Python Exception" co-exist,
    # treat cancellation as root cause and discard the Python exception
    # (the exception is likely a cascading effect of the cancellation signal)
    if "Cancelled by user" in symptoms and "Python Exception" in symptoms and len(symptoms) == 2:
        symptoms.discard("Python Exception")
    # Python Exception + Timeout likely indicate the exception is not real failure.
    # This allows the outcome reclassification to mark it as Pass
    if "Cancelled by Timeout" in symptoms and "Python Exception" in symptoms and len(symptoms) == 2:
        symptoms.discard("Python Exception")
    # Same for root / scheduler cancellation
    if "Cancelled by root" in symptoms and "Python Exception" in symptoms and len(symptoms) == 2:
        symptoms.discard("Python Exception")
    if "Cancelled by cluster operators" in symptoms and "Python Exception" in symptoms and len(symptoms) == 2:
        symptoms.discard("Python Exception")

    # Replace transient "Cancelled due to task failure" with "job failed"
    if "Cancelled due to task failure" in symptoms:
        symptoms.discard("Cancelled due to task failure")
        if not symptoms:
            symptoms.add(job_info.source_value("job failed", "task failure", "task failure cancellation"))

    if "Job requeued" in symptoms and len(symptoms) > 1:
        symptoms.discard("Job requeued")
    if "job failed" in symptoms and len(symptoms) > 1:
        symptoms.discard("job failed")
    if "Cancelled by PyTorch NCCL watchdog" in symptoms:
        if len(symptoms) > 1:
            symptoms.discard("Cancelled by PyTorch NCCL watchdog")
        cancelled_by = "PyTorch NCCL watchdog"
    if "Cancelled by Timeout" in symptoms:
        if len(symptoms) > 1:
            symptoms.discard("Cancelled by Timeout")
        cancelled_by = "Timeout"
    if "Cancelled by root" in symptoms:
        if len(symptoms) > 1:
            symptoms.discard("Cancelled by root")
        cancelled_by = "root"
    if "Cancelled by cluster operators" in symptoms:
        if len(symptoms) > 1:
            symptoms.discard("Cancelled by cluster operators")
    if "Cancelled by user" in symptoms and len(symptoms) > 1:
        symptoms.discard("Cancelled by user")
    if "local catastrophic error" in symptoms and len(symptoms) > 1:
        symptoms.discard("local catastrophic error")
    if "terminate called" in symptoms and len(symptoms) > 1:
        symptoms.discard("terminate called")

    # if "the process completes" in symptoms and len(symptoms) > 1:
    #     symptoms.discard("the process completes")

    if "Python Exception" in symptoms and len(symptoms) > 1:
        symptoms.discard("Python Exception")

    if "Segmentation fault" in symptoms and len(symptoms) > 1:
        symptoms.discard("Segmentation fault")

    for nccl_error in [
        "NCCL WARN Cuda failure",
        "ncclUnhandledCudaError",
        "ncclSystemError",
        # "ncclInternalError",
        "ncclInvalidArgument",
        "ncclInvalidUsage",
        "ncclRemoteError",
        "ncclTimeout",
    ]:
        # TODO: should still keep them for debugging purpose
        if nccl_error in symptoms and len(symptoms) > 1:
            symptoms.discard(nccl_error)

    if "UserBuffer timeout" in symptoms:
        if "IB Link Down" in symptoms or "ECC Error" in symptoms:
            symptoms.discard("UserBuffer timeout")

    filter_cascade_symptoms_symptomset(symptoms, context.my_bad_nodes)

    if context.attribution_result is not None:
        context.attribution_result.cancelled_by = cancelled_by


_XID145_DYNAMIC_SYMPTOMS = frozenset(
    {"XID-145-InfraHW", "XID-145-InfraSW", "XID-145-MultiCategory", "XID-145-Undetermined"}
)


@refiner_def(
    name="DemoteBadNodesOnIncompleteInfo",
    author="NVIDIA Attribution team",
    priority=_priority("DemoteBadNodesOnIncompleteInfo"),
)
def demote_bad_nodes_on_incomplete_info(context: RefinerContext):
    """When no root cause was determined, demote the bad-node symptoms to interesting.

    Moved out of core.attribute. JobNodeAttributions.clear() demotes (bad -> interesting)
    rather than removing, so the node stays for reporting but no longer counts as bad.
    Runs first in the post-outcome phase so it reads the pre-reclassification outcome,
    matching the old inline check.
    """
    outcome = context.outcome
    my_bad_nodes = context.my_bad_nodes
    if outcome is None or my_bad_nodes is None:
        return
    if outcome.value().outcome == "Incomplete Info":
        my_bad_nodes.clear()


@refiner_def(
    name="CleanupXid145DynamicSymptoms",
    author="NVIDIA Attribution team",
    priority=_priority("CleanupXid145DynamicSymptoms"),
)
def cleanup_xid145_dynamic_symptoms_refiner(context: RefinerContext):
    """Discard the XID-145 dynamic sub-symptoms after outcome selection.

    These are synthesized per-category variants that may drive outcome selection
    but are not reported as standalone symptoms. Formerly inlined at the end of
    ``outcome.calculate_outcome`` (both the matched and unmatched branches)."""
    symptoms = context.symptoms
    if symptoms is None:
        return
    for symptom in list(symptoms.symptoms()):
        if symptom in _XID145_DYNAMIC_SYMPTOMS:
            symptoms.discard(symptom)


@refiner_def(name="FilterFatalXidNoise", author="NVIDIA Attribution team", priority=_priority("FilterFatalXidNoise"))
def filter_fatal_xid_noise_refiner(context: RefinerContext):
    """Filter symptoms to prioritize fatal XID errors over noise."""
    symptoms = context.symptoms
    if symptoms is None:
        return

    fatal_prefixes = ["XID-79"]
    fatal_symptoms = [s for s in symptoms.symptoms() if any(s.startswith(p) for p in fatal_prefixes)]

    if fatal_symptoms:
        for symptom in list(symptoms.symptoms()):
            if symptom not in fatal_symptoms:
                symptoms.discard(symptom)


@refiner_def(
    name="OutcomeReclassification",
    author="NVIDIA Attribution team",
    priority=_priority("OutcomeReclassification"),
    derived_rules=[
        "slurm state",
        "reclassified: slurm preempted",
        "reclassified: user cancellation",
        "reclassified: pytorch timeout < slurm limit",
        "reclassified: timeout only symptom",
        "reclassified: nvrx exit success",
    ],
)
def outcome_reclassification_refiner(context: RefinerContext):
    """Apply post-processing outcome reclassification rules."""
    if context.attribution_result is None or context.job_info is None:
        return
    attribution_result = context.attribution_result
    job_info = context.job_info
    event_stream = context.event_stream

    # NVRx ft_launcher exit code: job-level result only, NVRx jobs only.
    # A nonzero code or signal means the workload failed and must not be flipped to
    # Pass by the PREEMPTED/timeout branches below; a clean 0:0 result can resolve an
    # ambiguous outcome to Pass. Missing/None is ignored (leave the outcome unchanged).
    nvrx_exit_fields = (job_info.exit_code, job_info.exit_code_signal)
    is_nvrx = NVRX_TAG in job_info.tags
    nvrx_failed = is_nvrx and any(v is not None and v != 0 for v in nvrx_exit_fields)

    # Reclassify PREEMPTED jobs to Pass (unless NVRx reports a failure)
    if not nvrx_failed and job_info.state is not None and job_info.state.upper() == "PREEMPTED":
        previous_source = None
        if attribution_result.outcome_attr is not None:
            previous_source = attribution_result.outcome_attr.one_source()

        if previous_source is None:
            previous_source = job_info.source(job_info.state, "slurm state")

        new_outcome = OutcomeAttribution(
            outcome="Pass",
            hw_cause="Not Applicable",
            root_cause_symptom="SLURM preempted",
        )

        attribution_result.outcome_attr = SourceSet(new_outcome).add(
            previous_source.derived(rule="reclassified: slurm preempted")
        )
        return

    # Reclassify TIMEOUT to Pass when it's the only symptom (unless NVRx reports a failure)
    if not nvrx_failed and job_info.time_limit is not None:
        normalized_symptoms = [s.lower() for s in attribution_result.symptoms.symptoms()]
        is_slurm_timeout = (
            attribution_result.cancelled_by is not None
            and bool(attribution_result.cancelled_by)
            and attribution_result.cancelled_by.lower() == "timeout"
        )
        has_user_cancel_symptom = any("cancelled by user" in s for s in normalized_symptoms)
        has_only_timeout_symptoms = (
            all(s.startswith("cancelled by timeout") for s in normalized_symptoms) or not normalized_symptoms
        )

        if is_slurm_timeout and not has_user_cancel_symptom and has_only_timeout_symptoms:
            pytorch_timeout_minutes = get_pytorch_timeout_minutes(event_stream)
            if pytorch_timeout_minutes is not None and pytorch_timeout_minutes.value() < float(job_info.time_limit):
                root_cause_symptom = "SLURM timeout with shorter user distributed timeout"
                derived_rule = "reclassified: pytorch timeout < slurm limit"
                fallback_source = pytorch_timeout_minutes.one_source()
            else:
                root_cause_symptom = "SLURM timeout (no other errors)"
                derived_rule = "reclassified: timeout only symptom"
                fallback_source = job_info.source(job_info.state, "slurm state")

            previous_source = (
                attribution_result.outcome_attr.one_source() if attribution_result.outcome_attr is not None else None
            )

            new_outcome = OutcomeAttribution(
                outcome="Pass",
                hw_cause="Not Applicable",
                root_cause_symptom=root_cause_symptom,
            )

            base_source = previous_source if previous_source is not None else fallback_source
            attribution_result.outcome_attr = SourceSet(new_outcome).add(base_source.derived(rule=derived_rule))
            return

    # NVRx clean success (0:0) resolves a genuinely ambiguous outcome to Pass. Uses an
    # allowlist (not a denylist) so it only touches outcomes with no specific category to
    # mask: any categorized failure (InfraHW/InfraSW/User/MultiCategory), a cancellation,
    # or an already-decided Pass is left alone, and new outcomes are safe by default.
    if is_nvrx and job_info.exit_code == 0 and job_info.exit_code_signal == 0:
        current_outcome = (
            attribution_result.outcome_attr.value().outcome if attribution_result.outcome_attr is not None else None
        )
        if current_outcome in ("Incomplete Info", "Fail Undetermined"):
            previous_source = (
                attribution_result.outcome_attr.one_source() if attribution_result.outcome_attr is not None else None
            )
            if previous_source is None:
                previous_source = job_info.source("ft_launcher exit 0:0", "nvrx exit code")
            new_outcome = OutcomeAttribution(
                outcome="Pass",
                hw_cause="Not Applicable",
                root_cause_symptom="NVRx ft_launcher exit 0",
            )
            attribution_result.outcome_attr = SourceSet(new_outcome).add(
                previous_source.derived(rule="reclassified: nvrx exit success")
            )
            return

    # Reclassify to "Cancelled by User" when job was cancelled by a regular user
    apply_user_cancellation_reclassification(attribution_result, job_info)


@refiner_def(
    name="SingleNodeInfraBackfill",
    author="NVIDIA Attribution team",
    priority=_priority("SingleNodeInfraBackfill"),
    derived_rules=["single-node infra failure -> sole node"],
)
def single_node_infra_backfill_refiner(context: RefinerContext):
    """Blame the sole node of a single-node job when the outcome is Fail InfraHW.

    A job that ran on exactly one node and was classified Fail InfraHW implicates
    its only node even when no node-scoped evidence named it, so the bad-node
    reporting fields are populated for these jobs. Fail InfraSW is intentionally
    excluded: single-node software/network failures (e.g. "enroot: curl empty
    reply from server") are often caused by the remote end, so the sole node is
    not reliably the bad node. Runs after OutcomeReclassification so it sees the
    final outcome.
    """
    attribution_result = context.attribution_result
    job_info = context.job_info
    my_bad_nodes = context.my_bad_nodes
    if attribution_result is None or job_info is None or my_bad_nodes is None:
        return
    if len(job_info.nodes) != 1:
        return
    outcome_attr = attribution_result.outcome_attr
    if outcome_attr is None or outcome_attr.value().outcome != "Fail InfraHW":
        return
    if any(group.has_bad() for group in my_bad_nodes.attributions.values()):
        return

    outcome_value = outcome_attr.value()
    symptom = (
        outcome_value.root_cause_symptom
        if (outcome_value.root_cause_symptom is not None and outcome_value.root_cause_symptom != "")
        else outcome_value.outcome
    )
    # Some infra symptoms are not node-attributable even on a single-node job
    # (fabric/collective/catastrophic signals). get_clean_bad_nodes strips these
    # from per-node attributions, so do not re-introduce them here.
    if is_non_node_attributable_symptom(symptom):
        return
    source = outcome_attr.one_source()
    if source is not None:
        source = source.derived(rule="single-node infra failure -> sole node")
    else:
        source = job_info.source(job_info.state, "single-node infra failure -> sole node")
    my_bad_nodes.add([job_info.nodes[0]], SourceSet.new_one(symptom, source))
