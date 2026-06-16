# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from datetime import timedelta
from typing import Optional

from attribution_lib.attr_types import (
    JobNodeAttributions,
)
from attribution_lib.events import BAD_PAIR_EVENT_TYPES, EventStream, sql_placeholders
from attribution_lib.filter import get_pytorch_timeout_minutes
from attribution_lib.hooks import singlehooks
from attribution_lib.job_info import JobInfo
from attribution_lib.sources import (
    SymptomSet,
)


def clear_bad_pairs(event_stream: EventStream, rule: str):
    """Deactivate bad pairs in stream"""
    type_ids = event_stream.mgr.types.get_ids(BAD_PAIR_EVENT_TYPES)
    if not type_ids:
        return

    rule_id = event_stream.mgr.rules.get_id(rule)
    type_ph = sql_placeholders(len(type_ids))
    event_stream.execute(
        f"UPDATE events SET status=0, source_rule=? WHERE context_id=? AND event_type IN ({type_ph}) AND status=1",
        [rule_id, event_stream.ctx_id] + type_ids,
    )


def clean_bad_nodes_of_cascaded_symptoms(my_bad_nodes: JobNodeAttributions, symptoms_to_keep: list[str]):
    """Given a symptom_to_keep, we believe all other symptoms are cascaded, so discard anything but this symptom"""
    for k in my_bad_nodes:
        for symptom in my_bad_nodes[k].symptoms():
            if symptom not in symptoms_to_keep:
                my_bad_nodes[k].discard(symptom)


def ib_symptom_cleanup(
    symptoms: SymptomSet[str],
    my_bad_nodes: JobNodeAttributions,
    event_stream: EventStream,
    job_info: JobInfo | None = None,
):
    def discard_cascaded_ib_symptoms(symptom_list: SymptomSet[str], ib_and_efa_errors_only: bool = False):
        # TODO: discard cascaded symptoms is very dirty, switch to state machine ASAP
        # dismiss all IB errors
        ib_and_efa_errors = ["IB Error-", "EFA Error-"]
        general_errors = [
            "Mellanox links are down",
            "ncclRemoteError",
            "ncclInternalError",
            "Connection reset by peer",
            "Connection closed",
            "PyTorch NCCL watchdog",
            "UserBuffer timeout",
            # if network flap was reported, we should not report ibv_modify_qp
            # as a symptom because it is likely to be caused by the flap
            # otherwise, it is recommended to update NCCL to the version with
            # the modify QP retry.
            # similarly discard ibv_set_ece reported by connection drop.
            "Call to ibv_modify_qp failed",
            "Call to ibv_set_ece failed",
        ]

        # Discard all IB and EFA errors
        for symptom in list(symptom_list.symptoms()):
            for error in ib_and_efa_errors:
                if error in symptom:
                    symptom_list.discard(symptom)

        if ib_and_efa_errors_only:
            return

        for error in general_errors:
            symptom_list.discard(error)

    def _should_discard_link_down(source, watchdog_cancellation_timestamp, threshold_seconds):
        """Return True if this link down source should be discarded."""
        if not source.time:
            return True  # No timestamp available, discard to be safe
        time_diff = (watchdog_cancellation_timestamp - source.time).total_seconds()
        # Discard if:
        # - time_diff < 0: link down happened after cancellation
        # - time_diff > threshold: link down happened too far before cancellation
        return time_diff < 0 or time_diff > threshold_seconds

    # ==========================================================================
    # Link Down Attribution Logic Summary:
    # ==========================================================================
    # This section determines whether "link down(east-west)" is a fatal symptom.
    #
    # CASE 1: IB Errors present (IB Error-12 or Error-5)
    #   - If "link down(east-west)" co-exists with IB errors → fabric issue
    #   - Action: Promote link down as root cause, discard cascaded IB symptoms
    #   - Rationale: IB-12/5 + link down(east-west) = fabric failure
    #                IB-12/5 + gpu failure = gpu failure (handled elsewhere)
    #
    # CASE 2: link down(east-west) alone (no IB errors) - PyTorch watchdog detection
    #   - Subcase 2a: Cancellation timestamp available (PyTorch NCCL watchdog / site automation)
    #       - Keep link down ONLY if it occurred within (pytorch_timeout + 60s buffer)
    #         before the watchdog cancellation
    #       - Events outside this window → moved to "interesting" (non-fatal)
    #       - If no valid link down events remain → discard the symptom entirely
    #   - Subcase 2b: No cancellation timestamp
    #       - Cannot determine if link down is fatal → discard all link down symptoms
    #
    # TODO: Improve link down attribution by considering NCCL timeout duration.
    #       Currently, we only check if link down occurred within the PyTorch timeout window.
    #       However, if link down and another symptom (the actual root cause) occur simultaneously,
    #       and the link recovers before NCCL timeout expires, the link down should not be
    #       attributed as the root cause. Future improvement: incorporate link recovery time
    #       and NCCL timeout to avoid false attribution when link down is transient.
    # ==========================================================================
    # Per Pasha suggestion:
    #       Current logic checks if the port error occurred in the time window of watchdog timeout and if this is the case,
    #       the port error is considered primary reason and not the watchdog timeout.
    #       Nevertheless, there is a chance that port becomes active and it is not a real reason for watchdog event.
    #       Unfortunately, counting port error vs port active and trying to figure out if all ports recovered
    #       without explicitly ordered events is challenging.
    #       I was suggesting to implement an methodology that takes in account NCCL retransmission time.
    #       E.g. if total retransmission time is 2 min but flap has happened 5 min before watchdog triggered,
    #       we have 3 min gap. We also do not see any error 12. This is likely sign that port error recovered and port error is not the main reason for watchdog timeout.
    #       In cases when NCCL total retransmission is 8min, 5-8=-3 , so watchdog happened while retransmissions still are going on.
    #       This is a sign that watchdog timeout likely triggered by port error.
    # ==========================================================================
    if (
        "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" in symptoms
        or "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)" in symptoms
        or "IB Error-5 (IBV_WC_WR_FLUSH_ERR)" in symptoms
    ):
        for _node, node_group in my_bad_nodes.items():
            errors = node_group.attributions
            found_issue: Optional[str] = None
            if "link down(east-west)" in errors:
                found_issue = "link down(east-west)"
            if (issue := found_issue) is not None:
                symptoms.add(errors[issue])
                errors.add(errors[issue])
                discard_cascaded_ib_symptoms(symptoms)
                discard_cascaded_ib_symptoms(errors)
                clear_bad_pairs(event_stream, "Found IB error")
                break
    elif "link down(east-west)" in symptoms:
        # pytorch watchdog based link down detection
        # the idea is to keep link down(east-west) symptom only if it is within the pytorch timeout before the watchdog cancellation
        watchdog_cancellation_timestamp = None
        for _pytorch_watchdog_symptom in [
            "Cancelled by PyTorch NCCL watchdog",
            "Cancelled by site automation",
            "PyTorch NCCL watchdog",
        ]:
            if _pytorch_watchdog_symptom in symptoms:
                watchdog_cancellation_source = symptoms[_pytorch_watchdog_symptom].one_source()
                watchdog_cancellation_timestamp = (
                    watchdog_cancellation_source.time if watchdog_cancellation_source else None
                )
                break

        if watchdog_cancellation_timestamp:
            # Get PyTorch timeout from runtime args, default to 10 minutes (600 seconds)
            timeout_seconds = 600  # default 600 seconds if no runtime args is provided
            buffer_seconds = 60  # 1 minute buffer
            if (pytorch_timeout := get_pytorch_timeout_minutes(event_stream)) is not None:
                timeout_seconds = int(pytorch_timeout.value() * 60)  # convert minutes to seconds

            threshold = timeout_seconds + buffer_seconds

            # Filter link down symptoms from my_bad_nodes (per-node attributions)
            # This ensures discarded events are tracked in interesting_nodes
            keep_link_down_in_symptoms = False
            for _node, node_group in my_bad_nodes.items():
                errors = node_group.attributions
                if "link down(east-west)" in errors:
                    node_link_down_set = errors["link down(east-west)"]
                    to_discard = [
                        s
                        for s in node_link_down_set
                        if _should_discard_link_down(s, watchdog_cancellation_timestamp, threshold)
                    ]

                    for source in to_discard:
                        node_link_down_set._interesting.add(source)
                        node_link_down_set._bad.discard(source)

                    if not node_link_down_set.has_bad():
                        errors.discard("link down(east-west)")
                    else:
                        keep_link_down_in_symptoms = True

            if not keep_link_down_in_symptoms:
                # since there is not link down(east-west) symptom in my_bad_nodes, we can discard the symptom from symptoms
                symptoms.discard("link down(east-west)")
            else:
                # discard cascaded IB symptoms, since link down(east-west) is the root cause
                discard_cascaded_ib_symptoms(symptoms)
        else:
            # remove all link down(east-west) symptoms, since we can not tell if they are fatal or not
            symptoms.discard("link down(east-west)")
            for _node, node_group in my_bad_nodes.items():
                errors = node_group.attributions
                if "link down(east-west)" in errors:
                    errors.discard("link down(east-west)")

    for _node, node_group in my_bad_nodes.items():
        errors = node_group.attributions
        errors_copy = list(errors.symptoms())
        for err in errors_copy:
            if (
                "XID-" in err
                or "ECC" in err
                or "Row Remap" in err
                or "Not responding" in err
                or "NaN reported by Megatron" in err
                or "megatron.core.dist_checkpointing.core.CheckpointingException" in err
                or "Page Retirement" in err
                or "Segmentation fault" in err
                or "Node unexpectedly rebooted" in err
                or "Disk quota exceeded" in err
                or "error running nvidia-smi" in err
                or "CE deadlock DETECTED" in err
                or "JSONDecodeError" in err
                or "Node failure reported by slurm" in err
                or "Power Supply Unit failure" in err
                or "System is powering down" in err
                or "EOFError: Ran out of input" in err
                or "http.client.RemoteDisconnected" in err
                or "DataLoader Exception" in err
                or "S3 Exception" in err
                or "cudaError out of memory" in err
                or "System-out of memory" in err
                or "LustreError" in err
                or "assert number_image_tags <= number_of_images" in err
                or "wandb.errors.CommError: Run initialization has timed out" in err
                or "Stale file handle" in err
                or "Python Exception" in err
                or "NCCL invalid NIC selection" in err
            ):
                discard_cascaded_ib_symptoms(symptoms)
                discard_cascaded_ib_symptoms(errors)
                break

    # else: IB-12/5 + vendor 129 = core fabric
    #       IB-12/5 = connection closed due to core switch fabric issue
    if "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)" in symptoms:
        symptoms.discard("IB Error-12 (IBV_WC_RETRY_EXC_ERR)")
        symptoms.discard("IB Error-5 (IBV_WC_WR_FLUSH_ERR)")
    elif "IB Error-12 (IBV_WC_RETRY_EXC_ERR)" in symptoms:
        symptoms.discard("IB Error-5 (IBV_WC_WR_FLUSH_ERR)")

    # "Call to ibv_modify_qp failed" or "Call to ibv_set_ece failed" discard "Connection reset by peer" and "Connection closed"
    # and possibly other nccl errors
    # TODO: fix it with state machine
    if any(
        _smpt in symptoms
        for _smpt in [
            "Call to ibv_modify_qp failed",
            "Call to ibv_set_ece failed",
            "Call to ibv_modify_qp failed: eno 22 invalid argument",
            "Call to ibv_modify_qp failed: eno 110 timeout",
            "Call to ibv_modify_qp failed: eno 22 invalid argument(Mismatched GID)",
        ]
    ):
        remove_cascaded_network_symptoms(symptoms, my_bad_nodes)

    # If SLURM cancels the job due to timeout and the length of the job is
    # longer than the time_limit, then any 'IB Error-' entries are not relevant.
    if "Cancelled by Timeout" in symptoms:
        if job_info is not None and job_info.time_limit is not None:
            if (job_info.interest_end_time - job_info.interest_start_time) > timedelta(minutes=job_info.time_limit):
                discard_cascaded_ib_symptoms(symptoms, ib_and_efa_errors_only=True)

    # On affected CSP fabrics, EAP certificates are good for 2 hours. If the job runs for less than 2 hours and never sees an
    # auth success, then we can't be certain if the failure was job fatal or not as the certificate may still be valid.
    # If the job runs longer than 2 hours, then we can be certain that the failure was job fatal.
    if "EAP authentication failed" in symptoms and job_info is not None and job_info.job_end_time is not None:
        elapsed_time_hours = (job_info.job_end_time - job_info.job_start_time).total_seconds() / 3600
        if elapsed_time_hours < 2:
            symptoms.discard("EAP authentication failed")

    if "cudaErrorInvalidValue" in symptoms:
        symptoms.discard("EFA Error-5")


@singlehooks.CancelledByCleanup.set_default
def cancelled_by_cleanup_default(job_info: JobInfo, symptoms: SymptomSet[str]) -> bool:
    return False


def remove_cascaded_network_symptoms(symptoms: SymptomSet[str], my_bad_nodes: JobNodeAttributions):
    to_discard = [
        "Connection reset by peer",
        "Connection closed",
        "PyTorch NCCL watchdog",
        "Software caused connection abort",
        "cudaErrorDevicesUnavailable",
    ]
    for symptom in to_discard:
        symptoms.discard(symptom)
    for _node, errors in my_bad_nodes.items():
        for symptom in to_discard:
            errors.discard(symptom)


def get_clean_bad_nodes(my_bad_nodes: JobNodeAttributions, cluster: str) -> JobNodeAttributions:
    non_node_compat = frozenset(["the process completes", "alias else"])
    nodes = list(my_bad_nodes.attributions.keys())  # a copy so we can delete while iterating
    for node in nodes:
        my_bad_nodes.attributions[node].attributions.discard_by_rule(non_node_compat)
        symptoms = list(my_bad_nodes.attributions[node].symptoms())
        for symptom in symptoms:
            if (
                "IB Error" in symptom
                or "local catastrophic error" in symptom
                or "CE deadlock DETECTED" in symptom
                or "PyTorch NCCL watchdog" in symptom
                or "pushrecv" in symptom
            ):
                my_bad_nodes.attributions[node].discard(symptom)
    return my_bad_nodes


def filter_xid_145_cascade_symptoms(
    symptoms: SymptomSet[str], my_bad_nodes: Optional[JobNodeAttributions] = None
) -> None:
    """Filter out XID-145 specific cascade symptoms.

    When XID-145 errors occur, they can cause cascading failures that show up as
    secondary symptoms. This function removes known XID-145 cascade symptoms
    from both job-level symptoms and per-node attributions.

    Args:
        symptoms: SymptomSet to filter (modified in place)
        my_bad_nodes: Per-node attributions to filter (modified in place)
    """
    cascade_symptoms_to_filter = [
        "cudaErrorLaunchFailure",
        "cudaErrorNvlinkUncorrectable",
        "XID-45-PREEMPTIVE_REMOVAL",
        "NV_ERR_INVALID_STATE",
    ]
    # RXPIPE-8 is treated as the primary symptom; SRC_TRACK variants are cascaded noise.
    if "XID-145-RLW_RXPIPE-8" in symptoms:
        cascade_symptoms_to_filter.extend(
            [
                "XID-145-RLW_SRC_TRACK-4",
                "XID-145-RLW_SRC_TRACK-8",
            ]
        )

    # Find symptoms to discard
    symptoms_to_discard = []
    for symptom in symptoms.symptoms():
        if any(cascade in symptom for cascade in cascade_symptoms_to_filter):
            symptoms_to_discard.append(symptom)

    # Discard cascade symptoms (using discard_seq for efficiency)
    if symptoms_to_discard:
        symptoms.discard_seq(symptoms_to_discard, purge=False)

    if my_bad_nodes is not None:
        my_bad_nodes.discard_matching_from_all_nodes(cascade_symptoms_to_filter)


def filter_cascade_symptoms_symptomset(
    symptoms: SymptomSet[str], my_bad_nodes: Optional[JobNodeAttributions] = None
) -> None:
    """Generic cascade symptom filter that dispatches to specific filtering functions.

    This function checks for various primary symptoms that can cause cascading failures
    and calls the appropriate specific filtering function for each type detected.

    Args:
        symptoms: SymptomSet to filter (modified in place)
        my_bad_nodes: Per-node attributions to filter (modified in place)
    """
    # Check for XID-145 and apply its specific cascade filtering
    has_xid_145 = any("XID-145" in symptom for symptom in symptoms.symptoms())
    if has_xid_145:
        filter_xid_145_cascade_symptoms(symptoms, my_bad_nodes)
