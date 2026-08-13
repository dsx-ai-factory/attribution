# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from datetime import datetime
from typing import Any, Optional

from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import EventStream
from attribution_lib.job_info import JobInfo
from attribution_lib.sources import SourceSet


class NCCLCudaVersion:
    """Class to help collect NCCL+Cuda Version from event stream (post-processing)."""

    @classmethod
    def collect(cls, event_stream: EventStream) -> dict[str, SourceSet[str]]:
        typ_id = event_stream.mgr.types.get_id("Application.nccl.version")

        res: dict[str, SourceSet[str]] = {}

        # We only need the first event, so use ORDER BY with a manual limit
        # Note: query_events_with_source returns an iterator, so we can just take the first item
        for eid, evt_type, payload, source in query_events_with_source(event_stream, typ_id, order_by="e.line_num ASC"):
            if source is not None:
                if (nccl_version := payload.get("nccl_version")) is not None:
                    res["nccl_version"] = SourceSet(nccl_version).add_interesting(source)
                if (cuda_version := payload.get("cuda_version")) is not None:
                    res["cuda_version"] = SourceSet(cuda_version).add_interesting(source)
            # Only process the first event
            break

        return res


class WatchdogFoundValues:
    """Helper container for watchdog values found during attribution."""

    @classmethod
    def collect(cls, event_stream: EventStream, job_info: JobInfo) -> Optional[SourceSet[dict[str, datetime | Any]]]:
        ## Might be better to have emit a pseudo-mfu at job start time for this, but we'll just use jobinfo for now.
        mfu_typ = event_stream.mgr.types.get_id("Application.mfu")
        watchdog_typ = event_stream.mgr.types.get_id("Application.pytorch.nccl_watchdog")

        type_ids = [mfu_typ, watchdog_typ]

        last_time = job_info.source_value(
            job_info.interest_start_time, datum="<JobInfo>", rule="jobinfo extract watchdog time"
        )

        # Need to process events in order since we track last_time across MFU events
        for eid, evt_type, payload, source in query_events_with_source(
            event_stream, type_ids, order_by="e.line_num ASC"
        ):
            if not payload:
                continue

            if evt_type == mfu_typ:
                if source is not None:
                    # Parse datetime from ISO string stored in payload
                    mfu_datetime = datetime.fromisoformat(payload["datetime"])
                    last_time = SourceSet(mfu_datetime).add_interesting(
                        source.derived(rule="mfu extract watchdog time")
                    )
            elif evt_type == watchdog_typ:
                if last_time is not None and source is not None:
                    error_time = datetime.strptime(payload["raw_error_time"], "%H:%M:%S").replace(
                        year=last_time.value().year,
                        month=last_time.value().month,
                        day=last_time.value().day,
                        tzinfo=last_time.value().tzinfo,
                    )

                    watchdog = {"error_time": error_time, "timeout_seconds": payload["timeout_seconds"]}
                    last_time_source = last_time.value_mapped_copy(
                        lambda x: watchdog
                    )  # these are some gymastics to enable us to merge the sources for the mfu/jobinfo time with the sources for the watchdog event
                    result = SourceSet(watchdog).add_interesting(source).update(last_time_source)
                    return result
        return None
