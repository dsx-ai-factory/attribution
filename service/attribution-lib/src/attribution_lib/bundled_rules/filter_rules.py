# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from common.nccl_watchdog import NcclWatchdogParseState, infer_process_group
from common.syslog_helpers import XidParserInfo
from common.xid_errors import parse_syslog_events
from loguru import logger

from attribution_lib.attr_types import EventHelper
from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import DemotionTypes, EventStream, sql_placeholders
from attribution_lib.rules import FilterContext, FilterPriority, declare_event_types, filter_def
from attribution_lib.sources import Source
from attribution_lib.symptoms import INFORMATIONAL_XIDS, RegisteredSymptoms

_DEMOTION_KEYS = frozenset(e.value for e in DemotionTypes)

# Constants for bad-to-interesting type mappings
_BAD_TYPES = ["Attribution.node.bad", "Attribution.node.bad_peers"]
_INTERESTING_TYPES = ["Attribution.node.interesting", "Attribution.node.interesting_peers"]

declare_event_types(*_BAD_TYPES, *_INTERESTING_TYPES, "XID_parser.info", "XID_parser.burst")


def _build_bad_to_interesting_type_map(event_stream: EventStream) -> dict[int, int]:
    """Build a mapping from bad event type IDs to interesting event type IDs.

    Returns:
        Dict mapping bad type IDs to their corresponding interesting type IDs.
    """
    return {
        event_stream.mgr.types.get_id(bad): event_stream.mgr.types.get_id(interesting)
        for bad, interesting in zip(_BAD_TYPES, _INTERESTING_TYPES)
    }


def _get_mfu_source_before_line(event_stream: EventStream, before_line: int):
    """Get the source of the last MFU event before a given line number."""
    mfu_type_id = event_stream.mgr.types.get_id("Application.mfu")

    # Use query_events_with_source and return first result
    for _eid, _evt_type, _payload, source in query_events_with_source(
        event_stream,
        mfu_type_id,
        additional_where="AND e.line_num < ?",
        params=[before_line],
        order_by="e.line_num DESC",
    ):
        return source
    return None


def _get_last_mfu_row(event_stream: EventStream, before_line: Optional[int] = None) -> Optional[tuple]:
    """Find the last MFU in the event stream and return (id, timestamp, line_num, payload_json)."""
    typ_id = event_stream.mgr.types.get_id("Application.mfu")

    if before_line is not None:
        rows = event_stream.query_raw(
            "SELECT id, timestamp, line_num, payload FROM events WHERE context_id=? AND event_type=? AND status=1 AND line_num<? ORDER BY line_num DESC LIMIT 1",
            [event_stream.ctx_id, typ_id, before_line],
        )
    else:
        rows = event_stream.query_raw(
            "SELECT id, timestamp, line_num, payload FROM events WHERE context_id=? AND event_type=? AND status=1 ORDER BY line_num DESC LIMIT 1",
            [event_stream.ctx_id, typ_id],
        )

    return rows[0] if rows else None


def _sql_demote_events(
    event_stream: EventStream,
    demote_key: str,
    where_clause: str,
    params: list,
    type_id_map: dict,
):
    """Demote events using SQL with minimal Python for JSON transformation.

    Args:
        event_stream: Event stream
        demote_key: JSON key for demotion rule (e.g., 'AFTER_MFU_DEMOTION')
        where_clause: SQL WHERE clause to select events to demote
        params: Parameters for WHERE clause
        type_id_map: Map of 'bad' type IDs to 'interesting' type IDs
    """
    # Build CASE statement for event type transformation
    type_cases = []
    for bad_id, interesting_id in type_id_map.items():
        type_cases.append(f"WHEN event_type={bad_id} THEN {interesting_id}")
    type_case_sql = " ".join(type_cases) if type_cases else "WHEN 1=0 THEN event_type"

    # Build IN clause for needs_transform check
    if type_id_map:
        in_clause = f"event_type IN ({','.join(map(str, type_id_map.keys()))})"
    else:
        in_clause = "1=0"

    # Flush interners and pending events to ensure all IDs exist
    event_stream._flush()

    # Get events to demote with type transformations done in SQL
    rows = event_stream.query_raw(
        f"""
        SELECT
            e.id,
            e.context_id,
            CASE {type_case_sql} ELSE e.event_type END as new_event_type,
            e.timestamp,
            e.line_num,
            e.source_datum,
            e.resources,
            e.obs_context_id,
            e.payload,
            json_extract_string(e.payload, '$.{demote_key}') as demote_rule,
            CASE WHEN {in_clause} THEN 1 ELSE 0 END as needs_transform
        FROM events e
        WHERE {where_clause}
        """,
        params,
    )

    if not rows:
        return

    # Step 1: Collect all unique rule values and batch insert/fetch them
    unique_rules = {row[9] for row in rows if row[9]}  # demote_rule_value is at index 9
    rule_id_map = {}

    if unique_rules:
        rule_list = list(unique_rules)
        # Batch insert new rules using flat tuple insert
        values_clause = ",".join(["(?)"] * len(rule_list))
        event_stream._conn.execute(
            f"INSERT INTO rules (name) VALUES {values_clause} ON CONFLICT (name) DO NOTHING",
            rule_list,
        )
        # Batch fetch all IDs in single query
        placeholders = sql_placeholders(len(rule_list))
        results = event_stream._conn.execute(
            f"SELECT name, id FROM rules WHERE name IN ({placeholders})",
            rule_list,
        ).fetchall()
        rule_id_map = {name: id for name, id in results}

    # Step 2: Process rows using the pre-fetched rule IDs
    insert_data = []
    event_ids_to_deactivate = []

    for row in rows:
        (
            orig_id,
            context_id,
            new_event_type,
            timestamp,
            line_num,
            source_datum,
            resources,
            obs_context_id,
            payload_json,
            demote_rule_value,
            needs_transform,
        ) = row

        # Skip events without a demote rule
        if not demote_rule_value:
            continue

        event_ids_to_deactivate.append(orig_id)

        # Look up pre-fetched rule ID (no query here!)
        source_rule = rule_id_map.get(demote_rule_value)

        # Only create a new event if this event type needs transformation
        if needs_transform:
            # Transform payload
            if payload_json:
                payload = json.loads(payload_json)
                new_payload = {k: v for k, v in payload.items() if k != demote_key}
                if "symptom" in new_payload:
                    new_payload["symptom"] = f"(non-fatal) {new_payload['symptom']}"
                payload_json = json.dumps(new_payload)

            # ID will be auto-generated by SQL
            insert_data.append(
                (
                    context_id,
                    new_event_type,
                    timestamp,
                    line_num,
                    source_datum,
                    source_rule,
                    resources,
                    obs_context_id,
                    payload_json,
                )
            )

    # Bulk insert - single executemany (IDs are auto-generated by SQL)
    if insert_data:
        event_stream._conn.executemany(
            """INSERT INTO events (context_id, event_type, timestamp, status, line_num, source_datum, source_rule,
                                    resources, obs_context_id, payload)
               VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
            insert_data,
        )

    # Bulk deactivate - single UPDATE
    if event_ids_to_deactivate:
        id_ph = sql_placeholders(len(event_ids_to_deactivate))
        event_stream.execute(
            f"UPDATE events SET status=0 WHERE context_id=? AND id IN ({id_ph})",
            [event_stream.ctx_id] + event_ids_to_deactivate,
        )


def _demote_bad_by_symptom(event_stream: EventStream, symptoms: frozenset[str], rule_name: str):
    """Demote bad events matching given symptoms to interesting.

    Uses the event stream API (event_adder) instead of raw SQL INSERT.
    Strips demotion keys from the payload; keeps symptom name unchanged.
    """
    bad_type_ids = event_stream.mgr.types.get_ids(_BAD_TYPES)
    if not bad_type_ids:
        return

    sym_ph = sql_placeholders(len(symptoms))
    additional_where = f"AND json_extract_string(payload, '$.symptom') IN ({sym_ph})"
    params = list(symptoms)

    event_adder = event_stream.event_adder()
    event_ids = []

    for eid, _evt_type, payload, source in query_events_with_source(
        event_stream, bad_type_ids, additional_where, params
    ):
        event_ids.append(eid)
        new_payload = {k: v for k, v in payload.items() if k not in _DEMOTION_KEYS}
        event_adder(
            "Attribution.node.interesting",
            new_payload,
            source.derived(rule=rule_name),
        )

    if event_ids:
        id_ph = sql_placeholders(len(event_ids))
        event_stream.execute(
            f"UPDATE events SET status=0 WHERE context_id=? AND id IN ({id_ph})",
            [event_stream.ctx_id] + event_ids,
        )


def _demote_bad_by_symptom_prefix(event_stream: EventStream, prefixes: list[str], rule_name: str):
    """Demote bad events whose symptom starts with any prefix to interesting.

    Args:
        event_stream: Event stream containing candidate bad events.
        prefixes: Literal symptom prefixes to demote.
        rule_name: Derived rule name attached to interesting events.
    """
    bad_type_ids = event_stream.mgr.types.get_ids(_BAD_TYPES)
    if not bad_type_ids:
        return

    prefix_clauses = " OR ".join(["starts_with(json_extract_string(payload, '$.symptom'), ?)"] * len(prefixes))
    additional_where = f"AND ({prefix_clauses})"

    event_adder = event_stream.event_adder()
    event_ids = []

    for eid, _evt_type, payload, source in query_events_with_source(
        event_stream, bad_type_ids, additional_where, prefixes
    ):
        event_ids.append(eid)
        new_payload = {k: v for k, v in payload.items() if k not in _DEMOTION_KEYS}
        event_adder("Attribution.node.interesting", new_payload, source.derived(rule=rule_name))

    if event_ids:
        id_ph = sql_placeholders(len(event_ids))
        event_stream.execute(
            f"UPDATE events SET status=0 WHERE context_id=? AND id IN ({id_ph})",
            [event_stream.ctx_id] + event_ids,
        )


# Filter rules (registered as plugins via @filter_def)


@filter_def(name="DemoteTflopsSymptoms", author="NVIDIA Attribution team")
def demote_tflops_symptoms(context: FilterContext):
    """Look for demotable bad events and demote them."""
    event_stream = context.event_stream

    last_mfu = _get_last_mfu_row(event_stream)
    if last_mfu is None:
        return

    _mfu_id, mfu_ts, mfu_ln, _mfu_payload = last_mfu

    # Build type ID mappings for bad->interesting transformations
    type_id_map = _build_bad_to_interesting_type_map(event_stream)

    # Get all type IDs we care about
    all_type_names = _BAD_TYPES + _INTERESTING_TYPES + ["XID_parser.info"]
    type_ids = event_stream.mgr.types.get_ids(all_type_names)
    if not type_ids:
        return

    type_ph = sql_placeholders(len(type_ids))
    where_clause = f"""
        context_id=? AND event_type IN ({type_ph}) AND status=1
        AND json_extract_string(payload, '$.AFTER_MFU_DEMOTION') IS NOT NULL
        AND ((timestamp IS NULL AND line_num<?) OR (timestamp IS NOT NULL AND timestamp<?))
    """
    params = [event_stream.ctx_id] + type_ids + [mfu_ln, mfu_ts]

    _sql_demote_events(event_stream, "AFTER_MFU_DEMOTION", where_clause, params, type_id_map)


@filter_def(name="DemoteWatchdogErrors", author="NVIDIA Attribution team")
def demote_watchdog_errors(context: FilterContext):
    """Demote certain events that are far away from watchdog events.

    If no watchdog: demote all NOT_NEAR_WATCHDOG_EVENT_DEMOTION events (unless dataloader should handle).
    If watchdog exists and no MFU events exist: demote all eligible events.
    If watchdog and MFU exist: demote only events not near the watchdog.
    """
    event_stream = context.event_stream

    lustre_read_timeout_itself = 30  # seconds

    # Get type IDs
    wd_type_id = event_stream.mgr.types.get_id("Application.pytorch.nccl_watchdog")
    type_ids = event_stream.mgr.types.get_ids(_BAD_TYPES)

    if not type_ids:
        return

    # Build type ID mappings
    type_id_map = _build_bad_to_interesting_type_map(event_stream)
    type_ph = sql_placeholders(len(type_ids))

    # Get watchdog event data in one query
    wd_rows = (
        event_stream.query_raw(
            "SELECT id, timestamp, line_num, payload FROM events WHERE context_id=? AND event_type=? AND status=1 ORDER BY line_num DESC LIMIT 1",
            [event_stream.ctx_id, wd_type_id],
        )
        if wd_type_id
        else []
    )

    if not wd_rows:
        # No watchdog: check if there's a dataloader event
        # If dataloader exists, let _demote_dataloader_errors handle those events
        # Otherwise, demote all (no watchdog = not near watchdog)
        dl_type_id = event_stream.mgr.types.get_id("Application.dataloader.exception")
        if dl_type_id:
            dl_rows = event_stream.query_raw(
                "SELECT 1 FROM events WHERE context_id=? AND event_type=? AND status=1 LIMIT 1",
                [event_stream.ctx_id, dl_type_id],
            )
            if dl_rows:
                # Dataloader exists - let dataloader logic handle demotion
                return

        # No watchdog and no dataloader: demote ALL events with NOT_NEAR_WATCHDOG_EVENT_DEMOTION
        where_clause = f"""
            context_id=? AND event_type IN ({type_ph}) AND status=1
            AND json_extract_string(payload, '$.NOT_NEAR_WATCHDOG_EVENT_DEMOTION') IS NOT NULL
        """
        params = [event_stream.ctx_id] + type_ids
        _sql_demote_events(event_stream, "NOT_NEAR_WATCHDOG_EVENT_DEMOTION", where_clause, params, type_id_map)
        return

    # Watchdog found: only demote events not near it
    _wd_id, _wd_ts, wd_ln, wd_payload_json = wd_rows[0]
    wd_payload = json.loads(wd_payload_json) if wd_payload_json else {}

    if _get_last_mfu_row(event_stream) is None:
        where_clause = f"""
            context_id=? AND event_type IN ({type_ph}) AND status=1
            AND json_extract_string(payload, '$.NOT_NEAR_WATCHDOG_EVENT_DEMOTION') IS NOT NULL
        """
        params = [event_stream.ctx_id] + type_ids
        _sql_demote_events(event_stream, "NOT_NEAR_WATCHDOG_EVENT_DEMOTION", where_clause, params, type_id_map)
        return

    # Get prior MFU source with correct timezone
    mfu_source = _get_mfu_source_before_line(event_stream, wd_ln)
    if mfu_source is None or mfu_source.time is None:
        return

    # Construct error_time from MFU timestamp components + watchdog time
    mfu_ts = mfu_source.time
    error_time = datetime.strptime(wd_payload["raw_error_time"], "%H:%M:%S").replace(
        year=mfu_ts.year,
        month=mfu_ts.month,
        day=mfu_ts.day,
        tzinfo=mfu_ts.tzinfo,
    )

    timeout_seconds = wd_payload["timeout_seconds"]

    # Get all candidate events
    rows = event_stream.query_raw(
        f"""SELECT e.id, e.timestamp
            FROM events e
            WHERE e.context_id=? AND e.event_type IN ({type_ph}) AND e.status=1
            AND json_extract_string(e.payload, '$.NOT_NEAR_WATCHDOG_EVENT_DEMOTION') IS NOT NULL""",
        [event_stream.ctx_id] + type_ids,
    )

    # Filter for events not near watchdog
    filtered_event_ids = []
    for eid, evt_ts in rows:
        if evt_ts is None:
            continue
        time_diff = (error_time - evt_ts).total_seconds()
        if not (0 < time_diff < timeout_seconds + lustre_read_timeout_itself):
            filtered_event_ids.append(eid)

    if filtered_event_ids:
        id_ph = sql_placeholders(len(filtered_event_ids))
        where_clause = f"""
            context_id=? AND id IN ({id_ph}) AND status=1
            AND json_extract_string(payload, '$.NOT_NEAR_WATCHDOG_EVENT_DEMOTION') IS NOT NULL
        """
        params = [event_stream.ctx_id] + filtered_event_ids

        _sql_demote_events(event_stream, "NOT_NEAR_WATCHDOG_EVENT_DEMOTION", where_clause, params, type_id_map)


@filter_def(name="DemoteDataloaderErrors", author="NVIDIA Attribution team")
def demote_dataloader_errors(context: FilterContext):
    """Demote events that occur before dataloader errors."""
    event_stream = context.event_stream
    job_info = context.job_info

    dl_type_id = event_stream.mgr.types.get_id("Application.dataloader.exception")
    mfu_type_id = event_stream.mgr.types.get_id("Application.mfu")
    type_ids = event_stream.mgr.types.get_ids(_BAD_TYPES)

    if not type_ids:
        return

    # Get first dataloader event
    dl_rows = event_stream.query_raw(
        "SELECT line_num FROM events WHERE context_id=? AND event_type=? AND status=1 ORDER BY line_num ASC LIMIT 1",
        [event_stream.ctx_id, dl_type_id],
    )
    if not dl_rows:
        return

    dl_ln = dl_rows[0][0]

    # Try to get MFU timestamp before dataloader (with correct timezone)
    last_time = job_info.interest_start_time
    if mfu_type_id:
        mfu_source = _get_mfu_source_before_line(event_stream, dl_ln)
        if mfu_source and mfu_source.time:
            last_time = mfu_source.time

    # Build type ID mappings
    type_id_map = _build_bad_to_interesting_type_map(event_stream)

    type_ph = sql_placeholders(len(type_ids))

    where_clause = f"""
        context_id=? AND event_type IN ({type_ph}) AND status=1 AND timestamp<=?
        AND json_extract_string(payload, '$.BEFORE_DATALOADER_DEMOTION') IS NOT NULL
    """
    params = [event_stream.ctx_id] + type_ids + [last_time]

    _sql_demote_events(event_stream, "BEFORE_DATALOADER_DEMOTION", where_clause, params, type_id_map)


@filter_def(
    name="ProcessCompletesCascade", author="NVIDIA Attribution team", derived_rules=["the process completes cascade"]
)
def process_completes_cascade(context: FilterContext):
    """Deactivate events with 'No such file or directory' after 'the process completes' events."""
    event_stream = context.event_stream

    warnings_to_discard = {"No such file or directory"}

    type_ids = event_stream.mgr.types.get_ids(["Attribution.node.bad", "Attribution.node.interesting"])
    if not type_ids:
        return

    type_ph = sql_placeholders(len(type_ids))

    # Find events with "the process completes" in payload
    rows = event_stream.query_raw(
        f"""SELECT id, line_num FROM events
            WHERE context_id=? AND event_type IN ({type_ph}) AND status=1
            AND CAST(payload AS VARCHAR) LIKE '%the process completes%'
            ORDER BY line_num ASC LIMIT 1""",
        [event_stream.ctx_id] + type_ids,
    )

    if not rows:
        return

    first_completes_id, first_completes_ln = rows[0]

    # Deactivate warning events after the first "completes"
    rule_id = event_stream.mgr.rules.get_id("the process completes cascade")

    for warning in warnings_to_discard:
        event_stream.execute(
            f"""UPDATE events SET status=0, source_rule=?
                WHERE context_id=? AND event_type IN ({type_ph}) AND status=1
                AND CAST(payload AS VARCHAR) LIKE ? AND line_num>?""",
            [rule_id, event_stream.ctx_id] + type_ids + [f"%{warning}%", first_completes_ln],
        )


@filter_def(
    name="HandleXidBurstEvents",
    author="NVIDIA Attribution team",
    priority=FilterPriority.EARLY,
    derived_rules=[
        "xid-64-pointer-error",
        "xid-64-event",
        "xid-63-event",
        "unknown-false-positive-xid-event",
        "xid-nvlink-event",
        "xid-event",
        "xid-event-outside-of-job-execution",
        "xid-burst-parser",
    ],
)
def handle_xid_burst_events(context: FilterContext):
    """Process XID burst events to produce XID parser info and node attributions.

    Reads XID_parser.burst events emitted by syslog extractors, feeds them through
    xid_errors.parse_syslog_events() for burst pattern analysis, then emits
    XID_parser.info events and bad/interesting node attributions.
    """
    event_stream = context.event_stream
    job_info = context.job_info

    typ_id = event_stream.mgr.types.get_id("XID_parser.burst")

    def _generator():
        for _eid, _evt_type, payload, source in query_events_with_source(event_stream, typ_id):
            if "node" not in payload:
                continue
            yield (source, payload["node"])

    generator = _generator()
    nodes_with_interruption = parse_syslog_events(generator, job_info)
    logger.info(nodes_with_interruption)
    for row in nodes_with_interruption:
        node = row["node"]
        sub_category = row["sub_category"]
        # add the event context to the source, this is to make the symptom consistent with the event
        nvlink_source = Source(
            row["source"].source_context,
            row["source"].line_num,
            row["source"].datum,
            row["source"].time,
            row["source"].resources,
            row["source"].rule,
        )
        event_adder = event_stream.event_adder()
        node_attr = EventHelper(event_adder, nvlink_source)

        # Create XidParserInfo with additional fields for NVLink XIDs
        xid_parser_info_kwargs = {"category": row["category"], "job_disruption": row["job_disruption"]}

        # Add new fields if they exist (for NVLINK XIDs)
        if "code" in row:
            xid_parser_info_kwargs["code"] = row["code"]
        if "subcode" in row:
            xid_parser_info_kwargs["subcode"] = row["subcode"]
        if "port_down_reason_code" in row:
            xid_parser_info_kwargs["port_down_reason_code"] = row["port_down_reason_code"]

        xid_parser_info = XidParserInfo(**xid_parser_info_kwargs)
        xid_pattern = row["xid_pattern"]
        payload = {
            "xid_parser_info": asdict(xid_parser_info),
            DemotionTypes.AFTER_MFU_DEMOTION: "xid-outside-of-job-execution",
        }
        event_adder(
            "XID_parser.info",
            payload,
            row["source"].derived(rule="xid-burst-parser"),
        )

        # Handle NVLink XIDs
        if row["category"] == "NvLink XID":
            if row.get("job_disruption") != "No":
                node_attr.bad(
                    node,
                    sub_category,
                    "xid-nvlink-event",
                    demotion={DemotionTypes.AFTER_MFU_DEMOTION: "xid-outside-of-job-execution"},
                )
                # Also add dynamic symptom to bad nodes for outcome calculation
                if row.get("dynamic_symptom"):
                    node_attr.bad(
                        node,
                        row["dynamic_symptom"],
                        "xid-nvlink-event",
                        demotion={DemotionTypes.AFTER_MFU_DEMOTION: "xid-outside-of-job-execution"},
                    )
        elif '64{"context":"internal pointer error"}' in xid_pattern:
            if "No" in xid_parser_info.job_disruption:
                node_attr.bad(
                    node,
                    "FalsePositive-XID-64",
                    "xid-64-pointer-error",
                    demotion={DemotionTypes.AFTER_MFU_DEMOTION: "xid-outside-of-job-execution"},
                )
            else:
                node_attr.bad(
                    node,
                    "DriverBug-XID-64",
                    "xid-64-event",
                    demotion={DemotionTypes.AFTER_MFU_DEMOTION: "xid-outside-of-job-execution"},
                )

        elif xid_parser_info.category == "False Positive":
            if "63" in xid_pattern:
                node_attr.bad(
                    node,
                    "FalsePositive-XID-63",
                    "xid-63-event",
                    demotion={DemotionTypes.AFTER_MFU_DEMOTION: "xid-outside-of-job-execution"},
                )
            else:
                node_attr.bad(
                    node,
                    f"FalsePositive-XID-{xid_pattern}",
                    "unknown-false-positive-xid-event",
                    demotion={DemotionTypes.AFTER_MFU_DEMOTION: "xid-outside-of-job-execution"},
                )

        if xid_parser_info.category == "Fallback":
            # TODO: lookup catalog
            pass
        # For NvLink XIDs, just use the sub_category
        if row["category"] == "NvLink XID":
            node_attr.interesting(
                node,
                sub_category,
                "xid-event",
                demotion={DemotionTypes.AFTER_MFU_DEMOTION: "xid-event-outside-of-job-execution"},
            )
        else:
            node_attr.interesting(
                node,
                f"{xid_parser_info.category}-{sub_category} : {xid_pattern}",
                "xid-event",
                demotion={DemotionTypes.AFTER_MFU_DEMOTION: "xid-event-outside-of-job-execution"},
            )


@filter_def(name="CleanInformationalXids", author="NVIDIA Attribution team", priority=FilterPriority.LATE)
def clean_informational_xids(context: FilterContext):
    """Demote informational-only XIDs from bad to interesting."""
    _demote_bad_by_symptom(context.event_stream, INFORMATIONAL_XIDS, "CleanInformationalXids")


@filter_def(name="DemoteNonFatalRowRemap", author="NVIDIA Attribution team", priority=FilterPriority.LATE)
def demote_nonfatal_row_remap(context: FilterContext):
    """Demote non-fatal row-remap telemetry to interesting.

    Args:
        context: Filter context containing the event stream.
    """
    _demote_bad_by_symptom(
        context.event_stream,
        frozenset((RegisteredSymptoms.ROW_REMAP.value,)),
        "DemoteNonFatalRowRemap",
    )


@filter_def(name="DemoteInsufficientPermissions", author="NVIDIA Attribution team", priority=FilterPriority.LATE)
def demote_insufficient_permissions(context: FilterContext):
    """Demote insufficient-permissions telemetry from bad to interesting.

    Args:
        context: Filter context containing the event stream.
    """
    symptom = "NV_ERR_INSUFFICIENT_PERMISSIONS"
    _demote_bad_by_symptom(context.event_stream, frozenset((symptom,)), "DemoteInsufficientPermissions")
    _demote_bad_by_symptom_prefix(
        context.event_stream,
        [f"{symptom} "],
        "DemoteInsufficientPermissions",
    )


@filter_def(name="DemoteBufferTooSmallJournal", author="NVIDIA Attribution team", priority=FilterPriority.LATE)
def demote_buffer_too_small_journal(context: FilterContext):
    """Demote RM journal-dump BUFFER_TOO_SMALL from bad to interesting.

    ``NV_ERR_BUFFER_TOO_SMALL`` at ``journal.c`` is ``rcdbDumpSystemInfo`` /
    ``prbEncNestedStart`` failing to encode a debug dump during CUDA process
    teardown. It is not a live GPU or driver fault. It appears on every node
    after SIGUSR1, scancel, or a rank crash and would otherwise outrank
    Cancelled by user (FALLBACK) because BUFFER_TOO_SMALL is MEDIUM_SW.

    Other ``NV_ERR_BUFFER_TOO_SMALL`` files are left as bad.

    Args:
        context: Filter context containing the event stream.
    """
    # Extractor emits "{error_code} {file}" without the line number, e.g.
    # "NV_ERR_BUFFER_TOO_SMALL journal.c". starts_with keeps any accidental suffix.
    _demote_bad_by_symptom_prefix(
        context.event_stream,
        ["NV_ERR_BUFFER_TOO_SMALL journal.c"],
        "DemoteBufferTooSmallJournal",
    )


@filter_def(name="DemoteNonFatalInvalidState", author="NVIDIA Attribution team", priority=FilterPriority.LATE)
def demote_nonfatal_invalid_state(context: FilterContext):
    """Demote the non-fatal osapi invalid-state symptom to interesting.

    Args:
        context: Filter context containing the event stream.
    """
    _demote_bad_by_symptom(
        context.event_stream,
        frozenset(("NV_ERR_INVALID_STATE osapi.c",)),
        "DemoteNonFatalInvalidState",
    )


@filter_def(name="FmNodeFailure", author="NVIDIA Attribution team", derived_rules=["fm-lost-connection-to-all-gpus"])
def fm_node_failure(context: FilterContext):
    """Aggregate 'Fabric manager lost connection to GPU' per node.

    If the number of distinct GPUs on the same node that lost their FM
    connection equals ``ranks_per_node`` (from the cluster config / job info),
    discard all individual symptoms and replace them with a single
    'Fabric manager lost connection to all GPUs' symptom.

    This indicates a likely node failure.
    """
    event_stream = context.event_stream
    job_info = context.job_info
    if job_info is None:
        return

    ranks_per_node = int(job_info.RANKS_PER_NODE)
    if ranks_per_node <= 0:
        return

    typ_id = event_stream.mgr.types.get_id("FM_parser.lost_gpu_connection")

    # Keep one representative source per distinct GPU GUID so every lost GPU's line is retained.
    node_gpu_sources: dict[str, dict[str, Source]] = {}
    for _eid, _evt_type, payload, source in query_events_with_source(event_stream, typ_id):
        node = payload.get("node")
        gpu_guid = payload.get("gpu_guid")
        if not node or not gpu_guid:
            continue
        node_gpu_sources.setdefault(node, {}).setdefault(gpu_guid, source)

    # Identify nodes where all GPUs lost FM connection
    nodes_to_aggregate = {
        node for node, guid_sources in node_gpu_sources.items() if len(guid_sources) == ranks_per_node
    }
    if not nodes_to_aggregate:
        return

    # Deactivate individual "Fabric manager lost connection to GPU" bad events
    bad_type_ids = event_stream.mgr.types.get_ids(_BAD_TYPES)
    if not bad_type_ids:
        return

    node_list = list(nodes_to_aggregate)
    node_ph = sql_placeholders(len(node_list))
    event_ids: list[int] = []
    for eid, _evt_type, _payload, _source in query_events_with_source(
        event_stream,
        bad_type_ids,
        additional_where=(
            f"AND json_extract_string(payload, '$.symptom') = ? "
            f"AND json_extract_string(payload, '$.node') IN ({node_ph})"
        ),
        params=["Fabric manager lost connection to GPU"] + node_list,
    ):
        event_ids.append(eid)

    if event_ids:
        id_ph = sql_placeholders(len(event_ids))
        event_stream.execute(
            f"UPDATE events SET status=0 WHERE context_id=? AND id IN ({id_ph})",
            [event_stream.ctx_id] + event_ids,
        )

    # Aggregated symptom per affected node, emitted once per GPU so all lines survive as evidence.
    event_adder = event_stream.event_adder()
    for node in nodes_to_aggregate:
        for source in node_gpu_sources[node].values():
            event_adder(
                "Attribution.node.bad",
                {"node": node, "symptom": "Fabric manager lost connection to all GPUs"},
                source.derived(rule="fm-lost-connection-to-all-gpus"),
            )


@filter_def(
    name="NCCLLateArrivalNodes",
    priority=FilterPriority.STANDARD,
    derived_event_types=["Application.NCCLWatchdog.report"],
)
def nccl_late_arrival_nodes(context: FilterContext) -> None:
    """Identify NCCL late-arrival (straggler) nodes and emit as interesting nodes.

    Reads watchdog timeout data from Application.NCCLWatchdog.report events. Computes
    the complement to find which ranks (and nodes) never arrived at the collective.

    Known non-handled cases (heuristic limitations):
    - Contiguous small process groups (e.g., intra-node allreduce):
        Ranks {0..7} reporting timeouts would be indistinguishable from a global
        allreduce operation.
    - Two independent single-GPU stragglers in different process groups simultaneously:
        Combined reported ranks break stride analysis and fall back to global.
    All of these cases would cause the straggler(s) to be ignored by the size guard.
    """
    event_stream, job_info = context.event_stream, context.job_info
    if job_info is None or not job_info.nodes or job_info.RANKS_PER_NODE <= 0:
        return

    ranks_per_node = int(job_info.RANKS_PER_NODE)
    world_size = len(job_info.nodes) * ranks_per_node
    if world_size == 0:
        return

    watchdog_reports = NcclWatchdogParseState.collect(event_stream)
    if len(watchdog_reports) == 0:
        return

    # There should only be one NCCL watchdog report per job. Take the first one.
    watchdog_report = watchdog_reports[0]
    reported_ranks: set[int] = set()
    for timeouts in watchdog_report.timeouts_by_key.values():
        reported_ranks.update(timeouts.ranks)
    if len(reported_ranks) == 0:
        return

    expected_ranks = infer_process_group(reported_ranks, world_size)
    late_ranks = sorted(expected_ranks - reported_ranks)
    if len(late_ranks) == 0:
        return

    missing_nodes: defaultdict[str, list[int]] = defaultdict(list)
    for rank in late_ranks:
        if job_info.rank_exists(rank):
            missing_nodes[job_info.get_name_from_rank(rank)].append(rank)

    # Apply a size guard before emitting.
    # The straggler pattern (one slow/dead node blocking a collective) only makes sense when a small fraction of nodes are missing.
    #   i.e. bulk cancellations or hardware failures across many nodes should not be treated as stragglers.
    # Threshold: max(1, 1% of nodes). The 1% value helps scale for larger jobs.
    num_nodes_reporting_threshold = max(1, int(0.01 * len(job_info.nodes)))
    if len(missing_nodes) > num_nodes_reporting_threshold:
        logger.debug(
            f"NCCLLateArrivalNodes: skipping - {len(missing_nodes)} missing nodes"
            f" exceeds {num_nodes_reporting_threshold} node reporting threshold"
        )
        return

    event_adder = event_stream.event_adder()
    for node, ranks in missing_nodes.items():
        event_adder(
            "Attribution.node.interesting",
            {"node": node, "symptom": "NCCL Late Arrival", "ranks": ranks},
            watchdog_report.source.derived(rule="NCCLLateArrivalNodes"),
        )
