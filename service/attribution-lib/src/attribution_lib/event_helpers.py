# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper utilities for efficiently querying events with full context.

This module provides reusable abstractions for common patterns in collect()
functions, eliminating code duplication and ensuring consistent optimization.
"""

import json
from collections.abc import Iterator
from typing import Optional
from zoneinfo import ZoneInfo

from attribution_lib.events import EventStream
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import (
    Format,
    Id,
    Source as ObsSource,
    TimeInterval,
)
from attribution_lib.resources import AllJobResources
from attribution_lib.resources.resource_set import validate_resource_set
from attribution_lib.sources import Source

# ============================================================================
# EventQueryHelper class
# ============================================================================


class EventQueryHelper:
    """Helper for efficiently querying events with full context (observation context, rules, etc)."""

    def __init__(self, event_stream: EventStream):
        self.event_stream = event_stream
        # Cache for reconstructed observation contexts
        self._obs_context_cache: dict[int, ObservationContext] = {}

    def _get_obs_source_str(self, obs_source_id: int) -> Optional[str]:
        """Get observation source string using EventManager's cached lookup."""
        return self.event_stream.mgr._obs_sources.get_str(obs_source_id)

    def _get_obs_format_str(self, obs_format_id: int) -> Optional[str]:
        """Get observation format string using EventManager's cached lookup."""
        return self.event_stream.mgr._obs_formats.get_str(obs_format_id)

    def query_events_with_context(
        self,
        event_type_ids: int | list[int],
        additional_where: str = "",
        params: Optional[list] = None,
        order_by: str = "",
    ) -> list[tuple]:
        """Execute a standard JOIN query for events with observation contexts and rules.

        Args:
            event_type_ids: Single event type ID or list of event type IDs to query
            additional_where: Additional WHERE clause conditions (e.g., "AND e.timestamp > ?")
            params: Additional query parameters for the additional_where clause
            order_by: ORDER BY clause (default: "" - no ordering for better performance)

        Returns:
            List of tuples with standardized columns:
            (eid, event_type, payload_json, timestamp, line_num, datum, rule,
             resources_str, obs_id, obs_time_start, obs_time_end,
             obs_source_id, obs_format_id, obs_timezone_str)
        """
        if isinstance(event_type_ids, int):
            event_type_ids = [event_type_ids]

        if not event_type_ids:
            return []

        type_ph = ",".join(["?"] * len(event_type_ids))
        order_clause = f"ORDER BY {order_by}" if order_by else ""
        query = f"""
            SELECT
                e.id, e.event_type, e.payload, e.timestamp, e.line_num, e.source_datum, r.name,
                e.resources,
                oc.id, oc.time_start, oc.time_end, oc.source, oc.format, oc.timezone
            FROM events e
            LEFT JOIN observation_contexts oc ON e.obs_context_id = oc.id
            LEFT JOIN rules r ON e.source_rule = r.id
            WHERE e.context_id=? AND e.event_type IN ({type_ph}) AND e.status=1
            {additional_where}
            {order_clause}
        """

        query_params = [self.event_stream.ctx_id] + event_type_ids
        if params:
            query_params.extend(params)

        return self.event_stream.query_raw(query, query_params)

    def iter_events_with_source(
        self,
        event_type_ids: int | list[int],
        additional_where: str = "",
        params: Optional[list] = None,
        order_by: str = "",
    ) -> Iterator[tuple[int, int, dict, Source]]:
        """Iterate over events, yielding (event_id, event_type, payload, source) tuples.

        This is a high-level convenience method that handles all the boilerplate:
        - Querying with JOINs
        - Reconstructing observation contexts (with caching)
        - Reconstructing sources

        Args:
            event_type_ids: Single event type ID or list of event type IDs to query
            additional_where: Additional WHERE clause conditions
            params: Additional query parameters
            order_by: ORDER BY clause (default: ""---no ordering). Specify "e.line_num ASC" if order matters.

        Yields:
            Tuples of (event_id, event_type, payload_dict, source)
        """
        rows = self.query_events_with_context(event_type_ids, additional_where, params, order_by)

        for row in rows:
            (
                eid,
                event_type,
                payload_json,
                timestamp,
                line_num,
                datum,
                rule,
                resources_str,
                obs_id,
                obs_time_start,
                obs_time_end,
                obs_source_id,
                obs_format_id,
                obs_timezone_str,
            ) = row

            # Parse payload (simple JSON types only)
            payload = json.loads(payload_json) if payload_json else {}

            # Reconstruct source
            source = self._reconstruct_source(
                timestamp=timestamp,
                line_num=line_num,
                datum=datum,
                rule=rule,
                resources_str=resources_str,
                obs_id=obs_id,
                obs_time_start=obs_time_start,
                obs_time_end=obs_time_end,
                obs_source_id=obs_source_id,
                obs_format_id=obs_format_id,
                obs_timezone_str=obs_timezone_str,
            )

            yield eid, event_type, payload, source

    def _reconstruct_source(
        self,
        timestamp,
        line_num: Optional[int],
        datum: Optional[str],
        rule: Optional[str],
        resources_str: Optional[str],
        obs_id: int,
        obs_time_start,
        obs_time_end,
        obs_source_id: int,
        obs_format_id: int,
        obs_timezone_str: str,
    ) -> Source:
        """Reconstruct a Source object from query row data."""
        # Reconstruct resources using Pydantic validator
        if resources_str:
            resources = validate_resource_set(json.loads(resources_str))
        else:
            resources = AllJobResources

        # Get or reconstruct observation context (cached)
        source_context = None
        if obs_id is not None:
            if obs_id not in self._obs_context_cache:
                obs_source_str = self._get_obs_source_str(obs_source_id)
                obs_format_str = self._get_obs_format_str(obs_format_id)

                time_interval = TimeInterval(obs_time_start, obs_time_end)
                obs_source = ObsSource(obs_source_str)
                obs_format = Format(obs_format_str)
                obs_timezone = ZoneInfo(obs_timezone_str)

                self._obs_context_cache[obs_id] = ObservationContext(
                    time_interval=time_interval,
                    source=obs_source,
                    format=obs_format,
                    timezone=obs_timezone,
                    id=Id(obs_id),
                )
            source_context = self._obs_context_cache[obs_id]

        # Convert timestamp to observation timezone
        if timestamp is not None and source_context and source_context.timezone and timestamp.tzinfo:
            timestamp = timestamp.astimezone(source_context.timezone)

        return Source(
            source_context=source_context,
            line_num=line_num,
            datum=datum or "",
            time=timestamp,
            resources=resources,
            rule=rule,
        )


def query_events_with_source(
    event_stream: EventStream,
    event_type_ids: int | list[int],
    additional_where: str = "",
    params: Optional[list] = None,
    order_by: str = "",
) -> Iterator[tuple[int, int, dict, Source]]:
    """Convenience function to iterate over events with full context.

    This is a module-level wrapper around EventQueryHelper.iter_events_with_source()
    for ease of use.

    Args:
        event_stream: The EventStream to query
        event_type_ids: Single event type ID or list of event type IDs to query
        additional_where: Additional WHERE clause conditions
        params: Additional query parameters
        order_by: ORDER BY clause

    Yields:
        Tuples of (event_id, event_type, payload_dict, source)

    Example:
        ```python
        mfu_id = event_stream.mgr.types.get_id("Application.mfu")
        for eid, evt_type, payload, source in query_events_with_source(
            event_stream, mfu_id
        ):
            print(f"Event {eid}: {payload['datetime']} from {source.rule}")
        ```
    """
    helper = EventQueryHelper(event_stream)
    yield from helper.iter_events_with_source(event_type_ids, additional_where, params, order_by)
