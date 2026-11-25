# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Mapping

from loguru import logger
from requests.adapters import DEFAULT_POOLSIZE, HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.connectors.cluster import query_cluster
from attribution_blueprint.connectors.logs import log_observation
from attribution_blueprint.connectors.logs.envelope import extract_body_and_node
from attribution_blueprint.connectors.logs.victorialogs.templates import TEMPLATES
from attribution_blueprint.exceptions import PartialCollectionError
from attribution_blueprint.telemetry import MeteredSession, QueryTruncation, record_paginated_query
from attribution_blueprint.types import (
    ExecutionContext,
    NodeResource,
    Observation,
    ReplicaResource,
    Source,
    TimeseriesEntry,
    assert_aware,
    context_run_id,
)

DEFAULT_LIMIT = 5000
DEFAULT_NODE_FIELD = "host_name"
DEFAULT_QUERY_CONCURRENCY = 1
MAX_PAGES = 20
QUERY_PATH = "/select/logsql/query"
_NS_PER_SECOND = 1_000_000_000

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$")
_BARE_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _timestamp_ns(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = _TIMESTAMP.fullmatch(value)
    if match is None:
        return None
    base, fraction, zone = match.groups()
    offset = "+00:00" if zone == "Z" else zone
    try:
        timestamp = datetime.fromisoformat(f"{base}{offset}")
    except ValueError:
        return None
    delta = timestamp - _EPOCH
    seconds = delta.days * 86400 + delta.seconds
    fractional_ns = int((fraction or "").ljust(9, "0") or "0")
    return seconds * 1_000_000_000 + fractional_ns


def _datetime_ns(value: datetime) -> int:
    delta = value - _EPOCH
    return (delta // timedelta(microseconds=1)) * 1000


@dataclass(frozen=True, slots=True)
class _WindowRead:
    """One rendered query over one slice of the collection window."""

    label: str
    query: str
    start_ns: int
    end_ns_excl: int


class VictoriaLogsConnector:
    """Collect timestamped LogsQL rows under a finite page budget per query and window slice."""

    def __init__(
        self,
        *,
        endpoint: str,
        source: Source,
        auth: Auth | None = None,
        query_kwargs: Mapping[str, str] | None = None,
        templates: Mapping[str, list[str]] = TEMPLATES,
        timeout_seconds: float = 60.0,
        limit: int = DEFAULT_LIMIT,
        max_pages: int = MAX_PAGES,
        node_field: str | None = DEFAULT_NODE_FIELD,
        cluster_from_context: bool = False,
        query_concurrency: int = DEFAULT_QUERY_CONCURRENCY,
        window_slice_seconds: float | None = None,
    ):
        """Configure queries, connection security, row limits, page budgets and request concurrency.

        Args:
            endpoint: VictoriaLogs base URL.
            source: Evidence source the templates collect.
            auth: Optional request authentication.
            query_kwargs: Values for template placeholders.
            templates: LogsQL templates by source.
            timeout_seconds: HTTP timeout per request.
            limit: Maximum rows per response.
            max_pages: Maximum responses per query, or per slice when the window is sliced.
            node_field: Row field holding the node name, or None to skip node filtering.
            cluster_from_context: Take `cluster` from each context when it has one.
            query_concurrency: Maximum requests one collection runs at the same time.
            window_slice_seconds: Split the collection window into slices of this length,
                each read and paged on its own, or None to read the window as one range.

        Raises:
            ValueError: A setting is out of range.
        """
        if source not in templates:
            raise ValueError(f"no templates registered for source {source}")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        if type(max_pages) is not int or max_pages <= 0:
            raise ValueError("max_pages must be a positive integer")
        if type(query_concurrency) is not int or query_concurrency <= 0:
            raise ValueError("query_concurrency must be a positive integer")
        self._window_slice_ns = _window_slice_ns(window_slice_seconds)
        self._query_concurrency = query_concurrency
        self._max_pages = max_pages
        self._source = source
        self._templates = list(templates[source])
        self._query_kwargs = dict(query_kwargs or {})
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_seconds
        self._limit = limit
        self._node_field = node_field or None
        self._cluster_from_context = cluster_from_context
        self._auth = auth

        self._session = MeteredSession("victorialogs", (QUERY_PATH,))
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=0.5,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
        )
        # Keep a connection per concurrent request so parallel reads reuse TLS sessions.
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=max(DEFAULT_POOLSIZE, query_concurrency))
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        if auth is not None:
            kwargs = auth.request_kwargs()
            if "headers" in kwargs:
                self._session.headers.update(kwargs["headers"])
            if "cert" in kwargs:
                self._session.cert = kwargs["cert"]
            if "verify" in kwargs:
                self._session.verify = kwargs["verify"]

    @property
    def source(self) -> Source:
        """Return the configured evidence source."""
        return self._source

    def collect(self, context: ExecutionContext) -> list[Observation]:
        """Collect deduplicated entries within the workload window.

        Args:
            context: Workload identity, nodes, cluster and collection interval.

        Returns:
            Timestamp-ordered observations without provider-specific context fields.

        Raises:
            PartialCollectionError: Paging or a configured query was incomplete; retains available observations.
            RequestException: Every attempted query failed without returning a response.
        """
        start = assert_aware(context["start_time"], "start_time")
        end = assert_aware(context["end_time"], "end_time")
        start_ns = _datetime_ns(start)
        end_ns = _datetime_ns(end) + 1
        rows, issues = self._collect_queries(context, start_ns, end_ns)
        entries, skipped = _rows_to_entries(rows, self._node_field)
        if skipped:
            issues.append(f"skipped {skipped} row(s) with invalid _time or non-string _msg")
        observations = [log_observation(source=self._source, entries=entries, context=context)] if entries else []
        if issues:
            raise PartialCollectionError("; ".join(dict.fromkeys(issues)), observations)
        return observations

    def _collect_queries(
        self, context: ExecutionContext, start_ns: int, end_ns_excl: int
    ) -> tuple[list[dict], list[str]]:
        """Try every configured query and retain evidence independently of query order.

        Args:
            context: Provider identity, nodes and query arguments.
            start_ns: Inclusive collection start.
            end_ns_excl: Exclusive collection end.

        Returns:
            Deduplicated rows and incomplete-coverage reasons.

        Raises:
            RequestException: All requests failed; preserves the first failure.
        """
        nodes = context.get("nodes") or []
        query_kwargs = self._context_query_kwargs(context)
        rows: list[dict] = []
        seen: set[str] = set()
        issues: list[str] = []
        request_error: RequestException | None = None
        received_response = False
        slices = self._window_slices(start_ns, end_ns_excl)
        reads: list[_WindowRead] = []
        for index, template in enumerate(self._templates, start=1):
            try:
                query = template.format(
                    job_id=context_run_id(context),
                    **query_kwargs,
                )
            except KeyError as error:
                issues.append(f"query {index}: template missing kwarg {error}; skipped")
                logger.warning(f"victorialogs template missing kwarg {error}; skipping query")
                continue
            query = self._inject_node_filter(query, nodes)
            for number, (slice_start, slice_end) in enumerate(slices, start=1):
                label = f"query {index}" if len(slices) == 1 else f"query {index} slice {number}/{len(slices)}"
                reads.append(_WindowRead(label, query, slice_start, slice_end))
        # Results come back in read order, so the first error and the row order do not depend on timing.
        for read, result in zip(reads, self._run_reads(reads), strict=True):
            if isinstance(result, RequestException):
                if request_error is None:
                    request_error = result
                issues.append(f"{read.label}: {_request_failure_reason(result)}")
                continue
            query_rows, query_issues = result
            received_response = True
            issues.extend(f"{read.label}: {issue}" for issue in query_issues)
            for row in query_rows:
                key = json.dumps(row, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

        if request_error is not None and not received_response:
            raise request_error
        return rows, issues

    def _window_slices(self, start_ns: int, end_ns_excl: int) -> list[tuple[int, int]]:
        """Split the collection window into the fewest equal contiguous slices, newest first.

        Equal slices avoid a sliver at one end: the exclusive end is one nanosecond past the
        inclusive window, which would otherwise cost a request for a one-nanosecond slice.

        Args:
            start_ns: Inclusive collection start.
            end_ns_excl: Exclusive collection end.

        Returns:
            Inclusive-start, exclusive-end bounds, none longer than the inclusive window
            divided by the slice count.
        """
        span = end_ns_excl - start_ns
        if self._window_slice_ns is None or span - 1 <= self._window_slice_ns:
            return [(start_ns, end_ns_excl)]
        count = -(-(span - 1) // self._window_slice_ns)
        bounds = [start_ns + span * index // count for index in range(count + 1)]
        return [(bounds[index], bounds[index + 1]) for index in reversed(range(count))]

    def _run_reads(self, reads: list[_WindowRead]) -> list[tuple[list[dict], list[str]] | RequestException]:
        """Read every query slice, up to `query_concurrency` at a time.

        Args:
            reads: Query slices in reporting order.

        Returns:
            Each read's rows and issues, or the request error that left it without rows, in read order.
        """
        if self._query_concurrency == 1 or len(reads) <= 1:
            return [self._read_window(read) for read in reads]
        workers = min(self._query_concurrency, len(reads))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="victorialogs") as pool:
            # Each read runs in a copy of the caller's context, so context-bound log fields reach its log lines.
            futures = [pool.submit(contextvars.copy_context().run, self._read_window, read) for read in reads]
            return [future.result() for future in futures]

    def _read_window(self, read: _WindowRead) -> tuple[list[dict], list[str]] | RequestException:
        """Read one query slice, returning a first-request failure instead of raising it.

        The caller reports a failed slice and keeps the others, so the failure is a result here.

        Args:
            read: Query slice to read.

        Returns:
            Rows and issues, or the request error that left the slice without rows.
        """
        try:
            return self._query_window(read.query, read.start_ns, read.end_ns_excl)
        except RequestException as error:
            return error

    def _context_query_kwargs(self, context: ExecutionContext) -> dict[str, str]:
        """Return query_kwargs with `cluster` taken from the context when enabled and present."""
        cluster = query_cluster(context, self._query_kwargs.get("cluster"), from_context=self._cluster_from_context)
        if cluster is None or cluster == self._query_kwargs.get("cluster"):
            return self._query_kwargs
        return {**self._query_kwargs, "cluster": cluster}

    def _read_page(self, query: str, start_ns: int, end_ns_excl: int) -> tuple[list[dict], int]:
        """Fetch and decode one bounded LogsQL response.

        Args:
            query: Rendered LogsQL query.
            start_ns: Inclusive window start in nanoseconds.
            end_ns_excl: Exclusive window end in nanoseconds.

        Returns:
            Decoded object rows and the count of invalid non-empty records.
            Transport errors propagate.
        """
        response = self._session.post(
            f"{self._endpoint}{QUERY_PATH}",
            data={
                "query": query,
                "start": start_ns,
                "end": end_ns_excl,
                "limit": self._limit,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        page: list[dict] = []
        malformed = 0
        for line in response.text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, RecursionError):
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue
            page.append(row)
        if malformed:
            logger.warning(f"victorialogs: skipped {malformed} invalid NDJSON line(s)")
        return page, malformed

    def _query_window(self, query: str, start_ns: int, end_ns_excl: int) -> tuple[list[dict], list[str]]:
        """Page backward within the budget, preserving reasons for incomplete coverage.

        Args:
            query: Rendered LogsQL query.
            start_ns: Inclusive window start in nanoseconds.
            end_ns_excl: Exclusive window end in nanoseconds.

        Returns:
            Available rows and any reasons paging could not prove complete coverage.

        Raises:
            RequestException: The first request failed, leaving no rows from this query.
        """
        rows: list[dict] = []
        issues: list[str] = []
        truncations: set[QueryTruncation] = set()
        cur_end = end_ns_excl
        for pages in range(1, self._max_pages + 1):
            try:
                page, skipped = self._read_page(query, start_ns, cur_end)
            except RequestException as error:
                if rows == []:
                    raise
                issues.append(_request_failure_reason(error))
                truncations.add(QueryTruncation.REQUEST_ERROR)
                break
            if skipped:
                issues.append(f"skipped {skipped} invalid NDJSON record(s)")
            if not page:
                if skipped >= self._limit:
                    truncations.add(QueryTruncation.MISSING_TIMESTAMP)
                break
            rows.extend(page)
            if len(page) + skipped < self._limit:
                break
            timestamps = [timestamp for row in page if (timestamp := _timestamp_ns(row.get("_time"))) is not None]
            if not timestamps:
                issues.append("saturated page has no parseable _time; stopping pagination")
                logger.warning(f"victorialogs: {issues[-1]}")
                truncations.add(QueryTruncation.MISSING_TIMESTAMP)
                break
            oldest = min(timestamps)
            if oldest <= start_ns:
                issues.append("saturated page reaches window start; boundary records may be truncated")
                truncations.add(QueryTruncation.WINDOW_START)
                break
            next_end = oldest + 1
            if next_end >= cur_end:
                issues.append(f"possible truncation at {_format_timestamp_ns(oldest)}")
                logger.warning(f"victorialogs: {issues[-1]}")
                truncations.add(QueryTruncation.TIMESTAMP_TIE)
                next_end = oldest
            cur_end = next_end
        else:
            issues.append(
                f"page limit of {self._max_pages} reached; older entries before ts={cur_end} may be truncated"
            )
            logger.warning(f"victorialogs: {issues[-1]}")
            truncations.add(QueryTruncation.PAGE_LIMIT)
        record_paginated_query("victorialogs", QUERY_PATH, entries=len(rows), pages=pages, truncations=truncations)
        return rows, issues

    def _inject_node_filter(self, query: str, nodes: list[str]) -> str:
        if self._node_field is None or not nodes:
            return query
        field = (
            self._node_field
            if _BARE_FIELD.fullmatch(self._node_field)
            else json.dumps(self._node_field, ensure_ascii=False)
        )
        values = ",".join(json.dumps(node, ensure_ascii=False) for node in nodes)
        filter_expr = f"{field}:in({values})"
        head, pipes = _split_top_level_pipe(query)
        filters = head.strip()
        rendered = f"{filter_expr} ({filters})" if filters else filter_expr
        return f"{rendered} {pipes}" if pipes else rendered


def _split_top_level_pipe(query: str) -> tuple[str, str]:
    quote: str | None = None
    depth = 0
    index = 0
    while index < len(query):
        char = query[index]
        if quote is not None:
            if quote != "`" and char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in ('"', "'", "`"):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "|" and depth == 0:
            return query[:index], query[index:].strip()
        index += 1
    return query, ""


def _window_slice_ns(window_slice_seconds: float | None) -> int | None:
    """Validate the slice length and convert it to nanoseconds.

    Args:
        window_slice_seconds: Slice length in seconds, or None for no slicing.

    Returns:
        Slice length in nanoseconds, or None.

    Raises:
        ValueError: The length is not a finite positive number of at least one nanosecond.
    """
    if window_slice_seconds is None:
        return None
    if isinstance(window_slice_seconds, bool) or not isinstance(window_slice_seconds, int | float):
        raise ValueError("window_slice_seconds must be a positive number")
    slice_ns = round(window_slice_seconds * _NS_PER_SECOND) if math.isfinite(window_slice_seconds) else 0
    if slice_ns <= 0:
        raise ValueError("window_slice_seconds must be a positive number")
    return slice_ns


def _format_timestamp_ns(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    timestamp = datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{timestamp}.{nanoseconds:09d}Z"


def _request_failure_reason(error: RequestException) -> str:
    """Describe a request failure without copying URLs, headers or response bodies.

    Args:
        error: The failed transport operation.

    Returns:
        Exception type and HTTP status when the server supplied a response.
    """
    status = error.response.status_code if error.response is not None else None
    detail = f" (HTTP {status})" if status is not None else ""
    return f"request failed: {type(error).__name__}{detail}"


def _rows_to_entries(rows: list[dict], node_field: str | None) -> tuple[list[TimeseriesEntry], int]:
    """Normalize valid rows and order entries by timestamp.

    Args:
        rows: Rows already deduplicated across queries and page boundaries.
        node_field: Row field containing the node hostname, or None.

    Returns:
        Timestamp-ordered entries and the count of rows dropped during normalization.
    """
    entries_with_timestamps = []
    skipped = 0
    for row in rows:
        timestamp = _timestamp_ns(row.get("_time"))
        message = row.get("_msg")
        if timestamp is None or not isinstance(message, str):
            logger.debug("victorialogs: skipping row with invalid _time or non-string _msg")
            skipped += 1
            continue
        row_node = row.get(node_field) if node_field else None
        if not isinstance(row_node, str) or not row_node:
            row_node = None
        body, node = extract_body_and_node(message, row_node)
        entry: TimeseriesEntry = {"body": body, "ts": timestamp}
        resources: list[NodeResource | ReplicaResource] = []
        if node is not None:
            resources.append({"kind": "node", "hostname": node})
        replica = row.get("pod")
        if isinstance(replica, str) and replica:
            resources.append({"kind": "replica", "name": replica})
        if resources:
            entry["resources"] = resources
        entries_with_timestamps.append((timestamp, entry))

    entries_with_timestamps.sort(key=lambda item: item[0])
    entries = [entry for _, entry in entries_with_timestamps]
    return entries, skipped
