# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Mapping

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.connectors.logs import log_observation
from attribution_blueprint.connectors.logs.envelope import extract_body_and_node
from attribution_blueprint.connectors.logs.victorialogs.templates import TEMPLATES
from attribution_blueprint.types import (
    ExecutionContext,
    NodeResource,
    Observation,
    ReplicaResource,
    Source,
    TimeseriesEntry,
    assert_aware,
)

DEFAULT_LIMIT = 5000
DEFAULT_NODE_FIELD = "host_name"
MAX_PAGES = 20
QUERY_PATH = "/select/logsql/query"

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


class VictoriaLogsConnector:
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
        node_field: str | None = DEFAULT_NODE_FIELD,
    ):
        if source not in templates:
            raise ValueError(f"no templates registered for source {source}")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        self._source = source
        self._templates = list(templates[source])
        self._query_kwargs = dict(query_kwargs or {})
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_seconds
        self._limit = limit
        self._node_field = node_field or None
        self._auth = auth

        self._session = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=0.5,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
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
        return self._source

    def collect(self, context: ExecutionContext) -> list[Observation]:
        start = assert_aware(context["start_time"], "start_time")
        end = assert_aware(context["end_time"], "end_time")
        start_ns = _datetime_ns(start)
        end_ns = _datetime_ns(end) + 1
        nodes = context.get("nodes") or []

        rows: list[dict] = []
        seen: set[str] = set()
        for template in self._templates:
            try:
                query = template.format(
                    job_id=context.get("id", ""),
                    **self._query_kwargs,
                )
            except KeyError as error:
                logger.warning(f"victorialogs template missing kwarg {error}; skipping query")
                continue
            query = self._inject_node_filter(query, nodes)
            for row in self._query_window(query, start_ns, end_ns):
                key = json.dumps(row, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

        if not rows:
            return []
        entries_with_timestamps = []
        for row in rows:
            timestamp = _timestamp_ns(row.get("_time"))
            message = row.get("_msg")
            if timestamp is None or not isinstance(message, str):
                logger.debug("victorialogs: skipping row with invalid _time or non-string _msg")
                continue
            row_node = row.get(self._node_field) if self._node_field else None
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
        return [log_observation(source=self._source, entries=entries, context=context)]

    def _query_window(self, query: str, start_ns: int, end_ns_excl: int) -> list[dict]:
        rows: list[dict] = []
        cur_end = end_ns_excl
        for _ in range(MAX_PAGES):
            response = self._session.post(
                f"{self._endpoint}{QUERY_PATH}",
                data={
                    "query": query,
                    "start": start_ns,
                    "end": cur_end,
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
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(row, dict):
                    malformed += 1
                    continue
                page.append(row)
            if malformed:
                logger.warning(f"victorialogs: skipped {malformed} invalid NDJSON line(s)")
            if not page:
                return rows
            rows.extend(page)
            if len(page) < self._limit:
                return rows
            timestamps = [timestamp for row in page if (timestamp := _timestamp_ns(row.get("_time"))) is not None]
            if not timestamps:
                logger.warning("victorialogs: saturated page has no parseable _time; stopping pagination")
                return rows
            oldest = min(timestamps)
            if oldest <= start_ns:
                return rows
            next_end = oldest + 1
            if next_end >= cur_end:
                logger.warning(f"victorialogs: possible truncation at {_format_timestamp_ns(oldest)}")
                next_end = oldest
            cur_end = next_end
        logger.warning(
            f"victorialogs: hit MAX_PAGES={MAX_PAGES} for query={query[:80]!r}; "
            f"older entries before ts={cur_end} may be truncated"
        )
        return rows

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


def _format_timestamp_ns(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    timestamp = datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{timestamp}.{nanoseconds:09d}Z"
