# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loki query client shared by the connector and other callers.

The client owns the mechanics of reading a complete window from Loki:

* `query_range` requests capped at `query_window` each, so no single request
  scans an unbounded range. Windows run oldest first, or newest first with
  `newest_first=True` so a spent budget drops the oldest entries instead.
* Bisection of saturated pages (a page that returns exactly `limit` entries
  hides its tail) instead of paging backwards, which keeps equal timestamps.
* One page budget per `query_range` call. When it runs out the partial
  result is raised as `LokiQueryBudgetExceeded` so callers can keep it.

Transient HTTP failures are retried by the session (see
`attribution_blueprint.connectors.auth.build_session`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import NotRequired, TypedDict

import requests

from attribution_blueprint.connectors.auth import Auth, build_session
from attribution_blueprint.types import assert_aware

DEFAULT_LIMIT = 5000
DEFAULT_MAX_PAGES = 500
DEFAULT_QUERY_WINDOW = timedelta(minutes=30)
DEFAULT_NODE_LABEL = "host_name"

_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def datetime_to_nanoseconds(value: datetime) -> int:
    """Return `value` as exact integer Unix nanoseconds."""
    delta = value.astimezone(timezone.utc) - _UNIX_EPOCH
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1_000


class LokiQueryBudgetExceeded(RuntimeError):
    """Raised with the entries fetched so far when the page budget runs out."""

    def __init__(self, entries: list, message: str = "Loki pagination limit reached before exhaustion") -> None:
        super().__init__(message)
        self.entries = entries


class LokiEntry(TypedDict):
    """One flattened Loki log line."""

    timestamp: int  # nanoseconds since the Unix epoch
    body: str
    labels: dict[str, str]
    node: str | None
    # Loki's per-line structured metadata, when the response carries it.
    structured_metadata: NotRequired[dict[str, str]]


def node_from_labels(
    labels: Mapping[str, object],
    preferred: str | None = None,
    fallbacks: Sequence[str] = (),
) -> str | None:
    """Return the node named by `labels`, preferring `preferred` then `fallbacks`."""
    candidates = ([preferred] if preferred else []) + list(fallbacks)
    for key in candidates:
        value = labels.get(key)
        if isinstance(value, str) and value != "":
            return value
    return None


def _in_time_order(parts: list[list[LokiEntry]], newest_first: bool) -> list[LokiEntry]:
    """Join parts fetched in read order into timestamp order."""
    return [entry for part in (reversed(parts) if newest_first else parts) for entry in part]


class LokiQueryClient:
    """Fetch complete Loki windows with bounded, bisected, forward requests."""

    def __init__(
        self,
        *,
        endpoint: str,
        auth: Auth | None = None,
        timeout_seconds: float = 60.0,
        limit: int = DEFAULT_LIMIT,
        max_pages: int = DEFAULT_MAX_PAGES,
        query_window: timedelta | None = DEFAULT_QUERY_WINDOW,
        node_label: str | None = DEFAULT_NODE_LABEL,
        node_label_fallbacks: Sequence[str] = (),
        session: requests.Session | None = None,
    ) -> None:
        """Create a client.

        `query_window=None` reads each requested range in one request (plus
        bisection). `node_label_fallbacks` are further stream labels tried when
        `node_label` is absent; the Blueprint assumes none. `session` is used
        as-is; otherwise `build_session(auth)` supplies one.
        """
        if not endpoint:
            raise ValueError("endpoint must be a nonempty Loki base URL")
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if max_pages <= 0:
            raise ValueError(f"max_pages must be positive, got {max_pages}")
        if query_window is not None and query_window <= timedelta(0):
            raise ValueError(f"query_window must be positive, got {query_window}")
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_seconds
        self._limit = limit
        self._max_pages = max_pages
        self._query_window_ns = None if query_window is None else int(query_window.total_seconds() * 1_000_000_000)
        self._node_label = node_label
        self._node_label_fallbacks = tuple(node_label_fallbacks)
        self._session = build_session(auth) if session is None else session

    def query_range(self, query: str, start: datetime, end: datetime, *, newest_first: bool = False) -> list[LokiEntry]:
        """Return every entry for `query` between `start` and `end`, both inclusive.

        Entries are ordered by timestamp. Raises `LokiQueryBudgetExceeded`
        carrying the partial result when the page budget runs out.
        """
        start = assert_aware(start, "start")
        end = assert_aware(end, "end")
        if end < start:
            raise ValueError("end must not precede start")
        return self._fetch_windows(query, datetime_to_nanoseconds(start), datetime_to_nanoseconds(end), newest_first)

    def _fetch_windows(self, query: str, start_ns: int, end_ns: int, newest_first: bool) -> list[LokiEntry]:
        budget = [self._max_pages]
        span = self._query_window_ns
        windows: list[tuple[int, int]] = []
        if newest_first:
            # Same span per window, anchored at `end_ns` and read newest first.
            window_end = end_ns
            while window_end >= start_ns:
                window_start = start_ns if span is None else max(window_end - span, start_ns)
                windows.append((window_start, window_end))
                window_end = window_start - 1
        else:
            window_start = start_ns
            while window_start <= end_ns:
                window_end = end_ns if span is None else min(window_start + span, end_ns)
                windows.append((window_start, window_end))
                window_start = window_end + 1
        fetched: list[list[LokiEntry]] = []
        try:
            for window_start, window_end in windows:
                fetched.append(self._fetch_interval(query, window_start, window_end, budget, newest_first))
        except LokiQueryBudgetExceeded as exc:
            fetched.append(exc.entries)
            raise LokiQueryBudgetExceeded(_in_time_order(fetched, newest_first), str(exc)) from None
        return _in_time_order(fetched, newest_first)

    def _fetch_interval(
        self, query: str, start_ns: int, end_ns: int, budget: list[int], newest_first: bool = False
    ) -> list[LokiEntry]:
        if budget[0] <= 0:
            raise LokiQueryBudgetExceeded(
                [], f"Loki page budget of {self._max_pages} exhausted before the window was complete"
            )
        budget[0] -= 1
        page = self._fetch_page(query, start_ns, end_ns)
        if len(page) < self._limit:
            return page
        if start_ns >= end_ns:
            raise RuntimeError(
                f"Loki pagination cannot split a saturated timestamp {start_ns}: raise limit above {self._limit}"
            )
        midpoint = start_ns + (end_ns - start_ns) // 2
        halves = [(start_ns, midpoint), (midpoint + 1, end_ns)]
        if newest_first:
            halves.reverse()
        try:
            first = self._fetch_interval(query, *halves[0], budget, newest_first)
        except LokiQueryBudgetExceeded as exc:
            raise LokiQueryBudgetExceeded(exc.entries if exc.entries else page, str(exc)) from None
        try:
            second = self._fetch_interval(query, *halves[1], budget, newest_first)
        except LokiQueryBudgetExceeded as exc:
            raise LokiQueryBudgetExceeded(_in_time_order([first, exc.entries], newest_first), str(exc)) from None
        return _in_time_order([first, second], newest_first)

    def _fetch_page(self, query: str, start_ns: int, end_ns: int) -> list[LokiEntry]:
        response = self._session.post(
            f"{self._endpoint}/loki/api/v1/query_range",
            data={"query": query, "start": start_ns, "end": end_ns, "limit": self._limit, "direction": "forward"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", {}) if isinstance(payload, Mapping) else {}
        streams = data.get("result", []) if isinstance(data, Mapping) else []
        return self.flatten_streams(streams)

    def flatten_streams(self, streams: object) -> list[LokiEntry]:
        """Flatten a Loki `data.result` payload into entries in payload order."""
        page: list[LokiEntry] = []
        for stream in streams if isinstance(streams, list) else []:
            if not isinstance(stream, Mapping):
                continue
            raw_labels = stream.get("stream", {})
            labels = (
                {str(key): str(value) for key, value in raw_labels.items()} if isinstance(raw_labels, Mapping) else {}
            )
            node = node_from_labels(labels, self._node_label, self._node_label_fallbacks)
            values = stream.get("values", [])
            for value in values if isinstance(values, list) else []:
                if not isinstance(value, list) or len(value) < 2:
                    continue
                try:
                    timestamp_ns = int(value[0])
                except (TypeError, ValueError):
                    continue
                entry: LokiEntry = {"timestamp": timestamp_ns, "body": str(value[1]), "labels": labels, "node": node}
                metadata = value[2] if len(value) >= 3 else None
                if isinstance(metadata, Mapping):
                    entry["structured_metadata"] = {str(key): str(item) for key, item in metadata.items()}
                page.append(entry)
        return page


__all__ = [
    "DEFAULT_LIMIT",
    "DEFAULT_MAX_PAGES",
    "DEFAULT_NODE_LABEL",
    "DEFAULT_QUERY_WINDOW",
    "LokiEntry",
    "LokiQueryBudgetExceeded",
    "LokiQueryClient",
    "datetime_to_nanoseconds",
    "node_from_labels",
]
