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
import re

import requests
from loguru import logger

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.connectors.logs.loki.templates import TEMPLATES
from attribution_blueprint.connectors.logs.loki.types import LokiSyslogEntry
from attribution_blueprint.types import (
    ExecutionContext,
    Format,
    Observation,
    Source,
    assert_aware,
)

_FORMAT_FOR_SOURCE: dict[Source, Format] = {
    Source.SYSLOG: Format.LOKI_DICT_LIST,
    Source.APPLOG: Format.RAW_LOKI_STREAMS,
}

DEFAULT_LIMIT = 5000
DEFAULT_NODE_LABEL = "host_name"

# RE2 metacharacters (https://github.com/google/re2/wiki/Syntax). Unlike
# Python re, RE2 rejects unrecognized escape sequences (e.g. `\-`), so
# re.escape() can't be used for Loki — it backslashes too many characters.
_RE2_METAS = frozenset(r".\+*?()[]{}|^$")


def _re2_escape(s: str) -> str:
    return "".join("\\" + c if c in _RE2_METAS else c for c in s)


# Hard cap on paginated requests per template so a pathologically chatty query
# can't run forever; at DEFAULT_LIMIT=5000 this is up to 100k entries.
MAX_PAGES = 20


class LokiConnector:
    """Collector that fetches one Source from a Loki backend.

    Each instance handles exactly one Source. To serve both SYSLOG and APPLOG
    from the same backend, instantiate twice — failure isolation per Source
    lives at the instance boundary.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        auth: Auth,
        source: Source,
        query_kwargs: dict[str, str],
        templates: dict[Source, list[str] | str] = TEMPLATES,
        timeout_seconds: float = 60.0,
        limit: int = DEFAULT_LIMIT,
        node_label: str | None = DEFAULT_NODE_LABEL,
    ):
        if source not in templates:
            raise ValueError(f"no templates registered for source {source}")
        self._source = source
        # The v1 design spec types templates as dict[Source, str] (one query
        # per source). The blueprint extends this to a list so a single
        # Source can carry multiple ORed queries — see the bundled SYSLOG
        # set. Accept either shape so customers following the PDF aren't
        # broken when they pass a string.
        tmpl_val = templates[source]
        self._templates = [tmpl_val] if isinstance(tmpl_val, str) else list(tmpl_val)
        self._query_kwargs = query_kwargs
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_seconds
        self._limit = limit
        # Retained so the CLI override path can inherit endpoint/auth/etc.
        # from a TOML-built instance when only some Loki flags are passed.
        self._auth = auth
        # When set, the connector splices `<node_label>=~"n1|n2|..."` into
        # each query's first label selector to keep per-ExecutionContext
        # fetches scoped to that ExecutionContext's nodes. Set to None for
        # cluster-wide queries.
        self._node_label = node_label

        self._session = requests.Session()
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
        # The service requires aware datetimes; reject naive at the boundary
        # rather than propagating them into the wire and triggering a 500.
        start = assert_aware(context["start_time"], "start_time")
        end = assert_aware(context["end_time"], "end_time")
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)
        nodes = context.get("nodes") or []

        all_streams: list[dict] = []
        for tmpl in self._templates:
            try:
                query = tmpl.format(
                    job_id=context.get("id", ""),
                    **self._query_kwargs,
                )
            except KeyError as e:
                logger.warning(f"loki template missing kwarg {e}; skipping query")
                continue

            query = self._inject_node_filter(query, nodes)
            all_streams.extend(self._query_paginated(query, start_ns, end_ns))

        if not all_streams:
            return []

        fmt = _FORMAT_FOR_SOURCE[self._source]
        body = self._encode_body(all_streams, fmt)
        return [
            {
                "context": {
                    "time_interval": {"start": context["start_time"], "end": context["end_time"]},
                    "source": self._source,
                    "format": fmt,
                    "resources": "AllJobResources",
                },
                "body": body,
            }
        ]

    def _encode_body(self, streams: list[dict], fmt: Format) -> str:
        # Multi-template runs and pagination can produce duplicate streams
        # (same labels, overlapping values). Coalesce by label set and dedup
        # (timestamp, line) pairs before encoding so the service doesn't
        # double-count.
        streams = self._merge_streams(streams)
        if fmt == Format.LOKI_DICT_LIST:
            # The service's LOKI_DICT_LIST consumer expects a flat list of
            # LokiSyslogEntry, not raw Loki streams.
            # Entries with node=None are dropped: attribution_lib's syslog
            # entry parser passes node directly into verify_node() with no
            # None-guard, so emitting them crashes the service. Whoever
            # configured the connector with the wrong node_label (or hit a
            # stream missing the label) should see the warning here.
            entries: list[LokiSyslogEntry] = []
            dropped_no_node = 0
            for stream in streams:
                labels = stream.get("stream", {}) or {}
                node = labels.get(self._node_label) if self._node_label else None
                if node is None:
                    dropped_no_node += sum(1 for _ in stream.get("values", []) or [])
                    continue
                for ts, line in stream.get("values", []) or []:
                    try:
                        timestamp = int(ts)
                    except (TypeError, ValueError):
                        continue
                    entries.append({"timestamp": timestamp, "body": line, "node": node})
            if dropped_no_node:
                logger.warning(
                    f"loki SYSLOG: dropped {dropped_no_node} entries with missing "
                    f"{self._node_label!r} label (service requires node identity)"
                )
            return json.dumps(entries)
        # RAW_LOKI_STREAMS: the connector forwards raw Loki stream objects.
        # The service-side parser flattens each value into a LokiAppEntry
        # (see `connectors.logs.loki.types.LokiAppEntry` for the per-entry
        # shape the flattened payload satisfies). attribution_lib's APPLOG
        # extractor reads `stream["node_name"]` exclusively, so the
        # configured `node_label` (default "host_name") is rewritten to
        # "node_name" here. Without this, every APPLOG entry attributes to
        # node=None.
        if self._node_label and self._node_label != "node_name":
            for stream in streams:
                labels = stream.get("stream") or {}
                if self._node_label in labels:
                    labels["node_name"] = labels.pop(self._node_label)
                stream["stream"] = labels
        return json.dumps(streams)

    @staticmethod
    def _merge_streams(streams: list[dict]) -> list[dict]:
        merged: dict[tuple, dict] = {}
        seen: dict[tuple, set[tuple]] = {}
        for stream in streams:
            labels = stream.get("stream", {}) or {}
            key = tuple(sorted(labels.items()))
            if key not in merged:
                merged[key] = {"stream": dict(labels), "values": []}
                seen[key] = set()
            for entry in stream.get("values", []) or []:
                if len(entry) < 2:
                    continue
                pair = (entry[0], entry[1])
                if pair in seen[key]:
                    continue
                seen[key].add(pair)
                merged[key]["values"].append(entry)
        return list(merged.values())

    def _query_paginated(self, query: str, start_ns: int, end_ns: int) -> list[dict]:
        """Walk Loki's query_range backward in time until the result is unsaturated.

        Loki returns entries newest-first and caps at `limit` per response, so a
        single call silently drops the tail for high-volume windows. We re-issue
        with `end` set to the oldest returned timestamp minus 1ns, until a page
        comes back under the cap or we hit MAX_PAGES.

        POST keeps the LogQL out of the URL (Loki's GET path caps at ~8 KiB and
        a long `host_name=~"n1|n2|..."` regex for multi-node attempts blows past it).
        """
        out: list[dict] = []
        cur_end = end_ns
        for page in range(MAX_PAGES):
            response = self._session.post(
                f"{self._endpoint}/loki/api/v1/query_range",
                data={"query": query, "start": start_ns, "end": cur_end, "limit": self._limit},
                timeout=self._timeout,
            )
            response.raise_for_status()
            streams = response.json().get("data", {}).get("result", [])
            entry_count = sum(len(s.get("values", []) or []) for s in streams)
            if entry_count == 0:
                break
            out.extend(streams)
            if entry_count < self._limit:
                break
            oldest_ns = min(int(ts) for s in streams for ts, _ in (s.get("values", []) or []))
            if oldest_ns <= start_ns:
                break
            cur_end = oldest_ns - 1
        else:
            logger.warning(
                f"loki: hit MAX_PAGES={MAX_PAGES} for query={query[:80]!r}; "
                f"older entries before ts={cur_end} may be truncated"
            )
        return out

    def _inject_node_filter(self, query: str, nodes: list[str]) -> str:
        """Inject `<node_label>=~"<n1|n2|...>"` into every top-level stream
        selector of `query`. Hostnames are RE2-escaped so dots in FQDNs
        don't accidentally broaden the match. Inner `{` / `}` that
        appear inside string literals (double-quoted or backtick-quoted)
        are ignored — stream selectors don't nest in LogQL, but selector
        values may legally contain braces.

        Limitation: this parser tracks selector context by counting `{`/`}`
        outside string literals only; any LogQL pipeline stage that uses a
        second brace-block syntax (`| label_format {...}`, `| json {...}`,
        etc.) will trigger a spurious second injection. Customer templates
        must use the stream-selector position for label-filter braces only.

        Note: Python's re.escape() is too aggressive for RE2 — it backslashes
        hyphens, which RE2 rejects as 'invalid char escape' (hyphens are
        regex-special only inside character classes). Escape only the RE2
        metacharacters that can appear in valid hostnames here.
        """
        if not self._node_label or not nodes:
            return query
        nodes_regex = "|".join(_re2_escape(n) for n in nodes)
        filter_expr = f', {self._node_label}=~"{nodes_regex}"'
        out: list[str] = []
        i = 0
        in_selector = False
        in_dq = False
        in_bq = False
        while i < len(query):
            ch = query[i]
            if in_dq:
                out.append(ch)
                if ch == "\\" and i + 1 < len(query):
                    out.append(query[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    in_dq = False
                i += 1
            elif in_bq:
                out.append(ch)
                if ch == "`":
                    in_bq = False
                i += 1
            elif ch == '"' and in_selector:
                in_dq = True
                out.append(ch)
                i += 1
            elif ch == "`" and in_selector:
                in_bq = True
                out.append(ch)
                i += 1
            elif ch == "{" and not in_selector:
                in_selector = True
                out.append(ch)
                i += 1
            elif ch == "}" and in_selector:
                out.append(filter_expr)
                out.append(ch)
                in_selector = False
                i += 1
            else:
                out.append(ch)
                i += 1
        return "".join(out)
