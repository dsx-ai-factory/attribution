# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests
from loguru import logger

from attribution_blueprint.connectors.auth import Auth, credential_identity
from attribution_blueprint.connectors.logs import log_observation
from attribution_blueprint.connectors.logs.loki.client import (
    LokiEntry,
    LokiQueryBudgetExceeded,
    LokiQueryClient,
    datetime_to_nanoseconds,
)
from attribution_blueprint.connectors.logs.loki.range_cache import LokiRangeCache
from attribution_blueprint.connectors.logs.loki.templates import TEMPLATES
from attribution_blueprint.types import (
    ExecutionContext,
    Observation,
    NodeResource,
    ReplicaResource,
    TimeseriesEntry,
    Source,
    assert_aware,
)

DEFAULT_LIMIT = 5000
DEFAULT_NODE_LABEL = "host_name"
DEFAULT_NODE_CHUNK_SIZE = 50
# Concurrent queries per batch; enough to overlap network latency without
# hammering the backend.
_MAX_IN_FLIGHT = 4

# Cap on cached entries materialized per query when extending a context's slice
# into the deep past. Without it, a long context on a cluster-wide query pulls
# every cached line back to its start, and a batch of them exhausts memory.
_MAX_CACHED_ENTRIES = 50_000


def _entry_node(entry: LokiEntry) -> str:
    """Return an entry's node lowercased, matching the per-context slice key."""
    node = entry["node"]
    return node.lower() if isinstance(node, str) else ""


# RE2 metacharacters (https://github.com/google/re2/wiki/Syntax). Unlike
# Python re, RE2 rejects unrecognized escape sequences (e.g. `\-`), so
# re.escape() can't be used for Loki — it backslashes too many characters.
_RE2_METAS = frozenset(r".\+*?()[]{}|^$")


def _re2_escape(s: str) -> str:
    return "".join("\\" + c if c in _RE2_METAS else c for c in s)


def _as_str(value: object) -> str:
    """Envelope fields come from arbitrary JSON, so only a str is usable downstream."""
    return value if isinstance(value, str) else ""


def _extract_body_and_node(payload: str, node: str | None) -> tuple[str, str | None]:
    """Pull the log line, and a hostname when the stream had none, out of a JSON-encoded Loki value.

    Handles OTel (`body` + `attributes.hostname`), `message`, and CloudEvents
    (`data.message` + `data.hostname`) envelopes; anything else is returned verbatim.
    """
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload, node
    if not isinstance(parsed, dict):
        return payload, node

    attributes = parsed.get("attributes")
    if node is None and isinstance(attributes, dict):
        node = _as_str(attributes.get("hostname")) or None
    body = _as_str(parsed.get("body")) or _as_str(parsed.get("message"))
    if body == "":
        data = parsed.get("data")
        if isinstance(data, dict):
            body = _as_str(data.get("message"))
            if node is None:
                node = _as_str(data.get("hostname")) or None
    if body == "":
        body = payload
    return body, node


class LokiConnector:
    """Collector that fetches one Source from a Loki backend.

    Each instance handles exactly one Source. To serve both SYSLOG and APPLOG
    from the same backend, instantiate twice — failure isolation per Source
    lives at the instance boundary.

    `collect(context)` queries per context. `collect_batch(contexts)` fetches
    one window for a whole batch and slices the result per context by node and
    time. Both read through `LokiQueryClient` (bounded, bisected, forward
    requests), optionally through a `LokiRangeCache` so overlapping windows
    only fetch the five-minute buckets they lack. With a cache and a trailing
    `batch_window_hours`, each context is still sliced over its whole window:
    the part before the fetched window comes from buckets earlier batches
    stored, so a long workload gains coverage with every pass at no extra
    query. The orchestrator uses `collect_batch` when `[collection] mode =
    "batch"`.
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
        batch_window_hours: float = 0.0,
        batch_node_filter: bool = False,
        node_chunk_size: int = DEFAULT_NODE_CHUNK_SIZE,
        range_cache: LokiRangeCache | None = None,
        session: requests.Session | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        if source not in templates:
            raise ValueError(f"no templates registered for source {source}")
        if batch_window_hours < 0:
            raise ValueError(f"batch_window_hours must not be negative, got {batch_window_hours}")
        if node_chunk_size <= 0:
            raise ValueError(f"node_chunk_size must be positive, got {node_chunk_size}")
        self._source = source
        # The v1 design spec types templates as dict[Source, str] (one query
        # per source). The blueprint extends this to a list so a single
        # Source can carry multiple ORed queries — see the bundled SYSLOG
        # set. Accept either shape so customers following the PDF aren't
        # broken when they pass a string.
        tmpl_val = templates[source]
        self._templates = [tmpl_val] if isinstance(tmpl_val, str) else list(tmpl_val)
        self._query_kwargs = query_kwargs
        # Retained so the CLI override path can inherit endpoint/auth/etc.
        # from a TOML-built instance when only some Loki flags are passed.
        self._endpoint = endpoint.rstrip("/")
        self._auth = auth
        self._timeout = timeout_seconds
        self._limit = limit
        # When set, the connector splices `<node_label>=~"n1|n2|..."` into
        # each query's first label selector to keep per-ExecutionContext
        # fetches scoped to that ExecutionContext's nodes, and reads each
        # entry's node from that label. Set to None for cluster-wide queries.
        self._node_label = node_label
        # Batch mode only: a positive value trims each context to its trailing
        # window before slicing; 0 keeps the whole context window, as
        # `collect` does.
        self._batch_window = timedelta(hours=batch_window_hours) if batch_window_hours > 0 else None
        self._batch_node_filter = batch_node_filter
        self._node_chunk_size = node_chunk_size
        self._range_cache = range_cache
        self._clock = clock if clock is not None else (lambda: datetime.now(timezone.utc))
        # Query-affecting settings that scope cached buckets; the credential
        # digest keeps tenants apart without storing secrets.
        self._cache_config: dict[str, object] = {
            "endpoint": self._endpoint,
            "limit": limit,
            "node_label": node_label,
            "source": source.value,
            "credential_identity": credential_identity(auth),
        }
        self._client = LokiQueryClient(
            endpoint=self._endpoint,
            auth=auth,
            timeout_seconds=timeout_seconds,
            limit=limit,
            node_label=node_label,
            session=session,
        )

    @property
    def source(self) -> Source:
        return self._source

    def collect(self, context: ExecutionContext) -> list[Observation]:
        # The service requires aware datetimes; reject naive at the boundary
        # rather than propagating them into the wire and triggering a 500.
        start = assert_aware(context["start_time"], "start_time")
        end = assert_aware(context["end_time"], "end_time")
        nodes = [node for node in (context.get("nodes") or []) if node]
        queries: list[str] = []
        for tmpl in self._templates:
            try:
                query = tmpl.format(job_id=context.get("id", ""), **self._query_kwargs)
            except KeyError as e:
                logger.warning(f"loki template missing kwarg {e}; skipping query")
                continue
            queries.append(self._inject_node_filter(query, nodes))
        entries = self._query_batch(queries, start, end)
        if not entries:
            return []
        return [self._observation(entries, context)]

    def collect_batch(self, contexts: Sequence[ExecutionContext]) -> dict[str, list[Observation]]:
        """Fetch one window for the batch and slice it per context.

        Each context receives the lines from its nodes inside its window
        (trimmed to `batch_window_hours` when set). Templates that reference
        `{job_id}` cannot share a query, so they fall back to per-context
        `collect`.
        """
        contexts = list(contexts)
        if not contexts:
            return {}
        if any("{job_id}" in tmpl for tmpl in self._templates):
            return {str(ctx["id"]): self.collect(ctx) for ctx in contexts}
        windows = {str(ctx["id"]): self._context_window(ctx) for ctx in contexts}
        fetch_start = min(window[0] for window in windows.values())
        fetch_end = max(window[1] for window in windows.values())
        queries = self._batch_queries(contexts)
        entries = self._query_batch(queries, fetch_start, fetch_end)
        if self._range_cache is not None:
            # Extend the batch's window back to its earliest context start with
            # buckets earlier batches stored, reading the batch's nodes once;
            # nothing before `fetch_start` is fetched.
            batch_start = min(assert_aware(ctx["start_time"], "start_time") for ctx in contexts)
            if batch_start < fetch_start:
                all_nodes = {node.lower() for ctx in contexts for node in (ctx.get("nodes") or []) if node}
                earlier = self._cached_entries(queries, batch_start, fetch_start, all_nodes)
                if earlier:
                    entries = self._merge_entries([entries, earlier])
        result: dict[str, list[Observation]] = {}
        for ctx in contexts:
            ctx_id = str(ctx["id"])
            window_start, window_end = windows[ctx_id]
            nodes = {node.lower() for node in (ctx.get("nodes") or []) if node}
            if self._range_cache is not None:
                context_start = assert_aware(ctx["start_time"], "start_time")
                if context_start < window_start:
                    window_start = context_start
            start_ns = datetime_to_nanoseconds(window_start)
            end_ns = datetime_to_nanoseconds(window_end)
            selected = [
                entry
                for entry in entries
                if entry["node"] is not None
                and entry["node"].lower() in nodes
                and start_ns <= entry["timestamp"] <= end_ns
            ]
            result[ctx_id] = (
                [self._observation(selected, {**ctx, "start_time": window_start, "end_time": window_end})]
                if selected
                else []
            )
        return result

    def _cached_entries(self, queries: list[str], start: datetime, end: datetime, nodes: set[str]) -> list[LokiEntry]:
        """Return cached entries for `queries` on `nodes` inside `[start, end)` without fetching.

        Node-scoped: only the requested nodes' lines are materialized, so a batch
        pins memory proportional to its nodes, not the cluster. The cap keeps the
        newest `_MAX_CACHED_ENTRIES` per query and reading stops there.
        """
        assert self._range_cache is not None
        now = self._clock()
        pages = [
            self._range_cache.read(
                query=query,
                config=self._cache_config,
                start=start,
                end=end,
                nodes=nodes,
                timestamp=lambda entry: entry["timestamp"],
                now=now,
                node_of=_entry_node,
                max_entries=_MAX_CACHED_ENTRIES,
            )
            for query in queries
        ]
        return self._merge_entries(pages)

    def _context_window(self, context: ExecutionContext) -> tuple[datetime, datetime]:
        start = assert_aware(context["start_time"], "start_time")
        end = assert_aware(context["end_time"], "end_time")
        if self._batch_window is not None:
            start = max(start, end - self._batch_window)
        return start, end

    def _batch_queries(self, contexts: list[ExecutionContext]) -> list[str]:
        rendered: list[str] = []
        for tmpl in self._templates:
            try:
                rendered.append(tmpl.format(**self._query_kwargs))
            except KeyError as e:
                logger.warning(f"loki template missing kwarg {e}; skipping query")
        if not (self._batch_node_filter and self._node_label):
            return rendered
        nodes = sorted({node for ctx in contexts for node in (ctx.get("nodes") or []) if node})
        chunks = [
            nodes[offset : offset + self._node_chunk_size] for offset in range(0, len(nodes), self._node_chunk_size)
        ]
        return (
            [self._inject_node_filter(query, chunk) for query in rendered for chunk in chunks] if chunks else rendered
        )

    def _query_batch(self, queries: list[str], start: datetime, end: datetime) -> list[LokiEntry]:
        """Run every query over one window; merge, dedup, and order the entries."""

        def run(query: str) -> list[LokiEntry]:
            try:
                return self._query_range(query, start, end)
            except LokiQueryBudgetExceeded as exc:
                logger.warning(f"loki query {query[:80]!r} truncated: {exc}; keeping {len(exc.entries)} entries")
                return list(exc.entries)

        if len(queries) <= 1:
            pages = [run(query) for query in queries]
        else:
            with ThreadPoolExecutor(max_workers=min(_MAX_IN_FLIGHT, len(queries))) as executor:
                pages = list(executor.map(run, queries))
        return self._merge_entries(pages)

    @staticmethod
    def _merge_entries(pages: Sequence[Sequence[LokiEntry]]) -> list[LokiEntry]:
        """Union entry lists, dropping duplicates by (labels, timestamp, body), in timestamp order."""
        seen: set[tuple[str, int, str]] = set()
        merged: list[LokiEntry] = []
        for page in pages:
            for entry in page:
                key = (json.dumps(entry["labels"], sort_keys=True), entry["timestamp"], entry["body"])
                if key in seen:
                    continue
                seen.add(key)
                merged.append(entry)
        merged.sort(key=lambda entry: entry["timestamp"])
        return merged

    def _query_range(self, query: str, start: datetime, end: datetime) -> list[LokiEntry]:
        """Read one inclusive window through the range cache when one is configured."""
        if self._range_cache is None:
            return self._client.query_range(query, start, end)

        def fetch_missing(rendered: str, range_start: datetime, range_end: datetime) -> list[LokiEntry]:
            range_end_ns = datetime_to_nanoseconds(range_end)
            entries = self._client.query_range(rendered, range_start, range_end)
            return [entry for entry in entries if entry["timestamp"] < range_end_ns]

        return self._range_cache.query(
            query=query,
            config=self._cache_config,
            start=start,
            end=end + timedelta(microseconds=1),
            inclusive_end_ns=datetime_to_nanoseconds(end),
            fetch=fetch_missing,
            timestamp=lambda entry: entry["timestamp"],
            now=self._clock(),
            node_of=_entry_node,
        )

    def _observation(self, entries: list[LokiEntry], context: ExecutionContext) -> Observation:
        # Every source emits the shared timeseries format, syslog included, so
        # batch slicing and single collection produce the same observation.
        # Node and pod labels ride along as per-entry resources only when the
        # stream carries them.
        timeseries: list[TimeseriesEntry] = []
        for entry in entries:
            line, node = _extract_body_and_node(entry["body"], entry["node"])
            item: TimeseriesEntry = {"body": line, "ts": entry["timestamp"]}
            resources: list[NodeResource | ReplicaResource] = []
            if node is not None:
                resources.append({"kind": "node", "hostname": node})
            replica = entry["labels"].get("pod")
            if replica is not None:
                resources.append({"kind": "replica", "name": replica})
            if resources != []:
                item["resources"] = resources
            timeseries.append(item)
        return log_observation(source=self._source, entries=timeseries, context=context)

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
