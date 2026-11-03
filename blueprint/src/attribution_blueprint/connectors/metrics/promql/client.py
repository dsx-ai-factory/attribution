# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from attribution_blueprint.connectors.auth import Auth

# Prometheus and Mimir use RE2 for label-match regexes. Python's re.escape()
# also escapes hyphens (`\-`), which RE2 rejects.
_RE2_METAS = frozenset(r".\+*?()[]{}|^$")


def re2_escape(value: str) -> str:
    return "".join("\\" + char if char in _RE2_METAS else char for char in value)


def promql_string(value: str) -> str:
    """Quote a string for use as a PromQL label-matcher value."""
    escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return f'"{escaped}"'


class PrometheusClient:
    """Small synchronous client for Prometheus-compatible query APIs."""

    def __init__(
        self,
        *,
        endpoint: str,
        auth: Auth | None = None,
        timeout_seconds: float = 60.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
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
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        if auth is not None:
            kwargs = auth.request_kwargs()
            if "headers" in kwargs:
                self.session.headers.update(kwargs["headers"])
            if "cert" in kwargs:
                self.session.cert = kwargs["cert"]
            if "verify" in kwargs:
                self.session.verify = kwargs["verify"]

    def query(self, *, name: str, query: str, timestamp: int) -> list[dict[str, Any]]:
        return self._post_result(
            name=name,
            path="/api/v1/query",
            data={"query": query, "time": timestamp},
        )

    def query_range(
        self,
        *,
        name: str,
        query: str,
        start: int,
        end: int,
        step: int,
    ) -> list[dict[str, Any]]:
        return self._post_result(
            name=name,
            path="/api/v1/query_range",
            data={"query": query, "start": start, "end": end, "step": step},
        )

    def _post_result(self, *, name: str, path: str, data: dict[str, Any]) -> list[dict[str, Any]]:
        response = self.session.post(
            f"{self.endpoint}{path}",
            data=data,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json() or {}
        if payload.get("status") != "success":
            logger.warning(
                f"promql query {name!r} returned status={payload.get('status')!r}: "
                f"{payload.get('error') or payload.get('errorType') or '<no detail>'}"
            )
            return []
        result = payload.get("data", {}).get("result", []) or []
        if not isinstance(result, list):
            logger.warning(f"promql query {name!r} returned a non-list result")
            return []
        return result
