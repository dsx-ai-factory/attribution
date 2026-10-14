# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Collection
from urllib.parse import parse_qs, urlparse

from loguru import logger

from attribution_blueprint.kubernetes.daemon.health import HealthState
from attribution_blueprint.kubernetes.daemon.store import PlacementReader


def serve_api(
    state: HealthState,
    placement_reader: PlacementReader,
    port: int,
    namespace: str,
    *,
    resource_uid_aliases: Collection[str] = (),
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/livez":
                status, body = 200, {"status": "live"}
            elif parsed.path == "/readyz":
                ready = state.ready()
                status, body = (200 if ready else 503), {"status": "ready" if ready else "not-ready"}
            elif parsed.path == "/v1/placements":
                query = parse_qs(parsed.query)
                requested_namespace = (query.get("namespace") or [""])[0]
                pod_uid = (query.get("pod_uid") or [""])[0]
                owner_uid = (query.get("owner_uid") or [""])[0]
                resource_uid = (query.get("resource_uid") or [""])[0]
                if not resource_uid:
                    for alias in resource_uid_aliases:
                        if resource_uid := (query.get(alias) or [""])[0]:
                            break
                selectors = tuple(value for value in (pod_uid, owner_uid, resource_uid) if value)
                if not requested_namespace or len(selectors) != 1:
                    status, body = 400, {
                        "error": "namespace and exactly one of pod_uid, owner_uid, or resource_uid are required"
                    }
                elif requested_namespace != namespace:
                    status, body = 403, {
                        "error": f"this daemon only serves namespace {namespace!r}, got {requested_namespace!r}"
                    }
                else:
                    try:
                        if pod_uid:
                            records = placement_reader.list_for_pod(requested_namespace, pod_uid)
                        elif owner_uid:
                            records = placement_reader.list_for_owner(requested_namespace, owner_uid)
                        else:
                            records = placement_reader.list_for_resource(requested_namespace, resource_uid)
                        status, body = 200, {"items": [record.to_resource() for record in records]}
                    except Exception:
                        logger.exception("placement lookup failed")
                        status, body = 500, {"error": "placement lookup failed"}
            else:
                status, body = 404, {"status": "not-found"}

            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, name="kubernetes-daemon-api-server", daemon=True).start()
    return server
