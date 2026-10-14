# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from loguru import logger

from attribution_blueprint.kubernetes.daemon.health import HealthState
from attribution_blueprint.kubernetes.daemon.models import ResourceObservation
from attribution_blueprint.kubernetes.daemon.store import PlacementStore


def serve_api(
    state: HealthState,
    evidence_store: PlacementStore,
    port: int,
    namespace: str,
    *,
    cluster_id: str = "default",
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/livez":
                status, body = 200, {"status": "live"}
            elif parsed.path == "/readyz":
                ready = state.ready()
                status, body = (200 if ready else 503), {
                    "status": "ready" if ready else "not-ready",
                    "disabledWatches": state.disabled_watch_status(),
                }
            elif parsed.path == "/v1/history":
                query = parse_qs(parsed.query)
                requested_namespace = (query.get("namespace") or [""])[0]
                controller_uid = (query.get("controller_uid") or [""])[0]
                if not requested_namespace or not controller_uid:
                    status, body = 400, {
                        "error": "namespace and controller_uid are required"
                    }
                elif requested_namespace != namespace:
                    status, body = 403, {
                        "error": f"this daemon only serves namespace {namespace!r}, got {requested_namespace!r}"
                    }
                else:
                    try:
                        observations = evidence_store.list_observations_for_controller(
                            cluster_id,
                            requested_namespace,
                            controller_uid,
                        )
                        status, body = 200, _history_response(
                            cluster_id,
                            requested_namespace,
                            controller_uid,
                            observations,
                        )
                    except Exception:
                        logger.exception("execution history lookup failed")
                        status, body = 500, {"error": "execution history lookup failed"}
            elif parsed.path == "/v1/placements":
                query = parse_qs(parsed.query)
                requested_namespace = (query.get("namespace") or [""])[0]
                resource_uid = (query.get("resource_uid") or [""])[0]
                if not requested_namespace or not resource_uid:
                    status, body = 400, {"error": "namespace and resource_uid are required"}
                elif requested_namespace != namespace:
                    status, body = 403, {
                        "error": f"this daemon only serves namespace {namespace!r}, got {requested_namespace!r}"
                    }
                else:
                    try:
                        records = evidence_store.list_for_resource(
                            requested_namespace,
                            resource_uid,
                        )
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


def _history_response(
    cluster_id: str,
    namespace: str,
    controller_uid: str,
    observations: list[ResourceObservation],
) -> dict:
    resources: dict[str, dict] = {}
    relationships: dict[tuple[str, str, bool], dict] = {}
    for observation in observations:
        resource = _resource_identity(
            observation.api_version,
            observation.kind,
            observation.namespace,
            observation.name,
            observation.uid,
        )
        resources[observation.uid] = resource
        for owner in observation.owner_references:
            owner_resource = _resource_identity(
                owner.api_version,
                owner.kind,
                observation.namespace,
                owner.name,
                owner.uid,
            )
            resources.setdefault(owner.uid, owner_resource)
            relationships[(owner.uid, observation.uid, owner.controller)] = {
                "owner": owner_resource,
                "resource": resource,
                "controller": owner.controller,
            }
    return {
        "clusterId": cluster_id,
        "namespace": namespace,
        "controllerUid": controller_uid,
        "resources": list(resources.values()),
        "relationships": list(relationships.values()),
        "observations": [_observation_response(observation) for observation in observations],
    }


def _observation_response(observation: ResourceObservation) -> dict:
    result = {
        "apiVersion": observation.api_version,
        "kind": observation.kind,
        "metadata": {
            "namespace": observation.namespace,
            "name": observation.name,
            "uid": observation.uid,
            "resourceVersion": observation.resource_version,
            "ownerReferences": [
                {
                    "apiVersion": owner.api_version,
                    "kind": owner.kind,
                    "name": owner.name,
                    "uid": owner.uid,
                    "controller": owner.controller,
                }
                for owner in observation.owner_references
            ],
        },
        "eventType": observation.event_type,
        "observedAt": observation.observed_at,
    }
    if observation.selected_metadata is not None:
        result["selectedMetadata"] = observation.selected_metadata
    if observation.source_timestamp is not None:
        result["sourceTimestamp"] = observation.source_timestamp
    if observation.status is not None:
        result["status"] = observation.status
    return result


def _resource_identity(
    api_version: str,
    kind: str,
    namespace: str,
    name: str,
    uid: str,
) -> dict:
    return {
        "apiVersion": api_version,
        "kind": kind,
        "namespace": namespace,
        "name": name,
        "uid": uid,
    }
