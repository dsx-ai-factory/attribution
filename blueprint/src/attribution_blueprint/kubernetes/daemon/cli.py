# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import signal

from loguru import logger

from attribution_blueprint.kubernetes.daemon.resolvers import KubernetesOwnershipResolver
from attribution_blueprint.kubernetes.daemon.runtime import KubernetesDaemon
from attribution_blueprint.kubernetes.daemon.server import serve_api
from attribution_blueprint.kubernetes.daemon.sqlite_store import SQLitePlacementStore

_API_REQUEST_TIMEOUT_SECONDS = (10.0, 120.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record durable Kubernetes evidence")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--queue-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--watch-timeout-seconds", type=int, default=60)
    parser.add_argument("--health-port", type=int, default=8080)
    parser.add_argument("--kubeconfig")
    parser.add_argument("--context")
    parser.add_argument("--jobset-version", default="v1alpha2")
    parser.add_argument("--sqlite-path", default="/data/placements.db")
    parser.add_argument("--retention-days", type=int, default=30)
    return parser


def main() -> None:
    from kubernetes import client, config

    args = _parser().parse_args()
    configuration = client.Configuration()
    if args.kubeconfig or args.context:
        config.load_kube_config(
            config_file=args.kubeconfig,
            context=args.context,
            client_configuration=configuration,
        )
    else:
        try:
            config.load_incluster_config(client_configuration=configuration)
        except config.ConfigException:
            config.load_kube_config(client_configuration=configuration)

    api_client = client.ApiClient(configuration)
    _set_default_api_timeout(api_client)
    custom_api = client.CustomObjectsApi(api_client)
    placement_store = SQLitePlacementStore(args.sqlite_path, retention_days=args.retention_days)
    daemon = KubernetesDaemon(
        client.CoreV1Api(api_client),
        placement_store,
        args.namespace,
        owner_resolver=KubernetesOwnershipResolver(
            client.BatchV1Api(api_client),
            custom_api,
            jobset_version=args.jobset_version,
        ),
        queue_size=args.queue_size,
        workers=args.workers,
        watch_timeout_seconds=args.watch_timeout_seconds,
    )
    api_server = serve_api(daemon.health, placement_store, args.health_port, args.namespace)

    def stop(_signum, _frame):
        logger.info("stopping KubernetesDaemon")
        daemon.stop()
        api_server.shutdown()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    daemon.run()


def _set_default_api_timeout(api_client) -> None:
    """Bound connect and read time for list, watch, and enrichment calls."""
    call_api = api_client.call_api

    def call_api_with_timeout(*args, **kwargs):
        if kwargs.get("_request_timeout") is None:
            kwargs["_request_timeout"] = _API_REQUEST_TIMEOUT_SECONDS
        return call_api(*args, **kwargs)

    api_client.call_api = call_api_with_timeout


if __name__ == "__main__":
    main()
