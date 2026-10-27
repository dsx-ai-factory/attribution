# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import signal

from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.runtime import (
    KubernetesIdentityResolver,
    PodPlacementRecorder,
)
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.server import serve_api
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.sqlite_store import SQLitePlacementStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record Kueue Pod placement in durable storage")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--queue-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--watch-timeout-seconds", type=int, default=60)
    parser.add_argument("--health-port", type=int, default=8080)
    parser.add_argument("--kubeconfig")
    parser.add_argument("--context")
    parser.add_argument("--workload-version", default="v1beta1")
    parser.add_argument("--jobset-version", default="v1alpha2")
    parser.add_argument("--sqlite-path", default="/data/placements.db")
    parser.add_argument("--retention-days", type=int, default=30)
    return parser


def main() -> None:
    from kubernetes import client, config

    args = _parser().parse_args()
    if args.kubeconfig or args.context:
        config.load_kube_config(config_file=args.kubeconfig, context=args.context)
    else:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

    custom_api = client.CustomObjectsApi()
    placement_store = SQLitePlacementStore(args.sqlite_path, retention_days=args.retention_days)
    recorder = PodPlacementRecorder(
        client.CoreV1Api(),
        KubernetesIdentityResolver(
            client.BatchV1Api(),
            custom_api,
            workload_version=args.workload_version,
            jobset_version=args.jobset_version,
        ),
        placement_store,
        args.namespace,
        queue_size=args.queue_size,
        workers=args.workers,
        watch_timeout_seconds=args.watch_timeout_seconds,
    )
    api_server = serve_api(recorder.health, placement_store, args.health_port, args.namespace)

    def stop(_signum, _frame):
        logger.info("stopping Kueue Pod placement recorder")
        recorder.stop()
        api_server.shutdown()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    recorder.run()


if __name__ == "__main__":
    main()
