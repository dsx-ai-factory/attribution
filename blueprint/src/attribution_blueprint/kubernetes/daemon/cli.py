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
from attribution_blueprint.kubernetes.daemon.watch_sources import (
    custom_watch_source,
    job_watch_source,
    pod_watch_source,
)

_API_CONNECT_TIMEOUT_SECONDS = 10.0
_API_READ_TIMEOUT_GRACE_SECONDS = 10.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record durable Kubernetes evidence")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--cluster-id", default="default")
    parser.add_argument("--queue-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--watch-timeout-seconds", type=int, default=60)
    parser.add_argument("--health-port", type=int, default=8080)
    parser.add_argument("--kubeconfig")
    parser.add_argument("--context")
    parser.add_argument("--watch-pods", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--watch-jobs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--watch-jobsets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jobset-version", default="v1alpha2")
    parser.add_argument("--enable-kueue", action="store_true")
    parser.add_argument("--kueue-workload-version", default="v1beta1")
    parser.add_argument("--enable-kai", action="store_true")
    parser.add_argument("--kai-pod-group-version", default="v2alpha2")
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
    _set_default_api_timeout(api_client, args.watch_timeout_seconds)
    core_api = client.CoreV1Api(api_client)
    batch_api = client.BatchV1Api(api_client)
    custom_api = client.CustomObjectsApi(api_client)
    resource_resolvers = []
    watch_sources = []
    if args.watch_pods:
        watch_sources.append(pod_watch_source(core_api, args.namespace))
    if args.watch_jobs:
        watch_sources.append(job_watch_source(batch_api, args.namespace))
    if args.watch_jobsets:
        watch_sources.append(
            custom_watch_source(
                custom_api,
                name="jobsets",
                group="jobset.x-k8s.io",
                version=args.jobset_version,
                namespace=args.namespace,
                plural="jobsets",
                kind="JobSet",
            )
        )
    if args.enable_kueue:
        from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.resource_resolver import (
            KueueResourceResolver,
        )

        resource_resolvers.append(
            KueueResourceResolver(custom_api, workload_version=args.kueue_workload_version)
        )
        watch_sources.append(
            custom_watch_source(
                custom_api,
                name="kueue-workloads",
                group="kueue.x-k8s.io",
                version=args.kueue_workload_version,
                namespace=args.namespace,
                plural="workloads",
                kind="Workload",
            )
        )
    if args.enable_kai:
        from attribution_blueprint.connectors.execution_contexts.kai.resource_resolver import (
            KaiResourceResolver,
        )

        resource_resolvers.append(
            KaiResourceResolver(custom_api, pod_group_version=args.kai_pod_group_version)
        )
        watch_sources.append(
            custom_watch_source(
                custom_api,
                name="kai-podgroups",
                group="scheduling.run.ai",
                version=args.kai_pod_group_version,
                namespace=args.namespace,
                plural="podgroups",
                kind="PodGroup",
            )
        )

    placement_store = SQLitePlacementStore(args.sqlite_path, retention_days=args.retention_days)
    daemon = KubernetesDaemon(
        core_api,
        placement_store,
        args.namespace,
        owner_resolver=KubernetesOwnershipResolver(
            batch_api,
            custom_api,
            jobset_version=args.jobset_version,
        ),
        resource_resolvers=tuple(resource_resolvers),
        queue_size=args.queue_size,
        workers=args.workers,
        watch_timeout_seconds=args.watch_timeout_seconds,
        cluster_id=args.cluster_id,
        watch_sources=tuple(watch_sources),
    )
    api_server = serve_api(daemon.health, placement_store, args.health_port, args.namespace)

    def stop(_signum, _frame):
        logger.info("stopping KubernetesDaemon")
        daemon.stop()
        api_server.shutdown()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    daemon.run()

def _set_default_api_timeout(api_client, watch_timeout_seconds: int) -> None:
    """Bound connect and read time for list, watch, and enrichment calls."""
    call_api = api_client.call_api
    request_timeout = (
        _API_CONNECT_TIMEOUT_SECONDS,
        float(watch_timeout_seconds) + _API_READ_TIMEOUT_GRACE_SECONDS,
    )

    def call_api_with_timeout(*args, **kwargs):
        if kwargs.get("_request_timeout") is None:
            kwargs["_request_timeout"] = request_timeout
        return call_api(*args, **kwargs)

    api_client.call_api = call_api_with_timeout


if __name__ == "__main__":
    main()
