# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from loguru import logger

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import PlacementQuery, PodPlacement
from attribution_blueprint.connectors.metrics.promql.client import (
    PrometheusClient,
    promql_string,
    re2_escape,
)
from attribution_blueprint.types import assert_aware

_PROMQL_METRIC_IDENTIFIER = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_PROMQL_LABEL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class PrometheusPlacementResolver:
    """Recover Job and JobSet Pod placements from retained kube-state-metrics."""

    def __init__(
        self,
        *,
        endpoint: str,
        auth: Auth | None = None,
        timeout_seconds: float = 60.0,
        lookback_padding_seconds: int = 300,
        cluster: str | None = None,
        cluster_label: str | None = None,
        pod_info_metric: str = "kube_pod_info",
        pod_labels_metric: str = "kube_pod_labels",
        pod_controller_uid_label: str = "label_batch_kubernetes_io_controller_uid",
        batch_size: int = 20,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if lookback_padding_seconds < 0:
            raise ValueError("lookback_padding_seconds must not be negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if bool(cluster) != bool(cluster_label):
            raise ValueError("cluster and cluster_label must be configured together")
        if cluster_label is not None and not _PROMQL_LABEL_IDENTIFIER.fullmatch(cluster_label):
            raise ValueError(f"cluster_label is not a valid PromQL label name: {cluster_label!r}")
        for field, value in (
            ("pod_info_metric", pod_info_metric),
            ("pod_labels_metric", pod_labels_metric),
        ):
            if not _PROMQL_METRIC_IDENTIFIER.fullmatch(value):
                raise ValueError(f"{field} is not a valid PromQL metric name: {value!r}")
        if not _PROMQL_LABEL_IDENTIFIER.fullmatch(pod_controller_uid_label):
            raise ValueError(
                f"pod_controller_uid_label is not a valid PromQL label name: {pod_controller_uid_label!r}"
            )

        self.client = PrometheusClient(
            endpoint=endpoint,
            auth=auth,
            timeout_seconds=timeout_seconds,
        )
        self.lookback_padding_seconds = lookback_padding_seconds
        self.cluster = cluster
        self.cluster_label = cluster_label
        self.pod_info_metric = pod_info_metric
        self.pod_labels_metric = pod_labels_metric
        self.pod_controller_uid_label = pod_controller_uid_label
        self.batch_size = batch_size

    def placements_for_many(self, queries: list[PlacementQuery]) -> dict[str, list[PodPlacement]]:
        """Collect shared UID batches, then partition placements by Workload."""
        # Preprocess Workload queries and cache their padded time bounds.
        by_workload: dict[str, dict[str, PodPlacement]] = {}
        conflicting_uids: dict[str, set[str]] = defaultdict(set)
        by_namespace: dict[str, list[PlacementQuery]] = defaultdict(list)
        bounds_by_workload_uid: dict[str, tuple[datetime, datetime]] = {}
        now = datetime.now(timezone.utc)
        for query in queries:
            if query.workload_uid in by_workload:
                raise ValueError(f"duplicate placement query for Workload UID {query.workload_uid}")
            start = assert_aware(query.start_time, "placement start_time")
            end = assert_aware(query.end_time, "placement end_time")
            if end < start:
                raise ValueError("placement end_time precedes start_time")
            by_workload[query.workload_uid] = {}
            if query.namespace and query.owner_job_uids:
                query_start = start - timedelta(seconds=self.lookback_padding_seconds)
                query_end = min(now, end + timedelta(seconds=self.lookback_padding_seconds))
                by_namespace[query.namespace].append(query)
                bounds_by_workload_uid[query.workload_uid] = (query_start, query_end)

        # Build Job UID -> workload mapping
        batches: list[tuple[str, set[str], datetime, datetime, dict[str, set[str]]]] = []
        for namespace, namespace_queries in sorted(by_namespace.items()):
            queries_by_job_uid: dict[str, list[PlacementQuery]] = defaultdict(list)
            for query in namespace_queries:
                for job_uid in query.owner_job_uids:
                    queries_by_job_uid[job_uid].append(query)

            # Group temporally adjacent Workloads so each fixed-size batch
            # scans the smallest practical Prometheus lookback window.
            unique_job_uids = sorted(
                (job_uid for job_uid, uid_queries in queries_by_job_uid.items() if len(uid_queries) == 1),
                key=lambda job_uid: (*bounds_by_workload_uid[queries_by_job_uid[job_uid][0].workload_uid], job_uid),
            )
            for batch_index in range(0, len(unique_job_uids), self.batch_size):
                batch = set(unique_job_uids[batch_index : batch_index + self.batch_size])
                batch_queries = [queries_by_job_uid[job_uid][0] for job_uid in batch]
                query_start = min(bounds_by_workload_uid[query.workload_uid][0] for query in batch_queries)
                query_end = max(bounds_by_workload_uid[query.workload_uid][1] for query in batch_queries)
                routes = {
                    job_uid: {queries_by_job_uid[job_uid][0].workload_uid}
                    for job_uid in batch
                }
                batches.append((namespace, batch, query_start, query_end, routes))

            # A shared Job UID is interval-scoped so a Pod returned for one
            # Workload window cannot leak into another Workload's result.
            shared_routes_by_scope: dict[
                tuple[datetime, datetime],
                dict[str, set[str]],
            ] = defaultdict(lambda: defaultdict(set))
            for query in namespace_queries:
                bounds = bounds_by_workload_uid[query.workload_uid]
                for job_uid in query.owner_job_uids:
                    if len(queries_by_job_uid[job_uid]) > 1:
                        shared_routes_by_scope[bounds][job_uid].add(query.workload_uid)

            for (query_start, query_end), routes in sorted(shared_routes_by_scope.items()):
                shared_job_uids = sorted(routes)
                for batch_index in range(0, len(shared_job_uids), self.batch_size):
                    batch = set(shared_job_uids[batch_index : batch_index + self.batch_size])
                    batch_routes = {job_uid: routes[job_uid] for job_uid in batch}
                    batches.append((namespace, batch, query_start, query_end, batch_routes))

        batch_number = 0
        for namespace, batch, query_start, query_end, workload_uids_by_job_uid in batches:
            window_seconds = max(1, int((query_end - query_start).total_seconds()))
            promql = self._job_uids_query(
                namespace=namespace,
                owner_job_uids=batch,
                window_seconds=window_seconds,
            )
            batch_number += 1
            try:
                result = self.client.query(
                    name=f"kueue_pod_placement_batch_{batch_number}",
                    query=promql,
                    timestamp=int(query_end.timestamp()),
                )
            except Exception as error:
                logger.warning(
                    f"Prometheus placement batch failed for namespace {namespace!r} "
                    f"and {len(batch)} Job UIDs: {error}"
                )
                continue

            # Validate every returned series, then route it only to Workloads
            # associated with its exact owner Job UID.
            for series in result:
                labels = series.get("metric") or {}
                owner_job_uid = labels.get(self.pod_controller_uid_label)
                pod_uid = labels.get("uid")
                pod_name = labels.get("pod")
                node_name = labels.get("node")
                if (
                    labels.get("namespace") != namespace
                    or owner_job_uid not in batch
                    or not all((pod_uid, pod_name, node_name))
                ):
                    continue
                placement: PodPlacement = {
                    "pod_uid": pod_uid,
                    "pod_name": pod_name,
                    "node_name": node_name,
                    "source": "prometheus",
                }
                for workload_uid in workload_uids_by_job_uid[owner_job_uid]:
                    # Resolve conflicting pod placements
                    if pod_uid in conflicting_uids[workload_uid]:
                        continue
                    previous = by_workload[workload_uid].get(pod_uid)
                    if previous is None:
                        by_workload[workload_uid][pod_uid] = placement
                    elif previous != placement:
                        by_workload[workload_uid].pop(pod_uid)
                        conflicting_uids[workload_uid].add(pod_uid)
                        logger.warning(
                            f"Prometheus returned conflicting placements for Pod UID {pod_uid}; "
                            f"discarding that Pod's placement"
                        )
        return {
            workload_uid: [placements[pod_uid] for pod_uid in sorted(placements)]
            for workload_uid, placements in by_workload.items()
        }

    def _job_uids_query(
        self,
        *,
        namespace: str,
        owner_job_uids: set[str],
        window_seconds: int,
    ) -> str:
        pod_matchers = [*self._base_matchers(namespace), 'node!=""']
        sorted_uids = sorted(owner_job_uids)
        if not sorted_uids:
            raise ValueError("owner_job_uids must not be empty")
        if len(sorted_uids) == 1:
            uid_matcher = f"{self.pod_controller_uid_label}={promql_string(sorted_uids[0])}"
        else:
            uid_pattern = "|".join(re2_escape(uid) for uid in sorted_uids)
            uid_matcher = f"{self.pod_controller_uid_label}=~{promql_string(uid_pattern)}"
        labels_matchers = [
            *self._base_matchers(namespace),
            uid_matcher,
        ]
        pod_selector = f"{self.pod_info_metric}{{{','.join(pod_matchers)}}}"
        labels_selector = f"{self.pod_labels_metric}{{{','.join(labels_matchers)}}}"

        # Window each raw metric independently, THEN join. A subquery 
        # re-evaluates its inner expression only at each step tick.
        labels_last = f"last_over_time({labels_selector}[{window_seconds}s])"
        pod_last = f"last_over_time({pod_selector}[{window_seconds}s])"

        # Aggregate away exporter-identity labels before matching. 
        # A genuine conflict (the same Pod UID reported on two different nodes) 
        # still yields two distinct groups on the right.
        labels_deduped = f"group by (namespace,pod,uid,{self.pod_controller_uid_label}) ({labels_last})"
        pod_deduped = f"group by (namespace,pod,uid,node) ({pod_last})"

        return (
            f"{labels_deduped} * on(namespace,pod,uid) "
            f"group_right({self.pod_controller_uid_label}) {pod_deduped}"
        )

    def _base_matchers(self, namespace: str) -> list[str]:
        matchers = [f"namespace={promql_string(namespace)}"]
        if self.cluster_label is not None and self.cluster is not None:
            matchers.insert(0, f"{self.cluster_label}={promql_string(self.cluster)}")
        return matchers
