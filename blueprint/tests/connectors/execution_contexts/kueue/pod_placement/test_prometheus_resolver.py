# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timezone

import pytest

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import PlacementQuery
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.prometheus_resolver import (
    PrometheusPlacementResolver,
)


def _series(*, namespace, pod, uid, node, controller_uid="job-uid-1"):
    return {
        "metric": {
            "namespace": namespace,
            "pod": pod,
            "uid": uid,
            "node": node,
            "label_batch_kubernetes_io_controller_uid": controller_uid,
        }
    }


def _placement_query(
    workload_uid,
    *,
    namespace="team-a",
    owner_job_uids=("job-uid-1",),
    start_hour=0,
    end_hour=1,
):
    return PlacementQuery(
        workload_uid=workload_uid,
        namespace=namespace,
        owner_job_uids=frozenset(owner_job_uids),
        start_time=datetime(2026, 5, 1, start_hour, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 1, end_hour, tzinfo=timezone.utc),
    )


class TestConstructorValidation:
    def test_non_positive_timeout_rejected(self):
        with pytest.raises(ValueError, match="timeout_seconds"):
            PrometheusPlacementResolver(endpoint="https://prom.example", timeout_seconds=0)

    def test_negative_lookback_padding_rejected(self):
        with pytest.raises(ValueError, match="lookback_padding_seconds"):
            PrometheusPlacementResolver(endpoint="https://prom.example", lookback_padding_seconds=-1)

    def test_non_positive_batch_size_rejected(self):
        with pytest.raises(ValueError, match="batch_size"):
            PrometheusPlacementResolver(endpoint="https://prom.example", batch_size=0)

    def test_cluster_without_cluster_label_rejected(self):
        with pytest.raises(ValueError, match="cluster and cluster_label"):
            PrometheusPlacementResolver(endpoint="https://prom.example", cluster="c1")

    def test_cluster_label_without_cluster_rejected(self):
        with pytest.raises(ValueError, match="cluster and cluster_label"):
            PrometheusPlacementResolver(endpoint="https://prom.example", cluster_label="cluster")

    def test_cluster_and_cluster_label_together_accepted(self):
        source = PrometheusPlacementResolver(endpoint="https://prom.example", cluster="c1", cluster_label="cluster")
        assert source.cluster == "c1"
        assert source.cluster_label == "cluster"

    def test_invalid_cluster_label_rejected(self):
        with pytest.raises(ValueError, match="cluster_label"):
            PrometheusPlacementResolver(endpoint="https://prom.example", cluster="c1", cluster_label="not valid!")

    @pytest.mark.parametrize("field", ["pod_info_metric", "pod_labels_metric"])
    def test_invalid_metric_name_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            PrometheusPlacementResolver(endpoint="https://prom.example", **{field: "not a metric!"})

    def test_invalid_controller_uid_label_rejected(self):
        with pytest.raises(ValueError, match="pod_controller_uid_label"):
            PrometheusPlacementResolver(
                endpoint="https://prom.example",
                pod_controller_uid_label="not valid!",
            )


class TestUIDQueryConstruction:
    def test_single_job_uid_query_shape(self):
        """G6: last_over_time must wrap each raw metric individually (never
        the joined expression, never a subquery). G7: each side must be
        dedup-aggregated over its semantic identity labels before the join.
        `node!=""` on the info side excludes unscheduled samples."""
        source = PrometheusPlacementResolver(endpoint="https://prom.example")
        query = source._job_uids_query(
            namespace="team-a",
            owner_job_uids={"job-uid-1"},
            window_seconds=3600,
        )
        assert query == (
            "group by (namespace,pod,uid,label_batch_kubernetes_io_controller_uid) (last_over_time("
            'kube_pod_labels{namespace="team-a",'
            'label_batch_kubernetes_io_controller_uid="job-uid-1"}'
            "[3600s])) "
            "* on(namespace,pod,uid) group_right(label_batch_kubernetes_io_controller_uid) "
            "group by (namespace,pod,uid,node) (last_over_time("
            'kube_pod_info{namespace="team-a",node!=""}'
            "[3600s]))"
        )

    def test_uid_query_includes_cluster_matcher_first(self):
        source = PrometheusPlacementResolver(endpoint="https://prom.example", cluster="prod", cluster_label="cluster")
        query = source._job_uids_query(
            namespace="team-a",
            owner_job_uids={"job-uid-1"},
            window_seconds=60,
        )
        assert 'kube_pod_info{cluster="prod",namespace="team-a",node!=""}' in query
        assert (
            'kube_pod_labels{cluster="prod",namespace="team-a",'
            'label_batch_kubernetes_io_controller_uid="job-uid-1"}'
        ) in query

    def test_uid_query_uses_custom_metric_and_label_names(self):
        source = PrometheusPlacementResolver(
            endpoint="https://prom.example",
            pod_info_metric="custom_pod_info",
            pod_labels_metric="custom_pod_labels",
            pod_controller_uid_label="controller_uid",
        )
        query = source._job_uids_query(
            namespace="team-a",
            owner_job_uids={"job-uid-1"},
            window_seconds=60,
        )
        assert "custom_pod_info{" in query
        assert 'custom_pod_labels{namespace="team-a",controller_uid="job-uid-1"}' in query

    def test_multiple_job_uids_use_sorted_escaped_regex(self):
        source = PrometheusPlacementResolver(endpoint="https://prom.example")
        query = source._job_uids_query(
            namespace="team-a",
            owner_job_uids={"job.uid-2", "job-uid-1"},
            window_seconds=1200,
        )
        assert 'label_batch_kubernetes_io_controller_uid=~"job-uid-1|job\\\\.uid-2"' in query

    def test_raw_metrics_are_windowed_before_the_join_not_the_joined_result(self):
        """G6 regression: `last_over_time` must appear once around each raw
        metric selector. It must NOT wrap the join itself (that structure
        is a subquery re-evaluated only at each step tick, which loses
        Pods that lived only between two ticks)."""
        source = PrometheusPlacementResolver(endpoint="https://prom.example")
        query = source._job_uids_query(
            namespace="team-a",
            owner_job_uids={"job-uid-1"},
            window_seconds=3600,
        )
        assert query.count("last_over_time(") == 2
        # The join operator must sit BETWEEN the two last_over_time(...)
        # results, not inside a last_over_time(...) itself.
        join_index = query.index(
            "* on(namespace,pod,uid) group_right(label_batch_kubernetes_io_controller_uid)"
        )
        first_last_over_time_end = query.index("]))") + len("]))")
        assert join_index >= first_last_over_time_end
        assert not query.startswith("last_over_time((")

    def test_dedup_aggregation_wraps_each_side_before_the_join(self):
        """G7 regression: each side of the join must be pre-aggregated by
        its semantic identity (namespace/pod/uid[/node]) so duplicate
        exporter-replica series (differing only in instance/prometheus/
        replica labels on an HA/federated backend) collapse to one series
        per Pod before matching, instead of triggering a many-to-many
        matching error."""
        source = PrometheusPlacementResolver(endpoint="https://prom.example")
        query = source._job_uids_query(
            namespace="team-a",
            owner_job_uids={"job-uid-1"},
            window_seconds=60,
        )
        assert query.startswith("group by (namespace,pod,uid,label_batch_kubernetes_io_controller_uid) (last_over_time(")
        assert "group by (namespace,pod,uid,node) (last_over_time(" in query

    def test_dedup_aggregation_preserves_owner_uid_label(self):
        """Regression: a PromQL aggregation drops every label not named in
        its `by` clause. placements_for_many() reads
        `pod_controller_uid_label` back out of each result series to know
        which owner Job (and therefore which batched Workload query) it
        belongs to (see TestPlacementsForMany below) - if the left-hand
        aggregation drops that label, every real Prometheus result is
        silently discarded in the batched path even though the query
        itself succeeds."""
        source = PrometheusPlacementResolver(
            endpoint="https://prom.example",
            pod_controller_uid_label="custom_controller_uid",
        )
        query = source._job_uids_query(
            namespace="team-a",
            owner_job_uids={"job-uid-1"},
            window_seconds=60,
        )
        left_aggregation = query.split(" * on(namespace,pod,uid)")[0]
        assert left_aggregation.startswith("group by (")
        by_clause = left_aggregation[len("group by (") : left_aggregation.index(")")]
        by_labels = set(by_clause.split(","))
        assert source.pod_controller_uid_label in by_labels


class TestPlacementsForMany:
    def test_shared_query_is_partitioned_by_job_uid(self, mocker):
        source = PrometheusPlacementResolver(endpoint="https://prom.example")
        mock_query = mocker.patch.object(
            source.client,
            "query",
            return_value=[
                _series(namespace="team-a", pod="pod-a", uid="pod-uid-a", node="node-a", controller_uid="job-a"),
                _series(namespace="team-a", pod="pod-b", uid="pod-uid-b", node="node-b", controller_uid="job-b"),
            ],
        )

        result = source.placements_for_many(
            [
                _placement_query("workload-a", owner_job_uids=("job-a",)),
                _placement_query("workload-b", owner_job_uids=("job-b",)),
            ]
        )

        mock_query.assert_called_once()
        assert 'label_batch_kubernetes_io_controller_uid=~"job-a|job-b"' in mock_query.call_args.kwargs["query"]
        assert result["workload-a"] == [
            {"pod_uid": "pod-uid-a", "pod_name": "pod-a", "node_name": "node-a", "source": "prometheus"}
        ]
        assert result["workload-b"] == [
            {"pod_uid": "pod-uid-b", "pod_name": "pod-b", "node_name": "node-b", "source": "prometheus"}
        ]

    def test_unique_job_uids_with_different_intervals_stay_batched(self, mocker):
        source = PrometheusPlacementResolver(
            endpoint="https://prom.example",
            lookback_padding_seconds=0,
        )
        mock_query = mocker.patch.object(
            source.client,
            "query",
            return_value=[
                _series(
                    namespace="team-a",
                    pod="pod-early",
                    uid="pod-uid-early",
                    node="node-a",
                    controller_uid="job-early",
                ),
                _series(
                    namespace="team-a",
                    pod="pod-late",
                    uid="pod-uid-late",
                    node="node-b",
                    controller_uid="job-late",
                ),
            ],
        )

        result = source.placements_for_many(
            [
                _placement_query(
                    "workload-early",
                    owner_job_uids=("job-early",),
                    start_hour=0,
                    end_hour=1,
                ),
                _placement_query(
                    "workload-late",
                    owner_job_uids=("job-late",),
                    start_hour=2,
                    end_hour=3,
                ),
            ]
        )

        mock_query.assert_called_once()
        assert mock_query.call_args.kwargs["timestamp"] == int(
            datetime(2026, 5, 1, 3, tzinfo=timezone.utc).timestamp()
        )
        assert "[10800s]" in mock_query.call_args.kwargs["query"]
        assert {placement["pod_uid"] for placement in result["workload-early"]} == {"pod-uid-early"}
        assert {placement["pod_uid"] for placement in result["workload-late"]} == {"pod-uid-late"}

    def test_unique_job_uid_batches_group_adjacent_intervals(self, mocker):
        source = PrometheusPlacementResolver(
            endpoint="https://prom.example",
            lookback_padding_seconds=0,
            batch_size=2,
        )
        mock_query = mocker.patch.object(source.client, "query", return_value=[])

        source.placements_for_many(
            [
                _placement_query(
                    "workload-late-a",
                    owner_job_uids=("job-a",),
                    start_hour=10,
                    end_hour=11,
                ),
                _placement_query(
                    "workload-early-b",
                    owner_job_uids=("job-b",),
                    start_hour=0,
                    end_hour=1,
                ),
                _placement_query(
                    "workload-late-c",
                    owner_job_uids=("job-c",),
                    start_hour=12,
                    end_hour=13,
                ),
                _placement_query(
                    "workload-early-d",
                    owner_job_uids=("job-d",),
                    start_hour=2,
                    end_hour=3,
                ),
            ]
        )

        assert mock_query.call_count == 2
        assert [call.kwargs["timestamp"] for call in mock_query.call_args_list] == [
            int(datetime(2026, 5, 1, 3, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 5, 1, 13, tzinfo=timezone.utc).timestamp()),
        ]
        assert all("[10800s]" in call.kwargs["query"] for call in mock_query.call_args_list)

    def test_shared_job_uid_does_not_cross_workload_intervals(self, mocker):
        source = PrometheusPlacementResolver(
            endpoint="https://prom.example",
            lookback_padding_seconds=0,
        )
        mock_query = mocker.patch.object(
            source.client,
            "query",
            side_effect=[
                [_series(namespace="team-a", pod="pod-early", uid="pod-uid-early", node="node-a")],
                [_series(namespace="team-a", pod="pod-late", uid="pod-uid-late", node="node-b")],
            ],
        )

        result = source.placements_for_many(
            [
                _placement_query("workload-early", start_hour=0, end_hour=1),
                _placement_query("workload-late", start_hour=2, end_hour=3),
            ]
        )

        assert mock_query.call_count == 2
        assert [call.kwargs["timestamp"] for call in mock_query.call_args_list] == [
            int(datetime(2026, 5, 1, 1, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 5, 1, 3, tzinfo=timezone.utc).timestamp()),
        ]
        assert {placement["pod_uid"] for placement in result["workload-early"]} == {"pod-uid-early"}
        assert {placement["pod_uid"] for placement in result["workload-late"]} == {"pod-uid-late"}

    def test_conflicting_pod_does_not_discard_valid_placements(self, mocker):
        source = PrometheusPlacementResolver(endpoint="https://prom.example")
        mocker.patch.object(
            source.client,
            "query",
            return_value=[
                _series(namespace="team-a", pod="pod-conflict", uid="pod-uid-1", node="node-a"),
                _series(namespace="team-a", pod="pod-conflict", uid="pod-uid-1", node="node-b"),
                _series(namespace="team-a", pod="pod-valid", uid="pod-uid-2", node="node-c"),
            ],
        )

        result = source.placements_for_many([_placement_query("workload-a")])

        assert result["workload-a"] == [
            {
                "pod_uid": "pod-uid-2",
                "pod_name": "pod-valid",
                "node_name": "node-c",
                "source": "prometheus",
            }
        ]

    def test_job_uids_are_split_into_bounded_batches(self, mocker):
        source = PrometheusPlacementResolver(endpoint="https://prom.example", batch_size=1)
        mock_query = mocker.patch.object(source.client, "query", return_value=[])

        source.placements_for_many(
            [
                _placement_query("workload-a", owner_job_uids=("job-a",)),
                _placement_query("workload-b", owner_job_uids=("job-b",)),
            ]
        )

        assert mock_query.call_count == 2
        assert {call.kwargs["name"] for call in mock_query.call_args_list} == {
            "kueue_pod_placement_batch_1",
            "kueue_pod_placement_batch_2",
        }

    def test_namespaces_are_queried_separately(self, mocker):
        source = PrometheusPlacementResolver(endpoint="https://prom.example")
        mock_query = mocker.patch.object(source.client, "query", return_value=[])

        source.placements_for_many(
            [
                _placement_query("workload-a", namespace="team-a", owner_job_uids=("job-a",)),
                _placement_query("workload-b", namespace="team-b", owner_job_uids=("job-b",)),
            ]
        )

        assert mock_query.call_count == 2
        queries = [call.kwargs["query"] for call in mock_query.call_args_list]
        assert any('namespace="team-a"' in query and '"job-a"' in query for query in queries)
        assert any('namespace="team-b"' in query and '"job-b"' in query for query in queries)

    def test_duplicate_workload_uid_is_rejected(self):
        source = PrometheusPlacementResolver(endpoint="https://prom.example")
        query = _placement_query("workload-a")

        with pytest.raises(ValueError, match="duplicate placement query"):
            source.placements_for_many([query, query])
