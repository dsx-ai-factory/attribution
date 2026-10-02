# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timedelta, timezone

import pytest

from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.connectors.metrics.promql import PromQLMetricsConnector
from attribution_blueprint.types import Source


def _ctx(nodes=("node001", "node002")):
    return {
        "id": "sluid-123",
        "nodes": list(nodes),
        "start_time": datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 12, 30, 0, tzinfo=timezone.utc),
    }


def _mock_response(mocker, result, status: str = "success", error: str | None = None):
    fake = mocker.Mock()
    payload: dict = {"status": status, "data": {"result": result}}
    if error is not None:
        payload["error"] = error
    fake.json.return_value = payload
    fake.raise_for_status = mocker.Mock()
    return mocker.patch("requests.Session.post", return_value=fake)


def test_status_error_returns_empty_and_warns(mocker, caplog):
    """Prometheus error JSON (HTTP 200 with status:"error") must be surfaced as
    a warning and produce no observations - silently treating it as zero-results
    masks misconfigured queries and Mimir resource limits."""
    _mock_response(mocker, [], status="error", error="query timeout")
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": "{name}"}},
    )
    observations = conn.collect(_ctx())
    assert observations == []


def test_substitutes_name_cluster_and_nodes(mocker):
    post = _mock_response(mocker, [])
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example/",
        auth=TokenAuth("tok"),
        cluster="example-cluster",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"my_metric": '{name}{{s_cluster="{cluster}", net_host_name=~"{nodes}"}}'}},
    )
    conn.collect(_ctx(nodes=("n1", "n2")))
    args, kwargs = post.call_args
    assert args[0] == "https://prom.example/api/v1/query_range"
    assert kwargs["data"]["query"] == 'my_metric{s_cluster="example-cluster", net_host_name=~"n1|n2"}'


def test_node_regex_does_not_backslash_escape_hyphens(mocker):
    post = _mock_response(mocker, [])
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example/",
        auth=TokenAuth("tok"),
        cluster="example-cluster",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"my_metric": '{name}{{net_host_name=~"{nodes}"}}'}},
    )

    conn.collect(_ctx(nodes=("example-node-014", "host.example.com")))

    query = post.call_args.kwargs["data"]["query"]
    assert "\\-" not in query
    assert "example-node-014" in query
    assert "host\\.example\\.com" in query


def test_uses_seconds_for_time_window(mocker):
    post = _mock_response(mocker, [])
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": "{name}"}},
    )
    ctx = _ctx()
    conn.collect(ctx)
    _, kwargs = post.call_args
    assert kwargs["data"]["start"] == int(ctx["start_time"].timestamp())
    assert kwargs["data"]["end"] == int(ctx["end_time"].timestamp())
    assert kwargs["data"]["step"] == 60


def test_step_seconds_is_configurable(mocker):
    post = _mock_response(mocker, [])
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": "{name}"}},
        step_seconds=15,
    )
    conn.collect(_ctx())
    _, kwargs = post.call_args
    assert kwargs["data"]["step"] == 15


def test_returns_empty_when_no_nodes_in_context(mocker):
    post = _mock_response(mocker, [])
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": "{name}"}},
    )
    ctx = _ctx()
    ctx["nodes"] = []
    assert conn.collect(ctx) == []
    assert post.call_count == 0


def test_returns_empty_when_backend_has_no_data(mocker):
    _mock_response(mocker, [])
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": "{name}"}},
    )
    assert conn.collect(_ctx()) == []


def test_body_is_one_timeseries_entry_per_sample(mocker):
    series_a = [
        {
            "metric": {"net_host_name": "node001", "s_cluster": "c", "device": "mlx5_0"},
            "values": [["1744734450", "1"], ["1744734510", "6253.417646976346"]],
        }
    ]
    series_b = [
        {
            "metric": {"net_host_name": "node002", "s_cluster": "c"},
            "values": [["1744734450", "0"]],
        }
    ]
    fake1 = mocker.Mock()
    fake1.json.return_value = {"status": "success", "data": {"result": series_a}}
    fake1.raise_for_status = mocker.Mock()
    fake2 = mocker.Mock()
    fake2.json.return_value = {"status": "success", "data": {"result": series_b}}
    fake2.raise_for_status = mocker.Mock()
    mocker.patch("requests.Session.post", side_effect=[fake1, fake2])

    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"link_down": "{name}", "up": "{name}"}},
    )
    ctx = _ctx()
    obs = conn.collect(ctx)
    assert len(obs) == 1
    o = obs[0]
    assert o["context"] == {
        "time_interval": {"start": ctx["start_time"], "end": ctx["end_time"]},
        "source": Source.NODE_METRICS,
    }
    decoded = json.loads(o["body"])
    assert decoded == [
        {
            "ts": 1744734450_000_000_000,
            "resources": [{"kind": "node", "hostname": "node001"}],
            "body": (
                "time: 1744734450, metric_name: link_down, metric_value: 1, threshold: None, "
                "cluster: c, node: node001, component: mlx5_0"
            ),
        },
        {
            "ts": 1744734510_000_000_000,
            "resources": [{"kind": "node", "hostname": "node001"}],
            "body": (
                "time: 1744734510, metric_name: link_down, metric_value: 6253.417646976346, threshold: None, "
                "cluster: c, node: node001, component: mlx5_0"
            ),
        },
        {
            "ts": 1744734450_000_000_000,
            "resources": [{"kind": "node", "hostname": "node002"}],
            "body": "time: 1744734450, metric_name: up, metric_value: 0, threshold: None, cluster: c, node: node002, component: None",
        },
    ]


def test_skips_series_without_node_label(mocker):
    _mock_response(
        mocker,
        [
            {"metric": {"s_cluster": "c"}, "values": [["1", "1"]]},
            {"metric": {"net_host_name": "n1"}, "values": [["2", "5"]]},
        ],
    )
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": "{name}"}},
    )
    obs = conn.collect(_ctx())
    body = json.loads(obs[0]["body"])
    assert [e["resources"] for e in body] == [[{"kind": "node", "hostname": "n1"}]]


def test_mtls_auth_sets_session_cert():
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=MTLSAuth(cert_path="/c", key_path="/k", ca_cert_path="/ca"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": "{name}"}},
    )
    assert conn._client._session.cert == ("/c", "/k")
    assert conn._client._session.verify == "/ca"


def test_token_auth_sets_authorization_header():
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok", org_id="tenant-42"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": "{name}"}},
    )
    assert conn._client._session.headers["Authorization"] == "Bearer tok"
    assert conn._client._session.headers["X-Scope-OrgID"] == "tenant-42"


def test_auth_is_optional():
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": "{name}"}},
    )
    assert "Authorization" not in conn._client._session.headers
    assert "X-Scope-OrgID" not in conn._client._session.headers


def test_source_without_templates_raises():
    with pytest.raises(ValueError, match="no templates registered"):
        PromQLMetricsConnector(
            endpoint="https://prom.example",
            auth=TokenAuth("tok"),
            cluster="c",
            source=Source.NODE_METRICS,
            templates={},
        )


def test_bundled_templates_render_with_no_overrides(mocker):
    post = _mock_response(mocker, [])
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="example-cluster",
        source=Source.NODE_METRICS,
    )
    conn.collect(_ctx(nodes=("n1",)))
    # All four bundled queries should fire: link_downed, carrier_down,
    # ib_roce_adp_retrans, up. Carrier_down is the second link-down(east-west)
    # signal alongside infiniband_link_downed - attribution_lib registers
    # both with the same symptom type; missing it leaves a dead symptom path.
    assert post.call_count == 4
    queries = [call.kwargs["data"]["query"] for call in post.call_args_list]
    assert any("node_infiniband_link_downed_total" in q for q in queries)
    assert any("node_network_carrier_down_changes_total" in q for q in queries)
    assert any("ib_roce_adp_retrans" in q for q in queries)
    assert any('s_cluster="example-cluster"' in q for q in queries)
    assert any('net_host_name=~"n1"' in q for q in queries)


def test_bundled_carrier_down_template_uses_changes_threshold_gt_zero(mocker):
    """attribution_lib's nm_carrier_down_changes treats value > 0 as a
    link-flap; the template must surface only points where the counter
    actually changed, mirroring the infiniband link-down query shape."""
    post = _mock_response(mocker, [])
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
    )
    conn.collect(_ctx(nodes=("n1",)))
    queries = [call.kwargs["data"]["query"] for call in post.call_args_list]
    carrier = [q for q in queries if "node_network_carrier_down_changes_total" in q]
    assert carrier, "no node_network_carrier_down_changes_total query rendered"
    assert "changes(" in carrier[0]
    assert "> 0" in carrier[0]


def test_node_names_are_regex_escaped(mocker):
    """FQDN hostnames must not over-match under `=~` - `.` is regex 'any char'."""
    post = _mock_response(mocker, [])
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": '{name}{{net_host_name=~"{nodes}"}}'}},
    )
    conn.collect(_ctx(nodes=("host.example.com", "host[01].cluster")))
    _, kwargs = post.call_args
    # `.` and `[`, `]` must be escaped; the pipe between alternates stays.
    query = kwargs["data"]["query"]
    assert r"host\.example\.com" in query
    assert r"host\[01\]\.cluster" in query
    # Plain pipe-join (which would let "hostXexample-com" match) must NOT appear.
    assert "host.example.com|host[01].cluster" not in query


def test_cluster_falls_back_to_self_cluster_when_label_dropped(mocker):
    """`max by (net_host_name) (...)` drops s_cluster; entry["cluster"] should still be set."""
    _mock_response(
        mocker,
        [
            # Aggregated series - only net_host_name survives.
            {"metric": {"net_host_name": "n1"}, "values": [["1744734450", "1"]]},
        ],
    )
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="example-cluster",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"up": "{name}"}},
    )
    obs = conn.collect(_ctx(nodes=("n1",)))
    body = json.loads(obs[0]["body"])
    assert len(body) == 1
    assert "cluster: example-cluster," in body[0]["body"]


# --- batch collection -------------------------------------------------------


def _node_and_value(entry: dict) -> tuple[str, str]:
    return entry["resources"][0]["hostname"], entry["body"].split("metric_value: ", 1)[1].split(",", 1)[0]


def _batch_ctx(ctx_id: str, nodes: list[str], *, start: datetime, end: datetime) -> dict:
    return {"id": ctx_id, "nodes": nodes, "start_time": start, "end_time": end}


def _batch_conn(mocker, result, **kwargs):
    post = _mock_response(mocker, result)
    defaults = dict(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": '{name}{{s_cluster="{cluster}", net_host_name=~"{nodes}"}}'}},
        lookahead_seconds=300,
    )
    defaults.update(kwargs)
    return PromQLMetricsConnector(**defaults), post


def test_collect_batch_queries_union_of_nodes_once_and_slices_per_context(mocker):
    t0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    ts = lambda minutes: int((t0 + timedelta(minutes=minutes)).timestamp())  # noqa: E731
    result = [
        {"metric": {"net_host_name": "n1", "s_cluster": "c"}, "values": [[ts(5), "1"], [ts(40), "2"]]},
        {"metric": {"net_host_name": "n2", "s_cluster": "c"}, "values": [[ts(63), "3"]]},
        {"metric": {"net_host_name": "n3", "s_cluster": "c"}, "values": [[ts(10), "4"]]},
    ]
    conn, post = _batch_conn(mocker, result)
    contexts = [
        _batch_ctx("a", ["n1"], start=t0, end=t0 + timedelta(minutes=30)),
        _batch_ctx("b", ["n2", "n1"], start=t0 + timedelta(minutes=30), end=t0 + timedelta(hours=1)),
    ]

    observations = conn.collect_batch(contexts)

    assert post.call_count == 1
    data = post.call_args.kwargs["data"]
    assert data["query"] == 'm{s_cluster="c", net_host_name=~"n1|n2"}'
    assert data["start"] == int(t0.timestamp())
    assert data["end"] == int((t0 + timedelta(hours=1, minutes=5)).timestamp())
    body_a = json.loads(observations["a"][0]["body"])
    assert [_node_and_value(entry) for entry in body_a] == [("n1", "1")]
    assert observations["a"][0]["context"]["time_interval"] == {"start": t0, "end": t0 + timedelta(minutes=30)}
    body_b = json.loads(observations["b"][0]["body"])
    # n1's second sample falls inside b's window; n2's sample sits in the lookahead.
    assert sorted(_node_and_value(entry) for entry in body_b) == [("n1", "2"), ("n2", "3")]


def test_collect_batch_chunks_nodes_and_keeps_successful_chunks(mocker):
    t0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    def respond(*_args, **kwargs):
        fake = mocker.Mock()
        fake.raise_for_status = mocker.Mock()
        if "n3" in kwargs["data"]["query"]:
            fake.raise_for_status.side_effect = RuntimeError("chunk failed")
            return fake
        fake.json.return_value = {
            "status": "success",
            "data": {"result": [{"metric": {"net_host_name": "n1"}, "values": [[int(t0.timestamp()), "1"]]}]},
        }
        return fake

    post = mocker.patch("requests.Session.post", side_effect=respond)
    conn = PromQLMetricsConnector(
        endpoint="https://prom.example",
        auth=TokenAuth("tok"),
        cluster="c",
        source=Source.NODE_METRICS,
        templates={Source.NODE_METRICS: {"m": '{name}{{net_host_name=~"{nodes}"}}'}},
        node_chunk_size=2,
    )

    observations = conn.collect_batch([_batch_ctx("a", ["n1", "n2", "n3"], start=t0, end=t0 + timedelta(minutes=5))])

    assert post.call_count == 2
    assert [json.loads(obs["body"])[0]["resources"][0]["hostname"] for obs in observations["a"]] == ["n1"]


def test_collect_batch_raises_when_every_query_fails(mocker):
    t0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    conn, _post = _batch_conn(mocker, [])
    fake = mocker.Mock()
    fake.raise_for_status.side_effect = RuntimeError("down")
    mocker.patch("requests.Session.post", return_value=fake)

    with pytest.raises(RuntimeError, match="down"):
        conn.collect_batch([_batch_ctx("a", ["n1"], start=t0, end=t0 + timedelta(minutes=5))])


def test_collect_batch_without_nodes_or_contexts(mocker):
    t0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    conn, post = _batch_conn(mocker, [])

    assert conn.collect_batch([]) == {}
    assert conn.collect_batch([_batch_ctx("a", [], start=t0, end=t0 + timedelta(minutes=5))]) == {"a": []}
    assert post.call_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"step_seconds": 0}, {"node_chunk_size": 0}, {"lookahead_seconds": -1}],
)
def test_connector_rejects_invalid_batch_settings(kwargs):
    with pytest.raises(ValueError):
        PromQLMetricsConnector(
            endpoint="https://prom.example",
            cluster="c",
            source=Source.NODE_METRICS,
            templates={Source.NODE_METRICS: {"m": "{name}"}},
            **kwargs,
        )
