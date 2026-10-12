# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timedelta, timezone

import pytest

from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.connectors.logs.loki import LokiConnector
from attribution_blueprint.types import Source


def _ctx():
    return {
        "id": "sluid-123",
        "nodes": ["node001"],
        "start_time": datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 12, 30, 0, tzinfo=timezone.utc),
    }


def _mock_response(mocker, streams):
    fake = mocker.Mock()
    fake.json.return_value = {"data": {"result": streams}}
    fake.raise_for_status = mocker.Mock()
    return mocker.patch("requests.Session.post", return_value=fake)


def test_renders_query_and_passes_time_window_in_ns(mocker):
    post = _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example/",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "cluster-x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
    )
    ctx = _ctx()
    ctx["nodes"] = []  # cluster-wide query, no host_name filter injected
    conn.collect(ctx)
    args, kwargs = post.call_args
    assert args[0] == "https://loki.example/loki/api/v1/query_range"
    assert kwargs["data"]["query"] == '{service_name="cluster-x"}'
    assert kwargs["data"]["start"] == int(ctx["start_time"].timestamp() * 1e9)
    assert kwargs["data"]["end"] == int(ctx["end_time"].timestamp() * 1e9)
    assert kwargs["data"]["limit"] == 5000


def test_returns_empty_when_no_streams(mocker):
    _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ["{{}}"]},
    )
    assert conn.collect(_ctx()) == []


def test_syslog_body_is_entry_timeseries_with_stream_node(mocker):
    streams_q1 = [{"stream": {"host_name": "n1"}, "values": [["1", "kernel: oops"]]}]
    streams_q2 = [{"stream": {"host_name": "n2"}, "values": [["2", "imex stop"]]}]
    fake1 = mocker.Mock()
    fake1.json.return_value = {"data": {"result": streams_q1}}
    fake1.raise_for_status = mocker.Mock()
    fake2 = mocker.Mock()
    fake2.json.return_value = {"data": {"result": streams_q2}}
    fake2.raise_for_status = mocker.Mock()
    mocker.patch("requests.Session.post", side_effect=[fake1, fake2])

    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{a="{service_name}"}}', '{{b="{service_name}"}}']},
    )
    ctx = _ctx()
    obs = conn.collect(ctx)
    assert len(obs) == 1
    o = obs[0]
    assert o["context"]["time_interval"] == {"start": ctx["start_time"], "end": ctx["end_time"]}
    assert json.loads(o["body"]) == [
        {"body": "kernel: oops", "ts": 1, "resources": [{"kind": "node", "hostname": "n1"}]},
        {"body": "imex stop", "ts": 2, "resources": [{"kind": "node", "hostname": "n2"}]},
    ]


def _applog_connector(**kwargs) -> LokiConnector:
    return LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.APPLOG,
        query_kwargs={"app": "myapp", "service_name": "x"},
        templates={Source.APPLOG: ["{{}}"]},
        **kwargs,
    )


def test_applog_body_is_unbounded_entry_timeseries_with_labelled_resources(mocker):
    streams = [
        {"stream": {"host_name": "n1", "pod": "trainer-0"}, "values": [["1", "[rank 0] step 1"]]},
        {"stream": {"app": "x"}, "values": [["2", "[rank 1] no labels"], ["not-a-timestamp", "dropped"]]},
    ]
    _mock_response(mocker, streams)

    obs = _applog_connector().collect(_ctx())

    assert obs[0]["context"]["time_interval"] == {"start": None, "end": None}
    assert json.loads(obs[0]["body"]) == [
        {
            "body": "[rank 0] step 1",
            "ts": 1,
            "resources": [{"kind": "node", "hostname": "n1"}, {"kind": "replica", "name": "trainer-0"}],
        },
        {"body": "[rank 1] no labels", "ts": 2},
    ]


def test_applog_json_payloads_unwrap_line_and_hostname(mocker):
    streams = [
        {
            "stream": {"app": "x"},
            "values": [
                ["1", json.dumps({"body": "[rank 1] oops", "attributes": {"hostname": "n2"}})],
                ["2", json.dumps({"data": {"message": "terminate called", "hostname": "n3"}})],
                ["3", json.dumps({"unrelated": True})],
            ],
        },
        {
            "stream": {"host_name": "n1"},
            "values": [["4", json.dumps({"body": "kept", "attributes": {"hostname": "n9"}})]],
        },
    ]
    _mock_response(mocker, streams)

    obs = _applog_connector().collect(_ctx())

    assert json.loads(obs[0]["body"]) == [
        {"body": "[rank 1] oops", "ts": 1, "resources": [{"kind": "node", "hostname": "n2"}]},
        {"body": "terminate called", "ts": 2, "resources": [{"kind": "node", "hostname": "n3"}]},
        {"body": '{"unrelated": true}', "ts": 3},
        {"body": "kept", "ts": 4, "resources": [{"kind": "node", "hostname": "n1"}]},
    ]


def test_mtls_auth_sets_session_cert(mocker):
    _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=MTLSAuth(cert_path="/c", key_path="/k", ca_cert_path="/ca"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ["{{}}"]},
    )
    assert conn._client._session.cert == ("/c", "/k")
    assert conn._client._session.verify == "/ca"


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="no templates registered"):
        LokiConnector(
            endpoint="https://loki.example",
            auth=TokenAuth("tok"),
            source=Source.APPLOG,
            query_kwargs={},
            templates={},  # empty -> no templates for any source
        )


def test_node_filter_does_not_backslash_escape_hyphens_in_hostnames(mocker):
    """Loki uses RE2, which rejects `\\-` as 'invalid char escape'. Python's
    re.escape() backslashes hyphens, which produces 400 on every SYSLOG call
    against real Loki. The connector must escape only RE2 metacharacters."""
    post = _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
    )
    ctx = _ctx()
    ctx["nodes"] = ["example-node-014", "example-node-015"]
    conn.collect(ctx)
    sent = post.call_args.kwargs["data"]["query"]
    assert "\\-" not in sent, f"hyphens should not be backslash-escaped: {sent}"
    # Dots still must be escaped: they're an RE2 meta meaning "any char".
    ctx["nodes"] = ["host.example.com"]
    post.reset_mock()
    conn.collect(ctx)
    sent2 = post.call_args.kwargs["data"]["query"]
    assert "host\\.example\\.com" in sent2


def test_injects_node_filter_into_first_selector(mocker):
    post = _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}", appName=`kernel`}} |~ `Xid`']},
    )
    ctx = _ctx()
    ctx["nodes"] = ["node001", "node002", "node003"]
    conn.collect(ctx)
    rendered = post.call_args.kwargs["data"]["query"]
    assert 'host_name=~"node001|node002|node003"' in rendered
    # Original selector body and pipeline preserved
    assert 'service_name="x"' in rendered
    assert "appName=`kernel`" in rendered
    assert "|~ `Xid`" in rendered


def test_skips_node_filter_when_nodes_empty(mocker):
    post = _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
    )
    ctx = _ctx()
    ctx["nodes"] = []
    conn.collect(ctx)
    assert "host_name=~" not in post.call_args.kwargs["data"]["query"]


def test_node_label_can_be_disabled(mocker):
    post = _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        node_label=None,
    )
    conn.collect(_ctx())
    assert "host_name=~" not in post.call_args.kwargs["data"]["query"]


def test_bisects_saturated_pages_forward(mocker):
    """A page that returns exactly `limit` entries hides its tail, so the client
    splits the interval and reads both halves instead of paging backwards."""
    n = 2
    ctx = _ctx()
    start_ns = int(ctx["start_time"].timestamp() * 1e9)
    end_ns = int(ctx["end_time"].timestamp() * 1e9)
    lines = {start_ns + 100: "a", start_ns + 200: "b", end_ns - 300: "c"}

    def respond(*_args, **kwargs):
        lo, hi = int(kwargs["data"]["start"]), int(kwargs["data"]["end"])
        values = [[str(ts), body] for ts, body in sorted(lines.items()) if lo <= ts <= hi][:n]
        fake = mocker.Mock()
        fake.json.return_value = {"data": {"result": [{"stream": {"host_name": "n1"}, "values": values}]}}
        fake.raise_for_status = mocker.Mock()
        return fake

    post = mocker.patch("requests.Session.post", side_effect=respond)
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        limit=n,
    )
    obs = conn.collect(ctx)
    assert post.call_count >= 3
    assert all(call.kwargs["data"]["direction"] == "forward" for call in post.call_args_list)
    decoded = json.loads(obs[0]["body"])
    assert [e["body"] for e in decoded] == ["a", "b", "c"]


def test_substitutes_job_id_into_template(mocker):
    post = _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.APPLOG,
        query_kwargs={"app": "ml-jobs", "service_name": "x"},
        templates={Source.APPLOG: ['{{app="{app}"}} | slurm_job_id = `{job_id}`']},
    )
    conn.collect(_ctx())
    rendered = post.call_args.kwargs["data"]["query"]
    assert 'app="ml-jobs"' in rendered
    assert "slurm_job_id = `sluid-123`" in rendered


def test_injects_node_filter_into_every_top_level_selector(mocker):
    """Multi-selector LogQL (e.g. ratio queries) must scope every stream selector."""
    post = _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={
            Source.SYSLOG: [
                'count_over_time({{service_name="{service_name}", appName="kernel"}}[1m]) '
                '/ count_over_time({{service_name="{service_name}"}}[1m])'
            ]
        },
    )
    ctx = _ctx()
    ctx["nodes"] = ["n1", "n2"]
    conn.collect(ctx)
    rendered = post.call_args.kwargs["data"]["query"]
    # Both selectors injected, not just the first.
    assert rendered.count('host_name=~"n1|n2"') == 2


def test_node_filter_respects_strings_in_selector():
    """A literal `}` inside a label value must not terminate the selector early.
    Tests `_inject_node_filter` directly because such literals can't survive str.format()."""
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},  # benign for ctor
    )
    # Pre-formatted query with a literal `}` inside a quoted value:
    pre_formatted = '{service_name="ab}cd", svc="ok"}'
    out = conn._inject_node_filter(pre_formatted, ["n1"])
    # Filter must appear AT the real closing brace, not at the `}` inside the quoted value.
    assert out.endswith('host_name=~"n1"}')
    # The inner `}` must still be inside the quoted value.
    assert 'service_name="ab}cd"' in out


def test_node_names_are_regex_escaped(mocker):
    """FQDN hostnames must not over-match: `.` and `[`/`]` are regex metacharacters."""
    post = _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
    )
    ctx = _ctx()
    ctx["nodes"] = ["host.example.com", "node[01]"]
    conn.collect(ctx)
    rendered = post.call_args.kwargs["data"]["query"]
    assert r"host\.example\.com" in rendered
    assert r"node\[01\]" in rendered


def test_streams_merged_and_values_deduped(mocker):
    """Two templates returning overlapping (stream, ts) pairs must not double-count."""
    page1 = [{"stream": {"host_name": "n1", "app": "x"}, "values": [["3", "c"], ["2", "b"], ["1", "a"]]}]
    page2 = [{"stream": {"host_name": "n1", "app": "x"}, "values": [["2", "b"], ["1", "a"]]}]
    fake1 = mocker.Mock()
    fake1.json.return_value = {"data": {"result": page1}}
    fake1.raise_for_status = mocker.Mock()
    fake2 = mocker.Mock()
    fake2.json.return_value = {"data": {"result": page2}}
    fake2.raise_for_status = mocker.Mock()
    mocker.patch("requests.Session.post", side_effect=[fake1, fake2])

    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{a="{service_name}"}}', '{{b="{service_name}"}}']},
    )
    obs = conn.collect(_ctx())
    body = json.loads(obs[0]["body"])
    # Only the three unique (timestamp, body) pairs survive, in timestamp order.
    assert [(e["ts"], e["body"]) for e in body] == [(1, "a"), (2, "b"), (3, "c")]


def test_string_template_accepted_for_pdf_compat():
    """PDF v1 spec types templates as dict[Source, str]; the connector must accept it."""
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: '{{service_name="{service_name}"}}'},  # str, not list
    )
    assert conn._templates == ['{{service_name="{service_name}"}}']


def test_custom_node_label_filters_and_decodes(mocker):
    """node_label=<X> routes both the injected filter and the per-entry node read."""
    _mock_response(
        mocker,
        [{"stream": {"hostname": "n1"}, "values": [["1", "msg"]]}],
    )
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        node_label="hostname",
    )
    obs = conn.collect(_ctx())
    body = json.loads(obs[0]["body"])
    assert body[0]["resources"] == [{"kind": "node", "hostname": "n1"}]


# --- batch collection -------------------------------------------------------


def _batch_ctx(ctx_id: str, nodes: list[str], *, end: datetime, minutes: int = 30) -> dict:
    return {"id": ctx_id, "nodes": nodes, "start_time": end - timedelta(minutes=minutes), "end_time": end}


def _batch_conn(mocker, streams, **kwargs):
    post = _mock_response(mocker, streams)
    defaults = dict(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "cluster-x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
    )
    defaults.update(kwargs)
    return LokiConnector(**defaults), post


def test_collect_batch_fetches_one_window_and_slices_by_node_and_time(mocker):
    end = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    ns = lambda dt: int(dt.timestamp() * 1e9)  # noqa: E731
    streams = [
        {"stream": {"host_name": "n1"}, "values": [[str(ns(end - timedelta(minutes=10))), "n1 inside"]]},
        {"stream": {"host_name": "n2"}, "values": [[str(ns(end - timedelta(minutes=50))), "n2 old"]]},
        {"stream": {"host_name": "n3"}, "values": [[str(ns(end - timedelta(minutes=5))), "n3 other"]]},
        {"stream": {"service": "no-node"}, "values": [[str(ns(end - timedelta(minutes=5))), "dropped"]]},
    ]
    conn, post = _batch_conn(mocker, streams)
    contexts = [
        _batch_ctx("a", ["n1", "n2"], end=end, minutes=30),
        _batch_ctx("b", ["N3"], end=end - timedelta(minutes=2), minutes=10),
    ]

    result = conn.collect_batch(contexts)

    assert post.call_count == 1
    data = post.call_args.kwargs["data"]
    assert data["query"] == '{service_name="cluster-x"}'  # cluster-wide, no node filter
    assert data["start"] == ns(end - timedelta(minutes=30))
    assert data["end"] == ns(end)
    assert data["direction"] == "forward"
    decoded_a = json.loads(result["a"][0]["body"])
    assert [entry["body"] for entry in decoded_a] == ["n1 inside"]  # n2 is before a's window
    assert result["a"][0]["context"]["time_interval"] == {"start": end - timedelta(minutes=30), "end": end}
    decoded_b = json.loads(result["b"][0]["body"])
    assert [entry["resources"] for entry in decoded_b] == [[{"kind": "node", "hostname": "n3"}]]


def test_collect_batch_trims_each_context_to_its_trailing_window(mocker):
    end = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    ns = lambda dt: int(dt.timestamp() * 1e9)  # noqa: E731
    streams = [
        {
            "stream": {"host_name": "n1"},
            "values": [[str(ns(end - timedelta(hours=3))), "too old"], [str(ns(end - timedelta(hours=1))), "recent"]],
        },
    ]
    conn, post = _batch_conn(mocker, streams, batch_window_hours=2.0)

    result = conn.collect_batch([_batch_ctx("long", ["n1"], end=end, minutes=6 * 60)])

    # Four 30-minute query windows cover the two-hour trailing window.
    assert post.call_args_list[0].kwargs["data"]["start"] == ns(end - timedelta(hours=2))
    assert post.call_args_list[-1].kwargs["data"]["end"] == ns(end)
    assert post.call_count == 4
    assert [entry["body"] for entry in json.loads(result["long"][0]["body"])] == ["recent"]


def test_collect_batch_returns_empty_lists_for_contexts_without_evidence(mocker):
    end = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    conn, _post = _batch_conn(mocker, [])

    assert conn.collect_batch([_batch_ctx("a", ["n1"], end=end)]) == {"a": []}
    assert conn.collect_batch([]) == {}


def test_collect_batch_applog_emits_timeseries_with_node_resources(mocker):
    end = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    ns = lambda dt: int(dt.timestamp() * 1e9)  # noqa: E731
    streams = [
        {
            "stream": {"host_name": "n1", "app": "train"},
            "values": [[str(ns(end - timedelta(minutes=1))), "l1"], [str(ns(end - timedelta(minutes=2))), "l2"]],
        },
        {"stream": {"host_name": "n2", "app": "train"}, "values": [[str(ns(end - timedelta(minutes=1))), "other"]]},
    ]
    conn, _post = _batch_conn(
        mocker,
        streams,
        source=Source.APPLOG,
        templates={Source.APPLOG: ['{{service_name="{service_name}", app="train"}}']},
    )

    result = conn.collect_batch([_batch_ctx("a", ["n1"], end=end)])

    body = json.loads(result["a"][0]["body"])
    assert result["a"][0]["context"]["time_interval"] == {"start": None, "end": None}
    assert body == [
        {"body": "l2", "ts": ns(end - timedelta(minutes=2)), "resources": [{"kind": "node", "hostname": "n1"}]},
        {"body": "l1", "ts": ns(end - timedelta(minutes=1)), "resources": [{"kind": "node", "hostname": "n1"}]},
    ]


def test_collect_batch_falls_back_to_per_context_for_job_scoped_templates(mocker):
    end = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    conn, post = _batch_conn(
        mocker,
        [],
        source=Source.APPLOG,
        templates={Source.APPLOG: ['{{service_name="{service_name}"}} | slurm_job_id = `{job_id}`']},
    )

    result = conn.collect_batch([_batch_ctx("j1", ["n1"], end=end), _batch_ctx("j2", ["n2"], end=end)])

    assert result == {"j1": [], "j2": []}
    assert post.call_count == 2
    assert "`j1`" in post.call_args_list[0].kwargs["data"]["query"]
    assert "`j2`" in post.call_args_list[1].kwargs["data"]["query"]


def test_collect_batch_node_filter_chunks_union_of_nodes(mocker):
    end = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    conn, post = _batch_conn(mocker, [], batch_node_filter=True, node_chunk_size=2)

    conn.collect_batch([_batch_ctx("a", ["n1", "n2"], end=end), _batch_ctx("b", ["n3"], end=end)])

    queries = sorted(call.kwargs["data"]["query"] for call in post.call_args_list)
    assert queries == [
        '{service_name="cluster-x", host_name=~"n1|n2"}',
        '{service_name="cluster-x", host_name=~"n3"}',
    ]


def test_collect_batch_keeps_partial_result_when_page_budget_runs_out(mocker):
    end = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    ns = lambda dt: int(dt.timestamp() * 1e9)  # noqa: E731

    def respond(*_args, **kwargs):
        start_ns = int(kwargs["data"]["start"])
        fake = mocker.Mock()
        fake.json.return_value = {
            "data": {
                "result": [{"stream": {"host_name": "n1"}, "values": [[str(start_ns), "a"], [str(start_ns + 1), "b"]]}]
            }
        }
        fake.raise_for_status = mocker.Mock()
        return fake

    post = mocker.patch("requests.Session.post", side_effect=respond)
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        limit=2,
    )
    conn._client._max_pages = 1

    result = conn.collect_batch([_batch_ctx("a", ["n1"], end=end, minutes=30)])

    assert post.call_count == 1
    assert [entry["body"] for entry in json.loads(result["a"][0]["body"])] == ["a", "b"]
    assert json.loads(result["a"][0]["body"])[0]["ts"] == ns(end - timedelta(minutes=30))


@pytest.mark.parametrize("kwargs", [{"batch_window_hours": -1}, {"node_chunk_size": 0}])
def test_connector_rejects_invalid_batch_settings(kwargs):
    with pytest.raises(ValueError):
        LokiConnector(
            endpoint="https://loki.example",
            auth=TokenAuth("tok"),
            source=Source.SYSLOG,
            query_kwargs={},
            templates={Source.SYSLOG: ["{{}}"]},
            **kwargs,
        )


def test_collect_batch_reuses_cached_buckets_across_calls(mocker, tmp_path):
    from attribution_blueprint.connectors.logs.loki import FileRangeCacheStore, LokiRangeCache

    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    end = now - timedelta(hours=1)
    ns = lambda dt: int(dt.timestamp() * 1e9)  # noqa: E731

    def respond(*_args, **kwargs):
        start_ns = int(kwargs["data"]["start"])
        fake = mocker.Mock()
        fake.json.return_value = {
            "data": {"result": [{"stream": {"host_name": "n1"}, "values": [[str(start_ns + 1), "line"]]}]}
        }
        fake.raise_for_status = mocker.Mock()
        return fake

    post = mocker.patch("requests.Session.post", side_effect=respond)
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        range_cache=LokiRangeCache(FileRangeCacheStore(tmp_path)),
        clock=lambda: now,
    )
    contexts = [_batch_ctx("a", ["n1"], end=end, minutes=10)]

    first = conn.collect_batch(contexts)
    second = conn.collect_batch(contexts)

    assert post.call_count == 1
    assert json.loads(first["a"][0]["body"])[0]["ts"] == ns(end - timedelta(minutes=10)) + 1
    assert second == first


def test_collect_batch_slices_a_long_context_from_the_live_fetch_and_cached_buckets(mocker, tmp_path):
    """Two polls, two hours apart, over a three-hour context: the second poll fetches only its
    trailing window but the context's slice also carries the bucket the first poll stored."""
    from attribution_blueprint.connectors.logs.loki import FileRangeCacheStore, LokiRangeCache

    ns = lambda dt: int(dt.timestamp() * 1e9)  # noqa: E731
    context_start = datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)
    first_end = context_start + timedelta(hours=1)
    second_end = context_start + timedelta(hours=3)
    lines = {
        ns(context_start + timedelta(minutes=30)): "first-poll",
        ns(second_end - timedelta(minutes=30)): "second-poll",
    }

    def respond(*_args, **kwargs):
        lo, hi = int(kwargs["data"]["start"]), int(kwargs["data"]["end"])
        values = [[str(ts), body] for ts, body in sorted(lines.items()) if lo <= ts <= hi]
        fake = mocker.Mock()
        fake.json.return_value = {"data": {"result": [{"stream": {"host_name": "n1"}, "values": values}]}}
        fake.raise_for_status = mocker.Mock()
        return fake

    post = mocker.patch("requests.Session.post", side_effect=respond)
    clock = [first_end + timedelta(hours=1)]
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        batch_window_hours=1.0,
        range_cache=LokiRangeCache(FileRangeCacheStore(tmp_path)),
        clock=lambda: clock[0],
    )

    first = conn.collect_batch([{"id": "job", "nodes": ["n1"], "start_time": context_start, "end_time": first_end}])
    fetched_after_first = post.call_count
    clock[0] = second_end + timedelta(hours=1)
    second = conn.collect_batch([{"id": "job", "nodes": ["n1"], "start_time": context_start, "end_time": second_end}])

    assert [e["body"] for e in json.loads(first["job"][0]["body"])] == ["first-poll"]
    # The second poll fetched only [second_end - 1 h, second_end] ...
    later_calls = post.call_args_list[fetched_after_first:]
    assert min(int(c.kwargs["data"]["start"]) for c in later_calls) == ns(second_end - timedelta(hours=1))
    # ... yet the context's observation spans its whole window and carries both lines.
    body = json.loads(second["job"][0]["body"])
    assert [e["body"] for e in body] == ["first-poll", "second-poll"]
    assert second["job"][0]["context"]["time_interval"] == {"start": context_start, "end": second_end}


def test_collect_batch_reads_cached_history_node_scoped_per_context(mocker, tmp_path):
    """A later batch gives each context only its own node's cached history from one node-scoped read."""
    from attribution_blueprint.connectors.logs.loki import FileRangeCacheStore, LokiRangeCache

    ns = lambda dt: int(dt.timestamp() * 1e9)  # noqa: E731
    start = datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)
    first_end = start + timedelta(hours=1)
    second_end = start + timedelta(hours=3)
    lines = {
        (ns(start + timedelta(minutes=30)), "n1"): "n1-deep",
        (ns(start + timedelta(minutes=30)), "n2"): "n2-deep",
    }

    def respond(*_args, **kwargs):
        lo, hi = int(kwargs["data"]["start"]), int(kwargs["data"]["end"])
        streams = [
            {"stream": {"host_name": node}, "values": [[str(ts), body]]}
            for (ts, node), body in lines.items()
            if lo <= ts <= hi
        ]
        fake = mocker.Mock()
        fake.json.return_value = {"data": {"result": streams}}
        fake.raise_for_status = mocker.Mock()
        return fake

    mocker.patch("requests.Session.post", side_effect=respond)
    clock = [first_end + timedelta(hours=1)]
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        batch_window_hours=1.0,
        range_cache=LokiRangeCache(FileRangeCacheStore(tmp_path)),
        clock=lambda: clock[0],
    )

    conn.collect_batch(
        [
            {"id": "a", "nodes": ["n1"], "start_time": start, "end_time": first_end},
            {"id": "b", "nodes": ["n2"], "start_time": start, "end_time": first_end},
        ]
    )
    clock[0] = second_end + timedelta(hours=1)
    result = conn.collect_batch(
        [
            {"id": "a", "nodes": ["n1"], "start_time": start, "end_time": second_end},
            {"id": "b", "nodes": ["n2"], "start_time": start, "end_time": second_end},
        ]
    )

    assert [entry["body"] for entry in json.loads(result["a"][0]["body"])] == ["n1-deep"]
    assert [entry["body"] for entry in json.loads(result["b"][0]["body"])] == ["n2-deep"]


def test_every_exported_name_is_importable():
    """`__all__` must name only what the package defines.

    Main removed LokiSyslogEntry when syslog moved to entry_timeseries; an
    export list that still named it would make a star import raise.
    """
    import attribution_blueprint.connectors.logs.loki as loki

    assert [name for name in loki.__all__ if not hasattr(loki, name)] == []


def test_bundled_syslog_queries_cover_panic_gsp_reboot_and_link_down():
    from attribution_blueprint.connectors.logs.loki.templates import TEMPLATES

    queries = "\n".join(TEMPLATES[Source.SYSLOG])
    for term in (
        "Kernel panic",
        "NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR",
        "is rebooting(?: with kexec)?",
        r"Link\s+DOWN\b",
    ):
        assert term in queries


def test_long_window_is_queried_in_thirty_minute_slices_newest_first(mocker):
    post = _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
    )
    ctx = _ctx()
    ctx["end_time"] = datetime(2026, 5, 1, 13, 10, 0, tzinfo=timezone.utc)
    conn.collect(ctx)
    windows = [(c.kwargs["data"]["start"], c.kwargs["data"]["end"]) for c in post.call_args_list]
    s = int(ctx["start_time"].timestamp() * 1e9)
    half_hour = 30 * 60 * 10**9
    e = int(ctx["end_time"].timestamp() * 1e9)
    # Newest first, so the page cap drops the oldest entries. Windows do not overlap.
    assert windows == [(e - half_hour, e), (e - 2 * half_hour - 1, e - half_hour - 1), (s, e - 2 * half_hour - 2)]


def test_page_cap_bounds_all_windows_of_a_long_query(mocker):
    """Every multi-instant response is saturated, so only the overall MAX_PAGES cap stops the walk."""
    from attribution_blueprint.connectors.logs.loki.connector import MAX_PAGES

    def saturated(url, data, timeout):
        fake = mocker.Mock()
        # A single instant comes back unsaturated, so bisection can always finish it.
        values = [[str(data["end"]), "x"]] if data["start"] < data["end"] else []
        fake.json.return_value = {"data": {"result": [{"stream": {"host_name": "n1"}, "values": values}]}}
        return fake

    post = mocker.patch("requests.Session.post", side_effect=saturated)
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        limit=1,
    )
    ctx = _ctx()
    ctx["end_time"] = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)  # 7 days = 336 windows
    conn.collect(ctx)
    assert post.call_count == MAX_PAGES == 500
    # The cap spent its budget on the newest entries.
    assert post.call_args_list[0].kwargs["data"]["end"] == int(ctx["end_time"].timestamp() * 1e9)


def test_quiet_week_long_query_visits_every_window_within_budget(mocker):
    """Empty windows cost a page each; a 7-day job (336 windows) still fits the budget."""
    post = _mock_response(mocker, [])
    warning = mocker.patch("attribution_blueprint.connectors.logs.loki.connector.logger.warning")
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
    )
    ctx = _ctx()
    ctx["end_time"] = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
    conn.collect(ctx)
    assert post.call_count == 336
    assert post.call_args_list[-1].kwargs["data"]["start"] == int(ctx["start_time"].timestamp() * 1e9)
    warning.assert_not_called()


@pytest.mark.parametrize(("cap", "warned"), [(3, False), (2, True)])
def test_page_cap_warns_only_when_windows_remain(mocker, cap, warned):
    """Three quiet windows: a cap of exactly 3 finishes cleanly, a cap of 2 truncates."""
    mocker.patch("attribution_blueprint.connectors.logs.loki.connector.MAX_PAGES", cap)
    _mock_response(mocker, [])
    warning = mocker.patch("attribution_blueprint.connectors.logs.loki.connector.logger.warning")
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
    )
    ctx = _ctx()
    ctx["end_time"] = datetime(2026, 5, 1, 13, 30, 0, tzinfo=timezone.utc)  # 90 minutes
    conn.collect(ctx)
    assert warning.called == warned


def test_page_cap_warns_when_last_window_is_cut_short(mocker):
    """Running out mid-window is truncation even when no later window remains."""
    mocker.patch("attribution_blueprint.connectors.logs.loki.connector.MAX_PAGES", 2)

    def saturated(url, data, timeout):
        fake = mocker.Mock()
        fake.json.return_value = {
            "data": {"result": [{"stream": {"host_name": "n1"}, "values": [[str(data["end"]), "x"]]}]}
        }
        return fake

    mocker.patch("requests.Session.post", side_effect=saturated)
    warning = mocker.patch("attribution_blueprint.connectors.logs.loki.connector.logger.warning")
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        limit=1,
    )
    conn.collect(_ctx())
    warning.assert_called_once()
