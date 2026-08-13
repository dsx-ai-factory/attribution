# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from datetime import datetime, timezone

import pytest

from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.connectors.logs.loki import LokiConnector
from attribution_blueprint.types import Format, Source


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


def test_syslog_body_is_flat_dict_list(mocker):
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
    obs = conn.collect(_ctx())
    assert len(obs) == 1
    o = obs[0]
    assert o["context"]["format"] == Format.LOKI_DICT_LIST
    decoded = json.loads(o["body"])
    assert decoded == [
        {"timestamp": 1, "body": "kernel: oops", "node": "n1"},
        {"timestamp": 2, "body": "imex stop", "node": "n2"},
    ]


def test_applog_format_is_raw_loki_streams(mocker):
    _mock_response(mocker, [{"stream": {}, "values": [["1", "x"]]}])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.APPLOG,
        query_kwargs={"app": "myapp", "service_name": "x"},
        templates={Source.APPLOG: ["{{}}"]},
    )
    obs = conn.collect(_ctx())
    assert obs[0]["context"]["format"] == Format.RAW_LOKI_STREAMS


def test_applog_renames_node_label_to_node_name_for_service(mocker):
    """attribution_lib.observation.processing reads `stream["node_name"]` only.
    The connector's configured node_label (default `host_name`) must be
    rewritten to `node_name` in each forwarded stream's labels, else every
    APPLOG entry is attributed with node=None."""
    streams = [
        {"stream": {"host_name": "n1", "app": "x"}, "values": [["1", "line1"]]},
        {"stream": {"host_name": "n2", "app": "x"}, "values": [["2", "line2"]]},
    ]
    _mock_response(mocker, streams)
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.APPLOG,
        query_kwargs={"app": "myapp", "service_name": "x"},
        templates={Source.APPLOG: ["{{}}"]},
    )
    obs = conn.collect(_ctx())
    decoded = json.loads(obs[0]["body"])
    assert {s["stream"].get("node_name") for s in decoded} == {"n1", "n2"}


def test_applog_node_label_already_node_name_is_noop(mocker):
    """When the configured node_label is already `node_name`, the encoded
    streams pass through unchanged."""
    streams = [{"stream": {"node_name": "n1"}, "values": [["1", "x"]]}]
    _mock_response(mocker, streams)
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.APPLOG,
        query_kwargs={"app": "myapp", "service_name": "x"},
        templates={Source.APPLOG: ["{{}}"]},
        node_label="node_name",
    )
    obs = conn.collect(_ctx())
    decoded = json.loads(obs[0]["body"])
    assert decoded[0]["stream"]["node_name"] == "n1"


def test_mtls_auth_sets_session_cert(mocker):
    _mock_response(mocker, [])
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=MTLSAuth(cert_path="/c", key_path="/k", ca_cert_path="/ca"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ["{{}}"]},
    )
    assert conn._session.cert == ("/c", "/k")
    assert conn._session.verify == "/ca"


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


def test_paginates_when_response_at_limit(mocker):
    # Page 1: limit entries returned (saturated) -> connector must re-query.
    # Page 2: fewer than limit -> connector stops.
    n = 3  # use a small limit so the test is readable
    end_ns = int(_ctx()["end_time"].timestamp() * 1e9)
    # Timestamps must fall inside the query window for the pagination loop to advance.
    ts = [end_ns - 100, end_ns - 200, end_ns - 300]  # newest -> oldest
    page1 = [{"stream": {"host_name": "n1"}, "values": [[str(ts[i]), f"e{i}"] for i in range(n)]}]
    page2 = [{"stream": {"host_name": "n1"}, "values": [[str(end_ns - 500), "last"]]}]
    fake1 = mocker.Mock()
    fake1.json.return_value = {"data": {"result": page1}}
    fake1.raise_for_status = mocker.Mock()
    fake2 = mocker.Mock()
    fake2.json.return_value = {"data": {"result": page2}}
    fake2.raise_for_status = mocker.Mock()
    post = mocker.patch("requests.Session.post", side_effect=[fake1, fake2])

    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        limit=n,
    )
    obs = conn.collect(_ctx())
    assert post.call_count == 2
    # Second call's `end` is oldest_returned_ns - 1 from page 1.
    assert post.call_args_list[1].kwargs["data"]["end"] == ts[-1] - 1
    decoded = json.loads(obs[0]["body"])
    assert [e["body"] for e in decoded] == ["e0", "e1", "e2", "last"]


def test_pagination_stops_when_window_collapses(mocker):
    # Saturated page whose oldest entry equals start_ns: nothing left to fetch.
    n = 2
    start_ns = int(_ctx()["start_time"].timestamp() * 1e9)
    page = [{"stream": {"host_name": "n1"}, "values": [[str(start_ns + 1), "a"], [str(start_ns), "b"]]}]
    fake = mocker.Mock()
    fake.json.return_value = {"data": {"result": page}}
    fake.raise_for_status = mocker.Mock()
    post = mocker.patch("requests.Session.post", return_value=fake)
    conn = LokiConnector(
        endpoint="https://loki.example",
        auth=TokenAuth("tok"),
        source=Source.SYSLOG,
        query_kwargs={"service_name": "x"},
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        limit=n,
    )
    conn.collect(_ctx())
    assert post.call_count == 1  # didn't try to page below start_ns


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
    """Two pages with overlapping (stream, ts) pairs should not double-count."""
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
        templates={Source.SYSLOG: ['{{service_name="{service_name}"}}']},
        limit=3,  # saturated on page1, triggers pagination
    )
    obs = conn.collect(_ctx())
    body = json.loads(obs[0]["body"])
    # Only the three unique (timestamp, body) pairs survive.
    assert sorted((e["timestamp"], e["body"]) for e in body) == [(1, "a"), (2, "b"), (3, "c")]


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
    assert body[0]["node"] == "n1"
