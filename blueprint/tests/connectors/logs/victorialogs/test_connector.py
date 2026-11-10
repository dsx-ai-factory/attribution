# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import requests
from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.connectors.logs.victorialogs import (
    VictoriaLogsConnector,
    _build,
    connector as victorialogs,
)
from attribution_blueprint.exceptions import PartialCollectionError
from attribution_blueprint.orchestrator import AttributionOrchestrator
from attribution_blueprint.types import Source

FIXTURES = Path(__file__).parent / "fixtures"


def _context():
    return {
        "id": "1000001",
        "nodes": [f"node-{index:02d}" for index in range(1, 17)],
        "start_time": datetime(2026, 1, 15, 0, 0, tzinfo=UTC),
        "end_time": datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
    }


def _response(text: str, mocker):
    response = mocker.Mock()
    response.text = text
    response.raise_for_status = mocker.Mock()
    return response


def _row(timestamp: str, message: str, **fields):
    return {"_time": timestamp, "_msg": message, **fields}


def _ndjson(rows):
    return "\n".join(json.dumps(row) for row in rows)


def _connector(source=Source.SYSLOG, **kwargs):
    return VictoriaLogsConnector(
        endpoint="http://victorialogs.example/",
        source=source,
        query_kwargs={"service_name": "example-cluster", "app": "example-training"},
        templates={source: ['service_name:="{service_name}"']},
        **kwargs,
    )


def test_renders_templates_and_sends_inclusive_window_as_exclusive_end(mocker):
    post = mocker.patch(
        "requests.Session.post",
        return_value=_response("", mocker),
    )
    conn = VictoriaLogsConnector(
        endpoint="http://victorialogs.example/",
        source=Source.APPLOG,
        query_kwargs={"service_name": "example-cluster", "app": "example-training"},
        templates={Source.APPLOG: ['service_name:="{service_name}" app:="{app}" slurm_job_id:="{job_id}"']},
        limit=37,
    )
    conn.collect(_context())

    args, kwargs = post.call_args
    assert args[0] == "http://victorialogs.example/select/logsql/query"
    assert kwargs["data"] == {
        "query": (
            'host_name:in("node-01","node-02","node-03","node-04","node-05","node-06",'
            '"node-07","node-08","node-09","node-10","node-11","node-12","node-13",'
            '"node-14","node-15","node-16") '
            '(service_name:="example-cluster" app:="example-training" slurm_job_id:="1000001")'
        ),
        "start": 1768435200000000000,
        "end": 1768471200000000001,
        "limit": 37,
    }
    assert kwargs["timeout"] == 60.0


def test_missing_template_kwarg_warns_and_skips(mocker):
    """A skipped query records incomplete coverage even when it provides no observations."""
    post = mocker.patch("requests.Session.post")
    warning = mocker.patch.object(victorialogs.logger, "warning")
    conn = VictoriaLogsConnector(
        endpoint="http://victorialogs.example",
        source=Source.SYSLOG,
        query_kwargs={},
        templates={Source.SYSLOG: ['service_name:="{missing}"']},
    )

    with pytest.raises(PartialCollectionError, match="query 1: template missing kwarg 'missing'") as raised:
        conn.collect(_context())
    assert raised.value.observations == []
    post.assert_not_called()
    warning.assert_called_once()
    assert "missing" in warning.call_args.args[0]


@pytest.mark.parametrize("missing_first", [True, False])
def test_skipped_template_preserves_other_query_and_cannot_report_complete(missing_first, mocker):
    """A skipped query marks coverage partial regardless of its order among successful queries."""
    templates = ['field:="{missing}"', "*"] if missing_first is True else ["*", 'field:="{missing}"']
    connector = VictoriaLogsConnector(
        endpoint="http://logs.example", source=Source.SYSLOG, templates={Source.SYSLOG: templates}
    )
    post = mocker.patch(
        "requests.Session.post", return_value=_response(_ndjson([_row("2026-01-15T05:00:00Z", "kept")]), mocker)
    )
    service = mocker.Mock()
    service.get_attribution.return_value = {"verdict": {"attribution_outcome": "Fail User"}}
    service.ingest_observation.return_value.ingest.accepted = 1
    service.ingest_observation.return_value.ingest.dropped = {}
    output = mocker.Mock()
    pipeline = AttributionOrchestrator([], [], [], service, [output])

    result, succeeded = pipeline._process_one(_context(), [connector])

    assert succeeded is True
    post.assert_called_once()
    [report] = result["connector_reports"]
    assert report["status"] == "partial"
    assert f"query {1 if missing_first is True else 2}: template missing kwarg 'missing'" in report["reason"]
    assert report["observation_count"] == 1
    assert result["evidence_status"] == "partial"
    wire = service.ingest_observation.call_args.args[1].to_dict()
    assert [entry["body"] for entry in json.loads(wire["body"])] == ["kept"]
    assert output.write.call_args.args[1]["connector_reports"] == [report]


@pytest.mark.parametrize(
    ("field", "nodes", "expected"),
    [
        ("host_name", ['node "11"', "node\\12", "nœud"], 'host_name:in("node \\"11\\"","node\\\\12","nœud")'),
        ("node field", ["nœud"], '"node field":in("nœud")'),
    ],
)
def test_node_filter_quotes_field_and_node_values(field, nodes, expected):
    conn = _connector(node_field=field)
    assert conn._inject_node_filter('service_name:="example-cluster"', nodes) == (
        f'{expected} (service_name:="example-cluster")'
    )


def test_node_filter_splits_only_top_level_pipes():
    conn = _connector()
    query = 'service_name:="example-cluster" _msg:~"x\\|y" | stats by (app)'
    assert conn._inject_node_filter(query, ["node-11"]) == (
        'host_name:in("node-11") (service_name:="example-cluster" _msg:~"x\\|y") | stats by (app)'
    )
    assert conn._inject_node_filter('service_name:="x" _msg:~`a|b`', ["node-11"]) == (
        'host_name:in("node-11") (service_name:="x" _msg:~`a|b`)'
    )


@pytest.mark.parametrize("node_field", [None, ""])
def test_node_filter_can_be_disabled(node_field, mocker):
    post = mocker.patch("requests.Session.post", return_value=_response("", mocker))
    conn = _connector(node_field=node_field)
    conn.collect(_context())
    assert post.call_args.kwargs["data"]["query"] == 'service_name:="example-cluster"'


def test_ndjson_skips_blank_invalid_and_non_object_lines(mocker):
    warning = mocker.patch.object(victorialogs.logger, "warning")
    text = '\n{"_time":"2026-01-15T00:00:01Z","_msg":"kept"}\nnot-json\n[]\n  \n'
    mocker.patch("requests.Session.post", return_value=_response(text, mocker))
    observations = _connector(node_field=None).collect(_context())

    assert json.loads(observations[0]["body"]) == [{"body": "kept", "ts": 1768435201000000000}]
    warning.assert_called_once_with("victorialogs: skipped 2 invalid NDJSON line(s)")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-01-15T05:00:45Z", 1768453245000000000),
        ("2026-01-15T05:00:41.9Z", 1768453241900000000),
        ("2026-01-15T05:00:41.123456789Z", 1768453241123456789),
        ("2026-01-15T06:30:45+01:30", 1768453245000000000),
        ("2026-01-15T05:00:41.1234567890Z", None),
        ("not-a-time", None),
        (None, None),
    ],
)
def test_timestamp_parser(value, expected):
    assert victorialogs._timestamp_ns(value) == expected


def test_applog_fixture_unwraps_otel_body_and_builds_resources(mocker):
    text = (FIXTURES / "applog-response.ndjson").read_text()
    mocker.patch("requests.Session.post", return_value=_response(text, mocker))
    conn = VictoriaLogsConnector(
        endpoint="http://victorialogs.example",
        source=Source.APPLOG,
        query_kwargs={"service_name": "example-cluster", "app": "example-training"},
        templates={Source.APPLOG: ['service_name:="{service_name}" app:="{app}"']},
    )

    entries = json.loads(conn.collect(_context())[0]["body"])
    assert entries == [
        {
            "body": "[rank 0] iteration 1200 | lm loss 2.3104 | grad norm 0.912",
            "ts": 1768452900000000000,
            "resources": [{"kind": "node", "hostname": "node-01"}],
        },
        {
            "body": "[rank 80] RuntimeError: CUDA error: uncorrectable ECC error encountered",
            "ts": 1768453241900000000,
            "resources": [{"kind": "node", "hostname": "node-11"}],
        },
        {
            "body": "[rank 0] Watchdog caught collective operation timeout",
            "ts": 1768453245000000000,
            "resources": [{"kind": "node", "hostname": "node-01"}],
        },
    ]


def test_syslog_fixture_keeps_only_query_results_and_sorts_ascending(mocker):
    text = (FIXTURES / "syslog-response.ndjson").read_text()
    mocker.patch("requests.Session.post", return_value=_response(text, mocker))
    conn = _connector()

    entries = json.loads(conn.collect(_context())[0]["body"])
    assert len(entries) == 9
    assert all("Xid (PCI:0000:c5:00): 94" in entry["body"] for entry in entries)
    assert all(entry["resources"] == [{"kind": "node", "hostname": "node-11"}] for entry in entries)
    timestamps = [entry["ts"] for entry in entries]
    assert timestamps == sorted(timestamps)


def test_custom_node_field_and_replica_resource(mocker):
    text = _ndjson(
        [
            _row(
                "2026-01-15T05:00:00Z",
                "plain message",
                custom_node="node-11",
                pod="worker-0",
            )
        ]
    )
    mocker.patch("requests.Session.post", return_value=_response(text, mocker))
    conn = _connector(node_field="custom_node")

    assert json.loads(conn.collect(_context())[0]["body"]) == [
        {
            "body": "plain message",
            "ts": 1768453200000000000,
            "resources": [
                {"kind": "node", "hostname": "node-11"},
                {"kind": "replica", "name": "worker-0"},
            ],
        }
    ]


def test_message_and_cloud_event_envelopes(mocker):
    text = _ndjson(
        [
            _row("2026-01-15T05:00:00Z", '{"message":"message body"}'),
            _row(
                "2026-01-15T05:00:01Z",
                '{"data":{"message":"cloud body","hostname":"node-11"}}',
                host_name="",
            ),
        ]
    )
    mocker.patch("requests.Session.post", return_value=_response(text, mocker))
    entries = json.loads(_connector().collect(_context())[0]["body"])

    assert entries == [
        {"body": "message body", "ts": 1768453200000000000},
        {
            "body": "cloud body",
            "ts": 1768453201000000000,
            "resources": [{"kind": "node", "hostname": "node-11"}],
        },
    ]


def test_invalid_time_or_message_rows_are_skipped(mocker):
    text = _ndjson(
        [
            _row("invalid", "bad timestamp"),
            _row("2026-01-15T05:00:00Z", 42),
        ]
    )
    mocker.patch("requests.Session.post", return_value=_response(text, mocker))
    debug = mocker.patch.object(victorialogs.logger, "debug")

    observations = _connector().collect(_context())
    assert json.loads(observations[0]["body"]) == []
    assert debug.call_count == 2


def test_empty_response_returns_no_observation(mocker):
    mocker.patch("requests.Session.post", return_value=_response("", mocker))
    assert _connector().collect(_context()) == []


def test_pagination_reincludes_boundary_and_deduplicates_rows(mocker):
    boundary = _row("2026-01-15T05:00:20Z", "boundary", host_name="node-11")
    first_page = _ndjson(
        [
            _row("2026-01-15T05:00:40Z", "newest", host_name="node-11"),
            _row("2026-01-15T05:00:30Z", "older", host_name="node-11"),
            boundary,
        ]
    )
    second_page = _ndjson([boundary, _row("2026-01-15T05:00:00Z", "oldest", host_name="node-11")])
    post = mocker.patch(
        "requests.Session.post",
        side_effect=[_response(first_page, mocker), _response(second_page, mocker)],
    )
    conn = _connector(limit=3)

    entries = json.loads(conn.collect(_context())[0]["body"])
    assert len(post.call_args_list) == 2
    assert post.call_args_list[1].kwargs["data"]["end"] == 1768453220000000001
    assert [entry["body"] for entry in entries] == ["oldest", "boundary", "older", "newest"]


def test_same_timestamp_overflow_warns_and_skips_boundary(mocker):
    """A saturated timestamp retains visible rows and reports potentially missing boundary records."""
    same_time = _ndjson(
        [
            _row("2026-01-15T05:00:20Z", "one"),
            _row("2026-01-15T05:00:20Z", "two"),
        ]
    )
    post = mocker.patch(
        "requests.Session.post",
        side_effect=[
            _response(same_time, mocker),
            _response(same_time, mocker),
            _response("", mocker),
        ],
    )
    warning = mocker.patch.object(victorialogs.logger, "warning")
    with pytest.raises(PartialCollectionError, match="possible truncation"):
        _connector(limit=2).collect(_context())

    assert [call.kwargs["data"]["end"] for call in post.call_args_list] == [
        1768471200000000001,
        1768453220000000001,
        1768453220000000000,
    ]
    assert any("possible truncation" in call.args[0] for call in warning.call_args_list)


def test_pagination_stops_when_oldest_is_at_window_start(mocker):
    """A saturated start boundary stops paging and reports its coverage as incomplete."""
    start_time = _context()["start_time"]
    rows = _ndjson(
        [
            _row(start_time.isoformat().replace("+00:00", "Z"), "at-start"),
            _row("2026-01-15T00:00:01Z", "after-start"),
        ]
    )
    post = mocker.patch("requests.Session.post", return_value=_response(rows, mocker))

    with pytest.raises(PartialCollectionError, match="window start"):
        _connector(limit=2).collect(_context())
    post.assert_called_once()


def test_saturated_page_without_parseable_timestamps_warns_and_stops(mocker):
    """Unusable timestamps stop saturated pagination with an explicit partial reason."""
    rows = _ndjson([{"_msg": "one"}, {"_msg": "two"}])
    post = mocker.patch("requests.Session.post", return_value=_response(rows, mocker))
    warning = mocker.patch.object(victorialogs.logger, "warning")

    with pytest.raises(PartialCollectionError, match="no parseable _time") as raised:
        _connector(limit=2).collect(_context())
    observations = raised.value.observations
    assert json.loads(observations[0]["body"]) == []
    post.assert_called_once()
    warning.assert_called_once_with("victorialogs: saturated page has no parseable _time; stopping pagination")


def test_max_pages_cap_preserves_rows_and_reports_incomplete(mocker):
    """Reaching the page budget carries the fetched rows with an explicit partial result."""
    first_page = _ndjson(
        [
            _row("2026-01-15T05:00:20Z", "one"),
            _row("2026-01-15T05:00:10Z", "two"),
        ]
    )
    second_page = _ndjson(
        [
            _row("2026-01-15T05:00:10Z", "two"),
            _row("2026-01-15T05:00:09Z", "three"),
        ]
    )
    post = mocker.patch(
        "requests.Session.post",
        side_effect=[
            _response(first_page, mocker),
            _response(second_page, mocker),
        ],
    )
    warning = mocker.patch.object(victorialogs.logger, "warning")

    with pytest.raises(PartialCollectionError, match="page limit of 2") as raised:
        _connector(limit=2, max_pages=2).collect(_context())
    assert len(json.loads(raised.value.observations[0]["body"])) == 3
    assert post.call_count == 2
    assert any("page limit of 2" in call.args[0] for call in warning.call_args_list)


def test_page_budget_preserves_evidence_but_does_not_report_complete(mocker):
    """A truncated query must retain its entries and make incomplete collection visible."""
    pages = [
        _ndjson([_row(f"2026-01-15T05:00:{second:02d}Z", str(second)) for second in seconds])
        for seconds in ((20, 10), (10, 9))
    ]
    mocker.patch("requests.Session.post", side_effect=[_response(page, mocker) for page in pages])
    service = mocker.Mock()
    service.get_attribution.return_value = {"verdict": {"attribution_outcome": "Fail User"}}
    service.ingest_observation.return_value.ingest.accepted = 3
    service.ingest_observation.return_value.ingest.dropped = {}
    output = mocker.Mock()
    pipeline = AttributionOrchestrator([], [], [], service, [output])

    result, succeeded = pipeline._process_one(_context(), [_connector(limit=2, max_pages=2)])

    assert succeeded is True
    [report] = result["connector_reports"]
    assert report["status"] == "partial"
    assert "page" in report["reason"]
    assert report["observation_count"] == 1
    assert result["evidence_status"] == "partial"
    wire = service.ingest_observation.call_args.args[1].to_dict()
    assert [entry["body"] for entry in json.loads(wire["body"])] == ["9", "10", "20"]
    assert output.write.call_args.args[1]["connector_reports"] == [report]


def test_configured_budget_can_read_beyond_twenty_pages(mocker):
    """A larger finite budget retrieves older entries without changing boundary deduplication."""
    start = _context()["start_time"] + timedelta(hours=1)
    rows = [_row((start + timedelta(seconds=index)).isoformat(), str(index)) for index in range(22)]
    pages = [_ndjson([rows[index], rows[index - 1]]) for index in range(21, 0, -1)] + [""]
    post = mocker.patch("requests.Session.post", side_effect=[_response(page, mocker) for page in pages])

    [observation] = _connector(limit=2, max_pages=22).collect(_context())

    assert post.call_count == 22
    assert [entry["body"] for entry in json.loads(observation["body"])] == [str(index) for index in range(22)]


def test_partial_query_retains_other_queries_and_does_not_leak_status_to_next_collection(mocker):
    """Each collection owns its partial status and preserves rows from all configured queries."""
    pages = [
        _ndjson([_row("2026-01-15T05:00:20Z", "one"), _row("2026-01-15T05:00:10Z", "two")]),
        _ndjson([_row("2026-01-15T05:00:09Z", "three")]),
        "",
        "",
    ]
    mocker.patch("requests.Session.post", side_effect=[_response(page, mocker) for page in pages])
    connector = VictoriaLogsConnector(
        endpoint="http://logs.example",
        source=Source.SYSLOG,
        templates={Source.SYSLOG: ["first", "second"]},
        limit=2,
        max_pages=1,
    )
    with pytest.raises(PartialCollectionError, match="query 1: page limit of 1") as raised:
        connector.collect(_context())

    assert len(json.loads(raised.value.observations[0]["body"])) == 3
    assert "query 2" not in str(raised.value)
    assert connector.collect(_context()) == []


@pytest.mark.parametrize("max_pages", [0, -1, True, False, 1.5, "2", None])
def test_page_budget_requires_a_positive_integer(max_pages):
    """Reject invalid budgets before sending any query."""
    with pytest.raises(ValueError, match="max_pages must be a positive integer"):
        _connector(max_pages=max_pages)


def test_configured_page_budget_reaches_the_connector(mocker):
    """The MCA factory forwards the configured budget to actual pagination."""
    post = mocker.patch(
        "requests.Session.post",
        return_value=_response(_ndjson([_row("2026-01-15T05:00:20Z", "one")]), mocker),
    )
    connector = _build(
        {"endpoint": "http://logs.example", "query": "*", "limit": 1, "max_pages": 1},
        {"source": Source.SYSLOG},
    )

    with pytest.raises(PartialCollectionError, match="page limit of 1"):
        connector.collect(_context())
    post.assert_called_once()


@pytest.mark.parametrize("status", [400, 500])
def test_http_errors_propagate(status, mocker):
    response = _response("", mocker)
    response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    mocker.patch("requests.Session.post", return_value=response)

    with pytest.raises(requests.HTTPError, match=f"HTTP {status}"):
        _connector().collect(_context())


def test_auth_none_and_token_header(mocker):
    mocker.patch("requests.Session.post", return_value=_response("", mocker))
    no_auth = _connector(auth=None)
    token_auth = _connector(auth=TokenAuth("example-token"))

    assert "Authorization" not in no_auth._session.headers
    assert token_auth._session.headers["Authorization"] == "Bearer example-token"


def test_mtls_auth_configures_session(mocker):
    mocker.patch("requests.Session.post", return_value=_response("", mocker))
    conn = _connector(auth=MTLSAuth(cert_path="/cert", key_path="/key", ca_cert_path="/ca"))

    assert conn._session.cert == ("/cert", "/key")
    assert conn._session.verify == "/ca"


def test_retry_adapter_matches_promql_configuration():
    conn = _connector()
    retry = conn._session.get_adapter("https://").max_retries

    assert retry.total == retry.connect == retry.read == retry.status == 5
    assert retry.backoff_factor == 0.5
    assert retry.status_forcelist == (429, 502, 503, 504)
    assert retry.allowed_methods == frozenset({"GET", "POST"})
    assert conn._session.get_adapter("http://") is conn._session.get_adapter("https://")


def test_invalid_source_and_limit_raise_value_error():
    with pytest.raises(ValueError, match="no templates registered"):
        VictoriaLogsConnector(endpoint="http://victorialogs.example", source="unknown")
    with pytest.raises(ValueError, match="limit"):
        _connector(limit=0)


def _cluster_connector(query_kwargs=None, **kwargs):
    return VictoriaLogsConnector(
        endpoint="http://victorialogs.example",
        source=Source.SYSLOG,
        query_kwargs={"cluster": "cluster-a", "service_name": "example-service"}
        if query_kwargs is None
        else query_kwargs,
        templates={Source.SYSLOG: ['cluster:="{cluster}" service_name:="{service_name}"']},
        node_field=None,
        **kwargs,
    )


def _queried(post):
    return [call.kwargs["data"]["query"] for call in post.call_args_list]


@pytest.mark.parametrize("context_cluster", [None, "", "cluster-b"])
def test_cluster_from_context_disabled_uses_configured_cluster(context_cluster, mocker):
    post = mocker.patch("requests.Session.post", return_value=_response("", mocker))
    context = _context()
    if context_cluster is not None:
        context["cluster"] = context_cluster

    _cluster_connector().collect(context)

    assert _queried(post) == ['cluster:="cluster-a" service_name:="example-service"']


def test_cluster_from_context_uses_each_context_cluster(mocker):
    post = mocker.patch("requests.Session.post", return_value=_response("", mocker))
    conn = _cluster_connector(cluster_from_context=True)

    conn.collect({**_context(), "cluster": "cluster-b"})
    conn.collect({**_context(), "cluster": "cluster-c"})
    conn.collect(_context())

    assert _queried(post) == [
        'cluster:="cluster-b" service_name:="example-service"',
        'cluster:="cluster-c" service_name:="example-service"',
        'cluster:="cluster-a" service_name:="example-service"',
    ]
    assert conn._query_kwargs["cluster"] == "cluster-a"


@pytest.mark.parametrize("context_cluster", [None, "", 7])
def test_cluster_from_context_falls_back_without_context_cluster(context_cluster, mocker):
    post = mocker.patch("requests.Session.post", return_value=_response("", mocker))
    context = _context()
    if context_cluster is not None:
        context["cluster"] = context_cluster

    _cluster_connector(cluster_from_context=True).collect(context)

    assert _queried(post) == ['cluster:="cluster-a" service_name:="example-service"']


def test_cluster_from_context_supplies_cluster_missing_from_query_kwargs(mocker):
    """A context cluster completes the query; omitting it later reports a skipped query."""
    post = mocker.patch("requests.Session.post", return_value=_response("", mocker))
    warning = mocker.patch.object(victorialogs.logger, "warning")
    conn = _cluster_connector(query_kwargs={"service_name": "example-service"}, cluster_from_context=True)

    conn.collect({**_context(), "cluster": "cluster-b"})
    assert _queried(post) == ['cluster:="cluster-b" service_name:="example-service"']

    with pytest.raises(PartialCollectionError, match="template missing kwarg 'cluster'") as raised:
        conn.collect(_context())
    assert raised.value.observations == []
    assert post.call_count == 1
    assert "cluster" in warning.call_args.args[0]


@pytest.mark.parametrize("context_cluster", ['cluster-b" or "', "cluster\\b", "cluster`b", "cluster\nb"])
def test_cluster_from_context_rejects_unsafe_context_cluster(context_cluster, mocker):
    post = mocker.patch("requests.Session.post")

    with pytest.raises(ValueError, match="context cluster"):
        _cluster_connector(cluster_from_context=True).collect({**_context(), "cluster": context_cluster})
    post.assert_not_called()
