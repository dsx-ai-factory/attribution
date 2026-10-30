# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from attribution_blueprint.config import load_config
from attribution_blueprint.connectors.metrics.promql import PromQLMetricsConnector
from attribution_blueprint.connectors.metrics.promql.templates import TEMPLATES
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca import (
    build_config,
    deep_merge,
    load_toml,
    parse_env,
    settings_from_cli,
)
from attribution_blueprint.types import Source

FIXTURE = Path(__file__).with_name("query_range.json")
EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "victoriametrics.toml"
LABELS = [("net_host_name", "s_cluster"), ("hostname", "cluster")]


@pytest.mark.parametrize("setting", ["node_label", "cluster_label"])
@pytest.mark.parametrize("label", ["host.name", "", "1host", "host-name", "host\n", None])
@pytest.mark.parametrize("surface", ["constructor", "config"])
def test_invalid_label_names(setting, label, surface):
    with pytest.raises(MCAError, match=rf"promql\.{setting}=.*must match"):
        if surface == "constructor":
            PromQLMetricsConnector(endpoint="http://localhost:8428", cluster="example-cluster", **{setting: label})
        else:
            settings = load_toml(EXAMPLE)
            settings["metrics"]["node_metrics"]["promql"][setting] = label
            build_config(settings)


@pytest.mark.parametrize("setting", ["node_label", "cluster_label"])
@pytest.mark.parametrize("label", ["hostname", "_host", "Host_01"])
def test_valid_label_names(setting, label):
    settings = load_toml(EXAMPLE)
    settings["metrics"]["node_metrics"]["promql"][setting] = label
    assert len(build_config(settings).metrics_connectors) == 1


@pytest.fixture
def replay():
    return json.loads(FIXTURE.read_text())


def _context(replay):
    return {
        "id": "1000001",
        "nodes": ["node-01", "node-02"],
        "start_time": datetime.fromtimestamp(replay["start"], timezone.utc),
        "end_time": datetime.fromtimestamp(replay["end"], timezone.utc),
    }


def _response(replay, name, node_label, cluster_label):
    payload = deepcopy(replay["responses"][name])
    for series in payload["data"]["result"]:
        labels = series["metric"]
        labels[node_label] = labels.pop("net_host_name")
        if "s_cluster" in labels:
            labels[cluster_label] = labels.pop("s_cluster")
    return payload


@pytest.mark.parametrize("node_label,cluster_label", LABELS)
def test_bundled_query_range_replay(mocker, replay, node_label, cluster_label):
    templates = TEMPLATES[Source.NODE_METRICS]
    responses = []
    expected = []
    for name in templates:
        payload = _response(replay, name, node_label, cluster_label)
        response = mocker.Mock()
        response.json.return_value = payload
        responses.append(response)
        for series in payload["data"]["result"]:
            labels = series["metric"]
            for ts, value in series["values"]:
                expected.append(
                    {
                        "ts": ts * 10**9,
                        "resources": [{"kind": "node", "hostname": labels[node_label]}],
                        "body": (
                            f"time: {ts}, metric_name: {name}, metric_value: {value}, "
                            f"threshold: None, cluster: example-cluster, node: {labels[node_label]}, "
                            f"component: {labels.get('device')}"
                        ),
                    }
                )
    post = mocker.patch("requests.Session.post", side_effect=responses)
    # Omit the new options entirely in the default-label case.
    options = {} if node_label == "net_host_name" else dict(node_label=node_label, cluster_label=cluster_label)
    connector = PromQLMetricsConnector(endpoint="http://localhost:8428/", cluster="example-cluster", **options)
    context = _context(replay)
    observations = connector.collect(context)

    assert len(observations) == 1
    assert observations[0]["context"] == {
        "source": Source.NODE_METRICS,
        "time_interval": {"start": context["start_time"], "end": context["end_time"]},
    }
    assert json.loads(observations[0]["body"]) == expected
    assert len(expected) == 24
    selector = f'{{{cluster_label}="example-cluster", {node_label}=~"node-01|node-02"}}'
    queries = [
        f"changes(node_infiniband_link_downed_total{selector}[10m]) > 0",
        f"changes(node_network_carrier_down_changes_total{selector}[10m]) > 0",
        f"increase(ib_roce_adp_retrans{selector}[10m]) > 50",
        f"max by ({node_label}) (up{selector}) == 0",
    ]
    assert post.call_count == 4
    for call, query, response in zip(post.call_args_list, queries, responses, strict=True):
        assert call.args == ("http://localhost:8428/api/v1/query_range",)
        assert call.kwargs == {
            "data": {"query": query, "start": replay["start"], "end": replay["end"], "step": 60},
            "timeout": 60.0,
        }
        response.raise_for_status.assert_called_once_with()
    assert "Authorization" not in connector._session.headers
    assert "X-Scope-OrgID" not in connector._session.headers


def test_result_mapping_uses_configured_labels(mocker, replay):
    payload = _response(replay, "node_infiniband_link_downed_total", "hostname", "cluster")
    payload["data"]["result"] = payload["data"]["result"][:1]
    labels = payload["data"]["result"][0]["metric"]
    labels.update(cluster="returned-cluster", s_cluster="wrong-cluster", net_host_name="wrong-node")
    # A series carrying only the old label must not be attributed with custom mapping.
    payload["data"]["result"].append({"metric": {"net_host_name": "wrong-node"}, "values": [[1, "1"]]})
    response = mocker.Mock()
    response.json.return_value = payload
    mocker.patch("requests.Session.post", return_value=response)
    connector = PromQLMetricsConnector(
        endpoint="http://localhost:8428",
        cluster="configured-cluster",
        node_label="hostname",
        cluster_label="cluster",
        templates={Source.NODE_METRICS: {"counter": "{name}"}},
    )
    entries = json.loads(connector.collect(_context(replay))[0]["body"])
    assert len(entries) == 3
    for entry in entries:
        assert entry["resources"] == [{"kind": "node", "hostname": "node-01"}]
        assert "cluster: returned-cluster," in entry["body"]


@pytest.mark.parametrize("surface", ["toml", "cli", "env"])
def test_label_mapping_config_surfaces(mocker, surface):
    settings = load_toml(EXAMPLE)
    if surface != "toml":
        params = settings["metrics"]["node_metrics"]["promql"]
        del params["node_label"], params["cluster_label"]
        if surface == "cli":
            overrides = settings_from_cli(
                [
                    ["metrics", "node_metrics", "promql", "node_label", "hostname"],
                    ["metrics", "node_metrics", "promql", "cluster_label", "cluster"],
                ]
            )
        else:
            overrides = parse_env(
                {
                    "NVAB_METRICS_NODE_METRICS_PROMQL_NODE_LABEL": "hostname",
                    "NVAB_METRICS_NODE_METRICS_PROMQL_CLUSTER_LABEL": "cluster",
                }
            )
        settings = deep_merge(settings, overrides)
    cfg = build_config(settings)
    response = mocker.Mock()
    response.json.return_value = {"status": "success", "data": {"result": []}}
    post = mocker.patch("requests.Session.post", return_value=response)
    cfg.metrics_connectors[0].collect(cfg.execution_context_adapters[0].discover()[0])
    assert post.call_count == 4
    for call in post.call_args_list:
        assert 'cluster="example-cluster", hostname=~"node-01|node-02"' in call.kwargs["data"]["query"]


@pytest.mark.parametrize(
    "template",
    [
        '{name}{{{cluster_label}="{cluster}", {node_label}=~"{nodes}"}}',
        '{name}{{cluster="{cluster}", hostname=~"{nodes}"}}',
    ],
)
def test_custom_queries_support_new_and_existing_placeholders(mocker, replay, template):
    settings = load_toml(EXAMPLE)
    settings["metrics"]["node_metrics"]["promql"]["queries"] = {"custom_counter": template}
    cfg = build_config(settings)
    payload = _response(replay, "node_infiniband_link_downed_total", "hostname", "cluster")
    response = mocker.Mock()
    response.json.return_value = payload
    post = mocker.patch("requests.Session.post", return_value=response)
    observations = cfg.metrics_connectors[0].collect(_context(replay))
    assert post.call_count == 5
    assert post.call_args.kwargs["data"]["query"] == (
        'custom_counter{cluster="example-cluster", hostname=~"node-01|node-02"}'
    )
    assert any("metric_name: custom_counter," in e["body"] for e in json.loads(observations[0]["body"]))


def test_example_loads_without_auth():
    cfg = load_config(EXAMPLE)
    connector = cfg.metrics_connectors[0]
    assert connector.source == Source.NODE_METRICS
    assert "Authorization" not in connector._session.headers
    assert "X-Scope-OrgID" not in connector._session.headers
