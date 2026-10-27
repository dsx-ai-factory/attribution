# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from attribution_blueprint.config import load_config
from attribution_blueprint.connectors.attribution import AttributionConnector
from attribution_blueprint.connectors.execution_contexts.slurm import SlurmExecutionContextAdapter
from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput
from attribution_blueprint.connectors.outputs.stdout import StdoutOutput
from attribution_blueprint.exceptions import MCAError

FIXTURE = Path(__file__).parent / "fixtures" / "config.toml"


def test_loads_valid_config():
    cfg = load_config(FIXTURE)

    assert isinstance(cfg.attribution, AttributionConnector)
    assert len(cfg.execution_context_adapters) == 1
    assert isinstance(cfg.execution_context_adapters[0], SlurmExecutionContextAdapter)
    assert len(cfg.outputs) == 1
    assert isinstance(cfg.outputs[0], StdoutOutput)
    assert cfg.log_connectors == []
    assert cfg.metrics_connectors == []
    assert cfg.gpus_per_node is None


def test_workload_gpus_per_node_loaded(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        "[workload]\ngpus_per_node = 8\n"
    )
    cfg = load_config(p)
    assert cfg.gpus_per_node == 8


def test_execution_context_alias_works(tmp_path):
    """[execution_context] is an alias for [eca]."""
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[execution_context]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    cfg = load_config(p)
    assert isinstance(cfg.execution_context_adapters[0], SlurmExecutionContextAdapter)


def test_static_eca_loaded_from_toml(tmp_path):
    from attribution_blueprint.connectors.execution_contexts.static import StaticExecutionContextAdapter

    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "static"\n'
        "[eca.static]\n"
        'id = "sample-static-job"\n'
        'start_time = "2026-05-08T12:14:30+00:00"\n'
        'end_time = "2026-05-08T18:01:01+00:00"\n'
        'timezone = "America/Los_Angeles"\n'
        'nodes = ["example-node-003", "example-node-005"]\n'
        'state = "FAILED"\n'
        "gpu_count = 6\n"
        '[output.console]\ncomponent = "stdout"\n'
    )
    cfg = load_config(p)
    assert len(cfg.execution_context_adapters) == 1
    eca = cfg.execution_context_adapters[0]
    assert isinstance(eca, StaticExecutionContextAdapter)
    assert eca._id == "sample-static-job"
    assert eca._attempt_id == "0"
    assert eca._nodes == ["example-node-003", "example-node-005"]
    assert eca._state == "FAILED"
    assert eca.discover()[0]["gpu_count"] == 6
    assert eca.discover()[0]["timezone"] == "America/Los_Angeles"


def test_static_eca_missing_fields_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "static"\n'
        '[eca.static]\nid = "1"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    with pytest.raises(MCAError, match="missing required keys"):
        load_config(p)


def test_static_eca_contexts_path_loaded_from_toml(tmp_path):
    contexts_path = tmp_path / "contexts.json"
    contexts_path.write_text(
        json.dumps(
            {
                "contexts": [
                    {
                        "id": "workload-context-0001",
                        "attempt_id": "1",
                        "start_time": "2026-05-23T00:00:00+00:00",
                        "end_time": "2026-05-23T00:01:00+00:00",
                        "timezone": "UTC",
                        "nodes": ["node-0001"],
                        "gpu_count": "3",
                        "state": "FAILED",
                        "failure_reason": "gpu_nvlink_unhealthy",
                    },
                    {
                        "id": "workload-context-0002",
                        "attempt_id": "2",
                        "start_time": "2026-05-23T00:02:00+00:00",
                        "end_time": "2026-05-23T00:03:00+00:00",
                        "nodes": ["node-0002"],
                        "state": "FAILED",
                        "failure_reason": "gpu_fall_off_bus",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "static"\n'
        "[eca.static]\n"
        f'contexts_path = "{contexts_path}"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )

    cfg = load_config(p)
    ctxs = cfg.execution_context_adapters[0].discover()

    assert [ctx["id"] for ctx in ctxs] == ["workload-context-0001", "workload-context-0002"]
    assert ctxs[0]["scheduler"] == "static"
    assert ctxs[0]["job_id"] == "workload-context-0001"
    assert ctxs[0]["gpu_count"] == 3
    assert ctxs[0]["timezone"] == "UTC"
    assert ctxs[0]["failure_reason"] == "gpu_nvlink_unhealthy"


def test_static_eca_contexts_path_rejects_invalid_timezone(tmp_path):
    contexts_path = tmp_path / "contexts.json"
    contexts_path.write_text(
        json.dumps(
            [
                {
                    "id": "workload-context-0001",
                    "start_time": "2026-05-23T00:00:00+00:00",
                    "end_time": "2026-05-23T00:01:00+00:00",
                    "timezone": "Not/A/Zone",
                    "nodes": ["node-0001"],
                }
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "c.toml"
    config_path.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "static"\n'
        "[eca.static]\n"
        f'contexts_path = "{contexts_path}"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )

    with pytest.raises(MCAError, match="valid IANA timezone key"):
        load_config(config_path)


def test_static_eca_attempt_id_from_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "static"\n'
        "[eca.static]\n"
        'id = "sample-job-001"\n'
        'attempt_id = "1"\n'
        'start_time = "2026-05-08T12:14:30+00:00"\n'
        'end_time = "2026-05-08T18:01:01+00:00"\n'
        'nodes = ["example-node-001"]\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    cfg = load_config(p)
    ctx = cfg.execution_context_adapters[0].discover()[0]
    assert ctx["attempt_id"] == "1"


def test_workload_gpus_per_node_invalid_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        "[workload]\ngpus_per_node = 0\n"
    )
    with pytest.raises(MCAError, match="gpus_per_node"):
        load_config(p)


def test_unknown_output_component_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n[eca]\ncomponent = "slurm"\n[output.console]\ncomponent = "carrier-pigeon"\n'
    )
    with pytest.raises(MCAError, match="invalid component"):
        load_config(p)


def test_unknown_eca_component_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n[eca]\ncomponent = "k8s"\n[output.console]\ncomponent = "stdout"\n'
    )
    with pytest.raises(MCAError, match="invalid component"):
        load_config(p)


def test_loki_log_connector_built_per_source(monkeypatch, tmp_path):
    monkeypatch.setenv("ATTRIBUTION_TEST_TOKEN", "x")
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[log.applog]\ncomponent = "loki"\n'
        '[log.applog.loki]\nendpoint = "https://loki.example"\n'
        'query_kwargs = { service_name = "c", app = "ml" }\n'
        '[log.applog.loki.auth]\ntype = "token"\ntoken_env = "ATTRIBUTION_TEST_TOKEN"\n'
        '[log.syslog]\ncomponent = "loki"\n'
        '[log.syslog.loki]\nendpoint = "https://loki.example"\n'
        'query_kwargs = { service_name = "c" }\n'
        '[log.syslog.loki.auth]\ntype = "token"\ntoken_env = "ATTRIBUTION_TEST_TOKEN"\n'
    )
    cfg = load_config(p)
    assert len(cfg.log_connectors) == 2
    sources = sorted(c.source.value for c in cfg.log_connectors)
    assert sources == ["applog", "syslog"]


def test_loki_unset_token_env_raises_mca_error(monkeypatch, tmp_path):
    """Unset token_env surfaces as MCAError so the CLI's parser.error path
    produces a clean usage message and exit code 2, rather than a Python
    traceback from an uncaught KeyError."""
    monkeypatch.delenv("DOES_NOT_EXIST_XYZ", raising=False)
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[log.syslog]\ncomponent = "loki"\n'
        '[log.syslog.loki]\nendpoint = "https://loki.example"\n'
        'query_kwargs = { service_name = "c" }\n'
        '[log.syslog.loki.auth]\ntype = "token"\ntoken_env = "DOES_NOT_EXIST_XYZ"\n'
    )
    with pytest.raises(MCAError, match="DOES_NOT_EXIST_XYZ"):
        load_config(p)


def test_loki_unknown_auth_type_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[log.syslog]\ncomponent = "loki"\n'
        '[log.syslog.loki]\nendpoint = "https://loki.example"\n'
        '[log.syslog.loki.auth]\ntype = "smoke-signal"\n'
    )
    with pytest.raises(MCAError, match="auth type"):
        load_config(p)


def test_sqlite_output(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "https://svc"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.primary]\ncomponent = "sqlite"\n[output.primary.sqlite]\npath = "./out.db"\n'
    )
    cfg = load_config(p)
    assert len(cfg.outputs) == 1
    assert isinstance(cfg.outputs[0], SQLiteOutput)


def test_promql_metrics_connector_built(monkeypatch, tmp_path):
    from attribution_blueprint.connectors.metrics.promql import PromQLMetricsConnector
    from attribution_blueprint.types import Source

    monkeypatch.setenv("ATTRIBUTION_TEST_MIMIR_TOKEN", "x")
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[metrics.node_metrics]\ncomponent = "promql"\n'
        "[metrics.node_metrics.promql]\n"
        'endpoint = "https://mimir.example/prometheus"\n'
        'cluster = "example-cluster"\n'
        "[metrics.node_metrics.promql.auth]\n"
        'type = "token"\ntoken_env = "ATTRIBUTION_TEST_MIMIR_TOKEN"\n'
    )
    cfg = load_config(p)
    assert len(cfg.metrics_connectors) == 1
    conn = cfg.metrics_connectors[0]
    assert isinstance(conn, PromQLMetricsConnector)
    assert conn.source == Source.NODE_METRICS


def test_promql_metrics_connector_built_without_auth(tmp_path):
    from attribution_blueprint.connectors.metrics.promql import PromQLMetricsConnector
    from attribution_blueprint.types import Source

    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[metrics.node_metrics]\ncomponent = "promql"\n'
        "[metrics.node_metrics.promql]\n"
        'endpoint = "https://prom.example"\n'
        'cluster = "example-cluster"\n'
    )
    cfg = load_config(p)
    assert len(cfg.metrics_connectors) == 1
    conn = cfg.metrics_connectors[0]
    assert isinstance(conn, PromQLMetricsConnector)
    assert conn.source == Source.NODE_METRICS


def test_promql_queries_extend_bundled_templates(monkeypatch, tmp_path):
    from attribution_blueprint.connectors.metrics.promql.templates import TEMPLATES
    from attribution_blueprint.types import Source

    monkeypatch.setenv("ATTRIBUTION_TEST_MIMIR_TOKEN", "x")
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[metrics.node_metrics]\ncomponent = "promql"\n'
        "[metrics.node_metrics.promql]\n"
        'endpoint = "https://mimir.example"\n'
        'cluster = "c"\n'
        "[metrics.node_metrics.promql.queries]\n"
        "my_custom_drop = 'increase({name}{{net_host_name=~\"{nodes}\"}}[5m]) > 100'\n"
        "[metrics.node_metrics.promql.auth]\n"
        'type = "token"\ntoken_env = "ATTRIBUTION_TEST_MIMIR_TOKEN"\n'
    )
    cfg = load_config(p)
    conn = cfg.metrics_connectors[0]
    # The connector merges bundled defaults with the customer's queries.
    assert "my_custom_drop" in conn._templates
    for bundled_name in TEMPLATES[Source.NODE_METRICS]:
        assert bundled_name in conn._templates


def test_promql_requires_cluster(monkeypatch, tmp_path):
    monkeypatch.setenv("ATTRIBUTION_TEST_MIMIR_TOKEN", "x")
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[metrics.node_metrics]\ncomponent = "promql"\n'
        "[metrics.node_metrics.promql]\n"
        'endpoint = "https://mimir.example"\n'
        "[metrics.node_metrics.promql.auth]\n"
        'type = "token"\ntoken_env = "ATTRIBUTION_TEST_MIMIR_TOKEN"\n'
    )
    with pytest.raises(MCAError, match="cluster"):
        load_config(p)


def test_promql_requires_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("ATTRIBUTION_TEST_MIMIR_TOKEN", "x")
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[metrics.node_metrics]\ncomponent = "promql"\n'
        '[metrics.node_metrics.promql]\ncluster = "c"\n'
        "[metrics.node_metrics.promql.auth]\n"
        'type = "token"\ntoken_env = "ATTRIBUTION_TEST_MIMIR_TOKEN"\n'
    )
    with pytest.raises(MCAError, match="endpoint"):
        load_config(p)


def test_loki_node_label_from_toml(monkeypatch, tmp_path):
    """`node_label` in TOML reaches the connector so non-default Loki schemas
    (hostname / net_host_name / etc.) can be wired without code edits."""
    monkeypatch.setenv("ATTRIBUTION_TEST_TOKEN", "x")
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[log.syslog]\ncomponent = "loki"\n'
        '[log.syslog.loki]\nendpoint = "https://loki.example"\n'
        'node_label = "hostname"\n'
        'query_kwargs = { service_name = "c" }\n'
        '[log.syslog.loki.auth]\ntype = "token"\ntoken_env = "ATTRIBUTION_TEST_TOKEN"\n'
    )
    cfg = load_config(p)
    assert len(cfg.log_connectors) == 1
    assert cfg.log_connectors[0]._node_label == "hostname"


def test_slurm_cluster_tz_from_toml(tmp_path):
    """Slurm ECA `cluster_tz` flows from TOML so the connector anchors sacct
    timestamps to the cluster timezone, not the blueprint host's."""
    from zoneinfo import ZoneInfo

    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[eca.slurm]\ncluster_tz = "America/Los_Angeles"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    cfg = load_config(p)
    assert cfg.execution_context_adapters[0].cluster_tz == ZoneInfo("America/Los_Angeles")


def test_static_eca_workload_framework_from_toml(tmp_path):
    """Static ECA accepts an optional `workload_framework` (e.g. megatron) and
    emits it on the produced ExecutionContext."""
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "static"\n'
        "[eca.static]\n"
        'id = "1234"\n'
        'start_time = "2026-05-08T12:00:00+00:00"\n'
        'end_time = "2026-05-08T18:00:00+00:00"\n'
        'nodes = ["n1"]\n'
        'workload_framework = "megatron"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    cfg = load_config(p)
    eca = cfg.execution_context_adapters[0]
    ctx = eca.discover()[0]
    assert ctx["workload_framework"] == "megatron"


def test_promql_queries_via_open_namespace_cli(monkeypatch, tmp_path):
    """Verify the `queries` _open namespace merges with bundled defaults: a customer-named
    metric is present alongside the bundled ones in the connector's templates."""
    monkeypatch.setenv("ATTRIBUTION_TEST_MIMIR_TOKEN", "x")
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[metrics.node_metrics]\ncomponent = "promql"\n'
        "[metrics.node_metrics.promql]\n"
        'endpoint = "https://mimir.example"\ncluster = "c"\n'
        "[metrics.node_metrics.promql.queries]\n"
        "foo_dropped = 'increase({name}{{net_host_name=~\"{nodes}\"}}[5m])'\n"
        "[metrics.node_metrics.promql.auth]\n"
        'type = "token"\ntoken_env = "ATTRIBUTION_TEST_MIMIR_TOKEN"\n'
    )
    cfg = load_config(p)
    conn = cfg.metrics_connectors[0]
    assert "foo_dropped" in conn._templates
    # Bundled defaults still present:
    assert "node_infiniband_link_downed_total" in conn._templates


def test_promql_unset_token_env_raises_mca_error(monkeypatch, tmp_path):
    """Unset token_env for promql must surface as MCAError so the CLI's
    parser.error path produces a clean usage message. Mirrors the loki
    behavior - previously promql raised an uncaught KeyError."""
    monkeypatch.delenv("DOES_NOT_EXIST_PROMQL", raising=False)
    p = tmp_path / "c.toml"
    p.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[metrics.node_metrics]\ncomponent = "promql"\n'
        "[metrics.node_metrics.promql]\n"
        'endpoint = "https://mimir.example/prometheus"\n'
        'cluster = "example-cluster"\n'
        "[metrics.node_metrics.promql.auth]\n"
        'type = "token"\ntoken_env = "DOES_NOT_EXIST_PROMQL"\n'
    )
    with pytest.raises(MCAError, match="DOES_NOT_EXIST_PROMQL"):
        load_config(p)
