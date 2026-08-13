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
from pathlib import Path

import pytest
from loguru import logger

from attribution_blueprint import cli
from attribution_blueprint.connectors.attribution import AttributionConnector
from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput
from attribution_blueprint.connectors.outputs.stdout import StdoutOutput


@pytest.fixture
def patched_pipeline(mocker, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    patched = mocker.patch("attribution_blueprint.cli.AttributionOrchestrator")
    # `cli.main` consults `pipeline.last_run_stats` to decide the exit code
    # after analyze(): default it to a successful single-context run so the
    # MagicMock auto-attribute doesn't break the int comparisons in cli.py.
    patched.return_value.last_run_stats = {"attempted": 1, "succeeded": 1}
    return patched


@pytest.fixture
def clean_env(monkeypatch):
    """Strip any NVAB_* env vars that might leak between tests."""
    import os

    for key in list(os.environ):
        if key.startswith("NVAB_"):
            monkeypatch.delenv(key, raising=False)


def _kwargs_passed_to_analyze(patched_pipeline):
    return patched_pipeline.return_value.analyze.call_args.kwargs


def _args_passed_to_pipeline(patched_pipeline):
    return patched_pipeline.call_args.args


def _log_connectors_passed(patched_pipeline):
    return patched_pipeline.call_args.args[1]


def _eca_list_passed(patched_pipeline):
    return patched_pipeline.call_args.args[0]


# --- basic CLI plumbing ------------------------------------------------------


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "nvidia-attribution-blueprint" in out


def test_ids_parses_to_list(patched_pipeline, clean_env):
    rc = cli.main(["--connector", "eca", "slurm", "ids", "1, 5, 7"])
    assert rc == 0
    kwargs = _kwargs_passed_to_analyze(patched_pipeline)
    assert kwargs["ids"] == ["1", "5", "7"]
    assert kwargs["since"] is None
    assert kwargs["until"] is None
    assert kwargs["nodelist"] is None


def test_run_summary_reports_context():
    messages = []
    sink_id = logger.add(lambda msg: messages.append(str(msg)), level="INFO", format="{message}")
    try:
        cli._write_run_summary(
            {"attempted": 1, "succeeded": 1},
            [
                {
                    "job_id": "sample-job",
                    "attempt_id": "0",
                    "terminal_state": "fail",
                    "sub_domain": "hardware",
                    "root_symptom": "XID-79",
                    "symptoms": "XID-79, NCCL timeout",
                }
            ],
        )
    finally:
        logger.remove(sink_id)

    assert any("Attribution complete: 1/1 execution contexts succeeded." in msg for msg in messages)
    assert any(
        "sample-job attempt=0 terminal_state=fail sub_domain=hardware "
        "root_symptom=XID-79 symptoms=XID-79, NCCL timeout" in msg
        for msg in messages
    )


def test_time_range_passes_raw_nodelist_to_connector(patched_pipeline, clean_env):
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "--connector",
            "eca",
            "slurm",
            "start_time",
            "2026-05-01T00:00:00",
            "--connector",
            "eca",
            "slurm",
            "end_time",
            "2026-05-01T01:00:00",
            "--connector",
            "eca",
            "slurm",
            "nodelist",
            "node[001-002]",
        ]
    )
    assert rc == 0
    kwargs = _kwargs_passed_to_analyze(patched_pipeline)
    assert kwargs["ids"] is None
    # Offsetless ISO inputs default to UTC at the CLI boundary so naive
    # datetimes can't flow through to the service's AwareDatetime contract.
    assert kwargs["since"] == datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert kwargs["until"] == datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert kwargs["nodelist"] == ["node[001-002]"]


def test_parse_dt_attaches_utc_for_offsetless_input():
    """An ISO string without a UTC offset must be normalized to UTC, not left
    naive. Naive datetimes propagate to the service's AwareDatetime contract
    and cause a 422, and on slurm they cause the BP/slurmctld TZ mismatch."""
    dt = cli._parse_dt("2026-05-01T00:00:00")
    assert dt.tzinfo is not None
    assert dt == datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_dt_preserves_explicit_offset():
    """An ISO string with an explicit offset must be preserved verbatim."""
    dt = cli._parse_dt("2026-05-01T00:00:00-05:00")
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == -5 * 3600


def test_ids_and_start_time_mutually_exclusive(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "eca",
                "slurm",
                "start_time",
                "2026-05-01T00:00:00",
            ]
        )


def test_start_time_without_end_time_errors(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(["--connector", "eca", "slurm", "start_time", "2026-05-01T00:00:00"])


def test_end_time_alone_errors(patched_pipeline, clean_env):
    """`end_time` requires `start_time`; the lone-end-time case must error."""
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "end_time",
                "2026-05-01T01:00:00",
            ]
        )


def test_end_time_with_ids_is_rejected(patched_pipeline, clean_env):
    """ids and time-range params are mutually exclusive - having both should error."""
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "eca",
                "slurm",
                "end_time",
                "2026-05-01T01:00:00",
            ]
        )


@pytest.mark.parametrize("component", ["slurm", "kueue"])
def test_nodelist_without_time_range_errors(component, patched_pipeline, clean_env):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--connector",
                "eca",
                component,
                "--connector",
                "eca",
                component,
                "nodelist",
                "node-a",
            ]
        )
    assert exc.value.code == 2


def test_service_url_override(patched_pipeline, clean_env):
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "attribution",
            "service_url",
            "http://example.test/",
        ]
    )
    assert rc == 0
    args = _args_passed_to_pipeline(patched_pipeline)
    attribution = args[3]
    assert isinstance(attribution, AttributionConnector)
    assert attribution.service_url == "http://example.test"


def test_output_sqlite_override(patched_pipeline, tmp_path, clean_env):
    db_path = tmp_path / "x.db"
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "output",
            "primary",
            "sqlite",
            "--connector",
            "output",
            "primary",
            "sqlite",
            "path",
            str(db_path),
        ]
    )
    assert rc == 0
    args = _args_passed_to_pipeline(patched_pipeline)
    outputs = args[4]
    assert len(outputs) == 1
    assert isinstance(outputs[0], SQLiteOutput)
    assert outputs[0]._path == str(db_path)


def test_default_config_when_none_provided(patched_pipeline, clean_env):
    rc = cli.main(["--connector", "eca", "slurm", "ids", "1"])
    assert rc == 0
    args = _args_passed_to_pipeline(patched_pipeline)
    attribution = args[3]
    outputs = args[4]
    assert isinstance(attribution, AttributionConnector)
    assert attribution.service_url == "http://localhost:8080"
    assert len(outputs) == 1
    assert isinstance(outputs[0], StdoutOutput)
    patched_pipeline.return_value.analyze.assert_called_once()


def test_loads_config_from_explicit_path(patched_pipeline, clean_env):
    fixture = Path(__file__).parent / "fixtures" / "config.toml"
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--config",
            str(fixture),
        ]
    )
    assert rc == 0
    args = _args_passed_to_pipeline(patched_pipeline)
    attribution = args[3]
    assert isinstance(attribution, AttributionConnector)


def test_invalid_slurm_cluster_tz_reports_cli_error(tmp_path, capsys, clean_env):
    cfg = tmp_path / "bad-slurm-tz.toml"
    cfg.write_text('[eca]\ncomponent = "slurm"\n[eca.slurm]\ncluster_tz = "Not/A_Timezone"\n')

    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(cfg)])

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "error: unknown cluster_tz: 'Not/A_Timezone'" in captured.err
    assert "Traceback" not in captured.err


def test_malformed_toml_reports_cli_error_without_traceback(tmp_path, capsys, clean_env):
    cfg = tmp_path / "bad.toml"
    cfg.write_text('[attribution\nservice_url = "http://127.0.0.1:8000"\n')

    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(cfg)])

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert captured.out == ""
    assert f"invalid TOML config {cfg}:" in captured.err
    assert "Expected ']'" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_utf8_toml_reports_cli_error_without_traceback(tmp_path, capsys, clean_env):
    cfg = tmp_path / "bad-encoding.toml"
    cfg.write_bytes(b"\xff")

    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(cfg)])

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert captured.out == ""
    assert f"invalid TOML config {cfg}:" in captured.err
    assert "Traceback" not in captured.err


def test_missing_sacct_path_reports_cli_error_without_traceback(tmp_path, capsys, clean_env):
    cfg = tmp_path / "bad-sacct-path.toml"
    cfg.write_text(
        '[eca]\ncomponent = "slurm"\n[eca.slurm]\nsacct_path = "/no/such/sacct"\ncluster_tz = "UTC"\nids = "123456"\n'
    )
    rc = cli.main(["--config", str(cfg)])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "error: file not found: /no/such/sacct" in captured.err
    assert "Traceback" not in captured.err


def test_repeated_config_runs_each_config(patched_pipeline, tmp_path, clean_env):
    cfg1 = tmp_path / "one.toml"
    cfg1.write_text(
        '[attribution]\nservice_url = "http://one.test"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    cfg2 = tmp_path / "two.toml"
    cfg2.write_text(
        '[attribution]\nservice_url = "http://two.test"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )

    rc = cli.main(["--config", str(cfg1), "--config", str(cfg2)])

    assert rc == 0
    assert patched_pipeline.call_count == 2
    service_urls = [call.args[3].service_url for call in patched_pipeline.call_args_list]
    assert service_urls == ["http://one.test", "http://two.test"]


def test_repeated_config_validates_all_paths_before_running(patched_pipeline, tmp_path, clean_env):
    cfg = tmp_path / "valid.toml"
    cfg.write_text(
        '[attribution]\nservice_url = "http://valid.test"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(cfg), "--config", str(tmp_path / "missing.toml")])

    assert exc.value.code == 2
    patched_pipeline.assert_not_called()


def test_repeated_config_applies_cli_overrides_to_each_config(patched_pipeline, tmp_path, clean_env):
    cfg1 = tmp_path / "one.toml"
    cfg1.write_text(
        '[attribution]\nservice_url = "http://one.test"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    cfg2 = tmp_path / "two.toml"
    cfg2.write_text(
        '[attribution]\nservice_url = "http://two.test"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )

    rc = cli.main(
        [
            "--config",
            str(cfg1),
            "--config",
            str(cfg2),
            "--connector",
            "attribution",
            "service_url",
            "http://override.test",
        ]
    )

    assert rc == 0
    assert patched_pipeline.call_count == 2
    service_urls = [call.args[3].service_url for call in patched_pipeline.call_args_list]
    assert service_urls == ["http://override.test", "http://override.test"]


def test_configs_directory_runs_sorted_top_level_toml_files(patched_pipeline, tmp_path, clean_env, mocker):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "ignore.txt").write_text("not a config")
    nested = config_dir / "nested"
    nested.mkdir()
    (nested / "nested.toml").write_text(
        '[attribution]\nservice_url = "http://nested.test"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    (config_dir / "b.toml").write_text(
        '[attribution]\nservice_url = "http://b.test"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    (config_dir / "a.toml").write_text(
        '[attribution]\nservice_url = "http://a.test"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )

    info = mocker.patch("attribution_blueprint.cli.logger.info")
    rc = cli.main(["--configs", str(config_dir), "--connector", "eca", "slurm", "ids", "1"])

    assert rc == 0
    assert patched_pipeline.call_count == 2
    service_urls = [call.args[3].service_url for call in patched_pipeline.call_args_list]
    assert service_urls == ["http://a.test", "http://b.test"]
    assert patched_pipeline.return_value.analyze.call_count == 2
    for call in patched_pipeline.return_value.analyze.call_args_list:
        assert call.kwargs["ids"] == ["1"]
    assert any(
        call.args[0] == "Attribution complete: 2/2 execution contexts succeeded." for call in info.call_args_list
    )


def test_no_args_runs_loopback_demo(patched_pipeline, clean_env):
    """With no args the blueprint must run the documented loopback demo
    successfully: slurm.discover() returns [], the pipeline attempts zero
    contexts, and the CLI exits 0. (Previously this errored, contradicting
    ARCHITECTURE.md "Defaults and standalone CLI".)"""
    # Mock zero-context run.
    patched_pipeline.return_value.last_run_stats = {"attempted": 0, "succeeded": 0}
    rc = cli.main([])
    assert rc == 0


def test_selection_alone_runs_loopback_demo(patched_pipeline, clean_env):
    """`--connector eca slurm` is a bare selection that picks the (default)
    slurm component and otherwise behaves identically to no-args."""
    patched_pipeline.return_value.last_run_stats = {"attempted": 0, "succeeded": 0}
    rc = cli.main(["--connector", "eca", "slurm"])
    assert rc == 0


def test_cli_recognises_new_runtime_eca_registered_at_runtime(patched_pipeline, clean_env):
    """Regression: cli.py used to hardcode component names ("slurm", "kueue")
    when partitioning runtime-selectable ECAs vs static-style ones. A newly
    registered ECA that declares the full ids/start_time/end_time/nodelist
    schema must be treated as runtime-selectable without any cli.py edit -
    the partition is now driven by inspecting ComponentSpec.schema."""
    from attribution_blueprint import mca, mca_registry
    from attribution_blueprint.mca_registry import _PARAM

    class DummyRuntimeECA:
        def __init__(self, marker: str):
            self.marker = marker

        def discover(self):
            return []

        def discover_by_ids(self, ids):
            return []

        def discover_by_time_range(self, since, until, nodelist):
            return []

    name = "dummy_runtime_eca"
    old_schema = mca.SCHEMA
    mca_registry._COMPONENTS.pop(("eca", name), None)
    mca_registry.register_component(
        framework="eca",
        name=name,
        schema={
            "ids": _PARAM,
            "start_time": _PARAM,
            "end_time": _PARAM,
            "nodelist": _PARAM,
            "marker": _PARAM,
        },
        build=lambda spec, ctx: DummyRuntimeECA(marker=spec.get("marker", "")),
    )
    mca.SCHEMA = mca_registry.build_schema()
    try:
        rc = cli.main(
            [
                "--connector",
                "eca",
                name,
                "--connector",
                "eca",
                name,
                "ids",
                "id-1, id-2",
            ]
        )
    finally:
        mca_registry._COMPONENTS.pop(("eca", name), None)
        mca.SCHEMA = old_schema

    assert rc == 0
    assert _kwargs_passed_to_analyze(patched_pipeline)["ids"] == ["id-1", "id-2"]


def test_non_slurm_eca_does_not_require_slurm_runtime(patched_pipeline, clean_env):
    from attribution_blueprint import mca, mca_registry
    from attribution_blueprint.mca_registry import _PARAM

    class DummyECA:
        def __init__(self, marker: str):
            self.marker = marker

        def discover_by_ids(self, ids):
            return []

        def discover_by_time_range(self, since, until, nodelist):
            return []

    name = "dummy_cli_eca"
    old_schema = mca.SCHEMA
    mca_registry._COMPONENTS.pop(("eca", name), None)
    mca_registry.register_component(
        framework="eca",
        name=name,
        schema={"marker": _PARAM},
        build=lambda spec, ctx: DummyECA(marker=spec.get("marker", "")),
    )
    mca.SCHEMA = mca_registry.build_schema()

    try:
        rc = cli.main(
            [
                "--connector",
                "eca",
                name,
                "--connector",
                "eca",
                name,
                "marker",
                "ok",
            ]
        )
    finally:
        mca_registry._COMPONENTS.pop(("eca", name), None)
        mca.SCHEMA = old_schema

    assert rc == 0
    eca = _eca_list_passed(patched_pipeline)[0]
    assert isinstance(eca, DummyECA)
    assert eca.marker == "ok"
    kwargs = _kwargs_passed_to_analyze(patched_pipeline)
    assert kwargs == {"ids": None, "since": None, "until": None, "nodelist": None}


def test_execution_context_alias(patched_pipeline, clean_env):
    """`execution_context` is an alias for `eca`."""
    rc = cli.main(["--connector", "execution_context", "slurm", "ids", "1"])
    assert rc == 0
    patched_pipeline.return_value.analyze.assert_called_once()


def test_log_alias_for_logs(patched_pipeline, clean_env):
    """`logs` is an alias for `log` (plural / singular)."""
    from attribution_blueprint.connectors.logs.local_file import LocalFileLogConnector

    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "logs",
            "applog",
            "local_file",
            "--connector",
            "logs",
            "applog",
            "local_file",
            "paths",
            "/tmp/x.out",
        ]
    )
    assert rc == 0
    collectors = _log_connectors_passed(patched_pipeline)
    assert any(isinstance(c, LocalFileLogConnector) for c in collectors)


# --- eca=static ----------------------------------------------------------


def test_static_eca_from_cli_flags(patched_pipeline, clean_env):
    from attribution_blueprint.connectors.execution_contexts.static import StaticExecutionContextAdapter

    rc = cli.main(
        [
            "--connector",
            "eca",
            "static",
            "--connector",
            "eca",
            "static",
            "id",
            "sample-static-job",
            "--connector",
            "eca",
            "static",
            "start_time",
            "2026-05-08T12:14:30+00:00",
            "--connector",
            "eca",
            "static",
            "end_time",
            "2026-05-08T18:01:01+00:00",
            "--connector",
            "eca",
            "static",
            "nodes",
            "example-node-003,example-node-005",
            "--connector",
            "eca",
            "static",
            "state",
            "FAILED",
            "--connector",
            "eca",
            "static",
            "name",
            "sample-training",
            "--connector",
            "eca",
            "static",
            "std_out",
            "/path/to/logs/job.out",
            "--connector",
            "eca",
            "static",
            "workload_type",
            "runai",
        ]
    )
    assert rc == 0
    ecas = _eca_list_passed(patched_pipeline)
    assert len(ecas) == 1
    eca = ecas[0]
    assert isinstance(eca, StaticExecutionContextAdapter)
    assert eca._id == "sample-static-job"
    assert eca._nodes == ["example-node-003", "example-node-005"]
    assert eca._state == "FAILED"
    assert eca._name == "sample-training"
    assert eca._std_out == "/path/to/logs/job.out"
    assert eca._workload_type == "runai"
    assert _kwargs_passed_to_analyze(patched_pipeline) == {
        "ids": None,
        "since": None,
        "until": None,
        "nodelist": None,
    }


def test_static_eca_id_defaults_to_hash(patched_pipeline, clean_env):
    rc = cli.main(
        [
            "--connector",
            "eca",
            "static",
            "--connector",
            "eca",
            "static",
            "start_time",
            "2026-05-08T12:14:30+00:00",
            "--connector",
            "eca",
            "static",
            "end_time",
            "2026-05-08T18:01:01+00:00",
            "--connector",
            "eca",
            "static",
            "nodes",
            "node2,node1",
        ]
    )
    assert rc == 0
    eca = _eca_list_passed(patched_pipeline)[0]
    assert eca._id.startswith("static-")
    assert len(eca._id) == len("static-") + 16


def test_static_eca_hash_id_is_stable(patched_pipeline, clean_env):
    def run_and_get_id():
        patched_pipeline.reset_mock()
        cli.main(
            [
                "--connector",
                "eca",
                "static",
                "--connector",
                "eca",
                "static",
                "start_time",
                "2026-05-08T12:14:30+00:00",
                "--connector",
                "eca",
                "static",
                "end_time",
                "2026-05-08T18:01:01+00:00",
                "--connector",
                "eca",
                "static",
                "nodes",
                "node1,node2",
            ]
        )
        return _eca_list_passed(patched_pipeline)[0]._id

    assert run_and_get_id() == run_and_get_id()


def test_static_eca_rejects_slurm_runtime_flags(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "static",
                "--connector",
                "eca",
                "slurm",
                "ids",
                "sample-static-job",
                "--connector",
                "eca",
                "static",
                "start_time",
                "2026-05-08T12:14:30+00:00",
                "--connector",
                "eca",
                "static",
                "end_time",
                "2026-05-08T18:01:01+00:00",
                "--connector",
                "eca",
                "static",
                "nodes",
                "node1",
            ]
        )


def test_static_eca_missing_required_field_errors(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "static",
                "--connector",
                "eca",
                "static",
                "id",
                "sample-static-job",
                "--connector",
                "eca",
                "static",
                "end_time",
                "2026-05-08T18:01:01+00:00",
                "--connector",
                "eca",
                "static",
                "nodes",
                "node1",
            ]
        )


def test_slurm_rejects_static_flags(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "eca",
                "static",
                "name",
                "x",
            ]
        )


def test_static_eca_cli_overrides_toml_fields(patched_pipeline, tmp_path, clean_env):
    from attribution_blueprint.connectors.execution_contexts.static import StaticExecutionContextAdapter

    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "static"\n'
        "[eca.static]\n"
        'id = "sample-static-job"\n'
        'start_time = "2026-05-08T12:14:30+00:00"\n'
        'end_time = "2026-05-08T18:01:01+00:00"\n'
        'nodes = ["toml-node-1", "toml-node-2"]\n'
        'state = "FAILED"\n'
        '[output.console]\ncomponent = "stdout"\n'
    )
    rc = cli.main(
        [
            "--config",
            str(cfg),
            "--connector",
            "eca",
            "static",
            "nodes",
            "cli-node-1",
        ]
    )
    assert rc == 0
    eca = _eca_list_passed(patched_pipeline)[0]
    assert isinstance(eca, StaticExecutionContextAdapter)
    assert eca._nodes == ["cli-node-1"]
    assert eca._id == "sample-static-job"
    assert eca._state == "FAILED"


# --- eca=kueue --------------------------------------------------------------


def test_kueue_ids_parses_to_list(patched_pipeline, clean_env):
    rc = cli.main(
        [
            "--connector",
            "eca",
            "kueue",
            "--connector",
            "eca",
            "kueue",
            "ids",
            "kueue-demo/foo, kueue-demo/bar",
        ]
    )
    assert rc == 0
    kwargs = _kwargs_passed_to_analyze(patched_pipeline)
    assert kwargs["ids"] == ["kueue-demo/foo", "kueue-demo/bar"]
    assert kwargs["since"] is None
    assert kwargs["until"] is None
    assert kwargs["nodelist"] is None


def test_kueue_ids_single_invocation_selects_kueue(patched_pipeline, clean_env):
    rc = cli.main(["--connector", "eca", "kueue", "ids", "kueue-demo/foo"])

    assert rc == 0
    kwargs = _kwargs_passed_to_analyze(patched_pipeline)
    assert kwargs["ids"] == ["kueue-demo/foo"]
    eca = _eca_list_passed(patched_pipeline)[0]
    assert type(eca).__name__ == "KueueExecutionContextAdapter"


def test_kueue_time_range_passes_to_analyze(patched_pipeline, clean_env):
    rc = cli.main(
        [
            "--connector",
            "eca",
            "kueue",
            "--connector",
            "eca",
            "kueue",
            "start_time",
            "2026-05-20T12:00:00",
            "--connector",
            "eca",
            "kueue",
            "end_time",
            "2026-05-20T16:00:00",
        ]
    )
    assert rc == 0
    kwargs = _kwargs_passed_to_analyze(patched_pipeline)
    assert kwargs["ids"] is None
    assert kwargs["since"] == datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    assert kwargs["until"] == datetime(2026, 5, 20, 16, 0, 0, tzinfo=timezone.utc)
    assert kwargs["nodelist"] is None


def test_kueue_nodelist_forwarded_verbatim_to_connector(patched_pipeline, clean_env):
    """CLI is scheduler-agnostic about nodelist syntax: the raw user string is
    forwarded as-is in a single-element list. Each adapter parses its own
    syntax internally - kueue CSV-splits, slurm expands bracket expressions -
    so the CLI doesn't need to know which scheduler is selected."""
    rc = cli.main(
        [
            "--connector",
            "eca",
            "kueue",
            "--connector",
            "eca",
            "kueue",
            "start_time",
            "2026-05-20T12:00:00",
            "--connector",
            "eca",
            "kueue",
            "end_time",
            "2026-05-20T16:00:00",
            "--connector",
            "eca",
            "kueue",
            "nodelist",
            "node-a, node-b",
        ]
    )
    assert rc == 0
    kwargs = _kwargs_passed_to_analyze(patched_pipeline)
    assert kwargs["nodelist"] == ["node-a, node-b"]


def test_kueue_ids_and_start_time_mutually_exclusive(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "kueue",
                "--connector",
                "eca",
                "kueue",
                "ids",
                "kueue-demo/foo",
                "--connector",
                "eca",
                "kueue",
                "start_time",
                "2026-05-20T12:00:00",
            ]
        )


def test_kueue_start_time_without_end_time_errors(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "kueue",
                "--connector",
                "eca",
                "kueue",
                "start_time",
                "2026-05-20T12:00:00",
            ]
        )


def test_kueue_end_time_without_start_time_errors(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "kueue",
                "--connector",
                "eca",
                "kueue",
                "end_time",
                "2026-05-20T16:00:00",
            ]
        )


def test_kueue_nodelist_requires_time_range(patched_pipeline, clean_env):
    """nodelist combined with ids (instead of a time range) must be rejected."""
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "kueue",
                "--connector",
                "eca",
                "kueue",
                "ids",
                "kueue-demo/foo",
                "--connector",
                "eca",
                "kueue",
                "nodelist",
                "node-a",
            ]
        )


def test_kueue_runtime_params_rejected_with_slurm_component(patched_pipeline, clean_env):
    """eca.kueue.* runtime params must not flow through when component != kueue."""
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "eca",
                "kueue",
                "ids",
                "kueue-demo/foo",
            ]
        )


def test_kueue_eca_rejects_slurm_runtime_flags(patched_pipeline, clean_env):
    """Symmetric to test_static_eca_rejects_slurm_runtime_flags."""
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "eca",
                "kueue",
            ]
        )


def test_kueue_eca_rejects_static_runtime_flags(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "kueue",
                "--connector",
                "eca",
                "static",
                "name",
                "x",
            ]
        )


def test_kueue_selection_alone_runs_discover_loopback(patched_pipeline, clean_env):
    """`--connector eca kueue` with no runtime params hits KueueExecutionContextAdapter.discover()
    (which is documented to return []), so analyze() runs with all-None kwargs and exits 0."""
    patched_pipeline.return_value.last_run_stats = {"attempted": 0, "succeeded": 0}
    rc = cli.main(["--connector", "eca", "kueue"])
    assert rc == 0
    kwargs = _kwargs_passed_to_analyze(patched_pipeline)
    assert kwargs == {"ids": None, "since": None, "until": None, "nodelist": None}


def test_kueue_eca_built_with_runtime_params(patched_pipeline, clean_env):
    """The kueue adapter itself is constructed when component=kueue and runtime params are present."""
    from attribution_blueprint.connectors.execution_contexts.kueue import (
        KueueExecutionContextAdapter,
    )

    rc = cli.main(
        [
            "--connector",
            "eca",
            "kueue",
            "--connector",
            "eca",
            "kueue",
            "namespace",
            "kueue-demo",
            "--connector",
            "eca",
            "kueue",
            "ids",
            "kueue-demo/foo",
        ]
    )
    assert rc == 0
    ecas = _eca_list_passed(patched_pipeline)
    assert len(ecas) == 1
    assert isinstance(ecas[0], KueueExecutionContextAdapter)
    assert ecas[0].namespace == "kueue-demo"
    assert _kwargs_passed_to_analyze(patched_pipeline)["ids"] == ["kueue-demo/foo"]


# --- workload --------------------------------------------------------------


def test_gpus_per_node_propagates_to_pipeline(patched_pipeline, clean_env):
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "workload",
            "gpus_per_node",
            "8",
        ]
    )
    assert rc == 0
    assert patched_pipeline.call_args.kwargs["gpus_per_node"] == 8


def test_gpus_per_node_defaults_to_none(patched_pipeline, clean_env):
    rc = cli.main(["--connector", "eca", "slurm", "ids", "1"])
    assert rc == 0
    assert patched_pipeline.call_args.kwargs["gpus_per_node"] is None


def test_gpus_per_node_non_positive_errors(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "workload",
                "gpus_per_node",
                "0",
            ]
        )


# --- log connectors --------------------------------------------------------


def test_applog_local_file_installs_collector(patched_pipeline, clean_env):
    from attribution_blueprint.connectors.logs.local_file import LocalFileLogConnector

    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "log",
            "applog",
            "local_file",
            "--connector",
            "log",
            "applog",
            "local_file",
            "paths",
            "{std_out}",
        ]
    )
    assert rc == 0
    collectors = _log_connectors_passed(patched_pipeline)
    assert len(collectors) == 1
    assert isinstance(collectors[0], LocalFileLogConnector)
    assert collectors[0]._templates[0] == "{std_out}"


def test_applog_local_file_requires_paths(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "log",
                "applog",
                "local_file",
            ]
        )


def test_applog_loki_requires_endpoint(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "log",
                "applog",
                "loki",
                "--connector",
                "log",
                "applog",
                "loki",
                "query",
                "{job=`x`}",
            ]
        )


def test_applog_loki_token_uses_env(patched_pipeline, monkeypatch, clean_env):
    from attribution_blueprint.connectors.logs.loki import LokiConnector

    monkeypatch.setenv("MY_LOKI_TOKEN", "shh")
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "log",
            "applog",
            "loki",
            "--connector",
            "log",
            "applog",
            "loki",
            "query",
            "{service_name=`cloud`}",
            "--connector",
            "log",
            "applog",
            "loki",
            "endpoint",
            "https://loki.example",
            "--connector",
            "log",
            "applog",
            "loki",
            "auth",
            "type",
            "token",
            "--connector",
            "log",
            "applog",
            "loki",
            "auth",
            "token_env",
            "MY_LOKI_TOKEN",
        ]
    )
    assert rc == 0
    collectors = _log_connectors_passed(patched_pipeline)
    assert len(collectors) == 1
    assert isinstance(collectors[0], LokiConnector)


def test_applog_loki_token_env_unset_errors(patched_pipeline, monkeypatch, clean_env):
    """Unset token_env must route through the CLI's MCAError path and exit
    cleanly via parser.error (SystemExit), not leak as an uncaught KeyError."""
    monkeypatch.delenv("MY_LOKI_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "log",
                "applog",
                "loki",
                "--connector",
                "log",
                "applog",
                "loki",
                "query",
                "{x=`y`}",
                "--connector",
                "log",
                "applog",
                "loki",
                "endpoint",
                "https://loki.example",
                "--connector",
                "log",
                "applog",
                "loki",
                "auth",
                "type",
                "token",
                "--connector",
                "log",
                "applog",
                "loki",
                "auth",
                "token_env",
                "MY_LOKI_TOKEN",
            ]
        )


def test_applog_loki_mtls_requires_cert_and_key(patched_pipeline, clean_env):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--connector",
                "eca",
                "slurm",
                "ids",
                "1",
                "--connector",
                "log",
                "applog",
                "loki",
                "--connector",
                "log",
                "applog",
                "loki",
                "query",
                "{x=`y`}",
                "--connector",
                "log",
                "applog",
                "loki",
                "endpoint",
                "https://loki.example",
                "--connector",
                "log",
                "applog",
                "loki",
                "auth",
                "type",
                "mtls",
            ]
        )


def test_syslog_local_file_installs_collector(patched_pipeline, clean_env):
    from attribution_blueprint.connectors.logs.local_file import LocalFileLogConnector
    from attribution_blueprint.types import Source

    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "log",
            "syslog",
            "local_file",
            "--connector",
            "log",
            "syslog",
            "local_file",
            "paths",
            "/var/log/syslog",
        ]
    )
    assert rc == 0
    collectors = _log_connectors_passed(patched_pipeline)
    assert len(collectors) == 1
    assert isinstance(collectors[0], LocalFileLogConnector)
    assert collectors[0].source is Source.SYSLOG
    assert collectors[0]._templates[0] == "/var/log/syslog"


def test_applog_and_syslog_can_coexist(patched_pipeline, clean_env):
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "log",
            "applog",
            "local_file",
            "--connector",
            "log",
            "applog",
            "local_file",
            "paths",
            "{std_out}",
            "--connector",
            "log",
            "syslog",
            "local_file",
            "--connector",
            "log",
            "syslog",
            "local_file",
            "paths",
            "/var/log/syslog",
        ]
    )
    assert rc == 0
    collectors = _log_connectors_passed(patched_pipeline)
    sources = sorted(c.source.name for c in collectors)
    assert sources == ["APPLOG", "SYSLOG"]


def test_per_source_loki_backends_can_differ(patched_pipeline, monkeypatch, clean_env):
    from attribution_blueprint.connectors.logs.loki import LokiConnector
    from attribution_blueprint.types import Source

    monkeypatch.setenv("APPLOG_TOKEN", "a")
    monkeypatch.setenv("SYSLOG_TOKEN", "b")
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "log",
            "applog",
            "loki",
            "--connector",
            "log",
            "applog",
            "loki",
            "query",
            "{app=`x`} | slurm_job_id = `{job_id}`",
            "--connector",
            "log",
            "applog",
            "loki",
            "endpoint",
            "https://applog-loki.example",
            "--connector",
            "log",
            "applog",
            "loki",
            "auth",
            "type",
            "token",
            "--connector",
            "log",
            "applog",
            "loki",
            "auth",
            "token_env",
            "APPLOG_TOKEN",
            "--connector",
            "log",
            "syslog",
            "loki",
            "--connector",
            "log",
            "syslog",
            "loki",
            "query",
            "{service_name=`cloud`} |~ `Xid`",
            "--connector",
            "log",
            "syslog",
            "loki",
            "endpoint",
            "https://syslog-loki.example",
            "--connector",
            "log",
            "syslog",
            "loki",
            "auth",
            "type",
            "token",
            "--connector",
            "log",
            "syslog",
            "loki",
            "auth",
            "token_env",
            "SYSLOG_TOKEN",
        ]
    )
    assert rc == 0
    collectors = _log_connectors_passed(patched_pipeline)
    assert len(collectors) == 2
    by_source = {c.source: c for c in collectors if isinstance(c, LokiConnector)}
    assert by_source[Source.APPLOG]._endpoint == "https://applog-loki.example"
    assert by_source[Source.SYSLOG]._endpoint == "https://syslog-loki.example"


def test_cli_loki_endpoint_overrides_toml(patched_pipeline, tmp_path, monkeypatch, clean_env):
    """CLI value wins over TOML for the same key."""
    from attribution_blueprint.connectors.auth import TokenAuth
    from attribution_blueprint.connectors.logs.loki import LokiConnector

    cert = tmp_path / "client.crt"
    key = tmp_path / "client.key"
    cert.write_text("")
    key.write_text("")
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[log.applog]\ncomponent = "loki"\n'
        '[log.applog.loki]\nendpoint = "https://toml-loki.example"\n'
        f'[log.applog.loki.auth]\ntype = "mtls"\ncert_path = "{cert}"\nkey_path = "{key}"\n'
    )
    monkeypatch.setenv("MY_TOKEN", "shh")
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--config",
            str(cfg),
            "--connector",
            "log",
            "applog",
            "loki",
            "query",
            "{job=`x`}",
            "--connector",
            "log",
            "applog",
            "loki",
            "endpoint",
            "https://cli-loki.example",
            "--connector",
            "log",
            "applog",
            "loki",
            "auth",
            "type",
            "token",
            "--connector",
            "log",
            "applog",
            "loki",
            "auth",
            "token_env",
            "MY_TOKEN",
        ]
    )
    assert rc == 0
    applog = next(
        c
        for c in _log_connectors_passed(patched_pipeline)
        if isinstance(c, LokiConnector) and c.source.value == "applog"
    )
    assert applog._endpoint == "https://cli-loki.example"
    assert isinstance(applog._auth, TokenAuth)


# --- env vars --------------------------------------------------------------


def test_env_var_sets_ids(patched_pipeline, monkeypatch, clean_env):
    monkeypatch.setenv("NVAB_ECA_SLURM_IDS", "42,43")
    rc = cli.main(["--connector", "eca", "slurm"])
    assert rc == 0
    assert _kwargs_passed_to_analyze(patched_pipeline)["ids"] == ["42", "43"]


def test_env_var_selects_component(patched_pipeline, monkeypatch, clean_env):
    monkeypatch.setenv("NVAB_ECA", "slurm")
    monkeypatch.setenv("NVAB_ECA_SLURM_IDS", "7")
    rc = cli.main([])
    assert rc == 0


def test_env_component_param_implies_selection(patched_pipeline, monkeypatch, clean_env):
    monkeypatch.setenv("NVAB_ECA_KUEUE_IDS", "team-alpha/foo")
    rc = cli.main([])

    assert rc == 0
    assert _kwargs_passed_to_analyze(patched_pipeline)["ids"] == ["team-alpha/foo"]
    eca = _eca_list_passed(patched_pipeline)[0]
    assert type(eca).__name__ == "KueueExecutionContextAdapter"


def test_env_var_routes_to_output_instance(patched_pipeline, monkeypatch, tmp_path, clean_env):
    """NVAB_OUTPUT_<INSTANCE>_<COMPONENT>_<PARAM> dynamically resolves the
    user-chosen instance name and routes the param under that instance."""
    db_path = tmp_path / "x.db"
    monkeypatch.setenv("NVAB_OUTPUT_PRIMARY", "sqlite")
    monkeypatch.setenv("NVAB_OUTPUT_PRIMARY_SQLITE_PATH", str(db_path))
    rc = cli.main(["--connector", "eca", "slurm", "ids", "1"])
    assert rc == 0
    outputs = _args_passed_to_pipeline(patched_pipeline)[4]
    assert len(outputs) == 1
    assert isinstance(outputs[0], SQLiteOutput)
    assert outputs[0]._path == str(db_path)


def test_env_output_param_implies_instance_component(patched_pipeline, monkeypatch, tmp_path, clean_env):
    db_path = tmp_path / "x.db"
    monkeypatch.setenv("NVAB_OUTPUT_PRIMARY_SQLITE_PATH", str(db_path))
    rc = cli.main(["--connector", "eca", "slurm", "ids", "1"])

    assert rc == 0
    outputs = _args_passed_to_pipeline(patched_pipeline)[4]
    assert len(outputs) == 1
    assert isinstance(outputs[0], SQLiteOutput)
    assert outputs[0]._path == str(db_path)


def test_cli_overrides_env(patched_pipeline, monkeypatch, clean_env):
    monkeypatch.setenv("NVAB_ATTRIBUTION_SERVICE_URL", "http://env.test")
    rc = cli.main(
        [
            "--connector",
            "eca",
            "slurm",
            "ids",
            "1",
            "--connector",
            "attribution",
            "service_url",
            "http://cli.test/",
        ]
    )
    assert rc == 0
    attribution = _args_passed_to_pipeline(patched_pipeline)[3]
    assert attribution.service_url == "http://cli.test"


def test_env_overrides_toml(patched_pipeline, monkeypatch, tmp_path, clean_env):
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[attribution]\nservice_url = "http://toml.test"\n[eca]\ncomponent = "slurm"\n[output.console]\ncomponent = "stdout"\n'
    )
    monkeypatch.setenv("NVAB_ATTRIBUTION_SERVICE_URL", "http://env.test")
    rc = cli.main(["--connector", "eca", "slurm", "ids", "1", "--config", str(cfg)])
    assert rc == 0
    attribution = _args_passed_to_pipeline(patched_pipeline)[3]
    assert attribution.service_url == "http://env.test"


def test_env_open_namespace_query_kwargs(patched_pipeline, monkeypatch, tmp_path, clean_env):
    """`query_kwargs` is an open namespace; sub-keys via env arrive as a dict."""
    from attribution_blueprint.connectors.logs.loki import LokiConnector

    cert = tmp_path / "client.crt"
    key = tmp_path / "client.key"
    cert.write_text("")
    key.write_text("")
    monkeypatch.setenv("NVAB_LOG_SYSLOG_LOKI_QUERY_KWARGS_SERVICE_NAME", "example-cluster")
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.console]\ncomponent = "stdout"\n'
        '[log.syslog]\ncomponent = "loki"\n'
        '[log.syslog.loki]\nendpoint = "https://loki.example"\n'
        f'[log.syslog.loki.auth]\ntype = "mtls"\ncert_path = "{cert}"\nkey_path = "{key}"\n'
    )
    rc = cli.main(["--connector", "eca", "slurm", "ids", "1", "--config", str(cfg)])
    assert rc == 0
    syslog = next(
        c
        for c in _log_connectors_passed(patched_pipeline)
        if isinstance(c, LokiConnector) and c.source.value == "syslog"
    )
    assert syslog._query_kwargs.get("service_name") == "example-cluster"


# --- MCA `help` action -----------------------------------------------------


def test_mca_root_help_lists_frameworks(capsys, clean_env):
    rc = cli.main(["--connector", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: <root>" in out
    for fw in ("eca", "log", "output", "attribution", "workload"):
        assert fw in out
    assert "execution_context" in out
    assert "logs" in out


def test_mca_eca_help_lists_components(capsys, clean_env):
    rc = cli.main(["--connector", "eca", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: eca" in out
    assert "slurm" in out
    assert "static" in out


def test_mca_eca_slurm_help_lists_params(capsys, clean_env):
    rc = cli.main(["--connector", "eca", "slurm", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: eca.slurm" in out
    for key in ("ids", "start_time", "end_time", "nodelist", "sacct_path"):
        assert key in out


def test_mca_eca_static_help_lists_params(capsys, clean_env):
    rc = cli.main(["--connector", "eca", "static", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: eca.static" in out
    for key in (
        "id",
        "start_time",
        "end_time",
        "nodes",
        "state",
        "name",
        "user_name",
        "std_out",
        "std_err",
        "workload_type",
        "gpu_count",
    ):
        assert key in out


def test_mca_log_help_lists_sources(capsys, clean_env):
    rc = cli.main(["--connector", "log", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: log" in out
    assert "applog" in out
    assert "syslog" in out


def test_mca_log_applog_help_lists_components(capsys, clean_env):
    rc = cli.main(["--connector", "log", "applog", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: log.applog" in out
    assert "loki" in out
    assert "local_file" in out


def test_mca_log_applog_loki_help_lists_keys_and_subsections(capsys, clean_env):
    rc = cli.main(["--connector", "log", "applog", "loki", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: log.applog.loki" in out
    for key in ("endpoint", "timeout_seconds", "query", "queries"):
        assert key in out
    assert "query_kwargs" in out
    assert "auth" in out


def test_mca_log_applog_loki_auth_help_lists_keys(capsys, clean_env):
    rc = cli.main(["--connector", "log", "applog", "loki", "auth", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: log.applog.loki.auth" in out
    for key in ("type", "cert_path", "key_path", "ca_cert_path", "token_env", "org_id"):
        assert key in out


def test_mca_output_help_lists_components(capsys, clean_env):
    rc = cli.main(["--connector", "output", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: output" in out
    assert "stdout" in out
    assert "sqlite" in out


def test_mca_output_instance_sqlite_help_lists_keys(capsys, clean_env):
    rc = cli.main(["--connector", "output", "primary", "sqlite", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: output.primary.sqlite" in out
    assert "path" in out


def test_mca_attribution_help_lists_keys(capsys, clean_env):
    rc = cli.main(["--connector", "attribution", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: attribution" in out
    assert "service_url" in out
    assert "timeout_seconds" in out


def test_mca_workload_help_lists_keys(capsys, clean_env):
    rc = cli.main(["--connector", "workload", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: workload" in out
    assert "gpus_per_node" in out


def test_mca_help_resolves_aliases(capsys, clean_env):
    rc = cli.main(["--connector", "execution_context", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: eca" in out
    assert "slurm" in out

    rc = cli.main(["--connector", "logs", "applog", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MCA help: log.applog" in out


def test_mca_help_short_circuits_before_config(capsys, clean_env, mocker):
    """Help must not load TOML or run the pipeline."""
    pipeline = mocker.patch("attribution_blueprint.cli.AttributionOrchestrator")
    rc = cli.main(["--connector", "eca", "slurm", "help"])
    assert rc == 0
    pipeline.assert_not_called()


def test_help_as_param_value_is_not_a_help_request(patched_pipeline, clean_env):
    """`eca slurm ids help` sets ids=['help']; the trailing `help` is the value, not the action."""
    rc = cli.main(["--connector", "eca", "slurm", "ids", "help"])
    assert rc == 0
    kwargs = _kwargs_passed_to_analyze(patched_pipeline)
    assert kwargs["ids"] == ["help"]


def test_mca_help_unknown_path_errors(capsys, clean_env):
    with pytest.raises(SystemExit):
        cli.main(["--connector", "eca", "nope", "help"])
    err = capsys.readouterr().err
    assert "unknown --connector key 'nope'" in err


def test_total_failure_exits_nonzero(patched_pipeline, clean_env):
    """If the orchestrator attempts contexts but none succeed (all errored, or
    sqlite couldn't persist a single verdict), the CLI must surface that as a
    non-zero exit. The previous code returned 0 even when 100/100 contexts
    failed, masking the regression."""
    patched_pipeline.return_value.last_run_stats = {"attempted": 100, "succeeded": 0}
    rc = cli.main(["--connector", "eca", "slurm", "ids", "1"])
    assert rc == 1


def test_partial_success_still_exits_zero(patched_pipeline, clean_env):
    """Per-context failure isolation is the documented contract. As long as
    at least one context produced a verdict, exit 0 - the loss surfaces in
    the logs, not the exit code."""
    patched_pipeline.return_value.last_run_stats = {"attempted": 100, "succeeded": 1}
    rc = cli.main(["--connector", "eca", "slurm", "ids", "1"])
    assert rc == 0
