# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from attribution_blueprint import mca_registry as reg
from attribution_blueprint.exceptions import RegistryError


@pytest.fixture(autouse=True)
def isolate_registry():
    """Each test gets a fresh registry so registrations don't bleed across tests.

    Also captures sys.modules entries for the connector packages: `_fresh_import`
    pops + re-imports them to re-run registration side effects, which creates
    NEW class objects. Without restoration, other test files that hold a
    pre-imported reference (e.g. `isinstance(eca, StaticExecutionContextAdapter)`
    in test_config.py) compare against the stale class identity and fail under
    cross-file collection orderings.
    """
    saved_fw = dict(reg._FRAMEWORKS)
    saved_comp = dict(reg._COMPONENTS)
    saved_modules = {k: v for k, v in sys.modules.items() if k.startswith("attribution_blueprint.")}
    reg._FRAMEWORKS.clear()
    reg._COMPONENTS.clear()
    yield
    # Drop anything the test imported under attribution_blueprint.*, then restore
    # the original module objects so later tests see the same classes they
    # imported at collection time.
    for k in [k for k in sys.modules if k.startswith("attribution_blueprint.")]:
        if k not in saved_modules:
            sys.modules.pop(k, None)
    for k, v in saved_modules.items():
        sys.modules[k] = v
    reg._FRAMEWORKS.clear()
    reg._FRAMEWORKS.update(saved_fw)
    reg._COMPONENTS.clear()
    reg._COMPONENTS.update(saved_comp)
    # Force mca.SCHEMA to reflect the restored registry. mca.SCHEMA is computed
    # once at import time, and the schema-shape test reads it after our clear()
    # above; without this, it observes an empty schema.
    if "attribution_blueprint.mca" in sys.modules:
        mca_mod = sys.modules["attribution_blueprint.mca"]
        mca_mod.SCHEMA = reg.build_schema()


def test_register_framework_stores_spec():
    reg.register_framework(name="eca", aliases=("execution_context",), has_selector=True)
    fw = reg.get_framework("eca")
    assert fw.name == "eca"
    assert fw.aliases == ("execution_context",)
    assert fw.has_selector is True


def test_register_component_keys_by_framework_and_name():
    reg.register_framework(name="output", instances=True)
    reg.register_component(
        framework="output",
        name="stdout",
        schema={},
        build=lambda spec, ctx: "stdout-instance",
    )
    comp = reg.get_component("output", "stdout")
    assert comp.framework == "output"
    assert comp.name == "stdout"
    assert comp.build({}, ctx=None) == "stdout-instance"


def test_register_component_under_unknown_framework_errors():
    with pytest.raises(RegistryError):
        reg.register_component(framework="nope", name="x", schema={}, build=lambda *_: None)


def test_duplicate_component_registration_errors():
    reg.register_framework(name="output", instances=True)
    reg.register_component(framework="output", name="stdout", schema={}, build=lambda *_: None)
    with pytest.raises(RegistryError):
        reg.register_component(framework="output", name="stdout", schema={}, build=lambda *_: None)


def _fresh_import(*modules: str) -> None:
    """Force each module to re-run its top-level code (registration side effects)."""
    import importlib
    import sys

    for mod in modules:
        sys.modules.pop(mod, None)
        importlib.import_module(mod)


def test_output_components_registered_by_connector_packages():
    """Importing the output connector packages alone is enough to register them."""
    from attribution_blueprint import mca_registry as r

    r.register_framework(name="output", instances=True)
    _fresh_import(
        "attribution_blueprint.connectors.outputs.stdout",
        "attribution_blueprint.connectors.outputs.sqlite",
    )
    assert r.get_component("output", "stdout").name == "stdout"
    assert r.get_component("output", "sqlite").name == "sqlite"


def test_eca_components_registered_by_connector_packages():
    """Importing the eca connector packages alone is enough to register them."""
    from attribution_blueprint import mca_registry as r

    r.register_framework(name="eca", aliases=("execution_context",), has_selector=True)
    _fresh_import(
        "attribution_blueprint.connectors.execution_contexts.slurm",
        "attribution_blueprint.connectors.execution_contexts.static",
    )
    assert r.get_component("eca", "slurm").name == "slurm"
    assert r.get_component("eca", "static").name == "static"


def test_log_components_registered_by_connector_packages():
    """Importing the log connector packages alone is enough to register them."""
    from attribution_blueprint import mca_registry as r

    r.register_framework(name="log", aliases=("logs",), sources=("applog", "syslog"))
    _fresh_import(
        "attribution_blueprint.connectors.logs.loki",
        "attribution_blueprint.connectors.logs.local_file",
    )
    assert r.get_component("log", "loki").name == "loki"
    assert r.get_component("log", "local_file").name == "local_file"


def test_built_schema_has_expected_shape():
    """The registry-derived SCHEMA must contain every existing framework, component,
    and key that mca.py exposed before the refactor."""
    from attribution_blueprint import mca

    schema = mca.SCHEMA
    assert "eca" in schema
    assert schema["eca"]["_aliases"] == ("execution_context",)
    assert set(schema["eca"]["_selector"]) == {"slurm", "static", "kueue"}
    assert set(schema["eca"]["_selector"]["slurm"]) == {
        "ids",
        "start_time",
        "end_time",
        "nodelist",
        "sacct_path",
        "cluster_tz",
    }
    assert set(schema["eca"]["_selector"]["static"]) == {
        "id",
        "attempt_id",
        "start_time",
        "end_time",
        "nodes",
        "state",
        "name",
        "user_name",
        "std_out",
        "std_err",
        "workload_type",
        "workload_framework",
        "gpu_count",
        "timezone",
        "contexts_path",
    }
    assert set(schema["eca"]["_selector"]["kueue"]) == {
        "ids",
        "start_time",
        "end_time",
        "nodelist",
        "kubeconfig",
        "context",
        "namespace",
        "timeout_seconds",
        "workload_api_version",
        "placement_records_enabled",
        "placement_recorder",
        "event_max_age_seconds",
        "prometheus_metrics",
    }
    assert set(schema["eca"]["_selector"]["kueue"]["placement_recorder"]) == {
        "endpoint",
        "timeout_seconds",
        "auth",
    }
    assert set(schema["eca"]["_selector"]["kueue"]["prometheus_metrics"]) == {
        "endpoint",
        "cluster",
        "cluster_label",
        "timeout_seconds",
        "lookback_padding_seconds",
        "batch_size",
        "pod_info_metric",
        "pod_labels_metric",
        "pod_controller_uid_label",
        "auth",
    }

    assert "log" in schema
    assert schema["log"]["_aliases"] == ("logs",)
    for src in ("applog", "syslog"):
        assert src in schema["log"]
        assert {"loki", "local_file", "kubectl_logs"} <= set(schema["log"][src]["_selector"])
        assert "query_kwargs" in schema["log"][src]["_selector"]["loki"]
        assert schema["log"][src]["_selector"]["loki"]["query_kwargs"].get("_open") is True
        assert set(schema["log"][src]["_selector"]["loki"]["auth"]) == {
            "type",
            "cert_path",
            "key_path",
            "ca_cert_path",
            "token_env",
            "org_id",
        }

    assert set(schema["output"]["_instances"]) == {"stdout", "sqlite"}
    assert set(schema["output"]["_instances"]["sqlite"]) == {"path"}
    assert set(schema["attribution"]) == {"service_url", "timeout_seconds"}
    assert set(schema["workload"]) == {"gpus_per_node"}
