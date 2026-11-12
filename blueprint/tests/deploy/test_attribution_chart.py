# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render checks for the in-cluster Helm chart at deploy/attribution.

They run `helm template` and inspect the manifests, so they need the `helm`
binary and are skipped without it. Nothing here talks to a cluster.

The chart deploys up to three components (the Service, the runner and the
dashboard) in one pod. The tests cover the bundled default, a Service-only
release, a runner that calls an external Service, a user-provided Blueprint
configuration file, and a dashboard-only release, plus the settings each one
accepts or rejects.
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

CHART = Path(__file__).resolve().parents[2] / "deploy" / "attribution"
EXAMPLES = CHART / "examples"
IMAGE = ["--set", "image.repository=registry.example/attribution", "--set-string", "image.tag=1.2.3"]
NS = ["--set", "namespaces={team-a,team-b}"]

pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")


def _helm(*args: str) -> subprocess.CompletedProcess:
    """Run `helm template` with the given arguments and capture its output."""
    return subprocess.run(["helm", "template", "rel", str(CHART), *args], text=True, capture_output=True, check=False)


def _render(*args: str, image: bool = True) -> list[dict]:
    """Render the chart and return its manifests, failing the test if helm fails."""
    result = _helm(*(IMAGE if image else []), *args)
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _render_values(tmp_path: Path, values: dict) -> list[dict]:
    """Render with a values file, which is how users pass numbers larger than a million."""
    path = tmp_path / "values.yaml"
    path.write_text(yaml.safe_dump(values))
    result = _helm("-f", str(path))
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _kind(docs: list[dict], kind: str) -> list[dict]:
    """The manifests of one kind."""
    return [doc for doc in docs if doc["kind"] == kind]


def _pod(docs: list[dict]) -> dict:
    """The pod spec of the chart's StatefulSet."""
    return _kind(docs, "StatefulSet")[0]["spec"]["template"]["spec"]


def _containers(docs: list[dict]) -> dict[str, dict]:
    """The pod's containers by name."""
    return {container["name"]: container for container in _pod(docs)["containers"]}


def _env(container: dict) -> dict[str, str]:
    """A container's literal environment values by name."""
    return {item["name"]: item.get("value") for item in container.get("env", [])}


def _config(docs: list[dict]) -> dict:
    """The data of the chart's ConfigMap."""
    return _kind(docs, "ConfigMap")[0]["data"]


def _generated_toml(docs: list[dict]) -> dict:
    """The generated Blueprint configuration, parsed."""
    return tomllib.loads(_config(docs)["blueprint.toml"])


# --- the bundled default ----------------------------------------------------


def test_bundled_default_runs_all_three_components_in_one_pod():
    docs = _render(*NS)

    assert list(_containers(docs)) == ["service", "runner", "dashboard"]
    assert len(_kind(docs, "StatefulSet")) == 1
    assert _kind(docs, "StatefulSet")[0]["spec"]["replicas"] == 1


def test_bundled_default_keeps_the_service_api_on_localhost():
    docs = _render(*NS)
    service = _containers(docs)["service"]

    assert service["command"][-4:] == ["--host", "127.0.0.1", "--port", "8000"]
    assert "ports" not in service
    assert [item["metadata"]["name"] for item in _kind(docs, "Service")] == ["rel-attribution"]
    assert [port["port"] for port in _kind(docs, "Service")[0]["spec"]["ports"]] == [8080]
    assert _generated_toml(docs)["attribution"]["service_url"] == "http://127.0.0.1:8000"
    assert _env(_containers(docs)["runner"])["SERVICE_URL"] == "http://127.0.0.1:8000"


def test_bundled_default_probes_the_service_health_endpoint_inside_the_container():
    probe = _containers(_render(*NS))["service"]["readinessProbe"]
    assert "httpGet" not in probe
    assert "/v2/healthz" in probe["exec"]["command"][-1]
    assert probe["timeoutSeconds"] == 5


def test_bundled_default_selects_kai_and_kubectl_logs():
    config = _generated_toml(_render(*NS))

    assert config["eca"]["component"] == "kai"
    assert config["log"]["applog"]["component"] == "kubectl_logs"
    assert "kubectl_logs" in config["log"]["applog"]
    assert config["output"]["db"]["sqlite"]["path"] == "/data/attributions.db"


def test_bundled_default_has_a_volume_and_read_only_rbac_per_namespace():
    docs = _render(*NS)
    roles = _kind(docs, "Role")

    assert [claim["metadata"]["name"] for claim in _kind(docs, "StatefulSet")[0]["spec"]["volumeClaimTemplates"]] == [
        "data"
    ]
    assert sorted(role["metadata"]["namespace"] for role in roles) == ["team-a", "team-b"]
    assert len(_kind(docs, "RoleBinding")) == 2
    assert not _kind(docs, "ClusterRole") and not _kind(docs, "ClusterRoleBinding")
    assert {verb for role in roles for rule in role["rules"] for verb in rule["verbs"]} == {"get", "list"}
    assert _pod(docs)["automountServiceAccountToken"] is True


def test_the_default_network_policy_admits_no_ingress():
    policy = _kind(_render(*NS), "NetworkPolicy")[0]
    assert not policy["spec"]["ingress"]


def test_the_runner_script_is_valid_shell_and_does_not_need_curl():
    docs = _render(*NS)
    for script in ("runner.sh", "dashboard.sh"):
        body = _config(docs)[script]
        result = subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "curl" not in "\n".join(line.split("#")[0] for line in body.splitlines())


def test_the_runner_stops_with_a_message_when_kubectl_is_missing():
    docs = _render(*NS)
    runner = _config(docs)["runner.sh"]

    assert _env(_containers(docs)["runner"])["LOG_CONNECTOR"] == "kubectl_logs"
    assert "command -v kubectl" in runner and "exit 1" in runner


def test_the_log_connector_name_is_passed_through_to_the_configuration():
    config = _generated_toml(_render(*NS, "--set", "logs.connector=other_pod_logs"))

    assert config["log"]["applog"]["component"] == "other_pod_logs"
    assert config["log"]["applog"]["other_pod_logs"]["since_context_start"] is True


def test_kueue_selects_workloads_and_passes_the_api_version():
    docs = _render(*NS, "--set", "eca.component=kueue", "--set", "eca.workloadApiVersion=kueue.x-k8s.io/v1beta2")
    resources = {r for role in _kind(docs, "Role") for rule in role["rules"] for r in rule["resources"]}

    assert _generated_toml(docs)["eca"]["kueue"]["workload_api_version"] == "kueue.x-k8s.io/v1beta2"
    assert "workloads" in resources and "podgroups" not in resources


def test_cluster_scope_uses_a_cluster_role_instead_of_roles():
    docs = _render(*NS, "--set", "rbac.scope=cluster")
    assert _kind(docs, "ClusterRole") and _kind(docs, "ClusterRoleBinding")
    assert not _kind(docs, "Role")


def test_rbac_can_be_left_to_the_user():
    docs = _render(*NS, "--set", "rbac.create=false")
    assert not any(doc["kind"] in {"Role", "RoleBinding", "ClusterRole", "ClusterRoleBinding"} for doc in docs)


def test_extra_args_reach_the_runner_command():
    assert '"--strict"' in _config(_render(*NS, "--set", "runner.extraArgs={--strict}"))["runner.sh"]


def test_runner_timing_renders_as_integers_from_a_values_file(tmp_path):
    """Helm reads a values file as floats, which print as 1.2096e+06 from a million up."""
    docs = _render_values(
        tmp_path,
        {
            "image": {"repository": "registry.example/attribution", "tag": "1.0"},
            "namespaces": ["team-a"],
            "runner": {"initialLookbackSeconds": 1209600, "settleSeconds": 900, "intervalSeconds": 300},
        },
    )
    env = _env(_containers(docs)["runner"])

    assert env["INITIAL_LOOKBACK_SECONDS"] == "1209600"
    assert env["SETTLE_SECONDS"] == "900" and env["INTERVAL_SECONDS"] == "300"


def test_network_policy_and_ingress_are_optional():
    docs = _render(*NS, "--set", "networkPolicy.enabled=false")
    assert not _kind(docs, "NetworkPolicy")
    assert not _kind(docs, "Ingress")


def test_ingress_renders_with_its_controller_allowed_through_the_policy():
    docs = _render(
        *NS,
        "--set", "ingress.enabled=true",
        "--set", "ingress.className=nginx",
        "--set", "ingress.hosts[0].host=attribution.example.com",
        "--set", r"networkPolicy.dashboard.from[0].namespaceSelector.matchLabels.kubernetes\.io/metadata\.name=ingress-nginx",
    )  # fmt: skip
    ingress = _kind(docs, "Ingress")[0]

    assert ingress["spec"]["ingressClassName"] == "nginx"
    assert ingress["spec"]["rules"][0]["host"] == "attribution.example.com"
    assert _kind(docs, "NetworkPolicy")[0]["spec"]["ingress"][0]["ports"] == [{"port": 8080, "protocol": "TCP"}]


# --- a Service-only release -------------------------------------------------


def test_service_only_release_needs_no_namespaces_rbac_volume_or_api_token():
    docs = _render("--set", "runner.enabled=false", "--set", "dashboard.enabled=false", "--set", "service.api.enabled=true",
                   "--set", r"networkPolicy.service.from[0].namespaceSelector.matchLabels.team=runners")  # fmt: skip
    statefulset = _kind(docs, "StatefulSet")[0]

    assert list(_containers(docs)) == ["service"]
    assert not any(
        doc["kind"] in {"Role", "RoleBinding", "ClusterRole", "ClusterRoleBinding", "ConfigMap", "Ingress"}
        for doc in docs
    )
    assert "volumeClaimTemplates" not in statefulset["spec"]
    assert "volumes" not in _pod(docs) or not _pod(docs)["volumes"]
    assert _pod(docs)["automountServiceAccountToken"] is False


def test_service_only_release_exposes_the_api_to_the_named_sources():
    docs = _render("--set", "runner.enabled=false", "--set", "dashboard.enabled=false", "--set", "service.api.enabled=true",
                   "--set", r"networkPolicy.service.from[0].namespaceSelector.matchLabels.team=runners")  # fmt: skip
    service = _containers(docs)["service"]
    api = _kind(docs, "Service")[0]
    probe = service["readinessProbe"]

    assert api["metadata"]["name"] == "rel-attribution-api"
    assert [port["port"] for port in api["spec"]["ports"]] == [8000]
    assert service["command"][-4:] == ["--host", "0.0.0.0", "--port", "8000"]
    assert probe["httpGet"] == {"path": "/v2/healthz", "port": "api"}
    assert _kind(docs, "NetworkPolicy")[0]["spec"]["ingress"] == [
        {
            "from": [{"namespaceSelector": {"matchLabels": {"team": "runners"}}}],
            "ports": [{"port": 8000, "protocol": "TCP"}],
        }
    ]


def test_service_settings_accept_env_secret_references_and_volumes():
    docs = _render(
        "--set", "runner.enabled=false", "--set", "dashboard.enabled=false",
        "--set", "service.env[0].name=ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "--set", "service.env[0].value=512MB",
        "--set", "service.envFrom[0].secretRef.name=attribution-service",
        "--set", "service.extraVolumeMounts[0].name=scratch", "--set", "service.extraVolumeMounts[0].mountPath=/scratch",
        "--set", "extraVolumes[0].name=scratch", "--set", "extraVolumes[0].emptyDir.sizeLimit=1Gi",
    )  # fmt: skip
    service = _containers(docs)["service"]

    assert {"name": "ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "value": "512MB"} in service["env"]
    assert service["envFrom"] == [{"secretRef": {"name": "attribution-service"}}]
    assert service["volumeMounts"] == [{"name": "scratch", "mountPath": "/scratch"}]
    assert _pod(docs)["volumes"] == [{"name": "scratch", "emptyDir": {"sizeLimit": "1Gi"}}]


# --- a runner that calls an external Service --------------------------------


def test_runner_with_an_external_service_starts_no_service_and_calls_the_url():
    url = "http://attribution-api.attribution-service.svc:8000/"
    docs = _render(*NS, "--set", "service.enabled=false", "--set", f"runner.serviceUrl={url}")

    assert list(_containers(docs)) == ["runner", "dashboard"]
    assert _generated_toml(docs)["attribution"]["service_url"] == url.rstrip("/")
    assert _env(_containers(docs)["runner"])["SERVICE_URL"] == url.rstrip("/")
    assert not any(item["metadata"]["name"].endswith("-api") for item in _kind(docs, "Service"))
    assert _kind(docs, "Role")


def test_runner_can_call_an_external_service_while_the_bundled_one_stays_enabled():
    docs = _render(*NS, "--set", "runner.serviceUrl=https://service.example.com")
    assert _generated_toml(docs)["attribution"]["service_url"] == "https://service.example.com"
    assert "service" in _containers(docs)


# --- a user-provided Blueprint configuration --------------------------------


@pytest.mark.parametrize("kind", ["configMap", "secret"])
def test_a_user_provided_file_is_mounted_as_the_whole_configuration(kind):
    docs = _render(
        *NS, "--set", f"config.existing.{kind}.name=my-config", "--set", f"config.existing.{kind}.key=custom.toml"
    )
    sources = next(volume for volume in _pod(docs)["volumes"] if volume["name"] == "config")["projected"]["sources"]
    user = [source for source in sources if kind in source and source[kind]["name"] == "my-config"]

    assert "blueprint.toml" not in _config(docs)
    assert user[0][kind]["items"] == [{"key": "custom.toml", "path": "blueprint.toml"}]
    assert "runner.sh" in _config(docs)


def test_a_user_provided_file_still_gets_the_chart_rbac_for_the_runner_namespaces():
    docs = _render(*NS, "--set", "config.existing.configMap.name=my-config")
    assert sorted(role["metadata"]["namespace"] for role in _kind(docs, "Role")) == ["team-a", "team-b"]


def test_the_example_configuration_file_is_valid_toml_that_matches_the_chart():
    manifest = yaml.safe_load((EXAMPLES / "user-provided-toml-configmap.yaml").read_text())
    config = tomllib.loads(manifest["data"]["blueprint.toml"])

    assert config["attribution"]["service_url"] == "http://127.0.0.1:8000"
    assert config["output"]["db"]["sqlite"]["path"] == "/data/attributions.db"
    assert config["eca"]["component"] == "kueue"


# --- a dashboard-only release ------------------------------------------------


def test_dashboard_only_release_reads_an_existing_claim_and_needs_no_rbac():
    docs = _render(
        "--set", "service.enabled=false", "--set", "runner.enabled=false", "--set", "persistence.existingClaim=results"
    )
    statefulset = _kind(docs, "StatefulSet")[0]

    assert list(_containers(docs)) == ["dashboard"]
    assert "volumeClaimTemplates" not in statefulset["spec"]
    assert {"name": "data", "persistentVolumeClaim": {"claimName": "results"}} in _pod(docs)["volumes"]
    assert not any(doc["kind"] in {"Role", "RoleBinding"} for doc in docs)
    assert "dashboard.sh" in _config(docs) and "runner.sh" not in _config(docs)


def test_dashboard_probes_are_configured_separately_from_the_service():
    docs = _render(
        *NS, "--set", "dashboard.probes.path=/healthz", "--set", "dashboard.probes.readiness.periodSeconds=3"
    )
    dashboard = _containers(docs)["dashboard"]

    assert dashboard["readinessProbe"] == {
        "initialDelaySeconds": 15,
        "periodSeconds": 3,
        "httpGet": {"path": "/healthz", "port": "web"},
    }
    assert "exec" in _containers(docs)["service"]["readinessProbe"]


# --- images, commands and storage -------------------------------------------


def test_a_component_can_use_its_own_image_and_command():
    docs = _render(
        *NS,
        "--set", "service.image.repository=registry.example/service", "--set-string", "service.image.tag=9.9",
        "--set", "service.command={uvicorn,app.main:app}", "--set", "dashboard.args={--verbose}",
    )  # fmt: skip
    containers = _containers(docs)

    assert containers["service"]["image"] == "registry.example/service:9.9"
    assert containers["service"]["command"] == ["uvicorn", "app.main:app"]
    assert containers["runner"]["image"] == "registry.example/attribution:1.2.3"
    assert containers["dashboard"]["args"] == ["--verbose"]


def test_without_a_volume_the_runner_and_dashboard_use_an_empty_dir():
    docs = _render(*NS, "--set", "persistence.enabled=false")
    statefulset = _kind(docs, "StatefulSet")[0]

    assert "volumeClaimTemplates" not in statefulset["spec"]
    assert {"name": "data", "emptyDir": {}} in _pod(docs)["volumes"]


def test_a_service_only_release_mounts_no_volume_even_when_persistence_is_on():
    docs = _render(
        "--set", "runner.enabled=false", "--set", "dashboard.enabled=false", "--set", "persistence.enabled=true"
    )
    assert "volumeClaimTemplates" not in _kind(docs, "StatefulSet")[0]["spec"]
    assert not _pod(docs).get("volumes")


def test_the_service_probe_command_can_be_replaced():
    docs = _render(*NS, "--set", "service.probes.command={python3,-c,pass}")
    assert _containers(docs)["service"]["readinessProbe"]["exec"]["command"] == ["python3", "-c", "pass"]


# --- the example values files -----------------------------------------------


@pytest.mark.parametrize(
    ("example", "containers"),
    [
        ("bundled.yaml", ["service", "runner", "dashboard"]),
        ("service-only.yaml", ["service"]),
        ("runner-external-service.yaml", ["runner", "dashboard"]),
        ("user-provided-toml.yaml", ["service", "runner", "dashboard"]),
        ("dashboard-only.yaml", ["dashboard"]),
    ],
)
def test_every_example_renders(example, containers):
    result = _helm("-f", str(EXAMPLES / example))
    assert result.returncode == 0, result.stderr
    docs = [doc for doc in yaml.safe_load_all(result.stdout) if doc]

    assert list(_containers(docs)) == containers


# --- settings that cannot work together -------------------------------------


@pytest.mark.parametrize(
    ("args", "image", "message"),
    [
        (NS, False, "an image repository is required"),
        (["--set", "image.repository=r/a", *NS], False, "an image tag is required"),
        ([], True, "namespaces is required"),
        (["--set", "runner.enabled=false", "--set", "service.enabled=false", "--set", "dashboard.enabled=false"], True, "at least one"),
        ([*NS, "--set", "service.enabled=false"], True, "runner.serviceUrl is required"),
        ([*NS, "--set", "eca.component=static"], True, "eca"),
        ([*NS, "--set", "logs.connector=Not-A-Name"], True, "logs"),
        ([*NS, "--set", "rbac.scope=everything"], True, "rbac"),
        (["--set", "namespaces={Team_A}"], True, "namespaces"),
        (["--set", "namespaces={team-a,team-a}"], True, "namespaces"),
        ([*NS, "--set", "bogusKey=1"], True, "bogusKey"),
        ([*NS, "--set", "runner.settleSeconds=600", "--set", "runner.initialLookbackSeconds=600"], True, "must be larger"),
        ([*NS, "--set", "runner.serviceUrl=attribution.example.com"], True, "serviceUrl"),
        ([*NS, "--set", "ingress.enabled=true"], True, "ingress.hosts is required"),
        ([*NS, "--set", "ingress.enabled=true", "--set", "ingress.hosts[0].host=a.example.com"], True, "networkPolicy.dashboard.from"),
        (["--set", "runner.enabled=false", "--set", "dashboard.enabled=false", "--set", "ingress.enabled=true"], True, "ingress.enabled needs dashboard"),
        ([*NS, "--set", "service.api.enabled=true"], True, "networkPolicy.service.from"),
        (["--set", "service.enabled=false", "--set", "runner.enabled=false", "--set", "service.api.enabled=true"], True, "service.api.enabled needs"),
        ([*NS, "--set", "config.existing.configMap.name=a", "--set", "config.existing.secret.name=b"], True, "only one of"),
        ([*NS, "--set", "config.existing.configMap.name=a", "--set", "extraConfig=[x]"], True, "extraConfig is not used"),
        (["--set", "runner.enabled=false", "--set", "config.existing.configMap.name=a"], True, "no effect when the runner is disabled"),
        ([*NS, "--set", "service.port=70000"], True, "port"),
        ([*NS, "--set", "dashboard.port=8000"], True, "must differ"),
        ([*NS, "--set", "dashboard.databasePath=/tmp/a.db"], True, "databasePath"),
        ([*NS, "--set", "persistence.enabled=false", "--set", "persistence.existingClaim=c"], True, "existingClaim has no effect"),
        ([*NS, "--set", "service.probes.readiness.httpGet.path=/x"], True, "httpGet"),
    ],
)  # fmt: skip
def test_invalid_values_are_rejected_with_a_message(args, image, message):
    result = _helm(*(IMAGE if image else []), *args)

    assert result.returncode != 0
    assert message in result.stderr
