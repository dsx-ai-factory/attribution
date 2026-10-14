# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_blueprint.connectors.execution_contexts.kubernetes_daemon import (
    build_kubernetes_daemon_client,
)
from attribution_blueprint.exceptions import MCAError


def test_build_client_requires_endpoint():
    with pytest.raises(MCAError, match="eca.test.kubernetes_daemon requires endpoint"):
        build_kubernetes_daemon_client({}, config_path="eca.test.kubernetes_daemon")


def test_build_client_configures_token_auth(monkeypatch):
    monkeypatch.setenv("DAEMON_TOKEN", "secret")

    client = build_kubernetes_daemon_client(
        {
            "endpoint": "https://daemon.example/",
            "timeout_seconds": 12,
            "auth": {"type": "token", "token_env": "DAEMON_TOKEN", "org_id": "tenant-a"},
        },
        config_path="eca.test.kubernetes_daemon",
    )

    assert client.endpoint == "https://daemon.example"
    assert client.timeout_seconds == 12
    assert client.session.headers["Authorization"] == "Bearer secret"
    assert client.session.headers["X-Scope-OrgID"] == "tenant-a"


def test_build_client_configures_mtls():
    client = build_kubernetes_daemon_client(
        {
            "endpoint": "https://daemon.example",
            "auth": {
                "type": "mtls",
                "cert_path": "/cert.pem",
                "key_path": "/key.pem",
                "ca_cert_path": "/ca.pem",
            },
        },
        config_path="eca.test.kubernetes_daemon",
    )

    assert client.session.cert == ("/cert.pem", "/key.pem")
    assert client.session.verify == "/ca.pem"


@pytest.mark.parametrize("timeout", [0, -1, "invalid"])
def test_build_client_rejects_invalid_timeout(timeout):
    with pytest.raises(MCAError, match="timeout_seconds"):
        build_kubernetes_daemon_client(
            {"endpoint": "http://daemon", "timeout_seconds": timeout},
            config_path="eca.test.kubernetes_daemon",
        )
