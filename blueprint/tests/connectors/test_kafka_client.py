# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amazon MSK IAM settings shared by the Kafka input and output connectors."""

import base64
import os
import sys
import types
from urllib.parse import parse_qs, urlparse

import pytest

from attribution_blueprint.connectors.execution_contexts.kafka import _consumer_config
from attribution_blueprint.connectors.kafka_client import security_config
from attribution_blueprint.connectors.outputs.kafka import _build as build_output
from attribution_blueprint.exceptions import MCAError

MSK_IAM = {"security_protocol": "SASL_SSL", "sasl_mechanism": "AWS_MSK_IAM", "aws_region": "us-east-2"}


@pytest.fixture
def signer(monkeypatch):
    """Install a fake `aws_msk_iam_sasl_signer` that records each signing call."""
    calls = []

    class MSKAuthTokenProvider:
        @staticmethod
        def generate_auth_token(region):
            calls.append(("default", region))
            return "example-token", 1_800_000_000_000

    module = types.ModuleType("aws_msk_iam_sasl_signer")
    module.MSKAuthTokenProvider = MSKAuthTokenProvider
    monkeypatch.setitem(sys.modules, "aws_msk_iam_sasl_signer", module)
    return calls


def test_msk_iam_uses_oauthbearer_with_signed_tokens(signer):
    config = security_config({**MSK_IAM, "ssl_ca_location": "/example/ca.pem"}, "eca.kafka")

    assert config["security.protocol"] == "SASL_SSL"
    assert config["ssl.ca.location"] == "/example/ca.pem"
    assert config["sasl.mechanism"] == "OAUTHBEARER"
    assert "sasl.username" not in config
    assert "sasl.password" not in config
    assert config["oauth_cb"]("") == ("example-token", 1_800_000_000.0)
    assert config["oauth_cb"]("") == ("example-token", 1_800_000_000.0)
    assert signer == [("default", "us-east-2"), ("default", "us-east-2")]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        *[({"aws_region": value}, "aws_region must be") for value in (None, "", " ", "us east-2", True, 123, [])],
        *[
            ({"security_protocol": value}, "requires security_protocol=SASL_SSL")
            for value in (None, "", "PLAINTEXT", "SSL", "SASL_PLAINTEXT")
        ],
    ],
)
def test_msk_iam_rejects_incomplete_settings(signer, override, message):
    with pytest.raises(MCAError, match=message):
        security_config({**MSK_IAM, **override}, "eca.kafka")


def test_msk_iam_requires_the_kafka_aws_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "aws_msk_iam_sasl_signer", None)
    with pytest.raises(MCAError, match=r"kafka-aws extra"):
        security_config(dict(MSK_IAM), "output.kafka")


def test_input_consumer_uses_msk_iam(signer):
    config = _consumer_config(
        {"bootstrap_servers": "broker:9098", "topic": "job-events", "group_id": "attribution", **MSK_IAM}
    )

    assert config["sasl.mechanism"] == "OAUTHBEARER"
    assert callable(config["oauth_cb"])


def test_output_producer_uses_msk_iam(signer, mocker):
    native = pytest.importorskip("confluent_kafka")
    factory = mocker.patch.object(native, "Producer")

    build_output({"bootstrap_servers": "broker:9098", "topic": "results", **MSK_IAM}, {})

    config = factory.call_args.args[0]
    assert config["sasl.mechanism"] == "OAUTHBEARER"
    assert config["oauth_cb"]("") == ("example-token", 1_800_000_000.0)


def test_msk_iam_rejects_omitted_security_protocol(signer):
    spec = {key: value for key, value in MSK_IAM.items() if key != "security_protocol"}
    with pytest.raises(MCAError, match="requires security_protocol=SASL_SSL"):
        security_config(spec, "eca.kafka")


def test_msk_iam_signing_failure_can_recover(signer, monkeypatch):
    provider = sys.modules["aws_msk_iam_sasl_signer"].MSKAuthTokenProvider
    callback = security_config(dict(MSK_IAM), "eca.kafka")["oauth_cb"]
    original = provider.generate_auth_token

    def fail(region):
        raise ValueError("synthetic credential failure")

    monkeypatch.setattr(provider, "generate_auth_token", fail)
    with pytest.raises(ValueError, match="synthetic credential failure"):
        callback("")
    monkeypatch.setattr(provider, "generate_auth_token", original)
    assert callback("") == ("example-token", 1_800_000_000.0)


def test_msk_iam_reads_rotated_credentials_file(tmp_path, monkeypatch):
    """Use the real optional signer with synthetic credentials and no AWS requests."""
    pytest.importorskip("aws_msk_iam_sasl_signer")
    for key in list(os.environ):
        if key.startswith("AWS_") or key == "BOTO_CONFIG":
            monkeypatch.delenv(key)
    credentials = tmp_path / "credentials"
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("BOTO_CONFIG", str(tmp_path / "boto"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    callback = security_config(dict(MSK_IAM), "eca.kafka")["oauth_cb"]

    for identity in ("EXAMPLE_FIRST", "EXAMPLE_SECOND"):
        credentials.write_text(
            f"[default]\naws_access_key_id = {identity}\naws_secret_access_key = synthetic-test-only\n"
        )
        token, expiry = callback("")
        signed_url = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
        query = parse_qs(urlparse(signed_url).query)
        assert query["X-Amz-Credential"][0].split("/")[0] == identity
        assert expiry > 0
