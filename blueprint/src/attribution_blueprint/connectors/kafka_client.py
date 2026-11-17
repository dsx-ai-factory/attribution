# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kafka client settings shared by the Kafka execution context adapter and output."""

import os
from collections.abc import Callable

from attribution_blueprint.exceptions import MCAError

SECURITY_PROTOCOLS = ("PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL")
AWS_MSK_IAM = "AWS_MSK_IAM"
SASL_MECHANISMS = ("PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512", AWS_MSK_IAM)


def security_config(spec: dict, path: str) -> dict:
    """Translate connection security keys into librdkafka settings.

    Credentials are read from the environment variables named by
    `sasl_username_env` / `sasl_password_env`, never from the TOML itself.
    `AWS_MSK_IAM` instead signs Amazon MSK IAM tokens with credentials from the
    standard AWS credential chain.

    Args:
        spec: Connector configuration section.
        path: Configuration path used in error messages, such as `eca.kafka`.

    Returns:
        `security.protocol`, optional `ssl.ca.location`, and SASL settings.
    """
    security_protocol = spec.get("security_protocol") or "PLAINTEXT"
    if security_protocol not in SECURITY_PROTOCOLS:
        raise MCAError(f"{path}.security_protocol must be one of {list(SECURITY_PROTOCOLS)}")
    config = {"security.protocol": security_protocol}
    if spec.get("ssl_ca_location"):
        config["ssl.ca.location"] = str(spec["ssl_ca_location"])

    mechanism = spec.get("sasl_mechanism")
    if mechanism == AWS_MSK_IAM:
        config.update(_msk_iam_config(spec, path, security_protocol))
        return config
    if security_protocol.startswith("SASL_"):
        if mechanism not in SASL_MECHANISMS:
            raise MCAError(f"{path}.sasl_mechanism must be one of {list(SASL_MECHANISMS)}")
        config["sasl.mechanism"] = mechanism
        for key, setting in (("sasl_username_env", "sasl.username"), ("sasl_password_env", "sasl.password")):
            env_name = spec.get(key)
            if not env_name:
                raise MCAError(f"{path} security_protocol={security_protocol} requires {key}")
            if env_name not in os.environ:
                raise MCAError(f"{path}.{key} references unset environment variable: {env_name}")
            config[setting] = os.environ[env_name]
    return config


def _msk_iam_config(spec: dict, path: str, security_protocol: str) -> dict:
    """Return OAUTHBEARER settings that authenticate to Amazon MSK with IAM.

    Args:
        spec: Connector configuration section.
        path: Configuration path used in error messages.
        security_protocol: The validated security protocol.

    Returns:
        `sasl.mechanism` and an `oauth_cb` that signs a fresh MSK IAM token.
    """
    if security_protocol != "SASL_SSL":
        raise MCAError(f"{path}.sasl_mechanism={AWS_MSK_IAM} requires security_protocol=SASL_SSL")
    region = spec.get("aws_region")
    if not isinstance(region, str) or not region or any(char.isspace() for char in region):
        raise MCAError(f"{path}.aws_region must be a non-empty string without whitespace")
    return {"sasl.mechanism": "OAUTHBEARER", "oauth_cb": msk_iam_token_callback(region, path)}


def msk_iam_token_callback(region: str, path: str) -> Callable[[str], tuple[str, float]]:
    """Build a librdkafka OAUTHBEARER callback that signs Amazon MSK IAM tokens.

    Each call resolves the AWS credential chain again, including updates to a
    shared credentials file. External changes cannot update a process's environment.

    Args:
        region: AWS region of the MSK cluster.
        path: Configuration path used in error messages.

    Returns:
        A callback returning the token and its expiry in seconds since the epoch.
    """
    try:
        from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
    except ImportError as exc:
        raise MCAError(
            f"{path}.sasl_mechanism={AWS_MSK_IAM} requires the kafka-aws extra: "
            "install nvidia-attribution-blueprint[kafka-aws]"
        ) from exc

    def token(_oauthbearer_config: str) -> tuple[str, float]:
        """Sign with current AWS credentials and convert expiry to epoch seconds."""
        value, expiry_ms = MSKAuthTokenProvider.generate_auth_token(region)
        return value, expiry_ms / 1000

    return token


SECURITY_SCHEMA_KEYS = (
    "security_protocol",
    "sasl_mechanism",
    "sasl_username_env",
    "sasl_password_env",
    "ssl_ca_location",
    "aws_region",
)
