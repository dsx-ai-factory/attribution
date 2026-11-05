# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kafka client settings shared by the Kafka execution context adapter and output."""

import os

from attribution_blueprint.exceptions import MCAError

SECURITY_PROTOCOLS = ("PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL")
SASL_MECHANISMS = ("PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512")


def security_config(spec: dict, path: str) -> dict:
    """Translate connection security keys into librdkafka settings.

    Credentials are read from the environment variables named by
    `sasl_username_env` / `sasl_password_env`, never from the TOML itself.

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

    if security_protocol.startswith("SASL_"):
        mechanism = spec.get("sasl_mechanism")
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


SECURITY_SCHEMA_KEYS = (
    "security_protocol",
    "sasl_mechanism",
    "sasl_username_env",
    "sasl_password_env",
    "ssl_ca_location",
)
