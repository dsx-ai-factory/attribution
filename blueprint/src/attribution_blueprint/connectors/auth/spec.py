# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from attribution_blueprint.connectors.auth.mtls import MTLSAuth
from attribution_blueprint.connectors.auth.protocol import Auth
from attribution_blueprint.connectors.auth.token import TokenAuth
from attribution_blueprint.exceptions import MCAError


def build_auth(spec: dict, component: str) -> Auth:
    atype = spec.get("type")
    if atype == "token":
        env_name = spec.get("token_env")
        if not env_name:
            raise MCAError(f"{component}.auth type=token requires token_env")
        if env_name not in os.environ:
            raise MCAError(f"auth.token_env references unset environment variable: {env_name}")
        return TokenAuth(token=os.environ[env_name], org_id=spec.get("org_id"))
    if atype == "mtls":
        if not spec.get("cert_path") or not spec.get("key_path"):
            raise MCAError(f"{component}.auth type=mtls requires cert_path and key_path")
        return MTLSAuth(
            cert_path=spec["cert_path"],
            key_path=spec["key_path"],
            ca_cert_path=spec.get("ca_cert_path"),
            org_id=spec.get("org_id"),
        )
    raise MCAError(f"unknown auth type: {atype!r}")
