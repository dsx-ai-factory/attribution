# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Build an `Auth` from the `auth` table every connector configuration shares."""

from __future__ import annotations

import os

from attribution_blueprint.connectors.auth.mtls import MTLSAuth
from attribution_blueprint.connectors.auth.protocol import Auth
from attribution_blueprint.connectors.auth.token import TokenAuth
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM

# Schema of the `auth` sub-table, shared by the connectors that accept one.
AUTH_SCHEMA = {
    "type": _PARAM,
    "cert_path": _PARAM,
    "key_path": _PARAM,
    "ca_cert_path": _PARAM,
    "token_env": _PARAM,
    "org_id": _PARAM,
}


def build_auth(spec: dict, prefix: str) -> Auth:
    """Return the `Auth` described by `spec`; `prefix` names the table in errors."""
    atype = spec.get("type")
    if atype == "token":
        env_name = spec.get("token_env")
        if not env_name:
            raise MCAError(f"{prefix} type=token requires token_env")
        if env_name not in os.environ:
            # Surface as MCAError so the CLI's parser.error path produces a
            # clean usage message and exit code 2, rather than a Python
            # traceback from an uncaught KeyError.
            raise MCAError(f"auth.token_env references unset environment variable: {env_name}")
        return TokenAuth(token=os.environ[env_name], org_id=spec.get("org_id"))
    if atype == "mtls":
        if not spec.get("cert_path") or not spec.get("key_path"):
            raise MCAError(f"{prefix} type=mtls requires cert_path and key_path")
        return MTLSAuth(
            cert_path=spec["cert_path"],
            key_path=spec["key_path"],
            ca_cert_path=spec.get("ca_cert_path"),
            org_id=spec.get("org_id"),
        )
    raise MCAError(f"unknown auth type: {atype!r}")


__all__ = ["AUTH_SCHEMA", "build_auth"]
