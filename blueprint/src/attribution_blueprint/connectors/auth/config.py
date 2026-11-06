# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the auth table schema shared by connector configurations."""

from __future__ import annotations

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


__all__ = ["AUTH_SCHEMA"]
