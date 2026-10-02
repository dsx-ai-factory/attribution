# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
