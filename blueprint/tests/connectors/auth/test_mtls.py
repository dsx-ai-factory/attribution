# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

from attribution_blueprint.connectors.auth import MTLSAuth


def test_expands_leading_tilde():
    auth = MTLSAuth("~/certs/client.crt", "~/certs/client.key", ca_cert_path="~/certs/ca.crt")
    home = os.path.expanduser("~")
    assert auth.request_kwargs() == {
        "cert": (f"{home}/certs/client.crt", f"{home}/certs/client.key"),
        "verify": f"{home}/certs/ca.crt",
    }


def test_cert_and_key_only():
    assert MTLSAuth("/c", "/k").request_kwargs() == {"cert": ("/c", "/k")}


def test_with_ca_cert():
    assert MTLSAuth("/c", "/k", ca_cert_path="/ca").request_kwargs() == {
        "cert": ("/c", "/k"),
        "verify": "/ca",
    }


def test_with_org_id():
    assert MTLSAuth("/c", "/k", org_id="tenant-42").request_kwargs() == {
        "cert": ("/c", "/k"),
        "headers": {"X-Scope-OrgID": "tenant-42"},
    }
