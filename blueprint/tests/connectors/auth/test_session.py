# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest

from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth, build_session, credential_identity


def test_build_session_applies_auth_and_mounts_retrying_adapter():
    session = build_session(MTLSAuth("/c.crt", "/c.key", ca_cert_path="/ca.crt", org_id="tenant"))

    assert session.cert == ("/c.crt", "/c.key")
    assert session.verify == "/ca.crt"
    assert session.headers["X-Scope-OrgID"] == "tenant"
    retry = session.get_adapter("https://loki.example").max_retries
    assert retry.total == 2
    assert retry.read == 2
    assert retry.backoff_jitter == 1.0
    assert retry.status_forcelist == (429, 500, 502, 503, 504)
    assert retry.allowed_methods == frozenset({"POST"})
    assert retry.raise_on_status is False
    assert session.get_adapter("http://loki.example").max_retries.total == 2


def test_build_session_without_auth_or_retries():
    session = build_session(None, retries=0)

    assert "Authorization" not in session.headers
    assert session.get_adapter("https://prom.example").max_retries.total == 0
    with pytest.raises(ValueError):
        build_session(None, retries=-1)


def test_credential_identity_is_a_non_secret_digest():
    assert credential_identity(None) == ""
    token = credential_identity(TokenAuth("secret-token"))
    assert "secret-token" not in token
    assert token != credential_identity(TokenAuth("other"))
    assert credential_identity(MTLSAuth("/c.crt", "/c.key")) == credential_identity(MTLSAuth("/c.crt", "/c.key"))
