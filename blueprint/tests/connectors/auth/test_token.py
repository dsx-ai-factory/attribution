# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.connectors.auth import TokenAuth


def test_token_only():
    assert TokenAuth("abc").request_kwargs() == {"headers": {"Authorization": "Bearer abc"}}


def test_token_with_org_id():
    assert TokenAuth("abc", "tenant-42").request_kwargs() == {
        "headers": {"Authorization": "Bearer abc", "X-Scope-OrgID": "tenant-42"},
    }
