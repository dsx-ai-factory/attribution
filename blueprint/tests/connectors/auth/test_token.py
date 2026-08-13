# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from attribution_blueprint.connectors.auth import TokenAuth


def test_token_only():
    assert TokenAuth("abc").request_kwargs() == {"headers": {"Authorization": "Bearer abc"}}


def test_token_with_org_id():
    assert TokenAuth("abc", "tenant-42").request_kwargs() == {
        "headers": {"Authorization": "Bearer abc", "X-Scope-OrgID": "tenant-42"},
    }
