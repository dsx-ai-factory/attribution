# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenAuth:
    token: str
    org_id: str | None = None

    def request_kwargs(self) -> dict:
        headers = {"Authorization": f"Bearer {self.token}"}
        if self.org_id is not None:
            headers["X-Scope-OrgID"] = self.org_id
        return {"headers": headers}
