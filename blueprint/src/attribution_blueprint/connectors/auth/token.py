# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
