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
from dataclasses import dataclass


@dataclass(frozen=True)
class MTLSAuth:
    cert_path: str
    key_path: str
    ca_cert_path: str | None = None
    org_id: str | None = None

    def __post_init__(self) -> None:
        # Expand a leading ~ so configs can reference ~/certs without baking in
        # an absolute home path. Absolute and relative paths pass through.
        object.__setattr__(self, "cert_path", os.path.expanduser(self.cert_path))
        object.__setattr__(self, "key_path", os.path.expanduser(self.key_path))
        if self.ca_cert_path is not None:
            object.__setattr__(self, "ca_cert_path", os.path.expanduser(self.ca_cert_path))

    def request_kwargs(self) -> dict:
        kwargs: dict = {"cert": (self.cert_path, self.key_path)}
        if self.ca_cert_path is not None:
            kwargs["verify"] = self.ca_cert_path
        if self.org_id is not None:
            kwargs["headers"] = {"X-Scope-OrgID": self.org_id}
        return kwargs
