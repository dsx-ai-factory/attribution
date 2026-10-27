# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from .base import Base, set_key


@set_key(key="hostname")
class Node(Base):
    kind: Literal["node"] = "node"
    hostname: str
