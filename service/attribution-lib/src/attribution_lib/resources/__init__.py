# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Annotated

from pydantic import Field, TypeAdapter

from .base import Base

# Import concrete model classes in dependency order
from .node import Node
from .replica import Replica

# Define discriminated union AFTER imports
_Resource = Annotated[Node | Replica, Field(discriminator="kind")]

Base.model_rebuild()
Node.model_rebuild()
Replica.model_rebuild()

Resource: TypeAdapter[_Resource] = TypeAdapter(_Resource)

# These imports come after `Resource` is defined because resource_set.py
# imports Resource from this module; ordering them before the TypeAdapter
# construction above would create a circular import.
from attribution_lib.exceptions import NoNodesError  # noqa: E402

from .resource_set import AllJobResources, ResourceSet  # noqa: E402


def get_single_node(resources: ResourceSet) -> Node:
    for r in resources:
        if r.kind == "node":
            assert isinstance(r, Node)
            return r
    raise NoNodesError("No nodes in ResourceSet")


# Define public API
__all__ = ["Node", "Replica", "ResourceSet", "AllJobResources", "Resource", "get_single_node"]
