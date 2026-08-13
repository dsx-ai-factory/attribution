# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.execution_contexts.slurm.connector import (
    SlurmExecutionContext,
    SlurmExecutionContextAdapter,
)
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM


def _build(spec: dict, ctx: dict) -> SlurmExecutionContextAdapter:
    try:
        return SlurmExecutionContextAdapter(
            sacct_path=spec.get("sacct_path", "sacct"),
            cluster_tz=spec.get("cluster_tz"),
        )
    except ValueError as exc:
        raise MCAError(str(exc)) from exc


_registry.register_component(
    framework="eca",
    name="slurm",
    schema={
        "ids": _PARAM,
        "start_time": _PARAM,
        "end_time": _PARAM,
        "nodelist": _PARAM,
        "sacct_path": _PARAM,
        # IANA tz name (e.g. "America/Los_Angeles"); attached to naive sacct
        # timestamps before normalization. Defaults to the blueprint host's
        # local tz, which is correct only when blueprint and slurmctld run on
        # the same host or in the same tz.
        "cluster_tz": _PARAM,
    },
    build=_build,
)


__all__ = ["SlurmExecutionContext", "SlurmExecutionContextAdapter"]
