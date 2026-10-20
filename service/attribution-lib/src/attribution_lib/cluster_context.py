# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated cluster-incident context supplied by attribution clients."""

from enum import StrEnum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class ClusterIncidentScope(StrEnum):
    """Supported physical scopes for a cluster-context candidate."""

    NODE = "node"
    DOMAIN = "domain"
    CLUSTER = "cluster"


class ClusterIncidentSignature(StrEnum):
    """Stable failure classes accepted from cluster-context producers."""

    GPU_ECC_FAULT = "gpu_ecc_fault"
    GPU_UNREACHABLE = "gpu_unreachable"
    FABRIC_NETWORK_FAULT = "fabric_network_fault"
    NODE_UNREACHABLE = "node_unreachable"
    STORAGE_FAULT = "storage_fault"
    OTHER = "other"


class ClusterIncidentCandidate(BaseModel):
    """Stable incident attributes that an outcome refiner may consume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_scope: ClusterIncidentScope
    signature_class: ClusterIncidentSignature
    non_job_nodes_affected: Optional[_NonNegativeInt] = None
    non_job_domains_affected: Optional[_NonNegativeInt] = None
    total_non_job_events: Optional[_NonNegativeInt] = None
    failure_episode_id: Optional[str] = None
    cluster_incident_id: Optional[str] = None
    incident_start_settled: Optional[bool] = None


class ClusterIncidentContext(BaseModel):
    """Narrow client-projected cluster context for outcome refinement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[ClusterIncidentCandidate, ...] = ()
