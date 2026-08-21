# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Provenance values for node-not-responding attribution."""

from enum import StrEnum


class NodeNotRespondingProvenance(StrEnum):
    """Describe the evidence source used to infer that a node did not respond."""

    EPILOG_HEALTH_CHECK = "epilog_health_check_not_responding"
    SLURM_NODE_REASON = "slurm_node_reason_not_responding"
