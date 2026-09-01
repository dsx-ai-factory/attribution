# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance values for scheduler failed-node attribution."""

from enum import StrEnum


class FailedNodeResolution(StrEnum):
    """Describe how attribution associated a Slurm node failure with a node."""

    SLURM_FAILED_NODE_FIELD = "slurm_failed_node_field"
    SINGLE_ALLOCATED_NODE_INFERENCE = "single_allocated_node_inference"
