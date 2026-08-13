# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Health-check observation filtering for the external rule pack."""

from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rules_engine import (
    PreFilterSpec,
    register_pre_filter,
)

rule_author = "NVIDIA Attribution team"


def _epilog_job_id_filter(entry, job_info):
    """Skip epilog entries for other jobs. Dynamic check - can't be a keyword."""
    line = entry.body()
    if "epilog" in line and f"epilog_{job_info.id}" not in line:
        return True  # skip
    return False


register_pre_filter(
    ObsSource.HEALTH_CHECK,
    PreFilterSpec(exclude_terms=["laso", "prolog"], callables=[_epilog_job_id_filter]),
)
