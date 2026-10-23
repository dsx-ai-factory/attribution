# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.connectors.execution_contexts.slurm.filename_patterns import substitute


def test_substitute_resolves_slurm_percent_patterns() -> None:
    """Each supported Slurm percent token resolves from execution context data."""
    context = {
        "job_id": "1200_7",
        "job_id_raw": "1200",
        "array_id": "1100",
        "attempt_index": 7,
        "name": "training",
        "user_name": "alice",
        "nodes": ["node-a", "node-b"],
    }

    rendered = substitute("%j-%x-%u-%A-%a-%n-%t-%N-%s-%%.log", context)

    assert rendered == "1200-training-alice-1100-7-0-0-node-a-0-%.log"


def test_substitute_resolves_environment_and_q_tokens() -> None:
    """Environment-style and q-style aliases use the same Slurm values."""
    context = {
        "job_id": "1200_7",
        "job_id_raw": "1200",
        "array_id": "1100",
        "attempt_index": 7,
        "name": "training",
    }
    template = (
        "${SLURM_JOB_NAME}-%q{SLURM_JOB_NAME}-"
        "${SLURM_JOBID}-%q{SLURM_JOB_ID}-"
        "${SLURM_ARRAY_JOB_ID}-%q{SLURM_ARRAY_TASK_ID}.log"
    )

    assert substitute(template, context) == "training-training-1200-1200-1100-7.log"


def test_substitute_uses_documented_fallbacks_for_missing_values() -> None:
    """Missing array, attempt, node, name, and user fields resolve predictably."""
    context = {"job_id": "1200"}

    assert substitute("%A-%a-%N-%x-%u.log", context) == "1200-0---.log"


def test_substitute_is_safe_for_empty_and_resolved_paths() -> None:
    """Empty and already resolved paths pass through unchanged."""
    assert substitute("", {}) == ""
    assert substitute("/logs/job-1200.out", {}) == "/logs/job-1200.out"
