# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slurm filename-pattern substitution.

Slurm's sbatch `--output=` / `--error=` templates support
`%j`/`%x`/`%u`/`%A`/`%a`/`%n`/`%t`/`%N`/`%s`/`%%` filename patterns and a
fixed set of `${SLURM_*}` env-style tokens (plus their `%q{...}`
equivalents). sacct's `StdOut`/`StdErr` fields may return the literal
template still containing these tokens, depending on the slurmctld
configuration. This module pre-renders those paths inside the Slurm ECA
so all downstream consumers (the `local_file` log connector, in
particular) only see fully-resolved filesystem paths and never need to
parse Slurm-specific token syntax.

This file deliberately lives inside the Slurm ECA package: Slurm
filename-pattern syntax is a Slurm-specific external contract, and per
ARCHITECTURE.md Rule 3 connector-specific parsing belongs next to the
connector that owns it. The `local_file` log connector handles only
generic Python-format `{X}` placeholders against cross-cutting
`ExecutionContext` fields.

Token resolution rules:
- `%j` / `${SLURM_JOBID}` / `${SLURM_JOB_ID}` → ctx["job_id_raw"] or ctx["job_id"]
- `%x` / `${SLURM_JOB_NAME}` → ctx["name"]
- `%u` → ctx["user_name"]
- `%A` / `${SLURM_ARRAY_JOB_ID}` → ctx["array_id"] or ctx["job_id"]
- `%a` / `${SLURM_ARRAY_TASK_ID}` → ctx["attempt_index"] (else "0")
- `%n` / `%t` / `%s` → "0" (rank/node/step proxy; pick rank-0 as the single-file proxy)
- `%N` → first hostname in ctx["nodes"]
- `%%` → "%"
"""

import re
from typing import Any

# Slurm filename patterns documented at
# https://slurm.schedmd.com/sbatch.html#SECTION_filename-pattern
_SLURM_PATTERN_RE = re.compile(r"%(j|x|u|A|a|n|t|N|s|%)")

# Env-style tokens (and their `%q{...}` equivalents) Slurm exposes in
# filename templates. The tuple is (literal token, ctx-field name).
_ENV_TOKEN_PATTERNS: list[tuple[str, str]] = [
    ("${SLURM_JOB_NAME}", "name"),
    ("%q{SLURM_JOB_NAME}", "name"),
    ("${SLURM_JOBID}", "job_id"),
    ("%q{SLURM_JOBID}", "job_id"),
    ("${SLURM_JOB_ID}", "job_id"),
    ("%q{SLURM_JOB_ID}", "job_id"),
    ("${SLURM_ARRAY_JOB_ID}", "array_id"),
    ("%q{SLURM_ARRAY_JOB_ID}", "array_id"),
    ("${SLURM_ARRAY_TASK_ID}", "attempt_index"),
    ("%q{SLURM_ARRAY_TASK_ID}", "attempt_index"),
]


def substitute(path: str, ctx: dict[str, Any]) -> str:
    """Resolve all Slurm filename patterns in `path` against the Slurm ctx.

    Idempotent: running on a fully-resolved path is a no-op (there's
    nothing left to match). Safe to call on the empty string.
    """
    if not path:
        return path

    # `%j` and `${SLURM_JOBID}` resolve to the unique numeric jobid Slurm
    # wrote at filename-creation time, NOT the array-spec form. For array
    # tasks `job_id` is "1113288_265" but the file on disk is "1117669.out".
    job_id_for_path = ctx.get("job_id_raw") or ctx.get("job_id", "") or ""
    array_id = ctx.get("array_id") or ctx.get("job_id", "")
    attempt_index = ctx.get("attempt_index")
    env_values: dict[str, str] = {
        "name": ctx.get("name", "") or "",
        "job_id": job_id_for_path,
        "array_id": array_id or "",
        "attempt_index": str(attempt_index) if attempt_index is not None else "0",
    }
    for token, key in _ENV_TOKEN_PATTERNS:
        path = path.replace(token, env_values[key])

    def replace(match: re.Match) -> str:
        code = match.group(1)
        if code == "%":
            return "%"
        if code == "j":
            return job_id_for_path
        if code == "x":
            return ctx.get("name", "") or ""
        if code == "u":
            return ctx.get("user_name", "") or ""
        if code == "A":
            return array_id or ""
        if code == "a":
            return str(attempt_index) if attempt_index is not None else "0"
        if code in ("n", "t", "s"):
            # Rank/node/step proxy — pick rank 0 as the single-file proxy.
            return "0"
        if code == "N":
            nodes = ctx.get("nodes") or []
            return nodes[0] if nodes else ""
        return ""

    return _SLURM_PATTERN_RE.sub(replace, path)
