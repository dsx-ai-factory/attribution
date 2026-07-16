# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re
import subprocess
from datetime import datetime
from typing import NotRequired, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

from attribution_blueprint.connectors.execution_contexts.slurm import filename_patterns


class SlurmExecutionContext(TypedDict):
    id: str
    workload_type: str
    # BP-internal connector identity (always "slurm"); consumed by
    # SQLiteOutput as the `scheduler_type` column. Distinct from
    # `workload_type` so wire-format and SQLite labeling can diverge.
    scheduler: str
    nodes: list[str]
    start_time: datetime
    end_time: datetime
    timezone: NotRequired[str]

    job_id: str
    job_id_raw: NotRequired[str]  # Internal numeric jobid; %j at write time
    sluid: str
    # Scheduler-agnostic per-attempt key (= sluid for Slurm). Consumed by
    # cross-cutting code like SQLiteOutput so the unique-row constraint
    # works identically across schedulers.
    attempt_id: str
    parent_job_id: str
    array_id: NotRequired[str]
    # `attempt_index` is the slurm array task id (%a in path templates), not a
    # retry counter. See local_file connector's SLURM_ARRAY_TASK_ID mapping.
    attempt_index: NotRequired[int]
    # `attempt_ordinal` is the 0-based requeue position within a single jobid
    # (earliest attempt by Start = 0). Distinct attempts of the same jobid each
    # have their own SLUID; this field gives downstream eval/UI a stable
    # ordering. Absent for jobids with only one attempt in the sacct output.
    attempt_ordinal: NotRequired[int]
    step_id: NotRequired[str]

    state: str
    name: NotRequired[str]
    user_name: NotRequired[str]
    failed_node: NotRequired[str]
    cancelled_by: NotRequired[str]
    time_limit: NotRequired[int]
    alloc_tres: NotRequired[str]
    std_out: NotRequired[str]
    std_err: NotRequired[str]


# JobName: needed for %x token. JobIDRaw: internal numeric jobid that Slurm
# substitutes for %j at filename-write time. For array tasks, JobID is the
# spec form ("1113288_265") but JobIDRaw is the unique numeric ("1117669")
# — and that's what's in the actual filesystem path.
SACCT_FORMAT = (
    "JobID,JobIDRaw,SLUID,State,Start,End,NodeList,FailedNode,TimelimitRaw,AllocTRES,StdOut,StdErr,User,JobName"
)

# Array job notation: "1113288_265" → array_id=1113288, task_id=265.
# Heterogeneous job notation ("12345+3") and step IDs ("12345.batch") are
# left untouched; %A/%a only apply to array tasks.
_ARRAY_JOB_RE = re.compile(r"^(?P<array_id>\d+)_(?P<task_id>\d+)$")

# A node group is `prefix[range_list]suffix` or a bare name. `suffix` captures
# domain-style trailing segments (e.g. `node[001-004].mgmt.example.com` →
# `node001.mgmt.example.com` ... `node004.mgmt.example.com`); without this,
# FQDN nodelists silently produce wrong node sets.
_NODE_GROUP_RE = re.compile(r"([A-Za-z0-9._-]*?)\[([^\]]+)\]([A-Za-z0-9._-]*)|([A-Za-z0-9._-]+)")


def _expand_nodelist(nodelist: str) -> list[str]:
    if not nodelist or nodelist.strip() in ("None assigned", "(null)"):
        return []

    nodes: list[str] = []
    for match in _NODE_GROUP_RE.finditer(nodelist):
        prefix, ranges, suffix, bare = match.group(1), match.group(2), match.group(3), match.group(4)
        if bare is not None:
            nodes.append(bare)
            continue
        suffix = suffix or ""
        for piece in ranges.split(","):
            piece = piece.strip()
            if "-" in piece:
                lo_s, hi_s = piece.split("-", 1)
                # Width is the wider of the two endpoints so `[9-010]` pads to 3
                # digits, matching slurm's convention that the larger zero-pad
                # wins.
                width = max(len(lo_s), len(hi_s))
                lo, hi = int(lo_s), int(hi_s)
                for n in range(lo, hi + 1):
                    nodes.append(f"{prefix}{str(n).zfill(width)}{suffix}")
            else:
                nodes.append(f"{prefix}{piece}{suffix}")
    return nodes


def _parse_dt(value: str, cluster_tz: ZoneInfo | None = None) -> datetime | None:
    """Parse a sacct timestamp into a tz-aware datetime.

    sacct emits naive datetimes in the slurmctld host's local timezone. When
    the blueprint runs on a different host (different container, different
    region), Python's `.astimezone()` on a naive datetime anchors to the
    *blueprint host*'s local timezone, not the cluster's — which shifts every
    ExecutionContext window by the offset delta. `cluster_tz`, when provided, attaches the
    cluster's timezone before normalization so the converted datetime reflects
    the correct absolute instant regardless of where the blueprint runs.
    """
    if not value or value in ("Unknown", "None"):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        if cluster_tz is not None:
            dt = dt.replace(tzinfo=cluster_tz)
        else:
            dt = dt.astimezone()  # local-host TZ fallback
    return dt


def _parse_state(state: str) -> tuple[str, str | None]:
    # "CANCELLED by 12345" -> ("CANCELLED", "12345")
    # The trailing "+" sacct uses to mark "still cleaning up" (e.g.
    # "CANCELLED+" or "COMPLETING+") is normalized away — it carries
    # liveness signal, not classification signal, and two attribution runs
    # on the same parent_job_id should not split verdicts based on whether sacct had
    # finished its bookkeeping when the row was read.
    state = state.rstrip("+")
    if state.startswith("CANCELLED by "):
        return "CANCELLED", state.removeprefix("CANCELLED by ").strip()
    return state, None


def _parse_row(row: str, cluster_tz: ZoneInfo | None = None) -> SlurmExecutionContext | None:
    # JobName is the last sacct field and can contain `|` (user-chosen text).
    # With --parsable2 sacct does NOT escape embedded delimiters, so rejoin
    # any overflow into the trailing JobName slot rather than dropping the
    # row or truncating with fields[:14].
    fields = row.split("|")
    if len(fields) < 14:
        logger.warning(f"sacct row has {len(fields)} fields, expected at least 14: {row!r}")
        return None
    if len(fields) > 14:
        fields = fields[:13] + ["|".join(fields[13:])]

    (
        job_id,
        job_id_raw,
        sluid,
        state_raw,
        start_s,
        end_s,
        nodelist,
        failed_node,
        time_limit,
        alloc_tres,
        std_out,
        std_err,
        user_name,
        job_name,
    ) = fields[:14]

    # `<jobid>.batch` and `<jobid>.extern` are Slurm-internal wrappers (the
    # batch script container and the extern-handler step). They carry no
    # user-relevant evidence beyond what the head/numeric-step rows already
    # have, so they are filtered. Numeric step rows (`.0`, `.1`, ...) ARE
    # attributed separately — each step has its own (start_time, end_time)
    # and the user's actual srun-launched code lives there.
    if job_id.endswith(".batch") or job_id.endswith(".extern"):
        return None

    # Extract the step suffix (`.N`) if present so downstream consumers can
    # distinguish steps from the head row. Head rows have step_id == None.
    step_id: str | None = None
    if "." in job_id:
        _, _, suffix = job_id.partition(".")
        if suffix.isdigit():
            step_id = suffix
        else:
            # Unknown suffix variant — skip to avoid producing a context with
            # no meaningful semantics.
            logger.debug(f"sacct row has unknown step suffix; skipping: {job_id!r}")
            return None

    start_time = _parse_dt(start_s, cluster_tz)
    end_time = _parse_dt(end_s, cluster_tz)
    state, cancelled_by = _parse_state(state_raw)

    # RUNNING ExecutionContexts report End=Unknown; treat "now" as the
    # evidence-collection end. PENDING ExecutionContexts have no Start
    # either and remain unattributable.
    # NOTE: substituting now() makes the context window non-deterministic
    # across re-runs of the same RUNNING ExecutionContext; callers that need
    # stable verdicts should wait for the workload to terminate before
    # attributing.
    if end_time is None and state == "RUNNING" and start_time is not None:
        if cluster_tz is not None:
            end_time = datetime.now(tz=cluster_tz)
        else:
            end_time = datetime.now().astimezone()
        logger.debug(f"sacct row {job_id} is RUNNING; substituting now() for end_time")

    if start_time is None or end_time is None:
        logger.warning(
            f"sacct row has unparseable Start/End ({start_s!r}, {end_s!r}); skipping ExecutionContext {job_id}"
        )
        return None

    if not sluid:
        logger.warning(f"sacct row {job_id} has no SLUID; SLURM 25.11+ required, skipping")
        return None

    # Step rows share their head's SLUID. To keep `id` globally unique within
    # one discovery call, suffix the step id when present; downstream service
    # derives the attributor handle from `id`.
    ctx_id = f"{sluid}.{step_id}" if step_id is not None else sluid

    # parent_job_id is the user-facing jobid stripped of any step suffix.
    parent_job_id = job_id.partition(".")[0] if step_id is not None else job_id

    ctx: SlurmExecutionContext = {
        "id": ctx_id,
        "workload_type": "slurm",
        "scheduler": "slurm",
        "nodes": _expand_nodelist(nodelist),
        "start_time": start_time,
        "end_time": end_time,
        "job_id": job_id,
        "sluid": sluid,
        # Scheduler-agnostic per-attempt key consumed by the sqlite output.
        # For Slurm this is the SLUID (shared by a head row and its steps,
        # distinct per requeue) — same semantics Kueue achieves via
        # Workload.metadata.uid.
        "attempt_id": sluid,
        "parent_job_id": parent_job_id,
        "state": state,
    }
    if cluster_tz is not None:
        ctx["timezone"] = cluster_tz.key
    if step_id is not None:
        ctx["step_id"] = step_id
    if job_name:
        ctx["name"] = job_name
    if job_id_raw and job_id_raw != job_id:
        ctx["job_id_raw"] = job_id_raw
    # Decompose array-task notation so downstream consumers can refer to
    # the array ID and task ID without re-parsing the joined "1234_56"
    # string. Match against parent_job_id so array-step rows (`1234_5.0`)
    # also get decomposed — the step suffix is already stripped in
    # parent_job_id.
    array_match = _ARRAY_JOB_RE.match(parent_job_id)
    if array_match:
        ctx["array_id"] = array_match.group("array_id")
        try:
            ctx["attempt_index"] = int(array_match.group("task_id"))
        except ValueError:
            pass
    if failed_node:
        ctx["failed_node"] = failed_node
    if cancelled_by is not None:
        ctx["cancelled_by"] = cancelled_by
    if time_limit:
        try:
            ctx["time_limit"] = int(time_limit)
        except ValueError:
            pass
    if alloc_tres:
        ctx["alloc_tres"] = alloc_tres
    if user_name:
        ctx["user_name"] = user_name
    # std_out/std_err may still carry Slurm filename patterns (`%j`, `%x`,
    # `${SLURM_JOBID}`, ...) depending on the slurmctld configuration —
    # sacct returns the literal template, not always the resolved path.
    # Pre-render here so the values written to the cross-cutting
    # ExecutionContext are fully-resolved filesystem paths; the local_file
    # log connector then only has to handle generic `{X}` placeholders and
    # never needs to know about Slurm token syntax.
    if std_out:
        ctx["std_out"] = filename_patterns.substitute(std_out, ctx)
    if std_err:
        ctx["std_err"] = filename_patterns.substitute(std_err, ctx)
    return ctx


def _assign_attempt_ordinals(
    contexts: list[SlurmExecutionContext],
) -> list[SlurmExecutionContext]:
    """For jobids with multiple sacct attempts, stamp a 0-based
    `attempt_ordinal` on every row of the attempt. An "attempt" is identified
    by SLUID: head and step rows share the same SLUID and therefore the same
    ordinal. Attempts are ordered by their earliest row's start_time.

    Single-attempt jobids (one SLUID across all rows for the parent_job_id)
    are left without `attempt_ordinal` set."""
    by_parent: dict[str, dict[str, list[SlurmExecutionContext]]] = {}
    for ctx in contexts:
        parent = ctx["parent_job_id"]
        sluid = ctx["sluid"]
        by_parent.setdefault(parent, {}).setdefault(sluid, []).append(ctx)
    for sluid_groups in by_parent.values():
        if len(sluid_groups) < 2:
            continue
        ordered_sluids = sorted(
            sluid_groups,
            key=lambda s: min(c["start_time"] for c in sluid_groups[s]),
        )
        for ordinal, sluid in enumerate(ordered_sluids):
            for ctx in sluid_groups[sluid]:
                ctx["attempt_ordinal"] = ordinal
    return contexts


class SlurmExecutionContextAdapter:
    def __init__(self, sacct_path: str = "sacct", cluster_tz: str | None = None):
        self.sacct_path = sacct_path
        if cluster_tz is None:
            self.cluster_tz: ZoneInfo | None = None
        else:
            try:
                self.cluster_tz = ZoneInfo(cluster_tz)
            except ZoneInfoNotFoundError as e:
                raise ValueError(f"unknown cluster_tz: {cluster_tz!r}") from e

    def discover(self) -> list[SlurmExecutionContext]:
        """No-runtime-selector entrypoint required by the ECA protocol.

        The Slurm adapter has no defensible "all ExecutionContexts" default —
        sacct without a `--jobs` or `--starttime` filter would return everything
        since the epoch. Returning [] keeps the no-arg loopback demo path
        documented in ARCHITECTURE.md working: build a pipeline, hit the
        service, persist zero verdicts, exit cleanly. Callers that actually
        want results use `discover_by_ids` or `discover_by_time_range`.
        """
        return []

    def _run_sacct(self, extra_args: list[str]) -> list[SlurmExecutionContext]:
        # `--duplicates` keeps every attempt of a requeued parent_job_id. We drop `-X`
        # so requeue history is visible (sacct's `-X` silently collapses to
        # the latest attempt); step records (`.batch`, `.extern`, `.0`) are
        # filtered by `_parse_row`.
        cmd = [
            self.sacct_path,
            *extra_args,
            f"--format={SACCT_FORMAT}",
            "--parsable2",
            "--noheader",
            "--duplicates",
        ]
        # sacct exits non-zero when the query matches nothing (e.g. an unknown
        # jobid, or a time range with no activity). Treat that as empty rather
        # than letting CalledProcessError abort the per-context loop upstream.
        try:
            result = subprocess.run(cmd, check=True, text=True, timeout=60, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"sacct exited {e.returncode}: {(e.stderr or '').strip() or '<no stderr>'}")
            return []
        contexts: list[SlurmExecutionContext] = []
        for row in result.stdout.splitlines():
            if not row.strip():
                continue
            parsed = _parse_row(row, cluster_tz=self.cluster_tz)
            if parsed is not None:
                contexts.append(parsed)
        return _assign_attempt_ordinals(contexts)

    def _format_sacct_time(self, dt: datetime) -> str:
        """sacct accepts `YYYY-MM-DD[THH:MM[:SS]]` only — no timezone offset.
        Convert tz-aware datetimes into the slurmctld's local TZ (cluster_tz
        if configured, else the blueprint host's local TZ) before formatting,
        so the absolute instant is preserved."""
        if dt.tzinfo is not None:
            dt = dt.astimezone(self.cluster_tz) if self.cluster_tz is not None else dt.astimezone()
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def discover_by_ids(self, ids: list[str]) -> list[SlurmExecutionContext]:
        return self._run_sacct([f"--jobs={','.join(ids)}"])

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[SlurmExecutionContext]:
        contexts = self._run_sacct(
            [f"--starttime={self._format_sacct_time(start)}", f"--endtime={self._format_sacct_time(end)}"]
        )
        if not nodelist:
            return contexts
        # Each element may be either an already-expanded hostname or a Slurm
        # range expression like "node[001-004,010]". Expand both into a single
        # set so the CLI doesn't need to know Slurm's syntax.
        wanted: set[str] = set()
        for item in nodelist:
            wanted.update(_expand_nodelist(item))
        return [ctx for ctx in contexts if wanted.intersection(ctx["nodes"])]
