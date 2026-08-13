# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import subprocess
from datetime import datetime
from pathlib import Path

from attribution_blueprint.connectors.execution_contexts.slurm import (
    SlurmExecutionContextAdapter,
)
from attribution_blueprint.connectors.execution_contexts.slurm.connector import _expand_nodelist

FIXTURE = Path(__file__).parent.parent.parent.parent / "fixtures" / "sacct_output.txt"


def _mock_sacct(mocker, stdout: str):
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    return mocker.patch("subprocess.run", return_value=completed)


def test_parses_fixture(mocker):
    _mock_sacct(mocker, FIXTURE.read_text())
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["1001", "1002", "1003", "1004", "2001"])

    # 4 single-attempt parent_job_ids + 2001 attempt 1 (head + .0) + 2001
    # attempt 2 (head + .0 + .1) = 9. `.batch` and `.extern` rows are filtered.
    assert len(contexts) == 9

    c0 = contexts[0]
    assert c0["job_id"] == "1001"
    assert c0["workload_type"] == "slurm"
    assert c0["scheduler"] == "slurm"
    assert c0["sluid"] == "sluid-aaaa-1001"
    assert c0["id"] == "sluid-aaaa-1001"
    assert c0["nodes"] == ["node005"]
    assert c0["state"] == "COMPLETED"
    assert c0["start_time"] == datetime(2026, 5, 1, 12, 0, 0).astimezone()
    assert c0["end_time"] == datetime(2026, 5, 1, 12, 30, 0).astimezone()
    assert c0["start_time"].tzinfo is not None
    assert c0["time_limit"] == 60
    assert c0["alloc_tres"] == "cpu=8,mem=32G,gres/gpu=1"
    assert c0["std_out"] == "/home/u/out-1001.log"
    assert c0["std_err"] == "/home/u/err-1001.log"
    assert "failed_node" not in c0
    assert "cancelled_by" not in c0
    assert "attempt_ordinal" not in c0  # single-attempt parent_job_ids are unstamped

    c1 = contexts[1]
    assert c1["nodes"] == ["node001", "node002", "node003"]
    assert c1["failed_node"] == "node002"
    assert c1["state"] == "FAILED"

    c2 = contexts[2]
    assert c2["state"] == "CANCELLED"
    assert c2["cancelled_by"] == "54321"
    assert c2["nodes"] == ["node001", "node002", "other5", "other6"]


def test_slurm_pre_2511_row_is_skipped(mocker):
    # SLURM 25.11+ is required; rows without SLUID are skipped with a warning
    # rather than fabricating an `id` from job_id.
    row = "9100|9100||COMPLETED|2026-05-01T15:00:00|2026-05-01T15:30:00|node[001,003,005]||60|cpu=8|/o|/e|alice|train\n"
    _mock_sacct(mocker, row)
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["9100"])
    assert contexts == []


def test_sluid_present(mocker):
    _mock_sacct(mocker, FIXTURE.read_text())
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["1001"])
    assert contexts[0]["id"] == "sluid-aaaa-1001"
    assert contexts[0]["id"] == contexts[0]["sluid"]


def test_nodelist_expansion_range():
    assert _expand_nodelist("node[001-003]") == ["node001", "node002", "node003"]


def test_nodelist_expansion_csv():
    assert _expand_nodelist("node[001,003,005]") == ["node001", "node003", "node005"]


def test_nodelist_expansion_mixed_groups():
    assert _expand_nodelist("node[001-002],other[5-6]") == ["node001", "node002", "other5", "other6"]


def test_nodelist_expansion_bare():
    assert _expand_nodelist("node005") == ["node005"]


def test_nodelist_expansion_empty():
    assert _expand_nodelist("") == []
    assert _expand_nodelist("None assigned") == []


def test_nodelist_expansion_combined_range_and_csv():
    assert _expand_nodelist("node[001-002,005]") == ["node001", "node002", "node005"]


def test_discover_by_ids_passes_jobs_arg(mocker):
    run_mock = _mock_sacct(mocker, "")
    SlurmExecutionContextAdapter().discover_by_ids(["1", "5"])
    cmd = run_mock.call_args.args[0]
    assert "--jobs=1,5" in cmd
    # --duplicates keeps requeue history; -X would silently collapse to the
    # latest attempt and is intentionally absent.
    assert "--duplicates" in cmd
    assert "-X" not in cmd
    assert "--parsable2" in cmd
    assert "--noheader" in cmd


def test_requeue_attempts_emit_separate_contexts(mocker):
    _mock_sacct(mocker, FIXTURE.read_text())
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["2001"])

    # Mock returns the whole fixture; restrict to the requeued parent_job_id.
    ctxs2001 = [c for c in contexts if c["parent_job_id"] == "2001"]
    heads = [c for c in ctxs2001 if "step_id" not in c]
    assert len(heads) == 2, "expected two head rows for requeued parent_job_id 2001"

    by_ord = {c["attempt_ordinal"]: c for c in heads}
    assert set(by_ord) == {0, 1}

    first = by_ord[0]
    assert first["sluid"] == "sluid-rq-2001a"
    assert first["state"] == "NODE_FAIL"
    assert first["failed_node"] == "node011"
    assert first["nodes"] == ["node010", "node011", "node012"]
    assert first["start_time"] == datetime(2026, 5, 2, 8, 0, 0).astimezone()

    second = by_ord[1]
    assert second["sluid"] == "sluid-rq-2001b"
    assert second["state"] == "TIMEOUT"
    assert "failed_node" not in second
    assert second["nodes"] == ["node020", "node021", "node022"]
    assert second["start_time"] == datetime(2026, 5, 2, 9, 35, 0).astimezone()


def test_batch_and_extern_rows_filtered_but_numeric_steps_kept(mocker):
    # `.batch` and `.extern` are Slurm-internal wrappers; only `.0`, `.1`, ...
    # rows survive as their own ExecutionContext.
    _mock_sacct(mocker, FIXTURE.read_text())
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["2001"])
    job_ids = [c["job_id"] for c in contexts]
    assert "2001.batch" not in job_ids
    assert "2001.extern" not in job_ids
    assert "2001.0" in job_ids
    assert "2001.1" in job_ids


def test_step_rows_carry_step_id_and_share_parent_attempt_ordinal(mocker):
    _mock_sacct(mocker, FIXTURE.read_text())
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["2001"])

    # 2001 attempt 1 (sluid-rq-2001a): head + .0
    a1 = [c for c in contexts if c.get("sluid") == "sluid-rq-2001a"]
    assert len(a1) == 2
    assert all(c["attempt_ordinal"] == 0 for c in a1)
    a1_head = next(c for c in a1 if "step_id" not in c)
    a1_step0 = next(c for c in a1 if c.get("step_id") == "0")
    # Step uses its own (start_time, end_time) from the .0 row.
    assert a1_step0["start_time"] == datetime(2026, 5, 2, 8, 0, 30).astimezone()
    assert a1_step0["job_id"] == "2001.0"
    assert a1_step0["parent_job_id"] == "2001"
    # `id` is SLUID-rooted but disambiguated by step suffix.
    assert a1_head["id"] == "sluid-rq-2001a"
    assert a1_step0["id"] == "sluid-rq-2001a.0"

    # 2001 attempt 2 (sluid-rq-2001b): head + .0 + .1
    a2 = [c for c in contexts if c.get("sluid") == "sluid-rq-2001b"]
    assert len(a2) == 3
    assert all(c["attempt_ordinal"] == 1 for c in a2)
    step_ids = sorted(c.get("step_id") for c in a2 if c.get("step_id"))
    assert step_ids == ["0", "1"]
    a2_step1 = next(c for c in a2 if c.get("step_id") == "1")
    # Step .1 starts much later than its head, exercising distinct windows.
    assert a2_step1["start_time"] == datetime(2026, 5, 2, 15, 0, 0).astimezone()
    assert a2_step1["id"] == "sluid-rq-2001b.1"


def test_discover_by_time_range_filters_by_nodelist(mocker):
    _mock_sacct(mocker, FIXTURE.read_text())
    start = datetime(2026, 5, 1, 0, 0, 0)
    end = datetime(2026, 5, 2, 0, 0, 0)
    contexts = SlurmExecutionContextAdapter().discover_by_time_range(start, end, nodelist=["node001"])

    job_ids = {c["job_id"] for c in contexts}
    assert job_ids == {"1002", "1003", "1004"}


def test_discover_by_time_range_no_filter_returns_all(mocker):
    _mock_sacct(mocker, FIXTURE.read_text())
    start = datetime(2026, 5, 1, 0, 0, 0)
    end = datetime(2026, 5, 2, 0, 0, 0)
    contexts = SlurmExecutionContextAdapter().discover_by_time_range(start, end)
    # 4 single-attempt parent_job_ids + 2001 attempt 1 (head + .0) +
    # attempt 2 (head + .0 + .1) = 9.
    assert len(contexts) == 9


def test_discover_by_time_range_passes_time_args(mocker):
    """sacct does NOT accept a timezone offset in --starttime/--endtime, so
    tz-aware inputs must be converted into the slurmctld's local TZ and
    serialized without the offset suffix. With `cluster_tz="UTC"` and a UTC
    input, the times pass through verbatim sans offset."""
    from datetime import timezone

    run_mock = _mock_sacct(mocker, "")
    start = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 1, 13, 0, 0, tzinfo=timezone.utc)
    SlurmExecutionContextAdapter(cluster_tz="UTC").discover_by_time_range(start, end)
    cmd = run_mock.call_args.args[0]
    assert "--starttime=2026-05-01T12:00:00" in cmd
    assert "--endtime=2026-05-01T13:00:00" in cmd
    # sacct rejects offsets - make sure none leaked through.
    assert not any("+00:00" in arg or "+0000" in arg for arg in cmd)


def test_discover_by_time_range_converts_to_cluster_tz(mocker):
    """When cluster_tz differs from the input's tz, the formatted sacct time
    must be the local equivalent of the same absolute instant."""
    from datetime import timezone

    run_mock = _mock_sacct(mocker, "")
    # 12:00 UTC == 08:00 America/New_York during DST (EDT, UTC-4).
    start = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 1, 13, 0, 0, tzinfo=timezone.utc)
    SlurmExecutionContextAdapter(cluster_tz="America/New_York").discover_by_time_range(start, end)
    cmd = run_mock.call_args.args[0]
    assert "--starttime=2026-05-01T08:00:00" in cmd
    assert "--endtime=2026-05-01T09:00:00" in cmd


def test_sacct_nonzero_exit_returns_empty(mocker):
    """sacct exits 1 for an unknown jobid; that must come back as [] rather
    than propagating CalledProcessError up out of the per-context loop."""
    err = subprocess.CalledProcessError(returncode=1, cmd=[], stderr="No matching ExecutionContexts")
    mocker.patch("subprocess.run", side_effect=err)
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["does-not-exist"])
    assert contexts == []


def test_nodelist_expansion_fqdn_suffix():
    """FQDN nodelists like `node[001-004].mgmt.example.com` must preserve the
    suffix on every expanded node, not lose it to the bare-name branch."""
    expanded = _expand_nodelist("node[001-004].mgmt.example.com")
    assert expanded == [
        "node001.mgmt.example.com",
        "node002.mgmt.example.com",
        "node003.mgmt.example.com",
        "node004.mgmt.example.com",
    ]


def test_nodelist_expansion_mixed_width_range():
    """`[9-010]` zero-pads to the wider of the two endpoints."""
    expanded = _expand_nodelist("node[9-010]")
    assert expanded == [f"node{n:03d}" for n in range(9, 11)]


def test_unknown_start_end_skipped_with_warning(mocker, caplog):
    bad_row = "9999|sluid-xxxx-9999|FAILED|Unknown|Unknown|node007||60|cpu=8|/o|/e\n"
    _mock_sacct(mocker, FIXTURE.read_text() + bad_row)
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["1001", "1002", "1003", "1004", "2001", "9999"])
    # 4 single-attempt parent_job_ids + 2001 head/step/step rows; bad row 9999 dropped.
    assert len(contexts) == 9
    assert {c["parent_job_id"] for c in contexts} == {"1001", "1002", "1003", "1004", "2001"}


def test_cluster_tz_anchors_naive_sacct_timestamps(mocker):
    """cluster_tz attaches the configured tz to naive sacct datetimes before normalization."""
    from zoneinfo import ZoneInfo

    row = (
        "5001|5001|sluid-tz-5001|COMPLETED|2026-05-01T12:00:00|2026-05-01T13:00:00|"
        "node100||60|cpu=8|/o|/e|alice|train\n"
    )
    _mock_sacct(mocker, row)
    contexts = SlurmExecutionContextAdapter(cluster_tz="America/Los_Angeles").discover_by_ids(["5001"])
    assert len(contexts) == 1
    ctx = contexts[0]
    # The naive sacct timestamp "12:00:00" is now anchored to LA tz, NOT the
    # test host's tz. Verify by checking the tzinfo on the returned datetime.
    assert ctx["start_time"].tzinfo == ZoneInfo("America/Los_Angeles")
    assert ctx["start_time"].hour == 12  # not shifted by host-tz conversion
    assert ctx["timezone"] == "America/Los_Angeles"


def test_cluster_tz_invalid_zone_raises():
    """An unknown IANA name fails fast at construction, not at first discover_by_ids."""
    import pytest

    with pytest.raises(ValueError, match="unknown cluster_tz"):
        SlurmExecutionContextAdapter(cluster_tz="Not/A/Zone")


def test_jobname_with_embedded_pipe_preserved(mocker):
    """sacct --parsable2 does not escape | inside JobName; the connector must rejoin."""
    # 15 fields: JobName is "etl|stage1|finalize" (3 pieces) split into 3 columns.
    row = (
        "6001|6001|sluid-pn-6001|COMPLETED|2026-05-01T12:00:00|2026-05-01T13:00:00|"
        "node200||60|cpu=8|/o|/e|bob|etl|stage1|finalize\n"
    )
    _mock_sacct(mocker, row)
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["6001"])
    assert len(contexts) == 1
    # JobName overflow must be reassembled, not truncated.
    assert contexts[0]["name"] == "etl|stage1|finalize"
    # And the parsed identity fields must be unaffected.
    assert contexts[0]["job_id"] == "6001"
    assert contexts[0]["user_name"] == "bob"


def test_cancelled_plus_state_normalized(mocker):
    """sacct's transient `+` suffix on a CANCELLED state must not split downstream verdicts."""
    row = (
        "7001|7001|sluid-cx-7001|CANCELLED+|2026-05-01T12:00:00|2026-05-01T13:00:00|"
        "node300||60|cpu=8|/o|/e|carol|short\n"
    )
    _mock_sacct(mocker, row)
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["7001"])
    assert contexts[0]["state"] == "CANCELLED"


def test_cancelled_by_state_normalized(mocker):
    """`CANCELLED by 9001` should yield state=CANCELLED, cancelled_by=9001 (independent of the
    `+` suffix path) so we lock down both branches of _parse_state."""
    row = (
        "7002|7002|sluid-cby-7002|CANCELLED by 9001|2026-05-01T12:00:00|2026-05-01T13:00:00|"
        "node300||60|cpu=8|/o|/e|carol|short\n"
    )
    _mock_sacct(mocker, row)
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["7002"])
    assert contexts[0]["state"] == "CANCELLED"
    assert contexts[0]["cancelled_by"] == "9001"


def test_array_step_row_decomposes_array_id_and_attempt_index(mocker):
    """An array step row (`<array>_<task>.<step>`) must decompose array_id and attempt_index."""
    # Head row for the array task, then its numeric step.
    head = (
        "8000_5|8123|sluid-arr-8000-5|COMPLETED|2026-05-01T12:00:00|2026-05-01T12:30:00|"
        "node400||60|cpu=8|/o|/e|dora|train-array\n"
    )
    step = (
        "8000_5.0|8123|sluid-arr-8000-5|COMPLETED|2026-05-01T12:01:00|2026-05-01T12:29:00|"
        "node400||60|cpu=8|/o|/e|dora|train-array\n"
    )
    _mock_sacct(mocker, head + step)
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["8000_5"])
    assert len(contexts) == 2
    step_ctx = next(c for c in contexts if c.get("step_id") == "0")
    # The bug being regressed: array decomposition used to fail on step rows
    # because _ARRAY_JOB_RE was matched against `job_id` (which has the `.0`
    # suffix), not `parent_job_id`. Now both head and step carry array_id /
    # attempt_index, so {std_out} templates with %A/%a render correctly on
    # array step rows.
    assert step_ctx["array_id"] == "8000"
    assert step_ctx["attempt_index"] == 5


def test_std_out_path_with_slurm_tokens_is_resolved_at_ctx_boundary(mocker):
    """The Slurm ECA pre-renders `%j`/`%x`/`${SLURM_*}` tokens in
    `StdOut`/`StdErr` BEFORE writing them to the cross-cutting
    ExecutionContext. Downstream consumers (local_file log connector) see
    fully-resolved filesystem paths and never need to parse Slurm-shaped
    token syntax. Verifies architectural rule 3: Slurm-specific parsing
    lives inside the Slurm connector, not in cross-cutting code."""
    # sacct row with StdOut/StdErr templates that carry Slurm tokens.
    # 14-field --parsable2 row: JobID|JobIDRaw|SLUID|State|Start|End|NodeList|
    # FailedNode|TimelimitRaw|AllocTRES|StdOut|StdErr|User|JobName
    row = (
        "9100|9100|sluid-9100|COMPLETED|2026-05-01T12:00:00|2026-05-01T12:30:00"
        "|node005||60|cpu=8,mem=32G|"
        "/path/to/logs/%u/%x/%j.out|/path/to/logs/${SLURM_JOB_NAME}/${SLURM_JOBID}.err|"
        "alice|train-mnist"
    )
    _mock_sacct(mocker, row)
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["9100"])
    assert len(contexts) == 1
    ctx = contexts[0]
    # %j → JobIDRaw, %x → JobName, %u → User
    assert ctx["std_out"] == "/path/to/logs/alice/train-mnist/9100.out"
    # ${SLURM_JOB_NAME} → name, ${SLURM_JOBID} → JobIDRaw
    assert ctx["std_err"] == "/path/to/logs/train-mnist/9100.err"


def test_std_out_with_array_tokens_resolves_at_ctx_boundary(mocker):
    """Array-task tokens %A (array_id) and %a (attempt_index) also
    pre-resolve at the Slurm ECA boundary, using the decomposed
    `array_id`/`attempt_index` fields rather than the joined `1234_5`
    string."""
    row = (
        "8200_3|8201|sluid-8201|COMPLETED|2026-05-01T12:00:00|2026-05-01T12:30:00"
        "|node005||60|cpu=8|"
        "/path/to/logs/%A_%a.out|/path/to/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err|"
        "alice|x"
    )
    _mock_sacct(mocker, row)
    contexts = SlurmExecutionContextAdapter().discover_by_ids(["8200_3"])
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["array_id"] == "8200"
    assert ctx["attempt_index"] == 3
    assert ctx["std_out"] == "/path/to/logs/8200_3.out"
    assert ctx["std_err"] == "/path/to/logs/8200_3.err"
