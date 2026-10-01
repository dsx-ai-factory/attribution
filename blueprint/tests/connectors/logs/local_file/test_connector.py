# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from datetime import datetime, timezone

import pytest

from attribution_blueprint.connectors.logs.local_file import LocalFileLogConnector, parse_size
from attribution_blueprint.types import Source


def _text(observation: dict) -> str:
    """Rejoin an APPLOG observation's per-line entries into the file text."""
    return "\n".join(entry["body"] for entry in json.loads(observation["body"]))


def _ctx(tmp_path, std_out_template: str, **extra) -> dict:
    return {
        "id": "sluid-1",
        "job_id": "12345",
        "nodes": ["node-a"],
        "start_time": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
        "std_out": std_out_template,
        **extra,
    }


def test_returns_empty_when_file_missing(tmp_path):
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"])
    obs = conn.collect(_ctx(tmp_path, str(tmp_path / "absent.log")))
    assert obs == []


def test_reads_existing_file_and_emits_one_observation(tmp_path):
    log = tmp_path / "12345.log"
    log.write_text("line one\nline two\nline three\n")
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"])

    obs = conn.collect(_ctx(tmp_path, str(log)))

    assert len(obs) == 1
    o = obs[0]
    assert o["context"]["source"] == Source.APPLOG
    assert o["context"]["time_interval"] == {"start": None, "end": None}
    assert json.loads(o["body"]) == [{"body": "line one"}, {"body": "line two"}, {"body": "line three"}, {"body": ""}]


def test_substitutes_job_id_placeholder(tmp_path):
    """`{job_id}` is the scheduler-agnostic placeholder in user-supplied
    templates. The previous Slurm-shaped `%j` token is no longer parsed
    here - that substitution moved into the Slurm ECA, which pre-renders
    `std_out`/`std_err` before they reach this connector. Users who
    previously wrote `%j` in TOML `paths` should migrate to `{job_id}`."""
    log = tmp_path / "12345" / "job.out"
    log.parent.mkdir()
    log.write_text("body")
    template = str(tmp_path / "{job_id}" / "job.out")
    conn = LocalFileLogConnector(log_path_templates=[template])

    obs = conn.collect(_ctx(tmp_path, std_out_template=""))

    assert len(obs) == 1
    assert _text(obs[0]) == "body"


def test_max_bytes_reads_tail(tmp_path):
    log = tmp_path / "big.log"
    head = "HEAD-content-that-should-be-dropped"
    tail = "TAIL-content-that-must-survive"
    filler = "x" * (1000 - len(head) - len(tail))
    log.write_text(head + filler + tail)
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"], max_bytes_per_file=100)

    obs = conn.collect(_ctx(tmp_path, str(log)))
    body = _text(obs[0])
    assert len(body) == 100
    assert body.endswith(tail)
    assert head not in body


def test_max_bytes_smaller_than_file_reads_all(tmp_path):
    log = tmp_path / "small.log"
    log.write_text("short content")
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"], max_bytes_per_file=1024)

    obs = conn.collect(_ctx(tmp_path, str(log)))
    assert _text(obs[0]) == "short content"


def test_empty_template_returns_empty(tmp_path):
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"])
    obs = conn.collect(_ctx(tmp_path, std_out_template=""))
    assert obs == []


def test_decode_errors_replaced(tmp_path):
    log = tmp_path / "binary.log"
    log.write_bytes(b"valid \xff\xfe binary trailing")
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"])
    obs = conn.collect(_ctx(tmp_path, str(log)))
    assert len(obs) == 1
    assert "valid" in _text(obs[0])


def test_source_is_applog_by_default():
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"])
    assert conn.source == Source.APPLOG


@pytest.mark.parametrize(
    "value,expected",
    [
        (1024, 1024),
        ("1024", 1024),
        ("1024B", 1024),
        ("1k", 1024),
        ("1kb", 1024),
        ("500MB", 500 * 1024**2),
        ("500mb", 500 * 1024**2),
        ("2G", 2 * 1024**3),
        ("2GB", 2 * 1024**3),
        ("1.5gb", int(1.5 * 1024**3)),
        ("1T", 1024**4),
    ],
)
def test_parse_size_accepts_human_readable(value, expected):
    assert parse_size(value) == expected


def test_parse_size_rejects_garbage():
    with pytest.raises(ValueError, match="unparseable"):
        parse_size("not a size")
    with pytest.raises(ValueError, match="unknown size suffix"):
        parse_size("100xyz")


def test_max_bytes_string_form_reads_tail(tmp_path):
    log = tmp_path / "big.log"
    log.write_text("x" * 900 + "FINAL")
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"], max_bytes_per_file="100B")
    obs = conn.collect(_ctx(tmp_path, str(log)))
    assert len(_text(obs[0])) == 100
    assert _text(obs[0]).endswith("FINAL")


def test_default_cap_full_read_when_below_cap(tmp_path):
    """Files smaller than DEFAULT_MAX_BYTES are read in full under the default cap."""
    log = tmp_path / "big.log"
    log.write_text("a" * (2 * 1024 * 1024))  # 2 MiB, well under 2 GB default cap
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"])

    obs = conn.collect(_ctx(tmp_path, str(log)))
    assert len(obs) == 1
    assert len(_text(obs[0])) == 2 * 1024 * 1024


def test_default_cap_is_2gb():
    """DEFAULT_MAX_BYTES is the constructor default and is 2 GB."""
    from attribution_blueprint.connectors.logs.local_file import DEFAULT_MAX_BYTES

    assert DEFAULT_MAX_BYTES == 2 * 1024 * 1024 * 1024
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"])
    assert conn._max_bytes == DEFAULT_MAX_BYTES


def test_explicit_none_disables_cap(tmp_path):
    """Passing max_bytes_per_file=None disables the cap entirely."""
    log = tmp_path / "big.log"
    log.write_text("z" * 1000)
    conn = LocalFileLogConnector(log_path_templates=["{std_out}"], max_bytes_per_file=None)
    assert conn._max_bytes is None
    obs = conn.collect(_ctx(tmp_path, str(log)))
    assert len(_text(obs[0])) == 1000


def test_directory_template_expands_to_all_files(tmp_path):
    """A template that resolves to a directory yields one Observation per file inside."""
    d = tmp_path / "12345"
    d.mkdir()
    (d / "node_0.err").write_text("rank0 error")
    (d / "node_1.err").write_text("rank1 error")
    (d / "node_2.err").write_text("rank2 error")
    conn = LocalFileLogConnector(log_path_templates=[str(d)])

    obs = conn.collect(_ctx(tmp_path, std_out_template=""))
    assert len(obs) == 3
    bodies = sorted(_text(o) for o in obs)
    assert bodies == ["rank0 error", "rank1 error", "rank2 error"]


def test_syslog_directory_expands_context_node_directories(tmp_path):
    """Local SYSLOG reads one directory per context node."""
    from loguru import logger

    base = tmp_path / "syslog"
    node_a = base / "node-a"
    node_b = base / "node-b"
    node_a.mkdir(parents=True)
    node_b.mkdir()
    (node_a / "kernel.log").write_text("2026-05-01T00:00:03Z node-a kernel\nnode-a nvrm\n")
    (node_b / "messages.log").write_text("May  1 00:00:04 node-b kernel: oops\n")
    (node_b / "undated.log").write_text("node-b kernel\n")
    conn = LocalFileLogConnector(log_path_templates=[str(base)], source=Source.SYSLOG)
    ctx = _ctx(tmp_path, std_out_template="", nodes=["node-a", "node-b", "node-c"])

    warnings: list[str] = []
    sink_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        obs = conn.collect(ctx)
    finally:
        logger.remove(sink_id)

    assert len(obs) == 3
    assert all(observation["context"]["source"] == Source.SYSLOG for observation in obs)
    assert all(
        observation["context"]["time_interval"] == {"start": ctx["start_time"], "end": ctx["end_time"]}
        for observation in obs
    )
    assert sorted(entry["body"] for observation in obs for entry in json.loads(observation["body"])) == [
        "2026-05-01T00:00:03Z node-a kernel",
        "May  1 00:00:04 node-b kernel: oops",
    ]
    assert sorted(
        entry["resources"][0]["hostname"] for observation in obs for entry in json.loads(observation["body"])
    ) == [
        "node-a",
        "node-b",
    ]
    missing_warnings = [
        warning
        for warning in warnings
        if "missing syslog directory" in warning and "node-c" in warning and str(base / "node-c") in warning
    ]
    assert missing_warnings == []


def test_syslog_warns_when_no_context_node_directories_exist(tmp_path):
    """Local SYSLOG warns when a context has no matching node directories."""
    from loguru import logger

    base = tmp_path / "syslog"
    base.mkdir()
    conn = LocalFileLogConnector(log_path_templates=[str(base)], source=Source.SYSLOG)
    ctx = _ctx(tmp_path, std_out_template="", nodes=["node-a", "node-b"])

    warnings: list[str] = []
    sink_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        obs = conn.collect(ctx)
    finally:
        logger.remove(sink_id)

    assert obs == []
    matching_warnings = [warning for warning in warnings if "no syslog directories found" in warning]
    assert len(matching_warnings) == 1


def test_multiple_paths_emit_multiple_observations(tmp_path):
    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"
    log_a.write_text("alpha")
    log_b.write_text("beta")
    conn = LocalFileLogConnector(log_path_templates=[str(log_a), str(log_b)])

    obs = conn.collect(_ctx(tmp_path, std_out_template=""))
    assert sorted(_text(o) for o in obs) == ["alpha", "beta"]


def test_duplicate_paths_are_deduped(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("once")
    conn = LocalFileLogConnector(log_path_templates=[str(log), str(log)])

    obs = conn.collect(_ctx(tmp_path, std_out_template=""))
    assert len(obs) == 1


def test_empty_templates_list_rejected():
    with pytest.raises(ValueError, match="at least one path template"):
        LocalFileLogConnector(log_path_templates=[])


def test_permission_denied_read_logs_warning_and_skips(tmp_path, mocker):
    """A file that exists but can't be read (EACCES) must NOT propagate and NOT emit an
    Observation - the behavior the eval harness depends on for excluding perm-denied attempts."""
    from loguru import logger

    log = tmp_path / "x.log"
    log.write_text("secret")
    # is_file() succeeds; the read raises PermissionError.
    mocker.patch.object(LocalFileLogConnector, "_read", side_effect=PermissionError(13, "denied"))
    conn = LocalFileLogConnector(log_path_templates=[str(log)])

    # Capture loguru warnings into a list via a temporary sink (loguru's stderr
    # sink bypasses pytest capsys because it binds to the import-time sys.stderr).
    warnings: list[str] = []
    sink_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        obs = conn.collect(_ctx(tmp_path, std_out_template=""))
    finally:
        logger.remove(sink_id)

    assert obs == []  # exception did not propagate; the context still attributes
    assert any("permission denied" in w.lower() for w in warnings)


def test_std_err_template_reads_stderr_file(tmp_path):
    """When `{std_err}` is in paths, the connector reads the StdErr file."""
    err_log = tmp_path / "12345.err"
    err_log.write_text("stderr body")
    conn = LocalFileLogConnector(log_path_templates=["{std_err}"])
    ctx = _ctx(tmp_path, std_out_template="", std_err=str(err_log))
    obs = conn.collect(ctx)
    assert len(obs) == 1
    assert _text(obs[0]) == "stderr body"


def test_std_out_and_std_err_emit_two_observations_when_distinct(tmp_path):
    """When sacct gives distinct StdOut and StdErr paths, both are read."""
    out_log = tmp_path / "12345.out"
    err_log = tmp_path / "12345.err"
    out_log.write_text("stdout body")
    err_log.write_text("stderr body")
    conn = LocalFileLogConnector(log_path_templates=["{std_out}", "{std_err}"])
    ctx = _ctx(tmp_path, std_out_template=str(out_log), std_err=str(err_log))
    obs = conn.collect(ctx)
    bodies = sorted(_text(o) for o in obs)
    assert bodies == ["stderr body", "stdout body"]


def test_std_err_empty_skipped_std_out_still_read(tmp_path):
    """Slurm default: --error unset → sacct StdErr empty → only StdOut is read.

    Asserts no exception and no spurious observation when {std_err} resolves to "".
    """
    out_log = tmp_path / "12345.out"
    out_log.write_text("stdout body")
    conn = LocalFileLogConnector(log_path_templates=["{std_out}", "{std_err}"])
    ctx = _ctx(tmp_path, std_out_template=str(out_log))  # no std_err key
    obs = conn.collect(ctx)
    assert len(obs) == 1
    assert _text(obs[0]) == "stdout body"


def test_std_err_same_as_std_out_deduped(tmp_path):
    """When user sets `#SBATCH --error` to the StdOut path, sacct returns the same
    path for both. The connector's path-dedup must collapse to a single read."""
    log = tmp_path / "12345.out"
    log.write_text("once")
    conn = LocalFileLogConnector(log_path_templates=["{std_out}", "{std_err}"])
    ctx = _ctx(tmp_path, std_out_template=str(log), std_err=str(log))
    obs = conn.collect(ctx)
    assert len(obs) == 1


def test_job_id_substitution_uses_cross_cutting_job_id(tmp_path):
    expected = tmp_path / "2001.0.log"
    expected.write_text("step content")
    conn = LocalFileLogConnector(log_path_templates=[str(tmp_path / "{job_id}.log")])

    ctx = _ctx(
        tmp_path,
        std_out_template="",
        id="0173B95DA9E7F1E.0",
        job_id="2001.0",
    )
    obs = conn.collect(ctx)
    assert len(obs) == 1
    assert _text(obs[0]) == "step content"


def test_scheduler_specific_placeholders_are_not_part_of_local_file_contract(tmp_path):
    from datetime import datetime, timezone

    log_file = tmp_path / "job-1234_56.out"
    log_file.write_text("hello\n")
    template = str(tmp_path / "job-{array_id}_{attempt_index}.out")
    conn = LocalFileLogConnector(log_path_templates=[template])
    ctx = {
        "id": "ctx-1",
        "nodes": ["n1"],
        "start_time": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
    }
    assert conn.collect(ctx) == []


def test_name_and_user_name_placeholders_substitute(tmp_path):
    """`{name}` and `{user_name}` are scheduler-agnostic placeholders for
    the workload's display name and owner - used to be Slurm's `%x` and
    `%u` tokens, which the Slurm ECA now pre-renders into `std_out`. Any
    user-supplied template referencing these uses generic placeholders."""
    from datetime import datetime, timezone

    log_file = tmp_path / "alice-train-mnist.out"
    log_file.write_text("from alice\n")
    template = str(tmp_path / "{user_name}-{name}.out")
    conn = LocalFileLogConnector(log_path_templates=[template])
    ctx = {
        "id": "ctx-1",
        "nodes": ["n1"],
        "start_time": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
        "name": "train-mnist",
        "user_name": "alice",
    }
    obs = conn.collect(ctx)
    assert len(obs) == 1
    assert _text(obs[0]) == "from alice\n"
