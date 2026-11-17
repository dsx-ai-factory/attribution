# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep JIT compiler evidence without assuming the cause of compilation failure."""

import pytest

from attribution_lib import core
from attribution_lib.bundled_rules.tests._shared import default_job_info
from attribution_lib.decision_trace import build_decision_graph
from attribution_lib.decision_trace_recorder import DecisionRecorder, NodeType
from attribution_lib.job_info import NVRX_TAG
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.resources import AllJobResources
from test_support.helpers import line_entries, make_applog_entry, make_syslog_entry


_SYMPTOM = "Triton/Inductor JIT C-compilation failure"
_RULE = "TritonInductorCCompilationFailure"
_TIME_NS = 1704103260000000000
_ERROR = (
    "torch._inductor.exc.InductorError: CalledProcessError: Command "
    "'['/usr/bin/gcc', '/tmp/jit/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', "
    "'-o', '/tmp/jit/cuda_utils.cpython-312-aarch64-linux-gnu.so', '-l:libcuda.so.1', "
    "'-L/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/lib', "
    "'-L/usr/lib/aarch64-linux-gnu', '-L/usr/local/cuda/compat/lib', "
    "'-I/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/include', "
    "'-I/tmp/jit', '-I/usr/include/python3.12', '-I/usr/local/cuda/include']' "
    "returned non-zero exit status 1."
)
# Preserve the observed traceback and rank-prefix shape, using synthetic ranks
# and application/cache paths. Only the terminal exception proves compilation failed.
_TRACEBACK = [
    " 1: [rank1]: Traceback (most recent call last):\n",
    ' 1: [rank1]:   File "/app/train.py", line 20, in warmup\n',
    " 1: [rank1]:     raise e.remove_dynamo_frames() from None\n",
    " 1: [rank1]:     raise InductorError(e, currentframe()).with_traceback(\n",
    ' 1: [rank1]:   File "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py", '
    "line 66, in __init__\n",
    ' 1: [rank1]:   File "/usr/local/lib/python3.12/dist-packages/triton/runtime/build.py", '
    "line 93, in compile_module_from_src\n",
    ' 1: [rank1]:   File "/usr/local/lib/python3.12/dist-packages/triton/runtime/build.py", line 48, in _build\n',
    ' 1: [rank1]:   File "/usr/lib/python3.12/subprocess.py", line 413, in check_call\n',
    " 1: [rank1]:     raise CalledProcessError(retcode, cmd)\n",
    f" 1: [rank1]: {_ERROR}\n",
]
_NVRX = "0: Cycle: 1 Event: failure_detected Node: node01 Rank: 0 Time: 2024-01-01 10:01:00.000 UTC"
_PEER_CLOSED = "0: NCCL WARN Connection closed by remote peer node02<1234>"
_EXT4 = (
    "EXT4-fs (md0): Delayed block allocation failed for inode 1234 at logical offset 0 with max blocks 1 with error 117"
)


@pytest.fixture
def attribute_logs(attribution_test_builder, obs_data):
    def _attribute(entries, *, syslog=(), **job_options):
        options = {"nodes": ["node01", "node02"], "tags": frozenset({NVRX_TAG}), "exit_code_signal": None}
        options.update(job_options)
        job = default_job_info(**options)
        extracted = attribution_test_builder(
            job_info=job, data=entries, source=ObservationSource.APPLOG, insert_tflops_line=False
        )
        syslog_in = obs_data(job, list(syslog), ObservationSource.SYSLOG) if syslog else None
        if syslog_in is not None:
            extracted.event_stream.register_observation_context(syslog_in.context)
        bad_nodes = core.calculate_bad_nodes(
            event_stream=extracted.event_stream,
            job_info=job,
            applog_in=extracted.observation_source,
            syslog_in=syslog_in,
        )
        recorder = DecisionRecorder(job.id)
        result = core.attribute(
            event_stream=extracted.event_stream,
            job_info=job,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
            recorder=recorder,
        )
        return result, build_decision_graph(recorder, result)

    return _attribute


@pytest.mark.parametrize(
    "nvrx_line",
    [
        _NVRX,
        "0: progress_tracker.py:248 EARLY TERMINATION: 3 consecutive training cycles without progress",
        "0: rank_monitor_server.py:523 [Cycle 1] Section 'step' has been open for 121 seconds. Timeout is 120",
    ],
)
def test_compiler_traceback_survives_nvrx_and_teardown(attribute_logs, nvrx_line):
    lines = [*_TRACEBACK, nvrx_line, "1: NCCL watchdog thread terminated", "1: exit code signal 15"]
    result, graph = attribute_logs(line_entries(lines))

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail MultiCategory"
    assert outcome.root_cause_symptom == _SYMPTOM
    assert outcome.doc == "jit-c-compilation-failure"
    assert outcome.hw_cause == "Underlying cause of JIT C-compilation failure is undetermined."
    assert "Python Exception" not in result.symptoms
    assert any(s.startswith("NVRx ") for s in result.symptoms.symptoms())
    source = result.symptoms[_SYMPTOM].one_source()
    assert source.rule == _RULE
    assert source.datum == _TRACEBACK[-1]
    assert source.line_num == len(_TRACEBACK)
    assert {r.hostname for r in source.resources} == {"node02"}
    evidence = [
        n for n in graph.nodes if n.rule_name == _RULE and n.node_type == NodeType.EXTRACTION and n.hostname == "node02"
    ]
    assert len(evidence) == 1
    assert evidence[0].datum == _TRACEBACK[-1]
    assert evidence[0].hostname == "node02"
    winner = next(n for n in graph.nodes if n.node_type == NodeType.OUTCOME)
    assert winner.symptom == _SYMPTOM
    evidence_ids = {n.node_id for n in graph.nodes if n.node_type == NodeType.EXTRACTION and n.datum == _TRACEBACK[-1]}
    assert any(e.parent_id in evidence_ids and e.child_id == winner.node_id for e in graph.edges)


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        " 1: ",
        "[rank1]: ",
        " 1: [rank1]: ",
        "[rank: 1]: ",
        "[default1]: ",
        " 1: [default1]: ",
        "[default1]:[rank1]: ",
        " 1: [default1]:[rank1]: ",
    ],
)
def test_compiler_failure_does_not_require_nvrx(attribute_logs, prefix):
    result, _ = attribute_logs(line_entries([prefix + _ERROR]), tags=frozenset())

    assert result.outcome_attr.value().outcome == "Fail MultiCategory"
    assert result.outcome_attr.value().root_cause_symptom == _SYMPTOM


@pytest.mark.parametrize(
    "compiler",
    [
        "aarch64-linux-gnu-gcc",
        "/usr/bin/aarch64-linux-gnu-gcc",
        "/usr/bin/aarch64-linux-gnu-gcc-13",
        "x86_64-linux-gnu-g++-13",
        "arm-none-eabi-gcc",
        "x86_64-unknown-linux-gnu-clang++-18",
    ],
)
@pytest.mark.parametrize("context", ["inductor", "triton", "unrelated"])
def test_target_prefixed_compiler_requires_framework_context(attribute_logs, compiler, context):
    error = _ERROR.replace("/usr/bin/gcc", compiler)
    if context != "inductor":
        error = error.replace(
            "torch._inductor.exc.InductorError: CalledProcessError:", "subprocess.CalledProcessError:"
        )
    if context != "triton":
        error = error.replace("/dist-packages/triton/", "/dist-packages/other_package/")
    line = "[default1]: " + error
    result, graph = attribute_logs(line_entries([line, _NVRX]))

    expected_match = context != "unrelated"
    assert (_SYMPTOM in result.symptoms) == expected_match
    assert any(node.rule_name == _RULE for node in graph.nodes) == expected_match
    assert result.outcome_attr.value().root_cause_symptom == (_SYMPTOM if expected_match else "NVRx detected failure")
    if expected_match:
        assert result.symptoms[_SYMPTOM].one_source().datum == line


@pytest.mark.parametrize("command", ["aarch64-linux-gnu-gcc-ar", "aarch64-linux-gnu-python", "wrapper-gcc", "notgcc"])
def test_similar_command_names_do_not_establish_compiler_failure(attribute_logs, command):
    result, graph = attribute_logs(line_entries([_ERROR.replace("/usr/bin/gcc", command), _NVRX]))

    assert _SYMPTOM not in result.symptoms
    assert result.outcome_attr.value().root_cause_symptom == "NVRx detected failure"
    assert not any(node.rule_name == _RULE for node in graph.nodes)


@pytest.mark.parametrize("prefix", ["[rank1]: ", "[default1]: "])
@pytest.mark.parametrize(
    "package_path, expected_match",
    [
        ("/usr/local/lib/python3.12/site-packages/triton/", True),
        ("/usr/local/lib/python3.12/dist-packages/triton/", True),
        ("/home/user/triton/", False),
        ("/tmp/triton/", False),
        ("/home/user/my-site-packages/triton/", False),
        ("/usr/local/lib/python3.12/site-packages/triton_helpers/", False),
    ],
)
def test_standalone_compiler_error_requires_installed_triton(attribute_logs, prefix, package_path, expected_match):
    error = _ERROR.replace("torch._inductor.exc.InductorError: CalledProcessError:", "subprocess.CalledProcessError:")
    line = prefix + error.replace("/usr/local/lib/python3.12/dist-packages/triton/", package_path)
    result, graph = attribute_logs(line_entries([line, _NVRX]))

    assert (_SYMPTOM in result.symptoms) == expected_match
    assert any(n.rule_name == _RULE for n in graph.nodes) == expected_match
    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail MultiCategory"
    assert outcome.root_cause_symptom == (_SYMPTOM if expected_match else "NVRx detected failure")
    if expected_match:
        assert result.symptoms[_SYMPTOM].one_source().datum == line


@pytest.mark.parametrize("has_timestamp", [False, True])
def test_missing_node_mapping_is_not_inferred(attribute_logs, has_timestamp):
    entry = {"body": "99: " + _ERROR, "resources": "AllJobResources"}
    if has_timestamp:
        entry["ts"] = _TIME_NS
    result, graph = attribute_logs([entry])

    source = result.symptoms[_SYMPTOM].one_source()
    assert source.resources == AllJobResources
    assert (source.time.timestamp() if source.time else None) == (_TIME_NS / 1e9 if has_timestamp else None)
    assert not result.bad_nodes_attr.my_bad_nodes.attributions
    evidence = next(n for n in graph.nodes if n.rule_name == _RULE)
    assert evidence.hostname is None
    assert evidence.timestamp == (source.time.isoformat() if source.time else None)


@pytest.mark.parametrize("has_peer_closure", [False, True])
def test_stronger_filesystem_evidence_wins_without_losing_compiler_detail(attribute_logs, has_peer_closure):
    line = "1: " + _ERROR
    entries = [make_applog_entry(line, _TIME_NS, "node02"), {"body": _NVRX}]
    if has_peer_closure:
        entries.append({"body": _PEER_CLOSED})
    result, graph = attribute_logs(
        entries,
        syslog=[make_syslog_entry(_EXT4, "node02", _TIME_NS - 1000000000)],
    )

    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail InfraSW"
    assert outcome.root_cause_symptom == "ext4 filesystem corruption"
    assert _SYMPTOM in result.symptoms
    assert result.symptoms[_SYMPTOM].one_source().datum == line
    assert any(n.rule_name == _RULE and n.hostname == "node02" for n in graph.nodes)
    assert any(n.symptom == "ext4 filesystem corruption" and n.hostname == "node02" for n in graph.nodes)


@pytest.mark.parametrize("has_nvrx", [False, True])
def test_compiler_failure_explains_closure_of_the_same_peer(attribute_logs, has_nvrx):
    line = "1: " + _ERROR
    lines = [line, _PEER_CLOSED]
    if has_nvrx:
        lines.append(_NVRX)
    result, graph = attribute_logs(line_entries(lines), tags=frozenset({NVRX_TAG}) if has_nvrx else frozenset())

    assert result.outcome_attr.value().outcome == "Fail MultiCategory"
    assert result.outcome_attr.value().root_cause_symptom == _SYMPTOM
    assert result.symptoms[_SYMPTOM].one_source().datum == line
    assert "Connection closed" not in result.symptoms
    assert any(n.rule_name == _RULE and n.datum == line and n.hostname == "node02" for n in graph.nodes)
    assert any(
        n.rule_name == "Connection closed by remote peer" and n.datum == _PEER_CLOSED and n.hostname == "node02"
        for n in graph.nodes
    )


@pytest.mark.parametrize("compiler_location", ["absent", "different_node", "unknown"])
def test_peer_closure_is_not_explained_without_compiler_failure_on_that_node(attribute_logs, compiler_location):
    entries = line_entries([_PEER_CLOSED, _NVRX])
    if compiler_location == "different_node":
        entries.insert(0, {"body": "0: " + _ERROR})
    elif compiler_location == "unknown":
        entries.insert(0, {"body": "99: " + _ERROR, "resources": "AllJobResources"})
    result, _ = attribute_logs(entries)

    assert result.outcome_attr.value().root_cause_symptom == "Connection closed"
    assert "Connection closed" in result.symptoms
    assert (_SYMPTOM in result.symptoms) == (compiler_location != "absent")


def test_repeated_compiler_failures_keep_each_source(attribute_logs):
    lines = ["0: " + _ERROR, "1: " + _ERROR.replace("/tmp/jit/", "/tmp/other-jit/")]
    result, _ = attribute_logs(line_entries([*lines, _NVRX]))

    assert {s.datum for s in result.symptoms[_SYMPTOM].all_sources()} == set(lines)
    assert {r.hostname for s in result.symptoms[_SYMPTOM].all_sources() for r in s.resources} == {"node01", "node02"}


@pytest.mark.parametrize(
    "job_options",
    [
        {"exit_code": 0, "exit_code_signal": 0},
        {"exit_code": 93, "exit_code_signal": 0},
        {"exit_code": 0, "exit_code_signal": 15},
        {"state": "CANCELLED", "cancelled_by": "test-user", "exit_code": 93},
        {"state": "TIMEOUT", "cancelled_by": "Timeout"},
    ],
)
def test_compiler_failure_survives_launcher_and_cancellation_metadata(attribute_logs, job_options):
    result, _ = attribute_logs(line_entries(["1: " + _ERROR, _NVRX]), **job_options)

    assert result.outcome_attr.value().outcome == "Fail MultiCategory"
    assert result.outcome_attr.value().root_cause_symptom == _SYMPTOM
    assert result.symptoms[_SYMPTOM].one_source().datum == "1: " + _ERROR


@pytest.mark.parametrize(
    "lines",
    [
        _TRACEBACK[:-1],
        ["1: Compiling Triton kernel with gcc", "1: Triton compilation completed successfully"],
        ["1: NCCL watchdog thread terminated"],
        ["1: handled " + _ERROR],
        [f'1: print("{_ERROR}")'],
        ["[default1]: handled " + _ERROR],
        [f'[default1]: print("{_ERROR}")'],
        ["1: subprocess.CalledProcessError: Command '['gcc', '/tmp/extension.c']' returned non-zero exit status 1."],
    ],
)
def test_generic_nvrx_failure_does_not_invent_compilation_failure(attribute_logs, lines):
    result, graph = attribute_logs(line_entries([*lines, _NVRX]))

    assert _SYMPTOM not in result.symptoms
    assert result.outcome_attr.value().outcome == "Fail MultiCategory"
    assert result.outcome_attr.value().root_cause_symptom == "NVRx detected failure"
    assert not any(n.rule_name == _RULE for n in graph.nodes)
