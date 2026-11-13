# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared synthetic evidence builders for bundled HCA rule tests."""

from collections import defaultdict
from dataclasses import asdict, replace
from datetime import timedelta

import pytest

from attribution_lib import core
from attribution_lib.decision_trace import build_decision_graph
from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.observation.inbound import (
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.sources import SourceSet
from attribution_lib.utils import hostname_mapping
from test_support.helpers import make_syslog_entry


@pytest.fixture
def graphs():
    return {}


@pytest.fixture
def job(simple_job_factory):
    return simple_job_factory(state="FAILED", exit_code_signal=1, nodes=["node-a", "node-b"], ranks_per_node=1)


def modify(errno=22, local="::ffff:192.0.2.1", remote="::ffff:192.0.2.2", device="mlx5_4:1"):
    return (
        f"node-a:123:456 [0] NCCL WARN Call to ibv_modify_qp failed with {errno} error, "
        f"on dev {device}, curr state RTR, next state RTS, local GID index -1, "
        f"local GID {local}, remote GID {remote}"
    )


def verb(name, errno=11, device="mlx5_4", process="123:456"):
    suffix = f", on dev {device}:1" if device else ""
    return f"node-a:{process} [0] NCCL WARN Call to {name} failed with error Resource temporarily unavailable errno {errno}{suffix}"


@pytest.fixture
def run(attribution_test_builder, obs_data, job, request, graphs):
    count = 0

    def _run(lines, syslogs=(), mapping=None, provider=(), job_script=None, declared_nodes=True):
        current_job = job._replace(job_script_content=job_script) if job_script else job
        nonlocal count
        count += 1
        start = job.job_start_time.timestamp()
        entries = []
        for line, offset in lines:
            entry = {"body": line}
            if offset is not None:
                entry["ts"] = int((start + 20 + offset) * 1e9)
            # Identity regressions deliberately omit producer-declared nodes.
            if declared_nodes:
                entry["resources"] = [
                    {"kind": "node", "hostname": "node-b" if line.startswith("node-b:") else "node-a"}
                ]
            entries.append(entry)
        extracted = attribution_test_builder(
            job_info=current_job,
            data=entries,
            source=ObservationSource.APPLOG,
            stream_id=f"{request.node.name}-{count}",
            insert_tflops_line=False,
        )
        app, stream = extracted.observation_source, extracted.event_stream
        sys = obs_data(
            job,
            [make_syslog_entry(line, node, (start + 20 + offset) * 1e9) for node, line, offset in syslogs],
            ObservationSource.SYSLOG,
            time_interval=TimeInterval(job.job_start_time, job.job_end_time + timedelta(minutes=3)),
        )
        stream.register_observation_context(sys.context)
        with hostname_mapping(mapping if mapping is not None else {"192.0.2.1": "node-a", "192.0.2.2": "node-b"}):
            bad = core.calculate_bad_nodes(event_stream=stream, job_info=current_job, applog_in=app, syslog_in=sys)
        # Exercise the normalized provider-evidence contract without embedding
        # provider-specific collection or inventory logic in public rules.
        for symptom, node, offset, kind in provider:
            context = replace(sys.context, source=kind)
            source = job.source("synthetic provider evidence", "test")._replace(
                source_context=context, time=job.job_start_time + timedelta(seconds=20 + offset)
            )
            bad.my_bad_nodes.add([node], SourceSet(symptom).add(source))
        recorder = DecisionRecorder(job.id)
        result = core.attribute(
            event_stream=stream,
            job_info=current_job,
            bad_node_result=bad,
            extract_job_metadata_from_logfile=False,
            run_collectors=True,
            recorder=recorder,
        )
        graphs[stream.ctx_id] = asdict(build_decision_graph(recorder, result))
        return result, stream

    return _run


def exported(result):
    bad, interesting, epilog = defaultdict(set), defaultdict(set), defaultdict(set)
    result.bad_nodes_attr.my_bad_nodes.extract_bad_interesting_epilog(bad, interesting, epilog)
    return bad, interesting


def events(stream, typ):
    return list(query_events_with_source(stream, stream.mgr.types.get_id(typ)))
