# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Minimal helper utilities for attribution tests."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pytest
from pytest import param as pparam

from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.events import EventStream
from attribution_lib.job_info import JobInfo, JobInfoBuilder
from attribution_lib.observation import (
    Context as ObservationContext,
    Observation,
)
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    TimeInterval,
)
from attribution_lib.observation.processing import EntrySourceBase
from attribution_lib.resources.resource_set import AllJobResources, ResourceSet
from attribution_lib.sources import Source, SourceSet

xfail = pytest.mark.xfail


def make_node_to_ranks(nodes: list[str], ranks_per_node: int) -> dict[str, list[int]]:
    """Test-only alias for `JobInfoBuilder.uniform_node_to_ranks`."""
    return JobInfoBuilder.uniform_node_to_ranks(nodes, ranks_per_node)


def create_simple_job_info(
    timezone=None,
    request=None,
    job_id="12345",
    cluster="test-cluster",
    state="RUNNING",
    job_type="training",
    framework="pytorch",
    ranks_per_node=8,
    start_timestamp=1700000000,
    end_timestamp=1700010000,
    nodes=None,
    node_to_ranks=None,
    time_limit=6000,
    exit_code_signal=0,
    srun_cmd="srun test.py",
    source_name="artificial slurm",
    slurm_node_infos=None,
    failed_node=None,
):
    if timezone is None:
        if request is None:
            raise ValueError(
                "Either 'timezone' or 'request' must be provided. "
                "Pass timezone explicitly or use request=request to auto-detect."
            )
        timezone = request.getfixturevalue("timezone")

    assert not (node_to_ranks is not None and nodes is not None), (
        "create_simple_job_info: pass either node_to_ranks=... or nodes=..., not both"
    )
    if node_to_ranks is None:
        if nodes is None:
            nodes = ["test-node-001"]
        node_to_ranks = make_node_to_ranks(nodes, ranks_per_node)

    builder = JobInfoBuilder(
        id=job_id,
        cluster=cluster,
        state=state,
        type=job_type,
        framework=framework,
        job_start_time=datetime.fromtimestamp(start_timestamp, tz=timezone),
        source_name=source_name,
        timezone=timezone,
        node_to_ranks=node_to_ranks,
    )

    builder = (
        builder.job_end_time(datetime.fromtimestamp(end_timestamp, tz=timezone))
        .srun_cmd(srun_cmd)
        .exit_code_signal(exit_code_signal)
        .time_limit(time_limit)
    )
    if slurm_node_infos is not None:
        builder = builder.slurm_node_infos(slurm_node_infos)
    if failed_node is not None:
        builder = builder.failed_node(failed_node)
    return builder.build()


def observation_base(jobinfo, timezone, source, format, data, *, time_interval=None):
    if time_interval is None:
        time_interval = TimeInterval(jobinfo.job_start_time, jobinfo.job_end_time)
    context = InboundContext(
        time_interval=time_interval,
        resources=AllJobResources,
        source=source,
        format=format,
        timezone=timezone,
    )
    observation = Observation.from_inbound(context=context, lines=data)
    return EntrySourceBase.from_observation(observation)


def match_bad_peers(node_attr, nodes, symptom, rule):
    for node_name in nodes:
        node_group = node_attr[node_name]
        assert symptom in node_group.attributions._data
        sources = node_group.attributions[symptom]
        assert any(source.rule == rule for source in sources._bad)


def match_bad(node_attr, node, symptom, rule):
    node_group = node_attr[node]
    assert symptom in node_group.attributions._data
    sources = node_group.attributions[symptom]
    assert any(source.rule == rule for source in sources._bad)


def match_interesting_peers(node_attr, nodes, symptom, rule, allow_bad=False):
    for node_name in nodes:
        node_group = node_attr[node_name]
        assert symptom in node_group.attributions._data
        sources = node_group.attributions[symptom]
        assert any(source.rule == rule for source in sources._interesting)
        if not allow_bad:
            assert not sources.has_bad()


def match_interesting(node_attr, node, symptom, rule, allow_bad=False):
    node_group = node_attr[node]
    assert symptom in node_group.attributions._data
    sources = node_group.attributions[symptom]
    assert any(source.rule == rule for source in sources._interesting)
    if not allow_bad:
        assert not sources.has_bad()


def make_syslog_entry(body: str, node: str, timestamp_ns: int | float, severity: str = "error") -> dict:
    payload = json.dumps({"body": body, "severity": severity, "attributes": {"hostname": node}})
    return {
        "stream": {
            "detected_level": severity.upper(),
            "job": "test-job",
            "service_name": "syslog",
        },
        "values": [
            [
                str(int(timestamp_ns)),
                payload,
            ],
        ],
    }


def make_applog_entry(
    body: str,
    timestamp_ns: int | float,
    severity: str = "info",
    hostname: Optional[str] = None,
    node: Optional[str] = None,
) -> dict:
    if hostname is None:
        hostname = node

    attributes = {}
    if hostname:
        attributes["hostname"] = hostname

    attrs_json = json.dumps(attributes) if attributes else "{}"

    return {
        "stream": {
            "detected_level": severity.upper(),
            "job": "test-job",
            "service_name": "applog",
        },
        "values": [
            [
                str(int(timestamp_ns)),
                f'{{"body":"{body}","severity":"{severity}","attributes":{attrs_json}}}',
            ],
        ],
    }


class UnknownAttr(Exception):
    def __init__(self, name):
        self.name = name


def make_tflops_line(time: datetime, rank: int = 0) -> str:
    return f"{rank}:  [{time.isoformat()}] iteration   516520/  900692 | consumed samples:   1067990784 | elapsed time per iteration (ms): 8096.8 | throughput per GPU (TFLOP/s/GPU): 399.3 | learning rate: 4.838E-05 | decoupled learning rate: 2.190E-04 | global batch size:  2304 | lm loss: 1.437022E+00 | loss scale: 1.0 | grad norm: 0.217 | num zeros: 1640372992.0 | number of skipped iterations:   0 | number of nan iterations:   0 |"


def rule_lines(lines, attr, nodes, symptom, rule, fail_reason=None):
    if fail_reason is not None:
        return [pparam(li, attr, nodes, symptom, rule, marks=xfail(reason=fail_reason)) for li in lines]
    return [(li, attr, nodes, symptom, rule) for li in lines]


@dataclass
class AttributionTestResult:
    event_stream: EventStream
    observation_source: EntrySourceBase
    job_info: JobInfo

    def get_node_attributions(self) -> JobNodeAttributions:
        return JobNodeAttributions.from_event_stream(self.event_stream, self.job_info)


def create_source_set(
    context: ObservationContext,
    symptom_name: str,
    datum: str = "TestDatum",
    line_num: int = 1,
    time: datetime | None = None,
    rule: str = "test_rule",
) -> SourceSet:
    src = Source(
        source_context=context,
        line_num=line_num,
        datum=datum,
        time=time or datetime.now(),
        resources=ResourceSet(),
        rule=rule,
    )
    src_set = SourceSet(symptom_name)
    src_set.add(src)
    return src_set
