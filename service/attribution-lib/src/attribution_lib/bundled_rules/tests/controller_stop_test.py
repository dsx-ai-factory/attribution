# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution of jobs whose node was stopped by the cluster controller."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import attribution_lib
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.core import get_bundled_rules_path
from attribution_lib.events import EventManager
from attribution_lib.filter import run_pre_outcome_refiners
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import (
    Format as ObsFormat,
    Id,
    Source as ObsSource,
    TimeInterval,
)
from attribution_lib.outcome import calculate_outcome
from attribution_lib.resources import AllJobResources, Node, ResourceSet
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.sources import Source, SourceSet, SymptomSet

CONTROLLER_STOP = "Node stopped by controller"


# --- harness (mirrors refiner_rules_test.py) ---


@pytest.fixture(autouse=True)
def rules_engine():
    original_engine = attribution_lib._rules_engine
    loader = RuleLoader()
    rules_dir = get_bundled_rules_path()
    loader.load(str(rules_dir / "filter_rules.py"))
    loader.load(str(rules_dir / "refiner_rules.py"))
    attribution_lib._rules_engine = loader.build()
    yield attribution_lib.get_rules_engine()
    attribution_lib._rules_engine = original_engine


@pytest.fixture
def mgr(rules_engine):
    m = EventManager(rules_engine=rules_engine)
    m.preload_rule_names({"test"})
    yield m
    m.close()


@pytest.fixture
def stream(mgr):
    return mgr.create_stream("test_job")


@pytest.fixture
def obs_ctx():
    return ObservationContext(
        time_interval=TimeInterval(
            datetime(2024, 1, 1, 9, 0, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC")),
        ),
        source=ObsSource.APPLOG,
        format=ObsFormat.ENTRY_TIMESERIES,
        timezone=ZoneInfo("UTC"),
        id=Id(999),
    )


def _build_job_info(*, nodes=None, state="NODE_FAIL"):
    nodes_ = nodes if nodes is not None else ["node01"]
    start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    builder = JobInfoBuilder(
        id="123",
        type="batch",
        state=state,
        cluster="test",
        framework="pytorch",
        source_name="test",
        timezone=ZoneInfo("UTC"),
        job_start_time=start,
        node_to_ranks=JobInfoBuilder.uniform_node_to_ranks(nodes_, 1),
    )
    builder.srun_cmd("srun dummy")
    builder.exit_code_signal(1)
    builder.interest_end_time(start + timedelta(hours=1))
    builder.job_end_time(start + timedelta(hours=1))
    builder.time_limit(3600)
    builder.num_gpus(8)
    return builder.build()


@pytest.fixture
def job_info():
    return _build_job_info()


@pytest.fixture(autouse=True)
def _register_ctx(stream, obs_ctx):
    stream.register_observation_context(obs_ctx)


def _make_source(obs_ctx):
    return Source(
        source_context=obs_ctx,
        line_num=1,
        datum="test",
        time=datetime(2024, 1, 1, 10, 30, 0, tzinfo=ZoneInfo("UTC")),
        resources=AllJobResources,
        rule="test",
    )


def _make_symptoms(obs_ctx, *symptom_names):
    symptoms = SymptomSet()
    for name in symptom_names:
        symptoms.add(SourceSet(name).add(_make_source(obs_ctx)))
    return symptoms


def _make_bad_nodes(obs_ctx, node_symptoms):
    bad_nodes = JobNodeAttributions()
    for node, syms in node_symptoms.items():
        for sym in syms:
            bad_nodes.add([node], SourceSet(sym).add(_make_source(obs_ctx)))
    return bad_nodes


def _run(stream, job_info, symptoms, bad_nodes):
    run_pre_outcome_refiners(stream, job_info, symptoms=symptoms, my_bad_nodes=bad_nodes)


def _outcome(symptoms):
    return calculate_outcome(symptoms).value().outcome


# --- tests ---


def test_controller_stop_symptom_maps_to_infrasw():
    """Increment 1: the controller-stop symptom resolves to Fail InfraSW."""
    registry = attribution_lib.get_symptom_registry()
    match = registry.outcome_for([CONTROLLER_STOP])
    assert match is not None, f"{CONTROLLER_STOP!r} is not registered in the symptom registry"
    assert match.outcome == "Fail InfraSW"


def test_controller_stop_discards_node_failure_yielding_infrasw(stream, obs_ctx, job_info):
    """Increment 2: controller-stop cascades over generic node-down -> Fail InfraSW (the flip)."""
    symptoms = _make_symptoms(obs_ctx, "Node failure reported by slurm", CONTROLLER_STOP)
    bad_nodes = _make_bad_nodes(obs_ctx, {"node01": ["Node failure reported by slurm", CONTROLLER_STOP]})
    _run(stream, job_info, symptoms, bad_nodes)
    assert "Node failure reported by slurm" not in symptoms
    assert _outcome(symptoms) == "Fail InfraSW"


def test_controller_stop_yields_to_hardware_xid(stream, obs_ctx, job_info):
    """Increment 3: a real HW root cause (XID) still wins over controller-stop."""
    symptoms = _make_symptoms(obs_ctx, "XID-94 (Containment error)", "Node failure reported by slurm", CONTROLLER_STOP)
    bad_nodes = _make_bad_nodes(
        obs_ctx,
        {"node01": ["XID-94 (Containment error)", "Node failure reported by slurm", CONTROLLER_STOP]},
    )
    _run(stream, job_info, symptoms, bad_nodes)
    assert _outcome(symptoms) == "Fail InfraHW"


# Emission from the slurmctld log line is covered by the SyslogControllerStop
# case in syslog_rules.test.toml; per-job node scoping happens in Attribution Service.
