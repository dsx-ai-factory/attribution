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
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Rule files import shared helpers via `from common...`, which depends on the
# bundled_rules directory being importable as a top-level path during tests.
# This assumes test_support/ stays directly under extern/attribution/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "attribution_lib" / "bundled_rules"))

import pytest
import zstd

import attribution_lib
import attribution_lib.core as core
from attribution_lib.events import EventManager
from attribution_lib.observation import Context
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Id,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.resources.resource_set import AllJobResources
from test_support.helpers import (
    AttributionTestResult,
    create_simple_job_info,
    make_tflops_line,
    observation_base,
)


@pytest.fixture(scope="session", autouse=True)
def setup_rules():
    core.setup_rules()


@pytest.fixture(autouse=True)
def deterministic_node_mapping(monkeypatch):
    """Provide the explicit test inventory mapping used by peer-rule cases."""
    monkeypatch.setattr(attribution_lib, "ip_to_nodenames", {"127.0.0.1": "localhost"})


@pytest.fixture(scope="session", autouse=True)
def timezone():
    return ZoneInfo("America/Los_Angeles")


@pytest.fixture
def time_interval(timezone):
    now = datetime.now(tz=timezone)
    return TimeInterval(now, now + timedelta(minutes=5))


@pytest.fixture(scope="class")
def obs_file(timezone):
    def _build_observation(jobinfo, filename, source, format, *, time_interval=None):
        match format:
            case ObservationFormat.LOKI_DICT_LIST:
                data = json.loads(zstd.decompress(Path(filename).read_bytes()))
            case ObservationFormat.RAW_LOKI_STREAMS:
                data = json.loads(Path(filename).read_bytes())
            case _:
                with open(filename, "r") as fp:
                    data = fp.readlines()

        return observation_base(jobinfo, timezone, source, format, data, time_interval=time_interval)

    return _build_observation


@pytest.fixture(scope="class")
def obs_data(timezone):
    def _build_observation(jobinfo, data, source, format, *, insert_tflops_line=True, time_interval=None):
        if source == ObservationSource.APPLOG and jobinfo.last_tflops_time is not None and insert_tflops_line:
            tflops_line = make_tflops_line(jobinfo.last_tflops_time)
            if jobinfo.last_tflops_time == jobinfo.job_start_time:
                data = [tflops_line] + data
            elif jobinfo.last_tflops_time == jobinfo.job_end_time:
                data = data + [tflops_line]
        return observation_base(jobinfo, timezone, source, format, data, time_interval=time_interval)

    return _build_observation


@pytest.fixture
def job_resources():
    return AllJobResources


_COMMON_TEST_RULE_NAMES = {"test_rule", "test"}


@pytest.fixture(scope="class")
def event_manager_class():
    mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
    mgr.preload_rule_names(_COMMON_TEST_RULE_NAMES)
    yield mgr
    mgr.close()


@pytest.fixture(scope="class")
def event_stream_class(event_manager_class):
    def _create_stream(id):
        return event_manager_class.create_stream(id)

    return _create_stream


@pytest.fixture
def event_manager():
    mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
    mgr.preload_rule_names(_COMMON_TEST_RULE_NAMES)
    yield mgr
    mgr.close()


@pytest.fixture
def event_stream(event_manager):
    def _create_stream(id):
        return event_manager.create_stream(id)

    return _create_stream


@pytest.fixture(scope="session")
def simple_job_factory(timezone):
    def _create(**kwargs):
        return create_simple_job_info(timezone, **kwargs)

    return _create


@pytest.fixture
def inbound_context(time_interval, job_resources, timezone):
    return InboundContext(
        time_interval, job_resources, ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES, timezone
    )


@pytest.fixture
def full_context(inbound_context):
    return Context(**vars(inbound_context), id=Id(12345))


@pytest.fixture
def dummy_ctx(full_context):
    return full_context


@pytest.fixture
def attribution_test_builder(event_manager, obs_data, obs_file, timezone):
    def _build(
        job_info, data, source, format, check_fn=None, events_to_add=None, stream_id=None, insert_tflops_line=True
    ):
        stream_id = stream_id or f"test_{hash(str(data))}"
        event_stream = event_manager.create_stream(stream_id)
        if isinstance(data, (str, Path)):
            observation_source = obs_file(job_info, data, source, format)
        else:
            observation_source = obs_data(job_info, data, source, format, insert_tflops_line=insert_tflops_line)
        event_stream.register_observation_context(observation_source.context)
        if events_to_add:
            for event in events_to_add:
                event_stream.register_observation_context(event[2].source_context)
                event_stream.add_event(event[0], event[1], event[2])
        if check_fn:
            check_kwargs = {
                "event_stream": event_stream,
                "job_info": job_info,
                "observation_source": observation_source,
            }
            if source in (
                ObservationSource.APPLOG,
                ObservationSource.SYSLOG,
                ObservationSource.HEALTH_CHECK,
                ObservationSource.UFMLOG,
                ObservationSource.NODE_METRICS,
                ObservationSource.NVLINK_METRICS,
            ):
                check_kwargs["rules_engine"] = attribution_lib.get_rules_engine()
            check_fn(**check_kwargs)
        return AttributionTestResult(
            event_stream=event_stream, observation_source=observation_source, job_info=job_info
        )

    return _build
