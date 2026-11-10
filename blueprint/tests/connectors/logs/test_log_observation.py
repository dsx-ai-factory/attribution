# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

from attribution_blueprint.connectors.logs import HEALTH_CHECK_GRACE, log_observation
from attribution_blueprint.types import Source

_START = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
_END = datetime(2026, 5, 1, 13, 0, 0, tzinfo=timezone.utc)
_CONTEXT = {"id": "1", "nodes": ["n1"], "start_time": _START, "end_time": _END}


def test_applog_defaults_to_unbounded_interval():
    obs = log_observation(source=Source.APPLOG, entries=[], context=_CONTEXT)
    assert obs["context"]["time_interval"] == {"start": None, "end": None}


def test_health_check_window_reaches_past_the_job_end():
    """A health check reports after the job ends, so its window keeps a grace past `end_time`."""
    obs = log_observation(source=Source.HEALTH_CHECK, entries=[], context=_CONTEXT)
    assert obs["context"]["time_interval"] == {"start": _START, "end": _END + HEALTH_CHECK_GRACE}


def test_explicit_interval_overrides_source_default():
    tighter = {"start": _START, "end": _START}
    obs = log_observation(source=Source.APPLOG, entries=[], context=_CONTEXT, time_interval=tighter)
    assert obs["context"]["time_interval"] == tighter
