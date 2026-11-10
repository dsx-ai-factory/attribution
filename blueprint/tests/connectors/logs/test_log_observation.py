# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest
from nvidia_attribution_client import ObservationV1, ObservationV2

from attribution_blueprint.connectors.logs import log_observation
from attribution_blueprint.types import Source

_START = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
_END = datetime(2026, 5, 1, 13, 0, 0, tzinfo=timezone.utc)
_CONTEXT = {"id": "1", "nodes": ["n1"], "start_time": _START, "end_time": _END}


@pytest.mark.parametrize("source", [Source.APPLOG, Source.SYSLOG])
def test_entry_format_is_explicit_and_accepted_by_both_service_models(source):
    """Consumers can identify normalized entries without guessing from their source."""
    obs = log_observation(source=source, entries=[{"body": "line", "ts": 1}], context=_CONTEXT)

    assert obs["context"]["format"] == "entry_timeseries"
    assert ObservationV1.model_validate(obs).to_dict()["context"]["format"] == "entry_timeseries"
    assert ObservationV2.model_validate(obs).body == obs["body"]


def test_applog_defaults_to_unbounded_interval():
    obs = log_observation(source=Source.APPLOG, entries=[], context=_CONTEXT)
    assert obs["context"]["time_interval"] == {"start": None, "end": None}


def test_explicit_interval_overrides_source_default():
    tighter = {"start": _START, "end": _START}
    obs = log_observation(source=Source.APPLOG, entries=[], context=_CONTEXT, time_interval=tighter)
    assert obs["context"]["time_interval"] == tighter
