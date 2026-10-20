# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import threading
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.attributor as attributor_module
from app.attributor import AttributorWrapper


def test_attribution_context_is_reused_until_successful_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated reads share one attribution run until an observation arrives."""
    attribute_calls = 0

    def run_attribution(checkpoint_backend, *, recorder, rule_params, run_collectors):
        del checkpoint_backend, recorder, rule_params, run_collectors
        nonlocal attribute_calls
        attribute_calls += 1
        return SimpleNamespace(call=attribute_calls)

    backend = SimpleNamespace(
        job_info=SimpleNamespace(id="job-a"),
        attribute=run_attribution,
        processed_ids=lambda: {"observation-a"},
        ingest=lambda _observation: "observation-b",
    )
    wrapper = AttributorWrapper.__new__(AttributorWrapper)
    wrapper.id = "job-a"
    wrapper.attributor = backend
    wrapper.parameters = SimpleNamespace(rule_params={})
    wrapper.lock = threading.Lock()
    wrapper.last_access_time = datetime.now()
    wrapper._checkpoint_backend = object()
    wrapper._timezone = "UTC"
    wrapper._attribution_context = None
    wrapper._closed = False
    monkeypatch.setattr(attributor_module, "_try_build_decision_graph", lambda _recorder, _result: object())

    # A graph run serves a later plain read; a plain run cannot serve a graph read.
    graph_context = wrapper.attribute(include_decision_graph=True)
    assert wrapper.attribute() is graph_context
    assert attribute_calls == 1

    wrapper.ingest(SimpleNamespace(to_lib_observation=lambda _timezone: object()))
    refreshed_plain = wrapper.attribute()
    refreshed_graph = wrapper.attribute(include_decision_graph=True)

    assert refreshed_plain is not graph_context
    assert refreshed_graph is not refreshed_plain
    assert wrapper.attribute() is refreshed_graph
    assert attribute_calls == 3

    # A deleted attributor must not serve its cached context to a request
    # that fetched it before the deletion.
    wrapper._event_manager = SimpleNamespace(close=lambda **_kwargs: None)
    wrapper.close()
    with pytest.raises(HTTPException) as refused:
        wrapper.attribute()
    assert refused.value.status_code == 404
    assert attribute_calls == 3
