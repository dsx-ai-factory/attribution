# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""When an attributor's DuckDB connection exceeds its memory budget with disk
spill disabled, `attribution_lib` raises `EventStorageOutOfMemoryError`. These
tests verify `app/attributor.py` translates that into a clean, catchable
507 response instead of a generic 500, isolating the failure to the one
attributor/request that triggered it, per the request that this be handled
robustly rather than crash the whole process.

The real overflow (memory_limit + no disk spill) is exercised in
`attribution-lib`'s own test suite; these tests verify only the translation at
the app layer, so they trigger it deterministically via monkeypatching
rather than depending on real DuckDB memory pressure.

Two levels are covered, matching the pattern in test_attributor_manager.py:
- AttributorWrapper.ingest()/attribute() raise AttributionHTTPException
  directly (pytest.raises, no HTTP layer involved).
- The full HTTP stack turns that into a 507 JSON response.
"""

import pytest
from attribution_lib.exceptions import EventStorageOutOfMemoryError as LibEventStorageOutOfMemoryError

from app.exceptions import AttributionHTTPException, EventStorageOverflowError
from tests.conftest import make_up_attributor_info, post_attributor
from tests.test_basic_functionality import fake_applog_observation


def _raise_overflow(*args, **kwargs):
    raise LibEventStorageOutOfMemoryError("simulated overflow")


def test_ingest_raises_attribution_http_exception_on_overflow(client, monkeypatch):
    att_id = post_attributor(client, make_up_attributor_info(id="overflow-ingest-raises"))
    wrapper = client.app.state.aim.attributors[att_id]
    monkeypatch.setattr(wrapper.attributor, "ingest", _raise_overflow)

    obs = fake_applog_observation("Example\nLines")
    with pytest.raises(AttributionHTTPException) as e:
        wrapper.ingest(obs)

    assert e.value.status_code == EventStorageOverflowError.code
    assert e.value.detail["att_id"] == att_id


def test_attribute_raises_attribution_http_exception_on_overflow(client, monkeypatch):
    att_id = post_attributor(client, make_up_attributor_info(id="overflow-attribute-raises"))
    wrapper = client.app.state.aim.attributors[att_id]
    monkeypatch.setattr(wrapper.attributor, "attribute", _raise_overflow)

    with pytest.raises(AttributionHTTPException) as e:
        wrapper.attribute()

    assert e.value.status_code == EventStorageOverflowError.code
    assert e.value.detail["att_id"] == att_id


def test_ingest_translates_overflow_to_507(client, monkeypatch):
    att_id = post_attributor(client, make_up_attributor_info(id="overflow-ingest"))
    wrapper = client.app.state.aim.attributors[att_id]
    monkeypatch.setattr(wrapper.attributor, "ingest", _raise_overflow)

    obs = fake_applog_observation("Example\nLines")
    response = client.post(f"/v1/attributor/{att_id}/observation", content=obs.model_dump_json())

    assert response.status_code == EventStorageOverflowError.code
    assert response.json()["att_id"] == att_id


def test_attribute_translates_overflow_to_507(client, monkeypatch):
    att_id = post_attributor(client, make_up_attributor_info(id="overflow-attribute"))
    wrapper = client.app.state.aim.attributors[att_id]
    monkeypatch.setattr(wrapper.attributor, "attribute", _raise_overflow)

    response = client.get(f"/v1/attributor/{att_id}/attribution")

    assert response.status_code == EventStorageOverflowError.code
    assert response.json()["att_id"] == att_id


@pytest.mark.parametrize(
    "path,method",
    [
        ("/v1/attributor/{att_id}/observation", "post"),
        ("/v1/attributor/{att_id}/attribution", "get"),
        ("/v2/attributor/{att_id}/observation", "post"),
        ("/v2/attributor/{att_id}/attribution", "get"),
    ],
)
def test_507_is_documented_on_routes_that_can_raise_it(app, path, method):
    """`ingest()`/`attribute()` (see `app/attributor.py`) can both raise
    `EventStorageOverflowError`, so every route that calls them should
    document 507 in its OpenAPI `responses`, not just the routes' happy path
    and client-fixable errors."""
    operation = app.openapi()["paths"][path][method]
    assert "507" in operation["responses"], f"{method.upper()} {path} is missing a documented 507 response"
