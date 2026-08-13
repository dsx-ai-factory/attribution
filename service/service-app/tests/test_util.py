# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.util import TimeInterval, to_zoneinfo


def test_TimeInterval_expected():
    now = datetime.now().astimezone()

    start = now - timedelta(minutes=20)
    end = now - timedelta(minutes=10)

    _ = TimeInterval(start=start, end=end)


def test_HashableModel_eq():
    now = datetime.now().astimezone()

    start = now - timedelta(minutes=20)
    end = now - timedelta(minutes=10)

    a = TimeInterval(start=start, end=end)
    b = TimeInterval(start=start, end=end)

    assert a == b
    assert a != start


@pytest.mark.parametrize(
    "tz_input, expected_key",
    [
        pytest.param(ZoneInfo("America/Chicago"), "America/Chicago", id="zoneinfo_passthrough"),
        pytest.param(None, "UTC", id="naive_falls_back_to_utc"),
        pytest.param(timezone.utc, "UTC", id="utc_offset_zero"),
        pytest.param(timezone(timedelta(hours=0)), "UTC", id="fixed_zero_offset"),
        pytest.param(timezone(timedelta(hours=-7)), "Etc/GMT+7", id="negative_offset_utc_minus_7"),
        pytest.param(timezone(timedelta(hours=5)), "Etc/GMT-5", id="positive_offset_utc_plus_5"),
        pytest.param(timezone(timedelta(hours=-12)), "Etc/GMT+12", id="negative_offset_utc_minus_12"),
        pytest.param(timezone(timedelta(hours=5, minutes=30)), "UTC", id="non_hour_boundary_falls_back_to_utc"),
    ],
)
def test_to_zoneinfo(tz_input, expected_key):
    dt = datetime(2025, 6, 1, 12, 0, 0, tzinfo=tz_input)
    result = to_zoneinfo(dt)
    assert isinstance(result, ZoneInfo)
    assert result.key == expected_key
