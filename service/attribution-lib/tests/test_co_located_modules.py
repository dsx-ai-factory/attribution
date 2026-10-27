# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Marker module for the generic co-located test collector.

Root conftest.py intercepts collection of this file and substitutes a dynamic
collector that walks bundled-rule test roots for `*_test.py` modules.
"""

# This file is intentionally empty. Root conftest.py replaces its collection
# with a generic path-based collector for co-located rule tests.
