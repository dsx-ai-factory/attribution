# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Marker module for the generic co-located test collector.

Root conftest.py intercepts collection of this file and substitutes a dynamic
collector that walks bundled-rule test roots for `*_test.py` modules.
"""

# This file is intentionally empty. Root conftest.py replaces its collection
# with a generic path-based collector for co-located rule tests.
