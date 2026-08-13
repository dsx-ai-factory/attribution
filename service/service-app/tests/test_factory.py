# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import subprocess
import sys


def test_importing_factory_does_not_initialize_rules():
    code = """
import attribution_lib

assert attribution_lib.initialized is False
import app.factory
assert attribution_lib.initialized is False
"""

    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
