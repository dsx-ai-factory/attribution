# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
UFM log pre-filter data: constants, rules mapping, and exclude keywords.

Ported from attribution_lib/rules/ufmlog_to_symptom.py.  Importing this
module registers the pre-filter with the rules-engine framework.
"""

import re

from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rules_engine import PreFilterSpec, register_pre_filter

# ---------------------------------------------------------------------------
# Constants  (from ufmlog_to_symptom.py)
# ---------------------------------------------------------------------------
LINK_DOWN_EAST_WEST = "link down(east-west)"
NODE_UP = "NODE_UP"
NODE_DOWN_UFM = "node down reported by ufm"
BER_ERROR = "BER_ERROR"

# Pre-compiled regex patterns for better performance
LINK_UP_PATTERN = re.compile(r".*Link is up.*")
LINK_DOWN_PATTERN = re.compile(r".*Link went down.*")

# Define rules for mapping log patterns to symptoms
rules = {
    # example log: 2025-06-06 23:59:08.664 [13232552] [328] INFO Site [ABC] [Fabric_Topology] Link [Source 1234567890_1  TO Dest: 1234567890_2]: Link is up: (Computer:host-y5z6a7b8 (946dae03008b05ac)) 946dae03008b05ac:1 - (Switch:1234) fc6a1c0300766b40:25
    # ".*Link is up.*": LINK_DOWN_EAST_WEST, # disable it to lower the ufmlog line
    # example log: 2025-06-07 00:01:41.957 [13232559] [329] WARNING Site [ABC] [Fabric_Topology] Link [Source 1234567890_1  TO Dest: 1234567890_2]:
    ".*Link went down.*": LINK_DOWN_EAST_WEST,
    # #example log: 2025-06-10 16:12:47.883 [13273126] [332] INFO Site [ABC] [Fabric_Topology] Site [default(1234) / NA / NA / NA]: Site configuration changes: 1234567890 (host-c9d0e1f2) node is Up
    # (".*node is Up.*", NetworkSymptomTypes.NODE_UP),
    # example log: 2025-06-10 16:12:16.484 [13273115] [331] WARNING Site [ABC] [Fabric_Topology] Site [default(1234) / NA / NA / NA]: Site configuration changes: 1234567890 (host-c9d0e1f2) node is Down
    ".*node is Down.*": NODE_DOWN_UFM,
    # #example log: 2025-06-10 16:34:40.038 [13273591] [1321] WARNING Site [ABC] [Fabric_Topology] Site [default(1234) / NA / NA / NA]: Link flap of link between node: 1234567890 port: 00 and node: 0x9c0591030090b680 port: 00
    # (".*Link flap.*", NetworkSymptomTypes.LINK_DOWN_EAST_WEST)
    # #example log: 2025-06-10 15:39:49.645 [13272904] [917] CRITICAL Site [ABC] [Hardware] IBPort [default(1234) / Computer: host-g3h4i5j6 / HCA-10] [dev_id: 1234567890]: [None]:Symbol BER Error on Port (4.000000e-05)  exceeded critical threshold (1.000000e-14)
    # (".*BER Error.*", NetworkSymptomTypes.BER_ERROR),
    # backup rules that we are not sure about whether or not they are useful
    # example log:  2025-06-10 15:37:12.161 [13272860] [65] WARNING Site [ABC] [Fabric_Notification] IBPort [default(1234) / Computer: host-c9d0e1f2 / HCA-3] [dev_id: 1234567890]: GID Address Out of Service: prefix fe80000000000000,guid 1234567890 Link Source 1234567890_1  TO Dest: 1234567890_2
    # "GID Address Out of Service": "",
}

exclude_keywords = ["message repeated", "MT4129 ConnectX7"]

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
# Include terms are plain substrings for engine pre-filtering and Loki query
# shaping. Extracted from the regex patterns in the rules dict above.
include_terms = ["Link went down", "node is Down"]

register_pre_filter(
    ObservationSource.UFMLOG,
    PreFilterSpec(
        include_terms=include_terms,
        exclude_terms=exclude_keywords,
    ),
)
