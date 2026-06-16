# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re
from typing import Optional

from loguru import logger

# Pre-compiled regex patterns for better performance
COMPUTER_TO_SWITCH_LINK_PATTERN = re.compile(
    r"\((?P<src_type>Computer):(?P<src_name>[\w\-. ()]+) \((?P<src_guid>[0-9a-f]+)\)\)\s+"
    r"(?P<src_guid_2>[0-9a-f]+):(?P<src_port>[\d/]+)\s+-\s+"
    r"\((?P<dst_type>Switch):(?P<dst_name>[\w\-. ]+):(?P<dst_port>[\d/]+)\)\s+"
    r"(?P<dst_guid>[0-9a-f]+):(?P<dst_port_2>[\d/]+)"
)

SWITCH_TO_COMPUTER_LINK_PATTERN = re.compile(
    r"\((?P<src_type>Switch):(?P<src_name>[\w\-. ]+):(?P<src_port>[\d/]+)\)\s+"
    r"(?P<src_guid>[0-9a-f]+):(?P<src_port2>[\d/]+)\s+-\s+"
    r"\((?P<dst_type>Computer):(?P<dst_name>[\w\-. ()]+) \((?P<dst_guid>[0-9a-f]+)\)\)\s+"
    r"(?P<dst_guid_2>[0-9a-f]+):(?P<dst_port>[\d/]+)"
)

SWITCH_TO_SWITCH_LINK_PATTERN = re.compile(
    r"\((?P<src_type>Switch):(?P<src_name>[\w\-. ]+):(?P<src_port>[\d/]+)\)\s+"
    r"(?P<src_guid>[0-9a-f]+):[\d/]+\s+-\s+"
    r"\((?P<dst_type>Switch):(?P<dst_name>[\w\-. ]+):(?P<dst_port>[\d/]+)\)\s+"
    r"(?P<dst_guid>[0-9a-f]+):[\d/]+"
)

# Pre-compiled patterns for link type detection
COMPUTER_SWITCH_PATTERN = re.compile(r".*Computer.*Switch.*")
SWITCH_COMPUTER_PATTERN = re.compile(r".*Switch.*Computer.*")
SWITCH_SWITCH_PATTERN = re.compile(r".*Switch.*Switch.*")

# Pattern for "node is Up/Down" log messages
# Example: Site configuration changes: 1234567890 (host-c9d0e1f2) node is Down
NODE_UP_DOWN_PATTERN = re.compile(r"\((?P<node_name>[\w\-]+)\)\s+node is (Up|Down)")


def _parse_link_up_down_log(line: str) -> Optional[dict]:
    """Parse a single link up/down log line into structured fields."""
    # Use pre-compiled patterns for link type detection
    if COMPUTER_SWITCH_PATTERN.search(line):
        match = COMPUTER_TO_SWITCH_LINK_PATTERN.search(line)
    elif SWITCH_COMPUTER_PATTERN.search(line):
        match = SWITCH_TO_COMPUTER_LINK_PATTERN.search(line)
    elif SWITCH_SWITCH_PATTERN.search(line):
        match = SWITCH_TO_SWITCH_LINK_PATTERN.search(line)
    else:
        # No link pattern found - this is expected for other UFM log types (e.g., node up/down)
        return None

    if match:
        return {
            "src_type": match.group("src_type").lower(),
            "src_name": match.group("src_name"),
            "src_guid": match.group("src_guid"),
            "src_port": match.group("src_port"),
            "dst_type": match.group("dst_type").lower(),
            "dst_name": match.group("dst_name"),
            "dst_guid": match.group("dst_guid"),
            "dst_port": match.group("dst_port"),
        }
    return None


def _parse_node_up_down_log(line: str) -> Optional[dict]:
    """Parse a node up/down log line into structured fields.

    Example log: Site configuration changes: 1234567890 (host-c9d0e1f2) node is Down
    """
    if match := NODE_UP_DOWN_PATTERN.search(line):
        return {
            "node_name": match.group("node_name"),
        }
    return None


def extract_ufmlog_line_node(log_line: str) -> Optional[str]:
    """Parse log_line as though it were a UFM log line body and try to extract the node name.

    Supports two log formats:
    1. Link up/down: (Computer:host-name ...) - (Switch:...)
    2. Node up/down: Site configuration changes: 1234567890 (host-name) node is Down
    """
    # Try link up/down pattern first
    if parsed_info := _parse_link_up_down_log(log_line):
        node_name = parsed_info["src_name"] if parsed_info["src_type"] == "computer" else parsed_info["dst_name"]
        return node_name

    # Try node up/down pattern
    if parsed_info := _parse_node_up_down_log(log_line):
        return parsed_info["node_name"]

    logger.warning(f"Failed to parse UFM log: {log_line}")
    return None
