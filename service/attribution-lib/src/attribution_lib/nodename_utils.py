# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import csv
import ipaddress
from pathlib import Path
from typing import Optional

import pandas as pd  # type: ignore[import-untyped]
from loguru import logger

# Internal caches
# This will be set by _load_node_instance_mapping()
_node_name_to_instance_id: dict[str, str] = {}  # Maps node names to cloud instance IDs
# This will be set by _load_ip_to_node_name()
_ip_to_node_name: dict[str, str] = {}  # Maps IP addresses to node names
# This will be set by configure_node_mapping_file()
_node_mapping_file: Optional[Path] = None  # File Path containing node name to cloud instance ID & IP mappings


def configure_node_mapping_file(csv_path: Optional[str]) -> None:
    """
    Configure the node name to cloud instance ID mapping file path.

    Supports cloud provider's instance identifiers.

    Args:
        csv_path: Path to CSV file containing node name to cloud instance ID mappings.
                 Can be None to disable.
    """
    global _node_mapping_file, _node_name_to_instance_id, _ip_to_node_name

    # Clear any cached data when reconfiguring
    _node_name_to_instance_id = {}
    _ip_to_node_name = {}

    # Convert string path to Path object
    if csv_path:
        _node_mapping_file = Path(csv_path)
    else:
        _node_mapping_file = None


def _load_node_instance_mapping(csv_path: Path) -> dict[str, str]:
    """
    Load node name to cloud instance ID mapping from CSV file.
    Args:
        csv_path: Path to the CSV file

    Returns:
        dict[str, str]: Dictionary mapping node names to cloud instance IDs
    """
    try:
        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        required_columns = ["id", "instanceId"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}. Found columns: {df.columns.tolist()}")

        mapping_dict = dict(zip(df["id"], df["instanceId"]))

        logger.info(f"Loaded {len(mapping_dict)} node name to cloud instance ID mappings")
        return mapping_dict

    except FileNotFoundError:
        logger.error(f"Mapping file not found at {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading mapping file: {e}")
        raise


def _load_ip_to_node_name(csv_path: Path) -> dict[str, str]:
    """
    Load IP to node name mapping from CSV file.
    Maps any column containing IPv4 addresses to their corresponding hostname.

    Args:
        csv_path (Path): Path to the CSV file
    """
    try:
        ip_2_node_name = dict()

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                hostname = row["id"]
                for _, value in row.items():
                    # Strip whitespace from value to handle any leading/trailing spaces
                    value = value.strip()
                    if value and is_ipv4(value):
                        ip_2_node_name[value] = hostname

        logger.info(f"Loaded {len(ip_2_node_name)} IP to node name mappings")
        return ip_2_node_name

    except Exception as e:
        logger.error(f"Error loading IP to node name mapping: {e}")
        raise


def convert_node_names_to_instance_ids(node_names: list[str]) -> list[str]:
    """
    Convert a list of node names to their corresponding cloud instance IDs.

    Args:
        node_names (list[str]): List of node names to convert

    Returns:
        list[str]: List of cloud instance IDs. Empty list if conversion fails or no valid mappings found.
    """
    try:
        if not isinstance(node_names, list):
            logger.error("node_names must be a list")
            return []

        if not node_names or not _node_mapping_file:
            return []

        global _node_name_to_instance_id
        if not _node_name_to_instance_id:
            _node_name_to_instance_id = _load_node_instance_mapping(_node_mapping_file)

        result = [
            str(_node_name_to_instance_id.get(node_name))
            for node_name in node_names
            if node_name in _node_name_to_instance_id
        ]
        return result
    except Exception as e:
        logger.error(f"Error converting node names to cloud instance IDs: {e}")
        return []


def get_ip_to_node_name_mapping() -> dict[str, str]:
    """
    Get the IP address to node name mapping.

    Returns:
        dict[str, str]: return the mapping dict from IP to node name.
    """
    if not _node_mapping_file:
        logger.info("get_ip_to_node_name_mapping() called but no node name mapping file configured")
        return {}

    global _ip_to_node_name
    if not _ip_to_node_name:
        _ip_to_node_name = _load_ip_to_node_name(_node_mapping_file)

    return _ip_to_node_name


def is_ipv4(ip: str) -> bool:
    """
    Check if the given string is an IPv4 address.

    Args:
        ip (str): IP address to check

    Returns:
        bool: True if the string is an IPv4 address, False otherwise
    """
    try:
        ipaddress.IPv4Address(ip)
    except ipaddress.AddressValueError:
        return False
    return True


def is_ipv6(ip: str) -> bool:
    """
    Check if the given string is an IPv6 address.

    Args:
        ip (str): IP address to check

    Returns:
        bool: True if the string is an IPv6 address, False otherwise
    """
    try:
        ipaddress.IPv6Address(ip)
    except ipaddress.AddressValueError:
        return False
    return True
