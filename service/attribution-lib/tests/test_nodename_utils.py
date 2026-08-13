# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Iterator
from pathlib import Path

import pytest

from attribution_lib.nodename_utils import (
    configure_node_mapping_file,
    convert_node_names_to_instance_ids,
    get_ip_to_node_name_mapping,
    is_ipv4,
    is_ipv6,
)


@pytest.fixture(autouse=True)
def reset_node_mapping() -> Iterator[None]:
    """Reset the process-wide node mapping before and after each test."""
    configure_node_mapping_file(None)
    yield
    configure_node_mapping_file(None)


def _write_mapping(tmp_path: Path, instance_id: str = "instance-a") -> Path:
    """Write a representative node mapping.

    Args:
        tmp_path: Temporary test directory.
        instance_id: Instance ID assigned to node-a.

    Returns:
        Path to the generated CSV file.
    """
    mapping_path = tmp_path / "nodes.csv"
    mapping_path.write_text(
        "id,instanceId,primaryIp,secondaryIp\n"
        f"node-a,{instance_id},192.0.2.10,2001:db8::10\n"
        "node-b,instance-b,192.0.2.11,\n",
        encoding="utf-8",
    )
    return mapping_path


def test_conversion_uses_configured_mapping_and_preserves_input_order(tmp_path: Path) -> None:
    """Known nodes convert in caller order while unknown nodes are omitted."""
    mapping_path = _write_mapping(tmp_path)
    configure_node_mapping_file(str(mapping_path))

    converted = convert_node_names_to_instance_ids(["node-b", "unknown", "node-a"])

    assert converted == ["instance-b", "instance-a"]


def test_reconfiguration_invalidates_the_cached_mapping(tmp_path: Path) -> None:
    """Changing the configured file replaces values loaded from the prior file."""
    first_mapping = _write_mapping(tmp_path, instance_id="instance-old")
    configure_node_mapping_file(str(first_mapping))
    assert convert_node_names_to_instance_ids(["node-a"]) == ["instance-old"]

    second_dir = tmp_path / "replacement"
    second_dir.mkdir()
    second_mapping = _write_mapping(second_dir, instance_id="instance-new")
    configure_node_mapping_file(str(second_mapping))

    assert convert_node_names_to_instance_ids(["node-a"]) == ["instance-new"]


def test_ip_mapping_includes_only_ipv4_values(tmp_path: Path) -> None:
    """IPv4 columns map to node names while IPv6 and identifiers do not."""
    mapping_path = _write_mapping(tmp_path)
    configure_node_mapping_file(str(mapping_path))

    assert get_ip_to_node_name_mapping() == {
        "192.0.2.10": "node-a",
        "192.0.2.11": "node-b",
    }


def test_disabled_mapping_returns_empty_results() -> None:
    """Public conversion functions fail closed when mapping is disabled."""
    assert convert_node_names_to_instance_ids(["node-a"]) == []
    assert get_ip_to_node_name_mapping() == {}


def test_invalid_mapping_schema_returns_no_instance_ids(tmp_path: Path) -> None:
    """A mapping without required columns cannot produce instance IDs."""
    mapping_path = tmp_path / "invalid.csv"
    mapping_path.write_text("id,primaryIp\nnode-a,192.0.2.10\n", encoding="utf-8")
    configure_node_mapping_file(str(mapping_path))

    assert convert_node_names_to_instance_ids(["node-a"]) == []


def test_ip_address_helpers_distinguish_protocol_versions() -> None:
    """IPv4 and IPv6 helpers accept only their respective address families."""
    assert is_ipv4("192.0.2.10") is True
    assert is_ipv4("2001:db8::10") is False
    assert is_ipv4("not-an-address") is False
    assert is_ipv6("2001:db8::10") is True
    assert is_ipv6("192.0.2.10") is False
    assert is_ipv6("not-an-address") is False
