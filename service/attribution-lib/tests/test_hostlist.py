# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_lib.hostlist import (
    DuplicateHostError,
    HostlistError,
    InvalidRangeError,
    InvalidSyntaxError,
    parse_hostlist,
)


def test_simple_hostname():
    """Test parsing of simple hostnames without expansions"""
    assert parse_hostlist("node1") == ["node1"]
    assert parse_hostlist("compute-01") == ["compute-01"]
    assert parse_hostlist("node_a") == ["node_a"]


def test_single_range():
    """Test parsing of hostnames with single range expressions"""
    assert parse_hostlist("node[1-3]extra") == ["node1extra", "node2extra", "node3extra"]
    assert parse_hostlist("node[1-3]") == ["node1", "node2", "node3"]
    assert parse_hostlist("compute-[01-03]") == ["compute-01", "compute-02", "compute-03"]
    assert parse_hostlist("gpu[1-2]") == ["gpu1", "gpu2"]
    assert parse_hostlist("gpu[08-12]") == ["gpu08", "gpu09", "gpu10", "gpu11", "gpu12"]
    assert parse_hostlist("gpu[008-12]") == ["gpu008", "gpu009", "gpu010", "gpu011", "gpu012"]


def test_multiple_ranges():
    """Test parsing of hostnames with multiple range expressions"""
    assert parse_hostlist("node[1-2][3-4]") == ["node13", "node14", "node23", "node24"]
    assert parse_hostlist("compute-[01-02][1-2]") == ["compute-011", "compute-012", "compute-021", "compute-022"]


def test_syntax_errors():
    """Test various syntax error cases"""
    test_cases = [
        ("", "Empty hostlist expression"),
        ("node[", "Unclosed bracket"),
        ("node[1-2", "Missing closing bracket"),
        ("node[1-2],", "Unexpected end of input"),
        ("node[1-2].", "Unexpected end of input"),
        ("node[abc]", "Expected number or range"),
        ("node[1--2]", "Missing closing bracket"),
        ("node[-1-2]", "Expected number or range"),
    ]

    for input_str, error_msg in test_cases:
        with pytest.raises(InvalidSyntaxError) as exc:
            parse_hostlist(input_str)
        assert error_msg in str(exc.value)


def test_range_errors():
    """Test various range error cases"""
    test_cases = [
        ("node[5-3]", "start must be less than end"),
    ]

    for input_str, error_msg in test_cases:
        with pytest.raises(InvalidRangeError) as exc:
            parse_hostlist(input_str)
        assert error_msg in str(exc.value)


def test_duplicate_errors():
    """Test various duplicate host cases"""
    test_cases = [
        "node[1,1]",
        "node[1-2],node2",
        "node1,node1",
        "node[1-2].dom,node1.dom",
    ]

    for input_str in test_cases:
        with pytest.raises(DuplicateHostError) as exc:
            parse_hostlist(input_str)
        assert "Duplicate hosts found" in str(exc.value)


def test_complex_expressions():
    """Test parsing of complex host expressions"""
    # Multiple components with ranges
    assert parse_hostlist("node[1-2]-rack[01-02]-u[1,3]") == [
        "node1-rack01-u1",
        "node1-rack01-u3",
        "node1-rack02-u1",
        "node1-rack02-u3",
        "node2-rack01-u1",
        "node2-rack01-u3",
        "node2-rack02-u1",
        "node2-rack02-u3",
    ]

    # Domain names with multiple ranges
    assert parse_hostlist("compute[01-02].cluster[1-2].org") == [
        "compute01.cluster1.org",
        "compute01.cluster2.org",
        "compute02.cluster1.org",
        "compute02.cluster2.org",
    ]


def test_error_hierarchy():
    """Test that all errors inherit properly from HostlistError"""
    error_cases = [
        (lambda: parse_hostlist(""), InvalidSyntaxError),
        (lambda: parse_hostlist("node[5-3]"), InvalidRangeError),
        (lambda: parse_hostlist("node[1,1]"), DuplicateHostError),
    ]

    for error_func, expected_type in error_cases:
        with pytest.raises(HostlistError) as exc:
            error_func()
        assert isinstance(exc.value, expected_type)


def test_error_positions():
    """Test that error positions are correctly reported"""
    # Test unclosed bracket position
    with pytest.raises(InvalidSyntaxError) as exc:
        parse_hostlist("node[1-2,gpu[1-3")
    assert exc.value.position == 9  # Position of the ',gpu'

    # Test range error position
    with pytest.raises(InvalidRangeError) as exc:
        parse_hostlist("node[1-2],gpu[5-3]")
    assert exc.value.position == 14  # Position of '5-3'

    # Test unexpected end position
    with pytest.raises(InvalidSyntaxError) as exc:
        parse_hostlist("node[1-2],")
    assert exc.value.position == 9  # Position of the comma

    # Test invalid character position
    with pytest.raises(InvalidSyntaxError) as exc:
        parse_hostlist("node[abc]")
    assert exc.value.position == 5  # Position where number was expected


def test_error_messages():
    """Test that error messages include position information"""
    with pytest.raises(InvalidSyntaxError) as exc:
        parse_hostlist("node[1-2")
    assert "position" in str(exc.value)
    assert str(exc.value.position) in str(exc.value)
