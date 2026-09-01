# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_lib.bundled_rules.common.xid_errors import create_symptom_nvl, format_xid_with_subcode

ERROR_EVENT = (
    "NVRM: Xid (PCI:0000:01:00): 150, SAW_MSEFW Fatal XC0 i0 Link 0 "
    "(0x00000002 0xffffffff 0x00000000 0x00000000 0x00000000 0x00000000)"
)
DECODED_ERROR = {
    "subcode": "SAW_MSEFW",
    "port_down_reason_code": "NO_LINK_DOWN",
    "port_down_reason_code_value": "0",
}
EXPECTED_SYMPTOMS = [
    (144, "XID-144-SAW_MSEFW-4294967295"),
    (145, "XID-145-SAW_MSEFW-4294967295"),
    (146, "XID-146-SAW_MSEFW-4294967295"),
    (147, "XID-147-SAW_MSEFW-4294967295"),
    (148, "XID-148-SAW_MSEFW-4294967295"),
    (149, "XID-149.0-SAW_MSEFW-NO_LINK_DOWN-4294967295"),
    (150, "XID-150-SAW_MSEFW"),
]


@pytest.mark.parametrize(("xid", "expected"), EXPECTED_SYMPTOMS)
def test_format_xid_with_subcode_omits_error_status_only_for_xid_150(xid: int, expected: str) -> None:
    """Verify PCI mapping keys retain status except for XID 150.

    Args:
        xid: XID number being formatted.
        expected: Expected PCI mapping key.
    """
    assert format_xid_with_subcode(xid, ERROR_EVENT) == expected


@pytest.mark.parametrize(("xid", "expected"), EXPECTED_SYMPTOMS)
def test_create_symptom_nvl_omits_error_status_only_for_xid_150(xid: int, expected: str) -> None:
    """Verify emitted symptoms retain status except for XID 150.

    Args:
        xid: XID number being formatted.
        expected: Expected emitted symptom.
    """
    assert create_symptom_nvl(xid, DECODED_ERROR, "4294967295") == expected
