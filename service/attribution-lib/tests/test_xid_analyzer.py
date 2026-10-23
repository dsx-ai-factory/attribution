# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for XID parsing, burst grouping, and resolution lookup."""

import subprocess
import sys
from datetime import datetime

import pytest

from attribution_lib.xid_analyzer import nvidia_xid_analyzer as xid_analyzer

# Codes exercised below are taken from the public Xid catalog.
_NON_STICKY = 13
_INTERRUPT = 79
_STICKY = 109
_DRIVER = "525.105.17"
_MACHINE = "gpu-node-01"


def _xid_message(code: int, detail: str = "test") -> str:
    """Build a representative public NVIDIA XID message.

    Args:
        code: Numeric XID code to include.
        detail: Message detail appended after the process fields.

    Returns:
        A public dmesg-style XID message.
    """
    return f"NVRM: Xid (PCI:0000:ae:00): {code}, pid=1, name=app, {detail}"


def _entry(code: int, timestamp: str, *, driver: str = _DRIVER, machine: str = _MACHINE):
    """Create a parsed XID entry for a test scenario.

    Args:
        code: Numeric XID code to parse.
        timestamp: Timestamp supplied with the message.
        driver: Driver version associated with the entry.
        machine: Machine name associated with the entry.

    Returns:
        The parsed XID entry.
    """
    entry = xid_analyzer.XIDEntry.from_message(
        message=_xid_message(code),
        timestamp=timestamp,
        driver=driver,
        machine=machine,
    )
    assert entry is not None
    return entry


def _analyze(entries):
    """Run the batch analysis path used by syslog attribution."""
    fleet = xid_analyzer.FleetData()
    for entry in entries:
        fleet._add_xid_entry(entry, drop_duplicates=True)
    fleet.find_resolutions()
    fleet.tag_xids()
    fleet.group_xids_into_bursts()
    fleet.match_bursts_to_patterns()
    return fleet


def test_setup_rules_loads_catalog_and_policy():
    """Verify setup loads the catalog, sticky codes, and parsing policy."""

    xid_analyzer.setup_rules()

    assert xid_analyzer.XID_CATALOG
    assert str(_NON_STICKY) in xid_analyzer.XID_CATALOG
    assert str(_STICKY) in xid_analyzer.STICKY_XIDS
    assert xid_analyzer.XID_POLICY is not None
    assert xid_analyzer.XID_POLICY.XIDPolicyEntries


def test_import_does_not_configure_root_logging():
    """Verify importing the embedded analyzer does not mutate host logging."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import logging; "
                "assert logging.getLogger().handlers == []; "
                "from attribution_lib.xid_analyzer import nvidia_xid_analyzer; "
                "assert logging.getLogger().handlers == []"
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_from_message_parses_public_xid_line():
    """Verify a public dmesg XID line populates the core entry fields."""

    xid_analyzer.setup_rules()
    message = _xid_message(_INTERRUPT, "fallen off the bus")

    entry = xid_analyzer.XIDEntry.from_message(
        message=message,
        timestamp="2024-01-01 10:00:00",
        driver=_DRIVER,
        machine=_MACHINE,
    )

    assert entry is not None
    assert entry.number == _INTERRUPT
    assert entry.machine == _MACHINE
    assert entry.driver == _DRIVER
    assert entry.pcie_bdf == "0000:ae:00"
    assert entry.context == message
    assert entry.timestamp is not None


@pytest.mark.parametrize(
    ("contained_detail", "expected"),
    [
        ("CE User Channel (0x36)", "CE"),
        ("SM (0x1)", "SM"),
        ("GSP", "GSP"),
    ],
)
def test_from_message_extracts_xid94_poison_victim(contained_detail: str, expected: str) -> None:
    """Verify XID 94 containment details produce a normalized poison victim."""

    entry = xid_analyzer.XIDEntry.from_message(
        message=f"NVRM: Xid (PCI:0000:ae:00): 94, Contained: {contained_detail}. RST: No, D-RST: No"
    )

    assert entry is not None
    assert entry.number == 94
    assert entry.poison_victim == expected


def test_from_message_extracts_xid94_poison_victim_from_redfish() -> None:
    """Verify accepted Redfish XID 94 messages retain poison-victim identity."""

    entry = xid_analyzer.XIDEntry.from_message(
        message=(
            "Driver Event Message has detected errors of type "
            "'[2024-01-01 10:00:00][GPU0] XID 94 "
            "Contained: SM (0x1). RST: No, D-RST: No'"
        ),
        driver=_DRIVER,
        machine=_MACHINE,
    )

    assert entry is not None
    assert entry.number == 94
    assert entry.poison_victim == "SM"


def test_from_message_leaves_poison_victim_unset_without_supported_xid94_detail() -> None:
    """Verify generic XID messages do not invent poison-victim identity."""

    entry = xid_analyzer.XIDEntry.from_message(message=_xid_message(94, "Contained: unknown engine. RST: No"))

    assert entry is not None
    assert entry.poison_victim is None


@pytest.mark.parametrize(
    "message",
    [
        "",
        "GPU is healthy",
        "NVRM: Xid (PCI:0000:ae:00): 94, Contained: SM",
    ],
)
def test_extract_xid94_poison_victim_rejects_malformed_input(message: str) -> None:
    """Return no identity for unrelated or incomplete input."""

    assert xid_analyzer.extract_xid94_poison_victim(message) is None


def test_from_message_parses_fractional_kernel_uptime():
    """Verify fractional dmesg uptime is converted deterministically."""

    entry = xid_analyzer.XIDEntry.from_message(message=f"[ 123.456789] {_xid_message(_INTERRUPT)}")

    assert entry is not None
    assert entry.timestamp == xid_analyzer.ensure_timezone_local(datetime.fromtimestamp(123.456789))


def test_from_message_without_timestamp_uses_epoch():
    """Verify timestamp-less dmesg entries remain safe for burst grouping."""

    xid_analyzer.setup_rules()
    entry = xid_analyzer.XIDEntry.from_message(message=_xid_message(_INTERRUPT))

    assert entry is not None
    assert entry.timestamp == xid_analyzer.ensure_timezone_local(datetime.fromtimestamp(0))
    assert len(xid_analyzer.SplitListToBursts([entry])) == 1


def test_from_message_parses_generic_redfish_event() -> None:
    """Verify generic Redfish events preserve their embedded timestamp."""

    entry = xid_analyzer.XIDEntry.from_message(
        message=(
            "Driver Event Message has detected errors of type '[2024-01-01 10:00:00][GPU0] XID 79 fallen off the bus'"
        ),
        driver=_DRIVER,
        machine=_MACHINE,
    )

    assert entry is not None
    assert entry.number == _INTERRUPT
    assert entry.timestamp == xid_analyzer.ensure_timezone_local(datetime(2024, 1, 1, 10, 0, 0))
    assert entry.driver == _DRIVER
    assert entry.machine == _MACHINE


def test_from_message_parses_and_decodes_nvl5_dmesg_event() -> None:
    """Verify an NVLink 5 event is parsed and resolved through its decode rule."""

    xid_analyzer.setup_rules()
    entry = xid_analyzer.XIDEntry.from_message(
        message=(
            "NVRM: Xid (PCI:0000:ae:00): 145, pid=1, name=app, RLW_CTRL NonFatal Local No Link 2 (0x62 0x80000000"
        ),
        timestamp="2024-01-01 10:00:00",
        driver=_DRIVER,
        machine=_MACHINE,
    )

    assert entry is not None
    assert entry.subcode == "RLW_CTRL"
    assert entry.intrinfo == 0x62
    assert entry.errorstatus == 0x80000000

    entry.lookup_resolution()

    assert entry.resolution == "IGNORE"
    assert entry.investigatory_action == "TBD"
    assert entry.job_failure_severity == "Job-Non-Fatal"
    assert entry.hw_sw_classification == "SW"
    assert entry.decoded_xid_string == "145.0h"
    assert entry.nvl5_error_code == 0x62


def test_from_message_rejects_non_xid_text() -> None:
    """Verify unrelated log messages do not produce analyzer entries."""

    assert xid_analyzer.XIDEntry.from_message("GPU is healthy") is None


def test_missing_catalog_action_remains_unset():
    """Verify JSON null actions do not become truthy NaN values."""

    assert xid_analyzer._restore_missing(None) is None


def test_catalog_actions_without_documented_recovery_resolve_to_tbd() -> None:
    """Verify support-contact and undocumented actions resolve to ``TBD``."""

    data = xid_analyzer.load_xid_catalog_data()
    catalog = data["xid_catalog"]
    decode_rules = data["nvl5_xid_decoding_rules"]

    assert catalog["79"]["recovery_action"] == "RESTART_BM"
    assert catalog["79"]["investigatory_action"] == "TBD"
    assert catalog["108"]["recovery_action"] == "IGNORE"
    assert catalog["108"]["investigatory_action"] == "XID_137_FLOW"
    assert all(
        entry["recovery_action"] == "TBD" and entry["investigatory_action"] == "TBD"
        for entry in catalog.values()
        if entry["type"] == "SXID"
    )
    assert all(
        action != "CONTACT_SUPPORT"
        for entry in catalog.values()
        for action in (entry["recovery_action"], entry["investigatory_action"])
    )
    assert all(
        action != "CONTACT_SUPPORT"
        for rules in decode_rules.values()
        for rule in rules
        for action in (rule["recovery_action"], rule["investigatory_action"])
    )


def test_lookup_resolution_fills_catalog_fields():
    """Verify catalog lookup populates the resolution and decoded fields."""

    xid_analyzer.setup_rules()
    entry = _entry(_INTERRUPT, "2024-01-01 10:00:00")

    entry.lookup_resolution()

    catalog = xid_analyzer.XID_CATALOG[str(_INTERRUPT)]
    assert entry.resolution == catalog["recovery_action"]
    assert entry.mnemonic == catalog["mnemonic"]
    assert entry.decoded_xid_string == str(_INTERRUPT)


def test_lookup_resolution_decodes_xid_154_driver_action():
    """Verify the embedded analyzer preserves XID 154's dynamic action."""

    xid_analyzer.setup_rules()
    entry = xid_analyzer.XIDEntry.from_message(
        message=(
            "NVRM: Xid (PCI:0009:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x4 (Drain and Reset)"
        ),
        timestamp="2024-01-01 10:00:00",
        driver=_DRIVER,
        machine=_MACHINE,
    )
    assert entry is not None

    entry.lookup_resolution()

    assert entry.resolution == "Drain and Reset"


def test_xid_154_decoder_ignores_later_parenthesized_text():
    """Verify the dynamic action is bound to the transition, not trailing text."""

    message = (
        "GPU recovery action changed from 0x0 (None) to 0x4 (Drain and Reset) followed by unrelated detail (ignored)"
    )

    assert xid_analyzer.get_resolution_for_xid154(message) == "Drain and Reset"


def test_xid_154_without_dynamic_action_uses_fallback_resolution() -> None:
    """Verify malformed XID 154 messages receive the documented fallback."""

    xid_analyzer.setup_rules()
    entry = _entry(154, "2024-01-01 10:00:00")

    entry.lookup_resolution()

    assert entry.resolution == "No resolution found for XID_154"


def test_fleet_backfills_driver_version_and_drops_duplicate_entries() -> None:
    """Verify known node versions are applied before duplicate suppression."""

    xid_analyzer.setup_rules()
    fleet = xid_analyzer.FleetData()
    known_version = _entry(_NON_STICKY, "2024-01-01 10:00:00", driver=_DRIVER)
    missing_version = _entry(31, "2024-01-01 10:01:00", driver="unknown")

    fleet._add_xid_entry(known_version, drop_duplicates=True)
    fleet._add_xid_entry(missing_version, drop_duplicates=True)
    fleet._add_xid_entry(missing_version, drop_duplicates=True)

    assert missing_version.driver == _DRIVER
    assert len(fleet.nodes) == 1
    assert fleet.nodes[0].xid_entries == [known_version, missing_version]
    assert fleet.driver_versions == {_DRIVER}


def test_xids_within_three_minutes_form_one_burst():
    """Verify nearby non-sticky XIDs form one burst."""

    xid_analyzer.setup_rules()
    fleet = _analyze(
        [
            _entry(_NON_STICKY, "2024-01-01 10:00:00"),
            _entry(_NON_STICKY, "2024-01-01 10:02:00"),
        ]
    )

    bursts = fleet.nodes[0].xid_bursts
    assert len(bursts) == 1
    assert bursts[0].xids == {str(_NON_STICKY)}


def test_xids_more_than_three_minutes_apart_form_separate_bursts():
    """Verify separated non-sticky XIDs form distinct bursts."""

    xid_analyzer.setup_rules()
    fleet = _analyze(
        [
            _entry(_NON_STICKY, "2024-01-01 10:00:00"),
            _entry(31, "2024-01-01 10:04:00"),
        ]
    )

    bursts = fleet.nodes[0].xid_bursts
    assert len(bursts) == 2
    assert bursts[0].xids == {str(_NON_STICKY)}
    assert bursts[1].xids == {"31"}


def test_driver_change_starts_new_burst_within_three_minutes():
    """Verify a driver change starts a new burst despite close timestamps."""

    xid_analyzer.setup_rules()
    fleet = _analyze(
        [
            _entry(_NON_STICKY, "2024-01-01 10:00:00", driver="1.0.0"),
            _entry(_NON_STICKY, "2024-01-01 10:01:00", driver="2.0.0"),
        ]
    )

    assert len(fleet.nodes[0].xid_bursts) == 2


def test_sticky_xid_stays_in_same_burst_across_three_minute_gap():
    """Sticky codes within a 3-hour window do not open a new burst."""
    xid_analyzer.setup_rules()
    fleet = _analyze(
        [
            _entry(_STICKY, "2024-01-01 10:00:00"),
            _entry(_STICKY, "2024-01-01 10:05:00"),
            _entry(_NON_STICKY, "2024-01-01 10:05:30"),
        ]
    )

    bursts = fleet.nodes[0].xid_bursts
    assert len(bursts) == 1
    assert bursts[0].xids == {str(_STICKY), str(_NON_STICKY)}


def test_sticky_xid_opens_new_burst_after_three_hours():
    """Verify a sticky XID starts a new burst after its 3-hour window."""

    xid_analyzer.setup_rules()
    fleet = _analyze(
        [
            _entry(_STICKY, "2024-01-01 10:00:00"),
            _entry(_STICKY, "2024-01-01 13:30:00"),
        ]
    )

    bursts = fleet.nodes[0].xid_bursts
    assert len(bursts) == 2
    assert all(burst.xids == {str(_STICKY)} for burst in bursts)


def test_analysis_pipeline_classifies_interrupt_burst():
    """Verify the analysis pipeline classifies an interrupting XID burst."""

    xid_analyzer.setup_rules()
    fleet = _analyze([_entry(_INTERRUPT, "2024-01-01 10:00:00")])

    assert len(fleet.nodes) == 1
    bursts = fleet.nodes[0].xid_bursts
    assert len(bursts) == 1
    burst = bursts[0]
    assert burst.xids == {str(_INTERRUPT)}
    assert burst.category
    assert burst.is_interruption is True
    assert fleet.nodes[0].interruptions_count == 1


def test_burst_pattern_rejects_unlisted_xids() -> None:
    """Verify wildcard exclusions allow declared XIDs and reject extras."""

    pattern = xid_analyzer.XIDBurstPattern(
        SET_XIDS_MUST_EQUAL={"13"},
        SET_XIDS_MUST_INTERSECT={"31"},
        SET_XIDS_MUST_NOT_INTERSECT={"*"},
        SET_XIDS_MAY_INTERSECT={"43"},
    )
    burst = xid_analyzer.XIDBurst(
        xids={"13", "31", "43"},
        xid_list=[],
        decoded_xid_set=set(),
        count=3,
        sources=[],
        is_interruption=False,
        category="",
        sub_category="",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 1),
        vbios_version="",
        driver_version=_DRIVER,
    )
    assert pattern.Matches(burst) is True

    burst.xids.add("79")
    assert pattern.Matches(burst) is False
