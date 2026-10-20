# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from attribution_blueprint.webui.diagnose import _load_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "src" / "attribution_blueprint" / "data" / "failures.yml"


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently keeping the last."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _catalog_entries() -> list[Any]:
    """Top-level catalog entries, read straight from the YAML list."""
    data = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{CATALOG_PATH.name} must parse to a mapping, got {type(data).__name__}"
    entries = data.get("failures")
    assert isinstance(entries, list), f"{CATALOG_PATH.name} must hold a top-level 'failures' list"
    return entries


def _entry_label(index: int, entry: Any) -> str:
    """Human-locatable label for an entry, e.g. `failures[2] (XID 79)`."""
    if isinstance(entry, dict):
        title = entry.get("title")
        if isinstance(title, str) and title != "":
            return f"failures[{index}] ({title})"
    return f"failures[{index}]"


def _entry_ids(entries: list[Any]) -> list[Any]:
    """One id per entry, None where the entry is not a mapping or has no id."""
    return [entry.get("id") if isinstance(entry, dict) else None for entry in entries]


def _duplicate_id_messages(entries: list[Any]) -> list[str]:
    """One message per id that more than one entry claims."""
    ids = _entry_ids(entries)
    repeated = {value for value, count in Counter(ids).items() if count > 1}
    messages = []
    for value in repeated:
        where = ", ".join(f"failures[{i}]" for i, other in enumerate(ids) if other == value)
        messages.append(f"{value!r} is claimed by {where}")
    return sorted(messages)


def test_every_catalog_entry_has_a_non_empty_string_id() -> None:
    """Every catalog entry carries a non-empty string id."""
    offenders = []
    for index, entry in enumerate(_catalog_entries()):
        if not isinstance(entry, dict):
            offenders.append(f"{_entry_label(index, entry)} is {type(entry).__name__}, not a mapping")
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or entry_id == "":
            offenders.append(f"{_entry_label(index, entry)} has id {entry_id!r}")
    assert offenders == [], "catalog entries without a usable id:\n" + "\n".join(offenders)


def test_catalog_entry_ids_are_unique() -> None:
    """No two catalog entries share an id."""
    duplicates = _duplicate_id_messages(_catalog_entries())
    assert duplicates == [], "duplicate catalog ids:\n" + "\n".join(duplicates)


def test_catalog_entry_count_matches_loaded_catalog_size() -> None:
    """Every list entry survives the id-keyed dict the persona card is looked up in."""
    _load_catalog.cache_clear()
    entries = _catalog_entries()
    entry_count = len(entries)
    reachable_count = len(_load_catalog())
    detail = "\n".join(_duplicate_id_messages(entries)) or "an entry has a missing or empty id"
    assert entry_count == reachable_count, (
        f"catalog has {entry_count} entries but only {reachable_count} are reachable by id:\n{detail}"
    )


def test_catalog_yaml_has_no_duplicate_keys_within_an_entry() -> None:
    """A dropped `- id:` marker folds an entry into its neighbour, which YAML accepts silently."""
    yaml.load(CATALOG_PATH.read_text(encoding="utf-8"), Loader=_StrictLoader)


def test_xid_38_remediation_stays_undeclared_pending_review() -> None:
    """Do not automate the unresolved XID 38 ignore-versus-reset decision."""
    entry = next(entry for entry in _catalog_entries() if entry["id"] == "xid-38")
    assert "recommended_action" not in entry
    action = entry["personas"]["tenant_sre"]["action"]
    assert "Retry on a fresh allocation" in action
    assert "avoid that node" in action


def test_xid_79_resets_gpu_before_escalating() -> None:
    """Keep the XID 79 sequence and tenant guidance aligned with the RFC."""
    entry = next(entry for entry in _catalog_entries() if entry["id"] == "xid-79")
    assert entry["recommended_action"] == {
        "operational_state": "drain",
        "remediation_actions": ["reset_gpu", "service_required"],
    }
    action = entry["personas"]["tenant_sre"]["action"]
    assert "nvidia-smi --gpu-reset" in action
    assert "restart the affected node's operating system" not in action


def test_xid_95_resets_gpu_without_changing_xid_94() -> None:
    """Follow the public XID catalog: RESET_GPU for XID 95, RESTART_APP for XID 94."""
    entries = {entry["id"]: entry for entry in _catalog_entries()}
    assert entries["xid-95"]["recommended_action"] == {
        "operational_state": "drain",
        "remediation_actions": ["reset_gpu"],
    }
    assert entries["xid-94"]["recommended_action"]["remediation_actions"] == ["restart_app", "service_required"]
    assert "recommended_action" not in entries["xid-94-95"]
    action = entries["xid-95"]["personas"]["tenant_sre"]["action"]
    assert "nvidia-smi --gpu-reset" in action
    assert "restart the affected node's operating system" not in action
    xid_pattern = next(signal["pattern"] for signal in entries["xid-95"]["signals"] if signal["kind"] == "xid")
    assert re.search(xid_pattern, "NVRM: Xid (PCI:0000:01:00): 95, uncontained error") is not None
    assert re.search(xid_pattern, "NVRM: Xid (PCI:0000:01:00): 94, contained error") is None


def test_nvlink5_declarations_follow_the_public_decode_table() -> None:
    """Declare only XIDs whose every error status has the same action; the rest depend on the decode."""
    entries = {entry["id"]: entry for entry in _catalog_entries()}
    for entry_id in ("workflow-nvlink5-err", "nvlink5-errors", "xid-144", "xid-145", "xid-146", "xid-149"):
        assert "recommended_action" not in entries[entry_id], entry_id
    for entry_id in ("xid-147", "xid-148"):
        assert entries[entry_id]["recommended_action"] == {"operational_state": "ready"}, entry_id
    assert entries["xid-150"]["recommended_action"] == {
        "operational_state": "drain",
        "remediation_actions": ["reset_gpu", "service_required"],
    }
    link_down = next(signal["pattern"] for signal in entries["link-down-nvlink"]["signals"] if signal["kind"] == "xid")
    assert re.search(link_down, "74") is not None
    assert all(re.search(link_down, str(xid)) is None for xid in range(144, 151))


@pytest.fixture(scope="module")
def catalog_by_id() -> dict[str, Any]:
    return {entry["id"]: entry for entry in _catalog_entries()}


@pytest.mark.parametrize("entry_id,vendor", [("ib-error-12", "0"), ("ib-error-12-vendor-129", "129")])
@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("status", ["12", "IBV_WC_RETRY_EXC_ERR(12)"])
def test_irrecoverable_ib_patterns_require_exact_status_and_vendor(
    catalog_by_id: dict[str, Any], entry_id: str, vendor: str, direction: str, status: str
) -> None:
    patterns = [
        signal["pattern"]
        for signal in catalog_by_id[entry_id]["signals"]
        if "Irrecoverable" in signal["pattern"] and "vendor" in signal["pattern"]
    ]
    assert len(patterns) == 1
    pattern = re.compile(patterns[0])
    prefix = f"Irrecoverable {direction} communicator error status={status} vendor_err="
    for suffix in ("", " ", ", localGid=::ffff:192.0.2.1"):
        assert pattern.search(prefix + vendor + suffix)
    for other_vendor in (vendor + "0", vendor + "1", vendor + "abc", "130"):
        assert pattern.search(prefix + other_vendor) is None
    for other_status in ("120", "112", "IBV_WC_WR_FLUSH_ERR(5)"):
        line = f"Irrecoverable {direction} communicator error status={other_status} vendor_err={vendor}"
        assert pattern.search(line) is None
    details = catalog_by_id[entry_id]["personas"]["tenant_sre"]["support_details"]
    assert "localGid" in details and "remoteGid" in details


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_irrecoverable_ib_status_only_pattern(catalog_by_id: dict[str, Any], direction: str) -> None:
    """Cover the named retry-exceeded status without requiring a vendor syndrome."""
    patterns = [
        signal["pattern"]
        for signal in catalog_by_id["ib-error-12"]["signals"]
        if "Irrecoverable" in signal["pattern"] and "status" in signal["pattern"] and "vendor" not in signal["pattern"]
    ]
    assert len(patterns) == 1
    pattern = re.compile(patterns[0])
    prefix = f"NCCL WARN NET/IB: Irrecoverable {direction} communicator error "
    for separator in ("=", " "):
        assert pattern.search(prefix + f"status{separator}IBV_WC_RETRY_EXC_ERR(12)")
    for other_status in ("IBV_WC_RETRY_EXC_ERR(120)", "IBV_WC_WR_FLUSH_ERR(5)"):
        assert pattern.search(prefix + f"status={other_status}") is None
    assert pattern.search("status=IBV_WC_RETRY_EXC_ERR(12)") is None


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_irrecoverable_ib_peer_metadata_pattern(catalog_by_id: dict[str, Any], direction: str) -> None:
    """Require both peer GIDs and the HCA in the separate metadata pattern."""
    patterns = [
        signal["pattern"]
        for signal in catalog_by_id["ib-error-12"]["signals"]
        if "Irrecoverable" in signal["pattern"] and "localGid" in signal["pattern"]
    ]
    assert len(patterns) == 1
    pattern = re.compile(patterns[0])
    prefix = f"NCCL WARN NET/IB: Irrecoverable {direction} communicator error "
    for separator in (":", " "):
        metadata = f"localGid{separator}::ffff:192.0.2.1 remoteGid{separator}::ffff:192.0.2.2 hca=mlx5_0"
        assert pattern.search(prefix + metadata)
        for field in ("localGid", "remoteGid", "hca"):
            assert pattern.search(prefix + metadata.replace(field, "missing_field")) is None
        assert pattern.search(metadata) is None


def test_legacy_ib_completion_requires_exact_values(catalog_by_id: dict[str, Any]) -> None:
    pattern = next(
        signal["pattern"]
        for signal in catalog_by_id["ib-error-12-vendor-129"]["signals"]
        if signal["pattern"].startswith("Got completion")
    )
    assert re.search(pattern, "Got completion with error 12 vendor err 129")
    assert re.search(pattern, "Got completion with error 120 vendor err 129") is None
    assert re.search(pattern, "Got completion with error 12 vendor err 1290") is None


@pytest.mark.parametrize(
    "source,missing,user_match,system_match",
    [
        ("/home/example/model", False, True, False),
        ("/lustre/example/model", False, True, False),
        ("/etc/resolv.conf", False, False, True),
        ("/opt/example/model", True, False, False),
    ],
)
def test_pyxis_mount_signals_distinguish_paths(
    catalog_by_id: dict[str, Any], source: str, missing: bool, user_match: bool, system_match: bool
) -> None:
    entry = catalog_by_id["pyxis-enroot-mount-failure"]
    user_pattern, system_pattern = [signal["pattern"] for signal in entry["signals"][:2]]
    line = f"enroot-mount: failed to mount: {source} at /tmp/container/path"
    if missing:
        line += ": No such file or directory"
    assert bool(re.search(user_pattern, line)) is user_match
    assert bool(re.search(system_pattern, line)) is system_match
    assert entry["likely_owner"] == "ambiguous"
    assert "Do not change system directories" in entry["personas"]["tenant_sre"]["action"]


@pytest.mark.parametrize("entry_id", ["broken-pipe", "connection-timed-out"])
def test_socket_guidance_does_not_assign_infrastructure_cause(catalog_by_id: dict[str, Any], entry_id: str) -> None:
    entry = catalog_by_id[entry_id]
    assert entry["likely_owner"] == "ambiguous"
    assert "Fail MultiCategory" in entry["technical_details"]
    assert "LOW_GENERIC" in entry["technical_details"]
    assert "cloud provider" in entry["personas"]["tenant_sre"]["action"]


def test_nccl_override_guidance_describes_text_detection(catalog_by_id: dict[str, Any]) -> None:
    entry = catalog_by_id["call-to-ibv-modify-qp-failed-eno-110-timeout-overwrite-nccl-parameters"]
    assert entry["likely_owner"] == "user"
    assert entry["aliases"] == ["Call to ibv_modify_qp failed: eno 110 timeout(Overwrite NCCL parameters)"]
    details = entry["technical_details"]
    assert "substring check, not shell evaluation" in details
    assert "`NCCL_IB_MQP_RETRY_ALL` is not one of the names checked" in details
    assert "does not prove that a variable was assigned" in details
    assert "Fail User" in details and "Fail InfraHW" in details


def test_reboot_guidance_distinguishes_initiated_and_unexpected(catalog_by_id: dict[str, Any]) -> None:
    entry = catalog_by_id["node-unexpectedly-rebooted"]
    assert "Node reboot initiated" in entry["aliases"]
    details = entry["technical_details"]
    assert "Fail InfraSW" in details and "Fail InfraHW" in details
    assert "explicit initiated signal is not evidence of an unexpected reset" in details
    assert "do not reboot shared nodes yourself" in entry["personas"]["tenant_sre"]["action"]


def test_scheduler_gpu_loss_is_not_a_hardware_diagnosis(catalog_by_id: dict[str, Any]) -> None:
    entry = catalog_by_id["gpu-is-lost-reported-by-slurm"]
    assert entry["likely_owner"] == "ambiguous"
    assert "does not establish XID 79 or a hardware fault" in entry["personas"]["tenant_sre"]["summary"]
    assert "optional corroborating" in next(signal["note"] for signal in entry["signals"] if signal["kind"] == "xid")
    assert "cloud provider" in entry["personas"]["tenant_sre"]["action"]


@pytest.mark.parametrize(
    "entry_id",
    [
        "broken-pipe",
        "connection-timed-out",
        "pyxis-enroot-mount-failure",
        "gpu-is-lost-reported-by-slurm",
        "node-unexpectedly-rebooted",
        "call-to-ibv-modify-qp-failed-eno-110-timeout-overwrite-nccl-parameters",
    ],
)
def test_conditional_entries_do_not_declare_blanket_recovery(catalog_by_id: dict[str, Any], entry_id: str) -> None:
    assert "recommended_action" not in catalog_by_id[entry_id]


def test_xid_32_keeps_restart_and_application_investigation(catalog_by_id: dict[str, Any]) -> None:
    entry = catalog_by_id["xid-32"]
    assert entry["recommended_action"]["remediation_actions"] == ["restart_app", "service_required"]
    assert "CHECK_APP/CUDA" in entry["technical_details"]
    assert "Fail MultiCategory" in entry["technical_details"]
    assert "restart the affected application" in entry["personas"]["tenant_sre"]["action"]
    assert "does not prove" in entry["personas"]["tenant_sre"]["impact"]


@pytest.mark.parametrize("entry_id", ["nvlink5-errors", "workflow-nvlink5-err"])
def test_nvlink_family_guidance_requires_event_decode(catalog_by_id: dict[str, Any], entry_id: str) -> None:
    entry = catalog_by_id[entry_id]
    assert "intrInfo" in entry["technical_details"] and "errorStatus" in entry["technical_details"]
    assert "If the job failed" in entry["personas"]["tenant_sre"]["action"]
    assert "cloud provider" in entry["personas"]["tenant_sre"]["action"]


@pytest.mark.parametrize(
    "entry_id",
    [
        "ib-error-12",
        "ib-error-12-vendor-129",
        "connection-timed-out",
        "broken-pipe",
        "xid-32",
        "gpu-is-lost-reported-by-slurm",
        "call-to-ibv-modify-qp-failed-eno-110-timeout-overwrite-nccl-parameters",
        "workflow-nvlink5-err",
        "nvlink5-errors",
        "pyxis-enroot-mount-failure",
        "node-unexpectedly-rebooted",
    ],
)
def test_corrected_entries_include_diagnosis_in_support_details(catalog_by_id: dict[str, Any], entry_id: str) -> None:
    """Keep the diagnosis with symptom evidence in the tenant handoff field."""
    details = catalog_by_id[entry_id]["personas"]["tenant_sre"]["support_details"]
    for part in ("NVIDIA Attribution diagnosis", "verdict", "symptom chain", "source logs"):
        assert part in details


def test_catalog_has_no_private_personas_or_notes(catalog_by_id: dict[str, Any]) -> None:
    for entry in catalog_by_id.values():
        assert "internal_notes" not in entry, entry["id"]
        assert set(entry["personas"]) == {"tenant_sre"}, entry["id"]


def test_related_catalog_ids_exist(catalog_by_id: dict[str, Any]) -> None:
    for entry in catalog_by_id.values():
        assert set(entry.get("related", [])) <= catalog_by_id.keys(), entry["id"]
