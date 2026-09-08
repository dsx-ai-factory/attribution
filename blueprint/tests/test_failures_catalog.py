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
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from attribution_blueprint.webui.diagnose import _load_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "src" / "attribution_blueprint" / "data" / "failures.yml"

_PROHIBITED_NVIDIA_ROUTING_PATTERNS = (
    re.compile(r"\bescalat(?:e|es|ed|ing|ion)\b[^.!?\n]*?\b(?:to|with)\s+NVIDIA\b", re.IGNORECASE),
    re.compile(
        r"\b(?:file|open|submit)\s+(?:an?\s+)?(?:bug|issue|ticket)\s+(?:to|with)\s+NVIDIA\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsupport\s+case\s+(?:to|with)\s+NVIDIA\b|\bNVIDIA\s+support\b", re.IGNORECASE),
)


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


def _string_values(value: Any) -> list[str]:
    """Return every string nested below a catalog value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for child in value.values() for text in _string_values(child)]
    if isinstance(value, list):
        return [text for child in value for text in _string_values(child)]
    return []


def _has_prohibited_nvidia_routing(text: str) -> bool:
    """Return whether text sends users to an NVIDIA-specific support path."""
    return any(pattern.search(text) for pattern in _PROHIBITED_NVIDIA_ROUTING_PATTERNS)


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


@pytest.mark.parametrize(
    "text",
    [
        "Escalate to NVIDIA.",
        "The failure requires escalation to NVIDIA.",
        "The issue was escalated to NVIDIA.",
        "File a bug with NVIDIA.",
        "File an issue with NVIDIA.",
        "Open a ticket with NVIDIA.",
        "Open a support case with NVIDIA.",
        "Investigate via NVIDIA support.",
    ],
)
def test_prohibited_nvidia_routing_variants_are_detected(text: str) -> None:
    """The guard covers grammatical variants instead of exact phrases only."""
    assert _has_prohibited_nvidia_routing(text)


def test_catalog_does_not_use_nvidia_specific_support_routing() -> None:
    """Public catalog prose sends evidence to generic public support paths."""
    offenders = []
    for index, entry in enumerate(_catalog_entries()):
        if not isinstance(entry, dict):
            continue
        for text in _string_values(entry):
            if _has_prohibited_nvidia_routing(text):
                offenders.append(f"{_entry_label(index, entry)}: {text}")
    assert offenders == [], "NVIDIA-specific support routing found:\n" + "\n".join(offenders)
