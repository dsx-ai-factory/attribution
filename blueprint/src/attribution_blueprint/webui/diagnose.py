# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failures-catalog lookup and the persona-aware Diagnose card.

This is the web-server-free logic layer of the webui. It depends only on the
read-only DB row shape and PyYAML (to read the vendored ``failures.yml``
catalog) — never on FastAPI/uvicorn. Both the payload builder (``bootstrap``)
and the HTTP layer (``app``) import ``diagnose_card`` from here, so the
dependency direction is strictly web -> logic with no cycle.
"""

from __future__ import annotations

import functools
from importlib.resources import files as _pkg_files
from typing import Any

import yaml

from attribution_blueprint.webui.db import RunRow

# Catalog of failure modes (vendored from the failures catalog). Indexed
# by entry id (e.g., "xid-79"). The webui resolves a run's verdict to a
# catalog entry via `verdict.root_symptom.catalog_id`, which the upstream
# attribution service emits when it knows the mapping. Some historical demo
# rows predate catalog IDs, so keep narrow rule-id fallbacks here for those
# rows until the taxonomy layer emits catalog IDs for them.
#
# The catalog lives inside the package (`attribution_blueprint/data/failures.yml`)
# so it resolves identically under editable installs, non-editable installs,
# and inside the Docker image where the source tree is not on disk.
_CATALOG_RESOURCE = _pkg_files("attribution_blueprint") / "data" / "failures.yml"
_ROOT_RULE_CATALOG_FALLBACKS = {
    "UnsquashfsWriteFailed": "pyxis-couldnt-start-container",
}


@functools.lru_cache(maxsize=1)
def _load_catalog() -> dict[str, dict]:
    """Load the failures catalog YAML. Cached per process.

    Returns an empty dict if the catalog resource is missing, unreadable,
    or the YAML fails to parse, rather than raising. A raise here would
    propagate through `lru_cache` and poison every subsequent call until
    the process restarts, breaking `/run/{id}` and `/run/{id}/ncp-payload.md`
    for the rest of the process lifetime. `OSError` is caught because
    `is_file()` and `open()` are not atomic, so the file can become
    unreadable in the gap. Persona rendering degrades gracefully when the
    catalog is unavailable, which is the desired failure mode here.
    """
    if not _CATALOG_RESOURCE.is_file():
        return {}
    try:
        with _CATALOG_RESOURCE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        return {}
    entries = data.get("failures") or []
    return {e["id"]: e for e in entries if isinstance(e, dict) and e.get("id")}


def _catalog_id_for_run(run: RunRow) -> str | None:
    """Resolve the failure-catalog ID for a run.

    Args:
        run: Read-only web UI database row.

    Returns:
        The explicit catalog ID, a known historical rule fallback, or None.
    """
    verdict = (run.raw_result or {}).get("verdict") or {}
    root = verdict.get("root_symptom") or {}
    explicit = root.get("catalog_id")
    if isinstance(explicit, str) and explicit != "":
        return explicit
    source = root.get("source") or {}
    rule = source.get("rule")
    if isinstance(rule, str) and rule != "":
        return _ROOT_RULE_CATALOG_FALLBACKS.get(rule)
    return None


def _catalog_entry_for_run(run: RunRow) -> dict | None:
    """Resolve a catalog entry for a run.

    Args:
        run: Read-only web UI database row.

    Returns:
        The matching failure-catalog entry, or None.
    """
    catalog = _load_catalog()
    catalog_id = _catalog_id_for_run(run)
    if catalog_id in catalog:
        return catalog[catalog_id]
    return None


def _split_support_details(support_raw: Any) -> list[str]:
    """Normalize support details without splitting sentences or numbered steps.

    Args:
        support_raw: Raw ``support_details`` value from the catalog.

    Returns:
        Non-empty support-detail strings, preserving punctuation and line breaks.
    """
    if isinstance(support_raw, str):
        return [support_raw.strip()] if support_raw.strip() != "" else []
    if isinstance(support_raw, list):
        return [s.strip() for s in support_raw if isinstance(s, str) and s.strip() != ""]
    return []


def diagnose_card(run: RunRow) -> dict[str, Any]:
    """Assemble tenant guidance for the verdict page and evidence export.

    Actions and escalation evidence both come from ``tenant_sre``. Never fall
    back to operator guidance or internal notes when tenant fields are missing.
    """
    entry = _catalog_entry_for_run(run)
    verdict = (run.raw_result or {}).get("verdict") or {}
    root = verdict.get("root_symptom") or {}
    raw_symptom = root.get("symptom") or ""
    raw_line = (root.get("source") or {}).get("line") or ""

    if entry is None:
        return {
            "run_id": run.run_id,
            "has_persona": False,
            "title": raw_symptom or "Unclassified failure",
            "summary": verdict.get("detail") or "",
            "impact": "",
            "action": "",
            "support_details": [],
            "category": None,
            "severity": None,
            "raw_symptom": raw_symptom,
            "raw_line": raw_line,
            "catalog_id": None,
        }

    personas = entry.get("personas") or {}
    tenant = personas.get("tenant_sre") or {}

    return {
        "run_id": run.run_id,
        "has_persona": len(tenant) > 0,
        "title": tenant.get("title") or entry.get("title") or raw_symptom,
        "summary": tenant.get("summary") or entry.get("summary") or "",
        "impact": tenant.get("impact") or "",
        "action": tenant.get("action") or "",
        "support_details": _split_support_details(tenant.get("support_details")),
        "category": entry.get("category"),
        "severity": entry.get("severity"),
        "raw_symptom": raw_symptom,
        "raw_line": raw_line,
        "catalog_id": entry.get("id"),
    }
