# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass, field
from typing import Any

from attribution_lib.declarative.errors import LoadError

"""Parse ``[catalog.<name>]`` element catalogs and ``[rows.<name>]`` record
tables from a declarative rule doc into engine-side registries.
"""


@dataclass
class Tables:
    """An *element catalog* maps a lookup key to a leaf-element dict (e.g. a
    ``bad_node`` template); a *record table* maps a lookup key to an arbitrary
    record dict. Keys are ``str()``-coerced so TOML string keys and any int keys a
    Python provider might supply compare consistently. A single name may not be
    defined as both a catalog and a record table.
    """

    element_catalogs: dict[str, dict[str, Any]] = field(default_factory=dict)
    record_tables: dict[str, dict[str, Any]] = field(default_factory=dict)


def _coerce_keys(section: str, name: str, rows: Any) -> dict[str, Any]:
    """``str()``-coerce the keys of ``rows``, raising :class:`LoadError` when
    ``rows`` is not a mapping or when two distinct keys collide after coercion."""
    if not isinstance(rows, dict):
        raise LoadError(f"[{section}.{name}] must be a table, got {type(rows).__name__}")
    out: dict[str, Any] = {}
    for k, v in rows.items():
        sk = str(k)
        if sk in out:
            raise LoadError(f"[{section}.{name}]: keys {k!r} and an earlier key both coerce to {sk!r}")
        if not isinstance(v, dict):
            raise LoadError(f"[{section}.{name}].{sk} must be a table, got {type(v).__name__}")
        out[sk] = v
    return out


def parse_tables(doc: dict[str, Any]) -> Tables:
    """Build a ``Tables`` from a declarative doc's ``[catalog.*]`` and ``[rows.*]`` sections.

    Keys are ``str()``-coerced so TOML integer keys and Python string keys compare
    consistently. Raises :class:`LoadError` if a name appears in both sections, if a
    section value is not a mapping, or if two keys in one table collide after coercion.
    """
    t = Tables()
    catalog_raw = doc.get("catalog", {})
    if not isinstance(catalog_raw, dict):
        raise LoadError(f"[catalog] must be a table, got {type(catalog_raw).__name__}")
    for name, rows in catalog_raw.items():
        t.element_catalogs[name] = _coerce_keys("catalog", name, rows)
    rows_raw = doc.get("rows", {})
    if not isinstance(rows_raw, dict):
        raise LoadError(f"[rows] must be a table, got {type(rows_raw).__name__}")
    for name, rows in rows_raw.items():
        if name in t.element_catalogs:
            raise LoadError(f"table name {name!r} defined as both [catalog.{name}] and [rows.{name}]")
        t.record_tables[name] = _coerce_keys("rows", name, rows)
    return t


def resolve_catalog_table(
    emit: dict[str, Any],
    tables: Tables,
    *,
    rule_name: str,
) -> tuple[dict[str, Any], bool]:
    """Resolve a ``catalog_lookup`` emit's ``catalog`` name to its backing table.

    Returns ``(table, is_element)`` where ``table`` is the ``{str-key -> row}`` dict
    and ``is_element`` is True for an element catalog, False for a record table.
    Raises :class:`LoadError` when the name resolves to neither. Shared by the
    compiler, the coverage checker, and the declaration collector so they all
    resolve a name identically.
    """
    name = emit.get("catalog")
    if name is None:
        raise LoadError(f"{rule_name}: catalog_lookup is missing the required 'catalog' field")
    is_element = name in tables.element_catalogs
    is_record = name in tables.record_tables
    if not (is_element or is_record):
        raise LoadError(f"{rule_name}: catalog_lookup references unknown table {name!r}")
    table = tables.element_catalogs[name] if is_element else tables.record_tables[name]
    return table, is_element
