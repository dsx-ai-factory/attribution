# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stamp ``[[family]]`` stanzas into concrete generated rules.

A family is a compact way to spell many near-identical extractors: one ``emit``
template plus one or more ``pattern``s, applied across a set of ``rows`` (inline
or drawn from a record table via ``rows_from``). Each row's columns are textually
substituted (``{col}`` -> ``str(row[col])``) into the pattern and the emit, and
the pattern is ``re.escape``-d so keyword literals with regex metacharacters match
verbatim. The loader compiles each :class:`GeneratedRule` into an ``ExtractorRule``.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from attribution_lib.declarative.catalogs import Tables
from attribution_lib.declarative.errors import LoadError

_COL = re.compile(r"\{([^{}]+)\}")


def substitute_columns(
    text: str,
    row: dict[str, Any],
    *,
    where: str,
) -> str:
    """Replace each ``{col}`` in ``text`` with ``str(row[col])``.

    Textual substitution (not simpleeval): a ``{col}`` naming a column absent from
    ``row`` raises :class:`LoadError`.

    Example:
        >>> substitute_columns("GPU {name} failed", {"name": "A100"}, where="test")
        'GPU A100 failed'
        >>> substitute_columns("{count} errors", {"count": 7}, where="test")
        '7 errors'
    """

    def sub(m: re.Match[str]) -> str:
        col = m.group(1)
        if col not in row:
            raise LoadError(f"{where}: family column {col!r} not in row (have {sorted(row)})")
        return str(row[col])

    return _COL.sub(sub, text)


@dataclass
class GeneratedRule:
    name: str
    match_re: str
    emit: dict[str, Any]
    credit_names: set[str] = field(default_factory=set)
    line: str = ""  # the substituted pattern text BEFORE re.escape (a literal
    # line this rule matches; used to synthesize coverage cases)
    row_key: str = ""  # the row's `key` (for `untested_rows` waivers)


def _subst_emit(
    emit: Any,
    row: dict[str, Any],
    *,
    where: str,
) -> Any:
    """Recursively textual-substitute ``{col}`` in every string field of a leaf emit dict.

    Example:
        >>> _subst_emit({"symptom": "GPU {name} lost"}, {"name": "A100"}, where="test")
        {'symptom': 'GPU A100 lost'}
        >>> _subst_emit("{msg}", {"msg": "ok"}, where="test")
        'ok'
    """
    match emit:
        case dict():
            return {
                substitute_columns(k, row, where=where) if isinstance(k, str) else k: _subst_emit(v, row, where=where)
                for k, v in emit.items()
            }
        case str():
            return substitute_columns(emit, row, where=where)
        case list():
            return [_subst_emit(v, row, where=where) for v in emit]
        case _:
            return emit


def _rows_for(family: dict[str, Any], tables: Tables) -> list[dict[str, Any]]:
    """Return the row list for a family stanza: inline ``rows`` or a ``rows_from`` record table.

    For a ``rows_from`` reference each record-table entry is expanded to
    ``{"key": k, **v}`` so downstream code can treat both sources uniformly."""
    if "rows" in family:
        raw = family["rows"]
        fname = family.get("name")
        if not isinstance(raw, list):
            raise LoadError(f"family {fname!r}: 'rows' must be a list of tables, got {type(raw).__name__!r}")
        for i, row in enumerate(raw):
            if not isinstance(row, dict):
                raise LoadError(f"family {fname!r}: 'rows[{i}]' must be a table, got {type(row).__name__!r}")
        return raw
    if "rows_from" not in family:
        raise LoadError(f"family {family.get('name')!r}: requires 'rows' or 'rows_from'")
    name = family["rows_from"]
    if name in tables.record_tables:
        fname = family.get("name")
        rows = []
        for k, v in tables.record_tables[name].items():
            if not isinstance(v, dict):
                raise LoadError(
                    f"family {fname!r}: rows_from {name!r} entry {k!r} must be a table, got {type(v).__name__!r}"
                )
            if "key" in v:
                raise LoadError(f"family {fname!r}: rows_from {name!r} entry {k!r} has a reserved 'key' field")
            rows.append({**v, "key": k})
        return rows
    raise LoadError(f"family {family.get('name')!r}: rows_from names unknown record table {name!r}")


def stamp_family(family: dict[str, Any], tables: Tables) -> list[GeneratedRule]:
    """Expand a ``[[family]]`` stanza into one :class:`GeneratedRule` per (row, pattern) pair.

    Each row's columns are textually substituted into the pattern (then
    ``re.escape``-d) and the emit dict, yielding a concrete extractor the loader
    compiles like any other ``[[extractor]]``. Raises :class:`LoadError` for
    missing required fields or duplicate generated names."""
    fname = family.get("name")
    if not fname:
        raise LoadError("[[family]] requires a 'name'")
    match family:
        case {"patterns": patterns} if isinstance(patterns, list):
            for j, pattern in enumerate(patterns):
                if not isinstance(pattern, str):
                    raise LoadError(
                        f"family {fname!r}: 'patterns[{j}]' must be a string, got {type(pattern).__name__!r}"
                    )
            single = False
        case {"patterns": patterns}:
            raise LoadError(f"family {fname!r}: 'patterns' must be a list of strings, got {type(patterns).__name__!r}")
        case {"pattern": str(pattern)}:
            patterns = [pattern]
            single = True
        case {"pattern": pattern}:
            raise LoadError(f"family {fname!r}: 'pattern' must be a string, got {type(pattern).__name__!r}")
        case _:
            raise LoadError(f"family {fname!r}: requires 'pattern' or 'patterns'")
    if "emit" not in family:
        raise LoadError(f"family {fname!r}: requires 'emit'")
    if not isinstance(family["emit"], dict):
        raise LoadError(f"family {fname!r}: 'emit' must be a table, got {type(family['emit']).__name__!r}")
    out, seen = [], set()
    for row in _rows_for(family, tables):
        if "key" not in row:
            raise LoadError(f"family {fname!r}: every row needs a 'key' column (row {row})")
        key = str(row["key"])
        for i, pat in enumerate(patterns):
            gname = f"{fname}:{key}" if single else f"{fname}:{key}:{i}"
            if gname in seen:
                raise LoadError(f"family {fname!r}: duplicate generated name {gname!r}")
            seen.add(gname)
            where = gname
            line = substitute_columns(pat, row, where=where)
            match_re = re.escape(line)
            emit = _subst_emit(family["emit"], row, where=where)
            credit = emit.get("derived_rule")
            out.append(
                GeneratedRule(
                    name=gname,
                    match_re=match_re,
                    emit=emit,
                    credit_names={credit} if credit else set(),
                    line=line,
                    row_key=key,
                )
            )
    return out
