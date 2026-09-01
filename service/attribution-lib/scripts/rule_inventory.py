#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect and compare the rules loaded by attribution-lib.

An inventory is captured through the production setup_rules() path and records
discovered artifacts, executable rules, declared rule names, and event types. Two
inventories can then be compared to report additions, removals, renames, and
signature changes. This is structural accounting, not behavioral equivalence.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import attribution_lib
from attribution_lib.core import get_bundled_rules_path, setup_rules
from attribution_lib.rule_types.definitions import (
    CollectorRule,
    ExtractorAccumulatorRule,
    ExtractorRule,
    FilterRule,
    RefinerRule,
    Rule,
)

SCHEMA_NAME = "attribution-rule-inventory"
SCHEMA_VERSION = 1
AUTO_PROJECTION = "auto"
RUNTIME_PROJECTION = "runtime/v1"
RULE_KINDS = (
    "extractor",
    "filter",
    "extractor_accumulator",
    "refiner",
    "collector",
)
RULE_TYPES: dict[str, type[Rule]] = {
    "extractor": ExtractorRule,
    "filter": FilterRule,
    "extractor_accumulator": ExtractorAccumulatorRule,
    "refiner": RefinerRule,
    "collector": CollectorRule,
}


def _runtime_schema(kind: str) -> str:
    """Return the representation-neutral runtime schema name for a rule kind."""
    return f"{kind}/runtime/v1"


def _python_schema(kind: str) -> str:
    """Return the Python-definition schema name for a rule kind."""
    return f"{kind}/python/v1"


def _runtime_data(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Project definition data onto the stable runtime fields for its kind."""
    fields = {
        "extractor": ("match_re", "sources"),
        "extractor_accumulator": ("match_re", "sources"),
        "filter": ("priority",),
        "refiner": ("priority",),
        "collector": ("params_model",),
    }[kind]
    return {field: data.get(field) for field in fields}


RuleProjector = Callable[[str, dict[str, Any]], dict[str, Any]]
RULE_PROJECTORS: dict[tuple[str, str], RuleProjector] = {
    **{(_python_schema(kind), _runtime_schema(kind)): _runtime_data for kind in RULE_KINDS},
    ("extractor/declarative/v1", "extractor/runtime/v1"): _runtime_data,
}


def _definition(kind: str, data: dict[str, Any], serialized_emit: str | None) -> dict[str, Any]:
    """Build a versioned Python or declarative definition for a loaded rule."""
    if kind != "extractor" or serialized_emit is None:
        return {"schema": _python_schema(kind), "data": data}
    descriptor = cast("dict[str, Any]", json.loads(serialized_emit))
    declarative_data = dict(data)
    if "processor" in descriptor:
        declarative_data.update(
            form="processor",
            processor=descriptor["processor"],
            derived_rules=descriptor.get("derived_rules", []),
            derived_event_types=descriptor.get("derived_event_types", []),
        )
    else:
        declarative_data.update(form="emit", emit=descriptor)
    return {"schema": "extractor/declarative/v1", "data": declarative_data}


def _artifact_id(path: str | Path, roots: Sequence[Path]) -> str:
    """Return a stable directory-indexed identifier for a rule artifact path."""
    resolved = Path(path).resolve()
    for index, root in enumerate(roots):
        try:
            return f"{index}:{resolved.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return resolved.as_posix()


def _rule_record(kind: str, rule: Rule, roots: Sequence[Path], declarative_emit: dict[str, str]) -> dict[str, Any]:
    """Serialize one registered rule into an inventory record."""
    record: dict[str, Any] = {
        "kind": kind,
        "name": rule.name,
        "author": rule.author,
        "tags": list(rule.tags),
        "location": {
            "file": _artifact_id(rule.location.file, roots),
            "line": rule.location.line,
        },
    }
    data: dict[str, Any] = {}
    if hasattr(rule, "match_re"):
        data["match_re"] = rule.match_re
    if hasattr(rule, "source_filters"):
        data["sources"] = sorted(source.value for source in rule.source_filters)
    if hasattr(rule, "priority"):
        data["priority"] = rule.priority
    if kind == "collector":
        model = cast(CollectorRule, rule).params_model
        data["params_model"] = None if model is None else f"{model.__module__}.{model.__qualname__}"
    record["definition"] = _definition(kind, data, declarative_emit.get(rule.name))
    return record


def dump_inventory(rules_dirs: Sequence[Path] | None = None) -> dict[str, Any]:
    """Load bundled rules through production setup and return a JSON-ready inventory."""
    roots = tuple(path.resolve() for path in (rules_dirs or [get_bundled_rules_path()]))
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"rules directory does not exist or is not a directory: {root}")
    discovered = sorted(
        _artifact_id(path, roots)
        for root in roots
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".py", ".toml"} and not path.name.endswith(".test.toml")
    )
    setup_rules(rules_paths=list(roots), force=True)
    engine = attribution_lib.get_rules_engine()
    declarative_emit = engine.declarative_emit
    collections: tuple[tuple[str, dict[str, Rule]], ...] = tuple(
        (kind, {name: rule for name, rule in engine.rules.items() if type(rule) is rule_type})
        for kind, rule_type in RULE_TYPES.items()
    )
    rules = [
        _rule_record(kind, rule, roots, declarative_emit) for kind, by_name in collections for rule in by_name.values()
    ]
    rules.sort(key=lambda item: (item["kind"], item["name"]))
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "rules_dirs": [root.as_posix() for root in roots],
            "discovered": discovered,
        },
        "rules": rules,
        "declared_rule_names": sorted(engine.declared_rule_names),
        "declared_event_types": sorted(engine.declared_event_types),
    }


def _signature(rule: dict[str, Any]) -> tuple[Any, ...]:
    """Return the runtime identity signature used for automatic rule pairing."""
    kind = rule["kind"]
    data = _project_rule(rule, _runtime_schema(kind))
    if kind in {"extractor", "extractor_accumulator"}:
        return (kind, data.get("match_re"), tuple(data.get("sources", ())))
    if kind in {"filter", "refiner"}:
        return (kind, data.get("priority"))
    if kind == "collector":
        return (kind, data.get("params_model"))
    raise ValueError(f"unknown rule kind {kind!r}")


def _normalize_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the inventory envelope."""
    required_fields = {
        "schema_name",
        "schema_version",
        "artifacts",
        "rules",
        "declared_rule_names",
        "declared_event_types",
    }
    missing_fields = required_fields - set(inventory)
    if len(missing_fields) > 0:
        raise ValueError(f"inventory is missing required fields {sorted(missing_fields)}")

    version = inventory.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported inventory schema version {version!r}; expected {SCHEMA_VERSION}")
    schema_name = inventory.get("schema_name")
    if schema_name != SCHEMA_NAME:
        raise ValueError(
            f"inventory schema version {version} requires schema_name={SCHEMA_NAME!r}, got {schema_name!r}"
        )

    artifacts = inventory["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("inventory field 'artifacts' must be an object")
    missing_artifact_fields = {"rules_dirs", "discovered"} - set(artifacts)
    if len(missing_artifact_fields) > 0:
        raise ValueError(f"inventory artifacts are missing required fields {sorted(missing_artifact_fields)}")
    for field in ("rules_dirs", "discovered"):
        value = artifacts[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"inventory artifacts field {field!r} must be an array of strings")

    stored_rules = inventory["rules"]
    if not isinstance(stored_rules, list):
        raise ValueError("inventory field 'rules' must be an array")
    rules = []
    for stored_rule in stored_rules:
        if not isinstance(stored_rule, dict):
            raise ValueError("each inventory rule must be an object")
        rule = dict(stored_rule)
        definition = rule.get("definition")
        if not isinstance(definition, dict) or not isinstance(definition.get("schema"), str):
            raise ValueError(f"rule {(rule.get('kind'), rule.get('name'))} has no versioned definition schema")
        if not isinstance(definition.get("data"), dict):
            raise ValueError(f"rule {(rule.get('kind'), rule.get('name'))} definition data must be an object")
        rules.append(rule)

    declarations: dict[str, list[str]] = {}
    for field in ("declared_rule_names", "declared_event_types"):
        value = inventory[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"inventory field {field!r} must be an array of strings")
        declarations[field] = value

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "artifacts": artifacts,
        "rules": rules,
        "declared_rule_names": declarations["declared_rule_names"],
        "declared_event_types": declarations["declared_event_types"],
    }


def _projection_distances(rule: dict[str, Any]) -> dict[str, int]:
    """Return the shortest registered projection distance to each reachable schema."""
    source = rule["definition"]["schema"]
    distances = {source: 0}
    pending = [source]
    while len(pending) > 0:
        current = pending.pop(0)
        for from_schema, to_schema in RULE_PROJECTORS:
            if from_schema == current and to_schema not in distances:
                distances[to_schema] = distances[current] + 1
                pending.append(to_schema)
    return distances


def _reachable_schemas(rule: dict[str, Any]) -> set[str]:
    """Return every schema reachable from a rule's native definition schema."""
    return set(_projection_distances(rule))


def _project_rule(rule: dict[str, Any], target_schema: str) -> dict[str, Any]:
    """Project a rule definition to a target schema along the shortest available path."""
    source = rule["definition"]["schema"]
    data = cast("dict[str, Any]", rule["definition"]["data"])
    if target_schema == source:
        return data
    # Breadth-first search retains the actual projector path, allowing future
    # schemas to project through intermediate versions.
    pending: list[tuple[str, dict[str, Any]]] = [(source, data)]
    seen = {source}
    while len(pending) > 0:
        current_schema, current_data = pending.pop(0)
        for (from_schema, to_schema), projector in RULE_PROJECTORS.items():
            if from_schema != current_schema or to_schema in seen:
                continue
            projected = projector(rule["kind"], current_data)
            if to_schema == target_schema:
                return projected
            seen.add(to_schema)
            pending.append((to_schema, projected))
    raise ValueError(f"rule {(rule['kind'], rule['name'])} cannot project {source!r} to {target_schema!r}")


def _common_projection(reference: dict[str, Any], candidate: dict[str, Any], forced: str) -> str | None:
    """Choose the most specific common schema for a paired rule definition."""
    kind = reference["kind"]
    reference_distances = _projection_distances(reference)
    candidate_distances = _projection_distances(candidate)
    if forced == RUNTIME_PROJECTION:
        target = _runtime_schema(kind)
        return target if target in reference_distances and target in candidate_distances else None
    common = set(reference_distances) & set(candidate_distances)
    if len(common) == 0:
        return None
    return min(
        common,
        key=lambda schema: (
            reference_distances[schema] + candidate_distances[schema],
            max(reference_distances[schema], candidate_distances[schema]),
            schema,
        ),
    )


def load_changes(path: Path | None) -> list[dict[str, str]]:
    """Load and validate explicit rule mappings from an optional TOML file."""
    if path is None:
        return []
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    changes = cast("list[dict[str, str]]", payload.get("change", []))
    for index, change in enumerate(changes):
        missing = {"kind", "reference", "candidate", "reason"} - set(change)
        if len(missing) > 0:
            raise ValueError(f"change {index} is missing keys {sorted(missing)}")
        if change["kind"] not in RULE_KINDS:
            raise ValueError(f"change {index} has unknown kind {change['kind']!r}")
        if change["reason"].strip() == "":
            raise ValueError(f"change {index} has an empty reason")
    return changes


def _index(inventory: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index inventory rules by kind and name, rejecting duplicate identities."""
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for rule in inventory["rules"]:
        key = (rule["kind"], rule["name"])
        if key in result:
            raise ValueError(f"inventory contains duplicate rule {key}")
        result[key] = rule
    return result


def compare_inventories(
    base: dict[str, Any],
    current: dict[str, Any],
    changes: Sequence[dict[str, str]] = (),
    projection: str = AUTO_PROJECTION,
) -> dict[str, Any]:
    """Compare a reference inventory with a candidate and summarize their differences."""
    if projection not in {AUTO_PROJECTION, RUNTIME_PROJECTION}:
        raise ValueError(f"unknown projection policy {projection!r}")
    base = _normalize_inventory(base)
    current = _normalize_inventory(current)

    base_by_key = _index(base)
    current_by_key = _index(current)
    remaining_base = set(base_by_key)
    remaining_current = set(current_by_key)
    pairs: list[dict[str, Any]] = []
    errors: list[str] = []

    def pair(base_key: tuple[str, str], current_key: tuple[str, str], status: str, reason: str | None = None) -> None:
        """Consume a reference/candidate pair and record its comparison status."""
        remaining_base.remove(base_key)
        remaining_current.remove(current_key)
        pairs.append(
            {
                "kind": base_key[0],
                "base": base_key[1],
                "current": current_key[1],
                "status": status,
                **({"reason": reason} if reason is not None else {}),
            }
        )

    # Explicit mappings take precedence, including approved same-name signature changes.
    for change in changes:
        base_key = (change["kind"], change["reference"])
        current_key = (change["kind"], change["candidate"])
        if base_key not in remaining_base:
            errors.append(f"explicit mapping has missing or reused reference rule {base_key}")
            continue
        if current_key not in remaining_current:
            errors.append(f"explicit mapping has missing or reused candidate rule {current_key}")
            continue
        pair(base_key, current_key, "approved_change", change["reason"])

    # Same identity and signature is the strongest automatic match.
    for key in sorted(remaining_base & remaining_current):
        base_rule = base_by_key[key]
        current_rule = current_by_key[key]
        runtime = _runtime_schema(key[0])
        runtime_comparable = runtime in _reachable_schemas(base_rule) & _reachable_schemas(current_rule)
        same_native = base_rule["definition"] == current_rule["definition"]
        if (runtime_comparable is True and _signature(base_rule) == _signature(current_rule)) or same_native is True:
            pair(key, key, "unchanged")

    # Then pair matcher-equivalent renames. Sorting makes repeated signatures deterministic.
    base_by_signature: dict[tuple[Any, ...], list[tuple[str, str]]] = defaultdict(list)
    current_by_signature: dict[tuple[Any, ...], list[tuple[str, str]]] = defaultdict(list)
    for key in remaining_base:
        if key[0] == "extractor" and _runtime_schema(key[0]) in _reachable_schemas(base_by_key[key]):
            base_by_signature[_signature(base_by_key[key])].append(key)
    for key in remaining_current:
        if key[0] == "extractor" and _runtime_schema(key[0]) in _reachable_schemas(current_by_key[key]):
            current_by_signature[_signature(current_by_key[key])].append(key)
    for signature in sorted(base_by_signature, key=repr):
        base_keys = sorted(base_by_signature[signature])
        current_keys = sorted(current_by_signature.get(signature, []))
        for base_key, current_key in zip(base_keys, current_keys, strict=False):
            pair(base_key, current_key, "renamed")

    # A same-name rule with a different signature is never silently accepted.
    changed = sorted(remaining_base & remaining_current)
    for key in changed:
        errors.append(f"rule {key} changed signature without an explicit mapping")
        pair(key, key, "unapproved_change")

    unmatched = sorted(remaining_base)
    additions = sorted(remaining_current)
    for key in unmatched:
        errors.append(f"reference rule is missing from candidate {key}")

    for field in ("declared_rule_names", "declared_event_types"):
        dropped = sorted(set(base.get(field, ())) - set(current.get(field, ())))
        if len(dropped) > 0:
            errors.append(f"dropped {field}: {dropped}")

    schema_transitions: list[dict[str, Any]] = []
    definition_changes: list[dict[str, Any]] = []
    tag_changes: list[dict[str, Any]] = []
    incomparable: list[dict[str, Any]] = []
    for paired in pairs:
        base_rule = base_by_key[(paired["kind"], paired["base"])]
        current_rule = current_by_key[(paired["kind"], paired["current"])]
        approved = paired["status"] == "approved_change"
        base_schema = base_rule["definition"]["schema"]
        current_schema = current_rule["definition"]["schema"]
        base_tags = base_rule.get("tags", [])
        current_tags = current_rule.get("tags", [])
        if base_tags != current_tags:
            # Informational only, no errors on tag changes.
            tag_changes.append(
                {
                    "kind": paired["kind"],
                    "reference": paired["base"],
                    "candidate": paired["current"],
                    "reference_tags": base_tags,
                    "candidate_tags": current_tags,
                }
            )
        common = _common_projection(base_rule, current_rule, projection)
        if base_schema != current_schema:
            schema_transitions.append(
                {
                    "kind": paired["kind"],
                    "reference": paired["base"],
                    "candidate": paired["current"],
                    "reference_schema": base_schema,
                    "candidate_schema": current_schema,
                    "projection": common,
                }
            )
        if common is None:
            difference = {
                "kind": paired["kind"],
                "reference": paired["base"],
                "candidate": paired["current"],
                "reference_schema": base_schema,
                "candidate_schema": current_schema,
                "approved": approved,
            }
            incomparable.append(difference)
            if approved is False:
                errors.append(
                    f"rule {(paired['kind'], paired['base'])} has no common projection "
                    f"between {base_schema!r} and {current_schema!r}"
                )
            continue
        before_data = _project_rule(base_rule, common)
        after_data = _project_rule(current_rule, common)
        for field in sorted(set(before_data) | set(after_data)):
            before = before_data.get(field)
            after = after_data.get(field)
            if before == after:
                continue
            definition_changes.append(
                {
                    "kind": paired["kind"],
                    "reference": paired["base"],
                    "candidate": paired["current"],
                    "projection": common,
                    "field": field,
                    "approved": approved,
                }
            )
            if approved is False:
                errors.append(
                    f"rule {(paired['kind'], paired['base'])} changed field {field!r} under projection {common!r}"
                )

    counts = {
        status: sum(pair_["status"] == status for pair_ in pairs)
        for status in ("unchanged", "renamed", "approved_change", "unapproved_change")
    }
    return {
        "safe": len(errors) == 0,
        "projection": projection,
        "base_rules": len(base_by_key),
        "current_rules": len(current_by_key),
        "counts": counts,
        "pairs": sorted(pairs, key=lambda item: (item["kind"], item["base"])),
        "unmatched": [{"kind": kind, "name": name} for kind, name in unmatched],
        "additions": [{"kind": kind, "name": name} for kind, name in additions],
        "schema_transitions": schema_transitions,
        "definition_changes": definition_changes,
        "tag_changes": tag_changes,
        "incomparable": incomparable,
        "errors": errors,
    }


def render_report(result: dict[str, Any]) -> str:
    """Render a comparison result as a human-readable Markdown report."""
    counts = result["counts"]
    lines = [
        "# Rule inventory comparison",
        "",
        f"Result: **{'PASS' if result['safe'] is True else 'FAIL'}**",
        "",
        f"- Projection: `{result['projection']}`",
        f"- Reference rules: {result['base_rules']}",
        f"- Candidate rules: {result['current_rules']}",
        f"- Unchanged: {counts['unchanged']}",
        f"- Matcher-equivalent renames: {counts['renamed']}",
        f"- Approved changes: {counts['approved_change']}",
        f"- Unapproved changes: {counts['unapproved_change']}",
        f"- Additions: {len(result['additions'])}",
        f"- Missing from candidate: {len(result['unmatched'])}",
        f"- Rule-schema transitions: {len(result['schema_transitions'])}",
        f"- Projected definition changes: {len(result['definition_changes'])}",
        f"- Tag changes: {len(result['tag_changes'])}",
        f"- Incomparable rules: {len(result['incomparable'])}",
    ]
    renamed = [pair_ for pair_ in result["pairs"] if pair_["status"] == "renamed"]
    approved = [pair_ for pair_ in result["pairs"] if pair_["status"] == "approved_change"]
    unapproved = [pair_ for pair_ in result["pairs"] if pair_["status"] == "unapproved_change"]
    if len(renamed) > 0:
        lines.extend(["", "## Matcher-equivalent renames", ""])
        lines.extend(f"- `{item['kind']}:{item['base']}` → `{item['current']}`" for item in renamed)
    if len(approved) > 0:
        lines.extend(["", "## Approved changes", ""])
        lines.extend(f"- `{item['kind']}:{item['base']}` → `{item['current']}`: {item['reason']}" for item in approved)
    if len(unapproved) > 0:
        lines.extend(["", "## Unapproved signature changes", ""])
        lines.extend(f"- `{item['kind']}:{item['base']}` → `{item['current']}`" for item in unapproved)
    if len(result["unmatched"]) > 0:
        lines.extend(["", "## Missing from candidate", ""])
        lines.extend(f"- `{item['kind']}:{item['name']}`" for item in result["unmatched"])
    if len(result["additions"]) > 0:
        lines.extend(["", "## Candidate additions", ""])
        lines.extend(f"- `{item['kind']}:{item['name']}`" for item in result["additions"])
    if len(result["schema_transitions"]) > 0:
        lines.extend(["", "## Rule-schema transitions", ""])
        transitions = Counter(
            (item["reference_schema"], item["candidate_schema"], item["projection"])
            for item in result["schema_transitions"]
        )
        lines.extend(
            f"- `{reference}` → `{candidate}` via `{common}`: {count} rules"
            for (reference, candidate, common), count in sorted(transitions.items(), key=repr)
        )
    if len(result["definition_changes"]) > 0:
        lines.extend(["", "## Projected definition changes", ""])
        lines.extend(
            f"- `{item['kind']}:{item['reference']}` field `{item['field']}` under `{item['projection']}`"
            f"{' (approved)' if item['approved'] is True else ''}"
            for item in result["definition_changes"]
        )
    if len(result["tag_changes"]) > 0:
        lines.extend(["", "## Tag changes", ""])
        lines.extend(
            f"- `{item['kind']}:{item['reference']}`: `{item['reference_tags']}` → `{item['candidate_tags']}`"
            for item in result["tag_changes"]
        )
    if len(result["incomparable"]) > 0:
        lines.extend(["", "## Incomparable rules", ""])
        lines.extend(
            f"- `{item['kind']}:{item['reference']}`: `{item['reference_schema']}` vs "
            f"`{item['candidate_schema']}`{' (approved)' if item['approved'] is True else ''}"
            for item in result["incomparable"]
        )
    if len(result["errors"]) > 0:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result["errors"])
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON inventory document from disk."""
    with path.open(encoding="utf-8") as stream:
        return cast("dict[str, Any]", json.load(stream))


def _write_text(path: Path | None, text: str) -> None:
    """Write text to a path or standard output when no path was supplied."""
    if path is None:
        sys.stdout.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inventory command-line interface and return its process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dump_parser = subparsers.add_parser("dump", help="load bundled rules and write an inventory")
    dump_parser.add_argument("--output", "-o", type=Path)
    dump_parser.add_argument(
        "--rules-dir",
        action="append",
        dest="rules_dirs",
        type=Path,
        help="rule directory to load; repeat to load several (default: bundled rules)",
    )
    compare_parser = subparsers.add_parser("compare", help="compare a reference inventory with a candidate")
    compare_parser.add_argument("base", metavar="REFERENCE", type=Path)
    compare_parser.add_argument("current", metavar="CANDIDATE", type=Path)
    compare_parser.add_argument("--changes", type=Path)
    compare_parser.add_argument(
        "--projection",
        choices=(AUTO_PROJECTION, RUNTIME_PROJECTION),
        default=AUTO_PROJECTION,
        help="per-rule projection policy (default: auto)",
    )
    compare_parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    if args.command == "dump":
        _write_text(args.output, json.dumps(dump_inventory(args.rules_dirs), indent=2, sort_keys=True) + "\n")
        return 0

    result = compare_inventories(
        _read_json(args.base),
        _read_json(args.current),
        load_changes(args.changes),
        projection=args.projection,
    )
    _write_text(args.report, render_report(result))
    return 0 if result["safe"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
