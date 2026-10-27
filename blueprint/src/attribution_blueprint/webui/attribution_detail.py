# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and project detail sections without rewriting stored V2 evidence."""

from __future__ import annotations

import json
from collections.abc import Callable
from graphlib import TopologicalSorter
from typing import Any

from nvidia_attribution_client.models.decision_graph import DecisionGraph
from nvidia_attribution_client.models.node_source import NodeSource
from nvidia_attribution_client.models.resourceless_source import ResourcelessSource
from nvidia_attribution_client.models.sourced_symptom import SourcedSymptom


def _validate_graph(value: Any) -> None:
    graph = DecisionGraph.model_validate_json(json.dumps(value), strict=True)
    node_ids = {node.node_id for node in graph.nodes}
    if len(node_ids) != len(graph.nodes):
        raise ValueError("Duplicate decision node IDs")
    predecessors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in graph.edges:
        if edge.parent_id not in node_ids or edge.child_id not in node_ids:
            raise ValueError("Decision edge references a missing node")
        predecessors[edge.child_id].add(edge.parent_id)
    TopologicalSorter(predecessors).prepare()


def _validate_sites(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("Symptoms must be a map")
    for sites in value.values():
        if not isinstance(sites, list):
            raise ValueError("Symptom sites must be an array")
        for site in sites:
            if not isinstance(site, dict):
                raise ValueError("Symptom site must be an object")
            if site.get("type") == "NodeSource":
                model = NodeSource
            elif site.get("type") == "ResourcelessSource":
                model = ResourcelessSource
            else:
                raise ValueError("Missing or invalid symptom site discriminator")
            model.model_validate_json(json.dumps(site), strict=True)


def _project_source_id(record: dict[str, Any]) -> dict[str, Any]:
    # Observation IDs are opaque identifiers. Encode them while Python still has
    # the exact integer, before a browser's JSON parser can round an int64 value.
    source = record["source"]
    return {**record, "source": {**source, "observation_id": str(source["observation_id"])}}


def _project_sites(sites: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {symptom: [_project_source_id(site) for site in entries] for symptom, entries in sites.items()}


def _section(
    parent: Any,
    key: str,
    validate: Callable[[Any], Any],
    is_empty: Callable[[Any], bool] = lambda _: False,
    project: Callable[[Any], Any] = lambda value: value,
) -> dict[str, Any]:
    if not isinstance(parent, dict):
        return {"state": "invalid", "data": None}
    if key not in parent:
        return {"state": "absent", "data": None}
    value = parent[key]
    if value is None:
        return {"state": "null", "data": None}
    try:
        validate(value)
    except (ValueError, TypeError):
        return {"state": "invalid", "data": None}
    # Models are used only for validation. Serializing them would insert defaults,
    # normalize timestamps and drop additional provenance from the stored response.
    return {"state": "empty" if is_empty(value) else "available", "data": project(value)}


def decision_graph_section(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Return the same validated graph state for list entries and run details."""
    verdict = raw_result.get("verdict", {}) if isinstance(raw_result, dict) else None
    return _section(verdict, "decision_graph", _validate_graph, lambda value: not value["nodes"] and not value["edges"])


def attribution_detail(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Expose root, sites and graph with separate availability states.

    The models come from the generated Service SDK in this repository. Invalid
    graph data cannot invalidate a sound verdict or its symptom-site evidence.
    Only known observation IDs are projected to strings for lossless browser
    transport; other evidence and unknown provenance fields remain unchanged.
    """
    verdict = raw_result.get("verdict", {}) if isinstance(raw_result, dict) else None
    return {
        "rootSymptom": _section(
            verdict,
            "root_symptom",
            lambda value: SourcedSymptom.model_validate_json(json.dumps(value), strict=True),
            project=_project_source_id,
        ),
        "symptomSites": _section(
            raw_result,
            "symptoms",
            _validate_sites,
            lambda value: not any(value.values()),
            project=_project_sites,
        ),
        "decisionGraph": decision_graph_section(raw_result),
    }
