# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from copy import deepcopy
from typing import Literal, NotRequired, TypedDict

from attribution_blueprint.taxonomy import taxonomy_for_service_value


class ConnectorReport(TypedDict):
    connector: str
    source: str
    status: Literal["ok", "error"]
    reason: str
    observation_count: int


class AttributionBlueprintResult(TypedDict, total=False):
    terminal_state: str
    sub_domain: str
    connector_reports: list[ConnectorReport]
    tenant_sre_action: str
    cloud_operator_sre_action: str
    verdict: dict
    symptoms: dict
    observation_context: list
    model_type: NotRequired[str]


def service_outcome(result: dict | None) -> str:
    """Return the raw service outcome."""
    if result is None:
        return ""
    verdict = result.get("verdict")
    if not isinstance(verdict, dict):
        return ""
    outcome = verdict.get("attribution_outcome")
    if outcome is not None:
        return str(outcome)
    return ""


def enrich_attribution_result(
    service_result: dict | None,
    connector_reports: list[ConnectorReport],
) -> AttributionBlueprintResult:
    """Add Blueprint contract fields around an attribution service response."""
    result = {} if service_result is None else deepcopy(service_result)
    taxonomy = taxonomy_for_service_value(service_outcome(result))

    result["terminal_state"] = taxonomy.terminal_state
    result["sub_domain"] = taxonomy.sub_domain
    result["connector_reports"] = [dict(report) for report in connector_reports]
    result["tenant_sre_action"] = _action_value(result, "tenant_sre_action")
    result["cloud_operator_sre_action"] = _action_value(result, "cloud_operator_sre_action")
    symptoms = result.get("symptoms")
    if isinstance(symptoms, dict):
        result["symptoms"] = _enrich_symptoms(symptoms)
    _enrich_root_symptom(result)
    return result


def _action_value(result: dict, field: str) -> str:
    verdict = result.get("verdict")
    if not isinstance(verdict, dict):
        verdict = {}
    value = result.get(field)
    if value is None:
        value = verdict.get(field)
    if value is None:
        return ""
    return str(value)


def _enrich_symptoms(symptoms: dict) -> dict:
    enriched: dict = {}
    for name, sites in symptoms.items():
        enriched[name] = (
            [_enrich_site(site) if isinstance(site, dict) else site for site in sites]
            if isinstance(sites, list)
            else sites
        )
    return enriched


def _enrich_site(site: dict) -> dict:
    enriched = dict(site)
    channel = _source_channel(site.get("source"))
    if channel != "":
        enriched["source_channel"] = channel
    return enriched


def _enrich_root_symptom(result: dict) -> None:
    verdict = result.get("verdict")
    if not isinstance(verdict, dict):
        verdict = {}
    root = verdict.get("root_symptom")
    if root is None:
        root = {}
    if not isinstance(root, dict):
        return
    channel = _source_channel(root.get("source"))
    if channel != "":
        root["source_channel"] = channel


def _source_channel(source: object) -> str:
    if not isinstance(source, dict):
        return ""
    value = source.get("source")
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).lower()
