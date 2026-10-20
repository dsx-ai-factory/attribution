# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


EvidenceStatus = Literal["complete", "partial", "none"]


class IngestionReport(TypedDict):
    connector: str
    source: str
    status: Literal["ok", "partial", "error", "empty"]
    attempted: int
    failed: int
    empty: int
    accepted: int
    dropped: dict[str, int]
    reason: str


class AttributionBlueprintResult(TypedDict, total=False):
    terminal_state: str
    sub_domain: str
    connector_reports: list[ConnectorReport]
    ingestion_reports: list[IngestionReport]
    evidence_status: EvidenceStatus
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
    ingestion_reports: list[IngestionReport],
) -> AttributionBlueprintResult:
    """Add Blueprint contract fields around an attribution service response."""
    result = {} if service_result is None else deepcopy(service_result)
    taxonomy = taxonomy_for_service_value(service_outcome(result))

    result["terminal_state"] = taxonomy.terminal_state
    result["sub_domain"] = taxonomy.sub_domain
    result["connector_reports"] = [dict(report) for report in connector_reports]
    result["ingestion_reports"] = [dict(report) for report in ingestion_reports]
    result["evidence_status"] = _evidence_status(connector_reports, ingestion_reports)
    result["tenant_sre_action"] = _action_value(result, "tenant_sre_action")
    result["cloud_operator_sre_action"] = _action_value(result, "cloud_operator_sre_action")
    symptoms = result.get("symptoms")
    if isinstance(symptoms, dict):
        result["symptoms"] = _enrich_symptoms(symptoms)
    _enrich_root_symptom(result)
    return result


def _evidence_status(
    connector_reports: list[ConnectorReport],
    ingestion_reports: list[IngestionReport],
) -> EvidenceStatus:
    """Classify how much collected evidence reached the service for this verdict."""
    if all(report["attempted"] == 0 for report in ingestion_reports):
        return "none"
    collectors_ok = all(report["status"] == "ok" for report in connector_reports)
    if collectors_ok and all(report["status"] == "ok" for report in ingestion_reports):
        return "complete"
    if any(report["attempted"] > 0 and report["status"] in ("ok", "partial") for report in ingestion_reports):
        return "partial"
    return "none"


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
