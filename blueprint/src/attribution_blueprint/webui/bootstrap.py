# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map read-only SQLite rows to the React summary and detail contracts."""

from __future__ import annotations

import json as _json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attribution_blueprint.taxonomy import taxonomy_for_service_value
from attribution_blueprint.webui.attribution_detail import attribution_detail, decision_graph_section
from attribution_blueprint.webui.diagnose import diagnose_card, operator_diagnose_card

# Raw lib outcome string -> React OutcomeKey. Unmapped values fall through to
# "multi" (the gray MultiCategory/other bucket present in the React palette).
_OUTCOME_MAP = {
    "Fail InfraHW": "hardware",
    "Fail InfraSW": "software",
    "Fail Multi": "multi",
    "Fail MultiCategory": "multi",
    "Fail User": "user",
    "Fail Undetermined": "undetermined",
    "Pass": "pass",
}

_SYSLOG_PREFIX_RE = re.compile(
    r"^(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})(?P<fraction>\.\d+)?\b"
)


def normalize_outcome(raw: str | None) -> str:
    return _OUTCOME_MAP.get(raw or "", "multi")


def display_outcome(raw: str | None) -> str:
    """Return the diagnosis label for a raw service outcome.

    The label comes from the documented `(terminal_state, sub_domain)` taxonomy
    in `taxonomy.py` rather than from the raw service string, so the dashboard
    shows only the documented vocabulary. Unrecognized service
    outcomes fall through to the taxonomy's unclassified row.

    Args:
        raw: Raw service outcome.

    Returns:
        Diagnosis label for dashboard tables.
    """

    return taxonomy_for_service_value(raw or "").blueprint_taxonomy.value


def _diagnose_camel(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "hasPersona": card.get("has_persona", False),
        "title": card.get("title", ""),
        "summary": card.get("summary", ""),
        "impact": card.get("impact", ""),
        "action": card.get("action", ""),
        "supportDetails": list(card.get("support_details") or []),
        "category": card.get("category"),
        "severity": card.get("severity"),
        "catalogId": card.get("catalog_id"),
    }


def _parse_time(value: Any) -> datetime | None:
    """Parse an ISO-like timestamp into a timezone-aware datetime.

    Args:
        value: Raw timestamp value.

    Returns:
        Parsed timestamp, or None when the value is empty or invalid.
    """
    if not isinstance(value, str) or value.strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _root_time_from_line(line: Any, start_time: str, end_time: str) -> datetime | None:
    """Infer root-symptom time from a syslog-style line prefix.

    Args:
        line: Raw matched log line.
        start_time: Run start timestamp.
        end_time: Run end timestamp.

    Returns:
        A timezone-aware UTC timestamp, or None when the line has no supported
        prefix.
    """
    if not isinstance(line, str) or line == "":
        return None
    match = _SYSLOG_PREFIX_RE.search(line)
    if match is None:
        return None

    end = _parse_time(end_time)
    if end is None:
        return None
    start = _parse_time(start_time)
    year = end.year
    raw = f"{year} {match.group('month')} {match.group('day')} {match.group('time')}{match.group('fraction') or ''}"
    fmt = "%Y %b %d %H:%M:%S.%f" if match.group("fraction") is not None else "%Y %b %d %H:%M:%S"
    try:
        parsed = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return None

    if start is not None and start.year != end.year:
        if parsed > end:
            parsed = parsed.replace(year=start.year)
        elif parsed < start:
            parsed = parsed.replace(year=start.year)
    return parsed


def _root_symptom_time(run, root_src: dict[str, Any]) -> str:
    """Return a parseable root-symptom timestamp for React.

    Args:
        run: Web UI database row.
        root_src: Raw root-symptom source object.

    Returns:
        ISO-8601 timestamp, preferring explicit source time and falling back to
        the matched log prefix or run end time.
    """
    explicit = _parse_time(root_src.get("time"))
    if explicit is not None:
        return explicit.isoformat()
    inferred = _root_time_from_line(root_src.get("line"), run.start_time, run.end_time)
    if inferred is not None:
        return inferred.isoformat()
    return run.end_time


def _log_from_sites(run) -> list[dict[str, Any]]:
    """Matched-line evidence as the 'log': root symptom line at level 'symptom',
    other symptom-site lines at level 'error', sorted by time ascending.

    Dedup is by the EXACT (line, time) pair, not by line text alone. Two sites
    with identical text but different times (or on different nodes at different
    times) are distinct evidence and both appear; only a site that exactly
    repeats an already-emitted (line, time) — e.g. the root entry's own site —
    is suppressed.
    """
    lines: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    verdict = (run.raw_result or {}).get("verdict") or {}
    root = verdict.get("root_symptom") or {}
    root_src = root.get("source") or {}
    root_line = root_src.get("line") or ""
    root_time = _root_symptom_time(run, root_src)
    if root_line:
        lines.append({"t": root_time, "level": "symptom", "text": root_line})
        seen.add((root_line, root_time))
    symptoms = run.raw_result.get("symptoms")
    for sites in symptoms.values() if isinstance(symptoms, dict) else []:
        for site in sites if isinstance(sites, list) else []:
            src = _object(site.get("source")) if isinstance(site, dict) else {}
            line = _text(src.get("line"))
            if not line:
                continue
            t = _root_symptom_time(run, src)
            if (line, t) in seen:
                continue
            seen.add((line, t))
            lines.append({"t": t, "level": "error", "text": line})
    lines.sort(key=lambda x: x["t"])
    return lines


def _elapsed_hours(run) -> float:
    """Job duration in hours from execution_context start_time/end_time."""
    try:
        start = datetime.fromisoformat(run.start_time)
        end = datetime.fromisoformat(run.end_time)
    except ValueError:
        return 0.0
    seconds = (end - start).total_seconds()
    return max(seconds / 3600.0, 0.0)


def _positive_int(value: Any) -> int | None:
    """Coerce positive integer values from scheduler JSON.

    Args:
        value: Raw scheduler JSON value.

    Returns:
        A positive integer, or None when no positive integer is present.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        if parsed > 0:
            return parsed
    return None


def _non_negative_int(value: Any) -> int | None:
    """Coerce non-negative integer values from scheduler JSON.

    Args:
        value: Raw scheduler JSON value.

    Returns:
        A non-negative integer, or None when no integer is present.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _gpus_per_node(run) -> int:
    """Return the GPUs per node recorded for this run.

    Args:
        run: Web UI database row.

    Returns:
        The positive GPUs-per-node value. Falls back to the orchestrator
        default when historical rows do not include the workload value.
    """
    raw_scheduler = run.raw_scheduler if run.raw_scheduler is not None else {}
    value = _positive_int(raw_scheduler.get("gpus_per_node"))
    if value is not None:
        return value
    return 4


def _gpu_count_from_alloc_tres(alloc_tres: Any) -> int | None:
    """Parse Slurm AllocTRES GPU count.

    Args:
        alloc_tres: Raw AllocTRES value, such as ``cpu=8,gres/gpu=4``.

    Returns:
        The job GPU count when AllocTRES is present. Returns 0 for a
        CPU-only AllocTRES string and None when AllocTRES is missing.
    """
    if not isinstance(alloc_tres, str):
        return None
    if alloc_tres.strip() == "":
        return None
    total = 0
    saw_resource = False
    saw_gpu = False
    for part in alloc_tres.split(","):
        key, sep, raw_value = part.partition("=")
        if sep == "":
            continue
        saw_resource = True
        if key.strip().startswith("gres/gpu") is False:
            continue
        saw_gpu = True
        value = _non_negative_int(raw_value)
        if value is not None:
            total += value
    if saw_gpu is True:
        return total
    if saw_resource is True:
        return 0
    return None


def _gpu_count(run) -> int:
    """Return the number of GPUs allocated to the job.

    Args:
        run: Web UI database row.

    Returns:
        The positive job GPU count. Falls back to full listed nodes only when
        the scheduler JSON does not carry a job-level count.
    """
    raw_scheduler = run.raw_scheduler if run.raw_scheduler is not None else {}
    for key in ("gpu_count", "gpuCount", "gpus", "ngpus", "num_gpus", "allocated_gpus", "requested_gpus"):
        value = _non_negative_int(raw_scheduler.get(key))
        if value is not None:
            return value
    alloc_tres_value = _gpu_count_from_alloc_tres(raw_scheduler.get("alloc_tres"))
    if alloc_tres_value is not None:
        return alloc_tres_value
    node_count = len(run.nodes)
    billed_nodes = node_count if node_count > 0 else 1
    return billed_nodes * _gpus_per_node(run)


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _display_run(run):
    """Keep malformed evidence out of display fields without changing storage."""
    raw = _object(run.raw_result)
    verdict = _object(raw.get("verdict"))
    root = _object(verdict.get("root_symptom"))
    source = _object(root.get("source"))
    return replace(
        run,
        raw_result={
            **raw,
            "verdict": {
                **verdict,
                "attribution_outcome": _text(verdict.get("attribution_outcome")),
                "detail": _text(verdict.get("detail")),
                "root_symptom": {
                    "symptom": _text(root.get("symptom")),
                    "catalog_id": _text(root.get("catalog_id")),
                    "source": {key: _text(source.get(key)) for key in ("rule", "line", "time")},
                },
            },
        },
    )


def _ingestion_reports(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Return both report lists for display, or None for rows saved before ingestion reports were recorded."""
    if "ingestion_reports" not in raw:
        return None
    return {
        "connector_reports": raw.get("connector_reports") or [],
        "ingestion_reports": raw.get("ingestion_reports") or [],
    }


def _run_summary_fields(run) -> dict[str, Any]:
    """Map shared display fields without loading logs or validating evidence."""
    run = _display_run(run)
    verdict = (run.raw_result or {}).get("verdict") or {}
    collectors = (run.raw_result or {}).get("collectors")
    evidence_status = (run.raw_result or {}).get("evidence_status")
    ingestion_reports = _ingestion_reports(run.raw_result or {})
    bad_pairs = collectors.get("bad_pairs") if isinstance(collectors, dict) else None
    root = verdict.get("root_symptom") or {}
    root_src = root.get("source") or {}
    job = run.job_id
    status = "COMPLETED" if (run.status or "").upper() in {"COMPLETED", "COMPLETE", "SUCCESS"} else "FAILED"
    from attribution_blueprint.webui.db import symptom_summary

    symptoms = [{"name": name, "sites": count} for name, count in symptom_summary(run)]
    diagnose = diagnose_card(run)
    operator_diagnose = operator_diagnose_card(run)
    return {
        "id": run.run_id,
        "jobId": job,
        "workload": job if "/" in job else "—",
        "attempt": 1,
        "attemptUid": run.attempt_id or run.run_id,
        "user": run.user_name or "",
        "status": status,
        "nodes": len(run.nodes),
        "nodeIds": list(run.nodes),
        "elapsedH": round(_elapsed_hours(run), 6),
        "gpusPerNode": _gpus_per_node(run),
        "gpuCount": _gpu_count(run),
        "endTime": run.end_time,
        "startTime": run.start_time,
        "outcome": normalize_outcome(run.outcome),
        "verdictLabel": display_outcome(verdict.get("attribution_outcome") or run.outcome),
        "detail": verdict.get("detail") or "",
        "rootSymptom": {
            "symptom": root.get("symptom") or "",
            "rule": root_src.get("rule") or "",
            "time": _root_symptom_time(run, root_src),
            "line": root_src.get("line") or "",
        },
        "symptoms": symptoms,
        "errorPairs": bad_pairs if isinstance(bad_pairs, str) else "",
        "responsibleNodes": list(run.responsible_nodes or []),
        "pipeline": run.pipeline_version or "",
        "service": run.service_api_version or "",
        "terminalStates": _list_of_str(verdict.get("terminal_states")),
        "failureDomains": _list_of_str(verdict.get("failure_domains")),
        "cancellationActors": _list_of_str(verdict.get("cancellation_actors")),
        "uncertaintyType": verdict.get("uncertainty_type") if isinstance(verdict.get("uncertainty_type"), str) else None,
        "evidenceStatus": evidence_status if isinstance(evidence_status, str) else None,
        "ingestionReports": ingestion_reports,
        "diagnose": _diagnose_camel(diagnose),
        "operatorDiagnose": _diagnose_camel(operator_diagnose),
    }


def run_summary_to_react(run) -> dict[str, Any]:
    """Return list fields and graph availability without graph data or logs."""
    return {
        **_run_summary_fields(run),
        "decisionGraphState": decision_graph_section(run.raw_result)["state"],
    }


def run_to_react(run) -> dict[str, Any]:
    """Return a summary plus logs and independently validated V2 evidence."""
    attribution = attribution_detail(run.raw_result)
    return {
        **_run_summary_fields(run),
        "decisionGraphState": attribution["decisionGraph"]["state"],
        "log": _log_from_sites(_display_run(run)),
        "attribution": attribution,
    }


def build_payload(
    db_path: str,
    since: str | None = None,
    until: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    from attribution_blueprint.webui import db as dbmod

    runs = dbmod.list_runs(db_path, since=since, until=until, job_query=q)
    windowed_total = dbmod.count_runs(db_path, since=since, until=until, job_query=q)
    db_total = dbmod.count_runs(db_path)
    summary = dbmod.db_summary(db_path)
    lo, hi = dbmod.date_range_bounds(db_path)
    pipeline = runs[0].pipeline_version if runs else None
    service = runs[0].service_api_version if runs else None
    mapped = [run_to_react(r) for r in runs]
    return {
        "runs": mapped,
        "meta": {
            "versions": {
                "schema": summary.get("schema_version"),
                "pipeline": pipeline,
                "service": service,
            },
            "dateBounds": {"min": lo[:10] if lo else None, "max": hi[:10] if hi else None},
            "windowedRuns": windowed_total,
            "dbTotalRuns": db_total,
            "database": Path(db_path).name,
            "returnedRuns": len(mapped),
        },
    }


_SCRIPT_ESCAPES = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"}


def inject(index_html: str, payload: dict[str, Any]) -> str:
    blob = _json.dumps(payload, separators=(",", ":"))
    for ch, esc in _SCRIPT_ESCAPES.items():
        blob = blob.replace(ch, esc)
    returned = payload.get("meta", {}).get("returnedRuns", 0)
    tag = f'<script type="application/json" id="attr-bootstrap">{blob}</script>'
    html = index_html.replace('<div id="root">', f'<div id="root" data-total-runs="{returned}">', 1)
    if "</body>" in html:
        return html.replace("</body>", tag + "</body>", 1)
    return html + tag
