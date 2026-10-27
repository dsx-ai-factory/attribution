# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from attribution_lib.symptoms import INFORMATIONAL_XIDS


@dataclass(frozen=True, slots=True)
class RuleDataCase:
    path: Path
    runner: str
    description: str
    case: dict

    @property
    def pytest_id(self) -> str:
        return f"{self.path.name}:{self.description}"


def _bundled_rules_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "attribution_lib" / "bundled_rules"


_COMMON_CASE_KEYS = {
    "description",
    "runner",
    "expect",
    "expected_symptom",
    "expected_rule",
    "xfail",
}

_RUNNER_CASE_KEYS = {
    "node_metrics": {"metric", "value", "node", "component", "timestamp"},
    "nvlink_metrics": {"metric", "value", "node", "component", "cluster", "timestamp"},
    "syslog_single_line_mfu": {
        "line",
        "node",
        "severity",
        "tflops_start",
        "timestamp_s",
        "multi_gpu",
        "inject_mfu",
        "expected_node",
        "extra_nodes",
    },
    "syslog_single_line_events": {
        "line",
        "node",
        "severity",
        "tflops_start",
        "timestamp_s",
        "multi_gpu",
        "expected_node",
        "events",
        "expectations",
        "expected_events",
    },
    "syslog_fm_gpu_series": {"node", "timestamp_s", "gpu_guids", "expectations"},
    "applog_single_line": {"line", "tflops_start", "job_variant", "expected_node", "expected_events"},
    "applog_bad_peers": {"line", "tflops_start", "job_variant", "expected_nodes"},
}

_EXPANSION_CASE_KEYS = {
    "fixed_catalog": {"catalog"},
    "syslog_xid_matrix": {"catalog", "job_phases", "multi_gpu_values"},
    "matrix": {"line_templates", "sub_cases", "job_phases"},
}


def _warn_unknown_case_keys(path: Path, case_data: dict, default_runner: str) -> None:
    runner = case_data.get("runner", default_runner)
    allowed_keys = set(_COMMON_CASE_KEYS)
    allowed_keys.update(_RUNNER_CASE_KEYS.get(runner, set()))

    expansion = case_data.get("expansion")
    if expansion is not None:
        allowed_keys.add("expansion")
        allowed_keys.update(_EXPANSION_CASE_KEYS.get(expansion, set()))
    elif "line_templates" in case_data:
        allowed_keys.update(_EXPANSION_CASE_KEYS["matrix"])

    unknown_keys = sorted(set(case_data) - allowed_keys)
    if unknown_keys:
        warnings.warn(
            f"{path} case {case_data.get('description', '<missing description>')!r} contains unknown keys: "
            f"{', '.join(unknown_keys)}",
            stacklevel=2,
        )


def _expand_matrix_case(path: Path, case_data: dict) -> list[dict]:
    description = case_data["description"]
    templates = case_data["line_templates"]
    sub_cases = case_data["sub_cases"]
    job_phases = case_data.get("job_phases", ["during"])
    base_case = {
        k: v for k, v in case_data.items() if k not in {"description", "line_templates", "sub_cases", "job_phases"}
    }
    expanded_cases = []

    for phase in job_phases:
        match phase:
            case "during":
                expect = "bad"
                tflops_start = True
                symptom_prefix = ""
            case "outside":
                expect = "interesting"
                tflops_start = False
                symptom_prefix = "(non-fatal) "
            case other:
                raise ValueError(f"{path} contains unsupported job phase {other!r}")

        for template_entry in templates:
            if isinstance(template_entry, str):
                template = template_entry
                template_description = template_entry
            else:
                template = template_entry["template"]
                template_description = template_entry.get("description", template)

            for sub_case in sub_cases:
                expanded_case = dict(base_case)
                expanded_case["description"] = (
                    f"{description}: {sub_case['symptom']} via {template_description} ({phase})"
                )
                expanded_case["line"] = template.format(sub_error=sub_case["sub_error"])
                expanded_case["expected_symptom"] = f"{symptom_prefix}{sub_case['symptom']}"
                expanded_case["expect"] = expect
                expanded_case["tflops_start"] = tflops_start
                if xfail_reason := sub_case.get("xfail"):
                    expanded_case["xfail"] = xfail_reason
                # No static expected_rule: each sub_case credits its own finding (the
                # bare symptom). A matrix that sets expected_rule (e.g. the deferred
                # "alias else" cases) keeps that one static rule for all sub_cases.
                if "expected_rule" not in base_case:
                    expanded_case["expected_rule"] = sub_case["symptom"]
                expanded_cases.append(expanded_case)

    return expanded_cases


def _expand_syslog_xid_matrix_case(path: Path, case_data: dict) -> list[dict]:
    description = case_data["description"]
    catalog = case_data["catalog"]
    expanded_cases = []

    for phase in case_data.get("job_phases", ["during", "outside"]):
        match phase:
            case "during":
                base_expect = "bad"
                base_rule = "known-xid-event"
                tflops_start = True
            case "outside":
                base_expect = "interesting"
                base_rule = "xid-outside-of-job-execution"
                tflops_start = False
            case other:
                raise ValueError(f"{path} contains unsupported job phase {other!r}")

        for multi_gpu in case_data.get("multi_gpu_values", [True, False]):
            for item in catalog:
                symptom = item["symptom"]
                expect = base_expect
                expected_rule = base_rule
                expected_symptom = symptom if phase == "during" else f"(non-fatal) {symptom}"
                if phase == "during" and symptom in INFORMATIONAL_XIDS:
                    expect = "interesting"
                    expected_rule = "CleanInformationalXids"
                    expected_symptom = symptom

                expanded_cases.append(
                    {
                        "description": f"{description}: {symptom} ({phase}, {'multi' if multi_gpu else 'single'}-gpu)",
                        "line": item["line"],
                        "expect": expect,
                        "expected_symptom": expected_symptom,
                        "expected_rule": expected_rule,
                        "tflops_start": tflops_start,
                        "multi_gpu": multi_gpu,
                    }
                )

    return expanded_cases


def _expand_fixed_catalog_case(path: Path, case_data: dict) -> list[dict]:
    expanded_cases = []
    for item in case_data["catalog"]:
        concrete = {k: v for k, v in case_data.items() if k not in {"description", "catalog", "expansion"}}
        concrete["description"] = f"{case_data['description']}: {item['symptom']}"
        concrete["line"] = item["line"]
        concrete["expected_symptom"] = item["symptom"]
        expanded_cases.append(concrete)
    return expanded_cases


def _iter_concrete_cases(path: Path, case_data: dict) -> list[dict]:
    if case_data.get("expansion") == "syslog_xid_matrix":
        return _expand_syslog_xid_matrix_case(path, case_data)
    if case_data.get("expansion") == "fixed_catalog":
        return _expand_fixed_catalog_case(path, case_data)
    if "line_templates" in case_data:
        return _expand_matrix_case(path, case_data)
    return [case_data]


def iter_rule_data_cases() -> list:
    params: list = []
    for path in sorted(_bundled_rules_dir().glob("*.test.toml")):
        with path.open("rb") as fh:
            payload = tomllib.load(fh)

        meta = payload.get("meta", {})
        default_runner = meta.get("runner")
        if not default_runner:
            raise ValueError(f"{path} is missing [meta].runner")

        for case_data in payload.get("cases", []):
            _warn_unknown_case_keys(path, case_data, default_runner)
            for concrete_case in _iter_concrete_cases(path, case_data):
                description = concrete_case.get("description")
                if not description:
                    raise ValueError(f"{path} contains a case without description")

                runner = concrete_case.get("runner", default_runner)
                rule_case = RuleDataCase(path=path, runner=runner, description=description, case=concrete_case)
                marks = []
                if xfail_reason := concrete_case.get("xfail"):
                    marks.append(pytest.mark.xfail(reason=xfail_reason))
                params.append(pytest.param(rule_case, id=rule_case.pytest_id, marks=marks))

    return params
