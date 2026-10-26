# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "symptom_doc_check", ROOT / ".ci/symptom_doc_check.py"
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def snapshot(name="Example failure", doc="example-failure", *, documented=True):
    return {
        "schema_version": 1,
        "commit_sha": "a" * 40,
        "declarations": [{"prefix": name, "doc": doc, "outcome": "Fail InfraSW"}],
        "documents": [{"id": doc, "title": name}] if documented else [],
        "dynamic": [],
    }


def test_existing_gaps_do_not_break_unrelated_changes():
    base = snapshot(documented=False)
    head = copy.deepcopy(base)
    head["documents"].append({"id": "other", "title": "Other"})
    assert check.compare(base, head) == []


def test_new_missing_entry_is_reported():
    base = snapshot()
    head = copy.deepcopy(base)
    head["declarations"].append(
        {"prefix": "New failure", "doc": "missing", "outcome": "Fail InfraHW"}
    )
    assert check.compare(base, head) == [
        ("missing-document", "New failure", "missing", "Fail InfraHW")
    ]


@pytest.mark.parametrize(
    "field,value",
    [("prefix", "Renamed"), ("doc", "another-page"), ("outcome", "Fail User")],
)
def test_changed_declaration_cannot_reuse_an_old_gap(field, value):
    base = snapshot(documented=False)
    head = copy.deepcopy(base)
    head["declarations"][0][field] = value
    assert len(check.compare(base, head)) == 1


def test_existing_but_unrelated_page_fails_and_alias_fixes_it():
    base = snapshot()
    head = copy.deepcopy(base)
    head["declarations"][0]["prefix"] = "Another spelling"
    assert check.compare(base, head)[0][0] == "missing-name"
    head["documents"][0]["aliases"] = ["Another spelling"]
    assert check.compare(base, head) == []


def test_removing_document_or_alias_is_a_regression():
    base = snapshot(name="Alias")
    base["documents"][0].update(title="Canonical", aliases=["Alias"])
    head = copy.deepcopy(base)
    head["documents"][0]["aliases"] = []
    assert check.compare(base, head)[0][0] == "missing-name"
    head["documents"] = []
    assert check.compare(base, head)[0][0] == "missing-document"


def test_adding_documentation_clears_existing_gap():
    assert check.compare(snapshot(documented=False), snapshot()) == []


def test_redirect_does_not_replace_a_canonical_catalog_id():
    base = snapshot(doc="canonical-id")
    base["documents"][0]["redirect_ids"] = ["old-id"]
    assert check.gaps(base) == set()
    head = copy.deepcopy(base)
    head["declarations"][0]["doc"] = "old-id"
    assert check.compare(base, head) == [
        ("missing-document", "Example failure", "old-id", "Fail InfraSW")
    ]
    head["declarations"][0]["doc"] = "canonical-id"
    assert check.compare(base, head) == []


def test_dynamic_redirect_is_not_a_resolving_public_catalog_id():
    data = snapshot(doc="canonical-id")
    data["documents"][0]["redirect_ids"] = ["old-id"]
    data["dynamic"] = [
        {
            "name": "Example failure suffix",
            "expected_doc": "old-id",
            "doc": "old-id",
            "prefix": "Example failure",
        }
    ]
    assert check.gaps(data) == {
        ("dynamic-document", "Example failure suffix", "old-id", "Example failure")
    }


@pytest.mark.parametrize(
    "field,value,kind",
    [
        ("doc", "unrelated", "dynamic-target"),
        ("prefix", None, "dynamic-target"),
        ("prefix", "Wrong name", "dynamic-name"),
    ],
)
def test_dynamic_mapping_regressions(field, value, kind):
    base = snapshot()
    base["dynamic"] = [
        {
            "name": "Example failure suffix",
            "expected_doc": "example-failure",
            "doc": "example-failure",
            "prefix": "Example failure",
        }
    ]
    head = copy.deepcopy(base)
    head["dynamic"][0][field] = value
    assert check.compare(base, head)[0][0] == kind


def test_dynamic_document_removal_is_detected():
    base = snapshot()
    base["dynamic"] = [
        {
            "name": "Example failure suffix",
            "expected_doc": "example-failure",
            "doc": "example-failure",
            "prefix": "Example failure",
        }
    ]
    head = copy.deepcopy(base)
    head["documents"] = []
    assert {x[0] for x in check.compare(base, head)} == {
        "missing-document",
        "dynamic-document",
    }


def test_invalid_catalog_does_not_pass():
    head = snapshot()
    head["documents"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        check.compare(snapshot(), head)


def test_invalid_base_does_not_report_success(tmp_path):
    (tmp_path / "result.json").write_text('{"status": "success"}')
    (tmp_path / "head.json").write_text("old contract")
    (tmp_path / "base.json").write_text("old baseline")
    assert (
        check.main(["check", "--base", "main; echo bad", "--output", str(tmp_path)])
        == 2
    )
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "error"
    assert not (tmp_path / "head.json").exists()
    assert not (tmp_path / "base.json").exists()


def test_workflow_preserves_credential_free_candidate_execution():
    workflow = yaml.load(
        (ROOT / ".github/workflows/tests.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert "pull_request_target" not in workflow["on"]
    assert "pull_request" not in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["symptom-docs"]
    assert "secret-scan" in job["needs"]
    assert "secrets." not in json.dumps(job)
    checkout = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["persist-credentials"] == "false"


def test_workflow_uses_advisory_without_masking_operational_failures():
    workflow = yaml.load(
        (ROOT / ".github/workflows/tests.yml").read_text(), Loader=yaml.BaseLoader
    )
    job = workflow["jobs"]["symptom-docs"]
    assert job["name"] == "Documentation coverage"
    validation = next(
        s for s in job["steps"] if s.get("name") == "Check symptom documentation"
    )
    assert validation["env"]["SYMPTOM_DOC_ADVISORY"] == "true"
    assert "continue-on-error" not in job
    assert "continue-on-error" not in validation
    assert "|| true" not in validation["run"]


@pytest.mark.parametrize("advisory", ["false", "true"])
def test_suite_runner_passes_advisory_only_when_requested(tmp_path, advisory):
    """Exercise the Bash entry point, including macOS Bash 3's nounset rules."""
    environment_file = tmp_path / "bash-env"
    log = tmp_path / "arguments"
    environment_file.write_text('uv() { printf "%s\\n" "$@" >> "$RUNNER_ARG_LOG"; }\n')
    result = subprocess.run(
        ["bash", str(ROOT / ".ci/run-tests.sh"), "symptom-docs"],
        env={
            **os.environ,
            "BASH_ENV": str(environment_file),
            "RUNNER_ARG_LOG": str(log),
            "SYMPTOM_DOC_BASE": "a" * 40,
            "SYMPTOM_DOC_ADVISORY": advisory,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    arguments = log.read_text().splitlines()
    assert "check" in arguments and "a" * 40 in arguments
    assert ("--advisory" in arguments) == (advisory == "true")


def test_workflow_uploads_only_explicit_contract_files():
    workflow = yaml.load(
        (ROOT / ".github/workflows/tests.yml").read_text(), Loader=yaml.BaseLoader
    )
    upload = next(
        step
        for step in workflow["jobs"]["symptom-docs"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["name"] == "symptom-doc-contract"
    assert upload["with"]["include-hidden-files"] == "true"
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["path"].splitlines() == [
        ".symptom-doc-check/head.json",
        ".symptom-doc-check/base.json",
        ".symptom-doc-check/result.json",
    ]


@pytest.fixture
def change_repo(tmp_path):
    root = tmp_path / "changes"
    root.mkdir()
    check.git(root, "init", "--template=", "--initial-branch=main")
    check.git(root, "config", "core.hooksPath", "/dev/null")
    (root / "README.md").write_text("Example repository\n")
    base = commit(root, "Initial source")
    check.git(root, "update-ref", "refs/remotes/origin/main", base)
    return root, base


def detect_changes(root, base, tmp_path, *, event="push", branch="feature"):
    """Execute the workflow's actual path detection against committed changes."""
    workflow = yaml.load(
        (ROOT / ".github/workflows/tests.yml").read_text(), Loader=yaml.BaseLoader
    )
    steps = workflow["jobs"]["symptom-docs"]["steps"]
    detect = next(step for step in steps if step.get("id") == "changes")
    output = tmp_path / "github-output"
    summary = tmp_path / "github-summary"
    result = subprocess.run(
        ["bash", "-c", detect["run"]],
        cwd=root,
        env={
            **os.environ,
            "BEFORE_SHA": base,
            "EVENT_NAME": event,
            "GITHUB_REF_NAME": branch,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
    )
    values = (
        dict(line.split("=", 1) for line in output.read_text().splitlines())
        if output.exists()
        else {}
    )
    return result, values, summary.read_text() if summary.exists() else ""


@pytest.mark.parametrize(
    "path,relevant",
    [
        ("service/attribution-lib/src/attribution_lib/rule.py", True),
        ("service/attribution-lib/pyproject.toml", True),
        (check.CATALOG, True),
        (".ci/symptom_doc_check.py", True),
        (".ci/symptom-docs-tests/test_symptom_doc_check.py", True),
        (".ci/run-tests.sh", True),
        (".github/workflows/tests.yml", True),
        ("README.md", False),
        ("docs/setup.md", False),
        ("blueprint/src/attribution_blueprint/webui/frontend/example.ts", False),
    ],
)
def test_workflow_validates_only_relevant_changes(
    change_repo, tmp_path, path, relevant
):
    root, base = change_repo
    changed = root / path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("Updated source\n")
    commit(root, "Change source")
    result, outputs, summary = detect_changes(root, base, tmp_path)
    assert result.returncode == 0, result.stderr
    assert outputs == {"base_sha": base, "run": str(relevant).lower()}
    if not relevant:
        assert "documentation validation skipped" in summary


@pytest.mark.parametrize("operation", ["delete", "rename"])
def test_workflow_does_not_skip_catalog_removal(change_repo, tmp_path, operation):
    root, _ = change_repo
    catalog = root / check.CATALOG
    catalog.parent.mkdir(parents=True)
    catalog.write_text("failures: []\n")
    base = commit(root, "Add catalog")
    if operation == "delete":
        catalog.unlink()
    else:
        catalog.rename(root / "moved-catalog.yml")
    commit(root, "Remove catalog from its required location")
    result, outputs, _ = detect_changes(root, base, tmp_path, branch="main")
    assert result.returncode == 0, result.stderr
    assert outputs["run"] == "true"
    assert outputs["base_sha"] == base


def test_manual_run_validates_even_without_changes(change_repo, tmp_path):
    root, base = change_repo
    result, outputs, _ = detect_changes(root, base, tmp_path, event="workflow_dispatch")
    assert result.returncode == 0, result.stderr
    assert outputs == {"base_sha": base, "run": "true"}


def test_missing_comparison_commit_is_an_error_not_a_skip(change_repo, tmp_path):
    root, _ = change_repo
    result, outputs, summary = detect_changes(root, "0" * 40, tmp_path, branch="main")
    assert result.returncode != 0
    assert "run" not in outputs
    assert "skipped" not in summary


def test_real_snapshot_contains_only_document_metadata(tmp_path):
    result = check.collect(ROOT)
    assert result["declarations"]
    assert len(result["dynamic"]) == len(check.DYNAMIC_CASES)
    assert all(
        set(row) <= {"id", "title", "aliases", "redirect_ids"}
        for row in result["documents"]
    )
    assert all(
        set(row) == {"prefix", "doc", "outcome"} for row in result["declarations"]
    )
    assert "tenant_sre" not in json.dumps(result)
    assert "support_details" not in json.dumps(result)
    assert check.compare(result, result) == []


def test_saved_contract_matches_real_producer():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/public-symptom-contract.json").read_text()
    )
    actual = check.collect(ROOT)
    assert actual.keys() == fixture.keys()
    assert actual["schema_version"] == fixture["schema_version"]
    prefixes = {r["prefix"] for r in fixture["declarations"]}
    ids = {r["id"] for r in fixture["documents"]}
    assert [r for r in actual["declarations"] if r["prefix"] in prefixes] == fixture[
        "declarations"
    ]
    assert [r for r in actual["documents"] if r["id"] in ids] == fixture["documents"]
    assert actual["dynamic"] == fixture["dynamic"]


def commit(root, message):
    check.git(root, "add", ".")
    check.git(
        root,
        "-c",
        "user.name=Test Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--no-gpg-sign",
        "-m",
        message,
    )
    return check.git(root, "rev-parse", "HEAD")


@pytest.fixture
def committed_repo(tmp_path):
    """A real disposable repository containing the real registry and catalog."""
    root = tmp_path / "repo"
    root.mkdir()
    check.git(root, "init", "--template=", "--initial-branch=main")
    check.git(root, "config", "core.hooksPath", "/dev/null")
    shutil.copytree(
        ROOT / "service/attribution-lib/src",
        root / "service/attribution-lib/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(
        ROOT / "service/attribution-lib/LICENSE",
        root / "service/attribution-lib/LICENSE",
    )
    (root / check.CATALOG).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / check.CATALOG, root / check.CATALOG)
    shutil.copyfile(ROOT / "blueprint/LICENSE", root / "blueprint/LICENSE")
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    return root, commit(root, "Initial test registry and catalog")


@pytest.mark.parametrize("operation", ["remove", "redirect"])
def test_real_commit_comparison_and_worktree_cleanup(
    committed_repo, tmp_path, operation
):
    root, base = committed_repo
    catalog = root / check.CATALOG
    original = catalog.read_text()
    payload = yaml.safe_load(original)
    if operation == "remove":
        payload["failures"] = [
            r for r in payload["failures"] if r["id"] != "detected-bad-nic-state"
        ]
    else:
        entry = next(
            r for r in payload["failures"] if r["id"] == "detected-bad-nic-state"
        )
        entry["id"] = "renamed-nic-state"
        entry["redirect_ids"] = ["detected-bad-nic-state"]
    catalog.write_text(yaml.safe_dump(payload, sort_keys=False))
    head = commit(root, "Remove a required documentation entry")
    output = tmp_path / "output"
    args = ["check", "--root", str(root), "--base", base, "--output", str(output)]
    assert check.main(args) == 1
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "failure"
    assert report["base_sha"] == base and report["head_sha"] == head
    assert report["new_gaps"] == [
        [
            "missing-document",
            "detected bad NIC state",
            "detected-bad-nic-state",
            "Fail InfraSW",
        ]
    ]
    assert json.loads((output / "head.json").read_text())["commit_sha"] == head
    assert json.loads((output / "base.json").read_text())["commit_sha"] == base
    assert check.git(root, "worktree", "list", "--porcelain").count("worktree ") == 1
    catalog.write_text(original)
    fixed_head = commit(root, "Restore the documentation entry")
    assert check.main(args) == 0
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "success" and report["new_gaps"] == []
    assert report["head_sha"] == fixed_head


def test_advisory_missing_then_corrected_real_catalog(
    committed_repo, tmp_path, monkeypatch, capsys
):
    root, base = committed_repo
    catalog = root / check.CATALOG
    original = catalog.read_text()
    payload = yaml.safe_load(original)
    payload["failures"] = [
        r for r in payload["failures"] if r["id"] != "detected-bad-nic-state"
    ]
    catalog.write_text(yaml.safe_dump(payload, sort_keys=False))
    head = commit(root, "Remove required documentation for advisory test")
    output = tmp_path / "output"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = [
        "check",
        "--root",
        str(root),
        "--base",
        base,
        "--output",
        str(output),
        "--advisory",
    ]
    assert check.main(args) == 0
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "warning"
    assert report["head_sha"] == head
    assert "::warning file=" in capsys.readouterr().out
    text = summary.read_text()
    assert "detected-bad-nic-state" in text
    assert "ask a maintainer" in text.lower()
    assert "does not block merging" in text
    catalog.write_text(original)
    fixed = commit(root, "Restore the advisory documentation")
    # Actions gives each run a new summary file.
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "fixed-summary.md"))
    assert check.main(args) == 0
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "success" and report["head_sha"] == fixed
    assert "::warning" not in capsys.readouterr().out
    assert (
        "No new documentation coverage gaps"
        in (tmp_path / "fixed-summary.md").read_text()
    )


def test_advisory_invalid_input_is_unavailable_not_pass(tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert (
        check.main(
            ["check", "--base", "invalid", "--output", str(tmp_path), "--advisory"]
        )
        == 2
    )
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "error"
    assert "Check unavailable" in summary.read_text()
    assert "::error title=Documentation coverage::" in capsys.readouterr().out


def test_annotation_and_summary_escape_untrusted_symptom_text(
    tmp_path, monkeypatch, capsys
):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    check.report_to_github(
        {
            "status": "warning",
            "head_sha": "a" * 40,
            "existing_gap_count": 0,
            "new_gaps": [
                ("missing-name", "x\n::error::injected%0A|`<b>", "doc", "Fail User")
            ],
        }
    )
    annotations = capsys.readouterr().out.splitlines()
    assert len(annotations) == 1 and annotations[0].startswith("::warning ")
    assert "%250A" in annotations[0]
    text = summary.read_text()
    assert "<b>" not in text and "&#124;" in text and "&#96;" in text


def test_no_actions_commands_outside_github(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    check.report_to_github({"status": "error"})
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure", ["base-process", "wrong-sha", "missing-sha"])
def test_failed_real_comparison_clears_success_and_cleans_worktree(
    committed_repo, tmp_path, monkeypatch, failure
):
    root, base = committed_repo
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.json").write_text('{"status":"success"}')
    real_snapshot = check.snapshot_process

    def snapshot_with_failure(checkout, destination):
        if destination.name == "base.json" and failure == "base-process":
            raise subprocess.CalledProcessError(1, "snapshot")
        real_snapshot(checkout, destination)
        if destination.name == "base.json":
            data = json.loads(destination.read_text())
            if failure == "wrong-sha":
                data["commit_sha"] = "0" * 40
            else:
                del data["commit_sha"]
            destination.write_text(json.dumps(data))

    monkeypatch.setattr(check, "snapshot_process", snapshot_with_failure)
    assert (
        check.main(
            ["check", "--root", str(root), "--base", base, "--output", str(output)]
        )
        == 2
    )
    assert json.loads((output / "result.json").read_text())["status"] == "error"
    assert check.git(root, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_real_producer_preserves_redirect_ids(committed_repo):
    root, _ = committed_repo
    catalog = root / check.CATALOG
    catalog.write_text(
        yaml.safe_dump(
            {
                "failures": [
                    {
                        "id": "canonical-id",
                        "title": "Example failure",
                        "redirect_ids": ["old-id"],
                    }
                ]
            }
        )
    )
    assert check.collect(root)["documents"][0]["redirect_ids"] == ["old-id"]


def test_interruption_cannot_leave_previous_success(tmp_path, monkeypatch):
    (tmp_path / "result.json").write_text('{"status":"success"}')

    def interrupted(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(check, "snapshot_process", interrupted)
    with pytest.raises(KeyboardInterrupt):
        check.main(
            [
                "check",
                "--root",
                str(ROOT),
                "--base",
                check.git(ROOT, "rev-parse", "HEAD"),
                "--output",
                str(tmp_path),
            ]
        )
    assert not (tmp_path / "result.json").exists()
