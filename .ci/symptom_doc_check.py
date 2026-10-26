# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the real symptom/catalog contract without introducing new historic gaps.

Snapshots contain public identifiers only, not guidance or source code. Consumers
must treat snapshots as untrusted data and bind them to the tested commit.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import yaml

from attribution_lib.symptoms.declarations import SymptomDeclaration
from attribution_lib.symptoms.doc_validation import (
    build_doc_name_index,
    find_undeclared_symptom_names,
    find_unresolved_symptom_docs,
    symptom_requires_doc,
)

CATALOG = "blueprint/src/attribution_blueprint/data/failures.yml"
SHA = re.compile(r"[0-9a-f]{40}")
DYNAMIC_CASES = {
    "NV_ERR_RESET_REQUIRED kernel_channel.c": "gpu-reset-required",
    "NV_ERR_NO_MEMORY mem_multicast_fabric.c": "nv-err-no-memory",
    "NV_ERR_INVALID_STATE nv_gpu_ops.c": "nv-err-invalid-state",
    "NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c": "nv-err-insufficient-resources",
    "exit code signal 15": "exit-code-signal",
    "IB Error-12 (IBV_WC_RETRY_EXC_ERR)": "ib-error-12",
    "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)": "ib-error-12-vendor-129",
}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def collect(root: Path) -> dict:
    # Only the credential-free producer imports and executes candidate rules.
    import attribution_lib
    from attribution_lib.core import setup_rules

    setup_rules(force=True)
    declarations = sorted(
        (
            {"prefix": d.prefix, "doc": d.doc, "outcome": d.outcome}
            for d in attribution_lib._rules_engine.symptom_declarations
            if symptom_requires_doc(d)
        ),
        key=lambda d: d["prefix"],
    )
    if not declarations:
        raise ValueError("the registry did not produce any documentable symptoms")
    source = yaml.safe_load((root / CATALOG).read_text())
    if not isinstance(source, dict) or not isinstance(source.get("failures"), list):
        raise ValueError("catalog must contain a failures list")
    documents = []
    for row in source["failures"]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
        ):
            raise ValueError("catalog entry must have a nonempty id")
        documents.append(
            {k: row[k] for k in ("id", "title", "aliases", "redirect_ids") if k in row}
        )
    build_doc_name_index(documents)
    dynamic = []
    registry = attribution_lib.get_symptom_registry()
    for name, expected in DYNAMIC_CASES.items():
        match = registry.outcome_for([name])
        dynamic.append(
            {
                "name": name,
                "expected_doc": expected,
                "doc": match.doc if match else None,
                "prefix": match.matched_prefix if match else None,
            }
        )
    return {
        "schema_version": 1,
        "commit_sha": git(root, "rev-parse", "HEAD"),
        "declarations": declarations,
        "documents": documents,
        "dynamic": dynamic,
    }


def gaps(snapshot: dict) -> set[tuple[str, str, str, str]]:
    """Identify exact gaps, including outcome so changed guidance is rechecked."""
    declarations = [SymptomDeclaration(**row) for row in snapshot["declarations"]]
    index = build_doc_name_index(snapshot["documents"])
    # The public UI indexes canonical catalog IDs only. Documentation-site
    # redirects do not make an old ID usable as a verdict's catalog_id.
    canonical_ids = {row["id"] for row in snapshot["documents"]}
    index = {
        doc_id: names for doc_id, names in index.items() if doc_id in canonical_ids
    }
    result = set()
    for kind, rows in (
        ("missing-document", find_unresolved_symptom_docs(declarations, set(index))),
        ("missing-name", find_undeclared_symptom_names(declarations, index)),
    ):
        for row in rows:
            result.add((kind, row.prefix, row.doc or "", row.outcome or ""))
    for row in snapshot["dynamic"]:
        if row["doc"] != row["expected_doc"] or row["prefix"] is None:
            result.add(
                ("dynamic-target", row["name"], row["doc"] or "", row["expected_doc"])
            )
        elif row["doc"] not in index:
            result.add(("dynamic-document", row["name"], row["doc"], row["prefix"]))
        elif row["prefix"] not in index[row["doc"]]:
            result.add(("dynamic-name", row["name"], row["doc"], row["prefix"]))
    return result


def compare(base: dict, head: dict) -> list[tuple[str, str, str, str]]:
    return sorted(gaps(head) - gaps(base))


def snapshot_process(root: Path, destination: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "service/attribution-lib/src")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "snapshot",
            "--root",
            str(root),
            "--output",
            str(destination),
        ],
        env=environment,
        cwd=root,
        check=True,
        timeout=180,
    )


def report_to_github(report: dict) -> None:
    """Render public findings safely as Actions annotations and a job summary."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    status = report["status"]
    lines = ["## Documentation coverage", ""]
    if status == "error":
        lines.append(
            "Check unavailable. Documentation coverage could not be evaluated."
        )
        print(
            "::error title=Documentation coverage::Check unavailable; inspect the failed validation step."
        )
    else:
        findings = report["new_gaps"]
        lines.append(f"Tested commit: `{report['head_sha']}`")
        lines.append("")
        if findings:
            explanation = "Documentation needs attention."
            if status == "warning":
                explanation += " This check is advisory and does not block merging."
            lines.extend(
                [
                    explanation,
                    "",
                    "| Symptom | Document ID | Finding |",
                    "| --- | --- | --- |",
                ]
            )
            for kind, name, doc, _ in findings:
                # Escape workflow-command data separately from Markdown. JSON
                # quoting prevents control characters from creating extra rows.
                message = (
                    f"Documentation coverage needs attention for {json.dumps(name)} "
                    f"(document {json.dumps(doc)}, {kind}). Add or update the public "
                    "catalog entry or mapping, or ask a maintainer to coordinate the update."
                )
                message = (
                    message.replace("%", "%25")
                    .replace("\r", "%0D")
                    .replace("\n", "%0A")
                )
                print(
                    f"::warning file={CATALOG},title=Documentation coverage::{message}"
                )
                cells = [
                    html.escape(json.dumps(value))
                    .replace("|", "&#124;")
                    .replace("`", "&#96;")
                    for value in (name, doc, kind)
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.extend(
                [
                    "",
                    f"Add or update `{CATALOG}` or correct the rule mapping. "
                    "Ask a maintainer if you need help coordinating the documentation. "
                    "Human review is still required.",
                ]
            )
        else:
            lines.append("No new documentation coverage gaps found.")
        lines.extend(
            [
                "",
                f"Existing baseline gaps: {report['existing_gap_count']} (not introduced by this change).",
            ]
        )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as stream:
            stream.write("\n".join(lines) + "\n")


def check(root: Path, base_sha: str, output: Path, *, advisory: bool = False) -> int:
    output.mkdir(parents=True, exist_ok=True)
    # These are this command's outputs, not inputs. Clear every previous run's
    # artifact before doing work so an interrupted/failed run cannot reuse it.
    for name in ("result.json", "head.json", "base.json"):
        (output / name).unlink(missing_ok=True)
    if SHA.fullmatch(base_sha) is None:
        raise ValueError("base must be a full commit SHA")
    if git(root, "rev-parse", f"{base_sha}^{{commit}}") != base_sha:
        raise ValueError("base does not resolve to the requested commit")
    head_sha = git(root, "rev-parse", "HEAD")
    # Compare committed source in CI. Local unstaged changes to the checker are
    # useful during development, but the snapshot is not a signed attestation.
    head_path = output / "head.json"
    snapshot_process(root, head_path)
    with tempfile.TemporaryDirectory(prefix="symptom-doc-base-") as temporary:
        base_root = Path(temporary) / "checkout"
        git(root, "worktree", "add", "--detach", str(base_root), base_sha)
        try:
            snapshot_process(base_root, output / "base.json")
        finally:
            git(root, "worktree", "remove", "--force", str(base_root))
    base = json.loads((output / "base.json").read_text())
    head = json.loads(head_path.read_text())
    if base["commit_sha"] != base_sha or head["commit_sha"] != head_sha:
        raise ValueError("snapshot does not match the requested commit")
    introduced = compare(base, head)
    report = {
        "schema_version": 1,
        "status": ("warning" if advisory else "failure") if introduced else "success",
        "base_sha": base_sha,
        "head_sha": head_sha,
        "existing_gap_count": len(gaps(base)),
        "new_gaps": introduced,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Existing gaps: {report['existing_gap_count']}; new gaps: {len(introduced)}")
    for kind, name, doc, _ in introduced:
        # JSON quoting keeps raw symptom text from becoming workflow commands.
        print(json.dumps({"gap": kind, "symptom": name, "doc": doc}))
    if introduced:
        print(
            f"Update the document id/title/aliases in {CATALOG}, or correct the rule mapping."
        )
    report_to_github(report)
    return 1 if introduced and not advisory else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--root", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("check")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--base", required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument(
        "--advisory",
        action="store_true",
        help="Warn about coverage gaps without failing; operational errors still fail.",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            args.output.write_text(
                json.dumps(collect(args.root.resolve()), indent=2) + "\n"
            )
            return 0
        return check(
            args.root.resolve(),
            args.base,
            args.output.resolve(),
            advisory=args.advisory,
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
        subprocess.SubprocessError,
        yaml.YAMLError,
    ) as exc:
        if args.command == "check":
            # Never leave a previous successful result behind after a failed run.
            try:
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / "result.json").write_text(
                    json.dumps({"schema_version": 1, "status": "error"}) + "\n"
                )
            except OSError:
                pass  # The nonzero exit still prevents a successful CI result.
        print(
            f"Symptom documentation check could not complete: {type(exc).__name__}",
            file=sys.stderr,
        )
        if args.command == "check":
            try:
                report_to_github({"status": "error"})
            except OSError:
                pass  # A summary failure must not replace the original failure.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
