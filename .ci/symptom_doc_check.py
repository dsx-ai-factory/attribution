# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the real symptom/catalog contract without introducing new historic gaps.

Snapshots contain symptom and document identifiers, not guidance or source code.
Consumers must treat snapshots as untrusted data and bind them to the tested commit.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import NamedTuple

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
# Bound each local Git operation within the 15-minute CI job budget, allowing
# filesystem work to finish while leaving time to report an unavailable check.
GIT_TIMEOUT_SECONDS = 120
# Allow registry initialization and catalog validation within the CI job budget.
SNAPSHOT_TIMEOUT_SECONDS = 180


class DocumentationGap(NamedTuple):
    """Identify a coverage gap without changing JSON array serialization."""

    kind: str
    symptom: str
    document: str
    outcome: str


class _WorkflowSafeFormatter(logging.Formatter):
    """Keep tracebacks readable without emitting embedded Actions commands."""

    def format(self, record: logging.LogRecord) -> str:
        """Escape command delimiters in exception messages and source lines."""
        return super().format(record).replace("::", r"\u003a\u003a")


def git(root: Path, *args: str) -> str:
    """Return stripped Git output; propagate command failures and bounded timeouts."""
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, timeout=GIT_TIMEOUT_SECONDS
    ).strip()


def collect(root: Path) -> dict:
    """Load rules from the active Python environment and catalog metadata from root.

    Return identifiers bound to root's HEAD, without guidance or rule code.
    The caller must select root's library in PYTHONPATH when isolating a checkout.
    """
    # Only the credential-free producer imports and executes candidate rules.
    import attribution_lib
    from attribution_lib.core import setup_rules

    setup_rules(force=True)
    declarations = sorted(
        (
            {"prefix": d.prefix, "doc": d.doc, "outcome": d.outcome}
            for d in attribution_lib.get_rules_engine().symptom_declarations
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
    return {
        "schema_version": 1,
        "commit_sha": git(root, "rev-parse", "HEAD"),
        "declarations": declarations,
        "documents": documents,
        # Retained empty for schema-v1 consumers. Rule-specific runtime examples
        # belong to attribution-lib tests, not the repository coverage producer.
        "dynamic": [],
    }


def gaps(snapshot: dict) -> set[DocumentationGap]:
    """Identify exact gaps, including outcome so changed guidance is rechecked."""
    declarations = [SymptomDeclaration(**row) for row in snapshot["declarations"]]
    index = build_doc_name_index(snapshot["documents"])
    # The UI indexes canonical catalog IDs only. Documentation-site
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
            result.add(
                DocumentationGap(kind, row.prefix, row.doc or "", row.outcome or "")
            )
    return result


def compare(base: dict, head: dict) -> list[DocumentationGap]:
    """Return sorted head gaps absent from base, leaving historical gaps unchanged."""
    return sorted(gaps(head) - gaps(base))


def snapshot_process(root: Path, destination: Path) -> None:
    """Write a checkout's snapshot in an isolated, time-bounded Python subprocess.

    Select the checkout's library via PYTHONPATH; propagate failures and timeouts.
    """
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
        timeout=SNAPSHOT_TIMEOUT_SECONDS,
    )


def report_to_github(report: dict) -> None:
    """Render coverage findings safely as Actions annotations and a job summary."""
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
                    f"(document {json.dumps(doc)}, {kind}). Add or update the "
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
    """Compare base and head coverage and write commit-bound reports.

    Return 1 for new gaps unless advisory mode is selected, otherwise return 0.
    Propagate operational errors after cleaning up the temporary base worktree.
    """
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
    """Run the CLI, returning 2 with diagnostics for handled operational failures."""
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
    ):
        if args.command == "check":
            # Never leave a previous successful result behind after a failed run.
            try:
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / "result.json").write_text(
                    json.dumps({"schema_version": 1, "status": "error"}) + "\n"
                )
            except OSError:
                pass  # The nonzero exit still prevents a successful CI result.
        # A local logger avoids changing rule logging or accumulating handlers
        # when this entry point is called repeatedly by tests or consumers.
        logger = logging.Logger(__name__)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_WorkflowSafeFormatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.exception("Symptom documentation check could not complete")
        if args.command == "check":
            try:
                report_to_github({"status": "error"})
            except OSError:
                pass  # A summary failure must not replace the original failure.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
