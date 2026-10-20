# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24"]
# ///
"""Release check and tag-on-merge for the components in this repository.

Needs a full clone with tags. Three subcommands:

    status  per component: version literal, latest tag (any / stable-only),
            first-parent distance since that tag, PRs merged since.
    check   validate a release PR; exit 1 unless all of these hold:
            - literal is a PEP 440 release like 2.0.0 or 2.0.0rc1
            - planned tag does not exist yet
            - every in-repo dependency is released at distance 0, with a
              stable tag when the consumer is stable
            - CHANGELOG.md has a `## [X.Y.Z]` entry
    tag     runs on main after a merge. Re-runs the `check` rules on the
            merged commit, then creates and pushes one `<component>-vX.Y.Z`
            tag per component that is releasing. Exit 1 on any FAIL.
            `--before SHA` is the push event's previous tip.

The rules are the executable form of RELEASE.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from release_git import (
    COMPONENTS,
    RELEASABLE,
    ReleaseError,
    TagInfo,
    default_repo,
    git,
    tag_for,
)
from release_rules import (
    ComponentStatus,
    Pass,
    RuleResult,
    check_release,
    is_release_pr,
    releasing_components,
    violations,
)
from release_tag import TagRow, nothing_to_tag, tag_push


def _release_pr_line(statuses: dict[str, ComponentStatus]) -> str:
    if not is_release_pr(statuses):
        return "not a release PR, nothing releasing"
    names = [name for name, s in statuses.items() if s.releasing]
    return "release PR: releasing " + (", ".join(names) if names else "none")


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _tag_cells(info: Optional[TagInfo]) -> tuple[str, str]:
    if info is None:
        return "unreleased", "unreleased"
    return info.tag, str(info.distance)


RELEASE_PR_RULE = (
    "Release PR: any component's literal differs from the base; a component is "
    "releasing when its own literal changed and did not decrease."
)
DISTANCE_RULE = (
    "Distance counts first-parent commits since the tag over the component and "
    "its in-repo dependencies; the shortlog covers the component's own path only."
)


def render_status(repo: Path, statuses: dict[str, ComponentStatus]) -> str:
    lines = [
        _release_pr_line(statuses),
        "",
        "| component | literal | base literal | latest-any | distance-any "
        "| latest-stable | distance-stable | releasing? | pre-release? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in statuses.items():
        any_tag, any_distance = _tag_cells(s.latest.any)
        stable_tag, stable_distance = _tag_cells(s.latest.stable)
        lines.append(
            f"| {name} | {s.literal} | {s.base_literal or '-'} | {any_tag} "
            f"| {any_distance} | {stable_tag} | {stable_distance} "
            f"| {_yes_no(s.releasing)} | {_yes_no(s.is_prerelease)} |"
        )
    lines.append("")
    for name, s in statuses.items():
        if s.decreased:
            lines.append(
                f"{name}: literal decreased relative to base; not releasing, "
                "the tag job will not tag"
            )
        if s.releasing and RELEASABLE.match(s.literal):
            lines.append(f"{name}: planned tag {tag_for(name, s.literal)}")
    for name, s in statuses.items():
        since = s.latest.any.tag if s.latest.any is not None else None
        rev_range = f"{since}..HEAD" if since is not None else "HEAD"
        shortlog = git(
            repo, "shortlog", "--first-parent", rev_range, "--", COMPONENTS[name].path
        ).rstrip()
        lines += [
            "",
            f"### {name}: PRs since {since or 'start of history'}",
            "",
            "```",
            shortlog or "(none)",
            "```",
        ]
    lines += ["", DISTANCE_RULE, RELEASE_PR_RULE]
    return "\n".join(lines)


def render_check(
    statuses: dict[str, ComponentStatus], results: list[RuleResult]
) -> str:
    lines = [_release_pr_line(statuses), ""]
    for name, s in statuses.items():
        if s.decreased:
            lines.append(
                f"{name}: literal decreased relative to base; not releasing, "
                "the tag job will not tag"
            )
    # Every rule outcome is listed, passes included, so a rule that passes for
    # the wrong reason is visible in the log rather than silent.
    for r in results:
        if isinstance(r, Pass):
            lines.append(f"PASS      {r.component}: {r.reason}")
            continue
        lines.append(f"VIOLATION {r.component}: {r.message}")
        if r.log:
            lines += ["", "```", r.log, "```", ""]
    if len(violations(results)) == 0:
        lines.append("PASS: no violations")
    lines += ["", RELEASE_PR_RULE]
    return "\n".join(lines)


def render_tag(rows: list[TagRow]) -> str:
    lines = ["nothing to tag", ""] if nothing_to_tag(rows) else []
    lines += [
        "| component | old | new | tag | outcome | detail |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.component} | {r.old or ''} | {r.new} | {r.tag or ''} "
            f"| {r.outcome} | {r.detail} |"
        )
    return "\n".join(lines)


def _emit(text: str) -> None:
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(text + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("status", "print literals, latest tags, distance and shortlogs"),
        ("check", "validate the release PR; exit 1 on any violation"),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument(
            "--base",
            default="origin/main",
            help="ref whose merge-base with HEAD the literals are compared against",
        )
        sub.add_argument("--repo", type=Path, default=None, help="repository root")

    tag = subparsers.add_parser(
        "tag", help="tag and push released components; main only, exit 1 on any FAIL"
    )
    tag.add_argument("--repo", type=Path, default=None, help="repository root")

    tag.add_argument(
        "--before",
        metavar="SHA",
        required=True,
        help="push event: tag literals that moved since SHA",
    )
    args = parser.parse_args(argv)

    try:
        repo = args.repo if args.repo is not None else default_repo()
        match args.command:
            case "status":
                statuses = releasing_components(repo, args.base)
                report, failed = render_status(repo, statuses), False
            case "check":
                statuses = releasing_components(repo, args.base)
                results = check_release(repo, statuses)
                report, failed = (
                    render_check(statuses, results),
                    bool(violations(results)),
                )
            case "tag":
                rows = tag_push(repo, args.before)
                report = render_tag(rows)
                failed = any(r.outcome == "FAIL" for r in rows)
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(report)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
