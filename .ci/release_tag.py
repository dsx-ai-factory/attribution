# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tag-on-merge: tag and push every component whose literal moved, once the rules pass.

`tag_push` compares each literal against the push's `before` SHA, then for every
candidate: name the tag, dedupe against origin, run the bundle rules, tag, verify,
push. A rerun finds its tags on origin and reports DONE.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Literal, Optional

from release_git import (
    COMPONENTS,
    ReleaseError,
    git,
    tag_for,
)
from release_rules import (
    ComponentStatus,
    check_release,
    is_release_pr,
    releasing_components,
    violations,
)

Outcome = Literal["TAGGED", "DONE", "NOOP", "SKIP", "FAIL"]

# Loaded by path, never imported: the tag job must not import a component package.
HELPER_FILES: dict[str, str] = {
    "attribution-lib": "service/attribution-lib/src/_build_version.py",
    "service-app": "service/service-app/_build_version.py",
    "blueprint": "blueprint/src/_build_version.py",
}


@dataclasses.dataclass(frozen=True)
class TagRow:
    """One line of the step-summary table: what happened to one component."""

    component: str
    """Component name like "service-app"."""
    old: Optional[str]
    """Literal at `before`; None when the component did not exist there yet."""
    new: str
    """Literal at HEAD."""
    tag: Optional[str]
    """Tag created or found for `new`; None when nothing was tagged."""
    outcome: Outcome
    """TAGGED pushed a tag; DONE found it already; NOOP/SKIP tagged nothing; FAIL stopped."""
    detail: str
    """Human-readable reason, e.g. "pushed service-app-v2.0.0" or a rule message."""


def _row(
    status: ComponentStatus, tag: Optional[str], outcome: Outcome, detail: str
) -> TagRow:
    return TagRow(
        status.name, status.base_literal, status.literal, tag, outcome, detail
    )


def _ordered(rows: dict[str, TagRow]) -> list[TagRow]:
    return [rows[name] for name in COMPONENTS if name in rows]


def _remote_tag_sha(repo: Path, tag: str) -> Optional[str]:
    """SHA of the commit `tag` marks on origin; None when origin has no such tag."""
    # Annotated tags point at a tag object; the `^{}` (peeled) line has the commit.
    out = git(
        repo,
        "ls-remote",
        "--tags",
        "origin",
        f"refs/tags/{tag}",
        f"refs/tags/{tag}^{{}}",
    )
    shas: dict[str, str] = {}
    for line in out.splitlines():
        sha, _, ref = line.partition("\t")
        shas[ref] = sha
    return shas.get(f"refs/tags/{tag}^{{}}", shas.get(f"refs/tags/{tag}"))


def _build_version(repo: Path, name: str) -> str:
    path = repo / HELPER_FILES[name]
    spec = importlib.util.spec_from_file_location(f"_build_version_{name}", path)
    if spec is None or spec.loader is None:
        raise ReleaseError(f"{path}: cannot load build-version helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_version()


def _tag_and_push(repo: Path, status: ComponentStatus, tag: str) -> TagRow:
    git(repo, "tag", "-a", tag, "-m", f"{status.name} {status.literal}", "HEAD")
    built = _build_version(repo, status.name)
    if built != status.literal:
        git(repo, "tag", "-d", tag)
        return _row(
            status, tag, "FAIL", f"helper returned {built}, literal is {status.literal}"
        )
    try:
        # Push only this tag; `--tags` would ship every local tag.
        git(repo, "push", "origin", f"refs/tags/{tag}")
    except ReleaseError as exc:
        git(repo, "tag", "-d", tag)
        return _row(status, tag, "FAIL", str(exc))
    return _row(status, tag, "TAGGED", f"pushed {tag}")


def _tag_candidates(
    repo: Path, statuses: dict[str, ComponentStatus], candidates: list[str]
) -> dict[str, TagRow]:
    """Common tail for the components that may be tagged in this run."""
    rows: dict[str, TagRow] = {}
    head = git(repo, "rev-parse", "HEAD").strip()
    tags: dict[str, str] = {}
    for name in candidates:
        status = statuses[name]
        try:
            tag = tag_for(name, status.literal)
        except ReleaseError as exc:
            rows[name] = _row(status, None, "FAIL", str(exc))
            continue
        remote = _remote_tag_sha(repo, tag)
        if remote == head:
            rows[name] = _row(status, tag, "DONE", "tag already at this commit")
        elif remote is not None:
            rows[name] = _row(status, tag, "FAIL", f"{tag} already exists at {remote}")
        else:
            tags[name] = tag

    # Rerun safety: a tag already at HEAD is DONE above. Running the rules on it
    # would FAIL instead, since the tag-unused rule sees it in the fetched tag list.
    check_statuses = {
        name: dataclasses.replace(status, releasing=name in tags)
        for name, status in statuses.items()
    }
    for name, tag in tags.items():
        problems = [
            v
            for v in violations(check_release(repo, check_statuses))
            if v.component == name
        ]
        if problems:
            rows[name] = _row(
                statuses[name], tag, "FAIL", "; ".join(v.message for v in problems)
            )
        else:
            rows[name] = _tag_and_push(repo, statuses[name], tag)
        if rows[name].outcome == "FAIL":
            # Dependants must see this one as unreleased, so check them against its old tag.
            check_statuses[name] = dataclasses.replace(
                check_statuses[name], releasing=False
            )
    return rows


def tag_push(repo: Path, before: str) -> list[TagRow]:
    """Tag the components whose literal moved between `before` and HEAD.

    `before` is trusted: main only takes squash merges, never a force push, so
    the push event's previous tip is always an ancestor of HEAD.
    """
    statuses = releasing_components(repo, before)
    if not is_release_pr(statuses):
        return [
            _row(s, None, "NOOP", "gate closed, nothing releasing")
            for s in statuses.values()
        ]

    rows: dict[str, TagRow] = {}
    candidates: list[str] = []
    for name, status in statuses.items():
        if status.decreased:
            rows[name] = _row(status, None, "SKIP", "literal decreased, no tag")
        elif not status.releasing:
            rows[name] = _row(status, None, "NOOP", "literal unchanged")
        else:
            candidates.append(name)

    rows.update(_tag_candidates(repo, statuses, candidates))
    return _ordered(rows)


def nothing_to_tag(rows: list[TagRow]) -> bool:
    """True when no component was a candidate: everything DONE, SKIP or NOOP."""
    return all(r.outcome in ("DONE", "SKIP", "NOOP") for r in rows)
