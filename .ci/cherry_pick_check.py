# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24"]
# ///
"""Check that every commit in a pull request into a release branch is a cherry-pick.

A commit passes when its `(cherry picked from commit <sha>)` trailer names a
commit reachable from `--main`, or when `git cherry` finds a patch-identical
commit there, or when it makes only release edits. Merge commits fail.
Exit 1 on any violation. See RELEASE.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

from release_git import ReleaseError, changes_beyond_release_edits, default_repo, git

PICK_TRAILER = re.compile(
    r"^\(cherry picked from commit ([0-9a-f]{40})\)$", re.MULTILINE
)
"""The line `git cherry-pick -x` appends to the message."""


class Violation(NamedTuple):
    sha: str
    subject: str
    reason: str


def find_violations(repo: Path, base: str, head: str, main_ref: str) -> list[Violation]:
    """Commits in `base..head` that are not picks of a commit on `main_ref`."""
    if git(repo, "rev-parse", "--is-shallow-repository").strip() == "true":
        raise ReleaseError("shallow clone; fetch the full history")

    equivalent = {
        line.split()[1]
        for line in git(repo, "cherry", main_ref, head, base).splitlines()
        if line.startswith("-")
    }
    found = []
    for sha in git(repo, "rev-list", "--reverse", f"{base}..{head}").split():
        reason = _reason(repo, sha, main_ref, equivalent)
        if reason is not None:
            subject = git(repo, "show", "-s", "--format=%s", sha).strip()
            found.append(Violation(sha, subject, reason))
    return found


def _reason(repo: Path, sha: str, main_ref: str, equivalent: set[str]) -> str | None:
    parents = git(repo, "show", "-s", "--format=%P", sha).split()
    if len(parents) > 1:
        return "is a merge commit; rebase the branch instead of merging into it"
    if sha in equivalent:
        return None

    # The release commit lands on the release branch first and reaches main later in a sync-back PR.
    if not changes_beyond_release_edits(repo, f"{sha}^", ["."], head=sha):
        return None
    trailers = PICK_TRAILER.findall(git(repo, "show", "-s", "--format=%B", sha))
    if not trailers:
        if _on(repo, sha, main_ref):
            return f"is on {main_ref} itself, not a pick of it"
        return f"has no `(cherry picked from commit ...)` trailer and matches no commit on {main_ref}"

    # The last trailer is the one `git cherry-pick -x` appended; earlier ones were copied.
    picked = trailers[-1]
    try:
        git(repo, "cat-file", "-e", f"{picked}^{{commit}}")
    except ReleaseError:
        return f"names {picked}, which is not in this clone; pick the commit on {main_ref}, not a branch commit"
    if not _on(repo, picked, main_ref):
        return f"names {picked}, which is not on {main_ref}"
    return None


def _on(repo: Path, sha: str, ref: str) -> bool:
    return not git(repo, "rev-list", "-1", sha, "--not", ref).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="tip of the release branch")
    parser.add_argument("--head", required=True, help="tip of the pull request branch")
    parser.add_argument("--main", required=True, help="ref the picks must be on")
    args = parser.parse_args(argv)

    try:
        violations = find_violations(default_repo(), args.base, args.head, args.main)
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for v in violations:
        print(f"FAIL {v.sha[:12]} {v.subject}: {v.reason}")

    if len(violations) > 0:
        print(
            f"\nEvery commit on a release branch must be a pick of a commit on {args.main}; see RELEASE.md."
        )
        return 1

    print(f"PASS: every commit names or matches a commit on {args.main}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
