# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .ci/ is not a package
import cherry_pick_check  # noqa: E402


class Repo:
    def __init__(self, path: Path) -> None:
        self.path = path

    def git(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.path, check=check, capture_output=True, text=True
        )
        return result.stdout.strip()

    def commit(self, relpath: str, text: str, msg: str) -> str:
        (self.path / relpath).write_text(text)
        self.git("add", "-A")
        self.git("commit", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def violations(self) -> list[str]:
        found = cherry_pick_check.find_violations(
            self.path, base="release", head="HEAD", main_ref="main"
        )
        return [v.sha for v in found]


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(tmp_path)
    r.git("init", "-b", "main")
    for key, value in (
        ("user.name", "tests"),
        ("user.email", "tests@example.com"),
        ("commit.gpgsign", "false"),
    ):
        r.git("config", key, value)
    r.commit("a.txt", "one\n", "initial")
    r.git("branch", "release")
    return r


def test_fix_picked_without_x_passes_by_patch_equivalence(repo: Repo) -> None:
    repo.git("switch", "release")
    repo.commit("c.txt", "y\n", "release-only change")
    repo.git("switch", "main")
    fix = repo.commit("b.txt", "fix\n", "fix: b")
    repo.git("switch", "-c", "pr", "release")
    repo.git("cherry-pick", fix)
    assert repo.violations() == []


def test_conflict_resolved_pick_with_x_passes(repo: Repo) -> None:
    repo.git("switch", "release")
    repo.commit("a.txt", "release\n", "release-only change")
    repo.git("switch", "main")
    fix = repo.commit("a.txt", "fixed\n", "fix: a")
    repo.git("switch", "-c", "pr", "release")
    repo.git("cherry-pick", "-x", fix, check=False)
    (repo.path / "a.txt").write_text("resolved\n")
    repo.git("add", "-A")
    repo.git("-c", "core.editor=true", "cherry-pick", "--continue")
    assert repo.violations() == []


def test_fresh_commit_fails_and_release_commit_passes(repo: Repo) -> None:
    repo.git("switch", "release")
    repo.commit("v.py", '__version__ = "1.0.0"\n', "release-only change")
    repo.git("switch", "-c", "pr", "release")
    repo.commit("v.py", '__version__ = "1.1.0"\n', "chore: release")
    fresh = repo.commit("b.txt", "new\n", "fix: only on the branch")
    assert repo.violations() == [fresh]


def test_trailer_naming_a_branch_or_missing_commit_fails(repo: Repo) -> None:
    repo.git("switch", "-c", "pr", "release")
    direct = repo.commit("b.txt", "new\n", "chore: release")
    repo.git("switch", "-c", "pr2", "release")
    forged = repo.commit(
        "c.txt", "x\n", f"fix: c\n\n(cherry picked from commit {direct})"
    )
    missing = repo.commit(
        "d.txt", "y\n", f"fix: d\n\n(cherry picked from commit {'0' * 40})"
    )
    assert repo.violations() == [forged, missing]


def test_merge_commit_fails(repo: Repo) -> None:
    repo.git("switch", "-c", "pr", "release")
    repo.commit("b.txt", "x\n", "wip")
    repo.git("switch", "release")
    repo.commit("c.txt", "y\n", "release-only change")
    repo.git("switch", "pr")
    repo.git("merge", "--no-ff", "-m", "Merge release", "release")
    merge = repo.git("rev-parse", "HEAD")
    assert merge in repo.violations()


def test_commit_already_on_main_fails(repo: Repo) -> None:
    on_main = repo.commit("b.txt", "x\n", "fix: b")
    assert repo.violations() == [on_main]
