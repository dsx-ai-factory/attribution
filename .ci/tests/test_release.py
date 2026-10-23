# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .ci/ is not a package
import release  # noqa: E402
import release_git  # noqa: E402
import release_rules  # noqa: E402
import release_tag  # noqa: E402

INTERIM = {
    "attribution-lib": "0.4.0",
    "service-app": "1.0.0",
    "blueprint": "0.1.0",
}
NEW_RELEASE = {
    "attribution-lib": "0.5.0",
    "service-app": "2.0.0",
    "blueprint": "0.2.0",
}


@dataclass
class Repo:
    path: Path

    def git(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.path, check=True, capture_output=True)

    def write(self, relpath: str, text: str) -> None:
        target = self.path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, msg: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-m", msg)

    def tag(self, name: str) -> None:
        self.git("tag", "-a", name, "-m", name)

    def mark_base(self) -> None:
        self.git("branch", "-f", "base")

    def set_literal(self, component: str, value: str) -> None:
        self.write(
            release_git.COMPONENTS[component].literal_file, f'__version__ = "{value}"\n'
        )

    def add_changelog(self, component: str, literal: str) -> None:
        changelog = self.path / release_git.COMPONENTS[component].path / "CHANGELOG.md"
        existing = changelog.read_text() if changelog.is_file() else "# Changelog\n"
        self.write(
            str(changelog.relative_to(self.path)), existing + f"\n## [{literal}]\n"
        )

    def statuses(self) -> dict[str, release_rules.ComponentStatus]:
        return release_rules.releasing_components(self.path, "base")

    def results(self) -> list[release_rules.RuleResult]:
        return release_rules.check_release(self.path, self.statuses())

    def violations(self) -> list[release_rules.Violation]:
        return release_rules.violations(self.results())


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(tmp_path)
    r.git("init", "-b", "main")
    for key, value in (
        ("user.name", "tests"),
        ("user.email", "tests@example.com"),
        ("commit.gpgsign", "false"),
        ("tag.gpgsign", "false"),
    ):
        r.git("config", key, value)
    for component, literal in INTERIM.items():
        r.set_literal(component, literal)
    r.write("docs/x", "")
    r.commit("initial")
    r.mark_base()
    return r


def _release_everything_at_distance_zero(repo: Repo) -> None:
    for component, literal in NEW_RELEASE.items():
        repo.set_literal(component, literal)
        repo.add_changelog(component, literal)
    repo.commit("first release")
    for component, literal in NEW_RELEASE.items():
        repo.tag(release_git.tag_for(component, literal))
    repo.mark_base()


def test_unchanged_literals_are_not_a_release_pr(repo: Repo) -> None:
    statuses = repo.statuses()

    assert release._release_pr_line(statuses) == "not a release PR, nothing releasing"
    assert not any(s.releasing for s in statuses.values())
    assert release_rules.check_release(repo.path, statuses) == []


def test_only_bumped_component_releases_and_a_decrease_never_does(repo: Repo) -> None:
    repo.set_literal("attribution-lib", "0.5.0")
    repo.commit("bump lib")
    statuses = repo.statuses()
    assert statuses["attribution-lib"].releasing
    assert not statuses["service-app"].releasing

    repo.set_literal("attribution-lib", "0.3.0")
    repo.commit("lower lib")
    statuses = repo.statuses()
    assert not statuses["attribution-lib"].releasing
    assert statuses["attribution-lib"].decreased
    assert release_rules.check_release(repo.path, statuses) == []
    hit = release_rules._rule_no_decrease(statuses["attribution-lib"])
    assert isinstance(hit, release_rules.Violation) and "below base" in hit.message


def test_first_release_needs_a_changelog_entry_per_component(repo: Repo) -> None:
    for component, literal in NEW_RELEASE.items():
        repo.set_literal(component, literal)
    repo.commit("first release")

    statuses = repo.statuses()
    assert all(s.releasing for s in statuses.values())
    violations = repo.violations()
    assert [v.component for v in violations] == list(release_git.COMPONENTS)
    assert all("CHANGELOG.md lacks" in v.message for v in violations)

    for component, literal in NEW_RELEASE.items():
        repo.add_changelog(component, literal)
    repo.commit("changelogs")
    results = repo.results()
    assert results and all(isinstance(r, release_rules.Pass) for r in results)
    rendered = release.render_check(statuses, results)
    assert 'PASS      attribution-lib: CHANGELOG.md has "## [0.5.0]"' in rendered
    assert "PASS: no violations" in rendered


def test_blueprint_bump_over_moved_service_app_needs_service_app_release(
    repo: Repo,
) -> None:
    assert release_git.COMPONENTS["blueprint"].closure == (
        "attribution-lib",
        "service-app",
    )
    assert release_git.closure_paths("blueprint") == [
        "blueprint",
        "service/attribution-lib",
        "service/service-app",
    ]
    _release_everything_at_distance_zero(repo)
    repo.write("service/service-app/x", "")
    repo.commit("touch service-app")
    repo.set_literal("blueprint", "0.2.1")
    repo.add_changelog("blueprint", "0.2.1")
    repo.commit("bump blueprint")

    violations = repo.violations()
    assert len(violations) == 1
    violation = violations[0]
    assert violation.component == "blueprint"
    assert (
        "service-app has moved since service-app-v2.0.0 (distance 1)"
        in violation.message
    )
    assert violation.log is not None and "touch service-app" in violation.log

    repo.set_literal("service-app", "2.0.1")
    repo.add_changelog("service-app", "2.0.1")
    repo.commit("bump service-app")
    assert repo.violations() == []


def test_blueprint_bump_over_moved_lib_needs_whole_closure_released(repo: Repo) -> None:
    _release_everything_at_distance_zero(repo)
    repo.write("service/attribution-lib/x", "")
    repo.commit("touch lib")
    repo.set_literal("blueprint", "0.2.1")
    repo.add_changelog("blueprint", "0.2.1")
    repo.commit("bump blueprint")

    violations = repo.violations()
    assert [v.component for v in violations] == ["blueprint", "blueprint"]
    assert "attribution-lib has moved since attribution-lib-v0.5.0 (distance 1)" in (
        violations[0].message
    )
    assert "service-app has moved since service-app-v2.0.0 (distance 1)" in (
        violations[1].message
    )

    repo.set_literal("service-app", "2.0.1")
    repo.add_changelog("service-app", "2.0.1")
    repo.commit("bump service-app")
    violations = repo.violations()
    assert [v.component for v in violations] == ["service-app", "blueprint"]
    assert all("attribution-lib has moved" in v.message for v in violations)

    repo.set_literal("attribution-lib", "0.5.1")
    repo.add_changelog("attribution-lib", "0.5.1")
    repo.commit("bump lib")
    assert repo.violations() == []


def test_stable_service_app_cannot_bundle_a_prerelease_lib_tag(repo: Repo) -> None:
    repo.tag("attribution-lib-v0.5.2")
    repo.write("service/attribution-lib/x", "")
    repo.commit("lib change")
    repo.tag("attribution-lib-v0.5.3-rc.1")
    repo.tag("service-app-v2.0.1")
    repo.mark_base()

    repo.set_literal("service-app", "2.0.2")
    repo.add_changelog("service-app", "2.0.2")
    repo.commit("bump service-app")
    violations = repo.violations()
    assert len(violations) == 1
    message = violations[0].message
    assert "cannot bundle pre-release attribution-lib" in message
    assert "attribution-lib-v0.5.3-rc.1" in message
    assert "attribution-lib-v0.5.2" in message

    repo.set_literal("service-app", "2.0.2rc1")
    repo.add_changelog("service-app", "2.0.2rc1")
    repo.commit("bump service-app to rc")
    assert repo.violations() == []

    latest = release_git.latest_tags(repo.path, "attribution-lib")
    assert (latest.any.tag, latest.any.distance) == ("attribution-lib-v0.5.3-rc.1", 0)
    assert (latest.stable.tag, latest.stable.distance) == ("attribution-lib-v0.5.2", 1)


def test_literals_are_compared_at_the_merge_base_not_the_tip_of_base(
    repo: Repo,
) -> None:
    repo.git("checkout", "-b", "stale")
    repo.write("docs/y", "")
    repo.commit("unrelated")
    repo.git("checkout", "main")
    repo.set_literal("attribution-lib", "0.5.0")
    repo.commit("bump lib on main")
    repo.git("checkout", "stale")

    statuses = release_rules.releasing_components(repo.path, "main")
    assert not statuses["attribution-lib"].releasing
    assert not statuses["attribution-lib"].decreased


def test_only_an_rc_literal_is_a_prerelease(repo: Repo) -> None:
    repo.set_literal("service-app", "2.0.0.dev0")
    repo.set_literal("blueprint", "0.2.0rc1")
    repo.commit("dev and rc")
    statuses = repo.statuses()
    assert not statuses["service-app"].is_prerelease
    assert statuses["blueprint"].is_prerelease


def test_distance_counts_first_parent_commits_in_the_closure_only(repo: Repo) -> None:
    repo.tag("service-app-v2.0.0")
    repo.git("checkout", "-b", "side")
    for i in range(3):
        repo.write(f"service/attribution-lib/side{i}", "")
        repo.commit(f"side {i}")
    repo.git("checkout", "main")
    repo.git("merge", "--no-ff", "-m", "merge", "side")
    assert release_git.distance(repo.path, "service-app", "service-app-v2.0.0") == 1

    repo.write("docs/y", "")
    repo.commit("docs")
    assert release_git.distance(repo.path, "service-app", "service-app-v2.0.0") == 1

    repo.write("blueprint/x", "")
    repo.commit("blueprint")
    assert release_git.distance(repo.path, "service-app", "service-app-v2.0.0") == 1


def test_tag_names_round_trip_and_releasable_shapes() -> None:
    for literal, tag in (
        ("2.0.0", "service-app-v2.0.0"),
        ("2.0.0rc1", "service-app-v2.0.0-rc.1"),
    ):
        assert release_git.tag_for("service-app", literal) == tag
        assert str(release_git.version_from_tag("service-app", tag)) == literal
    assert all(release_git.RELEASABLE.match(v) for v in ("2.0.0", "2.0.0rc1"))
    assert not any(
        release_git.RELEASABLE.match(v) for v in ("2.0.0.dev0", "2.0.0+1", "2.0.0a1")
    )
    for literal in ("2.0.0.dev0", "2.0.0a1"):
        with pytest.raises(release_git.ReleaseError):
            release_git.tag_for("service-app", literal)


WORKTREE = Path(__file__).resolve().parents[2]


@dataclass
class TaggingRepo(Repo):
    def out(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.path, check=True, capture_output=True, text=True
        ).stdout.strip()

    def origin_tags(self) -> dict[str, str]:
        """Tag name -> commit it points at, as origin sees it."""
        tags: dict[str, str] = {}
        for line in self.out("ls-remote", "--tags", "origin").splitlines():
            sha, _, ref = line.partition("\t")
            if ref.endswith("^{}"):
                tags[ref.removeprefix("refs/tags/").removesuffix("^{}")] = sha
        return tags

    def bump(self, component: str, literal: str) -> None:
        self.set_literal(component, literal)
        self.add_changelog(component, literal)

    def push_rows(self) -> list[release_tag.TagRow]:
        return release_tag.tag_push(self.path, self.out("rev-parse", "base"))

    def push_outcomes(self) -> dict[str, str]:
        return {r.component: r.outcome for r in self.push_rows()}


@pytest.fixture
def tagging_repo(repo: Repo, tmp_path_factory: pytest.TempPathFactory) -> TaggingRepo:
    r = TaggingRepo(repo.path)
    for relpath in release_tag.HELPER_FILES.values():
        (r.path / relpath).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(WORKTREE / relpath, r.path / relpath)
    r.commit("build-version helpers")
    origin = tmp_path_factory.mktemp("origin")
    r.git("init", "--bare", str(origin))
    r.git("remote", "add", "origin", str(origin))
    r.git("push", "origin", "main")
    r.mark_base()
    return r


def test_moved_literal_is_tagged_once_and_a_rerun_is_done(
    tagging_repo: TaggingRepo,
) -> None:
    tagging_repo.bump("attribution-lib", "0.5.0")
    tagging_repo.commit("bump lib")
    argv = [
        "tag",
        "--repo",
        str(tagging_repo.path),
        "--before",
        tagging_repo.out("rev-parse", "base"),
    ]

    assert release.main(argv) == 0
    assert tagging_repo.push_outcomes() == {
        "attribution-lib": "DONE",
        "service-app": "NOOP",
        "blueprint": "NOOP",
    }
    assert release.main(argv) == 0
    head = tagging_repo.out("rev-parse", "HEAD")
    assert tagging_repo.origin_tags() == {"attribution-lib-v0.5.0": head}
    assert tagging_repo.out("cat-file", "-t", "attribution-lib-v0.5.0") == "tag"


def test_tag_already_on_origin_at_another_commit_fails(
    tagging_repo: TaggingRepo,
) -> None:
    tagging_repo.tag("attribution-lib-v0.5.0")
    tagging_repo.git("push", "origin", "attribution-lib-v0.5.0")
    tagging_repo.git("tag", "-d", "attribution-lib-v0.5.0")
    before = tagging_repo.origin_tags()
    tagging_repo.bump("attribution-lib", "0.5.0")
    tagging_repo.commit("bump lib")

    assert tagging_repo.push_outcomes()["attribution-lib"] == "FAIL"
    assert tagging_repo.origin_tags() == before


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        (
            "0.3.0",
            {"attribution-lib": "SKIP", "service-app": "NOOP", "blueprint": "NOOP"},
        ),
        (
            None,
            {"attribution-lib": "NOOP", "service-app": "NOOP", "blueprint": "NOOP"},
        ),
    ],
    ids=["decreased", "unchanged"],
)
def test_push_without_a_release_tags_nothing(
    tagging_repo: TaggingRepo,
    literal: Optional[str],
    expected: dict[str, str],
) -> None:
    if literal is not None:
        tagging_repo.bump("attribution-lib", literal)
        tagging_repo.commit("change lib")

    rows = tagging_repo.push_rows()
    assert {r.component: r.outcome for r in rows} == expected
    assert release.render_tag(rows).splitlines()[0] == "nothing to tag"
    assert tagging_repo.origin_tags() == {}


def test_bundle_violation_fails_the_consumer_and_tags_nothing_for_it(
    tagging_repo: TaggingRepo,
) -> None:
    tagging_repo.bump("service-app", "2.0.0")
    tagging_repo.commit("bump service-app over unreleased lib")

    assert tagging_repo.push_outcomes()["service-app"] == "FAIL"
    assert tagging_repo.origin_tags() == {}


def test_rejected_dependency_push_fails_its_dependants_too(
    tagging_repo: TaggingRepo,
) -> None:
    origin = Path(tagging_repo.out("remote", "get-url", "origin"))
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\ngrep -q refs/tags/attribution-lib- && exit 1\nexit 0\n")
    hook.chmod(0o755)
    tagging_repo.bump("attribution-lib", "0.5.0")
    tagging_repo.bump("service-app", "2.0.0")
    tagging_repo.commit("release lib and service-app")

    outcomes = tagging_repo.push_outcomes()
    assert outcomes["attribution-lib"] == "FAIL"
    assert outcomes["service-app"] == "FAIL"
    assert tagging_repo.origin_tags() == {}


def test_rc_literal_gets_an_rc_tag(tagging_repo: TaggingRepo) -> None:
    tagging_repo.bump("attribution-lib", "0.5.0")
    tagging_repo.bump("service-app", "2.0.0rc1")
    tagging_repo.commit("release lib and service-app rc")

    outcomes = tagging_repo.push_outcomes()
    assert outcomes["attribution-lib"] == outcomes["service-app"] == "TAGGED"
    assert set(tagging_repo.origin_tags()) == {
        "attribution-lib-v0.5.0",
        "service-app-v2.0.0-rc.1",
    }
