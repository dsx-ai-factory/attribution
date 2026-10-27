# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Is a release PR, and the rules a release PR must satisfy.

Policy lives in RELEASE.md. Each `_rule_*` enforces one of its statements.
Changes to the policy must be reflected in the rule and its test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeAlias

from release_git import (
    COMPONENTS,
    RELEASABLE,
    LatestTags,
    ReleaseError,
    closure_paths,
    git,
    latest_tags,
    parse_version,
    read_literal,
    tag_for,
)


@dataclass(frozen=True)
class ComponentStatus:
    name: str
    """Component name like "service-app"."""
    literal: str
    """The committed version literal."""
    base_literal: Optional[str]
    """The committed version literal at `base`, or None if the component did not exist yet."""
    latest: LatestTags
    """Latest tags, any and stable."""
    releasing: bool
    """The literal changed and did not decrease relative to `base`."""
    is_prerelease: bool
    """The new literal is a release candidate (e.g. X.Y.ZrcN)."""
    decreased: bool
    """The new literal went below the `base` literal: advisory, never releasing."""


def releasing_components(repo: Path, base: Optional[str]) -> dict[str, ComponentStatus]:
    """Per-component status against `base`; a release PR is one where some literal changed.

    Literals are compared at the merge-base of `base` and HEAD, so a branch that has
    not been rebased does not report main's later literal changes as its own.
    """
    if base is not None:
        base = git(repo, "merge-base", base, "HEAD").strip()
    statuses: dict[str, ComponentStatus] = {}
    for name, comp in COMPONENTS.items():
        literal = read_literal(repo, comp, None)
        if literal is None:
            raise ReleaseError(f"{comp.literal_file}: no __version__ literal")
        base_literal = read_literal(repo, comp, base) if base is not None else None
        changed = base_literal is not None and literal != base_literal
        new_version = parse_version(literal)
        old_version = parse_version(base_literal)
        decreased = (
            changed
            and new_version is not None
            and old_version is not None
            and new_version < old_version
        )
        latest = latest_tags(repo, name)
        statuses[name] = ComponentStatus(
            name=name,
            literal=literal,
            base_literal=base_literal,
            latest=latest,
            releasing=changed and not decreased,
            is_prerelease=(
                new_version is not None
                and new_version.pre is not None
                and new_version.pre[0] == "rc"
            ),
            decreased=decreased,
        )
    return statuses


def is_release_pr(statuses: dict[str, ComponentStatus]) -> bool:
    """True iff some component's literal differs from the base, up or down.

    A decrease alone still makes it a release PR, but releases nothing.
    """
    return any(s.releasing or s.decreased for s in statuses.values())


@dataclass(frozen=True)
class Pass:
    component: str
    reason: str
    """Why the rule was satisfied; printed for verification."""


@dataclass(frozen=True)
class Violation:
    component: str
    message: str
    log: Optional[str] = None


RuleResult: TypeAlias = Pass | Violation


def violations(results: list[RuleResult]) -> list[Violation]:
    """All violations in a list of rule results."""
    return [r for r in results if isinstance(r, Violation)]


def _log_since(repo: Path, name: str, tag: Optional[str]) -> str:
    """The commit log since `tag` (or HEAD if `tag` is None)."""
    rev_range = f"{tag}..HEAD" if tag is not None else "HEAD"
    return git(
        repo,
        "log",
        "--first-parent",
        "--format=- %s",
        rev_range,
        "--",
        *closure_paths(name),
    ).rstrip()


# TODO(pforderique): Call this once first official literals land.
def _rule_no_decrease(status: ComponentStatus) -> RuleResult:
    """A literal must not go below the base literal; versions only move forward."""
    if not status.decreased:
        return Pass(
            status.name,
            f'literal "{status.literal}" is not below base "{status.base_literal}"',
        )
    return Violation(
        status.name,
        f'literal "{status.literal}" is below base "{status.base_literal}"',
    )


def _rule_literal_shape(status: ComponentStatus) -> RuleResult:
    """Literal must be a PEP 440 final or rc (2.0.0, 2.0.0rc1); nothing else gets a tag."""
    if RELEASABLE.match(status.literal):
        return Pass(
            status.name, f'literal "{status.literal}" is PEP 440 X.Y.Z or X.Y.ZrcN'
        )
    return Violation(
        status.name, f'literal "{status.literal}" is not PEP 440 X.Y.Z or X.Y.ZrcN'
    )


def _rule_tag_unused(repo: Path, status: ComponentStatus) -> RuleResult:
    """The planned tag must not exist yet; a version is released once."""
    tag = tag_for(status.name, status.literal)
    if not git(repo, "tag", "--list", tag).strip():
        return Pass(status.name, f"tag {tag} does not exist yet")
    return Violation(status.name, f"tag {tag} already exists")


def _rule_changelog_entry(repo: Path, status: ComponentStatus) -> RuleResult:
    """CHANGELOG.md must carry a `## [<literal>]` heading for the release."""
    changelog = repo / COMPONENTS[status.name].path / "CHANGELOG.md"
    heading = re.compile(rf"^## \[{re.escape(status.literal)}\]", re.MULTILINE)
    if changelog.is_file() and heading.search(changelog.read_text()) is not None:
        return Pass(status.name, f'CHANGELOG.md has "## [{status.literal}]"')
    return Violation(status.name, f'CHANGELOG.md lacks "## [{status.literal}]"')


def _rule_no_prerelease_dep_in_pr(
    consumer: ComponentStatus, dep: ComponentStatus
) -> RuleResult:
    """A stable consumer cannot bundle a dependency released as rc in this PR."""
    if not dep.is_prerelease:
        return Pass(
            consumer.name, f"{dep.name} {dep.literal} released in this PR is stable"
        )
    return Violation(
        consumer.name,
        f"stable release of {consumer.name} {consumer.literal} cannot bundle "
        f"pre-release {dep.name} {dep.literal} released in this PR",
    )


def _rule_no_prerelease_dep_at_head(
    repo: Path, consumer: ComponentStatus, dep: ComponentStatus
) -> RuleResult:
    """A stable consumer cannot ship a dependency whose latest tag is an rc."""
    if dep.latest.any is None:
        return Pass(consumer.name, f"{dep.name} has no pre-release tag")
    if not dep.latest.any.version.is_prerelease:
        return Pass(
            consumer.name, f"{dep.name} latest tag {dep.latest.any.tag} is stable"
        )
    if dep.latest.stable is None:
        since, detail = None, "no stable tag"
    else:
        since = dep.latest.stable.tag
        detail = f"latest stable {since} at distance {dep.latest.stable.distance}"
    return Violation(
        consumer.name,
        f"stable release of {consumer.name} {consumer.literal} cannot bundle "
        f"pre-release {dep.name}: latest tag {dep.latest.any.tag} is a pre-release "
        f"({detail}); release a stable {dep.name} in this PR",
        _log_since(repo, dep.name, since),
    )


def _rule_dep_released(
    consumer: ComponentStatus, dep: ComponentStatus, stable: bool
) -> RuleResult:
    """A dependency must have a tag of the consumer's flavour (stable or any)."""
    tag = dep.latest.stable if stable else dep.latest.any
    flavour = "stable " if stable else ""
    if tag is not None:
        return Pass(consumer.name, f"{dep.name} has {flavour}release {tag.tag}")
    return Violation(
        consumer.name, f"{dep.name} has no {flavour}release; release it in this PR"
    )


def _rule_dep_at_distance_zero(
    repo: Path, consumer: ComponentStatus, dep: ComponentStatus, stable: bool
) -> RuleResult:
    """A dependency's closure must not have moved since its tag (distance 0)."""
    tag = dep.latest.stable if stable else dep.latest.any
    if tag is None:
        raise ReleaseError(f"{dep.name}: _rule_dep_released must run first")
    if tag.distance == 0:
        return Pass(consumer.name, f"{dep.name} unchanged since {tag.tag} (distance 0)")
    return Violation(
        consumer.name,
        f"{dep.name} has moved since {tag.tag} (distance {tag.distance})",
        _log_since(repo, dep.name, tag.tag),
    )


def _check_dependency(
    repo: Path, consumer: ComponentStatus, dep: ComponentStatus, stable: bool
) -> list[RuleResult]:
    """Bundle rules for one (consumer, dependency) edge of the transitive closure."""
    if dep.releasing:
        # Released in this PR: only its flavour matters, its tag is about to exist.
        return [_rule_no_prerelease_dep_in_pr(consumer, dep)] if stable else []
    results: list[RuleResult] = []
    if stable:
        results.append(_rule_no_prerelease_dep_at_head(repo, consumer, dep))
        if isinstance(results[-1], Violation):
            return results
    results.append(_rule_dep_released(consumer, dep, stable))
    if isinstance(results[-1], Violation):
        return results  # no tag to measure distance from
    results.append(_rule_dep_at_distance_zero(repo, consumer, dep, stable))
    return results


def check_release(repo: Path, statuses: dict[str, ComponentStatus]) -> list[RuleResult]:
    """Every rule outcome for every releasing component, in COMPONENTS order."""
    results: list[RuleResult] = []
    for name, status in statuses.items():
        if not status.releasing:
            continue
        results.append(_rule_literal_shape(status))
        if isinstance(results[-1], Violation):
            continue  # no tag name to derive the remaining rules from
        results.append(_rule_tag_unused(repo, status))
        stable = not status.is_prerelease
        for dep in COMPONENTS[name].closure:
            results.extend(_check_dependency(repo, status, statuses[dep], stable))
        results.append(_rule_changelog_entry(repo, status))
    return results
