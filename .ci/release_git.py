# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git-facing half of the release tooling: components, literals, tags, distance."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from packaging.version import InvalidVersion, Version

LITERAL_RE = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
"""Literal is a PEP 440 release like 2.0.0 or 2.0.0rc1
(packaging tools reject anything else, like SemVer)."""
RELEASABLE = re.compile(r"^\d+\.\d+\.\d+(rc\d+)?$")
"""A release PR may commit only a final (2.0.0) or an rc (2.0.0rc1) literal."""
_TAG_SUFFIX_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$")
"""Tag with SemVer suffix after "<component>-v", same key as each component's _build_version.py."""


class ReleaseError(RuntimeError):
    """A git call failed or the checkout is not in a state the script can read."""


@dataclass(frozen=True)
class Component:
    name: str
    """Tag prefix like "service-app"."""
    path: str
    """Repo-relative root, used for the shortlog and CHANGELOG.md."""
    literal_file: str
    """Repo-relative path to the __version__ literal."""
    closure: tuple[str, ...]
    """In-repo dependencies, including indirect ones, in `COMPONENTS` order."""


# Insertion order is dependency order (lib, service, blueprint).
COMPONENTS: dict[str, Component] = {
    "attribution-lib": Component(
        "attribution-lib",
        "service/attribution-lib",
        "service/attribution-lib/src/attribution_lib/__init__.py",
        (),
    ),
    "service-app": Component(
        "service-app",
        "service/service-app",
        "service/service-app/app/__init__.py",
        ("attribution-lib",),
    ),
    "blueprint": Component(
        "blueprint",
        "blueprint",
        "blueprint/src/attribution_blueprint/__init__.py",
        ("attribution-lib", "service-app"),
    ),
}


def closure_paths(name: str) -> list[str]:
    """Own path plus the closure's paths: the pathspec distance is counted over."""
    return [COMPONENTS[name].path] + [
        COMPONENTS[d].path for d in COMPONENTS[name].closure
    ]


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def default_repo() -> Path:
    return Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())


def read_literal(repo: Path, comp: Component, rev: Optional[str]) -> Optional[str]:
    """The `__version__` literal at `rev`, or from the working tree when `rev` is None."""
    if rev is None:
        path = repo / comp.literal_file
        if not path.is_file():
            return None
        text = path.read_text()
    else:
        try:
            text = git(repo, "show", f"{rev}:{comp.literal_file}")
        except ReleaseError:
            return None  # file absent at `rev` (component added since): no base literal
    match = LITERAL_RE.search(text)
    return match.group(1) if match else None


def parse_version(literal: Optional[str]) -> Optional[Version]:
    if literal is None:
        return None
    try:
        return Version(literal)
    except InvalidVersion:
        return None


def tag_for(comp: str, literal: str) -> str:
    """PEP 440 literal to tag name: `2.0.0rc1` -> `service-app-v2.0.0-rc.1`.

    Only RELEASABLE literals have a tag.
    """
    if not RELEASABLE.match(literal):
        raise ReleaseError(
            f'literal "{literal}" is not X.Y.Z or X.Y.ZrcN; no tag for it'
        )
    version = Version(literal)
    tag = f"{comp}-v{version.major}.{version.minor}.{version.micro}"
    if version.pre is not None:
        tag += f"-rc.{version.pre[1]}"
    return tag


def version_from_tag(comp: str, tag: str) -> Optional[Version]:
    """Tag name to version; None when the tag is not this component's or unparsable."""
    prefix = f"{comp}-v"
    if not tag.startswith(prefix):
        return None
    match = _TAG_SUFFIX_RE.match(tag[len(prefix) :])
    if match is None:
        return None
    major, minor, micro, rc = match.groups()
    return Version(f"{major}.{minor}.{micro}" + (f"rc{rc}" if rc is not None else ""))


@dataclass(frozen=True)
class TagInfo:
    tag: str
    version: Version
    distance: int


def distance(repo: Path, name: str, tag: str) -> int:
    """First-parent commits since `tag` touching the component or its closure."""
    out = git(
        repo,
        "rev-list",
        "--count",
        "--first-parent",
        f"{tag}..HEAD",
        "--",
        *closure_paths(name),
    )
    return int(out.strip())


@dataclass(frozen=True)
class LatestTags:
    any: Optional[TagInfo]
    """Highest reachable tag, rc or final."""
    stable: Optional[TagInfo]
    """Highest reachable final tag."""


def latest_tags(repo: Path, name: str) -> LatestTags:
    """Highest reachable tag of the component, any flavour and stable-only."""
    out = git(repo, "tag", "--merged", "HEAD", "--list", f"{name}-v*")
    candidates = [
        (version, tag)
        for tag in out.split()
        if (version := version_from_tag(name, tag)) is not None
    ]

    def pick(pool: list[tuple[Version, str]]) -> Optional[TagInfo]:
        if not pool:
            return None
        version, tag = max(pool)  # by Version
        return TagInfo(tag, version, distance(repo, name, tag))

    stable = [(v, t) for v, t in candidates if not v.is_prerelease]
    return LatestTags(pick(candidates), pick(stable))
