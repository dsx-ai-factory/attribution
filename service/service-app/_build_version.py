# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build version for the attribution service: the committed `__version__` literal plus `+<distance>[.dirty]`.

Read by hatchling through `[tool.hatch.version] source = "code"`. Any git failure yields the bare literal.
"""

from __future__ import annotations

from collections.abc import Iterable
import re
import subprocess
from pathlib import Path
from typing import Optional, TypeAlias

_COMPONENT = "service-app"
_PATHS = ["service/service-app", "service/attribution-lib"]  # repo-relative; commits touching these count toward distance
_PROJECT_DIR = Path(__file__).resolve().parent
_LITERAL_FILE = Path("app/__init__.py")
_GIT_TIMEOUT_S = 5
_LITERAL_RE = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
_TAG_RE = re.compile(rf"^{re.escape(_COMPONENT)}-v(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$")

# (major, minor, patch, is_stable, rc): stable sorts above every rc of the same X.Y.Z, rcs sort numerically.
_TagKey: TypeAlias = tuple[int, int, int, bool, int]


def _read_literal(project_dir: Path = _PROJECT_DIR) -> str:
    """Return the committed `__version__` literal; raise ValueError if absent."""
    match = _LITERAL_RE.search((project_dir / _LITERAL_FILE).read_text())
    if match is None:
        raise ValueError(f"no __version__ literal in {project_dir / _LITERAL_FILE}")
    return match.group(1)


def _find_root(start: Path) -> Optional[Path]:
    """Return the nearest ancestor (or `start`) containing `.git`, as a file or directory."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git(cwd: Path, *args: str) -> Optional[str]:
    # `-C <root>` so the repo-relative pathspecs in _PATHS resolve regardless of the caller's cwd.
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _tag_key(tag: str) -> Optional[_TagKey]:
    match = _TAG_RE.match(tag)
    if match is None:
        return None
    major, minor, patch, rc = match.groups()
    return (int(major), int(minor), int(patch), rc is None, int(rc or 0))


def _latest_tag(tags: Iterable[str]) -> Optional[str]:
    keyed = [(key, tag) for tag in tags if (key := _tag_key(tag)) is not None]
    if not keyed:
        return None
    return max(keyed)[1]


def build_version(project_dir: Path = _PROJECT_DIR) -> str:
    literal = _read_literal(project_dir)
    root = _find_root(project_dir)
    if root is None:
        return literal
    # The component may be exported alone into another repository, in which case its tags and history are not ours.
    if (root / _PATHS[0]).resolve() != project_dir.resolve():
        return literal

    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no", "--", *_PATHS))

    # Distance is measured from the latest reachable tag; with none, only the dirty marker applies.
    distance = 0
    tags = _git(root, "tag", "--merged", "HEAD", "--list", f"{_COMPONENT}-v*")
    tag = _latest_tag(tags.splitlines()) if tags is not None else None
    if tag is not None:
        count = _git(root, "rev-list", "--count", "--first-parent", f"{tag}..HEAD", "--", *_PATHS)
        if count is not None and count.isdecimal():
            distance = int(count)

    if distance == 0:
        return f"{literal}+dirty" if dirty else literal

    return f"{literal}+{distance}.dirty" if dirty else f"{literal}+{distance}"
