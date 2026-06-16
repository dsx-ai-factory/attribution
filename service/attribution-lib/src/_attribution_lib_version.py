# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Reusable version helpers shared by build metadata and runtime package APIs."""

from __future__ import annotations

import re
import subprocess
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Optional

from packaging.version import InvalidVersion, Version

_GIT_TIMEOUT_SECONDS = 2
_SITE_PACKAGE_PARTS = frozenset({"site-packages", "dist-packages"})
_SHA_RE = re.compile(r"\A[0-9a-fA-F]{7,40}\Z")


class VersionError(RuntimeError):
    """Raised when package version information cannot be determined safely."""


def get_version() -> str:
    """Return the PEP-440 package version from git or installed metadata."""
    return get_version_from_file(__file__)


def get_commit_sha(*, short: bool = False) -> str:
    """Return the current git commit SHA for source and editable checkouts."""
    return get_commit_sha_from_file(__file__, short=short)


def get_version_from_file(module_file: str | Path) -> str:
    """Return the package version for the project containing ``module_file``."""
    return _get_version(Path(module_file).resolve())


def get_commit_sha_from_file(module_file: str | Path, *, short: bool = False) -> str:
    """Return the commit SHA for the git checkout containing ``module_file``."""
    return _get_commit_sha(Path(module_file).resolve(), short=short)


def _get_version(module_path: Path) -> str:
    """Choose git-derived versions for checkouts and metadata versions for installs."""
    if _is_installed_package(module_path):
        return _metadata_version(module_path)

    root = _git_root(module_path.parent)
    if root is not None:
        return _git_version(root)
    if _has_git_metadata(module_path.parent):
        raise VersionError(f"Cannot resolve git checkout root for {module_path}.")

    return _metadata_version(module_path)


def _get_commit_sha(module_path: Path, *, short: bool) -> str:
    """Resolve a commit SHA from the git checkout containing ``module_path``."""
    root = _git_root(module_path.parent)
    if root is None:
        raise VersionError(f"Cannot determine commit SHA because {module_path} is not inside a git checkout.")

    args = ["rev-parse"]
    if short:
        args.append("--short=12")
    args.append("HEAD")
    commit_sha = _run_git(root, args)
    if commit_sha is None:
        raise VersionError(f"Cannot determine commit SHA from git checkout at {root}.")
    _validate_commit_sha(commit_sha, f"git checkout at {root}")
    return commit_sha


def _git_version(root: Path) -> str:
    """Return a PEP-440 version from ``git describe`` and dirty status."""
    describe = _run_git(
        root,
        [
            "describe",
            "--tags",
            "--long",
            "--match",
            "v[0-9]*",
            "--match",
            "[0-9]*",
        ],
    )
    if describe is None:
        raise VersionError(
            f"Cannot determine package version because git checkout at {root} has no matching version tag. "
            "Create a vX.Y.Z tag."
        )

    parsed = _parse_describe(describe)
    if parsed is None:
        raise VersionError(f"Cannot parse git describe output {describe!r} from {root}.")

    tag, commit_count = parsed
    base_version = _strip_version_tag_prefix(tag)
    dirty = _is_dirty(root)
    if commit_count == 0 and not dirty:
        return _validate_version(base_version, f"git describe output {describe!r}")

    version = f"{base_version}.dev{commit_count}"
    if dirty:
        version = f"{version}+dirty"
    return _validate_version(version, f"git describe output {describe!r}")


def _metadata_version(module_path: Optional[Path] = None) -> str:
    """Return the installed distribution version that owns ``module_path``."""
    distribution_name = _distribution_name(module_path or Path(__file__).resolve())
    try:
        return _validate_version(metadata.version(distribution_name), f"{distribution_name!r} metadata")
    except metadata.PackageNotFoundError:
        raise VersionError(
            f"Cannot determine package version because {distribution_name!r} metadata is not installed."
        ) from None


def _distribution_name(module_path: Path) -> str:
    """Find the project/distribution name for a source or installed version module."""
    if _is_installed_package(module_path):
        metadata_name = _distribution_name_from_installed_metadata(module_path)
        if metadata_name is not None:
            return metadata_name
        raise VersionError(f"Cannot determine installed distribution name for module {module_path.stem!r}.")

    pyproject_path = _find_pyproject(module_path.parent)
    if pyproject_path is not None:
        return _project_name_from_pyproject(pyproject_path)

    metadata_name = _distribution_name_from_import_metadata(module_path.stem)
    if metadata_name is not None:
        return metadata_name

    raise VersionError(f"Cannot determine distribution name for module {module_path.stem!r}.")


def _distribution_name_from_installed_metadata(module_path: Path) -> Optional[str]:
    """Resolve the owning distribution by matching dist-info RECORD entries."""
    site_root = _site_packages_root(module_path)
    if site_root is not None:
        relative_module_path = module_path.relative_to(site_root).as_posix()
        distributions = []
        for distribution in metadata.distributions(path=[str(site_root)]):
            files = distribution.files
            if files is None:
                continue
            if any(file.as_posix() == relative_module_path for file in files):
                distributions.append(distribution.metadata["Name"])
        if len(distributions) == 1:
            return distributions[0]
        if len(distributions) > 1:
            raise VersionError(
                f"Cannot determine distribution name for module {module_path.stem!r}: "
                f"multiple distributions provide it: {', '.join(distributions)}."
            )

    return _distribution_name_from_import_metadata(module_path.stem)


def _site_packages_root(module_path: Path) -> Optional[Path]:
    """Return the nearest site-packages/dist-packages parent for ``module_path``."""
    for index, part in enumerate(module_path.parts):
        if part in _SITE_PACKAGE_PARTS:
            return Path(*module_path.parts[: index + 1])
    return None


def _find_pyproject(start_dir: Path) -> Optional[Path]:
    """Find the nearest parent pyproject.toml."""
    for path in (start_dir, *start_dir.parents):
        pyproject_path = path / "pyproject.toml"
        if pyproject_path.is_file():
            return pyproject_path
    return None


def _project_name_from_pyproject(pyproject_path: Path) -> str:
    """Read ``[project].name`` from a pyproject.toml."""
    try:
        pyproject = tomllib.loads(pyproject_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise VersionError(f"Cannot read project name from {pyproject_path}.") from exc

    project_name = pyproject.get("project", {}).get("name")
    if not isinstance(project_name, str) or not project_name:
        raise VersionError(f"Cannot read [project].name from {pyproject_path}.")
    return project_name


def _distribution_name_from_import_metadata(top_level_name: str) -> Optional[str]:
    """Fallback to importlib's top-level package to distribution mapping."""
    distributions = metadata.packages_distributions().get(top_level_name, [])
    if len(distributions) == 1:
        return distributions[0]
    if len(distributions) > 1:
        raise VersionError(
            f"Cannot determine distribution name for module {top_level_name!r}: "
            f"multiple distributions provide it: {', '.join(distributions)}."
        )
    return None


def _git_root(start_dir: Path) -> Optional[Path]:
    """Return the git checkout root for ``start_dir``, if git can resolve one."""
    root = _run_git(start_dir, ["rev-parse", "--show-toplevel"])
    if root is None:
        return None
    return Path(root)


def _parse_describe(describe: str) -> Optional[tuple[str, int]]:
    """Parse ``git describe --long`` output into tag and commit count."""
    try:
        tag, commit_count, commit = describe.rsplit("-", 2)
    except ValueError:
        return None

    if not commit_count.isdecimal() or not commit.startswith("g"):
        return None
    return tag, int(commit_count)


def _strip_version_tag_prefix(tag: str) -> str:
    """Strip a semantic-version tag's leading ``v`` prefix."""
    if len(tag) > 1 and tag[0] in {"v", "V"} and tag[1].isdigit():
        return tag[1:]
    return tag


def _is_dirty(root: Path) -> bool:
    """Return whether the checkout has committed or unexpected untracked changes."""
    status = _run_git(root, ["status", "--porcelain", "--untracked-files=normal", "--", ".", ":(exclude)uv.lock"])
    if status is None:
        raise VersionError(f"Cannot determine whether git checkout at {root} is dirty.")
    return bool(status)


def _is_installed_package(module_path: Path) -> bool:
    """Return whether ``module_path`` is under a Python install location."""
    return any(part in _SITE_PACKAGE_PARTS for part in module_path.parts)


def _has_git_metadata(start_dir: Path) -> bool:
    """Return whether ``start_dir`` is inside a directory containing .git metadata."""
    return any((path / ".git").exists() for path in (start_dir, *start_dir.parents))


def _validate_version(version: str, source: str) -> str:
    """Fail unless ``version`` is PEP-440 compliant."""
    try:
        Version(version)
    except InvalidVersion as exc:
        raise VersionError(f"{source} produced non-PEP-440 version {version!r}.") from exc
    return version


def _validate_commit_sha(commit_sha: str, source: str) -> None:
    """Fail unless ``commit_sha`` looks like a git SHA."""
    if not _SHA_RE.fullmatch(commit_sha):
        raise VersionError(f"{source} produced invalid commit SHA {commit_sha!r}.")


def _run_git(cwd: Path, args: list[str]) -> Optional[str]:
    """Run a git command and return stdout, or ``None`` for unavailable git."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.strip()
