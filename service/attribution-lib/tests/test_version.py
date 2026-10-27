# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import attribution_lib

_PROJECT_ROOT = Path(__file__).parents[1]
_HELPER_FILE = _PROJECT_ROOT / "src" / "_build_version.py"
_LIB_SUBDIR = Path("service/attribution-lib")
_LITERAL_FILE = Path("src/attribution_lib/__init__.py")

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Version Test",
    "GIT_AUTHOR_EMAIL": "version-test@example.invalid",
    "GIT_COMMITTER_NAME": "Version Test",
    "GIT_COMMITTER_EMAIL": "version-test@example.invalid",
}

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git executable required")


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_build_version", _HELPER_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = _load_helper()


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "commit.gpgsign=false", *args],
        env=_GIT_ENV,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_lib(project_dir: Path, literal: str) -> None:
    (project_dir / _LITERAL_FILE).parent.mkdir(parents=True, exist_ok=True)
    (project_dir / _LITERAL_FILE).write_text(f'__version__ = "{literal}"\n')
    shutil.copy(_HELPER_FILE, project_dir / "src" / _HELPER_FILE.name)


def _commit_all(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fresh repository with the lib committed at `<root>/service/attribution-lib`, untagged."""
    root = tmp_path / "repo"
    _write_lib(root / _LIB_SUBDIR, "1.2.3")
    _git(root.parent, "init", "-q", str(root))
    _commit_all(root, "initial")
    return root


def test_dunder_version_is_the_committed_literal():
    assert attribution_lib.__version__ == helper._read_literal()


def test_latest_tag_orders_stable_above_rc_and_ignores_other_tags():
    tags = [
        "attribution-lib-v1.0.0-rc.2",
        "attribution-lib-v1.0.0-rc.10",
        "attribution-lib-v0.10.0",
        "service-app-v9.9.9",
        "junk",
    ]
    assert helper._latest_tag(tags) == "attribution-lib-v1.0.0-rc.10"
    assert helper._latest_tag([*tags, "attribution-lib-v1.0.0"]) == "attribution-lib-v1.0.0"
    assert helper._latest_tag([]) is None
    assert helper._latest_tag(["service-app-v9.9.9", "junk"]) is None


@needs_git
def test_build_version_tracks_dirty_and_tag_distance(repo: Path):
    project_dir = repo / _LIB_SUBDIR
    (project_dir / "src" / _HELPER_FILE.name).write_text("# edited\n")
    assert helper.build_version(project_dir=project_dir) == "1.2.3+dirty"
    _git(repo, "checkout", "--", ".")

    _git(repo, "tag", "-a", "attribution-lib-v1.2.3", "-m", "x")
    assert helper.build_version(project_dir=project_dir) == "1.2.3"

    (repo / "blueprint").mkdir()
    (repo / "blueprint" / "README.md").write_text("unrelated\n")
    _commit_all(repo, "blueprint change")
    assert helper.build_version(project_dir=project_dir) == "1.2.3"

    (project_dir / "src" / "extra.py").write_text("x = 1\n")
    _commit_all(repo, "lib change")
    assert helper.build_version(project_dir=project_dir) == "1.2.3+1"

    (project_dir / "src" / "extra.py").write_text("x = 2\n")
    assert helper.build_version(project_dir=project_dir) == "1.2.3+1.dirty"


def test_helper_never_imports_the_package(monkeypatch):
    for name in [n for n in sys.modules if n == "attribution_lib" or n.startswith("attribution_lib.")]:
        monkeypatch.delitem(sys.modules, name)

    fresh = _load_helper()
    fresh.build_version()

    assert "attribution_lib" not in sys.modules


@needs_git
class TestFallsBackToLiteral:
    def test_project_outside_any_repo(self, tmp_path: Path):
        project_dir = tmp_path / "loose"
        _write_lib(project_dir, "1.2.3")
        assert helper.build_version(project_dir=project_dir) == "1.2.3"

    def test_repo_without_matching_tag(self, repo: Path):
        _git(repo, "tag", "service-app-v9.9.9")
        assert helper.build_version(project_dir=repo / _LIB_SUBDIR) == "1.2.3"

    def test_lib_vendored_outside_its_expected_path(self, repo: Path):
        project_dir = repo / "vendor" / "attribution-lib"
        _write_lib(project_dir, "1.2.3")
        _commit_all(repo, "vendor the lib")
        _git(repo, "tag", "attribution-lib-v1.2.3")
        assert helper.build_version(project_dir=project_dir) == "1.2.3"

    @pytest.mark.parametrize(
        "git_failure",
        [FileNotFoundError("git"), subprocess.TimeoutExpired(cmd="git", timeout=5)],
        ids=["missing", "timeout"],
    )
    def test_git_unavailable(self, repo: Path, monkeypatch, git_failure: Exception):
        _git(repo, "tag", "attribution-lib-v1.2.3")

        def _failing_run(*_args, **_kwargs):
            raise git_failure

        monkeypatch.setattr(helper.subprocess, "run", _failing_run)
        assert helper.build_version(project_dir=repo / _LIB_SUBDIR) == "1.2.3"
