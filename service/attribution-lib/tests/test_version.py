# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

import _attribution_lib_version as versioning

_PROJECT_ROOT = Path(__file__).parents[1]


def _run_command(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _is_status_command(args: list[str]) -> bool:
    return args[:3] == ["status", "--porcelain", "--untracked-files=normal"]


def _write_dist_info(site_packages: Path, *, distribution_name: str, version: str, files: list[str]) -> None:
    dist_info = site_packages / f"{distribution_name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: {version}\n")
    record_entries = [f"{file},," for file in files]
    record_entries.extend(
        [
            f"{dist_info.name}/METADATA,,",
            f"{dist_info.name}/RECORD,,",
        ]
    )
    (dist_info / "RECORD").write_text("\n".join(record_entries))


def _assert_pep440_version(value: str, expected: str) -> None:
    assert value == expected
    Version(value)


def test_release_version_matches_bundle_manifest():
    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text())
    manifest = json.loads((_PROJECT_ROOT.parents[1] / "bundle-manifest.json").read_text())

    assert "packaging>=24" in pyproject["build-system"]["requires"]
    assert pyproject["project"]["version"] == manifest["attribution_lib_version"]


def test_file_scoped_public_helpers_delegate_from_given_file(monkeypatch, tmp_path):
    module_file = tmp_path / "src" / "_demo_project_version.py"
    module_file.parent.mkdir()
    module_file.touch()
    seen: dict[str, object] = {}

    def fake_get_version(module_path: Path) -> str:
        seen["version_path"] = module_path
        return "1.2.3"

    def fake_get_commit_sha(module_path: Path, *, short: bool) -> str:
        seen["sha_path"] = module_path
        seen["short"] = short
        return "abcdef123456"

    monkeypatch.setattr(versioning, "_get_version", fake_get_version)
    monkeypatch.setattr(versioning, "_get_commit_sha", fake_get_commit_sha)

    assert versioning.get_version_from_file(module_file) == "1.2.3"
    assert versioning.get_commit_sha_from_file(str(module_file), short=True) == "abcdef123456"
    assert seen == {
        "version_path": module_file.resolve(),
        "sha_path": module_file.resolve(),
        "short": True,
    }


def test_distribution_name_uses_pyproject_for_source_checkout(tmp_path):
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text('[project]\nname = "demo-lib"\n')
    module_path = tmp_path / "src" / "_demo_version.py"
    module_path.parent.mkdir()
    module_path.touch()

    assert versioning._distribution_name(module_path) == "demo-lib"


def test_distribution_name_uses_dist_info_record_for_installed_package(tmp_path):
    site_packages = tmp_path / ".venv" / "lib" / "site-packages"
    module_path = site_packages / "_demo_version.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    _write_dist_info(
        site_packages,
        distribution_name="demo-lib",
        version="1.2.3",
        files=["_demo_version.py"],
    )

    assert versioning._distribution_name(module_path) == "demo-lib"


def test_distribution_name_does_not_use_consumer_pyproject_for_installed_package(tmp_path):
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text('[project]\nname = "consumer-project"\n')
    module_path = tmp_path / ".venv" / "lib" / "site-packages" / "_demo_version.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    _write_dist_info(
        module_path.parent,
        distribution_name="demo-lib",
        version="1.2.3",
        files=["_demo_version.py"],
    )

    assert versioning._distribution_name(module_path) == "demo-lib"


def test_distribution_name_falls_back_to_packages_distributions_for_installed_package(monkeypatch):
    module_path = Path("/repo/.venv/lib/site-packages/_demo_version.py")
    monkeypatch.setattr(
        versioning.metadata,
        "packages_distributions",
        lambda: {"_demo_version": ["demo-lib"]},
    )

    assert versioning._distribution_name(module_path) == "demo-lib"


def test_distribution_name_fails_when_installed_distribution_is_unknown(monkeypatch):
    module_path = Path("/repo/.venv/lib/site-packages/_demo_version.py")
    monkeypatch.setattr(versioning.metadata, "packages_distributions", lambda: {})

    with pytest.raises(versioning.VersionError, match="installed distribution name"):
        versioning._distribution_name(module_path)


def test_built_wheel_uses_installed_metadata_outside_git_checkout(tmp_path):
    git = shutil.which("git")
    uv = shutil.which("uv")
    if git is None:
        pytest.skip("git executable is required for the build integration test")
    if uv is None:
        pytest.skip("uv executable is required for the build integration test")

    project = tmp_path / "project"
    src = project / "src"
    package = src / "demo_pkg"
    package.mkdir(parents=True)
    (project / ".gitignore").write_text("__pycache__/\n*.egg-info/\nbuild/\ndist/\nuv.lock\n")
    (src / "_demo_project_version.py").write_text((_PROJECT_ROOT / "src" / "_attribution_lib_version.py").read_text())
    (package / "__init__.py").write_text(
        "\n".join(
            [
                "from _demo_project_version import get_version as get_version",
                "",
                "",
                "def __getattr__(name: str) -> str:",
                '    if name == "__version__":',
                "        return get_version()",
                "    raise AttributeError(name)",
                "",
            ]
        )
    )
    (project / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo-version-pkg"',
                'dynamic = ["version"]',
                'requires-python = ">=3.11"',
                'dependencies = ["packaging>=24"]',
                "",
                "[build-system]",
                'requires = ["setuptools>=41", "wheel", "setuptools-git-versioning>=2.0,<3", "packaging>=24"]',
                'build-backend = "setuptools.build_meta"',
                "",
                "[tool.setuptools-git-versioning]",
                "enabled = true",
                'version_callback = "src._demo_project_version:get_version"',
                "",
                "[tool.setuptools]",
                'py-modules = ["_demo_project_version"]',
                "",
                "[tool.setuptools.packages.find]",
                'where = ["src"]',
                "",
            ]
        )
    )

    _run_command([git, "init"], cwd=project)
    _run_command([git, "add", "."], cwd=project)
    _run_command(
        [
            git,
            "-c",
            "user.name=Version Test",
            "-c",
            "user.email=version-test@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=project,
    )
    _run_command([git, "tag", "v1.2.3"], cwd=project)

    wheelhouse = tmp_path / "wheelhouse"
    _run_command([uv, "build", "--wheel", "--out-dir", str(wheelhouse)], cwd=project)
    wheels = list(wheelhouse.glob("demo_version_pkg-1.2.3-*.whl"))
    assert len(wheels) == 1

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    site_packages = tmp_path / "isolated" / "site-packages"
    _run_command([uv, "pip", "install", "--target", str(site_packages), "--no-deps", str(wheels[0])], cwd=run_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(site_packages)
    env["PATH"] = str(tmp_path / "no-git-on-path")
    completed = _run_command(
        [
            sys.executable,
            "-c",
            (
                "import os\n"
                "from pathlib import Path\n"
                "import shutil\n"
                "import demo_pkg\n"
                "import _demo_project_version\n"
                "site_packages = Path(os.environ['PYTHONPATH']).resolve()\n"
                "version_module = Path(_demo_project_version.__file__).resolve()\n"
                "print(shutil.which('git'))\n"
                "print(version_module.is_relative_to(site_packages))\n"
                "print(demo_pkg.get_version())\n"
                "print(demo_pkg.__version__)\n"
                "print(_demo_project_version.get_version())\n"
            ),
        ],
        cwd=run_dir,
        env=env,
    )

    assert completed.stdout.splitlines() == ["None", "True", "1.2.3", "1.2.3", "1.2.3"]


def test_get_version_uses_exact_git_tag(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args[:3] == ["describe", "--tags", "--long"]:
            return "v1.2.3-0-gabcdef1"
        if _is_status_command(args):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    _assert_pep440_version(versioning._get_version(root / "src" / "_attribution_lib_version.py"), "1.2.3")


def test_get_version_uses_dev_version_downstream_of_tag(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args[:3] == ["describe", "--tags", "--long"]:
            return "1.2.3-5-gabcdef1"
        if _is_status_command(args):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    _assert_pep440_version(versioning._get_version(root / "src" / "_attribution_lib_version.py"), "1.2.3.dev5")


def test_get_version_marks_dirty_checkout(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args[:3] == ["describe", "--tags", "--long"]:
            return "v1.2.3-5-gabcdef1"
        if _is_status_command(args):
            return " M pyproject.toml"
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    _assert_pep440_version(
        versioning._get_version(root / "src" / "_attribution_lib_version.py"),
        "1.2.3.dev5+dirty",
    )


def test_get_version_marks_dirty_exact_tag_checkout(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args[:3] == ["describe", "--tags", "--long"]:
            return "v1.2.3-0-gabcdef1"
        if _is_status_command(args):
            return "?? src/new-file.py"
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    _assert_pep440_version(
        versioning._get_version(root / "src" / "_attribution_lib_version.py"),
        "1.2.3.dev0+dirty",
    )


def test_get_version_ignores_lockfile_and_ignored_artifact_changes(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git executable is required for git dirty-state tests")

    project = tmp_path / "project"
    src = project / "src"
    src.mkdir(parents=True)
    module_path = src / "_demo_project_version.py"
    module_path.touch()
    (project / ".gitignore").write_text("build/\n")
    (project / "uv.lock").write_text("version = 1\n")
    (project / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo-version-pkg"',
                'dynamic = ["version"]',
                "",
                "[tool.setuptools.packages.find]",
                'where = ["src"]',
                "",
            ]
        )
    )

    _run_command([git, "init"], cwd=project)
    _run_command([git, "add", "."], cwd=project)
    _run_command(
        [
            git,
            "-c",
            "user.name=Version Test",
            "-c",
            "user.email=version-test@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=project,
    )
    _run_command([git, "tag", "v1.2.3"], cwd=project)

    (project / "uv.lock").write_text("version = 2\n")
    (project / "build").mkdir()
    (project / "build" / "artifact.txt").write_text("generated\n")

    assert versioning._get_version(module_path) == "1.2.3"

    (project / "unexpected.txt").write_text("not ignored\n")

    assert versioning._get_version(module_path) == "1.2.3.dev0+dirty"


def test_get_version_fails_without_tags(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args[:3] == ["describe", "--tags", "--long"]:
            return None
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    with pytest.raises(versioning.VersionError, match="no matching version tag"):
        versioning._get_version(root / "src" / "_attribution_lib_version.py")


def test_get_version_fails_when_dirty_state_is_unknown(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args[:3] == ["describe", "--tags", "--long"]:
            return "v1.2.3-5-gabcdef1"
        if _is_status_command(args):
            return None
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    with pytest.raises(versioning.VersionError, match="dirty"):
        versioning._get_version(root / "src" / "_attribution_lib_version.py")


def test_get_version_fails_on_unparseable_git_describe(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args[:3] == ["describe", "--tags", "--long"]:
            return "not-a-git-describe-value"
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    with pytest.raises(versioning.VersionError, match="Cannot parse"):
        versioning._get_version(root / "src" / "_attribution_lib_version.py")


def test_get_version_fails_on_non_pep440_git_tag(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args[:3] == ["describe", "--tags", "--long"]:
            return "v1.bad-0-gabcdef1"
        if _is_status_command(args):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    with pytest.raises(versioning.VersionError, match="non-PEP-440"):
        versioning._get_version(root / "src" / "_attribution_lib_version.py")


def test_get_version_fails_when_git_root_cannot_be_resolved(monkeypatch, tmp_path):
    module_path = tmp_path / "src" / "_attribution_lib_version.py"
    module_path.parent.mkdir()
    module_path.touch()
    (tmp_path / ".git").mkdir()

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return None
        raise AssertionError(args)

    def fail_metadata(_module_path: Path) -> str:
        raise AssertionError("metadata fallback should not be used for a git checkout")

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)
    monkeypatch.setattr(versioning, "_metadata_version", fail_metadata)

    with pytest.raises(versioning.VersionError, match="Cannot resolve git checkout root"):
        versioning._get_version(module_path)


def test_get_version_fails_when_git_executable_is_missing(monkeypatch, tmp_path):
    module_path = tmp_path / "src" / "_attribution_lib_version.py"
    module_path.parent.mkdir()
    module_path.touch()
    (tmp_path / ".git").mkdir()

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(versioning.subprocess, "run", missing_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    with pytest.raises(versioning.VersionError, match="Cannot resolve git checkout root"):
        versioning._get_version(module_path)


def test_get_version_uses_metadata_for_installed_package(monkeypatch):
    monkeypatch.setattr(versioning, "_metadata_version", lambda _module_path: "2.0.0")
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: True)

    assert versioning._get_version(Path("/repo/.venv/lib/site-packages/_attribution_lib_version.py")) == "2.0.0"


def test_get_version_uses_metadata_outside_git_repo(monkeypatch):
    monkeypatch.setattr(versioning, "_git_root", lambda _path: None)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)
    monkeypatch.setattr(versioning, "_metadata_version", lambda _module_path: "2.0.0")
    monkeypatch.setattr(versioning, "_has_git_metadata", lambda _path: False)

    assert versioning._get_version(Path("/tmp/_attribution_lib_version.py")) == "2.0.0"


def test_metadata_version_fails_when_distribution_metadata_is_missing(monkeypatch):
    def missing_metadata(_name: str) -> str:
        raise versioning.metadata.PackageNotFoundError

    monkeypatch.setattr(versioning, "_distribution_name", lambda _module_path: "demo-lib")
    monkeypatch.setattr(versioning.metadata, "version", missing_metadata)

    with pytest.raises(versioning.VersionError, match="metadata is not installed"):
        versioning._metadata_version()


def test_metadata_version_fails_on_invalid_distribution_metadata(monkeypatch):
    monkeypatch.setattr(versioning, "_distribution_name", lambda _module_path: "demo-lib")
    monkeypatch.setattr(versioning.metadata, "version", lambda _name: "not a version")

    with pytest.raises(versioning.VersionError, match="non-PEP-440"):
        versioning._metadata_version()


def test_get_commit_sha_uses_git_for_source_checkout(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args == ["rev-parse", "--short=12", "HEAD"]:
            return "abcdef123456"
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    assert versioning._get_commit_sha(root / "src" / "_attribution_lib_version.py", short=True) == "abcdef123456"


def test_get_commit_sha_uses_full_sha_by_default(monkeypatch):
    root = Path("/repo")
    full_sha = "abcdef1234567890abcdef1234567890abcdef12"

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args == ["rev-parse", "HEAD"]:
            return full_sha
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    assert versioning._get_commit_sha(root / "src" / "_attribution_lib_version.py", short=False) == full_sha


def test_get_commit_sha_fails_on_invalid_git_output(monkeypatch):
    root = Path("/repo")

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args == ["rev-parse", "HEAD"]:
            return "not-a-sha"
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)

    with pytest.raises(versioning.VersionError, match="invalid commit SHA"):
        versioning._get_commit_sha(root / "src" / "_attribution_lib_version.py", short=False)


def test_get_commit_sha_fails_for_installed_package(monkeypatch):
    def fail_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return None
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fail_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: True)

    with pytest.raises(versioning.VersionError, match="not inside a git checkout"):
        versioning._get_commit_sha(Path("/repo/.venv/lib/site-packages/_attribution_lib_version.py"), short=True)


def test_get_commit_sha_fails_outside_git_repo(monkeypatch):
    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return None
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "_is_installed_package", lambda _path: False)

    with pytest.raises(versioning.VersionError, match="not inside a git checkout"):
        versioning._get_commit_sha(Path("/tmp/_attribution_lib_version.py"), short=True)


def test_attribution_lib_exports_version_api(monkeypatch):
    root = Path("/repo")
    full_sha = "abcdef1234567890abcdef1234567890abcdef12"

    def fake_run_git(_cwd: Path, args: list[str]) -> str | None:
        if args == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if args[:3] == ["describe", "--tags", "--long"]:
            return "v9.8.7-0-gabcdef1"
        if _is_status_command(args):
            return ""
        if args == ["rev-parse", "--short=12", "HEAD"]:
            return full_sha[:12]
        raise AssertionError(args)

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)

    sys.modules.pop("attribution_lib", None)
    sys.modules.pop("attribution_lib.regex_accelerator", None)

    import attribution_lib  # noqa: PLC0415
    import attribution_lib.regex_accelerator as regex_accelerator  # noqa: PLC0415

    assert attribution_lib.__version__ == attribution_lib.get_version()
    assert Version(attribution_lib.get_version())
    assert attribution_lib.get_commit_sha(short=True) == full_sha[:12]
    assert not hasattr(regex_accelerator, "__version__")
    assert not hasattr(regex_accelerator, "get_version")
    assert not hasattr(regex_accelerator, "get_commit_sha")
