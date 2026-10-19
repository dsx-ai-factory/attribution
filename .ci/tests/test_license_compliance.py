# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPOSITORY_ROOT / ".ci" / "check-license-compliance.py"
IDENTIFIER_TAG = "SPDX-License-" + "Identifier:"
COPYRIGHT_TAG = "SPDX-FileCopyright" + "Text:"
APACHE_IDENTIFIER = f"{IDENTIFIER_TAG} Apache-2.0"
CANONICAL_LICENSE_PATHS = (
    "LICENSE",
    "blueprint/LICENSE",
    "service/attribution-lib/LICENSE",
    "service/service-app/LICENSE",
    "service/service-app/openapi/extras/LICENSE",
    "service/service-app/nvidia_attribution_client/LICENSE",
)
PYPROJECT_PATHS = (
    "blueprint/pyproject.toml",
    "service/attribution-lib/pyproject.toml",
    "service/service-app/pyproject.toml",
    "service/service-app/openapi/extras/pyproject.toml",
    "service/service-app/nvidia_attribution_client/pyproject.toml",
)
DERIVED_TEMPLATE_PATHS = (
    "service/service-app/openapi/templates/__init__api.mustache",
    "service/service-app/openapi/templates/api_response.mustache",
    "service/service-app/openapi/templates/partial_api_args.mustache",
)
EXPECTED_DISTRIBUTIONS = (
    "attribution-lib",
    "nvidia-attribution-blueprint",
    "nvidia-attribution-client",
    "nvidia-attribution-service",
)
SOURCE_VERSION_PATHS = {
    "attribution-lib": "service/attribution-lib/src/attribution_lib/__init__.py",
    "nvidia-attribution-blueprint": "blueprint/src/attribution_blueprint/__init__.py",
    "nvidia-attribution-client": "service/service-app/nvidia_attribution_client/nvidia_attribution_client/__init__.py",
    "nvidia-attribution-service": "service/service-app/app/__init__.py",
}
EXPECTED_VERSIONS = {
    distribution: re.search(
        r'^__version__ = "([^"]+)"$', (REPOSITORY_ROOT / path).read_text(), re.MULTILINE
    ).group(1)
    for distribution, path in SOURCE_VERSION_PATHS.items()
}


HEADER_SURFACE_PATHS = (
    "AGENTS.md",
    "docs/attribution-tour/slides.md",
    "docs/attribution-tour/components/AttributionStoryboard.vue",
    "docs/attribution-tour/styles/index.css",
    "blueprint/src/attribution_blueprint/webui/frontend/src/app.tsx",
    "blueprint/src/attribution_blueprint/webui/frontend/src/app.test.tsx",
)

FIXTURE_PATHS = (
    *HEADER_SURFACE_PATHS,
    *SOURCE_VERSION_PATHS.values(),
    ".ci/build-and-check-license-artifacts.sh",
    ".ci/check-client-drift.sh",
    ".ci/check-license-compliance.py",
    ".ci/run-end-to-end-tests.sh",
    ".ci/run-tests.sh",
    ".ci/tests/test_license_compliance.py",
    ".coderabbit.yaml",
    ".github/workflows/tests.yml",
    ".pre-commit-config.yaml",
    "CODE_OF_CONDUCT.md",
    "LICENSE",
    "README.md",
    "blueprint/LICENSE",
    "blueprint/docker/Dockerfile",
    "blueprint/docker/Dockerfile.dockerignore",
    "blueprint/pyproject.toml",
    "blueprint/ruff.toml",
    "blueprint/src/attribution_blueprint/webui/frontend/package.json",
    "service/attribution-lib/LICENSE",
    "service/attribution-lib/docs/adding-a-rule-agent-guide.md",
    "service/attribution-lib/docs/adding-a-rule.md",
    "service/attribution-lib/pyproject.toml",
    "service/attribution-lib/src/attribution_lib/bundled_rules/applog_prefilter.toml",
    "service/attribution-lib/src/attribution_lib/xid_analyzer/nvidia_xid_analyzer.py",
    "service/attribution-lib/stubs/ahocorasick.pyi",
    "service/service-app/Dockerfile",
    "service/service-app/Dockerfile.dockerignore",
    "service/service-app/LICENSE",
    "service/service-app/app/__init__.py",
    "service/service-app/nvidia_attribution_client/LICENSE",
    "service/service-app/nvidia_attribution_client/nvidia_attribution_client/__init__.py",
    "service/service-app/nvidia_attribution_client/nvidia_attribution_client/models/__init__.py",
    "service/service-app/nvidia_attribution_client/pyproject.toml",
    "service/service-app/openapi/extras/LICENSE",
    "service/service-app/openapi/extras/pyproject.toml",
    "service/service-app/openapi/templates/LICENSE",
    *DERIVED_TEMPLATE_PATHS,
    "service/service-app/openapi/templates/partial_header.mustache",
    "service/service-app/pyproject.toml",
)


def run_checker(
    root: Path, artifacts: Path | None = None
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(CHECKER), "--root", str(root)]
    if artifacts is not None:
        command.extend(("--artifacts", str(artifacts)))
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@contextmanager
def valid_fixture() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        fixture_root = Path(temporary_directory)
        for relative_path in FIXTURE_PATHS:
            source = REPOSITORY_ROOT / relative_path
            destination = fixture_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        subprocess.run(
            ["git", "init", "--quiet", str(fixture_root)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(fixture_root), "add", "."],
            check=True,
            capture_output=True,
            text=True,
        )
        yield fixture_root


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def metadata_bytes(
    name: str,
    version: str = EXPECTED_VERSIONS["attribution-lib"],
    *,
    metadata_version: str = "2.4",
    license_expression: str = "Apache-2.0",
    license_files: tuple[str, ...] = ("LICENSE",),
    extra_fields: tuple[tuple[str, str], ...] = (),
) -> bytes:
    lines = [
        f"Metadata-Version: {metadata_version}",
        f"Name: {name}",
        f"Version: {version}",
        f"License-Expression: {license_expression}",
        *(f"License-File: {license_file}" for license_file in license_files),
        *(f"{field}: {value}" for field, value in extra_fields),
        "",
        "",
    ]
    return "\n".join(lines).encode()


def write_wheel(
    artifacts: Path,
    canonical: bytes,
    name: str,
    *,
    version: str = "1.0.0",
    filename_name: str | None = None,
    dist_info: str | None = None,
    license_member: str | None = None,
    license_bytes: bytes | None = None,
    include_static: bool = True,
    include_metadata: bool = True,
    include_license: bool = True,
    metadata: bytes | None = None,
    extra_members: tuple[tuple[str, bytes, int | None], ...] = (),
) -> Path:
    artifact_name = normalized_name(filename_name or name)
    path = artifacts / f"{artifact_name}-{version}-py3-none-any.whl"
    metadata_root = dist_info or f"{normalized_name(name)}-{version}.dist-info"
    metadata_path = f"{metadata_root}/METADATA"
    license_path = license_member or f"{metadata_root}/licenses/LICENSE"
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        if include_metadata:
            archive.writestr(metadata_path, metadata or metadata_bytes(name, version))
        if include_license:
            archive.writestr(
                license_path, canonical if license_bytes is None else license_bytes
            )
        if name == "nvidia-attribution-blueprint" and include_static:
            archive.writestr(
                "attribution_blueprint/webui/static/index.html",
                '<link rel="icon" href="/logo.svg">'
                '<link rel="stylesheet" href="/assets/app.css">'
                '<script type="module" src="/assets/app.js"></script>',
            )
            archive.writestr(
                "attribution_blueprint/webui/static/logo.svg", "<svg></svg>"
            )
            archive.writestr(
                "attribution_blueprint/webui/static/assets/app.css", "body{}"
            )
            archive.writestr(
                "attribution_blueprint/webui/static/assets/app.js", "export{}"
            )
            archive.writestr(
                "attribution_blueprint/webui/static/assets/THIRD_PARTY_NOTICES.txt",
                "notices",
            )
        for member_name, data, file_type in extra_members:
            info = zipfile.ZipInfo(member_name)
            if file_type is not None:
                info.external_attr = file_type << 16
            archive.writestr(info, data)
    return path


def add_tar_bytes(
    archive: tarfile.TarFile, name: str, data: bytes, type_: bytes | None = None
) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    if type_ is not None:
        member.type = type_
    archive.addfile(member, io.BytesIO(data))


def write_sdist(
    artifacts: Path,
    canonical: bytes,
    name: str,
    *,
    version: str = "1.0.0",
    filename_name: str | None = None,
    root_name: str | None = None,
    license_member: str | None = None,
    license_bytes: bytes | None = None,
    include_static: bool = True,
    include_metadata: bool = True,
    include_license: bool = True,
    metadata: bytes | None = None,
    extra_members: tuple[tuple[str, bytes, bytes | None], ...] = (),
) -> Path:
    artifact_name = normalized_name(filename_name or name)
    path = artifacts / f"{artifact_name}-{version}.tar.gz"
    package_root = root_name or f"{normalized_name(name)}-{version}"
    with tarfile.open(path, mode="w:gz") as archive:
        if include_metadata:
            add_tar_bytes(
                archive,
                f"{package_root}/PKG-INFO",
                metadata or metadata_bytes(name, version),
            )
        if include_license:
            add_tar_bytes(
                archive,
                license_member or f"{package_root}/LICENSE",
                canonical if license_bytes is None else license_bytes,
            )
        if name == "nvidia-attribution-blueprint" and include_static:
            add_tar_bytes(
                archive,
                f"{package_root}/src/attribution_blueprint/webui/static/index.html",
                b'<link rel="icon" href="/logo.svg">'
                b'<link rel="stylesheet" href="/assets/app.css">'
                b'<script type="module" src="/assets/app.js"></script>',
            )
            add_tar_bytes(
                archive,
                f"{package_root}/src/attribution_blueprint/webui/static/logo.svg",
                b"<svg></svg>",
            )
            add_tar_bytes(
                archive,
                f"{package_root}/src/attribution_blueprint/webui/static/assets/app.css",
                b"body{}",
            )
            add_tar_bytes(
                archive,
                f"{package_root}/src/attribution_blueprint/webui/static/assets/app.js",
                b"export{}",
            )
            add_tar_bytes(
                archive,
                f"{package_root}/src/attribution_blueprint/webui/static/assets/THIRD_PARTY_NOTICES.txt",
                b"notices",
            )
        for member_name, data, type_ in extra_members:
            add_tar_bytes(archive, member_name, data, type_)
    return path


def make_artifact_set(root: Path) -> Path:
    artifacts = root / "artifacts"
    artifacts.mkdir()
    canonical = (root / "LICENSE").read_bytes()
    for distribution in EXPECTED_DISTRIBUTIONS:
        version = EXPECTED_VERSIONS[distribution]
        write_wheel(artifacts, canonical, distribution, version=version)
        write_sdist(artifacts, canonical, distribution, version=version)
    return artifacts


class LicenseComplianceTest(unittest.TestCase):
    def assert_compliant(self, root: Path, artifacts: Path | None = None) -> None:
        result = run_checker(root, artifacts)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected_scope = "repository and artifacts" if artifacts else "repository"
        self.assertEqual(
            result.stdout, f"license: {expected_scope} compliance verified\n"
        )

    def assert_violation(
        self,
        root: Path,
        expected: str,
        artifacts: Path | None = None,
    ) -> None:
        result = run_checker(root, artifacts)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(expected, result.stderr)

    def git(self, root: Path, *arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )

    def stage(self, root: Path, relative_path: str) -> None:
        self.git(root, "add", "--", relative_path)

    def replace_and_stage(
        self,
        root: Path,
        relative_path: str,
        old: str,
        new: str,
    ) -> None:
        path = root / relative_path
        content = path.read_text(encoding="utf-8")
        self.assertIn(old, content)
        path.write_text(content.replace(old, new, 1), encoding="utf-8")
        self.stage(root, relative_path)

    def append_bytes_and_stage(
        self, root: Path, relative_path: str, addition: bytes
    ) -> None:
        path = root / relative_path
        path.write_bytes(path.read_bytes() + addition)
        self.stage(root, relative_path)

    def overwrite_artifact(
        self,
        root: Path,
        distribution: str,
        kind: str,
        **options: object,
    ) -> Path:
        artifacts = make_artifact_set(root)
        canonical = (root / "LICENSE").read_bytes()
        options.setdefault("version", EXPECTED_VERSIONS[distribution])
        if kind == "wheel":
            write_wheel(artifacts, canonical, distribution, **options)
        else:
            write_sdist(artifacts, canonical, distribution, **options)
        return artifacts

    def test_accepts_a_fully_compliant_index_snapshot(self) -> None:
        with valid_fixture() as root:
            self.assert_compliant(root)

    def test_accepts_html_and_svg_headers_with_indented_comment_closers(self) -> None:
        for suffix in ("html", "svg"):
            with self.subTest(suffix=suffix), valid_fixture() as root:
                relative_path = f"header-example.{suffix}"
                (root / relative_path).write_text(
                    "<!--\n"
                    f"{COPYRIGHT_TAG} Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
                    f"{APACHE_IDENTIFIER}\n"
                    " -->\n",
                    encoding="utf-8",
                )
                self.stage(root, relative_path)
                self.assert_compliant(root)
                self.replace_and_stage(root, relative_path, " -->", " --")
                self.assert_violation(
                    root, f"{relative_path}: expected a closed SPDX HTML comment"
                )

    def test_accepts_all_expected_wheels_and_sdists(self) -> None:
        with valid_fixture() as root:
            artifacts = make_artifact_set(root)
            self.assert_compliant(root, artifacts)

    def test_rejects_each_mutated_canonical_license(self) -> None:
        for relative_path in CANONICAL_LICENSE_PATHS:
            with self.subTest(relative_path=relative_path), valid_fixture() as root:
                self.append_bytes_and_stage(root, relative_path, b"\n")
                self.assert_violation(
                    root,
                    f"{relative_path}: expected canonical Apache-2.0 text",
                )

    def test_rejects_each_canonical_license_removed_only_from_the_index(self) -> None:
        for relative_path in CANONICAL_LICENSE_PATHS:
            with self.subTest(relative_path=relative_path), valid_fixture() as root:
                self.git(root, "rm", "--cached", "-f", "--", relative_path)
                self.assertTrue((root / relative_path).is_file())
                self.assert_violation(
                    root,
                    f"{relative_path}: required contract file is not tracked",
                )

    def test_rejects_changed_protected_legal_boundaries(self) -> None:
        cases = (
            (
                "CODE_OF_CONDUCT.md",
                "CODE_OF_CONDUCT.md: protected CC BY-SA 4.0 boundary changed",
            ),
            (
                "service/service-app/openapi/templates/LICENSE",
                "service/service-app/openapi/templates/LICENSE: protected upstream license changed",
            ),
        )
        for relative_path, diagnostic in cases:
            with self.subTest(relative_path=relative_path), valid_fixture() as root:
                self.append_bytes_and_stage(root, relative_path, b"\n")
                self.assert_violation(root, diagnostic)

    def test_rejects_all_legal_file_basename_variants(self) -> None:
        for filename in ("LICENSE.txt", "LICENCE.md", "COPYING", "NOTICE.rst", "EULA"):
            relative_path = f"unreviewed/{filename}"
            with self.subTest(filename=filename), valid_fixture() as root:
                path = root / relative_path
                path.parent.mkdir()
                path.write_text("unreviewed legal text\n", encoding="utf-8")
                self.stage(root, relative_path)
                self.assert_violation(root, f"{relative_path}: undeclared license file")

    def test_rejects_every_proprietary_relic_without_self_poisoning(self) -> None:
        markers = (
            ("LicenseRef-", "NvidiaProprietary"),
            ("Private :: Do Not ", "Upload"),
            ("GOVERNING DOWNLOAD ", "TERMS"),
            ("retain all intellectual ", "property and proprietary rights"),
            ("without an express license ", "agreement"),
            ("NVIDIA Software and Model Evaluation ", "License"),
        )
        for prefix, suffix in markers:
            with self.subTest(marker=f"{prefix}..."), valid_fixture() as root:
                self.append_bytes_and_stage(
                    root,
                    "README.md",
                    f"\n{prefix}{suffix}\n".encode(),
                )
                self.assert_violation(root, "README.md: proprietary license relic")

    def test_rejects_wrong_identifiers_for_every_header_syntax_class(self) -> None:
        header_paths = (
            *HEADER_SURFACE_PATHS,
            ".ci/check-license-compliance.py",
            ".ci/run-tests.sh",
            ".ci/tests/test_license_compliance.py",
            ".coderabbit.yaml",
            ".pre-commit-config.yaml",
            "blueprint/docker/Dockerfile",
            "blueprint/pyproject.toml",
            "blueprint/ruff.toml",
            "service/attribution-lib/docs/adding-a-rule.md",
            "service/attribution-lib/src/attribution_lib/bundled_rules/applog_prefilter.toml",
            "service/attribution-lib/src/attribution_lib/xid_analyzer/nvidia_xid_analyzer.py",
            "service/attribution-lib/stubs/ahocorasick.pyi",
            "service/service-app/nvidia_attribution_client/nvidia_attribution_client/__init__.py",
            "service/service-app/nvidia_attribution_client/nvidia_attribution_client/models/__init__.py",
            "service/service-app/openapi/templates/partial_header.mustache",
        )
        diagnostic = (
            "expected one exact Apache SPDX comment after the permitted preamble"
        )
        for relative_path in header_paths:
            with self.subTest(relative_path=relative_path), valid_fixture() as root:
                self.replace_and_stage(
                    root,
                    relative_path,
                    APACHE_IDENTIFIER,
                    f"{IDENTIFIER_TAG} MIT",
                )
                self.assert_violation(root, f"{relative_path}: {diagnostic}")

    def test_rejects_missing_duplicate_and_body_bypass_identifiers(self) -> None:
        relative_path = ".coderabbit.yaml"
        cases = (
            (APACHE_IDENTIFIER, "identifier removed"),
            (APACHE_IDENTIFIER, f"{APACHE_IDENTIFIER}\n# {APACHE_IDENTIFIER}"),
            (
                APACHE_IDENTIFIER,
                f"{IDENTIFIER_TAG} MIT\n# body decoy: {APACHE_IDENTIFIER}",
            ),
        )
        diagnostic = (
            "expected one exact Apache SPDX comment after the permitted preamble"
        )
        for old, new in cases:
            with self.subTest(replacement=new[:20]), valid_fixture() as root:
                self.replace_and_stage(root, relative_path, old, new)
                self.assert_violation(root, f"{relative_path}: {diagnostic}")

    def test_rejects_invalid_duplicate_and_missing_copyright_headers(self) -> None:
        relative_path = ".coderabbit.yaml"
        copyright_line = (
            f"{COPYRIGHT_TAG} Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
            "All rights reserved."
        )
        cases = (
            (copyright_line, "copyright removed"),
            (copyright_line, f"{copyright_line}\n# {copyright_line}"),
            ("Copyright (c) 2026", "Copyright (c) 2027-2026"),
        )
        diagnostic = "expected one exact NVIDIA SPDX copyright comment after the permitted preamble"
        for old, new in cases:
            with self.subTest(replacement=new[:20]), valid_fixture() as root:
                self.replace_and_stage(root, relative_path, old, new)
                self.assert_violation(root, f"{relative_path}: {diagnostic}")

    def test_rejects_an_unclosed_markdown_header(self) -> None:
        relative_path = "service/attribution-lib/docs/adding-a-rule.md"
        with valid_fixture() as root:
            self.replace_and_stage(root, relative_path, "-->\n", "-- >\n")
            self.assert_violation(
                root,
                f"{relative_path}: expected a closed SPDX HTML comment",
            )

    def test_rejects_nul_in_a_required_source(self) -> None:
        relative_path = "blueprint/ruff.toml"
        with valid_fixture() as root:
            self.append_bytes_and_stage(root, relative_path, b"\0")
            self.assert_violation(
                root,
                f"{relative_path}: required source file contains NUL bytes",
            )

    def test_rejects_license_and_license_file_metadata_for_all_pyprojects(self) -> None:
        mutations = (
            (
                'license = "Apache-2.0"',
                'license = "MIT"',
                "project.license must be Apache-2.0",
            ),
            ('license = "Apache-2.0"\n', "", "project.license must be Apache-2.0"),
            (
                'license-files = ["LICENSE"]',
                'license-files = ["NOTICE"]',
                'project.license-files must be ["LICENSE"]',
            ),
            (
                'license-files = ["LICENSE"]\n',
                "",
                'project.license-files must be ["LICENSE"]',
            ),
        )
        for relative_path in PYPROJECT_PATHS:
            for old, new, diagnostic in mutations:
                with (
                    self.subTest(relative_path=relative_path, diagnostic=diagnostic),
                    valid_fixture() as root,
                ):
                    self.replace_and_stage(root, relative_path, old, new)
                    self.assert_violation(root, f"{relative_path}: {diagnostic}")

    def test_rejects_old_build_backends_for_all_pyprojects(self) -> None:
        for relative_path in PYPROJECT_PATHS:
            expected = "hatchling>=1.27"
            replacement = "hatchling>=1.26"
            old_requirement = f'requires = ["{expected}"]'
            new_requirement = old_requirement.replace(expected, replacement)
            with self.subTest(relative_path=relative_path), valid_fixture() as root:
                self.replace_and_stage(
                    root,
                    relative_path,
                    old_requirement,
                    new_requirement,
                )
                self.assert_violation(
                    root,
                    f"{relative_path}: build-system.requires must include {expected}",
                )

    def test_rejects_private_classifiers_for_all_pyprojects(self) -> None:
        classifier = "Private :: Do Not " + "Upload"
        for relative_path in PYPROJECT_PATHS:
            with self.subTest(relative_path=relative_path), valid_fixture() as root:
                self.replace_and_stage(
                    root,
                    relative_path,
                    "[project]\n",
                    f'[project]\nclassifiers = ["{classifier}"]\n',
                )
                self.assert_violation(
                    root,
                    f"{relative_path}: private distribution classifier is forbidden",
                )

    def test_rejects_missing_and_wrong_frontend_license_metadata(self) -> None:
        relative_path = (
            "blueprint/src/attribution_blueprint/webui/frontend/package.json"
        )
        for replacement in ('"license": "MIT",', ""):
            with self.subTest(replacement=replacement), valid_fixture() as root:
                self.replace_and_stage(
                    root,
                    relative_path,
                    '"license": "Apache-2.0",',
                    replacement,
                )
                self.assert_violation(
                    root,
                    f"{relative_path}: license must be Apache-2.0",
                )

        for replacement in ('"private": false,', '"private": "true",', ""):
            with self.subTest(private=replacement), valid_fixture() as root:
                self.replace_and_stage(
                    root,
                    relative_path,
                    '"private": true,',
                    replacement,
                )
                self.assert_violation(
                    root,
                    f"{relative_path}: private must be true",
                )

    def test_rejects_documentation_boundary_mutations(self) -> None:
        cases = (
            (
                "licensed under the Apache License 2.0",
                "licensed under an unspecified license",
                "README.md: missing Apache License 2.0 notice",
            ),
            (
                "CC BY-SA 4.0",
                "a separate license",
                "README.md: missing CC BY-SA 4.0 Code of Conduct boundary",
            ),
            (
                "service/service-app/openapi/templates/LICENSE",
                "an unspecified upstream boundary",
                "README.md: missing vendored OpenAPI Generator license boundary",
            ),
        )
        for old, new, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), valid_fixture() as root:
                self.replace_and_stage(root, "README.md", old, new)
                self.assert_violation(root, diagnostic)

    def test_rejects_generated_code_boundary_mutations(self) -> None:
        exact_tag = "{{! " + APACHE_IDENTIFIER + " }}"
        notice = (
            "OpenAPI Generator does not license generated code; this project applies "
            "Apache-2.0 through partial_header.mustache."
        )
        for relative_path in DERIVED_TEMPLATE_PATHS:
            with (
                self.subTest(relative_path=relative_path, boundary="tag"),
                valid_fixture() as root,
            ):
                self.replace_and_stage(
                    root, relative_path, exact_tag, "{{! license tag removed }}"
                )
                self.assert_violation(
                    root,
                    f"{relative_path}: expected exact Apache-2.0 SPDX template tag",
                )
            with (
                self.subTest(relative_path=relative_path, boundary="copyright"),
                valid_fixture() as root,
            ):
                self.replace_and_stage(
                    root,
                    relative_path,
                    "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.",
                    "Copyright (c) 2026 Example Corp.",
                )
                self.assert_violation(
                    root,
                    f"{relative_path}: expected exact Apache-2.0 SPDX template tag",
                )
            for extra_tag in (IDENTIFIER_TAG + " MIT", COPYRIGHT_TAG + " Example"):
                with (
                    self.subTest(relative_path=relative_path, extra_tag=extra_tag),
                    valid_fixture() as root,
                ):
                    self.replace_and_stage(
                        root,
                        relative_path,
                        exact_tag,
                        f"{exact_tag}\n{{{{! {extra_tag} }}}}",
                    )
                    self.assert_violation(
                        root,
                        f"{relative_path}: expected exact Apache-2.0 SPDX template tag",
                    )
            with (
                self.subTest(relative_path=relative_path, boundary="provenance"),
                valid_fixture() as root,
            ):
                exact_provenance = (
                    "{{! This template is derived from OpenAPI Generator; see "
                    f"./LICENSE. {notice} }}}}"
                )
                self.replace_and_stage(
                    root,
                    relative_path,
                    exact_provenance,
                    "{{! provenance removed }}\n" + notice,
                )
                self.assert_violation(
                    root,
                    f"{relative_path}: missing exact generated-code licensing "
                    "boundary comment",
                )

        partial_api_args = DERIVED_TEMPLATE_PATHS[2]
        with (
            self.subTest(relative_path=partial_api_args, boundary="EOF"),
            valid_fixture() as root,
        ):
            self.append_bytes_and_stage(root, partial_api_args, b"\n")
            self.assert_violation(
                root,
                f"{partial_api_args}: partial template must not end with a newline",
            )

        partial_header = "service/service-app/openapi/templates/partial_header.mustache"
        with self.subTest(relative_path=partial_header), valid_fixture() as root:
            self.replace_and_stage(
                root,
                partial_header,
                APACHE_IDENTIFIER,
                f"{IDENTIFIER_TAG} MIT",
            )
            self.assert_violation(
                root,
                f"{partial_header}: expected Apache-2.0 generated header",
            )

    def test_rejects_generated_client_source_drift(self) -> None:
        cases = (
            (
                "service/service-app/nvidia_attribution_client/LICENSE",
                b"\n",
            ),
            (
                "service/service-app/nvidia_attribution_client/pyproject.toml",
                b"\n# generated drift\n",
            ),
        )
        for relative_path, addition in cases:
            with self.subTest(relative_path=relative_path), valid_fixture() as root:
                self.append_bytes_and_stage(root, relative_path, addition)
                self.assert_violation(
                    root,
                    f"{relative_path}: differs from OpenAPI extras source",
                )

    def test_rejects_commented_delivery_wiring_decoys(self) -> None:
        cases = (
            (
                "blueprint/docker/Dockerfile",
                "COPY LICENSE /licenses/LICENSE",
                "# COPY LICENSE /licenses/LICENSE",
                "final image must contain /licenses/LICENSE",
            ),
            (
                "blueprint/docker/Dockerfile",
                'LABEL com.nvidia.attribution.project.license="Apache-2.0"',
                '# LABEL com.nvidia.attribution.project.license="Apache-2.0"',
                "final image must declare the Apache-2.0 project license label",
            ),
            (
                "service/service-app/Dockerfile",
                "COPY LICENSE /licenses/LICENSE",
                "# COPY LICENSE /licenses/LICENSE",
                "final image must contain /licenses/LICENSE",
            ),
            (
                "service/service-app/Dockerfile",
                'LABEL com.nvidia.attribution.project.license="Apache-2.0"',
                '# LABEL com.nvidia.attribution.project.license="Apache-2.0"',
                "final image must declare the Apache-2.0 project license label",
            ),
            (
                "blueprint/docker/Dockerfile.dockerignore",
                "!blueprint/LICENSE",
                "# !blueprint/LICENSE",
                "must include the Blueprint package license",
            ),
            (
                "blueprint/docker/Dockerfile.dockerignore",
                "!LICENSE",
                "# !LICENSE",
                "must include the root license",
            ),
            (
                "service/service-app/Dockerfile.dockerignore",
                "!LICENSE",
                "# !LICENSE",
                "must include the root license",
            ),
            (
                ".ci/run-tests.sh",
                "    license-compliance)\n        run_license_compliance\n        ;;",
                "    # license-compliance)\n        # run_license_compliance\n        # ;;",
                "missing active license-compliance test dispatcher",
            ),
            (
                ".github/workflows/tests.yml",
                "          - license-compliance",
                "          # - license-compliance",
                "missing license-compliance CI suite",
            ),
            (
                ".github/workflows/tests.yml",
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "        # run: bash .ci/build-and-check-license-artifacts.sh",
                "missing published-artifact license validation",
            ),
            (
                ".ci/build-and-check-license-artifacts.sh",
                '    --artifacts "${artifacts}"',
                '    # --artifacts "${artifacts}"',
                "artifact build must run archive license validation",
            ),
            (
                ".ci/build-and-check-license-artifacts.sh",
                '    echo "source tree changed during artifact build" >&2',
                '    # echo "source tree changed during artifact build" >&2',
                "missing clean-source-tree invariant",
            ),
            (
                ".ci/build-and-check-license-artifacts.sh",
                'uv run --python 3.14.6 --no-project python "${source_root}/'
                '.ci/check-license-compliance.py" ' + "\\",
                '# uv run --python 3.14.6 --no-project python "${source_root}/'
                '.ci/check-license-compliance.py" ' + "\\",
                "checker must use the indexed managed Python runtime",
            ),
            (
                ".ci/build-and-check-license-artifacts.sh",
                'uv run --python 3.14.6 --no-project python "${source_root}/'
                '.ci/check-license-compliance.py" ' + "\\",
                "uv run --python 3.14.6 --no-project python "
                ".ci/check-license-compliance.py \\",
                "checker must use the indexed managed Python runtime",
            ),
            (
                ".ci/build-and-check-license-artifacts.sh",
                '    --root "${source_root}" \\',
                "    --root . \\",
                "checker must validate the indexed source root",
            ),
            (
                ".pre-commit-config.yaml",
                "      - id: license-compliance",
                "      # - id: license-compliance",
                "missing license compliance pre-commit hook",
            ),
            (
                ".pre-commit-config.yaml",
                "        entry: uv run --python 3.14.6 --no-project python .ci/check-license-compliance.py --root .",
                "        # entry: uv run --python 3.14.6 --no-project python .ci/check-license-compliance.py --root .",
                "checker must use the managed Python runtime",
            ),
            (
                ".ci/run-end-to-end-tests.sh",
                'assert_image_license "${service_image}"',
                '# assert_image_license "${service_image}"',
                "missing runtime license validation for service_image",
            ),
            (
                ".ci/run-end-to-end-tests.sh",
                'assert_image_license "${blueprint_image}"',
                '# assert_image_license "${blueprint_image}"',
                "missing runtime license validation for blueprint_image",
            ),
            (
                ".ci/run-end-to-end-tests.sh",
                "        docker image inspect " + "\\",
                "        # docker image inspect " + "\\",
                "runtime check must inspect the final image label",
            ),
            (
                ".ci/run-end-to-end-tests.sh",
                '        --volume "${source_root}/LICENSE:/expected/LICENSE:ro" '
                + "\\",
                '        # --volume "${source_root}/LICENSE:/expected/LICENSE:ro" '
                + "\\",
                "runtime check must mount the canonical license",
            ),
            (
                ".ci/run-end-to-end-tests.sh",
                '        -c \'from pathlib import Path; raise SystemExit(Path("/licenses/LICENSE").read_bytes() != Path("/expected/LICENSE").read_bytes())\'',
                '        # -c \'from pathlib import Path; raise SystemExit(Path("/licenses/LICENSE").read_bytes() != Path("/expected/LICENSE").read_bytes())\'',
                "runtime check must compare image and canonical license bytes",
            ),
        )
        for relative_path, old, new, diagnostic in cases:
            with (
                self.subTest(relative_path=relative_path, diagnostic=diagnostic),
                valid_fixture() as root,
            ):
                self.replace_and_stage(root, relative_path, old, new)
                self.assert_violation(root, f"{relative_path}: {diagnostic}")

    def test_rejects_unreachable_delivery_wiring(self) -> None:
        cases = (
            (
                ".github/workflows/tests.yml",
                "  python-tests:",
                "x-license-decoy:\n  python-tests:",
                "missing license-compliance CI suite",
            ),
            (
                ".github/workflows/tests.yml",
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "x-artifact-decoy:\n  run: bash "
                ".ci/build-and-check-license-artifacts.sh",
                "missing published-artifact license validation",
            ),
            (
                ".ci/run-tests.sh",
                'case "${suite}" in',
                'exit 0\ncase "${suite}" in',
                "unconditional successful termination can bypass license checks",
            ),
            (
                ".ci/build-and-check-license-artifacts.sh",
                "uv run --python 3.14.6 --no-project python ",
                "exit 0\nuv run --python 3.14.6 --no-project python ",
                "unconditional successful termination can bypass license checks",
            ),
            (
                ".ci/run-end-to-end-tests.sh",
                'assert_image_license "${service_image}"',
                'exit 0\nassert_image_license "${service_image}"',
                "unconditional successful termination can bypass license checks",
            ),
        )
        for relative_path, old, new, diagnostic in cases:
            with (
                self.subTest(relative_path=relative_path, diagnostic=diagnostic),
                valid_fixture() as root,
            ):
                self.replace_and_stage(root, relative_path, old, new)
                self.assert_violation(root, f"{relative_path}: {diagnostic}")

        workflow_path = ".github/workflows/tests.yml"
        with valid_fixture() as root:
            self.replace_and_stage(
                root, workflow_path, "          - license-compliance\n", ""
            )
            self.replace_and_stage(
                root,
                workflow_path,
                '    env:\n      COVERAGE: "true"',
                '    env:\n      COVERAGE: "true"\n      LICENSE_DECOY: |\n'
                "        - license-compliance",
            )
            self.assert_violation(
                root, f"{workflow_path}: missing license-compliance CI suite"
            )

        with valid_fixture() as root:
            self.replace_and_stage(
                root,
                workflow_path,
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "        run: echo skipped",
            )
            self.replace_and_stage(
                root,
                workflow_path,
                "  frontend-tests:\n    name: frontend",
                "  frontend-tests:\n    name: frontend\n    env:\n"
                "      ARTIFACT_DECOY: |\n        run: bash "
                ".ci/build-and-check-license-artifacts.sh",
            )
            self.assert_violation(
                root,
                f"{workflow_path}: missing published-artifact license validation",
            )

        workflow_cases = (
            (
                "\njobs:\n",
                "\ndefaults:\n  run:\n    shell: echo {0}\n\njobs:\n",
                "missing active matrix test dispatcher",
            ),
            (
                "    if: ${{ always() }}",
                "    if: false",
                "python-tests job must remain executable",
            ),
            (
                "    if: ${{ always() }}",
                "    if: ${{ false && always() }}",
                "python-tests job must remain executable",
            ),
            (
                "    if: ${{ always() }}\n    runs-on:",
                "    runs-on:",
                "python-tests job must remain executable",
            ),
            (
                "  python-tests:\n    name: ${{ matrix.suite }}",
                "  python-tests:\n    name: ${{ matrix.suite }}\n"
                "    continue-on-error: true",
                "python-tests job must fail when license compliance fails",
            ),
            (
                '        run: bash .ci/run-tests.sh "${{ matrix.suite }}"',
                "        run: echo skipped",
                "missing active matrix test dispatcher",
            ),
            (
                "      - name: Run ${{ matrix.suite }} with coverage\n"
                '        run: bash .ci/run-tests.sh "${{ matrix.suite }}"',
                "      - name: Run ${{ matrix.suite }} with coverage\n"
                "        shell: echo {0}\n"
                '        run: bash .ci/run-tests.sh "${{ matrix.suite }}"',
                "missing active matrix test dispatcher",
            ),
            (
                "  python-tests:\n    name: ${{ matrix.suite }}",
                "  python-tests:\n    name: ${{ matrix.suite }}\n"
                "    defaults:\n      run:\n        shell: echo {0}",
                "missing active matrix test dispatcher",
            ),
            (
                "      - name: Run ${{ matrix.suite }} with coverage\n"
                '        run: bash .ci/run-tests.sh "${{ matrix.suite }}"',
                "      - name: Run ${{ matrix.suite }} with coverage\n"
                "        if: ${{ false && always() }}\n"
                '        run: bash .ci/run-tests.sh "${{ matrix.suite }}"',
                "missing active matrix test dispatcher",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n"
                "          - suite: license-compliance\n",
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n"
                "        exclude: [{suite: license-compliance}]\n",
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n"
                "          - {suite: license-compliance}\n",
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n"
                '          - "suite": "license-compliance"\n',
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n"
                "          - 'suite': 'license-compliance'\n",
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n"
                '        exclude: [{"suite": "license-compliance"}]\n',
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n"
                "          - suite: &blocked license-compliance\n"
                "          - suite: *blocked\n",
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n"
                "          - &blocked {suite: license-compliance}\n"
                "          - *blocked\n",
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n"
                "        blocked:\n"
                "          - &blocked license-compliance\n"
                "        exclude:\n"
                "          - suite: *blocked\n",
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n"
                "        include:\n"
                "          - suite: &blocked license-compliance\n"
                "        exclude:\n"
                "          - suite: *blocked\n",
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n"
                '          - suite: "license-\\u0063ompliance"\n',
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n"
                "          - suite: >-\n"
                "              license-compliance\n",
                "license-compliance suite must not be excluded",
            ),
            (
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n"
                '          - suite: "license-' + "\\" + '\n            compliance"\n',
                "license-compliance suite must not be excluded",
            ),
            (
                "      - name: Build and validate published artifacts\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "      - name: Build and validate published artifacts\n"
                "        if: false\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "missing published-artifact license validation",
            ),
            (
                "      - name: Build and validate published artifacts\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "      - name: Build and validate published artifacts\n"
                "        continue-on-error: true\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "missing published-artifact license validation",
            ),
            (
                "      - name: Build and validate published artifacts\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "      - name: Build and validate published artifacts\n"
                "        shell: echo {0}\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "missing published-artifact license validation",
            ),
            (
                "  frontend-tests:\n    name: frontend\n    needs: secret-scan\n"
                "    if: ${{ always() }}",
                "  frontend-tests:\n    name: frontend\n    needs: secret-scan\n"
                "    if: false",
                "frontend-tests job must remain executable",
            ),
            (
                "  frontend-tests:\n    name: frontend",
                "  frontend-tests:\n    name: frontend\n    continue-on-error: true",
                "frontend-tests job must fail when artifact validation fails",
            ),
            (
                "  frontend-tests:\n    name: frontend",
                "  frontend-tests:\n    name: frontend\n"
                "    defaults:\n      run:\n        shell: echo {0}",
                "missing published-artifact license validation",
            ),
            (
                "      - name: Build and validate published artifacts\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "      - name: Build and validate published artifacts\n"
                "        if: ${{ false && always() }}\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "missing published-artifact license validation",
            ),
            (
                "  end-to-end:\n    name: end-to-end\n    needs: secret-scan\n"
                "    if: ${{ always() }}",
                "  end-to-end:\n    name: end-to-end\n    needs: secret-scan\n"
                "    if: false",
                "end-to-end job must remain executable",
            ),
            (
                "  end-to-end:\n    name: end-to-end\n    needs: secret-scan\n"
                "    if: ${{ always() }}",
                "  end-to-end:\n    name: end-to-end\n    needs: secret-scan\n"
                "    if: ${{ false && always() }}",
                "end-to-end job must remain executable",
            ),
            (
                "  end-to-end:\n    name: end-to-end\n    needs: secret-scan\n"
                "    if: ${{ always() }}\n    runs-on:",
                "  end-to-end:\n    name: end-to-end\n    needs: secret-scan\n"
                "    runs-on:",
                "end-to-end job must remain executable",
            ),
            (
                "  end-to-end:\n    name: end-to-end",
                "  end-to-end:\n    name: end-to-end\n    continue-on-error: true",
                "end-to-end job must fail when license validation fails",
            ),
            (
                "  end-to-end:\n    name: end-to-end",
                "  end-to-end:\n    name: end-to-end\n"
                "    defaults:\n      run:\n        shell: echo {0}",
                "missing active end-to-end license validation",
            ),
            (
                "      - name: Run end-to-end tests\n"
                "        run: bash .ci/run-tests.sh end-to-end",
                "      - name: Run end-to-end tests\n"
                "        if: false\n"
                "        run: bash .ci/run-tests.sh end-to-end",
                "missing active end-to-end license validation",
            ),
            (
                "        run: bash .ci/run-tests.sh end-to-end",
                "        run: echo skipped",
                "missing active end-to-end license validation",
            ),
            (
                "      - name: Run end-to-end tests\n"
                "        run: bash .ci/run-tests.sh end-to-end",
                "      - name: Run end-to-end tests\n"
                "        if: ${{ false && always() }}\n"
                "        run: bash .ci/run-tests.sh end-to-end",
                "missing active end-to-end license validation",
            ),
            (
                "      - name: Run end-to-end tests\n"
                "        run: bash .ci/run-tests.sh end-to-end",
                "      - name: Run end-to-end tests\n"
                "        continue-on-error: true\n"
                "        run: bash .ci/run-tests.sh end-to-end",
                "missing active end-to-end license validation",
            ),
            (
                "      - name: Run end-to-end tests\n"
                "        run: bash .ci/run-tests.sh end-to-end",
                "      - name: Run end-to-end tests\n"
                "        shell: echo {0}\n"
                "        run: bash .ci/run-tests.sh end-to-end",
                "missing active end-to-end license validation",
            ),
        )
        for old, new, diagnostic in workflow_cases:
            with self.subTest(diagnostic=diagnostic), valid_fixture() as root:
                self.replace_and_stage(root, workflow_path, old, new)
                self.assert_violation(root, f"{workflow_path}: {diagnostic}")

        with valid_fixture() as root:
            self.replace_and_stage(
                root,
                workflow_path,
                "          - blueprint\n",
                "          - blueprint\n        exclude:\n          - suite: blueprint\n",
            )
            self.assert_compliant(root)

        with valid_fixture() as root:
            self.replace_and_stage(
                root,
                workflow_path,
                "          - blueprint\n",
                "          - blueprint\n        exclude: [{suite: blueprint}]\n",
            )
            self.assert_compliant(root)

        for harmless_exclusion in (
            '        exclude:\n          - "suite": "blueprint"\n',
            '        exclude: [{"suite": "blueprint"}]\n',
        ):
            with (
                self.subTest(harmless_exclusion=harmless_exclusion),
                valid_fixture() as root,
            ):
                self.replace_and_stage(
                    root,
                    workflow_path,
                    "          - blueprint\n",
                    "          - blueprint\n" + harmless_exclusion,
                )
                self.assert_compliant(root)

        allowed_shell_cases = (
            (
                "\njobs:\n",
                "\ndefaults:\n  run:\n    shell: bash {0}\n\njobs:\n",
            ),
            (
                "  python-tests:\n    name: ${{ matrix.suite }}",
                "  python-tests:\n    name: ${{ matrix.suite }}\n"
                "    defaults:\n      run:\n        shell: bash {0}",
            ),
            (
                "      - name: Run ${{ matrix.suite }} with coverage\n"
                '        run: bash .ci/run-tests.sh "${{ matrix.suite }}"',
                "      - name: Run ${{ matrix.suite }} with coverage\n"
                "        shell: bash {0}\n"
                '        run: bash .ci/run-tests.sh "${{ matrix.suite }}"',
            ),
            (
                "  frontend-tests:\n    name: frontend",
                "  frontend-tests:\n    name: frontend\n"
                "    defaults:\n      run:\n        shell: bash {0}",
            ),
            (
                "      - name: Build and validate published artifacts\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
                "      - name: Build and validate published artifacts\n"
                "        shell: bash {0}\n"
                "        run: bash .ci/build-and-check-license-artifacts.sh",
            ),
            (
                "  end-to-end:\n    name: end-to-end",
                "  end-to-end:\n    name: end-to-end\n"
                "    defaults:\n      run:\n        shell: bash {0}",
            ),
            (
                "      - name: Run end-to-end tests\n"
                "        run: bash .ci/run-tests.sh end-to-end",
                "      - name: Run end-to-end tests\n"
                "        shell: bash {0}\n"
                "        run: bash .ci/run-tests.sh end-to-end",
            ),
        )
        for old, new in allowed_shell_cases:
            with self.subTest(allowed_shell=new), valid_fixture() as root:
                self.replace_and_stage(root, workflow_path, old, new)
                self.assert_compliant(root)

    def test_rejects_missing_docker_context_exclusions(self) -> None:
        common = (
            "**/.venv",
            "**/__pycache__",
            "**/.pytest_cache",
            "**/.ruff_cache",
            "**/.mypy_cache",
            "**/.cache",
            "**/build",
            "**/dist",
            "**/*.egg-info",
            "**/*.py[co]",
            "**/.env",
            "**/.env.*",
            "**/*secret*",
            "**/*credential*",
        )
        exclusions = {
            "blueprint/docker/Dockerfile.dockerignore": (
                "service/attribution-lib/build",
                "service/attribution-lib/src/*.egg-info",
                "blueprint/src/*.egg-info",
                *common,
            ),
            "service/service-app/Dockerfile.dockerignore": (
                "service/attribution-lib/build",
                "service/attribution-lib/src/*.egg-info",
                *common,
            ),
        }
        for relative_path, patterns in exclusions.items():
            for pattern in patterns:
                with (
                    self.subTest(relative_path=relative_path, pattern=pattern),
                    valid_fixture() as root,
                ):
                    self.replace_and_stage(root, relative_path, f"{pattern}\n", "")
                    self.assert_violation(
                        root,
                        f"{relative_path}: must exclude local or generated content: {pattern}",
                    )

            with (
                self.subTest(relative_path=relative_path, pattern="late include"),
                valid_fixture() as root,
            ):
                self.append_bytes_and_stage(root, relative_path, b"!**/.env\n")
                self.assert_violation(
                    root,
                    f"{relative_path}: exclusion must follow all include rules",
                )

    def test_rejects_invalid_archive_core_metadata(self) -> None:
        private_classifier = "Private :: Do Not " + "Upload"
        proprietary_identifier = "LicenseRef-" + "NvidiaProprietary"
        cases = (
            (
                metadata_bytes("attribution-lib", metadata_version="2.3"),
                "Metadata-Version must be at least 2.4",
            ),
            (
                metadata_bytes("attribution-lib", license_expression="MIT"),
                "License-Expression must be Apache-2.0",
            ),
            (
                metadata_bytes("attribution-lib", license_files=()),
                "License-File must be exactly LICENSE",
            ),
            (
                metadata_bytes(
                    "attribution-lib",
                    extra_fields=(("License", "legacy"),),
                ),
                "legacy License metadata field is forbidden",
            ),
            (
                metadata_bytes(
                    "attribution-lib",
                    extra_fields=(("Classifier", private_classifier),),
                ),
                "proprietary license relic",
            ),
            (
                metadata_bytes(
                    "attribution-lib",
                    extra_fields=(("License", proprietary_identifier),),
                ),
                "proprietary license relic",
            ),
            (
                metadata_bytes(
                    "attribution-lib",
                    extra_fields=(("Name", "duplicate-name"),),
                ),
                "metadata must contain one Name field",
            ),
        )
        for kind in ("wheel", "sdist"):
            for metadata, diagnostic in cases:
                with (
                    self.subTest(kind=kind, diagnostic=diagnostic),
                    valid_fixture() as root,
                ):
                    artifacts = self.overwrite_artifact(
                        root,
                        "attribution-lib",
                        kind,
                        metadata=metadata,
                    )
                    self.assert_violation(root, diagnostic, artifacts)

    def test_rejects_wheel_license_identity_and_frontend_mutations(self) -> None:
        cases = (
            (
                {"license_bytes": b"changed"},
                "wheel must contain the canonical dist-info LICENSE",
            ),
            (
                {"include_license": False},
                "wheel must contain the canonical dist-info LICENSE",
            ),
            (
                {
                    "license_member": f"attribution_lib-{EXPECTED_VERSIONS['attribution-lib']}.dist-info/LICENSE"
                },
                "wheel must contain the canonical dist-info LICENSE",
            ),
            (
                {
                    "dist_info": f"wrong-{EXPECTED_VERSIONS['attribution-lib']}.dist-info"
                },
                "wheel METADATA is outside its canonical dist-info root",
            ),
            (
                {"filename_name": "wrong-name"},
                "wheel filename does not match METADATA identity",
            ),
            (
                {"include_metadata": False},
                "wheel must contain one canonical METADATA file",
            ),
        )
        for options, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), valid_fixture() as root:
                artifacts = self.overwrite_artifact(
                    root,
                    "attribution-lib",
                    "wheel",
                    **options,
                )
                self.assert_violation(root, diagnostic, artifacts)

        with self.subTest(frontend="missing index"), valid_fixture() as root:
            artifacts = self.overwrite_artifact(
                root,
                "nvidia-attribution-blueprint",
                "wheel",
                include_static=False,
            )
            self.assert_violation(
                root,
                "Blueprint archive must contain the built frontend index",
                artifacts,
            )

        root_name = f"nvidia_attribution_blueprint-{EXPECTED_VERSIONS['nvidia-attribution-blueprint']}"
        static_root = f"{root_name}/src/attribution_blueprint/webui/static"
        index = (
            b'<link rel="stylesheet" href="/assets/app.css">'
            b'<script src="/assets/app.js"></script>'
        )
        for missing_suffix, present_name, present_data in (
            (".js", "app.css", b"body{}"),
            (".css", "app.js", b"export{}"),
        ):
            with (
                self.subTest(frontend=f"missing {missing_suffix}"),
                valid_fixture() as root,
            ):
                artifacts = self.overwrite_artifact(
                    root,
                    "nvidia-attribution-blueprint",
                    "sdist",
                    include_static=False,
                    extra_members=(
                        (f"{static_root}/index.html", index, None),
                        (f"{static_root}/assets/{present_name}", present_data, None),
                    ),
                )
                self.assert_violation(
                    root,
                    f"Blueprint frontend asset is missing: {static_root}/assets/app{missing_suffix}",
                    artifacts,
                )

        index = (
            b'<link rel="stylesheet" href="/assets/missing.css">'
            b'<script src="/assets/missing.js"></script>'
        )
        with self.subTest(frontend="missing assets"), valid_fixture() as root:
            artifacts = self.overwrite_artifact(
                root,
                "nvidia-attribution-blueprint",
                "wheel",
                include_static=False,
                extra_members=(
                    ("attribution_blueprint/webui/static/index.html", index, None),
                ),
            )
            self.assert_violation(
                root, "Blueprint frontend asset is missing", artifacts
            )

    def test_rejects_sdist_license_identity_and_frontend_mutations(self) -> None:
        cases = (
            (
                {"license_bytes": b"changed"},
                "sdist must contain the canonical root LICENSE",
            ),
            ({"include_license": False}, "sdist must contain one root LICENSE"),
            (
                {
                    "license_member": f"attribution_lib-{EXPECTED_VERSIONS['attribution-lib']}/legal/LICENSE"
                },
                "sdist must contain one root LICENSE",
            ),
            (
                {"root_name": f"wrong-{EXPECTED_VERSIONS['attribution-lib']}"},
                "sdist files are outside the canonical package root",
            ),
            (
                {"filename_name": "wrong-name"},
                "sdist filename does not match PKG-INFO identity",
            ),
            ({"include_metadata": False}, "sdist must contain one root PKG-INFO file"),
        )
        for options, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), valid_fixture() as root:
                artifacts = self.overwrite_artifact(
                    root,
                    "attribution-lib",
                    "sdist",
                    **options,
                )
                self.assert_violation(root, diagnostic, artifacts)

        with self.subTest(frontend="missing index"), valid_fixture() as root:
            artifacts = self.overwrite_artifact(
                root,
                "nvidia-attribution-blueprint",
                "sdist",
                include_static=False,
            )
            self.assert_violation(
                root,
                "Blueprint archive must contain the built frontend index",
                artifacts,
            )

    def test_rejects_duplicate_unsafe_and_nonregular_wheel_members(self) -> None:
        distribution = "attribution-lib"
        metadata_path = (
            f"attribution_lib-{EXPECTED_VERSIONS['attribution-lib']}.dist-info/METADATA"
        )
        cases = (
            (
                ((metadata_path, metadata_bytes(distribution), None),),
                "wheel contains duplicate members",
            ),
            (
                (("linked", b"target", stat.S_IFLNK | 0o777),),
                "wheel contains a non-regular member",
            ),
        )
        for members, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), valid_fixture() as root:
                artifacts = self.overwrite_artifact(
                    root,
                    distribution,
                    "wheel",
                    extra_members=members,
                )
                self.assert_violation(root, diagnostic, artifacts)

        for unsafe_path in (
            "../escape",
            "/absolute",
            "C:/escape",
            "dir\\escape",
            "a//b",
        ):
            with self.subTest(unsafe_path=unsafe_path), valid_fixture() as root:
                artifacts = self.overwrite_artifact(
                    root,
                    distribution,
                    "wheel",
                    extra_members=((unsafe_path, b"unsafe", None),),
                )
                self.assert_violation(
                    root, "wheel contains an unsafe member path", artifacts
                )

    def test_rejects_duplicate_unsafe_and_nonregular_sdist_members(self) -> None:
        distribution = "attribution-lib"
        metadata_path = (
            f"attribution_lib-{EXPECTED_VERSIONS['attribution-lib']}/PKG-INFO"
        )
        cases = (
            (
                ((metadata_path, metadata_bytes(distribution), None),),
                "sdist contains duplicate members",
            ),
            (
                (
                    (
                        f"attribution_lib-{EXPECTED_VERSIONS['attribution-lib']}/linked",
                        b"target",
                        tarfile.SYMTYPE,
                    ),
                ),
                "sdist contains a non-regular member",
            ),
            (
                (("another-root/file", b"extra", None),),
                "sdist must contain one top-level root",
            ),
        )
        for members, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), valid_fixture() as root:
                artifacts = self.overwrite_artifact(
                    root,
                    distribution,
                    "sdist",
                    extra_members=members,
                )
                self.assert_violation(root, diagnostic, artifacts)

        for unsafe_path in (
            "../escape",
            "/absolute",
            "C:/escape",
            "dir\\escape",
            "a//b",
        ):
            with self.subTest(unsafe_path=unsafe_path), valid_fixture() as root:
                artifacts = self.overwrite_artifact(
                    root,
                    distribution,
                    "sdist",
                    extra_members=((unsafe_path, b"unsafe", None),),
                )
                self.assert_violation(
                    root, "sdist contains an unsafe member path", artifacts
                )

    def test_rejects_oversized_archive_metadata(self) -> None:
        oversized = metadata_bytes("attribution-lib") + b"X" * 1_048_577
        for kind, diagnostic in (
            ("wheel", "wheel METADATA is too large"),
            ("sdist", "sdist PKG-INFO is too large"),
        ):
            with self.subTest(kind=kind), valid_fixture() as root:
                artifacts = self.overwrite_artifact(
                    root,
                    "attribution-lib",
                    kind,
                    metadata=oversized,
                )
                self.assert_violation(root, diagnostic, artifacts)

    def test_rejects_incomplete_artifact_sets_and_version_drift(self) -> None:
        with self.subTest(case="missing artifact"), valid_fixture() as root:
            artifacts = make_artifact_set(root)
            version = EXPECTED_VERSIONS["attribution-lib"]
            (artifacts / f"attribution_lib-{version}-py3-none-any.whl").unlink()
            self.assert_violation(
                root,
                "expected four wheels and four source distributions",
                artifacts,
            )

        with self.subTest(case="version drift"), valid_fixture() as root:
            artifacts = make_artifact_set(root)
            version = EXPECTED_VERSIONS["attribution-lib"]
            (artifacts / f"attribution_lib-{version}.tar.gz").unlink()
            write_sdist(
                artifacts,
                (root / "LICENSE").read_bytes(),
                "attribution-lib",
                version="2.0.0",
            )
            self.assert_violation(
                root,
                "wheel and sdist versions differ for attribution-lib",
                artifacts,
            )

    def test_rejects_proprietary_relics_in_archive_payloads(self) -> None:
        relic = ("LicenseRef-" + "NvidiaProprietary").encode()
        version = EXPECTED_VERSIONS["attribution-lib"]
        cases = (
            ("wheel", "attribution_lib/private.txt"),
            ("sdist", f"attribution_lib-{version}/private.txt"),
        )
        for kind, member_name in cases:
            with self.subTest(kind=kind), valid_fixture() as root:
                artifacts = self.overwrite_artifact(
                    root,
                    "attribution-lib",
                    kind,
                    extra_members=((member_name, relic, None),),
                )
                self.assert_violation(
                    root,
                    f"proprietary license relic in archive member {member_name}",
                    artifacts,
                )

    def test_accepts_git_stamped_component_artifacts(self) -> None:
        with valid_fixture() as root:
            artifacts = make_artifact_set(root)
            canonical = (root / "LICENSE").read_bytes()
            for distribution in EXPECTED_DISTRIBUTIONS:
                if distribution == "nvidia-attribution-client":
                    continue
                name = normalized_name(distribution)
                version = EXPECTED_VERSIONS[distribution]
                (artifacts / f"{name}-{version}-py3-none-any.whl").unlink()
                (artifacts / f"{name}-{version}.tar.gz").unlink()
                stamped = f"{version}+42.dirty"
                write_wheel(artifacts, canonical, distribution, version=stamped)
                write_sdist(artifacts, canonical, distribution, version=stamped)
            result = run_checker(root, artifacts)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_source_project_name_and_version_drift(self) -> None:
        name_cases = (
            (
                ("blueprint/pyproject.toml",),
                'name = "nvidia-attribution-blueprint"',
                'name = "wrong-blueprint"',
                (
                    "blueprint/pyproject.toml: project.name must be "
                    "nvidia-attribution-blueprint",
                ),
            ),
            (
                ("service/attribution-lib/pyproject.toml",),
                'name = "attribution-lib"',
                'name = "wrong-attribution-lib"',
                (
                    "service/attribution-lib/pyproject.toml: project.name must be "
                    "attribution-lib",
                ),
            ),
            (
                ("service/service-app/pyproject.toml",),
                'name = "nvidia-attribution-service"',
                'name = "wrong-attribution-service"',
                (
                    "service/service-app/pyproject.toml: project.name must be "
                    "nvidia-attribution-service",
                ),
            ),
            (
                (
                    "service/service-app/openapi/extras/pyproject.toml",
                    "service/service-app/nvidia_attribution_client/pyproject.toml",
                ),
                'name = "nvidia-attribution-client"',
                'name = "wrong-attribution-client"',
                (
                    "service/service-app/openapi/extras/pyproject.toml: "
                    "project.name must be nvidia-attribution-client",
                    "service/service-app/nvidia_attribution_client/pyproject.toml: "
                    "project.name must be nvidia-attribution-client",
                ),
            ),
        )
        for paths, old, new, diagnostics in name_cases:
            with self.subTest(paths=paths), valid_fixture() as root:
                for relative_path in paths:
                    self.replace_and_stage(root, relative_path, old, new)
                artifacts = make_artifact_set(root)
                result = run_checker(root, artifacts)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                for diagnostic in diagnostics:
                    self.assertIn(diagnostic, result.stderr)

        version_cases = (
            (
                relative_path,
                f'__version__ = "{EXPECTED_VERSIONS[distribution]}"',
                '__version__ = "9.9.9"',
                distribution,
            )
            for distribution, relative_path in SOURCE_VERSION_PATHS.items()
        )
        for relative_path, old, new, distribution in version_cases:
            with self.subTest(distribution=distribution), valid_fixture() as root:
                self.replace_and_stage(root, relative_path, old, new)
                artifacts = make_artifact_set(root)
                result = run_checker(root, artifacts)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                for artifact_kind in ("wheel", "sdist"):
                    self.assertIn(
                        f"{artifact_kind} version must match source version for "
                        f"{distribution}",
                        result.stderr,
                    )

    def test_rejects_missing_and_shadowed_blueprint_frontend_assets(self) -> None:
        blueprint_version = EXPECTED_VERSIONS["nvidia-attribution-blueprint"]
        roots = {
            "wheel": "attribution_blueprint/webui/static",
            "sdist": (
                f"nvidia_attribution_blueprint-{blueprint_version}/src/"
                "attribution_blueprint/webui/static"
            ),
        }
        complete_index = (
            b'<link rel="icon" href="/logo.svg">'
            b'<link rel="stylesheet" href="/assets/app.css">'
            b'<script src="/assets/app.js"></script>'
        )
        repeated_index = (
            b'<link rel="icon" href="/logo.svg">'
            b'<link rel="stylesheet" href="/assets/missing-first.css">'
            b'<link rel="stylesheet" href="/assets/app.css">'
            b'<script src="/assets/missing-first.js"></script>'
            b'<script src="/assets/app.js"></script>'
        )
        for kind, static_root in roots.items():
            valid_assets = (
                (f"{static_root}/assets/app.css", b"body{}", None),
                (f"{static_root}/assets/app.js", b"export{}", None),
            )
            with self.subTest(kind=kind, case="missing logo"), valid_fixture() as root:
                artifacts = self.overwrite_artifact(
                    root,
                    "nvidia-attribution-blueprint",
                    kind,
                    include_static=False,
                    extra_members=(
                        (f"{static_root}/index.html", complete_index, None),
                        *valid_assets,
                    ),
                )
                self.assert_violation(
                    root,
                    f"Blueprint frontend asset is missing: {static_root}/logo.svg",
                    artifacts,
                )

            with (
                self.subTest(kind=kind, case="missing third-party notices"),
                valid_fixture() as root,
            ):
                artifacts = self.overwrite_artifact(
                    root,
                    "nvidia-attribution-blueprint",
                    kind,
                    include_static=False,
                    extra_members=(
                        (f"{static_root}/index.html", complete_index, None),
                        (f"{static_root}/logo.svg", b"<svg></svg>", None),
                        *valid_assets,
                    ),
                )
                self.assert_violation(
                    root,
                    "Blueprint frontend asset is missing: "
                    f"{static_root}/assets/THIRD_PARTY_NOTICES.txt",
                    artifacts,
                )

            with (
                self.subTest(kind=kind, case="shadowed missing references"),
                valid_fixture() as root,
            ):
                artifacts = self.overwrite_artifact(
                    root,
                    "nvidia-attribution-blueprint",
                    kind,
                    include_static=False,
                    extra_members=(
                        (f"{static_root}/index.html", repeated_index, None),
                        (f"{static_root}/logo.svg", b"<svg></svg>", None),
                        *valid_assets,
                    ),
                )
                result = run_checker(root, artifacts)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                for missing_name in ("missing-first.css", "missing-first.js"):
                    self.assertIn(
                        "Blueprint frontend asset is missing: "
                        f"{static_root}/assets/{missing_name}",
                        result.stderr,
                    )

    def test_client_drift_uses_only_the_git_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            drift_script = root / ".ci/check-client-drift.sh"
            generator = root / "service/service-app/openapi/generate-client.sh"
            source_value = root / "service/service-app/openapi/templates/value.txt"
            spec = root / "service/service-app/openapi.json"
            client = (
                root / "service/service-app/nvidia_attribution_client/generated.txt"
            )
            for path in (drift_script, generator, source_value, spec, client):
                path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY_ROOT / ".ci/check-client-drift.sh", drift_script)
            generator.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'output="${1:?}"\n'
                'value="$(< openapi/templates/value.txt)"\n'
                'mkdir -p "${output}/nvidia_attribution_client"\n'
                'printf \'%s\\n\' "${value}" > "${output}/openapi.json"\n'
                "printf '%s\\n' \"${value}\" > "
                '"${output}/nvidia_attribution_client/generated.txt"\n',
                encoding="utf-8",
            )
            generator.chmod(0o755)
            source_value.write_text("A\n", encoding="utf-8")
            spec.write_text("A\n", encoding="utf-8")
            client.write_text("A\n", encoding="utf-8")
            self.git(root, "init", "--quiet")
            self.git(root, "add", ".")

            source_value.write_text("B\n", encoding="utf-8")
            clean_index = subprocess.run(
                ["bash", str(drift_script)],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(clean_index.returncode, 0, clean_index.stderr)

            spec.write_text("B\n", encoding="utf-8")
            client.write_text("B\n", encoding="utf-8")
            self.git(
                root, "add", str(spec.relative_to(root)), str(client.relative_to(root))
            )
            mismatched_index = subprocess.run(
                ["bash", str(drift_script)],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(mismatched_index.returncode, 0)
            self.assertIn(
                "the committed spec and client do not match a fresh regeneration",
                mismatched_index.stderr,
            )

    def test_reports_violations_in_stable_order(self) -> None:
        with valid_fixture() as root:
            for relative_path in ("blueprint/LICENSE", "LICENSE"):
                self.append_bytes_and_stage(root, relative_path, b"\n")
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            diagnostics = [
                line
                for line in result.stderr.splitlines()
                if line.startswith("license: ")
            ]
            self.assertEqual(diagnostics, sorted(diagnostics))


if __name__ == "__main__":
    unittest.main()
