#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate repository and published-artifact license contracts."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


APACHE_LICENSE_SHA256 = (
    "4f321b887d0350e672e06c8a8bc85cff17ee2d1f3c360617079041ceee324c81"
)
APACHE_LICENSE_SIZE = 11_432
COC_SHA256 = "2f336982e73e7c609ee949ea353b9ca016ba63c92e76fbbea6784129ec01f5e2"
OPENAPI_TEMPLATE_LICENSE_SHA256 = (
    "91a2fcdfc23cbd1188a22cc3b76647bf6eb05c87889e376a19fe478f0398ff02"
)

CANONICAL_LICENSE_PATHS = (
    "LICENSE",
    "blueprint/LICENSE",
    "service/attribution-lib/LICENSE",
    "service/service-app/LICENSE",
    "service/service-app/openapi/extras/LICENSE",
    "service/service-app/nvidia_attribution_client/LICENSE",
)
OPENAPI_TEMPLATE_LICENSE = "service/service-app/openapi/templates/LICENSE"
DECLARED_LICENSE_PATHS = frozenset((*CANONICAL_LICENSE_PATHS, OPENAPI_TEMPLATE_LICENSE))
LEGAL_FILE_BASENAME = re.compile(
    r"^(?:licen[cs]e|copying|notice|eula)(?:[._-].*)?$", re.IGNORECASE
)

PYPROJECT_PATHS = (
    "blueprint/pyproject.toml",
    "service/attribution-lib/pyproject.toml",
    "service/service-app/pyproject.toml",
    "service/service-app/openapi/extras/pyproject.toml",
    "service/service-app/nvidia_attribution_client/pyproject.toml",
)
EXPECTED_PROJECT_NAMES = {
    "blueprint/pyproject.toml": "nvidia-attribution-blueprint",
    "service/attribution-lib/pyproject.toml": "attribution-lib",
    "service/service-app/pyproject.toml": "nvidia-attribution-service",
    "service/service-app/openapi/extras/pyproject.toml": "nvidia-attribution-client",
    "service/service-app/nvidia_attribution_client/pyproject.toml": "nvidia-attribution-client",
}
SOURCE_VERSION_PATHS = {
    "attribution-lib": "service/attribution-lib/src/attribution_lib/__init__.py",
    "nvidia-attribution-blueprint": "blueprint/src/attribution_blueprint/__init__.py",
    "nvidia-attribution-service": "service/service-app/app/__init__.py",
    "nvidia-attribution-client": (
        "service/service-app/nvidia_attribution_client/"
        "nvidia_attribution_client/__init__.py"
    ),
}
FRONTEND_PACKAGE = "blueprint/src/attribution_blueprint/webui/frontend/package.json"
CLIENT_ROOT = "service/service-app/nvidia_attribution_client"
EXTRAS_ROOT = "service/service-app/openapi/extras"
PARTIAL_HEADER = "service/service-app/openapi/templates/partial_header.mustache"
HEADER_MARKDOWN_PATHS = frozenset(
    {
        "service/attribution-lib/docs/adding-a-rule-agent-guide.md",
        "service/attribution-lib/docs/adding-a-rule.md",
    }
)
DERIVED_TEMPLATE_PATHS = (
    "service/service-app/openapi/templates/__init__api.mustache",
    "service/service-app/openapi/templates/api_response.mustache",
    "service/service-app/openapi/templates/partial_api_args.mustache",
)
DERIVED_TEMPLATE_NOTICE = (
    "OpenAPI Generator does not license generated code; this project applies "
    "Apache-2.0 through partial_header.mustache."
)
EXPECTED_DISTRIBUTIONS = frozenset(
    {
        "attribution-lib",
        "nvidia-attribution-blueprint",
        "nvidia-attribution-client",
        "nvidia-attribution-service",
    }
)
MAX_METADATA_SIZE = 1_048_576
MAX_FRONTEND_ASSET_SIZE = 16_777_216
MAX_ARCHIVE_MEMBER_SCAN_SIZE = 67_108_864
DOCKER_CONTEXT_EXCLUSIONS = (
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
DELIVERY_PATHS = (
    ".ci/build-and-check-license-artifacts.sh",
    ".ci/check-client-drift.sh",
    ".ci/check-license-compliance.py",
    ".ci/run-end-to-end-tests.sh",
    ".ci/run-tests.sh",
    ".ci/tests/test_license_compliance.py",
    ".coderabbit.yaml",
    ".github/workflows/tests.yml",
    ".pre-commit-config.yaml",
    "README.md",
    "blueprint/docker/Dockerfile",
    "blueprint/docker/Dockerfile.dockerignore",
    "service/service-app/Dockerfile",
    "service/service-app/Dockerfile.dockerignore",
    PARTIAL_HEADER,
    *DERIVED_TEMPLATE_PATHS,
)
REQUIRED_TRACKED_PATHS = frozenset(
    (
        *CANONICAL_LICENSE_PATHS,
        "CODE_OF_CONDUCT.md",
        OPENAPI_TEMPLATE_LICENSE,
        *PYPROJECT_PATHS,
        *SOURCE_VERSION_PATHS.values(),
        FRONTEND_PACKAGE,
        *HEADER_MARKDOWN_PATHS,
        *DELIVERY_PATHS,
    )
)

IDENTIFIER_TAG = "SPDX-License-" + "Identifier:"
COPYRIGHT_TAG = "SPDX-FileCopyright" + "Text:"
APACHE_IDENTIFIER = f"{IDENTIFIER_TAG} Apache-2.0"
NVIDIA_COPYRIGHT_SUFFIX = " NVIDIA CORPORATION & AFFILIATES. All rights reserved."
PROPRIETARY_IDENTIFIER = "LicenseRef-" + "NvidiaProprietary"
PRIVATE_CLASSIFIER = "Private :: Do Not " + "Upload"
PROPRIETARY_MARKERS = (
    PROPRIETARY_IDENTIFIER,
    PRIVATE_CLASSIFIER,
    "GOVERNING DOWNLOAD " + "TERMS",
    "retain all intellectual " + "property and proprietary rights",
    "without an express license " + "agreement",
    "NVIDIA Software and Model Evaluation " + "License",
)
PROPRIETARY_MARKER_BYTES = tuple(
    marker.encode("utf-8") for marker in PROPRIETARY_MARKERS
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tracked_paths(root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot list tracked files: {detail}")
    return tuple(
        path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    )


def read_bytes(root: Path, relative_path: str, violations: list[str]) -> bytes | None:
    path = root / relative_path
    try:
        if path.is_symlink() or not path.is_file():
            violations.append(f"{relative_path}: required path is not a regular file")
            return None
        return path.read_bytes()
    except OSError as error:
        violations.append(f"{relative_path}: cannot read required file: {error}")
        return None


def read_text(root: Path, relative_path: str, violations: list[str]) -> str | None:
    data = read_bytes(root, relative_path, violations)
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        violations.append(f"{relative_path}: required text file is not UTF-8")
        return None


def requires_apache_header(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if relative_path in HEADER_MARKDOWN_PATHS or relative_path in PYPROJECT_PATHS:
        return True
    if relative_path in {".coderabbit.yaml", ".pre-commit-config.yaml"}:
        return True
    if relative_path == PARTIAL_HEADER:
        return True
    if relative_path.startswith("service/service-app/openapi/templates/"):
        return False
    if relative_path.startswith(".ci/") and path.suffix in {".py", ".sh"}:
        return True
    if relative_path.startswith(
        (
            "blueprint/src/attribution_blueprint/webui/frontend/",
            "docs/attribution-tour/",
        )
    ) and path.suffix in {".ts", ".tsx", ".js", ".jsx", ".css", ".vue"}:
        return True
    first_party_root = relative_path.startswith(
        ("blueprint/", "service/attribution-lib/", "service/service-app/")
    )
    if not first_party_root:
        return False
    if path.suffix in {".py", ".pyi", ".sh", ".in"} or path.name == "Dockerfile":
        return True
    if path.name == "ruff.toml":
        return True
    return path.suffix == ".toml" and relative_path.startswith(
        (
            "service/attribution-lib/src/attribution_lib/bundled_rules/",
            "service/attribution-lib/tests/declarative/fixtures/",
        )
    )


def header_style(relative_path: str) -> tuple[str, str | None]:
    suffix = PurePosixPath(relative_path).suffix
    if suffix in {".md", ".html", ".svg", ".vue"}:
        return "", "-->"
    if suffix == ".css":
        return " * ", " */"
    if suffix in {".ts", ".tsx", ".js", ".jsx"}:
        return "// ", None
    return "# ", None


def expected_header_start(relative_path: str, lines: list[str]) -> int:
    _, closing = header_style(relative_path)
    if closing is not None:
        index = 0
        if PurePosixPath(relative_path).suffix == ".md" and lines and lines[0] == "---":
            try:
                index = lines.index("---", 1) + 1
            except ValueError:
                return -1
            while index < len(lines) and not lines[index]:
                index += 1
        opening = "/*" if closing == " */" else "<!--"
        return index + 1 if index < len(lines) and lines[index] == opening else -1
    if relative_path == PARTIAL_HEADER:
        return (
            1
            if lines and lines[0].startswith("{{! Override of openapi-generator")
            else -1
        )

    index = 0
    if lines and re.fullmatch(r"// @vitest-environment [a-z-]+", lines[0]):
        index = 1
    if lines and (lines[0].startswith("#!") or lines[0].startswith("# syntax=")):
        index = 1
        while index < len(lines) and not lines[index]:
            index += 1
    if (
        relative_path.startswith(f"{CLIENT_ROOT}/")
        and index < len(lines)
        and lines[index] == "# coding: utf-8"
    ):
        index += 1
        while index < len(lines) and not lines[index]:
            index += 1
        if index < len(lines) and lines[index] == "# flake8: noqa":
            index += 1
            while index < len(lines) and not lines[index]:
                index += 1
    return index


def validate_license_files(
    root: Path, paths: tuple[str, ...], violations: list[str]
) -> None:
    tracked = frozenset(paths)
    for relative_path in sorted(REQUIRED_TRACKED_PATHS - tracked):
        violations.append(f"{relative_path}: required contract file is not tracked")

    for relative_path in CANONICAL_LICENSE_PATHS:
        data = read_bytes(root, relative_path, violations)
        if data is None:
            continue
        if len(data) != APACHE_LICENSE_SIZE or digest(data) != APACHE_LICENSE_SHA256:
            violations.append(f"{relative_path}: expected canonical Apache-2.0 text")

    coc = read_bytes(root, "CODE_OF_CONDUCT.md", violations)
    if coc is not None and digest(coc) != COC_SHA256:
        violations.append("CODE_OF_CONDUCT.md: protected CC BY-SA 4.0 boundary changed")

    template_license = read_bytes(root, OPENAPI_TEMPLATE_LICENSE, violations)
    if (
        template_license is not None
        and digest(template_license) != OPENAPI_TEMPLATE_LICENSE_SHA256
    ):
        violations.append(
            f"{OPENAPI_TEMPLATE_LICENSE}: protected upstream license changed"
        )

    for relative_path in paths:
        if (
            LEGAL_FILE_BASENAME.fullmatch(PurePosixPath(relative_path).name)
            and relative_path not in DECLARED_LICENSE_PATHS
        ):
            violations.append(f"{relative_path}: undeclared license file")


def validate_headers_and_relics(
    root: Path, paths: tuple[str, ...], violations: list[str]
) -> None:
    for relative_path in paths:
        path = root / relative_path
        try:
            data = path.read_bytes()
        except OSError as error:
            violations.append(f"{relative_path}: cannot read tracked file: {error}")
            continue
        text = data.decode("utf-8", errors="replace")
        if any(marker in text for marker in PROPRIETARY_MARKERS):
            violations.append(f"{relative_path}: proprietary license relic")

        if b"\0" in data:
            if requires_apache_header(relative_path):
                violations.append(
                    f"{relative_path}: required source file contains NUL bytes"
                )
            continue

        if relative_path in DERIVED_TEMPLATE_PATHS:
            # Mustache comment headers are checked with the template boundary.
            continue
        has_nvidia_spdx = COPYRIGHT_TAG in text and "NVIDIA CORPORATION" in text
        if not (requires_apache_header(relative_path) or has_nvidia_spdx):
            continue
        lines = text.splitlines()
        identifier_occurrences = text.count(IDENTIFIER_TAG)
        copyright_occurrences = text.count(COPYRIGHT_TAG)
        comment_prefix, closing = header_style(relative_path)
        expected_identifier_line = f"{comment_prefix}{APACHE_IDENTIFIER}"
        header_start = expected_header_start(relative_path, lines)
        if (
            identifier_occurrences != 1
            or header_start < 0
            or header_start + 1 >= len(lines)
            or lines[header_start + 1] != expected_identifier_line
        ):
            violations.append(
                f"{relative_path}: expected one exact Apache SPDX comment after the permitted preamble"
            )
        copyright_match = None
        if 0 <= header_start < len(lines):
            copyright_match = re.fullmatch(
                re.escape(f"{comment_prefix}{COPYRIGHT_TAG} Copyright (c) ")
                + r"(20\d{2})(?:-(20\d{2}))?"
                + re.escape(NVIDIA_COPYRIGHT_SUFFIX),
                lines[header_start],
            )
        valid_year_range = bool(
            copyright_match
            and (
                copyright_match.group(2) is None
                or int(copyright_match.group(1)) <= int(copyright_match.group(2))
            )
        )
        if (
            copyright_occurrences != 1
            or header_start < 0
            or header_start >= len(lines)
            or not valid_year_range
        ):
            violations.append(
                f"{relative_path}: expected one exact NVIDIA SPDX copyright comment "
                "after the permitted preamble"
            )
        if closing is not None and (
            header_start < 0
            or header_start + 2 >= len(lines)
            or lines[header_start + 2].strip() != closing.strip()
        ):
            violations.append(
                f"{relative_path}: expected a closed SPDX HTML comment"
                if closing == "-->"
                else f"{relative_path}: expected a closed SPDX CSS comment"
            )


def validate_package_metadata(root: Path, violations: list[str]) -> None:
    for relative_path in PYPROJECT_PATHS:
        data = read_bytes(root, relative_path, violations)
        if data is None:
            continue
        try:
            document = tomllib.loads(data.decode("utf-8"))
            project = document["project"]
        except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            violations.append(
                f"{relative_path}: cannot parse project metadata: {error}"
            )
            continue
        expected_name = EXPECTED_PROJECT_NAMES[relative_path]
        if project.get("name") != expected_name:
            violations.append(f"{relative_path}: project.name must be {expected_name}")
        if project.get("license") != "Apache-2.0":
            violations.append(f"{relative_path}: project.license must be Apache-2.0")
        if project.get("license-files") != ["LICENSE"]:
            violations.append(
                f'{relative_path}: project.license-files must be ["LICENSE"]'
            )
        if PRIVATE_CLASSIFIER in project.get("classifiers", []):
            violations.append(
                f"{relative_path}: private distribution classifier is forbidden"
            )
        expected_backend = "hatchling>=1.27"
        build_requirements = document.get("build-system", {}).get("requires", [])
        if expected_backend not in build_requirements:
            violations.append(
                f"{relative_path}: build-system.requires must include {expected_backend}"
            )

    package_text = read_text(root, FRONTEND_PACKAGE, violations)
    if package_text is not None:
        try:
            package = json.loads(package_text)
        except json.JSONDecodeError as error:
            violations.append(f"{FRONTEND_PACKAGE}: cannot parse JSON: {error}")
        else:
            if package.get("private") is not True:
                violations.append(f"{FRONTEND_PACKAGE}: private must be true")
            if package.get("license") != "Apache-2.0":
                violations.append(f"{FRONTEND_PACKAGE}: license must be Apache-2.0")

    for filename in ("LICENSE", "pyproject.toml"):
        extras_path = f"{EXTRAS_ROOT}/{filename}"
        client_path = f"{CLIENT_ROOT}/{filename}"
        extras = read_bytes(root, extras_path, violations)
        client = read_bytes(root, client_path, violations)
        if extras is not None and client is not None and extras != client:
            violations.append(f"{client_path}: differs from OpenAPI extras source")


def artifact_version_matches(distribution: str, actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    if distribution == "nvidia-attribution-client":
        return False
    # Component builds append Git distance and dirty state to the release literal.
    return (
        re.fullmatch(re.escape(expected) + r"\+(?:[0-9]+(?:\.dirty)?|dirty)", actual)
        is not None
    )


def source_distribution_versions(root: Path, violations: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    version_pattern = re.compile(r'^__version__ = "([^"\n]+)"$', re.MULTILINE)
    for distribution, relative_path in SOURCE_VERSION_PATHS.items():
        text = read_text(root, relative_path, violations)
        if text is None:
            continue
        matches = version_pattern.findall(text)
        if len(matches) != 1:
            violations.append(
                f"{relative_path}: expected one exact __version__ assignment"
            )
            continue
        versions[distribution] = matches[0]
    return versions


def require_text(
    root: Path,
    relative_path: str,
    expected: str,
    message: str,
    violations: list[str],
) -> None:
    text = read_text(root, relative_path, violations)
    if text is not None and expected not in text:
        violations.append(f"{relative_path}: {message}")


def require_active_line(
    root: Path,
    relative_path: str,
    pattern: str,
    message: str,
    violations: list[str],
) -> None:
    text = read_text(root, relative_path, violations)
    if text is not None and re.search(pattern, text, flags=re.MULTILINE) is None:
        violations.append(f"{relative_path}: {message}")


def yaml_mapping_bounds(
    text: str, path: tuple[str, ...]
) -> tuple[list[str], int, int] | None:
    """Return the exact indentation-defined block for a YAML mapping path."""
    lines = text.splitlines()
    search_start = 0
    search_end = len(lines)
    for depth, key in enumerate(path):
        indentation = depth * 2
        expected = f"{' ' * indentation}{key}:"
        try:
            key_index = next(
                index
                for index in range(search_start, search_end)
                if lines[index].rstrip() == expected
            )
        except StopIteration:
            return None
        block_end = next(
            (
                index
                for index in range(key_index + 1, search_end)
                if lines[index].strip()
                and not lines[index].lstrip().startswith("#")
                and len(lines[index]) - len(lines[index].lstrip()) <= indentation
            ),
            search_end,
        )
        search_start = key_index + 1
        search_end = block_end
    return lines, search_start, search_end


def yaml_sequence_has_scalar(text: str, path: tuple[str, ...], value: str) -> bool:
    bounds = yaml_mapping_bounds(text, path)
    if bounds is None:
        return False
    lines, start, end = bounds
    item_indentation = len(path) * 2
    expected = f"{' ' * item_indentation}- {value}"
    return sum(lines[index].rstrip() == expected for index in range(start, end)) == 1


def yaml_sequence_mapping_has_scalar(
    text: str,
    path: tuple[str, ...],
    field: str,
    value: str,
) -> bool:
    """Return whether an exclusion contains a value or unsupported YAML.

    GitHub Actions accepts many YAML spellings for matrix exclusions.  This
    checker deliberately supports the simple block and flow forms used by the
    repository.  It fails closed for aliases, anchors, block scalars, escapes,
    and other complex forms so an equivalent spelling cannot hide a protected
    suite from the structural workflow check.
    """
    if len(path) < 2:
        return True
    bounds = yaml_mapping_bounds(text, path[:-1])
    if bounds is None:
        return False
    lines, start, end = bounds
    key_indentation = (len(path) - 1) * 2
    sequence_key = re.escape(path[-1])
    sequence_key_pattern = re.compile(
        rf"^{' ' * key_indentation}(?:{sequence_key}|\"{sequence_key}\"|"
        rf"'{sequence_key}')\s*:\s*(.*)$"
    )
    declarations = [
        (index, match.group(1))
        for index in range(start, end)
        if (match := sequence_key_pattern.fullmatch(lines[index])) is not None
    ]
    if not declarations:
        return False
    if len(declarations) != 1:
        return True

    declaration_index, inline_value = declarations[0]
    block_end = next(
        (
            index
            for index in range(declaration_index + 1, end)
            if lines[index].strip()
            and not lines[index].lstrip().startswith("#")
            and len(lines[index]) - len(lines[index].lstrip()) <= key_indentation
        ),
        end,
    )
    body_lines = [inline_value.split(" #", 1)[0].strip()]
    body_lines.extend(
        line.strip()
        for line in lines[declaration_index + 1 : block_end]
        if line.strip() and not line.lstrip().startswith("#")
    )
    body = "\n".join(part for part in body_lines if part)
    if not body or body == "[]":
        return False

    # These features can change the semantic scalar value or import it from
    # another node.  Reject them instead of attempting a partial YAML parse.
    if any(marker in body for marker in ("&", "*", "\\", ">", "|")):
        return True

    field_key = re.escape(field)
    field_token = rf"(?:{field_key}|\"{field_key}\"|'{field_key}')"
    field_pattern = re.compile(rf"(?<![A-Za-z0-9_-]){field_token}\s*:")
    scalar_pattern = re.compile(
        rf"(?<![A-Za-z0-9_-]){field_token}\s*:\s*"
        r"(?P<scalar>\"[^\"\n]*\"|'(?:[^']|'')*'|[^,\]}\n]+)"
    )
    field_count = len(field_pattern.findall(body))
    scalar_matches = list(scalar_pattern.finditer(body))
    if len(scalar_matches) != field_count:
        return True

    for match in scalar_matches:
        candidate = match.group("scalar").split(" #", 1)[0].strip()
        if candidate.startswith('"'):
            try:
                decoded = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                return True
            if not isinstance(decoded, str):
                return True
            candidate = decoded
        elif candidate.startswith("'"):
            if not candidate.endswith("'"):
                return True
            candidate = candidate[1:-1].replace("''", "'")
        elif re.fullmatch(r"[A-Za-z0-9_.+-]+", candidate) is None:
            return True
        if candidate == value:
            return True
    return False


def yaml_mapping_field_values(
    text: str, path: tuple[str, ...], field: str
) -> list[str]:
    bounds = yaml_mapping_bounds(text, path)
    if bounds is None:
        return []
    lines, start, end = bounds
    field_indentation = len(path) * 2
    prefix = f"{' ' * field_indentation}{field}:"
    values = []
    for index in range(start, end):
        line = lines[index]
        if (
            line.startswith(prefix)
            and len(line) - len(line.lstrip()) == field_indentation
        ):
            values.append(line[len(prefix) :].strip())
    return values


def yaml_scalar_is_false(value: str) -> bool:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    normalized = re.sub(r"\s+", "", value).lower()
    return normalized in {"false", "no", "off", "0", "${{false}}"}


def yaml_mapping_is_disabled(text: str, path: tuple[str, ...]) -> bool:
    values = yaml_mapping_field_values(text, path, "if")
    return len(values) > 1 or any(yaml_scalar_is_false(value) for value in values)


def yaml_mapping_can_continue_on_error(text: str, path: tuple[str, ...]) -> bool:
    values = yaml_mapping_field_values(text, path, "continue-on-error")
    return len(values) > 1 or any(not yaml_scalar_is_false(value) for value in values)


def yaml_job_has_exact_condition(text: str, path: tuple[str, ...]) -> bool:
    return yaml_mapping_field_values(text, path, "if") == ["${{ always() }}"]


def yaml_job_fails_closed(text: str, path: tuple[str, ...]) -> bool:
    return not yaml_mapping_field_values(text, path, "continue-on-error")


def yaml_run_shell_executes_command(value: str) -> bool:
    """Accept only known shells that execute the generated run script."""
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value in {
        "bash",
        "sh",
        "bash {0}",
        "sh {0}",
        "bash -e {0}",
        "sh -e {0}",
        "bash --noprofile --norc -e -o pipefail {0}",
        "/bin/bash --noprofile --norc -e -o pipefail {0}",
    }


def yaml_steps_have_run(text: str, job_name: str, command: str) -> bool:
    path = ("jobs", job_name, "steps")
    bounds = yaml_mapping_bounds(text, path)
    if bounds is None:
        return False
    lines, start, end = bounds
    item_indentation = len(path) * 2
    field_indentation = item_indentation + 2
    expected_run = f"{' ' * field_indentation}run: {command}"
    job_shells = yaml_mapping_field_values(
        text, ("jobs", job_name, "defaults", "run"), "shell"
    )
    workflow_shells = yaml_mapping_field_values(text, ("defaults", "run"), "shell")
    if len(job_shells) > 1 or len(workflow_shells) > 1:
        return False
    step_starts = [
        index
        for index in range(start, end)
        if lines[index].strip()
        and not lines[index].lstrip().startswith("#")
        and len(lines[index]) - len(lines[index].lstrip()) == item_indentation
        and re.fullmatch(r"-[ \t]+[A-Za-z][A-Za-z0-9_-]*:.*", lines[index].strip())
        is not None
    ]
    matches = 0
    for position, step_start in enumerate(step_starts):
        step_end = step_starts[position + 1] if position + 1 < len(step_starts) else end
        step_lines = lines[step_start:step_end]
        if sum(line.rstrip() == expected_run for line in step_lines) != 1:
            continue
        direct_fields: dict[str, list[str]] = {}
        for line in step_lines:
            if len(line) - len(line.lstrip()) != field_indentation:
                continue
            field_match = re.fullmatch(
                rf"{' ' * field_indentation}([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)",
                line,
            )
            if field_match is not None:
                direct_fields.setdefault(field_match.group(1), []).append(
                    field_match.group(2).strip()
                )
        if direct_fields.get("if") or direct_fields.get("continue-on-error"):
            continue
        step_shells = direct_fields.get("shell", [])
        if len(step_shells) > 1:
            continue
        effective_shells = step_shells or job_shells or workflow_shells
        if effective_shells and not yaml_run_shell_executes_command(
            effective_shells[0]
        ):
            continue
        matches += 1
    return matches == 1


def reject_unconditional_success_termination(
    root: Path, relative_path: str, violations: list[str]
) -> None:
    text = read_text(root, relative_path, violations)
    if text is not None and re.search(
        r"^[ \t]*exit[ \t]+0[ \t]*(?:#.*)?$", text, flags=re.MULTILINE
    ):
        violations.append(
            f"{relative_path}: unconditional successful termination can bypass license checks"
        )


def validate_documentation_and_delivery(root: Path, violations: list[str]) -> None:
    require_text(
        root,
        "README.md",
        "licensed under the Apache License 2.0",
        "missing Apache License 2.0 notice",
        violations,
    )
    require_text(
        root,
        "README.md",
        "CC BY-SA 4.0",
        "missing CC BY-SA 4.0 Code of Conduct boundary",
        violations,
    )
    require_text(
        root,
        "README.md",
        OPENAPI_TEMPLATE_LICENSE,
        "missing vendored OpenAPI Generator license boundary",
        violations,
    )

    for dockerfile in ("blueprint/docker/Dockerfile", "service/service-app/Dockerfile"):
        require_active_line(
            root,
            dockerfile,
            r"^[ \t]*COPY[ \t]+LICENSE[ \t]+/licenses/LICENSE[ \t]*$",
            "final image must contain /licenses/LICENSE",
            violations,
        )
        require_active_line(
            root,
            dockerfile,
            r'^[ \t]*LABEL[ \t]+com\.nvidia\.attribution\.project\.license="Apache-2\.0"[ \t]*$',
            "final image must declare the Apache-2.0 project license label",
            violations,
        )

    require_active_line(
        root,
        "blueprint/docker/Dockerfile.dockerignore",
        r"^!blueprint/LICENSE$",
        "must include the Blueprint package license",
        violations,
    )
    for dockerignore in (
        "blueprint/docker/Dockerfile.dockerignore",
        "service/service-app/Dockerfile.dockerignore",
    ):
        require_active_line(
            root,
            dockerignore,
            r"^!LICENSE$",
            "must include the root license",
            violations,
        )

    dockerignore_exclusions = {
        "blueprint/docker/Dockerfile.dockerignore": (
            "service/attribution-lib/build",
            "service/attribution-lib/src/*.egg-info",
            "blueprint/src/*.egg-info",
        ),
        "service/service-app/Dockerfile.dockerignore": (
            "service/attribution-lib/build",
            "service/attribution-lib/src/*.egg-info",
        ),
    }
    for dockerignore, exclusions in dockerignore_exclusions.items():
        dockerignore_text = read_text(root, dockerignore, violations)
        active_rules = (
            [
                line.strip()
                for line in dockerignore_text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if dockerignore_text is not None
            else []
        )
        last_include_index = max(
            (index for index, rule in enumerate(active_rules) if rule.startswith("!")),
            default=-1,
        )
        for exclusion in (*exclusions, *DOCKER_CONTEXT_EXCLUSIONS):
            require_active_line(
                root,
                dockerignore,
                rf"^{re.escape(exclusion)}$",
                f"must exclude local or generated content: {exclusion}",
                violations,
            )
            exclusion_indices = [
                index for index, rule in enumerate(active_rules) if rule == exclusion
            ]
            if exclusion_indices and exclusion_indices[-1] < last_include_index:
                violations.append(
                    f"{dockerignore}: exclusion must follow all include rules: {exclusion}"
                )

    require_text(
        root,
        PARTIAL_HEADER,
        APACHE_IDENTIFIER,
        "expected Apache-2.0 generated header",
        violations,
    )
    template_copyright = (
        f"{{{{! {COPYRIGHT_TAG} Copyright (c) 2026{NVIDIA_COPYRIGHT_SUFFIX} }}}}"
    )
    template_tag = f"{{{{! {APACHE_IDENTIFIER} }}}}"
    template_provenance = (
        "{{! This template is derived from OpenAPI Generator; see ./LICENSE. "
        f"{DERIVED_TEMPLATE_NOTICE} }}}}"
    )
    for template_path in DERIVED_TEMPLATE_PATHS:
        template_text = read_text(root, template_path, violations)
        if template_text is None:
            continue
        template_lines = template_text.splitlines()
        # Provenance comments come first; the NVIDIA SPDX block follows them.
        if (
            template_lines[2:4] != [template_copyright, template_tag]
            or template_text.count(COPYRIGHT_TAG) != 1
            or template_text.count(IDENTIFIER_TAG) != 1
        ):
            violations.append(
                f"{template_path}: expected exact Apache-2.0 SPDX template tag"
            )
        if (
            not template_lines
            or template_lines[0] != template_provenance
            or template_text.count(template_provenance) != 1
        ):
            violations.append(
                f"{template_path}: missing exact generated-code licensing boundary comment"
            )
    partial_api_args = read_bytes(root, DERIVED_TEMPLATE_PATHS[2], violations)
    if partial_api_args is not None and partial_api_args.endswith((b"\n", b"\r")):
        violations.append(
            f"{DERIVED_TEMPLATE_PATHS[2]}: partial template must not end with a newline"
        )
    require_active_line(
        root,
        ".ci/run-tests.sh",
        r"(?m)^[ \t]*license-compliance\)[ \t]*\n[ \t]*run_license_compliance[ \t]*\n[ \t]*;;[ \t]*$",
        "missing active license-compliance test dispatcher",
        violations,
    )
    workflow_path = ".github/workflows/tests.yml"
    workflow_text = read_text(root, workflow_path, violations)
    if workflow_text is not None:
        python_job_path = ("jobs", "python-tests")
        if not yaml_job_has_exact_condition(workflow_text, python_job_path):
            violations.append(
                f"{workflow_path}: python-tests job must remain executable"
            )
        if not yaml_job_fails_closed(workflow_text, python_job_path):
            violations.append(
                f"{workflow_path}: python-tests job must fail when license compliance fails"
            )
        if not yaml_sequence_has_scalar(
            workflow_text,
            ("jobs", "python-tests", "strategy", "matrix", "suite"),
            "license-compliance",
        ):
            violations.append(f"{workflow_path}: missing license-compliance CI suite")
        if yaml_sequence_mapping_has_scalar(
            workflow_text,
            ("jobs", "python-tests", "strategy", "matrix", "exclude"),
            "suite",
            "license-compliance",
        ):
            violations.append(
                f"{workflow_path}: license-compliance suite must not be excluded"
            )
        if not yaml_steps_have_run(
            workflow_text,
            "python-tests",
            'bash .ci/run-tests.sh "${{ matrix.suite }}"',
        ):
            violations.append(f"{workflow_path}: missing active matrix test dispatcher")
        frontend_job_path = ("jobs", "frontend-tests")
        if not yaml_job_has_exact_condition(workflow_text, frontend_job_path):
            violations.append(
                f"{workflow_path}: frontend-tests job must remain executable"
            )
        if not yaml_job_fails_closed(workflow_text, frontend_job_path):
            violations.append(
                f"{workflow_path}: frontend-tests job must fail when artifact validation fails"
            )
        if not yaml_steps_have_run(
            workflow_text,
            "frontend-tests",
            "bash .ci/build-and-check-license-artifacts.sh",
        ):
            violations.append(
                f"{workflow_path}: missing published-artifact license validation"
            )
        end_to_end_job_path = ("jobs", "end-to-end")
        if not yaml_job_has_exact_condition(workflow_text, end_to_end_job_path):
            violations.append(f"{workflow_path}: end-to-end job must remain executable")
        if not yaml_job_fails_closed(workflow_text, end_to_end_job_path):
            violations.append(
                f"{workflow_path}: end-to-end job must fail when license validation fails"
            )
        if not yaml_steps_have_run(
            workflow_text,
            "end-to-end",
            "bash .ci/run-tests.sh end-to-end",
        ):
            violations.append(
                f"{workflow_path}: missing active end-to-end license validation"
            )
    require_active_line(
        root,
        ".ci/build-and-check-license-artifacts.sh",
        r"^[ \t]*--artifacts[ \t]+\"\$\{artifacts\}\"[ \t]*$",
        "artifact build must run archive license validation",
        violations,
    )
    require_active_line(
        root,
        ".ci/build-and-check-license-artifacts.sh",
        r'^[ \t]*echo[ \t]+"source tree changed during artifact build"[ \t]+>&2[ \t]*$',
        "missing clean-source-tree invariant",
        violations,
    )
    managed_checker_command = (
        "uv run --python 3.14.6 --no-project python .ci/check-license-compliance.py"
    )
    indexed_checker_command = (
        'uv run --python 3.14.6 --no-project python "${source_root}/'
        '.ci/check-license-compliance.py"'
    )
    require_active_line(
        root,
        ".ci/build-and-check-license-artifacts.sh",
        rf"^[ \t]*{re.escape(indexed_checker_command)}[ \t]+\\[ \t]*$",
        "checker must use the indexed managed Python runtime",
        violations,
    )
    require_active_line(
        root,
        ".ci/build-and-check-license-artifacts.sh",
        r'^[ \t]*--root[ \t]+"\$\{source_root\}"[ \t]+\\[ \t]*$',
        "checker must validate the indexed source root",
        violations,
    )
    artifact_script = read_text(
        root, ".ci/build-and-check-license-artifacts.sh", violations
    )
    if artifact_script is not None:
        command_lines = (
            f"{indexed_checker_command} \\",
            '    --root "${source_root}" \\',
            '    --artifacts "${artifacts}"',
        )
        if "\n".join(command_lines) not in artifact_script:
            violations.append(
                ".ci/build-and-check-license-artifacts.sh: indexed checker and "
                "artifact validation must execute as one command"
            )
    require_active_line(
        root,
        ".pre-commit-config.yaml",
        rf"^[ \t]*entry:[ \t]+{re.escape(managed_checker_command)}[ \t]+--root[ \t]+\.[ \t]*$",
        "checker must use the managed Python runtime",
        violations,
    )
    for shell_path in (
        ".ci/run-tests.sh",
        ".ci/build-and-check-license-artifacts.sh",
        ".ci/run-end-to-end-tests.sh",
    ):
        reject_unconditional_success_termination(root, shell_path, violations)
    for image in ("service_image", "blueprint_image"):
        require_active_line(
            root,
            ".ci/run-end-to-end-tests.sh",
            rf'^[ \t]*assert_image_license[ \t]+"\$\{{{image}\}}"[ \t]*$',
            f"missing runtime license validation for {image}",
            violations,
        )
    for pattern, message in (
        (
            r"^[ \t]*docker image inspect[ \t]*\\[ \t]*$",
            "runtime check must inspect the final image label",
        ),
        (
            r"^[ \t]*--volume[ \t]+\"\$\{source_root\}/LICENSE:/expected/LICENSE:ro\"[ \t]*\\[ \t]*$",
            "runtime check must mount the canonical license",
        ),
        (
            r"^[ \t]*-c[ \t]+'[^\n]*Path\(\"/licenses/LICENSE\"\)\.read_bytes\(\)[ \t]*!=[ \t]*Path\(\"/expected/LICENSE\"\)\.read_bytes\(\)[^\n]*'[ \t]*$",
            "runtime check must compare image and canonical license bytes",
        ),
    ):
        require_active_line(
            root,
            ".ci/run-end-to-end-tests.sh",
            pattern,
            message,
            violations,
        )
    require_active_line(
        root,
        ".pre-commit-config.yaml",
        r"^[ \t]*-[ \t]+id:[ \t]+license-compliance[ \t]*$",
        "missing license compliance pre-commit hook",
        violations,
    )


def normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def safe_archive_path(name: str) -> PurePosixPath | None:
    if not name or "\\" in name or "\0" in name or re.match(r"^[A-Za-z]:", name):
        return None
    candidate = name[:-1] if name.endswith("/") else name
    parts = candidate.split("/")
    path = PurePosixPath(candidate)
    if (
        not candidate
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
    ):
        return None
    return path


def zip_member_has_valid_type(member: zipfile.ZipInfo) -> bool:
    file_type = (member.external_attr >> 16) & 0o170000
    if member.is_dir():
        return file_type in {0, stat.S_IFDIR}
    return file_type in {0, stat.S_IFREG}


def metadata_singleton(
    metadata: email.message.Message,
    field: str,
    label: str,
    violations: list[str],
) -> str | None:
    values = metadata.get_all(field, [])
    if len(values) != 1:
        violations.append(f"{label}: metadata must contain one {field} field")
        return None
    return values[0]


def validate_core_metadata(
    metadata: email.message.Message,
    label: str,
    violations: list[str],
) -> tuple[str, str] | None:
    metadata_version = metadata_singleton(
        metadata, "Metadata-Version", label, violations
    )
    distribution = metadata_singleton(metadata, "Name", label, violations)
    version = metadata_singleton(metadata, "Version", label, violations)
    license_expression = metadata_singleton(
        metadata, "License-Expression", label, violations
    )
    if None in {metadata_version, distribution, version, license_expression}:
        return None
    assert metadata_version is not None
    assert distribution is not None
    assert version is not None
    assert license_expression is not None
    try:
        parsed_metadata_version = tuple(
            int(part) for part in metadata_version.split(".")
        )
    except ValueError:
        parsed_metadata_version = ()
    if parsed_metadata_version < (2, 4):
        violations.append(f"{label}: Metadata-Version must be at least 2.4")
    if license_expression != "Apache-2.0":
        violations.append(f"{label}: License-Expression must be Apache-2.0")
    if metadata.get_all("License-File", []) != ["LICENSE"]:
        violations.append(f"{label}: License-File must be exactly LICENSE")
    if metadata.get_all("License", []):
        violations.append(f"{label}: legacy License metadata field is forbidden")
    metadata_text = metadata.as_string()
    if any(marker in metadata_text for marker in PROPRIETARY_MARKERS):
        violations.append(f"{label}: proprietary license relic")
    return distribution, version


def parse_wheel_filename(
    path: Path, label: str, violations: list[str]
) -> tuple[str, str] | None:
    fields = path.name.removesuffix(".whl").split("-")
    if len(fields) not in {5, 6}:
        violations.append(f"{label}: invalid wheel filename")
        return None
    distribution, version = fields[:2]
    if not re.fullmatch(r"[A-Za-z0-9_]+", distribution) or not re.fullmatch(
        r"[A-Za-z0-9_.!+]+", version
    ):
        violations.append(f"{label}: invalid wheel filename")
        return None
    tag_fields = fields[-3:]
    if any(not re.fullmatch(r"[A-Za-z0-9_.]+", tag) for tag in tag_fields):
        violations.append(f"{label}: invalid wheel filename")
        return None
    if len(fields) == 6 and not re.fullmatch(r"[0-9][A-Za-z0-9_]*", fields[2]):
        violations.append(f"{label}: invalid wheel build tag")
        return None
    return distribution, version


def validate_blueprint_frontend(
    names: frozenset[str],
    read_member,
    prefix: str,
    label: str,
    violations: list[str],
) -> None:
    index_name = f"{prefix}/index.html"
    if index_name not in names:
        violations.append(
            f"{label}: Blueprint archive must contain the built frontend index"
        )
        return
    try:
        index_data = read_member(index_name)
    except (KeyError, OSError, RuntimeError) as error:
        violations.append(f"{label}: cannot read Blueprint frontend index: {error}")
        return
    if not index_data or len(index_data) > MAX_FRONTEND_ASSET_SIZE:
        violations.append(f"{label}: Blueprint frontend index has an invalid size")
        return
    try:
        index_text = index_data.decode("utf-8")
    except UnicodeDecodeError:
        violations.append(f"{label}: Blueprint frontend index is not UTF-8")
        return
    references: list[tuple[str, str]] = []
    for reference in re.findall(
        r"(?:src|href)\s*=\s*[\"']([^\"']+)[\"']", index_text, flags=re.IGNORECASE
    ):
        clean_reference = reference.split("#", 1)[0].split("?", 1)[0]
        if (
            not clean_reference
            or clean_reference.startswith("//")
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", clean_reference)
        ):
            continue
        clean_reference = clean_reference.removeprefix("/").removeprefix("./")
        asset_path = safe_archive_path(clean_reference)
        if asset_path is None:
            violations.append(
                f"{label}: Blueprint frontend index contains an unsafe local reference: "
                f"{reference}"
            )
            continue
        references.append(
            (asset_path.suffix.lower(), f"{prefix}/{asset_path.as_posix()}")
        )
    notices_name = f"{prefix}/assets/THIRD_PARTY_NOTICES.txt"
    references.append((".txt", notices_name))
    for suffix in (".js", ".css"):
        if not any(reference_suffix == suffix for reference_suffix, _ in references):
            violations.append(
                f"{label}: Blueprint frontend index must reference a local {suffix} asset"
            )
    for _, member_name in references:
        if member_name not in names:
            violations.append(
                f"{label}: Blueprint frontend asset is missing: {member_name}"
            )
            continue
        try:
            asset = read_member(member_name)
        except (KeyError, OSError, RuntimeError) as error:
            violations.append(f"{label}: cannot read Blueprint frontend asset: {error}")
            continue
        if not asset or len(asset) > MAX_FRONTEND_ASSET_SIZE:
            violations.append(
                f"{label}: Blueprint frontend asset has an invalid size: {member_name}"
            )


def validate_archive_member_payload(
    data: bytes, member_name: str, label: str, violations: list[str]
) -> None:
    if any(marker in data for marker in PROPRIETARY_MARKER_BYTES):
        violations.append(
            f"{label}: proprietary license relic in archive member {member_name}"
        )


def validate_wheel(
    path: Path, canonical: bytes, violations: list[str]
) -> tuple[str, str] | None:
    label = path.as_posix()
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            safe_paths = [safe_archive_path(name) for name in names]
            normalized_names = [
                member_path.as_posix()
                for member_path in safe_paths
                if member_path is not None
            ]
            if len(normalized_names) != len(set(normalized_names)):
                violations.append(f"{label}: wheel contains duplicate members")
                return None
            if any(member_path is None for member_path in safe_paths):
                violations.append(f"{label}: wheel contains an unsafe member path")
                return None
            if any(not zip_member_has_valid_type(member) for member in members):
                violations.append(f"{label}: wheel contains a non-regular member")
                return None
            file_members = [member for member in members if not member.is_dir()]
            for member in file_members:
                if member.file_size > MAX_ARCHIVE_MEMBER_SCAN_SIZE:
                    violations.append(
                        f"{label}: archive member is too large for license scan: "
                        f"{member.filename}"
                    )
                    continue
                validate_archive_member_payload(
                    archive.read(member), member.filename, label, violations
                )

            metadata_members = [
                member
                for member in file_members
                if len(PurePosixPath(member.filename).parts) == 2
                and PurePosixPath(member.filename).parent.name.endswith(".dist-info")
                and PurePosixPath(member.filename).name == "METADATA"
            ]
            if len(metadata_members) != 1:
                violations.append(
                    f"{label}: wheel must contain one canonical METADATA file"
                )
                return None
            metadata_member = metadata_members[0]
            if metadata_member.file_size > MAX_METADATA_SIZE:
                violations.append(f"{label}: wheel METADATA is too large")
                return None
            metadata = email.message_from_bytes(archive.read(metadata_member))
            identity = validate_core_metadata(metadata, f"{label}: wheel", violations)
            if identity is None:
                return None
            distribution, version = identity
            filename_identity = parse_wheel_filename(path, label, violations)
            if filename_identity is None:
                return None
            filename_distribution, filename_version = filename_identity
            if (
                filename_distribution.lower()
                != normalized_distribution_name(distribution)
                or filename_version != version
            ):
                violations.append(
                    f"{label}: wheel filename does not match METADATA identity"
                )
            dist_info = (
                f"{normalized_distribution_name(distribution)}-{version}.dist-info"
            )
            if PurePosixPath(metadata_member.filename).parent.as_posix() != dist_info:
                violations.append(
                    f"{label}: wheel METADATA is outside its canonical dist-info root"
                )
                return None
            license_name = f"{dist_info}/licenses/LICENSE"
            license_members = [
                member
                for member in file_members
                if PurePosixPath(member.filename).name == "LICENSE"
            ]
            if (
                len(license_members) != 1
                or license_members[0].filename != license_name
                or license_members[0].file_size != len(canonical)
                or archive.read(license_members[0]) != canonical
            ):
                violations.append(
                    f"{label}: wheel must contain the canonical dist-info LICENSE"
                )
            if distribution == "nvidia-attribution-blueprint":
                file_names = frozenset(member.filename for member in file_members)
                members_by_name = {member.filename: member for member in file_members}

                def read_frontend_member(name: str) -> bytes:
                    member = members_by_name[name]
                    if member.file_size > MAX_FRONTEND_ASSET_SIZE:
                        return b""
                    return archive.read(member)

                validate_blueprint_frontend(
                    file_names,
                    read_frontend_member,
                    "attribution_blueprint/webui/static",
                    f"{label}: wheel",
                    violations,
                )
            return distribution, version
    except (
        KeyError,
        NotImplementedError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
    ) as error:
        violations.append(f"{label}: cannot inspect wheel: {error}")
        return None


def validate_sdist(
    path: Path, canonical: bytes, violations: list[str]
) -> tuple[str, str] | None:
    label = path.as_posix()
    try:
        with tarfile.open(path, mode="r:*") as archive:
            all_members = archive.getmembers()
            names = [member.name for member in all_members]
            safe_paths = [safe_archive_path(name) for name in names]
            normalized_names = [
                member_path.as_posix()
                for member_path in safe_paths
                if member_path is not None
            ]
            if len(normalized_names) != len(set(normalized_names)):
                violations.append(f"{label}: sdist contains duplicate members")
                return None
            if any(path is None for path in safe_paths):
                violations.append(f"{label}: sdist contains an unsafe member path")
                return None
            if any(not (member.isfile() or member.isdir()) for member in all_members):
                violations.append(f"{label}: sdist contains a non-regular member")
                return None
            roots = {path.parts[0] for path in safe_paths if path is not None}
            if len(roots) != 1:
                violations.append(f"{label}: sdist must contain one top-level root")
                return None
            root_name = next(iter(roots))
            file_members = [member for member in all_members if member.isfile()]
            for member in file_members:
                if member.size > MAX_ARCHIVE_MEMBER_SCAN_SIZE:
                    violations.append(
                        f"{label}: archive member is too large for license scan: {member.name}"
                    )
                    continue
                extracted_member = archive.extractfile(member)
                if extracted_member is None:
                    violations.append(
                        f"{label}: cannot read archive member: {member.name}"
                    )
                    continue
                validate_archive_member_payload(
                    extracted_member.read(), member.name, label, violations
                )
            if any(
                len(PurePosixPath(member.name).parts) < 2 for member in file_members
            ):
                violations.append(f"{label}: sdist file is outside the top-level root")
                return None
            metadata_name = f"{root_name}/PKG-INFO"
            license_name = f"{root_name}/LICENSE"
            metadata_members = [
                member for member in file_members if member.name == metadata_name
            ]
            license_members = [
                member for member in file_members if member.name == license_name
            ]
            if len(metadata_members) != 1:
                violations.append(f"{label}: sdist must contain one root PKG-INFO file")
                return None
            if len(license_members) != 1:
                violations.append(f"{label}: sdist must contain one root LICENSE")
                return None
            metadata_file = archive.extractfile(metadata_members[0])
            if metadata_file is None:
                violations.append(f"{label}: cannot read sdist PKG-INFO")
                return None
            if metadata_members[0].size > MAX_METADATA_SIZE:
                violations.append(f"{label}: sdist PKG-INFO is too large")
                return None
            metadata = email.message_from_bytes(metadata_file.read())
            identity = validate_core_metadata(metadata, f"{label}: sdist", violations)
            if identity is None:
                return None
            distribution, version = identity
            expected_root = f"{normalized_distribution_name(distribution)}-{version}"
            if root_name != expected_root:
                violations.append(
                    f"{label}: sdist files are outside the canonical package root"
                )
                return None
            filename_root = path.name.removesuffix(".tar.gz")
            if filename_root != expected_root:
                violations.append(
                    f"{label}: sdist filename does not match PKG-INFO identity"
                )
            extracted = archive.extractfile(license_members[0])
            if (
                extracted is None
                or license_members[0].size != len(canonical)
                or extracted.read() != canonical
            ):
                violations.append(
                    f"{label}: sdist must contain the canonical root LICENSE"
                )
            if distribution == "nvidia-attribution-blueprint":
                members_by_name = {member.name: member for member in file_members}

                def read_frontend_member(name: str) -> bytes:
                    member = members_by_name[name]
                    if member.size > MAX_FRONTEND_ASSET_SIZE:
                        return b""
                    extracted_member = archive.extractfile(member)
                    if extracted_member is None:
                        raise KeyError(name)
                    return extracted_member.read()

                validate_blueprint_frontend(
                    frozenset(members_by_name),
                    read_frontend_member,
                    f"{root_name}/src/attribution_blueprint/webui/static",
                    f"{label}: sdist",
                    violations,
                )
            return distribution, version
    except (
        KeyError,
        NotImplementedError,
        OSError,
        RuntimeError,
        tarfile.TarError,
    ) as error:
        violations.append(f"{label}: cannot inspect sdist: {error}")
        return None


def validate_artifacts(root: Path, artifacts: Path, violations: list[str]) -> None:
    canonical = read_bytes(root, "LICENSE", violations)
    if canonical is None:
        return
    wheels = sorted(artifacts.rglob("*.whl"))
    sdists = sorted(artifacts.rglob("*.tar.gz"))
    if len(wheels) != 4 or len(sdists) != 4:
        violations.append(
            f"{artifacts.as_posix()}: expected four wheels and four source distributions"
        )
    wheel_identities = [
        validate_wheel(wheel, canonical, violations) for wheel in wheels
    ]
    sdist_identities = [
        validate_sdist(sdist, canonical, violations) for sdist in sdists
    ]
    wheel_distributions = [
        identity[0] for identity in wheel_identities if identity is not None
    ]
    sdist_distributions = [
        identity[0] for identity in sdist_identities if identity is not None
    ]
    if frozenset(wheel_distributions) != EXPECTED_DISTRIBUTIONS or len(
        wheel_distributions
    ) != len(EXPECTED_DISTRIBUTIONS):
        violations.append(
            f"{artifacts.as_posix()}: wheels must contain all four distributions"
        )
    if frozenset(sdist_distributions) != EXPECTED_DISTRIBUTIONS or len(
        sdist_distributions
    ) != len(EXPECTED_DISTRIBUTIONS):
        violations.append(
            f"{artifacts.as_posix()}: sdists must contain all four distributions"
        )
    wheel_versions = {
        distribution: version
        for identity in wheel_identities
        if identity is not None
        for distribution, version in (identity,)
    }
    sdist_versions = {
        distribution: version
        for identity in sdist_identities
        if identity is not None
        for distribution, version in (identity,)
    }
    source_versions = source_distribution_versions(root, violations)
    for distribution in sorted(EXPECTED_DISTRIBUTIONS):
        if (
            distribution in wheel_versions
            and distribution in sdist_versions
            and wheel_versions[distribution] != sdist_versions[distribution]
        ):
            violations.append(
                f"{artifacts.as_posix()}: wheel and sdist versions differ for {distribution}"
            )
        expected_version = source_versions.get(distribution)
        for archive_kind, archive_versions in (
            ("wheel", wheel_versions),
            ("sdist", sdist_versions),
        ):
            actual_version = archive_versions.get(distribution)
            if (
                expected_version is not None
                and actual_version is not None
                and not artifact_version_matches(
                    distribution, actual_version, expected_version
                )
            ):
                violations.append(
                    f"{artifacts.as_posix()}: {archive_kind} version must match "
                    f"source version for {distribution}"
                )


def validate(root: Path, artifacts: Path | None) -> list[str]:
    violations: list[str] = []
    try:
        paths = tracked_paths(root)
    except RuntimeError as error:
        return [str(error)]
    with tempfile.TemporaryDirectory(prefix="attribution-license-index.") as directory:
        index_root = Path(directory)
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "checkout-index",
                "--all",
                f"--prefix={index_root}/",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip()
            return [f"cannot materialize tracked files from the Git index: {detail}"]
        validate_license_files(index_root, paths, violations)
        validate_headers_and_relics(index_root, paths, violations)
        validate_package_metadata(index_root, violations)
        validate_documentation_and_delivery(index_root, violations)
        if artifacts is not None:
            validate_artifacts(index_root, artifacts, violations)
    return sorted(set(violations))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    violations = validate(root, args.artifacts)
    if violations:
        for violation in violations:
            print(f"license: {violation}", file=sys.stderr)
        return 1
    scope = "repository and artifacts" if args.artifacts else "repository"
    print(f"license: {scope} compliance verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
