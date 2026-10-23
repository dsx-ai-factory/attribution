#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Regenerate the OpenAPI client package from the service's own routes.
#
# Run this after any change that alters the HTTP surface: a new route, a
# changed request/response model, or an attribution-lib bump that reshapes a
# model the API returns. The client is committed to the repo, so the resulting
# diff belongs in the same PR as the change that caused it.
#
#     ./openapi/generate-client.sh [--fail-on-change] [output-root]
#
# output-root defaults to this project directory, so the spec and client land
# where they are committed. The drift check passes a scratch directory instead,
# to verify the committed output without rewriting the working tree.
#
# --fail-on-change regenerates in place as usual but exits non-zero if that
# changed anything, so the pre-commit hook repairs the client and still stops the
# commit.
#
# Everything the generator needs is pinned in this project's uv.lock.

set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(dirname "${TOOL_DIR}")"

FAIL_ON_CHANGE=0
if [[ "${1:-}" == "--fail-on-change" ]]; then
    FAIL_ON_CHANGE=1
    shift
fi

PACKAGE_NAME="nvidia_attribution_client"
OUTPUT_ROOT="${1:-${APP_ROOT}}"
CLIENT_DIR="${OUTPUT_ROOT}/${PACKAGE_NAME}"
SPEC="${OUTPUT_ROOT}/openapi.json"
# Generate into a staging directory and swap it in only on success.
STAGING_DIR="${OUTPUT_ROOT}/.openapi-client-staging"

cd "${APP_ROOT}"

# uv refuses to resolve the workspace if the client is missing its pyproject.toml.
# Reseed it so a half-finished earlier run, or a manually deleted client directory,
# does not wedge the script.
if [[ ! -f "${CLIENT_DIR}/pyproject.toml" ]]; then
    echo "==> reseeding ${CLIENT_DIR}/pyproject.toml"
    mkdir -p "${CLIENT_DIR}"
    cp "${TOOL_DIR}/extras/pyproject.toml" "${CLIENT_DIR}/pyproject.toml"
fi

cleanup() { rm -rf "${STAGING_DIR}"; }
trap cleanup EXIT

# The client's version tracks the service's. Read it straight out of the source
# of truth rather than shelling out to a version tool, so this works offline.
VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' app/__init__.py | head -1)"
if [[ -z "${VERSION}" ]]; then
    echo "error: no __version__ found in ${APP_ROOT}/app/__init__.py" >&2
    exit 1
fi
echo "==> version ${VERSION}"

echo "==> extracting OpenAPI schema"
PYTHONHASHSEED=0 uv run --frozen python "${TOOL_DIR}/extract_spec.py" "${SPEC}"

echo "==> generating client"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# .openapi-generator-ignore is only honoured in the output directory, so it has
# to be seeded before the generator runs.
# extras/ holds every file that is hand-maintained and need to be copied first.
cp "${TOOL_DIR}/generator-ignore" "${STAGING_DIR}/.openapi-generator-ignore"
cp "${TOOL_DIR}"/extras/* "${STAGING_DIR}/"

uv run --frozen openapi-generator-cli generate \
    --generator-name python \
    --template-dir "${TOOL_DIR}/templates" \
    --input-spec "${SPEC}" \
    --output "${STAGING_DIR}" \
    --additional-properties=packageName="${PACKAGE_NAME}" \
    --additional-properties=packageVersion="${VERSION}"

# Replace wholesale so files deleted from the API surface disappear from the
# tree instead of lingering as orphans.
rm -rf "${CLIENT_DIR}"
mv "${STAGING_DIR}" "${CLIENT_DIR}"

echo "==> done"

# Hooks inherit GIT_DIR without GIT_WORK_TREE, which makes the calls below treat
# this subdirectory as the repository root and report every file as untracked.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE

# Only meaningful when the output landed in the repo; a scratch run has nothing
# for git to compare against.
if [[ "${OUTPUT_ROOT}" == "${APP_ROOT}" ]] && git -C "${APP_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    CHANGED="$(
        git -C "${APP_ROOT}" diff --name-only -- "${CLIENT_DIR}" "${SPEC}"
        git -C "${APP_ROOT}" ls-files --others --exclude-standard -- "${CLIENT_DIR}" "${SPEC}"
    )"
    if [[ -z "${CHANGED}" ]]; then
        echo "no changes; client was already up to date"
    elif (( FAIL_ON_CHANGE )); then
        echo "error: the client was out of date and has been regenerated. Stage it:" >&2
        echo "${CHANGED}" >&2
        exit 1
    else
        echo "changed:"
        echo "${CHANGED}"
    fi
fi
