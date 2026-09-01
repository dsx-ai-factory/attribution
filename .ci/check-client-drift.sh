#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fail if the committed OpenAPI spec and client no longer match the service
# routes. Both are outputs of generate-client.sh, so a fresh regeneration should
# reproduce the committed files byte for byte, else it means route changes were
# introduced without regenerating or generated code was hand-edited.
#
# Client is generated into a scratch directory and compared against the exact
# Git index so the pre-commit verdict covers the content that will be committed
# and never includes unstaged or untracked local data.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

scratch="$(mktemp -d "${TMPDIR:-/tmp}/attribution-client-drift.XXXXXX")"
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT

source_root="${scratch}/source"
generated="${scratch}/generated"
mkdir -p "${source_root}" "${generated}"

# Run the generator from a full snapshot of the Git index. This makes every
# route, template, extra, lock file, and version source match the staged output
# that the check compares, even when the worktree contains unstaged changes.
index_tree="$(git -C "${repository_root}" write-tree)"
git -C "${repository_root}" archive "${index_tree}" | tar -x -C "${source_root}"
indexed_service_root="${source_root}/service/service-app"

(
    cd "${indexed_service_root}"
    ./openapi/generate-client.sh "${generated}"
)

if ! drift="$({
    diff -r \
        "${indexed_service_root}/nvidia_attribution_client" \
        "${generated}/nvidia_attribution_client"
    diff "${indexed_service_root}/openapi.json" "${generated}/openapi.json"
} 2>&1)"; then
    {
        echo "error: the committed spec and client do not match a fresh regeneration."
        echo "There may have been changes to a route, a template under openapi/, "
        echo "a file under openapi/extras/, the pinned generator version, or "
        echo "generated code was manually edited."
        echo "Run service/service-app/openapi/generate-client.sh and commit the result."
        echo "${drift}"
    } >&2
    exit 1
fi

echo "client is in sync with the OpenAPI spec"
