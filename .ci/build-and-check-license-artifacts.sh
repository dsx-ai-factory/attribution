#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="$(mktemp -d "${TMPDIR:-/tmp}/attribution-license-artifacts.XXXXXX")"
source_root="${artifact_root}/source"
artifacts="${artifact_root}/artifacts"
source_state_before="${artifact_root}/source-state-before"
source_state_after="${artifact_root}/source-state-after"

cleanup() {
    local status=$?
    if [[ -n "${artifact_root}" && "${artifact_root}" == "${TMPDIR:-/tmp}"/attribution-license-artifacts.* ]]; then
        rm -rf -- "${artifact_root}"
    fi
    exit "${status}"
}
trap cleanup EXIT

cd "${repository_root}"

record_source_state() {
    git diff --no-ext-diff --binary HEAD -- blueprint service/attribution-lib service/service-app
    git status --porcelain=v1 --ignored=matching --untracked-files=all -- \
        blueprint service/attribution-lib service/service-app
    find blueprint service/attribution-lib service/service-app \
        \( -name .venv -o -name node_modules \) -prune -o -type f \
        -exec shasum -a 256 {} + | LC_ALL=C sort
}

record_source_state >"${source_state_before}"

# Build from the exact Git index in a temporary directory. This prevents
# unstaged or ignored local content from entering a published artifact and
# keeps package-manager residue out of the checkout.
mkdir -p "${source_root}" "${artifacts}"
git checkout-index --all --prefix="${source_root}/"
git -C "${source_root}" init --quiet
git -C "${source_root}" add --all

frontend_root="${source_root}/blueprint/src/attribution_blueprint/webui/frontend"
(
    cd "${frontend_root}"
    pnpm install --frozen-lockfile
    pnpm build
)

uv build "${source_root}/blueprint" --out-dir "${artifacts}/blueprint"
uv build "${source_root}/service/attribution-lib" --out-dir "${artifacts}/attribution-lib"
uv build "${source_root}/service/service-app" \
    --package nvidia-attribution-service \
    --out-dir "${artifacts}/service-app"
uv build "${source_root}/service/service-app/nvidia_attribution_client" \
    --out-dir "${artifacts}/client"
uv run --python 3.14.6 --no-project python "${source_root}/.ci/check-license-compliance.py" \
    --root "${source_root}" \
    --artifacts "${artifacts}"

record_source_state >"${source_state_after}"
if ! cmp -s "${source_state_before}" "${source_state_after}"; then
    echo "source tree changed during artifact build" >&2
    diff -u "${source_state_before}" "${source_state_after}" >&2 || true
    exit 1
fi
