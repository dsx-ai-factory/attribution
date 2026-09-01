#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Fail if the committed OpenAPI spec and client no longer match the service
# routes. Both are outputs of generate-client.sh, so a fresh regeneration should
# reproduce the committed files byte for byte, else it means route changes were
# introduced without regenerating or generated code was hand-edited.
#
# Client is generated into a scratch directory and compared against HEAD so the
# verdict never depends on what else has already touched the checkout.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}/service/service-app"

scratch="$(mktemp -d "${TMPDIR:-/tmp}/attribution-client-drift.XXXXXX")"
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT

committed="${scratch}/committed"
generated="${scratch}/generated"
mkdir -p "${committed}" "${generated}"

# Place the committed spec and client in the same layout so they can be diffed.
git archive HEAD \
    -- ':/service/service-app/openapi.json' \
       ':/service/service-app/nvidia_attribution_client' \
    | tar -x -C "${committed}"

./openapi/generate-client.sh "${generated}"

if ! drift="$(cd "${scratch}" && diff -r committed generated 2>&1)"; then
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
