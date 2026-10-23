#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Print a CHANGELOG.md skeleton for one component: the heading for its committed
# version literal, the section headings, and one line per first-parent commit
# since the component's latest tag, for the releaser to sort into the sections.

set -euo pipefail

component="${1:-}"
case "${component}" in
    attribution-lib)
        path="service/attribution-lib"
        literal_file="${path}/src/attribution_lib/__init__.py"
        ;;
    service-app)
        path="service/service-app"
        literal_file="${path}/app/__init__.py"
        ;;
    blueprint)
        path="blueprint"
        literal_file="${path}/src/attribution_blueprint/__init__.py"
        ;;
    *)
        echo "usage: $0 {attribution-lib|service-app|blueprint}" >&2
        exit 2
        ;;
esac

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

literal="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "${literal_file}" | head -1)"
if [[ -z "${literal}" ]]; then
    echo "error: no __version__ found in ${literal_file}" >&2
    exit 1
fi

# sort -V orders -rc.1 after the final of the same number, unlike release.py;
# acceptable for a paste helper.
tag="$(git tag --merged HEAD --list "${component}-v*" | sort -V | tail -1)"
range="HEAD"
if [[ -n "${tag}" ]]; then
    range="${tag}..HEAD"
fi

printf '## [%s] - %s\n\n' "${literal}" "$(date +%F)"
printf '### ⚠ BREAKING CHANGES\n\n### Features\n\n### Bug Fixes\n\n'
echo "<!-- git log --first-parent --format='- %s' ${range} -- ${path} -->"
git log --first-parent --format='- %s' "${range}" -- "${path}"
