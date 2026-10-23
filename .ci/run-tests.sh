#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
suite="${1:-all}"
coverage_enabled="${COVERAGE:-false}"

run_attribution_lib() {
    (
        cd "${repository_root}/service/attribution-lib"
        if [[ "${coverage_enabled}" == "true" ]]; then
            COVERAGE_FILE="${TMPDIR:-/tmp}/attribution-lib.coverage" \
                uv run --frozen pytest -n 4 tests -q \
                --cov=attribution_lib \
                --cov-report=term-missing:skip-covered \
                --cov-fail-under=89
        else
            uv run --frozen pytest -n 4 tests -q
        fi
    )
}

run_service_app() {
    (
        cd "${repository_root}/service/service-app"
        if [[ "${coverage_enabled}" == "true" ]]; then
            COVERAGE_FILE="${TMPDIR:-/tmp}/service-app.coverage" \
                uv run --frozen pytest tests -q \
                --cov=app \
                --cov-report=term-missing:skip-covered \
                --cov-fail-under=92
        else
            uv run --frozen pytest tests -q
        fi
    )
}

run_service_app_client_drift() {
    (
        cd "${repository_root}"
        bash .ci/check-client-drift.sh
    )
}

run_license_compliance() {
    (
        cd "${repository_root}"
        uv run --project service/attribution-lib --frozen \
            pytest -q .ci/tests/test_license_compliance.py
        uv run --project service/attribution-lib --frozen \
            python .ci/check-license-compliance.py --root .
    )
}

run_blueprint() {
    (
        cd "${repository_root}/blueprint"
        if [[ "${coverage_enabled}" == "true" ]]; then
            COVERAGE_FILE="${TMPDIR:-/tmp}/blueprint.coverage" \
                uv run --frozen --extra dev --extra web --extra pod-placement-recorder pytest tests -q \
                --cov=attribution_blueprint \
                --cov-report=term-missing:skip-covered \
                --cov-fail-under=88
        else
            uv run --frozen --extra dev --extra web --extra pod-placement-recorder pytest tests -q
        fi
    )
}

run_frontend() {
    (
        cd "${repository_root}/blueprint/src/attribution_blueprint/webui/frontend"
        pnpm install --frozen-lockfile
        if [[ "${coverage_enabled}" == "true" ]]; then
            pnpm run test:coverage
        else
            pnpm test
        fi
        pnpm build
    )
}

run_end_to_end() {
    (
        cd "${repository_root}"
        bash .ci/run-end-to-end-tests.sh
    )
}

run_ci_tests() {
    (
        cd "${repository_root}"
        uv run --no-project --with pytest --with packaging pytest .ci/tests -q
    )
}

case "${suite}" in
    attribution-lib)
        run_attribution_lib
        ;;
    service-app)
        run_service_app
        ;;
    service-app-client-drift)
        run_service_app_client_drift
        ;;
    license-compliance)
        run_license_compliance
        ;;
    blueprint)
        run_blueprint
        ;;
    frontend)
        run_frontend
        ;;
    end-to-end)
        run_end_to_end
        ;;
    ci-tests)
        run_ci_tests
        ;;
    all)
        run_end_to_end
        run_attribution_lib
        run_service_app
        run_service_app_client_drift
        run_license_compliance
        run_blueprint
        run_frontend
        run_ci_tests
        ;;
    *)
        echo "usage: $0 {all|attribution-lib|service-app|blueprint|service-app-client-drift|license-compliance|frontend|end-to-end|ci-tests}" >&2
        exit 2
        ;;
esac
