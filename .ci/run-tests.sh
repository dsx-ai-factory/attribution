#!/usr/bin/env bash

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

run_blueprint() {
    (
        cd "${repository_root}/blueprint"
        if [[ "${coverage_enabled}" == "true" ]]; then
            COVERAGE_FILE="${TMPDIR:-/tmp}/blueprint.coverage" \
                uv run --frozen --extra dev --extra web pytest tests -q \
                --cov=attribution_blueprint \
                --cov-report=term-missing:skip-covered \
                --cov-fail-under=88
        else
            uv run --frozen --extra dev --extra web pytest tests -q
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

run_quickstart() {
    (
        cd "${repository_root}"
        bash .ci/validate-quickstart.sh
    )
}

case "${suite}" in
    attribution-lib)
        run_attribution_lib
        ;;
    service-app)
        run_service_app
        ;;
    blueprint)
        run_blueprint
        ;;
    frontend)
        run_frontend
        ;;
    quickstart)
        run_quickstart
        ;;
    all)
        run_quickstart
        run_attribution_lib
        run_service_app
        run_blueprint
        run_frontend
        ;;
    *)
        echo "usage: $0 {all|attribution-lib|service-app|blueprint|frontend|quickstart}" >&2
        exit 2
        ;;
esac
