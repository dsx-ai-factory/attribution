#!/usr/bin/env bash
# One-command local Attribution web UI.
#
# Builds the React SPA and serves it via FastAPI against a real SQLite DB,
# exercising the real bootstrap-injection link (not the empty `dev` shell).
# Run from anywhere; no cd.
#
#   scripts/webui.sh              build SPA + serve real data on :8080 (auto-seeds demo.db)
#   scripts/webui.sh --db x.db    serve your own SQLite DB
#   scripts/webui.sh --port 9000  custom port
#   scripts/webui.sh --no-build   skip the SPA rebuild, just serve
#   scripts/webui.sh dev          frontend hot-reload dev server (empty data, no backend)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND="$ROOT/src/attribution_blueprint/webui/frontend"
DEFAULT_DB="$ROOT/demo.db"
DB="$DEFAULT_DB"
PORT=8080
BUILD=1
MODE=serve
PNPM=()

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; }

die() {
  echo "error: $*" >&2
  exit 2
}

resolve_pnpm() {
  if command -v pnpm >/dev/null 2>&1; then
    PNPM=(pnpm)
  elif command -v corepack >/dev/null 2>&1; then
    PNPM=(corepack pnpm)
  else
    die "pnpm is required to build the web UI. Install pnpm or enable corepack, then rerun. Use --no-build only after assets exist."
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    dev) MODE=dev; shift ;;
    serve) MODE=serve; shift ;;
    --db)
      [[ $# -ge 2 ]] || die "--db requires a path"
      DB="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port requires a port"
      [[ "$2" =~ ^[0-9]+$ ]] || die "--port requires a numeric port, got: $2"
      PORT="$2"
      shift 2
      ;;
    --no-build) BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$MODE" == dev ]]; then
  resolve_pnpm
  echo ">> frontend dev server (hot reload, empty data, use plain 'serve' for real data)"
  exec "${PNPM[@]}" --dir "$FRONTEND" dev
fi

if [[ "$DB" != "$DEFAULT_DB" && ! -f "$DB" ]]; then
  die "sqlite db not found: $DB"
fi

if [[ "$BUILD" == 1 ]]; then
  resolve_pnpm
  echo ">> building SPA -> webui/static/"
  "${PNPM[@]}" --dir "$FRONTEND" install --frozen-lockfile
  "${PNPM[@]}" --dir "$FRONTEND" build
fi

if [[ "$DB" == "$DEFAULT_DB" && ! -f "$DB" ]]; then
  echo ">> seeding demo DB -> $DB"
  uv run --project "$ROOT" python "$ROOT/scripts/seed_demo_db.py" --out "$DB"
fi

echo ">> serving http://127.0.0.1:$PORT  (db: $DB)"
exec uv run --project "$ROOT" --extra web nvidia-attribution-blueprint-web --db "$DB" --port "$PORT"
