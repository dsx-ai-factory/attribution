# Attribution Blueprint — Web UI

React (Vite + TypeScript) dashboard for the NVIDIA Attribution Blueprint. It renders the
attribution verdicts stored in the blueprint's read-only SQLite database, served by the same
FastAPI process (`nvidia-attribution-blueprint-web`).

## Architecture

- Pure-render components + a single data hook, `useRuns()` — no mock data.
- Data arrives as a server-injected bootstrap blob (`#attr-bootstrap`), read through a swappable
  `AttributionDataSource` (Bootstrap by default; an `/api/v1` `HttpDataSource` is also available).
- Routes: `/` Fleet · `/verdict/:id` · `/node/:id`.

## Local development

From the repo root (no `cd` needed):

```bash
scripts/webui.sh             # build the SPA + serve real data on :8080 (auto-seeds demo.db)
scripts/webui.sh --db x.db   # serve your own SQLite DB
scripts/webui.sh --port 9000 # custom port
scripts/webui.sh dev         # frontend hot-reload dev server (empty data, no backend)
```

Frontend-only commands run from this directory: `pnpm build`, `pnpm test`, `pnpm exec tsc -b`.

## Tech

Vite 8 · React 19 · TypeScript · Tailwind v4 · shadcn/ui · ECharts · Monaco · react-router 7.
