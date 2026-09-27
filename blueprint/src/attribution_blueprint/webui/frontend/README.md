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

## Symptom documentation references

Set `VITE_SYMPTOM_DOCS_URL` at build time to explicitly enable the existing symptom
reference API. There is no default endpoint; missing or invalid configuration
disables documentation requests and hides Doc links. Use an absolute HTTPS URL
without credentials. The endpoint must be reachable from the browser and allow
the WebUI origin through CORS. Configure the endpoint directly; redirects are
rejected so they cannot change its transport.

For a local preview on a trusted network, an explicitly configured HTTP endpoint
is also accepted when the WebUI browser hostname is exactly `localhost`,
`127.0.0.1`, or `[::1]`. This applies to built previews as well as the development
server. Deployed WebUIs require HTTPS for the reference API; the frontend never
rewrites its URL or proxies the request.

Doc links use the raw recorded symptom as an exact key in
`response.symptoms[symptom]`. The reference API owns this mapping; the returned
record's canonical `id` can differ from the lookup key. Its `id` and `title` must
be nonempty strings, and its `doc_url` is used unchanged. There is no bundled
catalog, name normalization, alias search, slug construction, URL repair, root-ID
fallback, or WebUI backend proxy. Missing exact keys, malformed records, and
invalid links leave Doc hidden. API failures leave attribution and evidence
available; Doc stays hidden unless a previous API response is cached.
Only HTTPS document URLs without credentials are accepted, without checking whether the target
page exists. Incorrect destination paths must be fixed by the reference API owner.

Doc appears beside each symptom name in the Symptoms table and beside the
selected symptom in the graph's Step details panel. Graph selection updates the
link using that step's own symptom; neither location borrows the root symptom's
`catalog_id`. No rule documentation is inferred.
The reference response is shared through SWR, with focus refresh and automatic
error retries disabled; requests time out after ten seconds.

## Observation IDs

The WebUI serializes source observation IDs as decimal strings in both HTTP run
details and the injected bootstrap, preserving 64-bit IDs before they reach the
browser. Stored Service evidence is unchanged, and numeric quantities remain
numbers. The frontend uses ordinary JSON parsing without a custom reviver.

## Tech

Vite 8 · React 19 · TypeScript · Tailwind v4 · shadcn/ui · ECharts · Monaco · react-router 7.
