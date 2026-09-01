# `openapi/`

Tooling that generates `nvidia_attribution_client/` from the service's own
routes.

## Regenerating the API client

The client is committed to the repository. Regenerate it whenever the HTTP
surface changes: a new route, a changed request or response model, or a
dependency bump that reshapes a model the API returns:

```bash
cd service/service-app
./openapi/generate-client.sh
```

Commit the resulting client diff alongside the change that caused it.
`openapi.json` is committed too, so a spec change is reviewable on its own.

The generator (`openapi-generator-cli`) version is pinned by `uv.lock`, so two
runs of the script produce byte-identical output. `.ci/check-client-drift.sh`
regenerates in CI and fails if the committed tree moves.

The client package's own README (shipped in the distribution) lives in
`extras/README.md` and documents *using* the client; this file documents
*producing* it.

## Layout

| Path | Purpose |
|---|---|
| `generate-client.sh` | Entry point described above. |
| `extract_spec.py` | Dumps the OpenAPI schema without serving the app. |
| `generator-ignore` | Seeded into the client as `.openapi-generator-ignore`. The generator only honours that file inside its output directory, so it has to be copied in before each run. |
| `extras/` | Hand-maintained client files (`pyproject.toml`, `README.md`, `LICENSE`) copied in verbatim before generation. Paired with `generator-ignore`, which stops the generator overwriting them. |
| `templates/` | Mustache overrides. Each opens with a `{{! ... }}` banner explaining what changed and why. |

These templates are derived from
[openapi-generator](https://github.com/OpenAPITools/openapi-generator), which
is licensed under Apache-2.0; a copy of that license is in `templates/LICENSE`.
Code *generated* from the templates carries no such obligation; only the
templates themselves are derivative works.
