# Service Operator Guide

This guide is for the person responsible for running the attribution service,
the blueprint web UI, and the SQLite database used by the read-side dashboard.

The service runs attribution rules. The blueprint CLI runs separately when it
discovers workloads, collects evidence, calls the service, and writes results.
The web UI reads the SQLite database written by the blueprint CLI.

## Runtime Components

| Component | Purpose | Default |
|---|---|---|
| Attribution service | HTTP API for attribution rules | `0.0.0.0:8000` in Docker |
| Web UI | Read-only dashboard over SQLite | `0.0.0.0:8080` in Docker |
| SQLite database | Stores execution contexts and attribution runs | `/data/attributions.db` in Docker |

The bundled Docker entrypoint starts the service first, waits for
`/v2/healthz`, then starts the web UI against the configured database.

## Build The Image

Run from the release bundle root:

```bash
export ATTRIBUTION_IMAGE="attribution-blueprint-web:local"

docker build \
  -f blueprint/docker/Dockerfile \
  -t "$ATTRIBUTION_IMAGE" \
  "$PWD"
```

The Dockerfile expects the release bundle layout: the service application under
`service/service-app/`, the blueprint under `blueprint/`, and the attribution
library under `service/attribution-lib/`.

### Service-only image

To run only the attribution service, without the web UI or SQLite database,
build the service image from the same bundle root:

```bash
docker build \
  -f service/service-app/Dockerfile \
  -t attribution-service:local \
  "$PWD"

docker run -d --rm \
  --name attribution-service \
  -p 127.0.0.1:8000:8000 \
  attribution-service:local
```

This image listens on port `8000` and does not read `ATTRIBUTION_DB`,
`ATTRIBUTION_SERVICE_PORT`, or `ATTRIBUTION_WEB_PORT`. Use it for a shared
attribution endpoint that Blueprint runs elsewhere call over HTTP. Pass the
service variables in [Service environment variables](#service-environment-variables)
with `-e` as needed.

The `BASE_IMAGE` build argument sets only the final runtime image. The build
stage still pulls `python:3.14.6-slim-trixie` and `ghcr.io/astral-sh/uv:0.10.12`,
installs Debian packages with `apt-get`, and downloads Python dependencies with
`uv`. For a build with no internet access, mirror those images and package
sources as well, and point the build at them.

## Run The Container

Persist the database on the host:

```bash
export ATTRIBUTION_CONTAINER="attribution-blueprint-web"
export ATTRIBUTION_SERVICE_PORT="8000"
export ATTRIBUTION_WEB_PORT="8080"
export ATTRIBUTION_DB=".attribution-data/attributions.db"

mkdir -p "$(dirname "$ATTRIBUTION_DB")"

docker run -d --rm \
  --name "$ATTRIBUTION_CONTAINER" \
  -e ATTRIBUTION_DB=/data/attributions.db \
  -e ATTRIBUTION_SERVICE_PORT \
  -e ATTRIBUTION_WEB_PORT \
  -p "127.0.0.1:$ATTRIBUTION_SERVICE_PORT:$ATTRIBUTION_SERVICE_PORT" \
  -p "127.0.0.1:$ATTRIBUTION_WEB_PORT:$ATTRIBUTION_WEB_PORT" \
  -v "$PWD/$(dirname "$ATTRIBUTION_DB"):/data" \
  "$ATTRIBUTION_IMAGE"
```

Use a different host directory for each environment. The container creates the
SQLite schema when the database file does not exist.

## Environment Variables

| Variable | Default | Meaning |
|---|---|---|
| `ATTRIBUTION_DB` | `/data/attributions.db` | SQLite path used by the web UI |
| `ATTRIBUTION_SERVICE_PORT` | `8000` | Service listen port inside the container |
| `ATTRIBUTION_WEB_PORT` | `8080` | Web UI listen port inside the container |

When publishing ports with Docker, map the host port to the same container port
you set through the environment variable.

### DuckDB Event Storage

These settings apply per attributor, so allow for concurrent attributors when sizing storage:

| Variable | Default | Meaning |
|---|---|---|
| `ATTRIBUTOR_ALLOW_DISK_SPILL` | `false` | Set to `true` to enable spilling to disk |
| `ATTRIBUTOR_DUCKDB_MEMORY_LIMIT` | `128MB` | DuckDB memory limit |
| `ATTRIBUTOR_DUCKDB_TEMP_ROOT` | System temporary directory | Writable spill root; each attributor gets a private subdirectory |
| `ATTRIBUTOR_DUCKDB_MAX_TEMP_DIRECTORY_SIZE` | DuckDB default | Maximum spill storage, e.g. `1GB` |

Blank roots and spill limits use their defaults; both are ignored when spilling
is disabled. The system temporary directory honors `TMPDIR`. Private directories
are removed on close; failed cleanup is retained for retry while the service runs.
Forced termination can leave orphaned directories.

Sizes take a number and unit: `128MB`, `128MiB`, `1.5GB`, or `128 MB`.
Units are case-insensitive: `B`, `KB`, `MB`, `GB`, `TB` use powers of 1000;
`KiB`, `MiB`, `GiB`, `TiB` use powers of 1024. Negative values mean unlimited.
A blank memory limit is invalid.

The service checks these settings with DuckDB during startup. An invalid memory
limit, or an invalid spill capacity when spilling is enabled, fails startup
before the service accepts requests. The startup error names the environment
variable. Correct it and restart; for example, use `1GiB` instead of `1Gi`.
Spill capacity remains ignored when spilling is disabled.

Attributor creation also validates the settings. If configuration changes after
startup, an invalid size returns HTTP 500 with a `detail` naming the server
setting and giving valid examples; no attributor is registered.

<a id="service-environment-variables"></a>

### Service environment variables

The attribution service also reads these variables in both the bundled image and
the service-only image. The DuckDB event storage variables are described in
[DuckDB Event Storage](#duckdb-event-storage).

| Variable | Default | Meaning |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Service log level. An unrecognized value falls back to `INFO`. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | OTLP gRPC endpoint for log export. When unset, logs go to stdout only. |
| `OTEL_EXPORTER_OTLP_TOKEN` | unset | Optional bearer token sent with exported logs. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | unset | Ignored with a warning. The service exports over gRPC only. |

With `ATTRIBUTOR_ALLOW_DISK_SPILL` at `false`, an attributor that exceeds
`ATTRIBUTOR_DUCKDB_MEMORY_LIMIT` fails with an event-storage overflow error
scoped to that one request, not a generic 500. Enable disk spill only where the
container has writable local disk.

The Blueprint CLI has its own log level, `ATTRIBUTION_LOG_LEVEL` (default
`INFO`). See [CONFIGURATION.md](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/CONFIGURATION.md).

## Health Checks

Check the service:

```bash
curl -fsS "http://127.0.0.1:${ATTRIBUTION_SERVICE_PORT}/v2/healthz"
```

Check the web UI:

```bash
curl -fsS "http://127.0.0.1:${ATTRIBUTION_WEB_PORT}/" >/dev/null
```

The service must become healthy before the web UI starts. If the container exits
early, inspect logs:

```bash
docker logs "$ATTRIBUTION_CONTAINER"
```

## Run Attribution Against The Service

From the release bundle root, point the blueprint CLI at the running service and
the persisted database:

```bash
export ATTRIBUTION_SERVICE_URL="http://127.0.0.1:${ATTRIBUTION_SERVICE_PORT}"

uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --configs blueprint/examples/static/configs \
  --connector attribution service_url "$ATTRIBUTION_SERVICE_URL" \
  --connector output db sqlite path "$ATTRIBUTION_DB"
```

For a real deployment, replace the sample configs with the environment-specific
TOML described in [CONFIGURATION.md](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/CONFIGURATION.md).

## Data Management

SQLite is the system of record for the bundled web UI. Treat the mounted
database directory as persistent service data.

For a simple backup, stop writes first or take a filesystem snapshot of the
mounted database directory. SQLite may create `attributions.db-wal` and
`attributions.db-shm`; keep those files with the main database when they exist.

To reset the sample database:

```bash
docker rm -f "$ATTRIBUTION_CONTAINER"
rm -f "$ATTRIBUTION_DB" "$ATTRIBUTION_DB-wal" "$ATTRIBUTION_DB-shm"
```

The next container start recreates an empty schema.

## Upgrade And Rollback

Use a versioned image tag for each release:

```bash
docker build \
  -f blueprint/docker/Dockerfile \
  -t "attribution-blueprint-web:${version}" \
  "$PWD"
```

Before upgrading:

- stop blueprint CLI runs that are writing to the database
- back up the database files
- start the new container against the existing mounted database
- verify service health and the web UI
- run a small attribution job or the static sample

For rollback, stop the new container and start the previous image tag against
the same database backup. If the new version changed the SQLite schema, restore
the backup taken before the upgrade.

## Common Failures

| Symptom | First check |
|---|---|
| Container exits before the web UI starts | `docker logs "$ATTRIBUTION_CONTAINER"` and service health |
| Service health endpoint fails | Port mapping, service startup logs, and whether another process uses the host port |
| Web UI returns database errors | Mounted database path, file ownership, and whether the database was created by the current release |
| Blueprint CLI cannot reach the service | `ATTRIBUTION_SERVICE_URL`, firewall rules, and `curl -fsS "$ATTRIBUTION_SERVICE_URL/v2/healthz"` |
| No rows appear in the web UI | Confirm the CLI wrote to the same host `ATTRIBUTION_DB` mounted into the container |

## Related Docs

- [quickstart.md](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/quickstart.md), sample Docker workflow
- [README.md](https://github.com/dsx-ai-factory/attribution/blob/main/README.md), end-user workflow
- [CONFIGURATION.md](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/CONFIGURATION.md), connector configuration
- [SCHEDULER_REQUIREMENTS.md](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/SCHEDULER_REQUIREMENTS.md), scheduler access checklist
