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

The Dockerfile expects the release bundle layout: service source at the bundle
root, the blueprint under `blueprint/`, and the attribution library under
`service/attribution-lib/`.

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
TOML described in [CONFIGURATION.md](../blueprint/docs/CONFIGURATION.md).

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

- [quickstart.md](../blueprint/docs/quickstart.md), sample Docker workflow
- [README.md](../README.md), end-user workflow
- [CONFIGURATION.md](../blueprint/docs/CONFIGURATION.md), connector configuration
- [SCHEDULER_REQUIREMENTS.md](../blueprint/docs/SCHEDULER_REQUIREMENTS.md), scheduler access checklist
