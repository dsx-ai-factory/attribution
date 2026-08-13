# Quickstart

Run the attribution service and web UI in Docker, attribute sample AI factory
evidence, then view the results.

## Prerequisites

Use macOS Terminal, a Linux shell, or another POSIX-style shell for the
commands below.

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Install Docker using the
[Docker installation instructions](https://docs.docker.com/get-started/get-docker/).
On macOS, use Docker Desktop. On Linux, use Docker Engine or Docker Desktop.
Then start Docker and check that the CLI can reach it:

```bash
docker version
```

## Prepare Workspace

Clone the repository and change into it:

```bash
git clone <repository-url> nvidia-attribution
cd nvidia-attribution
```

The bundle contains the attribution service under `service/service-app/`, the
bundled attribution library under `service/attribution-lib/`, and the blueprint
reference implementation under `blueprint/`.

This quickstart assumes you are running commands from the root directory of the attribution service bundle.

## Set Variables

```bash
export ATTRIBUTION_IMAGE="attribution-blueprint-web:quickstart"
export ATTRIBUTION_CONTAINER="attribution-blueprint-web-local"
export ATTRIBUTION_SERVICE_PORT="8000"
export ATTRIBUTION_WEB_PORT="8080"
export ATTRIBUTION_SERVICE_URL="http://127.0.0.1:$ATTRIBUTION_SERVICE_PORT"
export ATTRIBUTION_DB=".attribution-sample/attributions.db"
```

## Start Docker

The Docker image builds the React UI from `blueprint/src/attribution_blueprint/webui/frontend`
and serves it from the same container as the web backend. No separate frontend
build step is required.

```bash
docker build \
  -f blueprint/docker/Dockerfile \
  -t "$ATTRIBUTION_IMAGE" \
  "$PWD"

mkdir -p "$(dirname "$ATTRIBUTION_DB")"

docker run -d --rm \
  --name "$ATTRIBUTION_CONTAINER" \
  -e ATTRIBUTION_DB=/data/attributions.db \
  -p "127.0.0.1:$ATTRIBUTION_SERVICE_PORT:8000" \
  -p "127.0.0.1:$ATTRIBUTION_WEB_PORT:8080" \
  -v "$PWD/$(dirname "$ATTRIBUTION_DB"):/data" \
  "$ATTRIBUTION_IMAGE"
```

Check that the containerized attribution service is healthy before running
attribution:

```bash
curl -fsS "http://127.0.0.1:${ATTRIBUTION_SERVICE_PORT}/v2/healthz"
```

Expected output:

```text
null
```

If the check fails, inspect the container logs:

```bash
docker logs "$ATTRIBUTION_CONTAINER"
```

This also helps diagnose port conflicts with an already-running local service.

## Run Attribution

This run uses the committed per-job TOML files under
`blueprint/examples/static/configs`.

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --configs blueprint/examples/static/configs \
  --connector attribution service_url "$ATTRIBUTION_SERVICE_URL" \
  --connector output db sqlite path "$ATTRIBUTION_DB"
```

The command should finish with:

```text
Attribution complete: 118/118 execution contexts succeeded.
```

The web UI shows the outcomes produced by the attribution rules in the service
version you are running. If the page was already open before attribution
finished, refresh it to load the new run data from SQLite.

If `ATTRIBUTION_DB` already exists, the command appends a new attribution run
for each sample job. Use a different database path if you want a separate run
history.

## View Results

Open `http://127.0.0.1:8080`, or the port you set in `ATTRIBUTION_WEB_PORT`.
The attribution service health endpoint remains available on
`http://127.0.0.1:8000/v2/healthz`, or the port you set in
`ATTRIBUTION_SERVICE_PORT`.

## Cleanup

```bash
docker rm -f "$ATTRIBUTION_CONTAINER"
```

## Next Steps

- [README.md](../../README.md), release-bundle overview and end-user workflow
- [examples/static/configs](../examples/static/configs), static sample configs
- [sample-data/static](./sample-data/static), sample evidence consumed by the configs
