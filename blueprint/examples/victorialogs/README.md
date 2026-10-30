<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VictoriaLogs Example

This example loads sanitized log records into VictoriaLogs, collects SYSLOG and
APPLOG for a static execution context, and writes attribution results to SQLite.

From the repository root:

```bash
docker run -d --name victorialogs-example -p 127.0.0.1:9428:9428 \
  victoriametrics/victoria-logs:latest -retentionPeriod=100y

curl -X POST \
  -H 'Content-Type: application/stream+json' \
  --data-binary @blueprint/examples/victorialogs/sample-logs.ndjson \
  'http://127.0.0.1:9428/insert/jsonline?_stream_fields=host_name,service_name'
curl 'http://127.0.0.1:9428/metrics' | grep -F 'vl_rows_ingested_total{type="jsonline"}'
curl -X POST 'http://127.0.0.1:9428/internal/force_flush'

docker build -f blueprint/docker/Dockerfile \
  -t attribution-blueprint-web:local .
mkdir -p .attribution-victorialogs
docker run -d --rm --name attribution-blueprint-web-local \
  -e ATTRIBUTION_DB=/data/attributions.db \
  -p 127.0.0.1:8000:8000 -p 127.0.0.1:8080:8080 \
  -v "$PWD/.attribution-victorialogs:/data" \
  attribution-blueprint-web:local

uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --config blueprint/examples/victorialogs/attribution-blueprint.victorialogs.toml \
  --connector output db sqlite path .attribution-victorialogs/victorialogs-output.db
```

The connector uses the bundled LogsQL templates and does not require auth for
the local endpoint. See
[VictoriaLogs configuration](../../docs/CONFIGURATION.md#victorialogs-syslog-or-applog)
for custom queries and authentication settings.
