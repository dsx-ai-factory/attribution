# Attribution Service REST API

Use the REST API when another orchestrator already has the workload context
and evidence. The service does not query schedulers, Kubernetes, logs, or
metrics. A client creates one temporary attributor, submits zero or more
observations, requests the verdict, and deletes the attributor.

The Blueprint implements this same lifecycle for every discovered execution
context. Its client uses the v2 routes documented here.

## Prerequisites

Start the Attribution Service and set its base URL. The commands below also
require `curl` and `jq`.

```bash
export SVC="http://127.0.0.1:8000"
curl --fail-with-body --silent --show-error "${SVC}/v2/healthz"
```

The Service API has no built-in authentication, and the Blueprint client does
not attach credentials. Keep access restricted, or place remote deployments
behind an authenticated ingress, gateway, or sidecar. See the
[authentication FAQ](../../docs/FAQ.md#how-does-the-blueprint-authenticate-against-the-service).

All timestamps must include a UTC offset. Set `workload.timezone` to the IANA
timezone used by the workload, such as `UTC` or `America/Los_Angeles`. Node
names should be lowercase, every node's rank list must have the same length,
and every rank must appear in the explicit `nodes` map on exactly one node.

## 1. Create an attributor

```bash
export ATT="$(
  curl --fail-with-body --silent --show-error \
    --request POST \
    --header 'Content-Type: application/json' \
    --data @- \
    "${SVC}/v2/attributor" <<'JSON' | jq -er '.attributor_id'
{
  "workload": {
    "id": "rest-example-001",
    "type": "kueue",
    "job_start_time": "2026-07-14T10:00:00Z",
    "timezone": "UTC",
    "job_end_time": "2026-07-14T10:30:00Z",
    "interest_start_time": "2026-07-14T10:00:00Z",
    "interest_end_time": "2026-07-14T10:30:00Z",
    "status": "COMPLETED",
    "nodes": {
      "node-a": [0, 1, 2, 3],
      "node-b": [4, 5, 6, 7]
    },
    "name": "rest-example",
    "username": "example-user",
    "framework": "pytorch"
  },
  "metadata": {
    "cluster": "example-cluster",
    "agent": "rest-example-client",
    "tenant": "example-tenant",
    "ruleset": "default"
  }
}
JSON
)"

printf 'attributor_id=%s\n' "${ATT}"
```

V2 accepts any string for `workload.type`. The `status` value is also passed
to the rules engine as a string. When no stronger evidence exists,
`COMPLETED` produces the successful fallback symptom. Other unrecognized
states produce the failed fallback symptom.

Creating an identical attributor while the first one still exists returns
HTTP 409 (`DuplicateAttributorError`). The response's `att_id` identifies the
existing session; it does not establish that the session is stale or safe to
delete. The Blueprint preserves that session and its accumulated evidence and
rejects the duplicate creation. See the
[Blueprint session flow](../../docs/ARCHITECTURE.md#end-to-end-flow-for-one-workload)
for conflict reporting.

Serialize runs with identical workload information against the same Service.
Retry after the existing session's owner finishes and deletes it. If a caller
crashed, delete its session only after confirming that no caller still uses it,
or wait for the Service to expire it after 24 hours of inactivity (checked
hourly, normally about 24–25 hours after the last activity). Normal sequential
Blueprint runs clean up their own sessions and can be repeated. Recovery before
expiry requires an operator to explicitly delete the session; ownership-aware
recovery would require a separate Service lifecycle change.

## 2. Submit an observation

The outer observation is JSON, but `body` is always a string. For
`entry_timeseries`, that string is a JSON array with one object per log line:
* `body` (required)
* `ts` (unix nanoseconds; optional in the schema, but `syslog`, `health_check` and `csp_event` drop an entry without it)
* `resources` (`[{"kind": "node", "hostname": ...}]`, optionally with `{"kind": "replica", "name": ...}` objects)

[Observation entry format](../../service/service-app/docs/api/observation-format.md) is the authority
on these fields: what each source does when `ts` or the node is missing, what
it parses out of the body instead, and why a 201 does not mean the entries were
used. A `null` bound in `time_interval` is unbounded on that side; **entries
outside the declared interval are dropped.**

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header 'Content-Type: application/json' \
  --data @- \
  "${SVC}/v2/attributor/${ATT}/observation" <<'JSON'
{
  "context": {
    "time_interval": {
      "start": null,
      "end": null
    },
    "source": "applog"
  },
  "body": "[{\"body\": \"[rank 0] Main process ended cleanly\"}]"
}
JSON
```

Every bundled Blueprint connector sends `entry_timeseries`: the log connectors
for `applog` and `syslog`, and the PromQL connector for `node_metrics`. A
`node_metrics` entry must declare its node in `resources`; the service never
parses one out of a metric body.

`body` is a JSON-encoded string containing the entry array. The service
rejects an invalid body with HTTP 400. Each distinct observation may be submitted
once to an attributor, and a duplicate returns HTTP 409.

## 3. Request attribution

```bash
curl --fail-with-body --silent --show-error \
  "${SVC}/v2/attributor/${ATT}/attribution" | jq .
```

The v2 response contains:

- `verdict`, the top-level outcome, detail, and root symptom
- `symptoms`, a symptom-to-evidence-site map
- `observation_context`, the IDs of processed observations
- `collectors`, any rule collector digests
- `model_type`, the response discriminator, `AttributionResponseV2`

These are raw service fields. Blueprint-only fields such as `terminal_state`,
`sub_domain`, `connector_reports`, and `source_channel` are added after the
REST response by `enrich_attribution_result()`.

## 4. Delete the attributor

```bash
curl --fail-with-body --silent --show-error \
  --request DELETE \
  "${SVC}/v2/attributor/${ATT}"
```

A successful delete returns HTTP 204 with no response body. Attributors are
temporary service state, not a result database. Always delete them after
retrieving the result, and only delete sessions you own. The Blueprint cleans
up successfully created sessions in a `finally` block and writes
the enriched result through its configured output connectors.

## API discovery and errors

The running service publishes Swagger UI at `${SVC}/docs` and its OpenAPI
schema at `${SVC}/openapi.json`.

Common responses are:

| Status | Meaning |
|---:|---|
| 201 | Attributor or observation created. |
| 204 | Attributor deleted. |
| 400 | Observation body cannot be ingested. |
| 404 | Attributor ID is unknown or has expired. |
| 409 | Duplicate attributor or observation. |
| 422 | Request fields, timestamps, or workload and rule values fail v2 validation. |
