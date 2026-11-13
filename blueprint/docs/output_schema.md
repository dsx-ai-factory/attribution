# Output Schemas

This guide describes SQLite persistence and [Kafka result events](#kafka-result-events).

The Blueprint SQLite output stores execution contexts and attribution runs in a
local database. This document records schema version `1`, implemented by the
SQLite output connector.

The schema has three core tables:

- `schema_version`, records the applied schema version.
- `execution_context`, stores one normalized workload context per scheduler
  job, attempt, and scheduler type.
- `attribution_run`, stores one enriched attribution result for an execution
  context.

There is no `connector_run` table in schema version `1`. Collector coverage
(`connector_reports`), observation delivery (`ingestion_reports`), and the
verdict-level `evidence_status` are stored inside
`attribution_run.raw_result_json` as part of the enriched attribution payload.

## Schema Version

`schema_version` stores the current schema marker.

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version     TEXT PRIMARY KEY,
    description TEXT,
    applied_at  TEXT NOT NULL
);
```

The current version is:

```text
1
```

The current description is:

```text
execution_context + attribution_run normalized schema; scheduler_type and attempt_id read from connector-supplied context; attribution_run carries terminal_state, sub_domain, and responsible node taxonomy
```

## `execution_context`

`execution_context` stores the scheduler-derived workload context that the
Blueprint sent to the Attribution Service.

```sql
CREATE TABLE IF NOT EXISTS execution_context (
    id                 TEXT PRIMARY KEY,
    job_id             TEXT NOT NULL,
    attempt_id         TEXT,
    scheduler_type     TEXT NOT NULL,
    start_time         TEXT NOT NULL,
    end_time           TEXT NOT NULL,
    nodes_json         TEXT NOT NULL,
    status             TEXT,
    workload_framework TEXT,
    user_name          TEXT,
    raw_scheduler_json TEXT NOT NULL DEFAULT '{}',
    created_at         TEXT NOT NULL,
    UNIQUE(job_id, attempt_id, scheduler_type)
);
```

`attempt_id` is connector-supplied: Slurm uses SLUID and Kueue uses a
`kueue:v1:...` ID derived from Workload UID, admission, and restart generation.
After upgrading, old UID-based Kueue rows can coexist with canonical-ID rows;
no automatic migration or schema-version change is applied.

`nodes_json` contains the scheduler node list serialized as a JSON array, for
example `["node-a", "node-b"]`. It is distinct from the Attribution Service v2
`workload.nodes` payload, which the Blueprint builds as a node-to-ranks map
from this node list and the configured GPUs per node (ranks per node).
`raw_scheduler_json` preserves the connector-supplied scheduler payload for
debugging and downstream inspection.

## `attribution_run`

`attribution_run` stores the enriched attribution result returned by the
Blueprint for an execution context.

```sql
CREATE TABLE IF NOT EXISTS attribution_run (
    id                   TEXT PRIMARY KEY,
    execution_context_id TEXT NOT NULL REFERENCES execution_context(id),
    schema_version       TEXT NOT NULL,
    pipeline_version     TEXT,
    service_api_version  TEXT,
    ruleset_version      TEXT,
    started_at           TEXT NOT NULL,
    completed_at         TEXT,
    outcome              TEXT NOT NULL,
    terminal_state       TEXT,
    sub_domain           TEXT,
    raw_result_json      TEXT NOT NULL,
    responsible_nodes_json TEXT NOT NULL DEFAULT '[]'
);
```

`outcome` is the raw service outcome. `terminal_state` and `sub_domain` are the
Blueprint taxonomy fields derived from that outcome. `raw_result_json` stores
the full enriched result payload, including connector reports, ingestion
reports, and `evidence_status`.

Each ingestion report counts that connector's observation POSTs in `attempted`
(failed POSTs included) and the failures in `failed`. It also carries the
Service's ingest counts for that connector: `accepted` (entries read, summed), `dropped` (discarded entries by
reason), and `empty` (delivered observations with zero accepted entries). See
[Missing signals and data gaps](../ATTRIBUTION_GUIDE.md#missing-signals-and-data-gaps)
for how these set `status`.
`responsible_nodes_json` stores the extracted responsible-node list as JSON.

## Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_execution_context_job_id ON execution_context(job_id);
CREATE INDEX IF NOT EXISTS idx_execution_context_end_time ON execution_context(end_time);
CREATE INDEX IF NOT EXISTS idx_run_outcome              ON attribution_run(outcome);
CREATE INDEX IF NOT EXISTS idx_run_execution_context    ON attribution_run(execution_context_id);
```

The dashboard uses these tables to list runs, filter by outcome, and join
attribution results back to their execution contexts.

## Optional Web UI Rollup

Quickstart sample databases may include `webui_run_fact`, a denormalized table
and `webui_node_day_fact`, denormalized tables used only to accelerate
dashboard KPIs and charts. The canonical attribution payload remains in
`execution_context` and `attribution_run`.

```sql
CREATE TABLE IF NOT EXISTS webui_run_fact (
    run_id       TEXT PRIMARY KEY,
    end_time     TEXT NOT NULL,
    end_day      TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    user_name    TEXT NOT NULL DEFAULT '',
    outcome      TEXT NOT NULL,
    gpu_hours    REAL NOT NULL,
    gpu_count    INTEGER NOT NULL,
    root_symptom TEXT NOT NULL,
    root_rule    TEXT NOT NULL DEFAULT '(unnamed)'
);

CREATE INDEX IF NOT EXISTS idx_webui_run_fact_end_time
    ON webui_run_fact(end_time);
CREATE INDEX IF NOT EXISTS idx_webui_run_fact_outcome_end_time
    ON webui_run_fact(outcome, end_time);
CREATE INDEX IF NOT EXISTS idx_webui_run_fact_user_end_time
    ON webui_run_fact(user_name, end_time);
CREATE INDEX IF NOT EXISTS idx_webui_run_fact_root_symptom_end_time
    ON webui_run_fact(root_symptom, end_time);
CREATE INDEX IF NOT EXISTS idx_webui_run_fact_root_rule_end_time
    ON webui_run_fact(root_rule, end_time);
```

`webui_node_day_fact` stores one row per observed node per end-day so the web UI
can display window GPU capacity as unique observed nodes multiplied by GPUs per
node without expanding every execution context on page load.

```sql
CREATE TABLE IF NOT EXISTS webui_node_day_fact (
    end_day       TEXT NOT NULL,
    node_name     TEXT NOT NULL,
    gpus_per_node INTEGER NOT NULL,
    PRIMARY KEY(end_day, node_name)
);
```

## Kafka result events

Kafka output publishes structured CloudEvents 1.0 JSON with an
`application/cloudevents+json` header. The record key is the execution context
`run_id` when it is a non-empty string, otherwise `id`.
This lets callers use a distinct processing `id` for each attempt while retaining
the original run as the partition key. See [configuration](CONFIGURATION.md) for connection,
authentication, and size/timeout settings.

| Event field | Value |
| --- | --- |
| `specversion`, `datacontenttype` | `1.0`, `application/json`. |
| `id`, `time` | A new UUID per emit and the UTC production time. |
| `source`, `type` | Configured values; defaults are `/attribution/blueprint` and `com.nvidia.attribution.job.attributed.v1`. |
| `subject` | Triggering `event_subject`, falling back to `jobs/<job_id>`. |
| `data.job_id`, `data.attempt_id` | Context job/attempt identity; defaults are the context run identity (`run_id`, otherwise `id`) and `"0"`. |
| `data.cluster`, `data.scheduler` | Context values; cluster can fall back to configuration. Missing values are null. |
| `data.trigger_event_id` | Original completion event ID, or null for contexts without a triggering event. |
| `data.terminal_state`, `data.sub_domain` | Blueprint summary fields. |
| `data.tenant_sre_action`, `data.cloud_operator_sre_action` | Blueprint action fields; empty strings when unavailable. |
| `data.result` | Service result fields, including verdict, symptoms, collectors, and observation IDs. |
| `data.connector_reports` | Collector reports; defaults to an empty list. |
| `data.ingestion_reports`, `data.evidence_status` | Blueprint evidence extensions, included when present. |

### Delivery behavior

A write succeeds only after broker acknowledgement, using `acks=all` and an
idempotent producer. Enqueue errors, delivery failures, and unconfirmed delivery
raise to the caller. Broker acceptance does not confirm RunAPI consumption.

A timeout can occur after the broker stored the record. Native idempotence
covers client transport retries; separate writes and application restarts can
produce duplicate results. Consumers must handle repeats and stale results
using job/attempt identity and trigger correlation. No evidence is trimmed to
meet the size limit. Closing flushes pending records and rejects later writes.

The current Blueprint orchestrator counts a context as successful if any output
succeeds; SQLite success can therefore coexist with Kafka failure. This sink
does not change input offset commits or provide durable application retries.
Required-output policy, acknowledgement, and recovery belong to Runtime
integration.
