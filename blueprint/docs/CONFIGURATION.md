# Blueprint Configuration Reference

The blueprint reads configuration from three places:

1. TOML, via `--config ./attribution-blueprint.toml` or the default `./attribution-blueprint.toml`
2. environment variables prefixed with `NVAB_`
3. CLI `--connector` assignments

Precedence is `CLI > environment > TOML > built-in defaults`.

For first-time use, create a TOML file and pass per-run job IDs on the CLI.

## Minimal SLURM Config

This is the smallest useful config for attributing real SLURM jobs and keeping results in SQLite:

```toml
[attribution]
service_url = "http://127.0.0.1:8000"
timeout_seconds = 60

[eca]
component = "slurm"

[eca.slurm]
sacct_path = "sacct"
cluster_tz = "America/Los_Angeles"

[workload]
gpus_per_node = 8

[log.applog]
component = "local_file"

[log.applog.local_file]
paths = ["{std_out}", "{std_err}"]
max_bytes = "2GB"

[output.db]
component = "sqlite"

[output.db.sqlite]
path = "./attributions.db"
```

Run it with:

```bash
uv run nvidia-attribution-blueprint \
  --config ./attribution-blueprint.toml \
  --connector eca slurm ids "123456"
```

## Required Sections

### `[attribution]`

The blueprint uses this section to call the attribution service.

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `service_url` | Yes in practice | `http://localhost:8080` | Base URL for the attribution service. Set this explicitly because the service examples normally run on port `8000`. |
| `timeout_seconds` | No | `60.0` | HTTP timeout for service calls. |

Example:

```toml
[attribution]
service_url = "http://127.0.0.1:8000"
timeout_seconds = 60
```

### `[eca]` and `[eca.slurm]`

`eca` means execution context adapter. For SLURM, it discovers jobs through `sacct`.

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `[eca].component` | Yes | `slurm` | Selects the execution context adapter. Use `slurm` for a SLURM cluster. |
| `sacct_path` | No | `sacct` | Path to the `sacct` binary. Use an absolute path if `sacct` is not on `PATH`. |
| `cluster_tz` | Strongly recommended | blueprint host timezone | IANA timezone for the SLURM controller, used to anchor naive `sacct` timestamps. |
| `ids` | Per run | none | Comma-separated job IDs. Usually pass this by CLI, not TOML. |
| `start_time` | Per run | none | ISO-8601 time for time-range mode. |
| `end_time` | Per run | none | ISO-8601 time for time-range mode. Required with `start_time`. |
| `nodelist` | Per run | none | Optional node or SLURM node range filter for time-range mode. |

Example:

```toml
[eca]
component = "slurm"

[eca.slurm]
sacct_path = "/usr/bin/sacct"
cluster_tz = "UTC"
```

Runtime selector examples:

```bash
uv run nvidia-attribution-blueprint --config ./attribution-blueprint.toml \
  --connector eca slurm ids "123456,123457"

uv run nvidia-attribution-blueprint --config ./attribution-blueprint.toml \
  --connector eca slurm start_time "2026-05-25T00:00:00+00:00" \
  --connector eca slurm end_time "2026-05-25T06:00:00+00:00" \
  --connector eca slurm nodelist "node[001-004]"
```

**Cluster Timezone Gotcha**

If the blueprint host and slurmctld run in different timezones, `cluster_tz` is effectively required, not optional. The default ("blueprint host timezone") silently mis-anchors `sacct` timestamps when the two hosts disagree.

- Symptom: sacct timestamps are shifted by hours, causing attribution windows to miss the real job range.
- How to find the right value: run `timedatectl` on the slurmctld host, or ask cluster ops for the canonical IANA timezone (for example, `America/Los_Angeles` or `UTC`).

### `[eca.kueue]`

Use the Kueue adapter for Kubernetes batch workloads managed by Kueue. It shells
out to `kubectl`, reads Kueue Workload objects, follows owner references to Job
or JobSet pods, and normalizes the result into execution contexts.

Example:

```toml
[eca]
component = "kueue"

[eca.kueue]
kubectl_path = "kubectl"
kubeconfig = "/path/to/kubeconfig"
context = "prod-batch"
namespace = "team-a"
timeout_seconds = 60
```

Runtime selector examples:

```bash
uv run nvidia-attribution-blueprint --config ./attribution-blueprint.toml \
  --connector eca kueue ids "team-a/workload-a,team-a/workload-b"

uv run nvidia-attribution-blueprint --config ./attribution-blueprint.toml \
  --connector eca kueue start_time "2026-05-25T00:00:00+00:00" \
  --connector eca kueue end_time "2026-05-25T06:00:00+00:00" \
  --connector eca kueue nodelist "gpu-node-a,gpu-node-b"
```

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `ids` | Per run | none | Comma-separated Workload refs. Use `<namespace>/<name>` unless `namespace` is configured. |
| `start_time` | Per run | none | ISO-8601 time for time-range mode. |
| `end_time` | Per run | none | ISO-8601 time for time-range mode. Required with `start_time`. |
| `nodelist` | Per run | none | Optional comma-separated Kubernetes node names for time-range mode. |
| `kubectl_path` | No | `kubectl` | Path to the `kubectl` binary. |
| `kubeconfig` | No | kubectl default | Optional kubeconfig path. |
| `context` | No | current context | Optional kubeconfig context name. |
| `namespace` | No | all namespaces | Optional namespace scope. Also allows bare workload names in `ids` mode. |
| `timeout_seconds` | No | `60` | Timeout for each `kubectl` command. |

Kueue time-range filtering is client-side because the Workload API does not
provide server-side time filtering. Pod evidence and `kubectl_logs` collection
depend on the owned pods still existing when the blueprint runs.

### `[eca.static]`

Use the static adapter for sample data, local smoke tests, or cases where an
operator already has the execution context fields.

Inline context example:

```toml
[eca]
component = "static"

[eca.static]
id = "sample-job"
start_time = "2026-05-25T00:00:00+00:00"
end_time = "2026-05-25T00:30:00+00:00"
nodes = ["node-a"]
state = "FAILED"
name = "sample-job"
user_name = "alice"
workload_type = "slurm"
workload_framework = "pytorch"
std_out = "docs/sample-data/static/applog/stdout.log"
std_err = "docs/sample-data/static/applog/stderr.log"
```

JSON context list example:

```toml
[eca]
component = "static"

[eca.static]
contexts_path = "path/to/static-contexts.json"
```

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `contexts_path` | Alternative to inline fields | none | JSON file containing either a context list or an object with a `contexts` list. |
| `id` | No for inline mode | derived hash | Execution context ID. |
| `start_time` | Yes for inline mode | none | Timezone-aware ISO-8601 start time. |
| `end_time` | Yes for inline mode | none | Timezone-aware ISO-8601 end time. |
| `nodes` | Yes for inline mode | none | Node list. |
| `state` | No | `UNKNOWN` | Scheduler state string. |
| `name` | No | none | Workload name. |
| `user_name` | No | none | Workload owner. |
| `std_out` | No | none | Path used by local file APPLOG placeholders. |
| `std_err` | No | none | Path used by local file APPLOG placeholders. |
| `workload_type` | No | empty string | Workload type sent to the service, such as `slurm`. |
| `workload_framework` | No | none | Optional framework label, such as `pytorch`. |
| `attempt_id` | No | `0` | Attempt key used by outputs. |
| `gpu_count` | No | none | Optional allocated GPU count used by outputs and the web UI. |

### `[workload]`

This section controls workload shape sent to the attribution service.

**Important:** `workload.gpus_per_node` defaults to `4`. This is also the
ranks-per-node value. In the supported training model, one rank runs on one GPU,
so GPUs per node and ranks per node are the same number. Review this for your
cluster before deploying; mismatches silently break rank-to-node attribution.

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `gpus_per_node` | No | `4` | GPUs per node (ranks per node). The blueprint uses this to build the service v2 node-to-ranks payload. Default: `4`, applied when this key is unset. **You should review and override this for your cluster.** Clusters with a different GPU count per node will produce incorrect rank-to-node attribution if this is left at the default. |

Example:

```toml
[workload]
gpus_per_node = 8
```

For `nodes = ["node-a", "node-b"]`, the service v2 payload contains
`{"node-a": [0, 1, 2, 3, 4, 5, 6, 7], "node-b": [8, 9, 10, 11, 12, 13, 14, 15]}`.

### `[output.<name>]`

Outputs are named instances. The instance name is yours, such as `db`, `console`, or `primary`.

If you omit all output blocks, the blueprint prints one JSON object per verdict to stdout.

SQLite output:

```toml
[output.db]
component = "sqlite"

[output.db.sqlite]
path = "./attributions.db"
```

| Field | Required | Meaning |
|---|---:|---|
| `path` | Yes | SQLite database file path. Example: `./attributions.db`. Build fails with `MCAError` if absent. |

Stdout output:

```toml
[output.console]
component = "stdout"
```

Multiple outputs can be enabled together:

```toml
[output.console]
component = "stdout"

[output.db]
component = "sqlite"

[output.db.sqlite]
path = "./attributions.db"
```

## Log Collection

Log collection is optional, but attribution quality is usually poor without at least application logs.

The supported log sources are:

- `applog`, application logs
- `syslog`, system logs

The supported log components are:

- `local_file`, read files visible on the blueprint host
- `loki`, query Loki over HTTP
- `kubectl_logs`, collect Kubernetes pod logs for Kueue workloads

### Local File APPLOG

Use this when SLURM `StdOut` and `StdErr` paths are on a filesystem mounted on the blueprint host.

```toml
[log.applog]
component = "local_file"

[log.applog.local_file]
paths = ["{std_out}", "{std_err}"]
max_bytes = "2GB"
```

Supported placeholders include:

| Placeholder | Meaning |
|---|---|
| `{std_out}` | Resolved `StdOut` path from `sacct`. |
| `{std_err}` | Resolved `StdErr` path from `sacct`. |
| `{job_id}` | Numeric job ID. |
| `{name}` | SLURM job name. |
| `{user_name}` | SLURM user. |
| `{start_time_ns}` | Execution context start time in nanoseconds. |
| `{end_time_ns}` | Execution context end time in nanoseconds. |

Scheduler-specific placeholders such as `{array_id}` and `{attempt_index}` are not supported by the local file connector. Use `{std_out}` and `{std_err}` for SLURM output files; the SLURM execution context adapter resolves scheduler filename tokens before the local file connector sees them.

`max_bytes` caps each file by reading the last N bytes. This keeps large training logs from exhausting memory.

### Local File SYSLOG

Use this for static or mounted node syslogs. Each configured path is a base directory, and each execution-context node must have a matching child directory:

```text
/path/to/syslog/node-a/messages.log
/path/to/syslog/node-b/messages.log
```

```toml
[log.syslog]
component = "local_file"

[log.syslog.local_file]
paths = ["/path/to/syslog"]
max_bytes = "2GB"
```

For an execution context whose `nodes` are `["node-a", "node-b"]`, the connector reads regular files directly under `/path/to/syslog/node-a/` and `/path/to/syslog/node-b/`. Missing per-node directories are logged as warnings. Each emitted syslog observation uses `raw_log`, with the regular syslog file text as its body.

### Loki SYSLOG Or APPLOG

Use Loki when system logs or application logs are stored centrally.

Token authentication:

```toml
[log.syslog]
component = "loki"

[log.syslog.loki]
endpoint = "https://loki.example"
query_kwargs = { service_name = "slurm-cluster" }
node_label = "host_name"

[log.syslog.loki.auth]
type = "token"
token_env = "LOKI_TOKEN"
```

Then set the token before running:

```bash
export LOKI_TOKEN="..."
```

mTLS authentication:

```toml
[log.syslog]
component = "loki"

[log.syslog.loki]
endpoint = "https://loki.example"
query_kwargs = { service_name = "slurm-cluster" }

[log.syslog.loki.auth]
type = "mtls"
cert_path = "/etc/attribution/client.crt"
key_path = "/etc/attribution/client.key"
ca_cert_path = "/etc/attribution/ca.crt"
```

Fields:

| Field | Required | Meaning |
|---|---:|---|
| `endpoint` | Yes | Loki base URL. |
| `auth.type` | Yes | `token` or `mtls`. |
| `auth.token_env` | Yes for token | Environment variable that contains the bearer token. |
| `auth.cert_path` | Yes for mTLS | Client certificate path. |
| `auth.key_path` | Yes for mTLS | Client private key path. |
| `auth.ca_cert_path` | No | CA bundle path. |
| `auth.org_id` | No | Optional `X-Scope-OrgID` value. |
| `query_kwargs` | Usually | Label values used by bundled LogQL templates. |
| `query` | No | Single custom LogQL template. |
| `queries` | No | List of custom LogQL templates. |
| `node_label` | No | Loki stream label used as the node name. Defaults to `host_name`. |
| `timeout_seconds` | No | HTTP timeout. Defaults to `60.0`. |

The bundled Loki templates are examples. If your labels are not `service_name`, `app`, or `host_name`, set `query`, `queries`, `query_kwargs`, and `node_label` for your environment.

### Kubectl Pod Logs

Use `kubectl_logs` with the Kueue adapter to collect pod stdout as APPLOG or
SYSLOG observations. The Kueue adapter populates each execution context with
`namespace` and `pod_names`; this connector reads those fields and runs
`kubectl logs` for each pod.

```toml
[log.applog]
component = "kubectl_logs"

[log.applog.kubectl_logs]
kubectl_path = "kubectl"
context = "prod-batch"
container = "trainer"
timestamps = true
previous = false
since_context_start = true
max_bytes_per_pod = "16MB"
timeout_seconds = 60
```

Fields:

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `kubectl_path` | No | `kubectl` | Path to the `kubectl` binary. |
| `kubeconfig` | No | kubectl default | Optional kubeconfig path. |
| `context` | No | current context | Optional kubeconfig context name. |
| `container` | No | all containers | Container name. When omitted, logs are read with `--all-containers=true`. |
| `timestamps` | No | `true` | Include timestamps in `kubectl logs` output. |
| `previous` | No | `false` | Read logs from a previous container instance. |
| `since_context_start` | No | `false` | Add `--since-time` using the execution context start time. |
| `max_bytes_per_pod` | No | `16MB` | Tail cap applied to each pod log. |
| `timeout_seconds` | No | `60` | Timeout for each `kubectl logs` command. |

This connector does not use `{std_out}` or `{std_err}`. Those placeholders are
for local file log collection.

## Metrics Collection

PromQL metrics are optional. Configure them when node metrics are available from Prometheus, Mimir, or a compatible endpoint.

```toml
[metrics.node_metrics]
component = "promql"

[metrics.node_metrics.promql]
endpoint = "https://mimir.example/prometheus"
cluster = "slurm-cluster"
step_seconds = 60

[metrics.node_metrics.promql.auth]
type = "token"
token_env = "MIMIR_TOKEN"
```

Fields:

| Field | Required | Meaning |
|---|---:|---|
| `endpoint` | Yes | Prometheus-compatible HTTP API base URL. |
| `cluster` | Yes | Cluster label value used by bundled PromQL templates. |
| `auth.type` | No | Optional. If present, `token` or `mtls`; same auth fields as Loki. Omit the `auth` block for unauthenticated endpoints. |
| `step_seconds` | No | Query step size. Defaults to `60`. |
| `timeout_seconds` | No | HTTP timeout. Defaults to `60.0`. |
| `queries` | No | Custom named PromQL templates. These extend the bundled templates. |

The bundled PromQL templates expect labels such as `s_cluster` and `net_host_name`. If your metrics use different labels, override the relevant query templates.

## Environment Variables

Every TOML value can be set with an `NVAB_` environment variable. Examples:

```bash
export NVAB_ATTRIBUTION_SERVICE_URL="http://127.0.0.1:8000"
export NVAB_ECA="slurm"
export NVAB_ECA_SLURM_CLUSTER_TZ="UTC"
export NVAB_OUTPUT_DB="sqlite"
export NVAB_OUTPUT_DB_SQLITE_PATH="./attributions.db"
```

CLI values still win over environment variables.

**Warning:** Env vars override TOML, and CLI flags override env vars. A stale `NVAB_*` in your shell will silently win against your config file. Run `env | grep NVAB_` before debugging unexpected config behavior.

## Config Checks

List valid config keys:

```bash
uv run nvidia-attribution-blueprint --connector help
uv run nvidia-attribution-blueprint --connector eca slurm help
uv run nvidia-attribution-blueprint --connector eca kueue help
uv run nvidia-attribution-blueprint --connector log syslog loki help
uv run nvidia-attribution-blueprint --connector log applog kubectl_logs help
uv run nvidia-attribution-blueprint --connector metrics node_metrics promql help
uv run nvidia-attribution-blueprint --connector output db sqlite help
```

Smoke-test a config without selecting any jobs:

```bash
uv run nvidia-attribution-blueprint --config ./attribution-blueprint.toml
```

This builds the configured connectors and exits with zero verdicts. It does not scan the entire cluster.
