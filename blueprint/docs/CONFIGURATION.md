# Blueprint Configuration Reference

The blueprint reads configuration from three places:

1. TOML, via `--config ./attribution-blueprint.toml`, `--configs <dir>`, or the default `./attribution-blueprint.toml`
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

[output.db]
component = "sqlite"

[output.db.sqlite]
path = "./attributions.db"
```

Run it with:

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --config ./attribution-blueprint.toml \
  --connector eca slurm ids "123456"
```



## CLI Flags


| Flag                      | Repeatable | Meaning                                                                                                                                                                                                                                        |
| ------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--connector TOKEN...`    | Yes        | Configuration assignment. The last token is the value; earlier tokens form the path. `--connector eca slurm` selects a component, `--connector eca slurm ids "1,5"` sets a parameter. End a path with `help` to list valid keys at that level. |
| `--config PATH`           | Yes        | TOML config file. When neither `--config` nor `--configs` is given, `./attribution-blueprint.toml` is used if it exists.                                                                                                                       |
| `--configs DIR`           | Yes        | Directory of TOML configs. Runs every top-level `*.toml` file in sorted order, one attribution pass per file. Fails if the directory contains no top-level `.toml` files.                                                                      |
| `--dump-observations DIR` | No         | Write collected observations to `DIR/<ctx-id>.<source>.json` before service-specific adaptation and submission. Multiple observations with the same context ID and source use the same path, so the last one replaces earlier dumps.           |


Notes:

- Section aliases: `execution_context` is accepted for `eca`, and `logs` for `log`, on the CLI, in environment variables, and in TOML.
- A component-qualified CLI or environment assignment also selects that component. `--connector eca kueue ids "team-a/wl-1"` sets `eca.kueue.ids` and implies `eca.component = "kueue"`. TOML does not imply selection; set `component` explicitly in TOML.
- With `--configs`, environment variables and `--connector` assignments apply to every config file in the run.



## Required Sections



### `[attribution]`

The blueprint uses this section to call the attribution service.


| Field             | Required        | Default                 | Meaning                                                                                                             |
| ----------------- | --------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `service_url`     | Yes in practice | `http://localhost:8080` | Base URL for the attribution service. Set this explicitly because the service examples normally run on port `8000`. |
| `timeout_seconds` | No              | `60.0`                  | HTTP read timeout for service calls.                                                                                |


For the service v2 request lifecycle and wire schemas, see
[Attribution Service REST API](./rest-api.md).

Example:

```toml
[attribution]
service_url = "http://127.0.0.1:8000"
timeout_seconds = 60
```



### `[eca]` and `[eca.slurm]`

`eca` means execution context adapter. For SLURM, it discovers jobs through `sacct`.


| Field             | Required             | Default                     | Meaning                                                                                                             |
| ----------------- | -------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `[eca].component` | No                   | `slurm`                     | Selects the execution context adapter: `slurm`, `kueue`, or `static`. Defaults to `slurm` when unset.               |
| `sacct_path`      | No                   | `sacct`                     | Path to the `sacct` binary. Use an absolute path if `sacct` is not on `PATH`.                                       |
| `cluster_tz`      | Strongly recommended | blueprint host fixed offset | IANA timezone for the SLURM controller, used to anchor naive `sacct` timestamps and preserve regional/DST identity. |
| `ids`             | Per run              | none                        | Comma-separated job IDs. Usually pass this by CLI, not TOML.                                                        |
| `start_time`      | Per run              | none                        | ISO-8601 time for time-range mode.                                                                                  |
| `end_time`        | Per run              | none                        | ISO-8601 time for time-range mode. Required with `start_time`.                                                      |
| `nodelist`        | Per run              | none                        | Optional node or SLURM node range filter for time-range mode.                                                       |


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
uv run --project blueprint --frozen nvidia-attribution-blueprint --config ./attribution-blueprint.toml \
  --connector eca slurm ids "123456,123457"

uv run --project blueprint --frozen nvidia-attribution-blueprint --config ./attribution-blueprint.toml \
  --connector eca slurm start_time "2026-05-25T00:00:00+00:00" \
  --connector eca slurm end_time "2026-05-25T06:00:00+00:00" \
  --connector eca slurm nodelist "node[001-004]"
```

Runtime selector rules, validated on the merged configuration before the run starts, for `slurm` and `kueue`:

- `ids` and `start_time` are mutually exclusive.
- `start_time` and `end_time` must be set together.
- `nodelist` is only valid in time-range mode.
- ISO-8601 times without a UTC offset are interpreted as UTC.

**Cluster Timezone Gotcha**

Set `cluster_tz` whenever the blueprint can run in a different timezone from
slurmctld, the controller's timezone observes daylight-saving changes, or its
UTC offset is not a whole hour. The host fallback loses the regional timezone
identity. If an unconfigured job spans a UTC offset change or uses a non-hour
offset, the blueprint rejects it instead of sending an incorrect timezone to
the Service.

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
placement_records_enabled = false
```

Runtime selector examples:

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint --config ./attribution-blueprint.toml \
  --connector eca kueue ids "team-a/workload-a,team-a/workload-b"

uv run --project blueprint --frozen nvidia-attribution-blueprint --config ./attribution-blueprint.toml \
  --connector eca kueue start_time "2026-05-25T00:00:00+00:00" \
  --connector eca kueue end_time "2026-05-25T06:00:00+00:00" \
  --connector eca kueue nodelist "gpu-node-a,gpu-node-b"
```


| Field                    | Required | Default         | Meaning                                                                                                           |
| ------------------------ | -------- | --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `ids`                    | Per run  | none            | Comma-separated Workload refs. Use `<namespace>/<name>` unless `namespace` is configured.                         |
| `start_time`             | Per run  | none            | ISO-8601 time for time-range mode.                                                                                |
| `end_time`               | Per run  | none            | ISO-8601 time for time-range mode. Required with `start_time`.                                                    |
| `nodelist`               | Per run  | none            | Optional comma-separated Kubernetes node names for time-range mode.                                               |
| `kubectl_path`           | No       | `kubectl`       | Path to the `kubectl` binary.                                                                                     |
| `kubeconfig`             | No       | kubectl default | Optional kubeconfig path.                                                                                         |
| `context`                | No       | current context | Optional kubeconfig context name.                                                                                 |
| `namespace`              | No       | all namespaces  | Optional namespace scope. Also allows bare workload names in `ids` mode. Required when `placement_records_enabled` is set. |
| `timeout_seconds`        | No       | `60`            | Timeout for each `kubectl` command.                                                                               |
| `placement_records_enabled` | No    | `false`         | Reads retained placement from the configured recorder; discovery fails if the recorder cannot be read. Requires `namespace`, matching the one namespace the recorder watches. |


The baseline Kueue adapter does not require the placement recorder or Prometheus. It
uses live Pods and a best-effort Kubernetes Events fallback. If Pods and their
Events are cleaned up before attribution, the execution context can have
missing `nodes` or `pod_names`; downstream node-scoped metrics and deleted Pod
logs will then be incomplete.

#### Pod placement recorder setup

Install the recorder before the workloads whose placement must be retained. A
single-replica StatefulSet writes PlacementRecord JSON to SQLite on a
`ReadWriteOnce` PVC and serves it through a ClusterIP Service.
See the
[recorder architecture, usage, and troubleshooting reference](../src/attribution_blueprint/connectors/execution_contexts/kueue/pod_placement/recorder/docs/README.md).

Build the image, then install the SQLite backend:

```bash
kubectl config current-context
docker build \
  -f blueprint/docker/pod-placement-recorder.Dockerfile \
  -t registry.example/pod-placement-recorder:VERSION .
docker push registry.example/pod-placement-recorder:VERSION

YES=1 RELEASE_NAME=attribution \
  ./blueprint/scripts/install-pod-placement-recorder.sh team-a \
  --set image.repository=registry.example/pod-placement-recorder \
  --set image.tag=VERSION \
  --set image.pullPolicy=IfNotPresent
```

For kind, load the locally built image with
`kind load docker-image pod-placement-recorder:local --name CLUSTER` and keep
the chart's default image values. Place the recorder on a system node pool
outside the failure domain it observes. The chart accepts `nodeSelector` and
`tolerations` values for that purpose; a recorder that fails with the workload
cannot retain the workload's final Pod update. SQLite PVC settings are
`storage.sqlite.size`, `storage.sqlite.storageClassName`, and
`storage.sqlite.accessModes`. Records are removed automatically after
`retentionDays`, which defaults to `30`.

After installation, wait for the StatefulSet and verify its read API:

```bash
kubectl -n team-a rollout status statefulset/attribution-pod-placement-recorder
kubectl -n team-a port-forward service/attribution-pod-placement-recorder 18080:8080
curl -fsS \
  'http://127.0.0.1:18080/v1/placements?namespace=team-a&workload_uid=WORKLOAD_UID'
```

Point an in-cluster Blueprint at the Service:

```toml
[eca.kueue]
placement_records_enabled = true

[eca.kueue.placement_recorder]
endpoint = "http://attribution-pod-placement-recorder.team-a.svc:8080"
timeout_seconds = 10
```

`endpoint` is required when this section is present; `timeout_seconds`
defaults to `10`. Optional `auth` supports `token` (`token_env`, optional
`org_id`) or `mtls` (`cert_path`, `key_path`, optional `ca_cert_path` and
`org_id`).

When enabled, recorder lookup is fail-closed. After deleting a completed Pod,
a validation run should retain the
same Pod UID and node with `pod_placements[].source = "placement_record"`.
For node-loss validation, `nodeFailureReason` must be present before the
Pod disappears; post-hoc attribution should then return `NODE_FAIL` and the
recorded `failed_node`. If Blueprint runs outside the cluster, use an
authenticated proxy or port-forward and set the endpoint to that reachable
URL. Optional `placement_recorder.auth` supports the same `token` and `mtls`
fields as the Prometheus source. The recorder itself does not terminate
authentication, so keep its Service private or place an authenticated proxy
in front of it.

To recover Job and JobSet placement from retained kube-state-metrics, configure a
Prometheus-compatible endpoint under the Kueue adapter:

```toml
[eca.kueue.prometheus_metrics]
endpoint = "https://mimir.example/prometheus"
cluster = "prod-batch"
cluster_label = "cluster"
timeout_seconds = 60
lookback_padding_seconds = 300
batch_size = 20

[eca.kueue.prometheus_metrics.auth]
type = "token"
token_env = "MIMIR_TOKEN"
org_id = "tenant-a"
```

Fields:


| Field                      | Required | Meaning                                                                                                                                    |
| -------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `endpoint`                 | Yes      | Prometheus-compatible HTTP API base URL.                                                                                                   |
| `cluster`                  | No       | Cluster label value to scope every query. Must be set together with `cluster_label`.                                                       |
| `cluster_label`            | No       | Label name carrying `cluster` on kube-state-metrics series. Must be set together with `cluster`.                                           |
| `auth.type`                | No       | Optional. If present, `token` or `mtls`; same auth fields as Loki and PromQL metrics. Omit the `auth` block for unauthenticated endpoints. |
| `timeout_seconds`          | No       | HTTP timeout. Defaults to `60.0`.                                                                                                          |
| `lookback_padding_seconds` | No       | Seconds padded onto both ends of the Workload interval before querying. Defaults to `300`.                                                 |
| `batch_size`               | No       | Maximum owner Job UIDs included in one shared query. Defaults to `20`.                                                                     |
| `pod_info_metric`          | No       | kube-state-metrics Pod-to-node metric name. Defaults to `kube_pod_info`.                                                                   |
| `pod_labels_metric`        | No       | kube-state-metrics Pod-label metric name. Defaults to `kube_pod_labels`.                                                                   |
| `pod_controller_uid_label` | No       | Prometheus label containing the Pod controller Job UID. Defaults to `label_batch_kubernetes_io_controller_uid`.                            |


`cluster` and `cluster_label` are optional for a single-cluster endpoint, but
must be set together for a shared backend. The source evaluates an instant
`last_over_time` query over the relevant Workload intervals. Following FACT's
node-metadata collection pattern, the adapter combines owner Job UIDs from
every Workload in the run that has one, orders unique Job UIDs by padded
Workload interval, and queries them in namespace-scoped, bounded batches. 
The query joins `kube_pod_labels` to `kube_pod_info` on namespace, Pod name, 
and Pod UID. For JobSets, the adapter first resolves the current child Job UIDs 
from Kubernetes. Configure kube-state-metrics with:

```text
--metric-labels-allowlist=pods=[batch.kubernetes.io/controller-uid]
```

This exposes `batch.kubernetes.io/controller-uid` as
`label_batch_kubernetes_io_controller_uid`. If a deployment uses a different
converted label name, set `pod_controller_uid_label`.

For the `prometheus-community/kube-state-metrics` chart, the equivalent values
are:

```yaml
metricLabelsAllowlist:
  - pods=[batch.kubernetes.io/controller-uid]
```

When kube-state-metrics is installed as a dependency of the
`prometheus-community/prometheus` chart, nest this under
`kube-state-metrics`. Confirm the running container has the allowlist argument,
then run a Job long enough to cross at least one scrape and verify that this
query is non-empty:

```promql
kube_pod_labels{
  namespace="team-a",
  label_batch_kubernetes_io_controller_uid="<exact Job UID>"
}
```

To prove the fallback rather than live-Pod or recorder recovery, leave
`placement_records_enabled = false`, delete the Pod after the series has been
scraped, and run attribution while the Workload remains available. The
result must contain the original Pod UID and node with
`pod_placements[].source = "prometheus"`. Repeat with a JobSet and retain its
child Job objects so the adapter can resolve their exact UIDs.

Kueue time-range filtering is client-side because the Workload API does not
provide server-side time filtering. The adapter reads pod and node names from
live owned pods and recorder records, then always merges in configured
Prometheus placement data for Workloads with an owner Job (already-known Pod
UIDs take precedence). If nodes or pod names are still missing afterward, it
falls back to querying Kubernetes Events for UID-matched Pod scheduling
evidence, bounded by `event_max_age_seconds`.

Prometheus recovery is bounded by kube-state-metrics scrape cadence and backend
retention. A Pod that exists entirely between scrapes is not recoverable.
JobSet recovery also requires its child Job objects to remain available at
attribution time so their UIDs can be resolved without a name-based metrics
join. If both a JobSet's Pods and child Jobs have been deleted, this UID-safe
Prometheus route returns no placement even when the old metrics remain.

For durable pod-to-node placement, install one pod-placement-recorder release per namespace:

```bash
YES=1 blueprint/scripts/install-pod-placement-recorder.sh team-a
```

The preview supports Job and JobSet owners and stores node names without
reading Node objects; JobSet support does not require its API on Jobs-only
clusters. Recorder `queueSize` and `workers` must be positive. Runtime API
defaults are Kueue Workload `v1beta1` and JobSet `v1alpha2`; override
`workloadVersion` or `jobsetVersion` to match the installed controllers.
SQLite records older than `retentionDays` are removed automatically; the
default is 30 days.

The adapter derives `kueue_state` from Kueue conditions, using finished
conditions before evicted, preempted, admitted, quota-reserved, and pending
conditions. For example, a `Finished=True` condition with reason `Succeeded`
becomes `FINISHED_SUCCEEDED`. It also exposes a service-facing `state`.
`FINISHED_SUCCEEDED` is normalized to `COMPLETED`. `FINISHED_FAILED` is
normalized to `TIMEOUT` when its direct Job or a JobSet child Job failed with
reason `DeadlineExceeded`, and otherwise to `FAILED`. A selected pod carrying
either the legacy reason `NodeLost` or a `DisruptionTarget=True` condition with
reason `DeletionByPodGC` normalizes the
service-facing state to `NODE_FAIL`, even when Kueue still reports the workload
as admitted. The same retained PlacementRecord evidence restores `NODE_FAIL`
after Pod deletion, without changing the existing `EVICTED_NODEFAILURES`
mapping. Both
`EVICTED_PREEMPTED` and `PREEMPTED` are normalized to `PREEMPTED`, while
`EVICTED_NODEFAILURES` becomes `NODE_FAIL`. A native `Requeued=True` condition
becomes the existing service state `REQUEUED`. Other eviction reasons (queue
stopped, admission check failed, pods-ready timeout, deactivated, ...)
normalize to the generic service state `EVICTED`, which produces
`Incomplete Info` when no stronger symptoms are available. The original `kueue_state`
remains available in `raw_scheduler_json`.

Kubernetes Job `activeDeadlineSeconds` values are converted from seconds to
the positive integer minutes expected by the Attribution Service, rounding
up. For JobSets, the shortest child-Job deadline is used.

The adapter marks a node as failed when one of its selected pods has a
container with a non-zero terminated exit code or carries either node-loss
signal described above. If more than one node meets that condition,
`failed_node` is the alphabetically first node name so the result is stable
across Kubernetes response orderings.

For a complete runnable configuration, see
[examples/kueue](../examples/kueue/README.md).

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
timezone = "UTC"
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


| Field                | Required                     | Default            | Meaning                                                                                                                                                                                                                                                         |
| -------------------- | ---------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `contexts_path`      | Alternative to inline fields | none               | JSON file containing either a context list or an object with a `contexts` list. Each context requires `id`, `start_time`, `end_time` (timezone-aware), and `nodes`; duplicate `id` values are rejected. Optional `timezone` values are validated as IANA names. |
| `id`                 | No for inline mode           | `static-<hash>`    | Execution context ID. The default hash is derived from `start_time`, `end_time`, and `nodes`.                                                                                                                                                                   |
| `start_time`         | Yes for inline mode          | none               | Timezone-aware ISO-8601 start time.                                                                                                                                                                                                                             |
| `end_time`           | Yes for inline mode          | none               | Timezone-aware ISO-8601 end time.                                                                                                                                                                                                                               |
| `timezone`           | No                           | inferred when safe | IANA timezone used to interpret offsetless evidence. Set it when timestamps use a non-hour numeric offset or the workload uses a regional timezone, such as `Asia/Kolkata` or `America/Los_Angeles`.                                                            |
| `nodes`              | Yes for inline mode          | none               | Node list. Accepts a TOML list or a comma-separated string.                                                                                                                                                                                                     |
| `state`              | No                           | `UNKNOWN`          | Scheduler state sent unchanged as the service workload `status`. This is not the attribution verdict.                                                                                                                                                           |
| `name`               | No                           | none               | Workload name.                                                                                                                                                                                                                                                  |
| `user_name`          | No                           | none               | Workload owner.                                                                                                                                                                                                                                                 |
| `std_out`            | No                           | none               | Path used by local file APPLOG placeholders.                                                                                                                                                                                                                    |
| `std_err`            | No                           | none               | Path used by local file APPLOG placeholders.                                                                                                                                                                                                                    |
| `workload_type`      | No                           | empty string       | Workload type sent unchanged to service v2. The v2 schema accepts any string. Examples include `slurm`, `runai`, and `kueue`; the empty string means unspecified.                                                                                               |
| `workload_framework` | No                           | none               | Optional framework label, such as `pytorch`.                                                                                                                                                                                                                    |
| `attempt_id`         | No                           | `0`                | Attempt key used by outputs.                                                                                                                                                                                                                                    |
| `gpu_count`          | No                           | none               | Optional allocated GPU count used by outputs and the web UI.                                                                                                                                                                                                    |


`state` is an open scheduler string, not a closed enum. Its value can affect
fallback attribution when no stronger log or metric symptom exists. The
current rules treat exactly `COMPLETED` as a successful fallback. Values such
as `FAILED`, `UNKNOWN`, or an unrecognized scheduler-specific state fall back
to `job failed`. Use the scheduler's real state, and do not use `state` to
preselect the desired attribution outcome.

`workload_type` identifies the workload manager to the service. The Kueue
adapter sends `kueue` and also records `scheduler = "kueue"` internally for
SQLite output. Service v2 accepts open string values.

### `[workload]`

This section controls workload shape sent to the attribution service.

**Important:** `workload.gpus_per_node` defaults to `4`. This is also the
ranks-per-node value. In the supported training model, one rank runs on one GPU,
so GPUs per node and ranks per node are the same number. Review this for your
cluster before deploying; mismatches silently break rank-to-node attribution.


| Field           | Required | Default | Meaning                                                                                                                                                                                                                                                                                                                                     |
| --------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gpus_per_node` | No       | `4`     | GPUs per node (ranks per node). The blueprint uses this to build the service v2 node-to-ranks payload. Default: `4`, applied when this key is unset. **You should review and override this for your cluster.** Clusters with a different GPU count per node will produce incorrect rank-to-node attribution if this is left at the default. |


Example:

```toml
[workload]
gpus_per_node = 8
```

For `nodes = ["node-a", "node-b"]`, the service v2 payload contains
`{"node-a": [0, 1, 2, 3, 4, 5, 6, 7], "node-b": [8, 9, 10, 11, 12, 13, 14, 15]}`.

### `[output.<name>]`

Outputs are named instances. The instance name is yours, such as `db`, `console`, or `primary`. The name `component` and names starting with `_` are reserved.

If you omit all output blocks, the blueprint prints one JSON object per verdict to stdout.

SQLite output:

```toml
[output.db]
component = "sqlite"

[output.db.sqlite]
path = "./attributions.db"
```


| Field  | Required | Meaning                                                                                         |
| ------ | -------- | ----------------------------------------------------------------------------------------------- |
| `path` | Yes      | SQLite database file path. Example: `./attributions.db`. Build fails with `MCAError` if absent. |


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

Log evidence is required for attribution. Logs must either be sent directly to
the Attribution Service or collected by Attribution Blueprint. Attribution
quality is usually poor without at least SYSLOG evidence. APPLOG is output from
workload processes, including application and framework messages. SYSLOG is
node-level operating-system, kernel, and GPU-driver output.

The supported log sources are:

- `applog`, application logs
- `syslog`, system logs

The supported log components are:

- `local_file`, read files visible on the blueprint host
- `loki`, query Loki over HTTP
- `kubectl_logs`, collect Kubernetes pod logs for Kueue workloads



### Evidence Coverage


| Logs available to the Attribution Service | Expected coverage                                                                                                                                                                          | Main limitation                                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| APPLOG and SYSLOG                         | Recommended for the highest accuracy. SYSLOG provides node and GPU evidence, while APPLOG adds process and framework context plus more visibility into network and communication failures. | Coverage still depends on the events present in the collected logs.                               |
| SYSLOG only                               | Supports most GPU-related attributions from kernel, driver, and node events.                                                                                                               | Application context is absent, and network visibility is reduced.                                 |
| APPLOG only                               | Supports application and framework evidence, including network or communication failures reported in workload output.                                                                      | GPU and hardware attributions may be missed when the relevant evidence exists only in node logs.  |
| No logs                                   | Insufficient evidence for attribution.                                                                                                                                                     | Send logs directly to the Attribution Service or configure Attribution Blueprint to collect them. |


If logs are available for only some workload nodes, attribution has reduced
evidence coverage for the missing nodes and may be unable to assign node
responsibility accurately.

### Local File APPLOG

Use this when SLURM `StdOut` and `StdErr` paths are on a filesystem mounted on the blueprint host.

```toml
[log.applog]
component = "local_file"

[log.applog.local_file]
paths = ["{std_out}", "{std_err}"]
```


| Field       | Required | Default | Meaning                                                                                                                         |
| ----------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `paths`     | Yes      | none    | Path templates to read. Accepts a list or a single string.                                                                      |
| `max_bytes` | No       | `2GB`   | Per-file cap. Accepts an integer byte count or a size string such as `500MB` or `2G` (binary multiples, suffixes through `TB`). |


Supported placeholders include:


| Placeholder       | Meaning                                                                     |
| ----------------- | --------------------------------------------------------------------------- |
| `{std_out}`       | Resolved `StdOut` path from `sacct`.                                        |
| `{std_err}`       | Resolved `StdErr` path from `sacct`.                                        |
| `{job_id}`        | Job ID from the execution context. Falls back to the context ID when unset. |
| `{name}`          | SLURM job name.                                                             |
| `{user_name}`     | SLURM user.                                                                 |
| `{start_time_ns}` | Execution context start time in nanoseconds.                                |
| `{end_time_ns}`   | Execution context end time in nanoseconds.                                  |


Scheduler-specific placeholders such as `{array_id}` and `{attempt_index}` are not supported by the local file connector. Use `{std_out}` and `{std_err}` for SLURM output files; the SLURM execution context adapter resolves scheduler filename tokens before the local file connector sees them.

`max_bytes` caps each file by reading the last N bytes. This keeps large training logs from exhausting memory.

If a rendered path is a directory, every regular file directly inside it is read (non-recursive). Paths that do not exist are skipped silently; paths that exist but cannot be read are logged as warnings.

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
```

For an execution context whose `nodes` are `["node-a", "node-b"]`, the connector reads regular files directly under `/path/to/syslog/node-a/` and `/path/to/syslog/node-b/`. Missing per-node directories are logged at debug level; the connector warns when no node directory is found under any configured base path. `max_bytes` behaves as for APPLOG (default `2GB`). Each emitted syslog observation uses `entry_timeseries`: one entry per line, dated from the line's own timestamp and tagged with the directory's node. Lines without a recognizable timestamp are dropped; see [Manually Collected Syslogs](use-cases/manually-collected-syslogs.md#syslog-file-format) for the accepted forms.

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


| Field               | Required      | Meaning                                                                                                                           |
| ------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `endpoint`          | Yes           | Loki base URL.                                                                                                                    |
| `auth.type`         | Yes           | `token` or `mtls`.                                                                                                                |
| `auth.token_env`    | Yes for token | Environment variable that contains the bearer token.                                                                              |
| `auth.cert_path`    | Yes for mTLS  | Client certificate path.                                                                                                          |
| `auth.key_path`     | Yes for mTLS  | Client private key path.                                                                                                          |
| `auth.ca_cert_path` | No            | CA bundle path.                                                                                                                   |
| `auth.org_id`       | No            | Optional `X-Scope-OrgID` value.                                                                                                   |
| `query_kwargs`      | Usually       | Label values used by bundled LogQL templates.                                                                                     |
| `query`             | No            | Single custom LogQL template. Replaces the bundled templates for this source.                                                     |
| `queries`           | No            | List of custom LogQL templates. Replaces the bundled templates for this source. Takes precedence over `query` when both are set.  |
| `node_label`        | No            | Loki stream label used as the node name, and spliced into each query to scope it to the context's nodes. Defaults to `host_name`. |
| `timeout_seconds`   | No            | HTTP timeout. Defaults to `60.0`.                                                                                                 |


The bundled Loki templates are examples. If your labels are not `service_name`, `app`, or `host_name`, set `query`, `queries`, `query_kwargs`, and `node_label` for your environment.

Custom templates may use `{job_id}` (the execution context ID) and any `{key}` supplied in `query_kwargs`. Literal `{` and `}` in LogQL label selectors must be doubled (`{{`, `}}`).

### Kubectl Pod Logs

Use `kubectl_logs` to collect pod stdout as APPLOG or SYSLOG observations. It
reads `namespace` and `pod_names` from the execution context and runs
`kubectl logs` for each pod. Entries are attributed to the pod's node via the
context's `pod_nodes` mapping (pod name to node). The Kueue adapter populates all
three fields.

```toml
[log.applog]
component = "kubectl_logs"

[log.applog.kubectl_logs]
kubectl_path = "kubectl"
context = "prod-batch"
container = "trainer"
timestamps = false
previous = false
since_context_start = true
timeout_seconds = 60
```

Fields:


| Field                 | Required | Default         | Meaning                                                                                          |
| --------------------- | -------- | --------------- | ------------------------------------------------------------------------------------------------ |
| `kubectl_path`        | No       | `kubectl`       | Path to the `kubectl` binary.                                                                    |
| `kubeconfig`          | No       | kubectl default | Optional kubeconfig path.                                                                        |
| `context`             | No       | current context | Optional kubeconfig context name.                                                                |
| `container`           | No       | all containers  | Container name. When omitted, logs are read with `--all-containers=true`.                        |
| `timestamps`          | No       | `true`          | Include timestamps in `kubectl logs` output.                                                     |
| `previous`            | No       | `false`         | Read logs from a previous container instance.                                                    |
| `since_context_start` | No       | `false`         | Add `--since-time` using the execution context start time.                                       |
| `max_bytes_per_pod`   | No       | `16MB`          | Tail cap applied to each pod log. Accepts an integer byte count or a size string such as `16MB`. |
| `timeout_seconds`     | No       | `60`            | Timeout for each `kubectl logs` command.                                                         |


`timestamps`, `previous`, and `since_context_start` accept TOML booleans, or
the strings `true`/`false` (also `yes`/`no`/`1`/`0`) when set from the CLI or
environment.

For APPLOG, set `timestamps = false` when application lines already begin with
a rank prefix such as `[rank 0]`. The current raw-log parser expects that prefix
at the start of the line. A timestamp added by `kubectl` would appear first and
prevent rank parsing. This setting only controls the prefix added by `kubectl`;
the application log itself must still put its rank prefix first.

This connector does not use `{std_out}` or `{std_err}`. Those placeholders are
for local file log collection.

The Kueue adapter and `kubectl_logs` connector are configured independently.
Settings under `[eca.kueue]` are not inherited by
`[log.<source>.kubectl_logs]`. Repeat `kubectl_path`, `kubeconfig`, and
`context` when both components must use the same non-default values.

Omitting `max_bytes_per_pod` keeps the default 16 MiB cap. It does not disable
truncation. The current TOML configuration surface supports changing the cap,
but not disabling it entirely.

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


| Field             | Required | Meaning                                                                                                                 |
| ----------------- | -------- | ----------------------------------------------------------------------------------------------------------------------- |
| `endpoint`        | Yes      | Prometheus-compatible HTTP API base URL.                                                                                |
| `cluster`         | Yes      | Cluster label value used by bundled PromQL templates.                                                                   |
| `auth.type`       | No       | Optional. If present, `token` or `mtls`; same auth fields as Loki. Omit the `auth` block for unauthenticated endpoints. |
| `step_seconds`    | No       | Query step size. Defaults to `60`.                                                                                      |
| `timeout_seconds` | No       | HTTP timeout. Defaults to `60.0`.                                                                                       |
| `queries`         | No       | Custom named PromQL templates. These extend the bundled templates and override bundled templates with the same name.    |


The bundled PromQL templates expect labels such as `s_cluster` and `net_host_name`. If your metrics use different labels, override the relevant query templates.

Each template may use `{name}` (the template's own name), `{cluster}`, and `{nodes}` (a regex alternation of the context's node names). Literal `{` and `}` in PromQL label selectors must be doubled (`{{`, `}}`).

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

**Warning:** Env vars override TOML, and CLI flags override env vars. A stale `NVAB_`* in your shell will silently win against your config file. Run `env | grep NVAB_` before debugging unexpected config behavior.

Unknown `NVAB_`-prefixed variables are logged as warnings and ignored, so unrelated `NVAB_` variables from other tooling do not abort a run.

The stderr log level is controlled by the `ATTRIBUTION_LOG_LEVEL` environment variable (default `INFO`). This is a plain environment variable, not an `NVAB_` config key.

## Config Checks

List valid config keys:

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint --connector help
uv run --project blueprint --frozen nvidia-attribution-blueprint --connector eca slurm help
uv run --project blueprint --frozen nvidia-attribution-blueprint --connector eca kueue help
uv run --project blueprint --frozen nvidia-attribution-blueprint --connector log syslog loki help
uv run --project blueprint --frozen nvidia-attribution-blueprint --connector log applog kubectl_logs help
uv run --project blueprint --frozen nvidia-attribution-blueprint --connector metrics node_metrics promql help
uv run --project blueprint --frozen nvidia-attribution-blueprint --connector output db sqlite help
```

Smoke-test a config without selecting any jobs:

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint --config ./attribution-blueprint.toml
```

This builds the configured connectors and exits with zero verdicts. It does not scan the entire cluster.

## Web UI Flags

The read-side dashboard, `nvidia-attribution-blueprint-web`, is a separate binary that reads the SQLite database written by the `sqlite` output. It takes no TOML or `NVAB_` configuration.


| Flag     | Required | Default     | Meaning                                  |
| -------- | -------- | ----------- | ---------------------------------------- |
| `--db`   | Yes      | none        | Path to the attribution SQLite database. |
| `--host` | No       | `127.0.0.1` | Listen address.                          |
| `--port` | No       | `8080`      | Listen port.                             |


The web binary needs the optional `web` extra (FastAPI and uvicorn) and a built
front-end bundle. The helper script `blueprint/scripts/webui.sh` builds the
front-end and starts the server in one step, and is the recommended entry point.
To run the binary directly once the front-end is built:

```bash
uv run --project blueprint --extra web nvidia-attribution-blueprint-web --db ./attributions.db
```
