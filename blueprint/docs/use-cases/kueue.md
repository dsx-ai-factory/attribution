# Attribute Kueue Workloads

Use this workflow to attribute Kubernetes Jobs or JobSets that Kueue admitted.
Attribution Blueprint reads the Kueue Workload, resolves its Job or JobSet
owner, and builds Pod placement from live Pods and any configured historical
sources. The included configuration also collects application logs with
`kubectl logs` and queries Loki for node-system logs.

The files for this use case are:

- [examples/kueue](../../examples/kueue/README.md), the runnable and fully
  commented configuration
- [kueue-sample-output.md](./kueue-sample-output.md), representative healthy
  and unhealthy results

## Prerequisites

The blueprint host must have:

- network access to a running Attribution Service
- `kubectl` access to the target cluster
- permission to read Kueue Workloads, their Job or JobSet owners, pods

Optional capabilities require:

- permission to run `kubectl logs` for the workload pods
- network access and credentials for Loki when collecting node syslogs
- network access to the Pod placement recorder or Prometheus when either
  fallback placement source is configured
- permission to read Kubernetes Events for the best-effort placement
  fallback

The Kueue Workload must still exist when attribution runs. Run attribution
before its owned Pods are deleted when application logs are required:
Pod and node identity can be recovered, but no source can restore `kubectl logs`
for a deleted Pod.

Pod Placement sources are merged in this order: live Pods, the Pod placement
recorder, and Prometheus metrics. If Pod or node names are still missing, the
adapter makes a best-effort recovery from short-lived Kubernetes Events.
Event recovery is a heuristic: it parses human-readable Event messages, which
have no versioned schema.

Loki configuration requires an auth block. The included configuration uses
mTLS, so its client certificate and private key must exist on the blueprint
host and be readable by the account running the CLI.

## Identify the Kueue Workload

The CLI accepts Kueue Workload names, not the names of the owning Job or
JobSet. Set these shell variables to the target values:

```bash
KUBECONFIG_PATH="/CHANGE_ME_PATH/kubeconfig"
KUBE_CONTEXT="CHANGE_ME_KUBECTL_CONTEXT"
NAMESPACE="CHANGE_ME_NAMESPACE"
WORKLOAD="CHANGE_ME_KUEUE_WORKLOAD"
```

Confirm the Workload exists and inspect its owner and terminal conditions:

```bash
kubectl --kubeconfig "${KUBECONFIG_PATH}" \
  --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" \
  get workload "${WORKLOAD}" \
  --output yaml
```

The Workload should have an owner reference to a `Job` or `JobSet`. Attribution
Blueprint uses the owner to find pods, node placement, exit codes, and GPU
requests. Unsupported owners can produce a context without pod evidence.

## Configure the Use Case

Edit
[examples/kueue/attribution-blueprint.kueue.toml](../../examples/kueue/attribution-blueprint.kueue.toml)
and replace every `CHANGE_ME` value.

Baseline settings:

| Entry | When needed | Value |
|---|---|---|
| `attribution.service_url` | Every run | Base URL of the Attribution Service. |
| `workload.gpus_per_node` | For correct rank-to-node mapping | GPUs, or training ranks, allocated per node. |
| `eca.component` | Every Kueue run | `kueue`, which selects Kueue Workload discovery. |
| `eca.kueue.kubeconfig` | When the kubectl default is not suitable | Kubeconfig file for the target cluster. |
| `eca.kueue.context` | When the current context is not suitable | Kubeconfig context for the target cluster. |
| `eca.kueue.namespace` | For bare Workload names | Namespace containing the target Workload. Explicit `namespace/name` IDs do not require it. |

Optional evidence and output settings:

| Entry | When needed | Value |
|---|---|---|
| `log.applog.component` | To collect Pod logs | `kubectl_logs`. |
| `log.applog.kubectl_logs.kubeconfig` | When the kubectl default is not suitable | The same kubeconfig file used for discovery. |
| `log.applog.kubectl_logs.context` | When the current context is not suitable | The same kubeconfig context used for discovery. |
| `log.applog.kubectl_logs.timestamps` | For application-provided rank prefixes | `false`, so kubectl does not insert a timestamp before the rank prefix. |
| `log.syslog.component` | To collect node-system evidence | `loki`. |
| `log.syslog.loki.endpoint` | When Loki is enabled | Base URL of the Loki service. |
| `log.syslog.loki.query` | When Loki is enabled | LogQL selector for journal streams from the target cluster. |
| `log.syslog.loki.auth.*` | When Loki is enabled | An authentication block; the example uses an mTLS certificate and key. |
| `output.db.component` | For the persisted sample-output workflow | `sqlite`. |
| `output.db.sqlite.path` | When SQLite output is enabled | Location of the result database. |

The doubled braces in the `query` value are intentional. They produce literal
LogQL braces after template rendering. When pods provide node names, the Loki
connector injects an additional regular-expression filter for those nodes. It
uses the default `host_name` stream label as the node name, so the Loki streams
must provide `host_name` values that exactly match Kubernetes node names.

Set `workload.gpus_per_node` to the workload's actual per-node GPU allocation.
The blueprint assumes one training rank per GPU and uses this value to build the
rank-to-node mapping sent to the Attribution Service. Omitting it silently uses
the default of four GPUs per node.

If node syslogs are unavailable, remove the complete `[log.syslog]`,
`[log.syslog.loki]`, and `[log.syslog.loki.auth]` sections. Pod-log attribution
can still run, but hardware diagnoses that depend on kernel logs, including
XID-149.4, will not be available unless the same evidence appears elsewhere.
See [Evidence Coverage](../CONFIGURATION.md#evidence-coverage) for the accuracy
tradeoffs when APPLOG or SYSLOG is unavailable.

Relative paths in the configuration are resolved from the shell's current
working directory, not from the directory containing the TOML file.

The Kueue adapter and pod-log connector do not inherit configuration from each
other. Repeat non-default `kubeconfig` and `context` values in both sections.
`kubectl_path` applies only to the pod-log connector; the Kueue adapter uses the
Kubernetes Python API client.

The current service parser expects an APPLOG rank prefix at the start of a raw
log line. `kubectl --timestamps=true` inserts its timestamp before that prefix,
which prevents rank parsing and can prevent application rules from matching.
The example therefore sets `timestamps = false`. This setting only controls
the prefix added by `kubectl`; the application log itself must still put its
rank prefix first.

### Recover Pod placement after deletion

The recorder and Prometheus sources are optional. The baseline adapter can use
live Pods and the final Kubernetes Events fallback without either source.

#### Pod placement recorder

The namespace-scoped recorder stores Pod placement in SQLite. Install one recorder
in each workload namespace before submitting workloads whose placement must
survive Pod deletion. Schedule it outside the workload failure domain so it can
observe the final Pod update.

Enable recorder reads in the Blueprint:

```toml
[eca.kueue]
placement_records_enabled = true

[eca.kueue.placement_recorder]
endpoint = "http://attribution-pod-placement-recorder.team-a.svc:8080"
timeout_seconds = 10
```

Recorder reads are fail-closed. When `placement_records_enabled = true`, the
endpoint is required and attribution fails if it cannot be read. The recorder
retains Pod UID, name, node, and observed `NodeLost`, `DeletionByPodGC`, or
non-zero-exit evidence. It does not retain Pod logs or Pod specifications.

See the
[recorder reference](../../src/attribution_blueprint/connectors/execution_contexts/kueue/pod_placement/recorder/docs/README.md)
for installation, retention, RBAC, and troubleshooting.

#### Prometheus metrics

The adapter can recover placement from retained `kube_pod_info` and
`kube_pod_labels` metrics. Configure `kube-state-metrics` to expose the controller
Job UID:

```yaml
metricLabelsAllowlist:
  - pods=[batch.kubernetes.io/controller-uid]
```

Then configure the Prometheus-compatible endpoint:

```toml
[eca.kueue.prometheus_metrics]
endpoint = "https://prometheus.example"
cluster = "prod-batch"
cluster_label = "cluster"
lookback_padding_seconds = 300
batch_size = 20
```

`cluster` and `cluster_label` are optional for a single-cluster endpoint but
must be set together for a shared backend. Authentication is optional and
supports token and mTLS.

When configured, Prometheus placement is always queried and merged; it is not
limited to Workloads whose live placement appears incomplete. Query failures
are fail-open: the adapter logs a warning and continues with other sources.
Prometheus restores placement only, not Pod logs, exit status, or node-loss
conditions.

## Verify Data Access

Confirm that the adapter can read the Workload and that pods still exist:

```bash
kubectl --kubeconfig "${KUBECONFIG_PATH}" \
  --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" \
  get workload "${WORKLOAD}" \
  --output json

kubectl --kubeconfig "${KUBECONFIG_PATH}" \
  --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" \
  get pods \
  --output wide
```

The second command lists every pod in the namespace. Use the Workload owner and
pod labels to confirm which pods belong to the workload. For a Job, Kubernetes
uses `batch.kubernetes.io/controller-uid`. For a JobSet, the adapter uses
`jobset.sigs.k8s.io/jobset-name`.

To verify recorder recovery, query the recorder with the exact Workload UID:

```bash
kubectl -n "${NAMESPACE}" port-forward \
  service/attribution-pod-placement-recorder 18080:8080
curl -fsS \
  "http://127.0.0.1:18080/v1/placements?namespace=${NAMESPACE}&workload_uid=WORKLOAD_UID"
```

To verify Prometheus recovery, confirm this query returns the target Job UID
after at least one kube-state-metrics scrape:

```promql
kube_pod_labels{
  namespace="CHANGE_ME_NAMESPACE",
  label_batch_kubernetes_io_controller_uid="CHANGE_ME_JOB_UID"
}
```

## Run Attribution

From the bundle root, create the output directory and run the CLI:

```bash
mkdir -p .attribution-kueue/observations

uv sync --project blueprint --frozen
uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --config blueprint/examples/kueue/attribution-blueprint.kueue.toml \
  --connector eca kueue ids "${NAMESPACE}/${WORKLOAD}" \
  --dump-observations .attribution-kueue/observations
```

The command performs these steps:

1. Reads the Kueue Workload and derives its state and execution interval.
2. Follows the Job or JobSet owner and discovers pods, nodes, and GPU requests.
3. Collects application logs from all containers in each pod.
4. Queries Loki over the same interval, scoped to the discovered nodes.
5. Sends the normalized workload and observations to the Attribution Service.
6. Prints the CLI summary and stores the result in
   `.attribution-kueue/attributions.db`.

To process several Workloads, pass a comma-separated list:

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --config blueprint/examples/kueue/attribution-blueprint.kueue.toml \
  --connector eca kueue ids \
  "${NAMESPACE}/workload-a,${NAMESPACE}/workload-b"
```

Missing Workload names are skipped. If no supplied names resolve, the CLI
reports `0 execution contexts discovered`.

For time-range discovery with optional node filtering, see
[the Kueue configuration reference](../CONFIGURATION.md#ecakueue).

## Interpret the Result

See [Kueue sample output](./kueue-sample-output.md) for complete, annotated
examples. A successful CLI run and a healthy workload are different concepts:

- `Attribution complete: 1/1 execution contexts succeeded` means the pipeline
  obtained and wrote a result.
- `terminal_state=pass`, `sub_domain=none`, and `root_symptom=the process completes` indicate a healthy
  workload result. This can come from recognized completion evidence in the
  collected application log or, when no stronger symptoms exist, the
  service-facing `COMPLETED` state.
- `terminal_state=fail`, `sub_domain=hardware`, and an XID root symptom indicate an unhealthy workload
  even when the attribution pipeline itself completed successfully.

Kueue derives `FINISHED_SUCCEEDED` from a `Finished=True` condition with reason
`Succeeded`. The adapter normalizes it to the service-facing state `COMPLETED`
and preserves the original value as `kueue_state` in `raw_scheduler_json`. For
`FINISHED_FAILED`, the adapter uses the direct Job or JobSet child Job reason
to normalize the service-facing state to `TIMEOUT` or `FAILED`. Legacy
`NodeLost` pod status and `DisruptionTarget=True` with
`DeletionByPodGC` normalize the state to `NODE_FAIL`, even if Kueue remains
`ADMITTED`. Recorder records can preserve this node-loss evidence and the
failed node after Pod deletion. Prometheus placement alone cannot restore
Pod-status evidence or derive `NODE_FAIL`. The adapter also normalizes
`EVICTED_PREEMPTED` and
`PREEMPTED` to the service-facing state `PREEMPTED`, and
`EVICTED_NODEFAILURES` to `NODE_FAIL`. A native `Requeued=True` condition uses
the existing service state `REQUEUED`. Other eviction reasons (queue
stopped, admission check failed, pods-ready timeout, deactivated, ...)
normalize to the generic service state `EVICTED`, which produces
`Incomplete Info` without stronger evidence. SQLite writes the service-facing state to
`execution_context.status`. Scheduler state and the attribution verdict answer
different questions, so inspect both. When no meaningful failure symptoms are
present, the Attribution Service recognizes `COMPLETED` as successful and
produces the fallback symptom `the process completes`.

## Troubleshooting

### No execution contexts are discovered

Confirm the kubeconfig context, namespace, and Kueue Workload name. A bare
Workload name requires `eca.kueue.namespace`; an explicit
`namespace/workload-name` reference is unambiguous.

### Pod-log observation count is zero

Confirm that owned pods still exist and that `kubectl logs` is authorized. The
adapter supports standard Job and JobSet ownership. Pod garbage collection or
an unsupported owner can leave no pod names for the log connector. The adapter
can recover Pod names from recorder records, retained Prometheus metrics, or
Kubernetes Events, but `kubectl logs` still returns `NotFound` after the Pod
object is deleted.

### Loki returns no syslog observations

Check the endpoint, mTLS files, cluster selector, and the default `host_name`
node label. The `host_name` value must exist on Loki streams and match the
Kubernetes node names discovered from pods. A connector report with `status="ok"` and
`observation_count=0` means the request completed but returned no evidence.

### Only the end of a pod log is present

The pod output exceeded the default 16 MiB per-pod limit. The connector retains
the trailing bytes because crash symptoms commonly appear near the end. Add the
optional `max_bytes_per_pod` setting when a different limit is required.
Removing the setting keeps the 16 MiB default and does not disable the cap.

For every Kueue connector option, see
[Configuration](../CONFIGURATION.md#ecakueue).
