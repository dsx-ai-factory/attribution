# Run the Blueprint in a Cluster

This guide deploys NVIDIA Attribution inside the Kubernetes cluster whose
workloads it attributes. The chart at
[`blueprint/deploy/attribution`](../../deploy/attribution) has three components,
each with its own enable flag:

| Component | What it is | Port |
|---|---|---|
| `service` | The Attribution Service: the rules engine and its HTTP API | 8000 |
| `runner` | The Blueprint on a loop: discovers workloads, collects evidence, asks the Service, writes results | none |
| `dashboard` | The web UI that reads the results database | 8080 |

By default all three run in one pod, the bundled deployment. To give them
separate lifecycles, install the chart more than once with different components
enabled. The [examples](../../deploy/attribution/examples) show each shape.

The pod is a one-replica StatefulSet: the results database is SQLite, which has
a single writer, on a `ReadWriteOnce` volume that the runner and the dashboard
share, and the Service keeps its sessions in memory.

## Prerequisites

- An image built from [`blueprint/docker/Dockerfile`](../../docker/Dockerfile),
  pushed to a registry the cluster can pull from. The chart has no default image.
- For the runner, the prerequisites of the adapter you choose:
  [KAI](./kai.md#prerequisites) or [Kueue](./kueue.md#prerequisites). The chart
  creates the read-only access they describe.
- For the default `kubectl_logs` connector, an image that contains `kubectl` (see
  [Logs](#logs)).
- A storage class with a `ReadWriteOnce` volume, when the runner or the dashboard
  is enabled.

## Install the bundled deployment

```yaml
# values.yaml
image:
  repository: registry.example/attribution
  tag: "1.0.0"
namespaces: [team-a, team-b]
eca:
  component: kai   # or kueue
```

```bash
helm install attribution blueprint/deploy/attribution \
  --namespace attribution --create-namespace --values values.yaml
kubectl -n attribution port-forward statefulset/attribution 8080:8080
kubectl -n attribution logs statefulset/attribution -c runner
```

The object names start with the release name, so a release named `attribution` gives `statefulset/attribution`.

## Deploy the components separately

Each component has `enabled`, and optionally its own `image`, `command` and
`args`. The chart renders and validates only what is enabled: a Service-only
release needs no `namespaces`, no scheduler settings, no RBAC and no volume.

| Goal | Example values |
|---|---|
| Everything in one pod | [`bundled.yaml`](../../deploy/attribution/examples/bundled.yaml) |
| Only the Service, for other installations to call | [`service-only.yaml`](../../deploy/attribution/examples/service-only.yaml) |
| A runner (and dashboard) that calls a Service elsewhere | [`runner-external-service.yaml`](../../deploy/attribution/examples/runner-external-service.yaml) |
| A complete Blueprint TOML file you provide | [`user-provided-toml.yaml`](../../deploy/attribution/examples/user-provided-toml.yaml) |
| Only the dashboard, on an existing database claim | [`dashboard-only.yaml`](../../deploy/attribution/examples/dashboard-only.yaml) |

### The Service

Set Service environment variables, such as the DuckDB limits in the operator
guide, with `service.env` and `service.envFrom`, which take the Kubernetes
`EnvVar` and `EnvFromSource` shapes, so a value can come from an existing
Secret. Mount extra volumes with `extraVolumes` and `service.extraVolumeMounts`.

By default the Service listens on the pod's `127.0.0.1` and has no Kubernetes
Service, because only the runner in the same pod calls it. Set
`service.api.enabled: true` to create a `ClusterIP` Service named
`<release>-attribution-api` on `service.port` and listen on all addresses. The
chart then requires `networkPolicy.service.from` to say who may call it.

The Service is probed at `/v2/healthz`: over HTTP when its API is exposed, and by
a command inside the container when it listens on localhost, because the kubelet
connects to the pod address.

### The runner

`runner.serviceUrl` is the Service the runner calls. Empty means the Service in
the same pod. It is required when `service.enabled` is `false`. For a Service in
another release, use its API Service name (`attribution-service-api` for a release named `attribution-service`, `<release>-attribution-api` otherwise):

```yaml
runner:
  serviceUrl: http://attribution-service-api.attribution-service.svc:8000
```

### The dashboard

The dashboard has its own probes, `dashboard.probes`, and reads
`dashboard.databasePath`. A dashboard-only release reads a database through
`persistence.existingClaim`.

## Use your own Blueprint configuration

By default the chart generates the Blueprint TOML from the `eca`, `logs`,
`workload` and `extraConfig` values. To provide a complete file instead, point
`config.existing` at a key of an existing ConfigMap or Secret:

```yaml
config:
  existing:
    configMap: {name: my-blueprint, key: blueprint.toml}
```

The chart mounts the file as the whole configuration. It generates no TOML and
rejects `extraConfig`, so put every setting in your file. The runner still adds
the namespace and the time window of each pass on the command line, so
`eca.component` must name the adapter your file configures, and the file must
write the results database to the path the dashboard reads
(`/data/attributions.db` by default).
[`user-provided-toml-configmap.yaml`](../../deploy/attribution/examples/user-provided-toml-configmap.yaml)
is a complete file you can start from.

## How a pass works

Every `runner.intervalSeconds`, for each namespace, the runner runs the Blueprint
in time-range mode over the window from that namespace's last mark to
`now - runner.settleSeconds`. It moves the mark forward when the Blueprint exits
successfully.

Know these limits before you rely on the results:

- **Running workloads are attributed too.** The adapters select every workload
  whose lifetime overlaps the window, and a running workload's lifetime ends at
  "now". It is attributed with the state `RUNNING`, and again by each later pass
  until it finishes. `runner.settleSeconds` only holds back a workload that
  started in the last few minutes. The last run of a workload reflects its
  terminal state.
- **Each attribution adds a row.** Attributing a workload again updates its
  `execution_context` row and adds an `attribution_run` row, so the web UI can
  show several runs for one workload.
- **A partial failure moves the mark.** The Blueprint exits with an error only
  when every context fails or discovery is incomplete. If some workloads fail
  and others succeed, the mark moves forward and the failed workloads are not
  retried by a later pass unless they are still running.
- **The first pass looks back `runner.initialLookbackSeconds`.** Keep it inside
  how long your cluster keeps finished workload objects. The adapters skip a
  workload once its owning Job or JobSet is gone, and pod logs exist only while
  the pods do. It must be larger than `runner.settleSeconds`.

## Logs

`logs.connector` names the pod-log component, `kubectl_logs` by default. It runs
`kubectl logs` with the pod's service account, so the image must contain
`kubectl`, and the runner exits with a message at start if it does not. A thin
image on top of the bundle image is enough:

```dockerfile
FROM registry.example/attribution:1.0.0
COPY --from=registry.k8s.io/kubectl:v1.33.0 /bin/kubectl /usr/local/bin/kubectl
```

Use a `kubectl` version within one minor version of your API server. The pod's
logs exist only while the pod does, and the container runtime rotates them. For
syslog or metrics sources, add sections to `extraConfig`, or use your own file.

## Access control

- **The Service API has no authentication.** Anyone who can reach port 8000 can
  call it. In the bundled deployment it listens only on the pod's localhost. If
  you expose it, set `networkPolicy.service.from` to the sources that may call
  it, and use a CNI that enforces NetworkPolicy: without enforcement every pod in
  the cluster can reach it.
- **RBAC** exists only when the runner is enabled. It is a read-only `Role` and
  `RoleBinding` in each listed namespace: PodGroups or Workloads, Jobs, JobSets,
  Pods, Pod logs and Events, with `get` and `list`. `rbac.scope: cluster` uses a
  `ClusterRole` instead. The pod mounts a service account token only when the
  runner is enabled.
- **The NetworkPolicy** (on by default) admits no ingress except the sources you
  list for each port. The chart refuses to render an exposed Service API or an
  Ingress that its own policy would block.
- **The web UI has no login.** It shows workload names, node names and error text
  taken from logs. Put authentication in front of the Ingress, or keep it behind
  the port-forward.

## What the chart does not do

- It runs the `kai` and `kueue` adapters. The `static` and `slurm` adapters read
  a host's files or scheduler and do not apply inside a cluster.
- It does not install [`kubernetes-daemon`](../../src/attribution_blueprint/kubernetes/daemon/docs/README.md),
  which keeps pod placement after pods are deleted. Without it, a workload whose
  pods are already gone loses its node and pod names.
- It runs one replica for each release and does not scale out.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Install fails with a message naming a value | The chart rejects combinations that cannot work, such as a runner with no `namespaces` or an exposed API with no `networkPolicy.service.from`. Read the message. |
| Runner exits with `kubectl is not in this image` | `logs.connector` is `kubectl_logs` and the image has no `kubectl`. Build the thin image in [Logs](#logs). |
| Runner log shows `waiting for the service` | The Service did not answer `/v2/healthz`. For a separate Service, check `runner.serviceUrl`, the Service's NetworkPolicy and its pod log. |
| `0 execution contexts discovered` on every pass | No workload in that namespace overlapped the window. This is normal when idle. |
| A namespace's pass fails with `forbidden` | Its `Role` is missing: the namespace is not in `namespaces`, or the release was not upgraded after adding it. |
| Log reads warn about a missing pod | The pod was deleted before the pass. Shorten `runner.intervalSeconds` and `runner.settleSeconds`, or keep finished workloads longer. |
| The dashboard is empty | The runner has not completed a pass, or your own TOML writes the database somewhere other than `dashboard.databasePath`. |
