# Kubernetes daemon

`kubernetes-daemon` retains Pod-to-node placement and ownership evidence after
Kubernetes deletes the Pods, independent of whether a Kueue Workload, a KAI
PodGroup, both, or neither can be resolved for that Pod.

## Architecture

One daemon runs in each workload namespace:

```text
Pod LIST/WATCH
  -> record namespace, Pod UID, node, and owner references immediately
  -> resolve the Job or JobSet owner (Kubernetes-native, no scheduler required)
  -> resolve optional scheduler-specific associations (Kueue Workload, KAI
     PodGroup, ...) via configured resource resolvers
  -> serve scheduler-neutral records over HTTP
  -> each adapter merges retained and live placement
```

Base placement, Pod UID, namespace, node, and owner references, is retained
even when ownership or scheduler resolution is still pending or fails; "owner
not resolved yet" describes the evidence rather than discarding it.

The Helm chart deploys a single-replica StatefulSet with a PVC, a
namespace-scoped ServiceAccount and Role, health probes, and a ClusterIP
Service. SQLite is deliberately single-writer; do not scale the StatefulSet
above one replica or mount the database from multiple writers.

Records are removed automatically after `retentionDays` since their last
update. The default is 30 days. Cleanup runs when the store starts and during
normal daemon and API activity.

The daemon captures placement plus `NodeLost`, PodGC `DeletionByPodGC`, and
non-zero container exit evidence. It does not retain Pod logs or Pod
specifications.

## Usage

Build and publish the image, then install the daemon before submitting the
workloads whose placement must be retained:

```bash
docker build \
  -f blueprint/docker/kubernetes-daemon.Dockerfile \
  -t registry.example/kubernetes-daemon:VERSION .
docker push registry.example/kubernetes-daemon:VERSION

helm upgrade --install attribution blueprint/deploy/kubernetes-daemon \
  --namespace team-a --create-namespace \
  --set image.repository=registry.example/kubernetes-daemon \
  --set image.tag=VERSION
```

For kind, load the locally built image with
`kind load docker-image kubernetes-daemon:local --name CLUSTER` and keep the
chart's default image values.

### Enabling scheduler integrations

By default the daemon records scheduler-neutral Pod and owner evidence only.
Enable one or both scheduler-specific resource resolvers per installation:

```bash
helm upgrade --install attribution blueprint/deploy/kubernetes-daemon \
  --namespace team-a \
  --set integrations.kueue.enabled=true \
  --set integrations.kai.enabled=true
```

Enabling an integration also adds the matching RBAC rule (`kueue.x-k8s.io`
Workloads or `scheduling.run.ai` PodGroups) to the generated Role. Enable only
the scheduler(s) actually present in the cluster; an absent CRD causes that
resolver to retain owner-level evidence without the scheduler association,
not a crash.

| Value                                  | Default   | Meaning                                                     |
| --------------------------------------- | --------- | ------------------------------------------------------------ |
| `integrations.kueue.enabled`            | `false`   | Resolve and retain the Kueue Workload associated with a Pod. |
| `integrations.kueue.workloadVersion`    | `v1beta1` | Kueue Workload API version to watch.                         |
| `integrations.kai.enabled`              | `false`   | Resolve and retain the KAI PodGroup associated with a Pod.   |
| `integrations.kai.podGroupVersion`      | `v2alpha2`| KAI PodGroup API version to watch.                            |

Other useful chart settings are:

- `retentionDays` (default `30`)
- `storage.sqlite.size`, `storage.sqlite.storageClassName`, and
  `storage.sqlite.accessModes`
- `nodeSelector` and `tolerations`
- `queueSize`, `workers`, and `watchTimeoutSeconds`
- `jobsetVersion`

Run the daemon on a system node outside the workload failure domain. JobSet
ownership support is used when its API is installed; Jobs-only clusters do not
need the JobSet CRD.

### Configuring each adapter's endpoint

Both the Kueue and KAI adapters read from the same `kubernetes-daemon`
installation. Point each at the daemon Service in its namespace:

```toml
[eca.kueue]
namespace = "team-a"
placement_records_enabled = true

[eca.kueue.kubernetes_daemon]
endpoint = "http://attribution-kubernetes-daemon.team-a.svc:8080"
timeout_seconds = 10
```

```toml
[eca.kai]
namespace = "team-a"
placement_records_enabled = true

[eca.kai.kubernetes_daemon]
endpoint = "http://attribution-kubernetes-daemon.team-a.svc:8080"
timeout_seconds = 10
```

When `placement_records_enabled` is `true` for either adapter, the endpoint is
required and attribution fails if retained placement cannot be read. Keep it
`false` when the daemon is not deployed for that namespace. Each `[eca.*.kubernetes_daemon]`
section accepts the same optional `auth` block (`token` with `token_env`, or
`mtls` with `cert_path`/`key_path`/`ca_cert_path`); the daemon's API has no
built-in authentication, so keep its Service private or place an
authenticated proxy in front of it.

### Verify the installation

```bash
kubectl -n team-a rollout status statefulset/attribution-kubernetes-daemon
kubectl -n team-a logs -l app.kubernetes.io/name=kubernetes-daemon --tail=100
```

Run `port-forward` in its own terminal (it stays in the foreground), then query it from another:

```bash
kubectl -n team-a port-forward service/attribution-kubernetes-daemon 18080:8080
```

```bash
curl -fsS \
  'http://127.0.0.1:18080/v1/placements?namespace=team-a&resource_uid=WORKLOAD_OR_PODGROUP_UID'
```

The service exposes `/livez`, `/readyz`, and the scoped `/v1/placements` read
endpoint, which requires `resource_uid` alongside `namespace`. The resource
UID matches a resource associated with that Pod by a configured integration.

## Troubleshooting

**The daemon is not ready**

Check `/readyz` and logs. Readiness requires the initial Pod list to complete
and the worker queue to remain healthy. Verify access to Pods and, when
enabled, Kueue Workloads, KAI PodGroups, Jobs, and any installed JobSet API.

**Placement is missing**

The daemon cannot recover a Pod created and deleted entirely while it was
unavailable. Confirm it was installed before the workload, ran in the same
namespace, and had enough queue capacity. Then query `/v1/placements` with
the associated Kueue Workload or KAI PodGroup resource UID.

**A scheduler association never appears**

Confirm the corresponding `integrations.*.enabled` value is `true` and that
the RBAC generated by the chart includes that scheduler's API group. Base
placement and owner evidence remain retained even when scheduler resolution
has not completed or the relevant CRD is absent.

**Attribution fails when placement is enabled**

Enabled reads are intentionally fail-closed. Verify the configured endpoint is
reachable from the Blueprint host and that any proxy token or mTLS settings
are valid.

**Records disappear too early or consume too much storage**

Set `retentionDays` to a positive integer appropriate for the attribution
window and confirm the PVC is persistent. Deleting the PVC or namespace
deletes the records.

**Node failure is not retained**

The daemon must observe the Pod update carrying `NodeLost` or
`DeletionByPodGC` before the Pod disappears. `DeletionByTaintManager` is not
classified as node failure because it can represent transient unreachability
or routine eviction.
