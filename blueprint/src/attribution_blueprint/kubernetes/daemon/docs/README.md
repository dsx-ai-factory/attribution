# Kubernetes daemon

`kubernetes-daemon` retains Pod-to-node placement, ownership, and selected
lifecycle evidence after Kubernetes deletes resources. Evidence collection is
independent of whether a Kueue Workload, a KAI PodGroup, both, or neither can
be resolved for a Pod.

## Architecture

One daemon runs in each workload namespace. For each observed bound Pod, it
writes Pod identity, node, owner references, and observed `NodeLost`,
`DeletionByPodGC`, or non-zero-exit evidence before asynchronous enrichment.
Within configured capacity, it retries enabled scheduler associations and
enrichment writes without requiring another Pod event. Required base,
deletion, and reconciliation write failures replay from the last accepted
watch resource version. Per-Pod enrichment is serialized, newer observations
reject stale writes, and successful associations remain queryable after Pod
deletion until retention expires.

```text
Pod, Job, and optional controller/scheduler LIST/WATCH
  -> journal selected lifecycle transitions and owner references immediately
  -> derive namespace, Pod UID, and node placement
  -> resolve the Job or JobSet owner (Kubernetes-native, no scheduler required)
  -> resolve optional scheduler-specific associations (Kueue Workload, KAI
     PodGroup, ...) via configured resource resolvers
  -> serve scheduler-neutral records over HTTP
  -> each adapter merges retained and live placement
```

Base placement, Pod UID, namespace, node, and owner references, is retained
even when ownership or scheduler resolution is still pending or fails; "owner
not resolved yet" describes the evidence rather than discarding it.

This is bounded recovery, not complete history. Capacity exhaustion can skip
enrichment for a new Pod UID while retaining its base placement, so
resource-UID queries can omit that Pod. Retries can exhaust, in-memory retries
are lost on restart, deleted relationship evidence may be unavailable, and
Pods created and deleted while the daemon is unavailable can be missed.

The Helm chart deploys a single-replica StatefulSet with a PVC, a
namespace-scoped ServiceAccount and Role, health probes, and a ClusterIP
Service. SQLite is deliberately single-writer; do not scale the StatefulSet
above one replica or mount the database from multiple writers.

Deleted placements and completed or deleted controller histories are removed
after `retentionDays`; active histories are retained. The default is 30 days.
Cleanup runs at startup and during normal store activity.

The daemon captures placement plus `NodeLost`, PodGC `DeletionByPodGC`, and
non-zero container exit evidence. Its lifecycle journal retains selected
status and attempt-identity metadata, not logs or complete resource
specifications.

## Usage

Build and publish the image:

```bash
docker build \
  -f blueprint/docker/kubernetes-daemon.Dockerfile \
  -t registry.example/kubernetes-daemon:VERSION .
docker push registry.example/kubernetes-daemon:VERSION
```

When distributing an image, preserve the applicable licenses and notices for
the Blueprint, generated client, base image, and installed dependencies. The
build does not create a consolidated notice bundle.

### Enabling scheduler integrations

By default the daemon records scheduler-neutral Pod and owner evidence only.
Enable one or both scheduler-specific resource resolvers per installation:

```bash
helm upgrade --install attribution blueprint/deploy/kubernetes-daemon \
  --namespace team-a \
  --set image.repository=registry.example/kubernetes-daemon \
  --set image.tag=VERSION \
  --set integrations.kueue.enabled=true \
  --set integrations.kai.enabled=true
```

Enabling an integration also watches the matching resource and adds its RBAC
rule (`kueue.x-k8s.io` Workloads or `scheduling.run.ai` PodGroups) to the
generated Role. Enable only schedulers whose CRDs are installed.

| Value                                  | Default   | Meaning                                                     |
| --------------------------------------- | --------- | ------------------------------------------------------------ |
| `integrations.kueue.enabled`            | `false`   | Resolve and retain the Kueue Workload associated with a Pod. |
| `integrations.kueue.workloadVersion`    | `v1beta1` | Kueue Workload API version to watch.                         |
| `integrations.kai.enabled`              | `false`   | Resolve and retain the KAI PodGroup associated with a Pod.   |
| `integrations.kai.podGroupVersion`      | `v2alpha2`| KAI PodGroup API version to watch.                            |

Other useful chart settings are:

- `clusterId`
- `watches.pods`, `watches.jobs`, and `watches.jobsets`
- `retentionDays` (default `30`)
- `storage.sqlite.size`, `storage.sqlite.storageClassName`, and
  `storage.sqlite.accessModes`
- `nodeSelector` and `tolerations`
- `queueSize`, `workers`, and `watchTimeoutSeconds`
- `jobsetVersion`

Run the daemon on a system node outside the workload failure domain. JobSet
watching is enabled by default. If the JobSet API is not installed, the daemon
disables that optional watch, reports it through `/readyz`, and remains ready
for ordinary Jobs. Set `watches.jobsets=false` to disable it explicitly.

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

When `placement_records_enabled` is `true`, attribution fails if the endpoint
cannot be read. The chart serves plain HTTP and provides no authentication or
TLS. Keep the Service private or terminate those controls at a proxy. Adapter
`auth` settings configure requests to that proxy: `token` adds credentials,
while `mtls` requires an HTTPS endpoint that terminates client TLS.

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
curl -fsS \
  'http://127.0.0.1:18080/v1/history?namespace=team-a&controller_uid=JOB_OR_JOBSET_UID'
```

The service exposes `/livez`, `/readyz`, and two namespace-scoped read
endpoints. `/v1/placements` requires `resource_uid`, which matches a resource
associated with a Pod by a configured integration. `/v1/history` accepts
`controller_uid` and returns retained observations, resources, and recursive
ownership relationships for a Job or JobSet.

## Troubleshooting

**The daemon is not ready**

Check `/readyz` and logs. Readiness requires every configured initial list to
complete and required storage writes to succeed. Verify access to Pods, Jobs,
and each enabled JobSet or scheduler API.

**Placement is missing**

The daemon cannot recover a Pod created and deleted entirely while it was
unavailable. Confirm it was installed before the workload, ran in the same
namespace, and had enough queue capacity. Then query `/v1/placements` with
the associated Kueue Workload or KAI PodGroup resource UID.

**A scheduler association never appears**

Confirm the corresponding `integrations.*.enabled` value is `true` and that
the RBAC generated by the chart includes that scheduler's API group. Base
placement remains available when scheduler resolution is incomplete, but
resource-UID lookup requires the corresponding association.

**Attribution fails when placement is enabled**

Enabled reads are intentionally fail-closed. Verify the configured endpoint is
reachable and that any proxy authentication or TLS settings are valid.

**Records disappear too early or consume too much storage**

`retentionDays` applies only to the derived placement view. Lifecycle
observations are append-only and currently have no automatic retention limit.
Size and monitor the PVC for the expected event volume. Confirm the PVC is
persistent; deleting the PVC or namespace deletes both placements and
observations.

**Node failure is not retained**

The daemon must observe the Pod update carrying `NodeLost` or
`DeletionByPodGC` before the Pod disappears. `DeletionByTaintManager` is not
classified as node failure because it can represent transient unreachability
or routine eviction.
