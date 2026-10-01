# Kueue Pod placement recorder

The recorder preserves Kueue Pod-to-node placement after Kubernetes deletes
the Pods. It is optional: without it, the Kueue adapter uses live Pods,
retained Prometheus metrics when configured, and Kubernetes Events as a
best-effort fallback.

## Architecture

One recorder runs in each workload namespace:

```text
Pod LIST/WATCH
  -> resolve Job or JobSet and Kueue Workload
  -> write one immutable placement per Pod UID to SQLite
  -> serve workload-scoped records over HTTP
  -> Kueue adapter merges retained and live placement
```

The Helm chart deploys a single-replica StatefulSet with a PVC, a
namespace-scoped ServiceAccount and Role, health probes, and a ClusterIP
Service. SQLite is deliberately single-writer; do not scale the StatefulSet
above one replica or mount the database from multiple writers.

Records are removed automatically after `retentionDays` since their last
update. The default is 30 days. Cleanup runs when the store starts and during
normal recorder and API activity.

The recorder captures placement plus `NodeLost`, PodGC
`DeletionByPodGC`, and non-zero container exit evidence. It does not retain
Pod logs or Pod specifications.

## Usage

Build and publish the image, then install the recorder before submitting the
workloads whose placement must be retained:

```bash
docker build \
  -f blueprint/docker/pod-placement-recorder.Dockerfile \
  -t registry.example/pod-placement-recorder:VERSION .
docker push registry.example/pod-placement-recorder:VERSION

YES=1 RELEASE_NAME=attribution \
  ./blueprint/scripts/install-pod-placement-recorder.sh team-a \
  --set image.repository=registry.example/pod-placement-recorder \
  --set image.tag=VERSION
```

Useful chart settings are:

- `retentionDays` (default `30`)
- `storage.sqlite.size`, `storage.sqlite.storageClassName`, and
  `storage.sqlite.accessModes`
- `nodeSelector` and `tolerations`
- `queueSize`, `workers`, and `watchTimeoutSeconds`
- `workloadVersion` and `jobsetVersion`

Run the recorder on a system node outside the workload failure domain. JobSet
support is used when its API is installed; Jobs-only clusters do not need the
JobSet CRD.

Enable fail-closed recorder reads in the Blueprint:

```toml
[eca.kueue]
namespace = "team-a"
placement_records_enabled = true

[eca.kueue.placement_recorder]
endpoint = "http://attribution-pod-placement-recorder.team-a.svc:8080"
timeout_seconds = 10
```

When `placement_records_enabled` is `true`, the endpoint is required and
attribution fails if retained placement cannot be read. Keep it `false` when
the recorder is not deployed.

Validate the installation:

```bash
kubectl -n team-a rollout status statefulset/attribution-pod-placement-recorder
kubectl -n team-a logs -l app.kubernetes.io/name=pod-placement-recorder --tail=100
kubectl -n team-a port-forward service/attribution-pod-placement-recorder 18080:8080
curl -fsS \
  'http://127.0.0.1:18080/v1/placements?namespace=team-a&workload_uid=WORKLOAD_UID'
```

The service exposes `/livez`, `/readyz`, and the scoped
`/v1/placements?namespace=...&workload_uid=...` read endpoint. It has no
built-in authentication; keep it private or put an authenticated proxy in
front of it.

## Troubleshooting

**The recorder is not ready**

Check `/readyz` and logs. Readiness requires the initial Pod list to complete
and the worker queue to remain healthy. Verify access to Pods, Kueue Workloads,
Jobs, and any installed JobSet API.

**Placement is missing**

The recorder cannot recover a Pod created and deleted entirely while it was
unavailable. Confirm it was installed before the workload, ran in the same
namespace, and had enough queue capacity. Then query `/v1/placements` with the
exact Workload UID.

**Attribution fails when placement is enabled**

Enabled reads are intentionally fail-closed. Verify the configured endpoint is
reachable from the Blueprint host and that any proxy token or mTLS settings
are valid.

**Records disappear too early or consume too much storage**

Set `retentionDays` to a positive integer appropriate for the attribution
window and confirm the PVC is persistent. Deleting the PVC or namespace
deletes the records.

**Node failure is not retained**

The recorder must observe the Pod update carrying `NodeLost` or
`DeletionByPodGC` before the Pod disappears. `DeletionByTaintManager` is not
classified as node failure because it can represent transient unreachability
or routine eviction.
