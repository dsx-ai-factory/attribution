# Attribute KAI PodGroups

The KAI execution-context adapter discovers PodGroups owned by Kubernetes
`Job` or `JobSet` resources. It validates owner UIDs, joins member Pods, and
uses the owner lifecycle to report completion, failure, timeout, and end time.

Configure the adapter with:

```toml
[eca]
component = "kai"

[eca.kai]
namespace = "CHANGE_ME_NAMESPACE"
kubeconfig = "CHANGE_ME_KUBECONFIG"
context = "CHANGE_ME_CONTEXT"
```

Use `<namespace>/<podgroup-name>` for ID discovery. Time-range discovery
requires `eca.kai.namespace`.

## Recover Pod placement after deletion

The baseline adapter uses live Pods only. If Pods are deleted before
attribution runs, the resulting context can have missing `nodes` or
`pod_names`.

### Deploy kubernetes-daemon with the KAI integration enabled

Install one `kubernetes-daemon` release per workload namespace, with the KAI
integration enabled, before submitting workloads whose placement must survive
Pod deletion. Schedule it outside the workload failure domain so it can
observe the final Pod update.

```bash
docker build \
  -f blueprint/docker/kubernetes-daemon.Dockerfile \
  -t registry.example/kubernetes-daemon:VERSION .
docker push registry.example/kubernetes-daemon:VERSION

helm upgrade --install attribution blueprint/deploy/kubernetes-daemon \
  --namespace team-a --create-namespace \
  --set image.repository=registry.example/kubernetes-daemon \
  --set image.tag=VERSION \
  --set integrations.kai.enabled=true
```

The same installation, with the Kueue integration also enabled, serves the
Kueue adapter; see [Attribute Kueue Workloads](./kueue.md).

### Configure the adapter's endpoint

```toml
[eca.kai]
namespace = "team-a"
placement_records_enabled = true

[eca.kai.placement_recorder]
endpoint = "http://attribution-kubernetes-daemon.team-a.svc:8080"
timeout_seconds = 10
```

`endpoint` is required when this section is present; `timeout_seconds`
defaults to `10`. Optional `auth` supports `token` (`token_env`, optional
`org_id`) or `mtls` (`cert_path`, `key_path`, optional `ca_cert_path` and
`org_id`). Daemon reads are fail-closed: when `placement_records_enabled =
true`, the endpoint is required and attribution fails if it cannot be read.
The daemon retains Pod UID, name, node, owner, and observed `NodeLost`,
`DeletionByPodGC`, or non-zero-exit evidence scoped by PodGroup UID. It does
not retain Pod logs or Pod specifications.

### Verify recovery

```bash
kubectl -n team-a port-forward service/attribution-kubernetes-daemon 18080:8080
curl -fsS \
  'http://127.0.0.1:18080/v1/placements?namespace=team-a&resource_uid=PODGROUP_UID'
```

After deleting a completed Pod, a validation run should retain the same Pod
UID and node with `pod_placements[].source = "kubernetes_daemon"`. For
node-loss validation, `nodeFailureReason` must be present before the Pod
disappears; post-hoc attribution should then return `NODE_FAIL` and the
recorded `failed_node`.

See the
[kubernetes-daemon reference](../../src/attribution_blueprint/kubernetes/daemon/docs/README.md)
for architecture, retention, RBAC, and troubleshooting.

## Current Scope

- Supported owners: Kubernetes Jobs and JobSets.
- Supported normalized states: `RUNNING`, `COMPLETED`, `FAILED`, `TIMEOUT`,
  and `NODE_FAIL`.
- Job deadlines are forwarded from `activeDeadlineSeconds`; for JobSets, the
  shortest verified child deadline and earliest child failure reason are used.
- `PREEMPTED`, `EVICTED`, and `REQUEUED` are not detected yet.
- A context represents the lifetime of one PodGroup incarnation, not an
  individual scheduling attempt.

The owning Job or JobSet must still exist when discovery runs. KAI PodGroups
do not retain authoritative completion state or end time, so the adapter skips
a PodGroup after its owner is deleted even if Pods or placement evidence
remain. Run attribution within the owner-retention window. Retaining owner
lifecycle metadata for later historical attribution is planned follow-up work.
