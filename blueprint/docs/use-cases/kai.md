# Attribute KAI PodGroups

The KAI execution-context adapter discovers PodGroups owned by Kubernetes
`Job` or `JobSet` resources. It validates owner UIDs, joins member Pods, and
uses the owner lifecycle to report completion, failure, timeout, and end time.

To run the Blueprint inside the cluster instead of from an operator host, see
[Run the Blueprint in a cluster](./in-cluster.md).

## Prerequisites

The Kubernetes identity used by the Blueprint needs `get` and `list` access to
KAI PodGroups and Pods, `get` access to Jobs, and, for JobSets, `get` access to
JobSets plus `list` access to their Jobs. Confirm the PodGroup CRD and access:

```bash
kubectl get crd podgroups.scheduling.run.ai \
  -o jsonpath='{.spec.versions[?(@.served==true)].name}{"\n"}'
kubectl auth can-i get podgroups.scheduling.run.ai -n team-a
kubectl auth can-i list podgroups.scheduling.run.ai -n team-a
kubectl auth can-i list pods -n team-a
kubectl auth can-i get jobs.batch -n team-a
# JobSet-owned PodGroups only:
kubectl auth can-i get jobsets.jobset.x-k8s.io -n team-a
kubectl auth can-i list jobs.batch -n team-a
```

Use the served PodGroup version for
`integrations.kai.podGroupVersion` when deploying `kubernetes-daemon`.

## Configure and run

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

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --config ./attribution-blueprint.toml \
  --connector eca kai ids "team-a/podgroup-a"
```

## Recover Pod placement after deletion

The baseline adapter uses live Pods only. If Pods are deleted before
attribution runs, the resulting context can have missing `nodes` or
`pod_names`.

### Deploy kubernetes-daemon with the KAI integration enabled

Install one daemon per workload namespace as described in the
[daemon reference](../../src/attribution_blueprint/kubernetes/daemon/docs/README.md),
with `integrations.kai.enabled=true`, before submitting workloads.

### Configure the adapter's endpoint

```toml
[eca.kai]
namespace = "team-a"
placement_records_enabled = true

[eca.kai.kubernetes_daemon]
endpoint = "http://attribution-kubernetes-daemon.team-a.svc:8080"
timeout_seconds = 10
```

`endpoint` is required when this section is present; `timeout_seconds`
defaults to `10`. The chart serves plain HTTP without authentication; `auth`
only configures requests to a terminating proxy, and `mtls` requires an HTTPS
proxy endpoint. Daemon reads are fail-closed: when
`placement_records_enabled = true`, attribution fails if the endpoint cannot
be read.

### Verify recovery

For a Pod observed and associated before deletion, a validation run should
return the same Pod UID and node with
`pod_placements[].source = "kubernetes_daemon"`. For node-loss validation,
`nodeFailureReason` must be observed before deletion. The daemon reference
defines retention and recovery limits.

## Current Scope

- Supported owners: Kubernetes Jobs and JobSets.
- Supported normalized states: `RUNNING`, `COMPLETED`, `FAILED`, `TIMEOUT`,
  and `NODE_FAIL`.
- Job deadlines are forwarded from `activeDeadlineSeconds`; for JobSets, the
  shortest verified child deadline and earliest child failure reason are used.
- `PREEMPTED`, `EVICTED`, and `REQUEUED` are not detected yet.
- A context represents the lifetime of one PodGroup incarnation, not an
  individual scheduling attempt.
- Hierarchical PodGroups are emitted with `podgroup_shape = "hierarchical"`;
  subgroup-specific attempt semantics are not inferred.

The PodGroup and its owning Job or JobSet must still exist when discovery
runs. The KAI adapter uses daemon evidence for placement recovery, not
discovery of deleted PodGroups. Run attribution within both resources'
retention windows.
