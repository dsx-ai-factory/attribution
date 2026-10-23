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
