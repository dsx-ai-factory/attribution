# Attribution Blueprint Use Cases

These guides show complete configurations for common Attribution Blueprint
workflows.

- [Manually collected syslogs](./manually-collected-syslogs.md) uses a static
  execution context and log files saved on the blueprint host.
- [Kueue workloads](./kueue.md) discovers Kubernetes batch workloads through
  Kueue, collects pod logs with `kubectl`, and queries node syslogs from Loki.
  It includes a [commented configuration](../../examples/kueue/README.md) and
  [representative healthy and unhealthy output](./kueue-sample-output.md).
- [KAI PodGroups](./kai.md) discovers supported Job- and JobSet-owned KAI
  workloads and documents the current lifecycle and retention scope.
- [Run the Blueprint in a cluster](./in-cluster.md) deploys the Service, web UI
  and a Blueprint runner as one pod with a Helm chart, and attributes finished
  KAI or Kueue workloads on a loop.
