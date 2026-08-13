# Attribution Blueprint Use Cases

These guides show complete configurations for common Attribution Blueprint
workflows.

- [Manually collected syslogs](./manually-collected-syslogs.md) uses a static
  execution context and log files saved on the blueprint host.
- [Kueue workloads](./kueue.md) discovers Kubernetes batch workloads through
  Kueue, collects pod logs with `kubectl`, and queries node syslogs from Loki.
  It includes a [commented configuration](../../examples/kueue/README.md) and
  [representative healthy and unhealthy output](./kueue-sample-output.md).
