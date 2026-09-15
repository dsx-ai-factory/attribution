# Kueue Example

This example discovers Kueue Workloads, collects owned pod logs with
`kubectl logs`, queries node syslogs from Loki, and stores results in SQLite.

Before running it:

1. Replace every `CHANGE_ME` value in
   [attribution-blueprint.kueue.toml](./attribution-blueprint.kueue.toml).
2. Set `workload.gpus_per_node` to the workload's actual per-node GPU count.
3. Confirm that the Kueue Workload still exists. Owned pods are required for
   `kubectl logs`, but placement can be recovered after pod deletion.
4. Confirm the Blueprint host can reach the Attribution Service and Loki.
5. For historical placement, install the recorder before workloads run and
   schedule it outside their failure domain. When using its SQLite backend,
   configure `[eca.kueue.placement_recorder]` with a reachable recorder
   Service URL. Alternatively, configure kube-state-metrics with the
   controller-UID label allowlist. See
   [Configuration](../../docs/CONFIGURATION.md#ecakueue).

The Kueue adapter and `kubectl_logs` connector do not share settings. Keep
`kubectl_path`, `kubeconfig`, and `context` consistent in both sections.
The example sets `timestamps = false` so a pod log line beginning with a rank
prefix still begins with that prefix when it reaches the service.

From the bundle root, run:

```bash
export NAMESPACE="CHANGE_ME_NAMESPACE"
export WORKLOAD="CHANGE_ME_KUEUE_WORKLOAD"

mkdir -p .attribution-kueue/observations
uv sync --project blueprint --frozen
uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --config blueprint/examples/kueue/attribution-blueprint.kueue.toml \
  --connector eca kueue ids "${NAMESPACE}/${WORKLOAD}" \
  --dump-observations .attribution-kueue/observations
```

For cluster permissions, evidence requirements, result interpretation, and
troubleshooting, see the [Kueue use-case guide](../../docs/use-cases/kueue.md).
For all connector fields, see
[Configuration](../../docs/CONFIGURATION.md#ecakueue).
