# Static Sample Data

This directory has two local static-log sample modes:

- `static-sample.toml` is the tracked one-job smoke test.
- `applog/workload-*` and `syslog/workload-*` are the committed 118-workload sample corpus used by `docs/quickstart.md`.
- `sample-jobs.json` is the manifest for the 118-workload corpus.

The baseline 98-workload corpus uses single-symptom multi-node jobs with 32,
64, and 1024 node allocations, committed syslog evidence, and benign
application logs. Generic failed, timeout-cancelled, and prior MMU-fault samples
are intentionally excluded from the baseline set. No baseline root-cause symptom
appears more than three times. Samples identify one responsible node when the
evidence supports one. Pair-only `ibv_modify_qp` failures keep the endpoint pair
without assigning either endpoint as the responsible node.

The additional 20 workloads are a repeat-offender demo. They use 20 distinct
users, 1 to 10 nodes per job, and the same responsible node (`node-0101`) for
every XID-31 event so the UI can show repeat-offender detection across jobs.

Hardware Fault coverage includes XID-48, XID-64, XID-74, XID-79, XID-95,
XID-110, XID-119, XID-120, XID-121, XID-140, XID-143, XID-149, XID-156,
XID-171, XID-172, and GSP allocation failure. Non-XID coverage also includes
SLURM node faults, system power-down node faults, IB device-open failures,
IB Error-12, network route and link failures, Lustre client eviction, GPU lost through
SLURM/NVML, CUDA illegal address, CUDA devices unavailable, NUMA memory
exhaustion, OOM kills, fabric manager absence, NMI watchdog lockups, local
catastrophic errors, general protection faults, and kernel invalid memory access.

The expected local layout is:

```text
applog/workload-0001/application.log
syslog/workload-0001/node-0001/syslog.log
```

The quickstart runs from the release bundle root. From that location, it uses
the committed per-workload TOML files under `blueprint/examples/static/configs`,
then runs:

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --configs blueprint/examples/static/configs \
  --connector attribution service_url "$ATTRIBUTION_SERVICE_URL" \
  --connector output db sqlite path "$ATTRIBUTION_DB"
```

If you use `static-sample.toml` directly, the output database should contain
exactly one attribution. Use the config directory for the full sample set.

`static-sample.toml` is intended for the source checkout root. The per-workload TOML
files under `examples/static/configs` are intended for the release bundle root,
where the blueprint tree lives under `blueprint/`.
