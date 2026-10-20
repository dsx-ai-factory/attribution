# Frequently asked questions

## What's the per-workload latency at production scale?

Attribution handles workloads from single-GPU jobs to 10k+ GPU multi-rank training runs. Latency is dominated by telemetry retrieval, not by the rules engine.

## How do you know the rules are accurate?

Bundled rules are validated with synthetic fixtures and regression tests before they ship with the product.

## How does attribution quality degrade when log coverage is incomplete?

Attribution returns a verdict from whatever signal coverage it has. A collector that errors out or returns zero observations shows up in `connector_reports` with a status and reason. Observation submission failures after collection are logged and skipped by the Blueprint, but they are not represented in `connector_reports`. Rules that depend on the missing source don't fire. Two practical consequences:

* **Coverage matters more than completeness.** Better to have applog from Loki and node_metrics from Prometheus than to wait until syslog is also wired up. Attribution returns useful verdicts on partial coverage.
* **The `connector_reports` field is the collector coverage record.** A diagnosis that ran with one collector erroring is honestly different from one that ran with all collectors green. Downstream consumers should inspect `connector_reports` and the run logs when treating verdicts as evidence.

The minimum useful set for most rules is one log source (applog or syslog) plus scheduler state. Adding node_metrics broadens cascade detection. Switch-log access is required only for the narrow multi-host NVLink cascade rules. The rest of the catalog runs without it.

## Does NVIDIA Attribution need real-time switch logs?

No. Most rules run on node-local signals (XIDs, DCGM counters via PromQL, dmesg, scheduler events). A narrow set of multi-host NVLink cascade rules requires switch logs to fire. Without switch-log access those specific rules don't fire. The rest of the catalog runs normally without real-time switch logs.

## How does the Blueprint authenticate against the Service?

The Blueprint's attribution connector calls the Service over HTTP with `service_url` and `timeout_seconds`. It does not configure bearer-token or mTLS credentials for Service requests. If your deployment requires auth in front of the Service API, enforce it outside the Service process, for example at an ingress, gateway, or sidecar.

## Can I extend the rule set?

Propose changes to the bundled rules through [CONTRIBUTING.md](https://github.com/dsx-ai-factory/attribution/blob/main/CONTRIBUTING.md). The Library and Blueprint do not offer stable external interfaces. Upstream bug reports must reproduce on the unmodified main branch, as described in [SUPPORT.md](https://github.com/dsx-ai-factory/attribution/blob/main/SUPPORT.md).

## What's the schema-versioning policy?

NVIDIA Attribution product V1 uses the Service's `/v2/...` HTTP API routes. The `/v2` route prefix is the Service API version and is independent of the product version. The compound outcome representation (`terminal_state` + `sub_domain`) is the product V1 contract. During pre-release development, SQLite schema changes are applied in place and sample databases should be regenerated with the current tooling.

## What happens when the symptoms are ambiguous?

When evidence supports multiple plausible causes, `sub_domain` is `"ambiguous"`. When evidence is insufficient to attribute a cause, `sub_domain` is `"unattributed"`. Downstream consumers branch on the normalized `terminal_state` and `sub_domain` fields.

## What hardware does NVIDIA Attribution cover?

A100, H100, L40S, B200, GB200, GB300. The same rule library runs across all six. GB200 and GB300 rule coverage continues to mature as coverage expands. No timeline commitments for future generations.

## What happens on Kueue when pods are garbage-collected before attribution runs?

Node attribution comes from live pod objects. When Kueue preempts a workload, when the JobSet/Job retention policy collects completed pods, or when an operator runs `kubectl delete pod`, those pods are gone by the time the connector queries. The Workload object survives with its `Finished` / `Evicted` conditions, so state is still derived correctly, but `nodes` ends up empty and `failed_node` is absent. This is captured by a regression test in `tests/connectors/execution_contexts/kueue/`.

## How does step-level attribution work for Slurm?

A single `sacct` invocation returns one row per Slurm scheduling entity: the head row plus one row per step. The Blueprint emits one `ExecutionContext` per head row and per numeric user step (`.0`, `.1`, etc.). Slurm-internal wrappers (`.batch`, `.extern`) are filtered. Numeric steps are attributed independently because the user's real workload lives inside them. Each step produces its own verdict, keyed by the SLUID and step number.

## Does Kueue support cover JobSet-shaped distributed training?

Yes. The Kueue execution-context adapter detects whether a Workload's owner is a `batch/v1 Job` or a `JobSet`, and selects pods accordingly (`batch.kubernetes.io/controller-uid` for Job-owned, `jobset.sigs.k8s.io/jobset-name` for JobSet-owned). JobSet v0.5+ is required.
