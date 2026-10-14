# NVIDIA Attribution overview

NVIDIA Attribution turns the raw symptoms of a failed GPU workload into one structured diagnosis, with the symptom chain that produced it. This page is for the platform and SRE teams who operate large GPU clusters and gigawatt-scale AI factories. It explains what you get, how Attribution fits into your environment, what it needs from it, and where to go next.

## The problem it solves

AI training and inference workloads fail in dozens of ways. A multi-rank workload at 10k-GPU scale can produce hundreds to tens of thousands of log lines across application logs, host syslogs, scheduler database entries, GPU driver `dmesg`, fabric manager logs, and metric backends. Application logs scale with rank count, so the volume grows with the size of the workload. Triaging that by hand breaks down at production-cluster volume.

Attribution automates the cross-correlation. Each failed workload becomes one diagnosis, with the cascade rendered as the symptom chain that produced it.

## What you get

For each failed workload, Attribution returns:

- a failure class, as `terminal_state` and `sub_domain`
- the responsible node, when the cascade points there
- the specific cause, when the symptoms support it
- the chain of events that led to the failure

For example, hundreds of transport-level symptoms such as `Connection closed` and `ncclRemoteError` across 128 nodes, plus one `NCCL invalid NIC selection`, collapse to a single verdict:

```text
terminal_state:       fail
sub_domain:           system_software
```

The chain keeps context that the verdict alone does not. If a node was rebooted mid-run and an XID followed, the verdict can still say `hardware`, because the cascade matched a hardware-class rule. The chain shows the reboot upstream of the XID. You see the reboot, recognize the cause, and keep the node in production. The verdict says what the cascade matched, the chain shows the broader context, and you decide.

## How operators use it

Two cases from production use:

- **Fabric reliability.** On a large RDMA-backed GPU cluster, Attribution concentrated a noisy mix of RDMA port errors, connection resets, and scheduler "Not Responding" events into a small set of fabric patterns and specific host-port and switch-port pairs. The operator used that node-pair evidence to repair infrastructure and tune host-side congestion. Those actions improved mean time between workload interruptions by roughly 3x on that cluster. Attribution supplied the evidence that scoped the work.
- **Host-level events.** On a multi-rank GB200 training fleet, scattered job failures each looked like an isolated termination. Attribution clustered them by node, time, and workload and showed they concentrated on a subset of hosts silently entering the scheduler "Not Responding" state. A backlog of individual job tickets became one tracked infrastructure failure with a clear owner. That evidence scoped a firmware fix, and recurrence is now tracked as a named failure mode.

## How it fits into your environment

Attribution has two components that share one rules engine.

- **Attribution Service.** A self-hosted HTTP REST service that owns the rules engine and returns one diagnosis per workload.
- **Attribution Blueprint.** A reference orchestrator that discovers workloads from your scheduler, collects log and metric evidence from your telemetry stack, calls the Service, and stores the diagnosis. It includes a read-only dashboard.

The Service does not query schedulers, read logs, or scrape metrics. The Blueprint does that and sends the Service normalized observations over HTTP. You can therefore run the Service centrally and the Blueprint at the edge of each cluster. Use the Blueprint when you want a working deployment quickly, or as the starting point for your own topology. Use the Service alone when you want one shared attribution endpoint across many workloads. See [Architecture](ARCHITECTURE.md) for the full picture.

## What it does not do

- **It does not probe.** It correlates logs. It does not run diagnostic commands or test hardware, and it does not make replace-or-return-to-service decisions. Those calls are yours.
- **It does not replace your monitoring.** It consumes the signals your stack already produces and does not ingest, route, or alert on healthy workloads.
- **It does not overclaim.** When several causes fit, `sub_domain` is `ambiguous`. When the evidence is not enough, it is `unattributed`. Output quality scales with the logs the Service sees.

## What you need

- **Scheduler.** Slurm 25.11 or later, because `SLUID` is required, or Kueue v0.6 or later with JobSet v0.5 or later for JobSet-shaped distributed training. NVIDIA KAI Scheduler support is pending. The Service API accepts `kai`, but attribution for it is not implemented yet.
- **Telemetry.** Any subset of Loki for application or system logs, a PromQL-compatible Prometheus or Mimir for node metrics, `kubectl` access for Kueue pod logs, or local log files.
- **Network.** The Blueprint must reach the Service over HTTP. The Service needs no outbound access.
- **Hardware coverage.** A100, H100, L40S, B200, GB200, and GB300. GB200 and GB300 rule coverage is still maturing.
- **Runtime.** Python 3.14.6 or later in the 3.14 series for the Service, Python 3.11 or later for the Blueprint, and `uv`.

Two Slurm details catch operators out. The Blueprint can only attribute jobs still in the Slurm accounting database, so if `sacct` returns no rows for older jobs, check `MaxJobAge` in `slurm.conf`. And if accounting data is private, the Blueprint sees only the jobs its user may see, so run it as an admin or service account for cluster-wide coverage. [Scheduler requirements](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/SCHEDULER_REQUIREMENTS.md) has the full checklist.

## Your path

1. **Understand how a verdict is reached.** [How Attribution Works](attribution.md) explains symptoms, rules, and the phases from extraction to outcome.
2. **Try it on sample data.** The [Quickstart](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/quickstart.md) runs the Service and web UI in Docker and attributes 118 sample workloads.
3. **Check your scheduler.** [Scheduler requirements](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/SCHEDULER_REQUIREMENTS.md) lists the access the Blueprint needs.
4. **Run it.** The [Service Operator Guide](service-operators.md) covers building the images, ports, environment variables, health checks, data, and upgrades.
5. **Connect your cluster.** [Configuration](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/CONFIGURATION.md) documents every connector, and the [use-case guides](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/use-cases/README.md) walk through complete Kueue and syslog setups.
6. **Read the results.** The [output schema](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/output_schema.md) describes the stored diagnosis, and the [REST API walkthrough](https://github.com/dsx-ai-factory/attribution/blob/main/blueprint/docs/rest-api.md) shows the Service calls.
7. **Get answers and extend.** The [FAQ](FAQ.md) covers common questions, and [Add a New Extractor Rule](https://github.com/dsx-ai-factory/attribution/blob/main/service/attribution-lib/docs/adding-a-rule.md) shows how to extend the rule set.
