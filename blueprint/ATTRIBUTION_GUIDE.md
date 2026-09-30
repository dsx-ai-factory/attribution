# Attribution capability and operator guidance

This guide is the detailed companion to the README's "What it doesn't do"
section. It is part of the distributed source tree.

## Capability boundaries

* **Attribution is post-hoc, not predictive.** The unit of analysis is a workload that has reached a terminal state. Attribution cannot tell you a node will fail before symptoms appear.
* **Attribution doesn't probe.** It reads the logs and metrics already collected by the operator's stack. It doesn't run `nvidia-smi`, `dcgmi diag`, `nvidia-fielddiag`, or any other diagnostic. Operators run those, and the output may land in syslog or a log store the Blueprint reads, but the running of the command is outside Attribution's scope.
* **Attribution provides evidence. It does not make hardware-call decisions.** The verdict describes the cascade, identifies the root symptom, and names responsible nodes when the evidence supports it. It does not make binary hardware-call decisions (replace, return to service). Those calls are the operator's.
* **Attribution doesn't replace incident management.** A diagnosis is one input to whatever ticketing, paging, and runbook system the operator already runs. It is not itself an incident-management surface.

## Common operator misreads

* **"Fail / hardware" doesn't mean replace the hardware.** It means the cascade's root cause sits in the hardware sub-domain. The operational action that follows is the operator's, not Attribution's.
* **An empty responsible-node list isn't an error.** When the cascade doesn't point at a single node, `responsible_nodes_json` is an empty list. That is honest output, not a failure.
* **An `ambiguous` or `unattributed` `sub_domain` isn't a bug.** When the evidence supports multiple plausible causes, `sub_domain` is `ambiguous`. When the evidence is insufficient to attribute a cause, `sub_domain` is `unattributed`. The distinction tells you whether the uncertainty is multiple causes fitting or insufficient evidence. Both are information, not failure.

## Missing signals and data gaps

Attribution output quality scales with the quantity and coverage of the logs the Service sees. If a signal source is missing entirely (no PromQL backend configured, Loki not reachable, sacct StdOut paths not resolvable), the affected collector reports zero observations in `connector_reports` and the rules that depended on that source don't fire. Observation submission failures after collection are logged and skipped by the Blueprint, but they are not represented in `connector_reports`. The verdict is still issued. Two consequences:

* **Coverage matters more than completeness.** Better to have applog from Loki and node_metrics from Prometheus than to wait until syslog is also wired up. Attribution returns a verdict from whatever it has.
* **The `connector_reports` field is the collector coverage record.** A diagnosis that ran with one collector erroring out is honestly different from one that ran with all collectors green. Downstream consumers should inspect `connector_reports` and the run logs when treating verdicts as evidence.

## Taxonomy: terminal_state and sub_domain

The Blueprint normalizes each diagnosis to a `(terminal_state, sub_domain)` pair and a diagnosis label (`taxonomy.py`). The mapping is deterministic.

| terminal_state | sub_domain | diagnosis |
|---|---|---|
| pass | none | Pass |
| fail | hardware | Hardware Fault |
| fail | system_software | System Software Fault |
| fail | application | Application Fault |
| fail | ambiguous | Ambiguous Fault |
| fail | unattributed | Unattributed |
| cancelled | workload_owner | Cancelled, Workload Owner |
| cancelled | operator | Cancelled, Operator |
| expected_failure | expected | Expected Failure |
| expected_failure | artificial | Expected Failure |
| unknown | unattributed | Unattributed |
| unknown | unclassified | Unattributed |

Downstream consumers should branch on the normalized `terminal_state` and `sub_domain` fields.
