# Changelog

## [0.1.0] - 2026-10-14

Initial release of the blueprint: a reference integration that collects logs and metrics from Slurm, pushes them to the attribution service, and presents the verdicts in a web UI.

### Features
- Reject a duplicate run as a conflict instead of replacing the existing session (#235)
- Report ingestion status and evidence completeness (#246)
- Present compound outcomes in the Web UI (#232)
- Publish the failure catalog with remediation actions (#177)
- Kueue support: normalized workload states, durable Pod placement recording, and placement recovery from Prometheus (#133, #134, #135)
- Connectors emit `entry_timeseries` for applog, syslog and PromQL (#130, #155, #170)
- Web UI: decision graph viewer, symptom evidence and documentation links, and a polish pass (#150, #154, #168)
- Switch to the generated attribution client (#63)

### Bug Fixes
- Persist ingestion status and flag incomplete-evidence verdicts (#234)
- Stamp `kubectl_logs` entries with the pod's node (#159)
- Date syslog lines from the line itself and drop undatable ones (#151)
- Update vulnerable dependencies (#169, #171, #202)
