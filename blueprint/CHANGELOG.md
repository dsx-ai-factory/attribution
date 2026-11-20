# Changelog

## [0.2.0rc1] - 2026-11-20

### BREAKING CHANGES
- Console script `pod-placement-recorder` is replaced by `kubernetes-daemon`: flag `--workload-version` is now `--kueue-workload-version`, config section `[eca.kueue.placement_recorder]` is now `[eca.kueue.kubernetes_daemon]`, and `/v1/placements` takes `resource_uid` instead of `workload_uid`. The `placement_record` table changed with no migration, so start from a new database (#255, #256)
- `eca.kueue.kubectl_path` is removed; the Kueue adapter uses the Kubernetes API client. Delete the key from `[eca.kueue]` (#217)
- Kueue attempt IDs change from the Workload UID to `kueue:v1:<workload-uid>:<digest>` (#301)
- The Web UI run payload no longer includes `operatorDiagnose` (#269)
- Connector reports can have a new `partial` status, and `VictoriaLogs.collect()` raises `PartialCollectionError` when coverage is incomplete (#349)

### Features
- Support Karta workload owner kinds in KAI connector (#364)
- Reconstruct KAI attempts from retained history (#322)
- Collect HEALTH_CHECK logs (#346)
- Add raw Kafka intake with explicit acknowledgements (#390)
- Add Kafka input consumer metrics (#370)
- Record query result metrics for log and PromQL connectors (#369)
- Add ncclMemAlloc OOM and single-controller run-complete rules (#358)
- Add per-connector request metrics (#343)
- Add explicit acknowledgements for Kafka input (#326)
- Let telemetry connectors take the cluster from the execution context (#325)
- Recognize runapi as a workload scheduler (#324)
- Add AWS MSK IAM authentication to Kafka connectors (#321)
- Add Kafka attribution result output (#319)
- Record Kubernetes watch coverage gaps (#300)
- Capture Kubernetes lifecycle events (#299)
- Add Kafka execution context adapter (#315)
- Support VictoriaMetrics in the PromQL connector (#309)
- Add VictoriaLogs log connector (#308)
- Derive KAI workload state and map to attribution-lib semantics (#228)
- Discover KAI PodGroups as workload contexts (#226)
- Detect irisc not responding (#278)
- Let `local_file` read pre-shaped entries from `.jsonl` (#270)
- Detect bad NIC state from correlated mlx5 failures (#272)

### Bug Fixes
- Resolve a UID 0 sacct canceller to root or Timeout (#345)
- Render scheduler-aware wording (#384)
- Filter dashboard date windows by UTC instant (#353)
- Restore Kafka consumption after partition reassignment (#380)
- Preserve provider run identity across processing contexts (#371)
- Request the decision graph with each attribution (#335)
- Show Slurm attempt ordinal in runs table (#332)
- Warn on invalid `LOG_LEVEL` (#337)
- Search allocated nodes and attempt UID in runs query (#336)
- Attribute local RDMA command failures to their emitter (#314)
- Drop a zero Kafka event time limit (#329)
- Stop trusting the compromised emSign ECC Root CA - G3 (#316)
- Attribute ext4 filesystem corruption to the node (#276)
- Detect batch-size assertions (#275)

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
