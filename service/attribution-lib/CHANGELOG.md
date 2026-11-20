# Changelog

## [0.6.0rc1] - 2026-11-20

### BREAKING CHANGES
- `Context` no longer has a `resources` field; resources come from each entry (#274)
- `BodyFallbacks.year_fallback` is replaced by `job_span`, the job start and end (#328)
- Refiner rule `SlurmTimeoutXid145Cascade` (`slurm_timeout_xid145_cascade`) is renamed `ExternalTerminationCascade` (`external_termination_cascade`) (#311)
- The PyTorch DataLoader worker bus-error line now emits `DataLoader worker bus error` instead of `bus error`, and a full `/dev/shm` emits `Shared memory exhausted` (#368)
- Jobs on schedulers other than Slurm now get scheduler-specific wording, for example `Node failure reported by kueue` instead of `Node failure reported by slurm`; Slurm strings are unchanged (#384)

### Features
- Attribute Megatron rerun verdicts instead of Bad Token (#351)
- Add ncclMemAlloc OOM and single-controller run-complete rules (#358)
- Recognize runapi as a workload scheduler (#324)
- Derive KAI workload state and map to attribution-lib semantics (#228)
- Expose cluster context through `JobInfo` (#261)
- Detect irisc not responding (#278)
- Detect bad NIC state from correlated mlx5 failures (#272)
- Add NVRx launcher health and progress symptoms (#237)
- Configure DuckDB spill capacity and retry cleanup (#253)

### Bug Fixes
- Keep IB errors unless syslog reports an OOM (#350)
- Remove spill dirs when pool workers exit (#382)
- Collapse bad nodes for a lone XID-149.21 (#366)
- Ignore crashpad's cpufreq ENOENT lines in application logs (#357)
- Clean shared lookups when pool workers exit (#356)
- Attribute local RDMA command failures to their emitter (#314)
- Rank NVRx detected failure below user failures (#320)
- Preserve unknown Kubernetes outcomes (#290)
- Stop `ibv_modify_qp` remote GID at ANSI color codes (#312)
- Handle unknown socketProgressOpt source hostnames (#297)
- Prefer user errors over NVRx detected failure (#295)
- Read node from Megatron rank errors (#183)
- Match short log hostnames to fully qualified allocations (#286)
- Keep XID-149.21 when XID-149.4 is present (#289)
- Detect JIT compilation failures (#279)
- Prioritize kernel crashes over XID-145 (#280)
- Map NVRx detected failure to Fail MultiCategory (#260)
- Attribute ext4 filesystem corruption to the node (#276)
- Detect batch-size assertions (#275)
- Preserve documentation for fallback symptoms (#263)
- Treat `NV_ERR_RESET_REQUIRED` as a GPU fault (#238)
- Keep symptoms from APPLOG lines without a node prefix (#198)

## [0.5.0] - 2026-10-14

First release from this repository. The changes below are relative to the previous version of this component.

### BREAKING CHANGES
- Remove legacy observation formats from the library (#180)
- Drop the deprecated `pre_filtered_sources` parameter from the attributor (#41)

### Features
- Add the `entry_timeseries` observation format (#94), accepted for applog (#114), metrics and fabric logs (#144), syslog and NVLink metrics (#147), health checks (#157) and CSP events (#163)
- Report accepted and dropped entries per observation (#236)
- Typed compound-outcome translation layer (#152)
- Semantic symptom families and selectors in the symptom registry (#136, #143)
- Key scheduler-specific rules off `job_info.scheduler`, with Kueue and KAI support (#132, #188, #205)
- Reject inverted time intervals at construction (#186)
- Identify XID-94 poison victims (#241)
- Detect poisoned GSP failures from correlated syslog evidence (#137)
- Identify node hardware failures from SLURM and epilog evidence (#67)

### Bug Fixes
- Give each EventManager a private DuckDB spill directory (#240)
- Preserve allocation-owned scheduler failure evidence (#213)
- Preserve NCCL source hostnames and address information (#178)
- Drop `Connection closed` when CUDA runs out of memory (#243)
- Prioritize kernel panic over XID-145 (#192)
- Classify XID-38 as Fail InfraSW (#182) and XID-140 as an ECC hardware failure (#146)
- Match fully qualified hostnames to the allocation (#174)
- Remove DNS from node resolution (#76)
- Recompute incompatible extraction cache snapshots (#122, #184)
