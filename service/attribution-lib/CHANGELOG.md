# Changelog

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
