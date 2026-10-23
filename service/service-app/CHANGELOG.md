# Changelog

## [2.0.0] - 2026-10-14

First release from this repository. The changes below are relative to the previous version of this component.

### Features
- Accept `entry_timeseries` observations at the API boundary (#95), for applog (#114), metrics and fabric logs (#144), syslog and NVLink metrics (#147), health checks (#157) and CSP events (#163)
- Split the observation model per API version; `/v1` decodes the legacy formats (#173, #179)
- Report accepted and dropped entries per observation in `/v2` responses (#236)
- Surface interesting evidence in the v2 attribution response (#79)
- Typed compound-outcome translation layer (#152)
- Add Kueue workload semantics and KAI to the workload scheduler contract (#132, #205)
- Reject inverted time intervals on `/v2`; `/v1` still accepts them (#186)
- Generate the attribution client from the service OpenAPI spec (#60)

### Bug Fixes
- Give each attributor a private DuckDB spill directory, rooted at `ATTRIBUTOR_DUCKDB_TEMP_ROOT` (#240)
- Cut `entry_timeseries` ingest memory and DuckDB resource use (#176, #117)
- Close an attributor's event manager when it is deleted, and stop the cleanup task on shutdown (#166, #39)
- Take the `raw_log` year from the declared interval end (#187)
- Ignore progress beyond the legacy health-check window (#204)
- Include method and path in HTTP error logs (#51)
- Update vulnerable dependencies (#169, #171)
