# Explicit NVL Domain Membership Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement these tasks in order.

**Goal:** Transport optional hostname-to-NVL-domain mappings without changing existing Slurm discovery or observations.

**Architecture:** Add optional `node_to_nvl_domain` to each workload. Explicit maps are authoritative, including empty or partial maps; absent maps retain the existing library callback. Blueprint accepts an optional JSON inventory file and filters it to each allocation. IDs represent physical domains independently of hostname conventions and clique IDs.

**Tech Stack:** Python, Pydantic, generated OpenAPI client, Blueprint MCA.

**Spec:** The design in this chat approved by the request to prepare an MR.

## Constraints

- Preserve existing notices, sacct fields, node/rank shapes and telemetry selectors.
- No production access or automatic hardware discovery.
- Reject invalid/ambiguous assignments; omitted nodes remain unknown.
- Deploy the service before enabling Blueprint mappings.
- Preserve unrelated work in the original checkout.

## Review Focus

- Missing maps preserve callbacks and legacy payloads.
- Partial/empty maps cannot inherit global membership.
- Case/FQDN normalization cannot merge conflicting assignments.
- Workloads with identical hostnames remain isolated.
- Inventory never expands the allocation or changes ranks.

## Task 1: Library membership

- [x] Write and observe failing tests for normalization, invalid maps, snapshots, precedence and isolation.
- [x] Add a trailing default to JobInfo and an optional builder input; resolve pruning per workload.
- [x] Run the library suite.

## Task 2: Service contract

- [x] Write and observe failing tests for old/new requests, normalization and HTTP rejection.
- [x] Add optional v2 field and translate through JobInfoBuilder.
- [x] Regenerate OpenAPI/client using the documented generator.
- [x] Run service and drift suites.

## Task 3: Blueprint transition

- [x] Write and observe failing config/orchestrator tests for inventory filtering, overrides and unchanged legacy payloads.
- [x] Add workload.nvl_domain_mapping_path and enrich copied contexts before collection.
- [x] Forward mappings, retain snapshots with results and document rollout/source requirements.
- [x] Run Blueprint and available full checks; inspect diff and prepare MR content locally.

## Progress

- Prepared feature-only branch on main at 25b9ad4; published target remains codex/nvl-domain-transition.
- Preserved current main's ingestion reporting, job context fields and unique short/FQDN hostname semantics.
- Regression coverage includes authority of empty/partial maps, workload isolation, conflicting/nil inventory entries, canonical sibling identity and IP preservation.
- Historical inventory selection and automated discovery remain deferred; per-context mappings support mixed-cluster input.
- Local full gate remains blocked by unavailable Docker and required frontend tool versions.

- Current-main verification: Library 7068 passed / 6 xfailed; Service 290 passed; Blueprint 1709 passed / 35 skipped. Independent refreshed-port review reported no regressions.
