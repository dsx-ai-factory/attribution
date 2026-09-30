# Support for NVIDIA Attribution

This document describes support for NVIDIA Attribution beginning with its general availability (GA) release. It covers the Attribution Service, Attribution Blueprint, and Attribution Library. Support under an NVIDIA commercial agreement is governed by that agreement.

## Support channels

| Channel | Best for | Who | Response expectation |
| --- | --- | --- | --- |
| [GitHub Discussions](https://github.com/dsx-ai-factory/attribution/discussions) | Usage questions, how-to help, feature requests, and design discussion | Anyone | Best effort; maintainers aim to respond within 3 business days if a question remains unanswered |
| [GitHub Issues](https://github.com/dsx-ai-factory/attribution/issues) | Reproducible bugs in the upstream repository | Anyone | Maintainers aim to triage within 3–5 business days |
| NVIDIA support under your commercial agreement or through your NVIDIA account team | Production incidents and contract-governed support | Customers with an applicable NVIDIA support agreement | Defined in the support contract |

GitHub Issues is for bug reports and accepted work. Please use Discussions for usage questions and feature requests. A maintainer may open a tracking issue after a feature request is accepted. Public community channels are not appropriate for production incidents or confidential information.

## Support requests and bug reports

A support request asks how to use Attribution; a bug report describes behavior that is incorrect or unexpected in the upstream project. If a Discussion reveals a bug, a maintainer can open an issue and link back to the Discussion.

### What to put in a bug report

All bug reports must include a reproduction using the current `main` branch of this repository. Include the commit SHA, the steps and inputs needed to reproduce the problem, the actual result, and the expected result. Include the installation or deployment method, operating system and kernel version, and relevant component and dependency versions. A minimal reproduction helps maintainers investigate.

For an **attribution decision** that appears incorrect, also include:

- The diagnosis returned by Attribution and your analysis of why that decision is incorrect.
- The observations supplied to the system, the expected diagnosis, and the evidence supporting that expectation.
- The relevant hardware topology, including NVLink, CPU, GPU, and NIC details and relationships, plus GPU driver and CUDA versions where applicable.

For a **Service or Blueprint defect**, also include:

- The affected API call or workflow, relevant configuration, exact command or request, and full error output.
- The connectors involved and the versions of external systems they use, such as Slurm, Kubernetes, KAI, Kueue, JobSet, or telemetry backends, as applicable.

Remove credentials, tokens, customer data, tenant and cluster identifiers, and private hostnames before posting logs or configuration publicly. Use your commercial support channel if the issue cannot be described without confidential information.

Changes to rules, connectors, custom layers, the Service, or the Blueprint outside this repository's `main` branch are not supported upstream. An issue that occurs only with such a customization cannot be accepted as an upstream bug. If you can reproduce the same defect on `main` without the customization, report that reproduction.

## Upstream community support

The upstream repository is supported by its maintainers and community on a best-effort basis. These are response targets, not promises to fix a reported problem:

- Maintainers aim to triage new bug issues within 3–5 business days. Triage may mean acknowledging the report, requesting details, or confirming the defect.
- Pull requests aim to receive an initial response within 5–7 business days. An initial response is not necessarily a full review.
- Maintainers aim to respond to an unanswered Discussion within 3 business days. There is no hard community-support SLA.
- Security reports sent through [SECURITY.md](./SECURITY.md) are acknowledged within 48–72 hours. Do not disclose a vulnerability in a public issue, Discussion, or pull request.

Resolution timing depends on severity, reproducibility, and maintainer capacity. Upstream maintainers do not provide production incident support or a guaranteed fix date. Customers who need contract-governed support should use their NVIDIA support channel.

## Support lifecycle

NVIDIA Attribution components use [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`). A new major version of an individual component may introduce breaking changes to that component's supported interfaces. Minor versions add backward-compatible functionality to a supported interface; patch versions provide backward-compatible fixes.

Each minor release has one of these stages:

| Stage | Meaning |
| --- | --- |
| Current | Receives features, bug fixes, and security fixes. |
| Maintenance | Receives security and critical bug fixes only. |
| End of life (EOL) | Receives no fixes. Upgrade to a supported release. |

Within a supported major version, the latest minor is Current and the previous minor is in Maintenance. When a new minor ships, the former Current minor enters Maintenance and the former Maintenance minor reaches EOL. When a new major version ships, the latest minor of the previous major enters Maintenance for three months and then reaches EOL. The release notes identify supported versions and migration guidance.

Breaking changes to a supported interface are announced in the new major version's release notes, with a migration path where feasible. No fixed advance notice period is promised.

### Interface compatibility

The Attribution Service exposes a versioned, stable interface. Breaking changes to that interface may be introduced in a future major version of the Service. The Attribution Library and Blueprint do not offer stable external interfaces. This includes the Blueprint's CLI, configuration, connector contracts, and persisted output.

### Supported environments

For supported environments and prerequisites, see the [README](./README.md#prerequisites).

### Upgrades and downgrades

| From → to | Support status |
| --- | --- |
| Older minor → latest Current minor in the same major | Supported |
| Older patch → newer patch in the same minor, skipping intermediate patches | Supported |
| Previous major's latest minor → new major's first release | Supported; follow its migration guidance |
| Skipping a major entirely | Untested; upgrade through the intermediate major |
| Downgrading to an older release | Unsupported |

## Enterprise and production support

If you run Attribution under an NVIDIA commercial support agreement, use the support route specified by that agreement or contact your NVIDIA account team for production incidents and contract-governed requests. The agreement determines its response times and scope. Bugs that also reproduce on the upstream `main` branch may be reported publicly once confidential information is removed.

## Escalation paths

| Situation | Action |
| --- | --- |
| A support question reveals a reproducible upstream bug | Open a GitHub issue and link to the Discussion. |
| A feature request is accepted for implementation | A maintainer opens a tracking issue and links to the Discussion. |
| A report may involve a security vulnerability | Direct the reporter to [SECURITY.md](./SECURITY.md). Do not discuss details publicly; ask a repository administrator to remove any public disclosure. |
| A production incident occurs under a commercial agreement | Use the NVIDIA support route specified by the agreement or contact your NVIDIA account team. |
| A bug occurs only in an unsupported customization | Reproduce it on the upstream `main` branch before filing an upstream issue. |
| A bug occurs on an EOL release | Upgrade to a supported release or use contract-governed support if available. |
