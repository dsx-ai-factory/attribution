<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Blueprint upgrade guide

## VictoriaLogs partial collection reporting

This change affects callers upgrading from versions where VictoriaLogs returned
potentially incomplete observations as a normal list.

### Direct connector callers

`VictoriaLogsConnector.collect()` raises `PartialCollectionError` when collection
is incomplete. The exception carries the available `observations` and a reason.
Retain both, including when the observation list is empty:

```python
from attribution_blueprint.exceptions import PartialCollectionError

try:
    observations = connector.collect(context)
    collection_status, collection_reason = "ok", ""
except PartialCollectionError as error:
    observations = error.observations
    collection_status, collection_reason = "partial", str(error)
```

Submit the retained observations through your normal ingestion path and preserve
the partial status and reason in your coverage reporting. The Blueprint
orchestrator handles this exception and records the partial result automatically.
Complete collection still returns a list, including an empty list when no logs match.

### Report consumers

Consumers of `connector_reports` must accept `"partial"` alongside `"ok"` and
`"error"`. Available evidence does not establish complete coverage. For the
conditions that produce partial collection, refer to
[VictoriaLogs configuration](CONFIGURATION.md#victorialogs-syslog-or-applog).

### Multiple queries and request failures

The connector attempts all configured queries after an HTTP error or timeout.
If another query returns a response, or earlier pages supplied evidence, collection
is partial and retains the available observations. A successful but empty query
counts as a response; a failure in another query still reports partial coverage.
If every attempted query fails without returning a response, the first request
exception propagates as an error. Reordering queries does not turn retained
evidence into a complete collection or prevent later queries from running.

Missing template arguments and skipped malformed response records also report
partial coverage. Blank response lines are ignored.
