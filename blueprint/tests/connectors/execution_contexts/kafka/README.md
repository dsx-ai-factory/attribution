<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kafka input tests

Run from the repository root against disposable local Kafka and Attribution
Service instances. Tests create/delete unique topics and write temporary SQLite
results; do not point them at a shared or production broker.

```bash
ATTRIBUTION_TEST_SERVICE_URL=http://127.0.0.1:18001 \
ATTRIBUTION_TEST_KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:39093 \
uv run --project blueprint --frozen --extra dev --extra kafka pytest \
  blueprint/tests/connectors/execution_contexts/kafka -q
```

The repository Blueprint gate installs the `kafka` extra so native-exception
coverage runs even without a broker. Without the endpoint settings above, live
tests skip. Unit tests still check payload mapping,
malformed/filtered events, exact `(source, id)` redeliveries, latest-attempt selection,
cross-partition warnings, timeouts, commit boundaries,
revocation/loss, and propagation of programming and native client errors.

Live tests cover both CloudEvent encodings, source-scoped redeliveries, restarts
from committed offsets, partial commit failure, delayed fetching, and two
consumers rebalancing without missing/overlapping contexts. One structured-event
smoke test checks real Service attribution and SQLite persistence of correlation
metadata. Expanded payload combinations and repeated-batch pipeline attribution
are separate Blueprint integration follow-up work. Service/output failures
explicitly verify the documented commit-before-attribution limitation.

Each discovery closes the consumer before attribution starts. Filtered-only
batches succeed with visible counts; invalid pod-to-node mappings are skipped.
The adapter closes instead of seeking or retrying after failures. Retry tests construct a new consumer and verify that only
uncommitted records replay. Rebalance callbacks never commit. Native Kafka
exceptions are injected separately from programming errors, which must escape.
The only catch-all performs failure cleanup and re-raises, including cancellation.

Optional infrastructure tests:

- `ATTRIBUTION_TEST_KAFKA_CONTAINER`: the dedicated local broker container to
  pause/unpause for an interruption test. No other workload may use it.
- `ATTRIBUTION_TEST_KAFKA_SASL_BOOTSTRAP_SERVERS`,
  `ATTRIBUTION_TEST_KAFKA_USERNAME`, `ATTRIBUTION_TEST_KAFKA_PASSWORD`: a disposable
  PLAIN/SASL_PLAINTEXT listener. The plaintext admin listener must allow ACL
  administration, with `StandardAuthorizer` and
  `allow.everyone.if.no.acl.found=true`. The test installs a temporary topic
  READ-deny ACL, checks failure without a commit, removes it, and retries.

Run sequentially, without pytest-xdist. The source is split between `events.py`
(payload decoding/mapping), `connector.py` (batch and ownership lifecycle), and
`__init__.py` (MCA registration and native client settings). Runtime storage and
deferred acknowledgment remain follow-up work. Kafka result publishing is
implemented by `output.kafka` and covered in the
[Kafka output tests](../../outputs/kafka/).
