<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kafka integration tests

Run from the repository root against disposable local Kafka and Attribution
Service instances. Tests create and delete unique topics, change test ACLs, and
write SQLite databases under pytest's temporary directory.

```bash
ATTRIBUTION_TEST_SERVICE_URL=http://127.0.0.1:18001 \
ATTRIBUTION_TEST_KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:39093 \
uv run --project blueprint --frozen --extra dev --extra kafka pytest \
  blueprint/tests/connectors/execution_contexts/kafka -q
```

Without these two environment variables, live tests skip; fake-consumer unit
tests still run. Use a sequential run, without pytest-xdist.

The live tests cover:

- The complete shared completion-event fixture in structured and binary mode,
  with empty and populated cancellation actors and one or multiple failed nodes.
- Actual Service requests, hardware attribution, all context/Pod fields in SQLite,
  and restart with committed group offsets.
- State-only attribution for completed, failed, user-cancelled,
  operator-cancelled, and timed-out jobs, without log evidence. Timeout-only
  jobs with a known wall-clock limit follow the existing Service policy of
  returning Pass; without that limit, the timeout fallback remains Fail User.
- Two consumers joining the same group during a batch: a revoke never commits
  past an unreturned record, and no records are missing or duplicated.
- CLI exit code 1 and the documented absence of replay after Service or SQLite
  output failure.
- An unreachable broker and preservation of uncommitted records after a commit
  exception.
- The native mixed-commit failure contract: a real KafkaException can occur
  after another partition's offset has committed. Adapter commits run one
  partition at a time; a deliberately invalid partition request generates a real
  broker/client failure, and only that partition's contexts are dropped and retried.
- Short polls with delayed native fetching: paused partitions remain unread
  through several empty polls, then resume and deliver their records.
- Same-consumer and restarted-consumer retry after a commit or read error,
  without cleanup accidentally committing the failed discovery batch.
- Filtered-only schema changes return CLI exit code 1 with type/count diagnostics;
  mixed batches process valid events and report their skipped-type counts.

Two additional tests require explicit infrastructure settings:

- Set `ATTRIBUTION_TEST_KAFKA_CONTAINER` to the name of the dedicated local
  Kafka Docker container to test a broker interruption. This test pauses and
  unpauses that container; no other workload should use it.
- Set `ATTRIBUTION_TEST_KAFKA_SASL_BOOTSTRAP_SERVERS`,
  `ATTRIBUTION_TEST_KAFKA_USERNAME`, and
  `ATTRIBUTION_TEST_KAFKA_PASSWORD` for the authorization test. The SASL listener
  must support PLAIN over SASL_PLAINTEXT and the test user must authenticate.
  The plaintext admin listener must permit topic and ACL administration.
  The broker must enable `org.apache.kafka.metadata.authorizer.StandardAuthorizer`
  with `allow.everyone.if.no.acl.found=true`. The test installs a temporary
  READ-deny ACL for its own unique topic and user, confirms CLI failure without
  a commit, removes the ACL, and confirms the authorized consumer can retry.

These settings describe test infrastructure only. The broker interruption and
authorization tests skip unless their respective settings are present.
