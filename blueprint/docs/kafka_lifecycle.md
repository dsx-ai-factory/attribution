# Kafka consumption and acknowledgements

The Kafka input supports two lifecycles through the same registered component
and [connection configuration](CONFIGURATION.md#ecakafka):

- `discover()` reads a bounded batch, commits, closes, and returns contexts. The
  existing Blueprint CLI uses this best-effort path.
- `open_consumer()` opens an independent long-lived consumer using the same
  configuration. Its `poll(max_records, timeout_seconds)` returns a list of
  `KafkaReceipt` records without committing offsets. An empty list can mean an
  idle poll, EOF, a pause, or intentional skips; keep polling.

The shared `AcknowledgingExecutionContextSource` and
`AcknowledgingExecutionContextAdapter` protocols describe this optional lifecycle.
Other ECAs need not implement it. A consumer uses the lifecycle selected by its
first read; mixing `discover()` and `poll()` on one instance raises an error.

## Processing handoff

```python
from collections.abc import Callable
from threading import Event

from attribution_blueprint.connectors.execution_contexts import (
    AcknowledgingExecutionContextSource,
    ExecutionContextDelivery,
)


def receive(
    source: AcknowledgingExecutionContextSource,
    handle: Callable[[ExecutionContextDelivery], None],
    stop: Event,
) -> None:
    consumer = source.open_consumer()
    try:
        while not stop.is_set():
            for receipt in consumer.poll(max_records=100, timeout_seconds=1):
                handle(receipt)  # Application decides when work is handled.
                consumer.ack(receipt)
    finally:
        consumer.close()
```

This sequential example requires processing the entire batch, including all
acknowledgements, within Kafka's `max.poll.interval.ms`.
For slow processing or a telemetry delay, save pending work and process it
separately; keep the intake thread polling and acknowledge on that same thread
when required processing and outputs finish. Acknowledgements can arrive across
many polls. The connector does not implement a ledger, processing delay, worker
pool, or result retries, and does not choose the application's acknowledgement
boundary. Merely saving pending work does not acknowledge it.

Each `KafkaReceipt` has `topic`, `partition`, `offset`, `context`, `event`, and an
opaque `assignment_id`. `event` retains the CloudEvent envelope and original
JSON data, including `type`; binary headers are normalized into that envelope.
`context` contains the mapped fields, including parsed times. Receipt metadata
stays outside the Attribution Service payload. Save provenance for diagnostics;
a saved receipt cannot acknowledge work after restart or reassignment.

## Offset and failure rules

- `ack(receipt)` commits synchronously per partition, using the next offset after
  its acknowledged prefix. A later acknowledgement cannot pass an earlier
  unacknowledged delivery. Gaps between Kafka offsets do not block the prefix.
- Repeated acknowledgements are harmless during the same ownership. Foreign,
  unknown, closed-consumer, and revoked/lost receipts raise `InvalidReceiptError`.
- Every valid event is delivered, including repeated events and attempts. The
  caller owns replay handling and its policy for distinct triggers. Bounded
  discovery keeps its existing batch deduplication.
- Malformed and filtered records are intentional skips. Polling never commits
  them; their offsets are covered by a later acknowledged prefix. Trailing skips
  may replay after restart. They do not accumulate receipt state in memory.
- Polling or commit failures close the consumer and propagate the error.
  Unexpected decoder failures also close and propagate. A failed commit can be
  uncertain; recreate the consumer and recover any replay through application state.
- Rebalance callbacks and shutdown do not commit. Revocation/loss invalidates
  receipts for those partitions; other owned partitions remain independent.
- `pause()` stops delivery without leaving the group. Continue polling while
  paused. New assignments inherit the pause; `resume()` restarts delivery.

Use one owner thread for polling, acknowledgements, flow control, and shutdown.
Keep polling within Kafka's `max.poll.interval.ms` and bound outstanding intake
in the application. Kafka can replay unfinished work only while records remain
within topic retention; offset commits do not delete records. `max_messages`
bounds `discover()`, not the long-lived consumer's lifetime.

Streaming statistics accumulate across polls. `accepted` counts decoded contexts,
not completed work. Streaming does not compute batch `redeliveries` or `superseded`
counts. It logs the first mismatch for each event type and retains at most 100
unmatched types plus an `<other>` count.
