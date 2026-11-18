# Kafka consumption and acknowledgements

The Kafka input supports bounded discovery and long-lived consumption through the same registered component
and [connection configuration](CONFIGURATION.md#ecakafka):

- `discover()` reads a bounded batch, commits, closes, and returns contexts. The
  existing Blueprint CLI uses this best-effort path.
- `open_consumer()` opens an independent long-lived consumer using the same
  configuration. Its `poll(max_records, timeout_seconds)` returns a list of
  `KafkaReceipt` records without committing offsets. An empty list can mean an
  idle poll, EOF, a pause, or intentional skips; keep polling.
- `open_raw_consumer()` opens an independent long-lived consumer whose
  `poll_raw(max_records, timeout_seconds)` returns `KafkaRawReceipt` records.
  This opt-in path delivers the original payload without CloudEvent decoding,
  execution-context mapping, or event-type filtering. Existing `discover()` and
  `poll()` behavior remains unchanged.

The shared `AcknowledgingExecutionContextSource` and
`AcknowledgingExecutionContextAdapter` protocols describe this optional lifecycle.
The shared delivery contract exposes only `context` and provider `event` data;
its lifecycle methods describe handling deliveries without assuming a transport.
Kafka-specific provenance stays on `KafkaReceipt`, outside `ExecutionContext`.
Raw receipts do not implement the execution-context delivery protocol because
their callers have not yet constructed execution contexts.
Other ECAs need not implement this lifecycle. A consumer uses the lifecycle
selected by its factory or first read. Mixing `discover()`, `poll()`, and
`poll_raw()` on one instance raises an error. Pause and resume do not select
whether a directly constructed adapter decodes its payloads.

## Raw records

Use `source.open_raw_consumer()` followed by `consumer.poll_raw()` when your
application must validate, retain, or correlate records before it can construct
execution contexts. The returned consumer uses the same `ack()`, `pause()`,
`resume()`, and `close()` methods as decoded consumption, on one owner thread.
It requires the same configured native consumer factory as `open_consumer()`.
You can also call `poll_raw()` on a directly constructed adapter.

Each `KafkaRawReceipt` contains `topic`, `partition`, `offset`, `assignment_id`,
`key`, `value`, and `headers`. Keys and values retain their original bytes.
An absent key or a tombstone value remains `None`; empty bytes remain empty.
Headers retain order, duplicate names, and null values. A missing header list
remains `None`, and a present list is copied when the record is delivered.
CloudEvent envelopes remain intact in the original bytes.

Raw intake requires a non-deserializing consumer. Keys, values, and header values
must be `bytes` or `None`; headers must be a list of `(str, bytes | None)` tuples
or `None`. A consumer that returns other types causes `InvalidReceiptError` and
closes without committing pending work. Payloads are never coerced into bytes.

Every raw record receives a receipt, including invalid JSON, unsupported event
types, tombstones, and duplicate events. The caller owns validation, filtering,
recovery, and the acknowledgement boundary. Raw polling never commits a record
or skips it because of its payload. Handle or durably retain rejected records
before acknowledging them; a later receipt cannot advance past an unacknowledged
raw record. Decoding or correlation does not happen in the connector.

Raw delivery increments `consumed`, `raw_delivered`, and `returned`, and emits
the `attribution.kafka.consumer.raw.delivered` counter. It does not increment
`accepted`, which counts valid decoded contexts. The existing Kafka metrics
report raw delivery backlog, acknowledgements, commits, and ownership changes
through the same lifecycle.

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
            for delivery in consumer.poll(max_records=100, timeout_seconds=1):
                handle(delivery)  # Application decides when work is handled.
                consumer.ack(delivery)
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

To monitor a long-lived consumer, use the
[Kafka input consumer metrics](CONFIGURATION.md#kafka-input-consumer-metrics).
They report consumer lag, unacknowledged deliveries, and the age of the oldest
unacknowledged delivery for each owned partition.

Streaming statistics accumulate across polls. `accepted` counts decoded contexts,
not completed work. Streaming does not compute batch `redeliveries` or `superseded`
counts. It logs the first mismatch for each event type and retains at most 100
unmatched types plus an `<other>` count.
