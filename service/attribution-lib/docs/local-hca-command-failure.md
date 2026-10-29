# Local HCA command failures

A failure of an RDMA control operation can identify the node that logged it
without establishing a fault on its remote communication peer.

For `ibv_modify_qp` errno 22 with both GIDs in IPv4-mapped form, the reporting
node is bad and the endpoint pair remains interesting. Mismatched-GID cases
retain their existing handling. Timeouts with resolved remote peers also keep
pair attribution. Pair records and their source lines remain available.

For errno 11, 110 or 125 with an unavailable or unresolved remote GID,
`ibv_modify_qp` contributes a local HCA command failure. The library also
recognizes these errno values for `ibv_query_qp`, `ibv_create_cq`, `ibv_reg_mr`
and `ibv_dereg_mr`, including NCCL's text-only resource-unavailable message.
The symptom includes the operation and errno; provenance retains the log line
and any device identity it provides. `NCCL Error 2: unhandled system error`
maps to the existing `ncclSystemError` symptom.

An isolated `ibv_reg_mr` EAGAIN (errno 11) can reflect a memory-registration or
memlock limit. It remains interesting unless either:

- A different verbs operation fails on the same node and device within five
  minutes. Repetitions of one operation do not satisfy this condition.
- An existing NIC syslog diagnosis corroborates the same node and device
  within five minutes. Supported diagnoses are `irisc not responding` and
  the two initialization messages required by `DetectBadNicState`.

Device-less wrapper messages can use a nearby device anchor only when it is
from the same emitting PID/thread and all nearby anchors agree. Missing times
or ambiguous devices cannot satisfy the guard. PCI addresses and `mlx5_N`
identifiers are never assumed equivalent without explicit evidence.
`health compromised` alone is not corroboration.

Qualifying local HCA symptoms without corroboration map to
`Fail InfraHW / Unknown HW Cause`. A local HCA failure plus an independent,
matching NIC syslog diagnosis or normalized provider system-check/NIC-fault
evidence produces `Corroborated local HCA command failure`, mapping to
`Fail InfraHW / Known HW Cause - Node (NIC)`. Provider evidence must already
be resolved to the same node by its connector; event identity and lifecycle
validation remain the connector's responsibility. An instance-status check
alone does not corroborate a NIC hardware diagnosis or bypass the registration
guard.

The compound uses the existing `DetectBadNicState` result instead of repeating
its initialization-message correlation. Consumed corroborating sources remain
interesting evidence. Other nodes and independent failures remain intact.
Standalone NIC syslog diagnoses retain their existing `Fail InfraSW` mappings.
Additional syslog command-timeout rules and provider collection changes are
separate work. A local command error does not by itself prove a physical
adapter defect.
