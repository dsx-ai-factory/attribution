# Local HCA command failures

A failure of an RDMA control operation can identify the node that logged it
without establishing a fault on its remote communication peer.

For `ibv_modify_qp` errno 22 with both GIDs in IPv4-mapped form, the reporting
node is bad and the endpoint pair remains interesting. Mismatched-GID cases
retain their existing handling. Timeouts with resolved remote peers also keep
pair attribution. A timeout associated with overwritten NCCL parameters retains
its existing Fail User outcome even when the peer cannot be resolved. Pair
records and their source lines remain available.

For errno 11, 110 or 125 with an unavailable or unresolved remote GID,
`ibv_modify_qp` contributes a local HCA command failure. The library also
recognizes these errno values for `ibv_query_qp`, `ibv_create_cq`, `ibv_reg_mr`
and `ibv_dereg_mr`, including NCCL's text-only resource-unavailable message.
The symptom includes the operation and errno; provenance retains the log line
and any device identity it provides. `NCCL Error 2: unhandled system error`
maps to the existing `ncclSystemError` symptom.

An isolated `ibv_reg_mr` or `ibv_create_cq` EAGAIN (errno 11) can reflect
memlock or resource limits. It remains interesting unless either:

- A different verbs operation fails on the same node and device within five
  minutes. Repetitions of one operation and another resource-limited
  `ibv_reg_mr`/`ibv_create_cq` EAGAIN do not satisfy this condition.
- An existing NIC syslog diagnosis corroborates the same node and device
  within five minutes. Supported diagnoses are `irisc not responding` and
  the two initialization messages required by `DetectBadNicState`.

Device-less wrapper messages can use a nearby device anchor only when it is
from the same emitting PID/thread and all nearby anchors agree. Missing times
or ambiguous devices cannot satisfy the guard. PCI addresses and `mlx5_N`
identifiers are never assumed equivalent. Real kernel messages usually identify
only PCI addresses, whereas application logs usually identify `mlx5_N`. These
cannot corroborate each other without a mapping source, which is not currently
implemented. Exact PCI matches can corroborate when both logs provide them.
`health compromised` alone is not corroboration.

Qualifying local HCA symptoms without corroboration map to
`Fail InfraHW / Unknown HW Cause`. A local HCA failure plus an independent,
matching NIC syslog diagnosis or an existing `CSP Event NIC Fault` on the same
node within five minutes produces `Corroborated local HCA command failure`,
mapping to `Fail InfraHW / Known HW Cause - Node (NIC)`. Authoritative CSP
evidence retains its priority, classification and cascade behavior. Provider
evidence must already identify the node and come from the CSP event source.
There is no EC2 system-check event producer in this library; system-check
collection and normalization are not implemented here. Provider evidence does
not bypass the resource-limit guard.

The compound uses the existing `DetectBadNicState` result instead of repeating
its initialization-message correlation. Consumed syslog sources remain
interesting evidence. CSP sources remain selected. Other nodes and independent
failures remain intact.
Standalone NIC syslog diagnoses retain their existing `Fail InfraSW` mappings.
Additional syslog command-timeout rules and provider collection changes are
separate work. A local command error does not by itself prove a physical
adapter defect.

Emitter identity follows the shared network rules: prefer the explicit NCCL
emitter, accept a verified leading hostname or mapped local GID, then use
explicit observation metadata. Rank alone and remote GIDs do not identify the
emitter. Unknown emitters remain interesting evidence without a hardware
verdict. Unmapped peer addresses remain in pair details, not node identities.
Local command candidates use the existing after-MFU demotion filter, including
its timestamp comparison and line-number fallback. Only surviving candidates
can corroborate another failure. Unresolved QP timeouts preserve
the existing timeout priority tier.
