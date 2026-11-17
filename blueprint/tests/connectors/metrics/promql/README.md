<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PromQL replay fixture

`query_range.json` contains Prometheus-compatible matrix responses for all four
bundled templates and the input series used for a local VictoriaMetrics import.

The fixture was built from a workload's node-scoped counter and availability
series. The original window had no threshold-matching results for any bundled
template, and the retransmission metric was absent. Counter changes, a
retransmission increase, and an availability outage are therefore **synthetic**;
they prove collection and node/cluster attribution, not a historical diagnosis.
The retransmission series uses the same minimal counter label shape.

Only the node, cluster, and device labels needed by the connector are retained
(plus the public metric name on input series). Their values are replaced with
`node-01`, `node-02`, `example-cluster`, and `adapter-01`; timestamps are shifted
to the synthetic example window. All other identifying labels are discarded.
The `up` response retains only the node label, matching `max by (...)` and
exercising the configured-cluster fallback. Tests rename these labels in memory
to replay the same envelopes with `hostname` / `cluster` mapping and no auth.

Unit tests mock HTTP and require neither a network connection nor containers.
