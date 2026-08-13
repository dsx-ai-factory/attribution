# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import socket
from datetime import datetime
from unittest.mock import patch

import pytest
from attribution_lib.core import process_check
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
    Source as ObservationSource,
)
from loguru import logger
from test_support.helpers import make_applog_entry, make_node_to_ranks


@pytest.fixture
def app_tst_bldr(attribution_test_builder):
    def _build(
        job_info,
        data,
        *,
        format=ObservationFormat.RAW_LOG_LINES,
        stream_id=None,
        insert_tflops_line=False,
    ):
        return attribution_test_builder(
            job_info=job_info,
            data=data,
            source=ObservationSource.APPLOG,
            format=format,
            check_fn=process_check,
            stream_id=stream_id,
            insert_tflops_line=insert_tflops_line,
        )

    return _build


@pytest.fixture(scope="class")
def big_jobinfo_builder(timezone):
    def _build_jobinfo(tflops_start):
        start = datetime.fromtimestamp(1741860016, tz=timezone)
        end = datetime.fromtimestamp(1741862869, tz=timezone)
        tflops_time = start if tflops_start else end
        return (
            JobInfoBuilder(
                id="12345678",
                cluster="cluster_b722552b",
                state="CANCELLED",
                type="training",
                framework="megatron",
                node_to_ranks=make_node_to_ranks(
                    [
                        "host-9ad02e1b",
                        "host-a222193a",
                        "host-9162d81d",
                        "host-a29ce5ce",
                        "host-dc7744c0",
                        "host-317c4378",
                        "host-2a0e0225",
                        "host-0fa931d7",
                    ],
                    8,
                ),
                job_start_time=start,
                source_name="artifical slurm",
                timezone=timezone,
            )
            .job_end_time(end)
            .srun_cmd("srun test.py")
            .time_limit(600)
            .last_tflops_time(tflops_time)
            .build()
        )

    return _build_jobinfo


@pytest.fixture(scope="class")
def big_jobinfo(timezone, big_jobinfo_builder):
    return big_jobinfo_builder(True)


class TestIBPairRules:
    """Rule-owned tests for IB Error-12 bad_pairs detection.

    These stay as co-located imperative tests instead of TOML cases because
    they validate emitted event types, payload structure, parser fallbacks,
    and mocked hostname-resolution behavior rather than just final
    attribution.
    """

    def _get_events_by_type(self, event_stream, event_types):
        if isinstance(event_types, str):
            event_types = [event_types]

        placeholders = ",".join(["?"] * len(event_types))
        rows = event_stream.query_raw(
            f"""
            SELECT e.id, et.name, e.payload
            FROM events e
            JOIN event_types et ON e.event_type = et.id
            WHERE et.name IN ({placeholders})
            """,
            event_types,
        )
        return [{"id": row[0], "type": row[1], "payload": json.loads(row[2]) if row[2] else {}} for row in rows]

    def test_got_completion_from_peer_new_creates_bad_pair_and_bad_peers(self, app_tst_bldr, big_jobinfo):
        data = [
            "transport/net_ib.cc:2359 NCCL WARN NET/IB: Got completion from peer 192.0.2.10<10904> with status=12 opcode=129 len=374784 vendor err 129 (Recv) localGid ::ffff:192.0.2.20 remoteGids::ffff:198.51.100.20 hca rocep206s0"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(
            result.event_stream, ["Attribution.node.bad_peers", "Application.bad_pair.complete"]
        )
        bad_peers_events = [e for e in events if e["type"] == "Attribution.node.bad_peers"]
        bad_pair_events = [e for e in events if e["type"] == "Application.bad_pair.complete"]

        assert len(bad_peers_events) >= 1
        assert len(bad_pair_events) == 1
        assert bad_peers_events[0]["payload"]["symptom"] == "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)"
        assert "192.0.2.10" in bad_peers_events[0]["payload"]["peer"]
        assert bad_pair_events[0]["payload"]["local_dev_ip"] == "192.0.2.20"
        assert bad_pair_events[0]["payload"]["remote_dev_ip"] == "198.51.100.20"
        assert bad_pair_events[0]["payload"]["local_device"] == "rocep206s0"

    def test_got_completion_from_peer_new_partial_fallback(self, app_tst_bldr, big_jobinfo):
        data = [
            "transport/net_ib.cc:2359 NCCL WARN NET/IB: Got completion from peer 192.0.2.10<10904> with status=12 opcode=129 len=374784 vendor err 129 (Recv) localGid 192.0.2.40 remoteGids::ffff:198.51.100.20 hca rocep206s0"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(result.event_stream, "Application.bad_pair.partial")
        assert len(events) >= 1
        bad_pair = events[0]
        assert "local_dev_ip" not in bad_pair["payload"] or bad_pair["payload"].get("local_dev_ip") is None
        assert "local_node" in bad_pair["payload"]
        assert "remote_node" in bad_pair["payload"]

    def test_got_completion_from_peer_old_creates_partial_bad_pair(self, app_tst_bldr, big_jobinfo):
        data = [
            "transport/net_ib.cc:2359 NCCL WARN NET/IB: Got completion from peer 192.0.2.10<10904> with status=12 opcode=129 vendor err 129 (Recv)"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(
            result.event_stream, ["Attribution.node.bad_peers", "Application.bad_pair.partial"]
        )
        assert len(events) == 2
        bad_pair = [e for e in events if e["type"] == "Application.bad_pair.partial"][0]
        assert "local_node" in bad_pair["payload"]
        assert "remote_node" in bad_pair["payload"]
        assert "local_dev_ip" not in bad_pair["payload"] or bad_pair["payload"].get("local_dev_ip") is None

    def test_ibv_modify_qp_failed_creates_bad_pair_and_bad_peers(self, app_tst_bldr, big_jobinfo):
        data = [
            "Call to ibv_modify_qp failed with 110 Connection timed out, on dev mlx5_4:1, curr state INIT, next state RTR, local GID index 3, local GID ::ffff:192.0.2.30, remote GID ::ffff:198.51.100.30"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(
            result.event_stream, ["Attribution.node.bad_peers", "Application.bad_pair.complete"]
        )
        bad_peers_events = [e for e in events if e["type"] == "Attribution.node.bad_peers"]
        bad_pair_events = [e for e in events if e["type"] == "Application.bad_pair.complete"]

        assert len(bad_peers_events) >= 1
        assert len(bad_pair_events) == 1
        assert "Call to ibv_modify_qp failed" in bad_peers_events[0]["payload"]["symptom"]
        assert "timeout" in bad_peers_events[0]["payload"]["symptom"]
        assert bad_pair_events[0]["payload"]["local_dev_ip"] == "192.0.2.30"
        assert bad_pair_events[0]["payload"]["remote_dev_ip"] == "198.51.100.30"
        assert "mlx5_4" in bad_pair_events[0]["payload"]["local_device"]

    def test_ibv_modify_qp_failed_partial_fallback(self, app_tst_bldr, big_jobinfo):
        data = [
            "Call to ibv_modify_qp failed with 22 Invalid argument, on dev mlx5_0:1, curr state INIT, next state RTR, local GID index 3, local GID 192.0.2.40, remote GID ::ffff:198.51.100.30"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(result.event_stream, "Application.bad_pair.partial")
        assert len(events) >= 1
        bad_pair = events[0]
        assert "local_node" in bad_pair["payload"]
        assert "remote_node" in bad_pair["payload"]

    def test_ib_error_symptom_format(self, app_tst_bldr, big_jobinfo):
        data = [
            "transport/net_ib.cc:2359 NCCL WARN NET/IB: Got completion from peer 192.0.2.10<10904> with status=12 opcode=129 len=374784 vendor err 129 (Recv) localGid ::ffff:192.0.2.20 remoteGids::ffff:198.51.100.20 hca rocep206s0"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(result.event_stream, "Attribution.node.bad_peers")
        assert len(events) >= 1
        symptom = events[0]["payload"]["symptom"]
        assert "IB Error-12" in symptom
        assert "Vendor-129" in symptom
        assert "IBV_WC_RETRY_EXC_ERR" in symptom

    def test_got_completion_from_peer_nccl_2_30_ib_plugin_format(self, app_tst_bldr, big_jobinfo):
        """NCCL 2.30 ib_plugin.c format: status=IBV_WC_RETRY_EXC_ERR(12), vendor_err=129.

        Distinct from the legacy transport/net_ib.cc format which used
        "status=12 ... vendor err 129" (no symbolic name, space-separated).
        Based on representative NCCL 2.30.4+cuda13.2 application logs.
        """
        data = [
            "[2026-05-11 16:52:49] node-01:3298233:3301236 [0] ib_plugin.c:2033 NCCL WARN NET/IB: Got completion from peer 192.0.2.10<31197> with status=IBV_WC_RETRY_EXC_ERR(12) opcode=IBV_WC_SEND(0) reqSize=262144 vendor_err=129 req_type=Send localGid ::ffff:192.0.2.20 remoteGids::ffff:198.51.100.20 hca mlx5_0"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(
            result.event_stream, ["Attribution.node.bad_peers", "Application.bad_pair.complete"]
        )
        bad_peers_events = [e for e in events if e["type"] == "Attribution.node.bad_peers"]
        bad_pair_events = [e for e in events if e["type"] == "Application.bad_pair.complete"]

        assert len(bad_peers_events) >= 1
        assert len(bad_pair_events) == 1
        assert bad_peers_events[0]["payload"]["symptom"] == "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)"
        assert bad_pair_events[0]["payload"]["local_dev_ip"] == "192.0.2.20"
        assert bad_pair_events[0]["payload"]["remote_dev_ip"] == "198.51.100.20"
        assert bad_pair_events[0]["payload"]["local_device"] == "mlx5_0"

    def test_got_completion_from_peer_nccl_2_30_ib_plugin_no_separator(self, app_tst_bldr, big_jobinfo):
        """NCCL 2.30 ib_plugin.c with reqSize=0vendor_err=129 (no separator before vendor_err).

        Some RECV_RDMA_WITH_IMM completion lines omit whitespace between
        reqSize and vendor_err. The regex must not depend on a separator
        there, and the "new" extractor must still emit a complete bad_pair
        event.
        """
        data = [
            "[2026-05-11 16:52:53] node-02:487245:494388 [0] ib_plugin.c:2033 NCCL WARN NET/IB: Got completion from peer 192.0.2.10<48078> with status=IBV_WC_RETRY_EXC_ERR(12) opcode=IBV_WC_RECV_RDMA_WITH_IMM(129) reqSize=0vendor_err=129 req_type=Recv localGid ::ffff:192.0.2.20 remoteGids::ffff:198.51.100.20 hca mlx5_0"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(
            result.event_stream, ["Attribution.node.bad_peers", "Application.bad_pair.complete"]
        )
        bad_peers_events = [e for e in events if e["type"] == "Attribution.node.bad_peers"]
        bad_pair_events = [e for e in events if e["type"] == "Application.bad_pair.complete"]

        assert len(bad_peers_events) >= 1
        assert len(bad_pair_events) == 1
        assert bad_peers_events[0]["payload"]["symptom"] == "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)"
        assert bad_pair_events[0]["payload"]["local_dev_ip"] == "192.0.2.20"
        assert bad_pair_events[0]["payload"]["remote_dev_ip"] == "198.51.100.20"
        assert bad_pair_events[0]["payload"]["local_device"] == "mlx5_0"

    def test_irrecoverable_net_ib_communicator_error_creates_partial_bad_pair(self, app_tst_bldr, big_jobinfo):
        """Irrecoverable communicator errors emit a bad-peer and partial bad-pair event."""
        data = [
            "[2025-01-01 00:00:00] test-node-delta:1234:5678 [2] p2p.cc:886 NCCL WARN NET/IB : Irrecoverable receive communicator error on device 0. Reason: No active devices left. Peer: 198.51.100.40<12345>, wr_id=42, req=0x000000001234, status=IBV_WC_RETRY_EXC_ERR(12), opcode=IBV_WC_RDMA_WRITE(1), vendor_err=0, reqSize=129 (Recv) localGid:2001:db8::1 remoteGid:2001:db8::2, qpn=42, hca=mlx5_0"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(
            result.event_stream, ["Attribution.node.bad_peers", "Application.bad_pair.partial"]
        )
        bad_peers_events = [e for e in events if e["type"] == "Attribution.node.bad_peers"]
        bad_pair_events = [e for e in events if e["type"] == "Application.bad_pair.partial"]

        assert len(bad_peers_events) >= 1
        assert len(bad_pair_events) == 1
        assert bad_peers_events[0]["payload"]["symptom"] == "IB Error-12 (IBV_WC_RETRY_EXC_ERR)"
        assert "198.51.100.40" in bad_peers_events[0]["payload"]["peer"]
        assert bad_pair_events[0]["payload"]["local_node"] == "test-node-delta"
        assert "remote_node" in bad_pair_events[0]["payload"]

    def test_irrecoverable_net_ib_communicator_error_creates_complete_bad_pair(self, app_tst_bldr, big_jobinfo):
        """IPv4-mapped irrecoverable communicator errors emit a complete bad-pair event."""
        data = [
            "[2025-01-01 00:00:00] test-node-delta:1234:5678 [2] p2p.cc:886 NCCL WARN NET/IB : Irrecoverable send communicator error on device 0. Reason: No active devices left. Peer: 127.0.0.1<12345>, wr_id=42, req=0x000000001234, status=IBV_WC_RETRY_EXC_ERR(12), opcode=IBV_WC_RDMA_WRITE(1), vendor_err=0, reqSize=129 (Send) localGid:::ffff:192.0.2.20 remoteGid:::ffff:198.51.100.20, qpn=42, hca=mlx5_0"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(
            result.event_stream,
            ["Attribution.node.bad_peers", "Application.bad_pair.complete", "Application.bad_pair.partial"],
        )
        bad_peers_events = [e for e in events if e["type"] == "Attribution.node.bad_peers"]
        complete_pair_events = [e for e in events if e["type"] == "Application.bad_pair.complete"]
        partial_pair_events = [e for e in events if e["type"] == "Application.bad_pair.partial"]

        assert len(bad_peers_events) >= 1
        assert len(complete_pair_events) == 1
        assert len(partial_pair_events) == 0
        assert bad_peers_events[0]["payload"]["symptom"] == "IB Error-12 (IBV_WC_RETRY_EXC_ERR)"
        assert complete_pair_events[0]["payload"]["local_node"] == "test-node-delta"
        assert complete_pair_events[0]["payload"]["remote_node"] == "localhost"
        assert complete_pair_events[0]["payload"]["local_dev_ip"] == "192.0.2.20"
        assert complete_pair_events[0]["payload"]["remote_dev_ip"] == "198.51.100.20"
        assert complete_pair_events[0]["payload"]["local_device"] == "mlx5_0"

    def test_legacy_got_completion_with_error_creates_node_ib_symptom(self, app_tst_bldr, big_jobinfo):
        """Legacy no-peer NCCL completion warnings emit a node-level IB symptom."""
        data = [
            "host-98f6741a:1234567:1234568 [2] ib_plugin.c:1007 NCCL WARN NET/IB : Got completion with error 12, opcode 1, len 0, vendor err 129"
        ]
        result = app_tst_bldr(big_jobinfo, data)
        events = self._get_events_by_type(result.event_stream, "Attribution.node.bad")

        assert len(events) >= 1
        assert events[0]["payload"]["symptom"] == "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)"

    def test_hostname_resolution_failure_uses_ip(self, app_tst_bldr, big_jobinfo):
        data = [
            "transport/net_ib.cc:2359 NCCL WARN NET/IB: Got completion from peer 192.0.2.1<10904> with status=12 opcode=129 len=374784 vendor err 0 (Recv) localGid ::ffff:192.0.2.20 remoteGids::ffff:192.0.2.1 hca rocep206s0"
        ]
        with patch("socket.gethostbyaddr", side_effect=socket.herror):
            result = app_tst_bldr(big_jobinfo, data)

        events = self._get_events_by_type(result.event_stream, "Attribution.node.bad_peers")
        assert len(events) >= 1
        assert "192.0.2.1" in events[0]["payload"]["peer"]


def test_multicast_skip_marker_logs_from_applog(app_tst_bldr, big_jobinfo):
    test_data = [
        make_applog_entry(
            body="Nov 15 10:30:00 host-9ad02e1b Gracefully skipping multicast initialization",
            hostname=big_jobinfo.nodes[0],
            timestamp_ns=1700005000000000000,
            severity="info",
        )
    ]

    with patch.object(logger, "info") as mock_info:
        app_tst_bldr(big_jobinfo, test_data, format=ObservationFormat.RAW_LOKI_STREAMS)

    mock_info.assert_any_call("Applog marker detected: Gracefully skipping multicast initialization")
