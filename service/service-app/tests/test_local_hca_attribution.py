# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify local RDMA diagnoses and provenance over the v2 API."""

import json
from datetime import datetime, timedelta, timezone

import attribution_lib
import pytest

from tests.conftest import make_up_attributor_info, post_attributor


@pytest.mark.parametrize("corroborated", [False, True])
def test_local_hca_endpoint_and_corroborated_diagnosis(client, monkeypatch, corroborated):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    info = make_up_attributor_info(
        "v2",
        nodes={"node-a": [0], "node-b": [1]},
        status="FAILED",
        job_start_time=start,
        job_end_time=end,
        timezone="UTC",
    )
    att_id = post_attributor(client, info)
    monkeypatch.setattr(attribution_lib, "ip_to_nodenames", {"192.0.2.1": "node-a", "192.0.2.2": "node-b"})
    line = (
        "node-a:123:456 [0] NCCL WARN Call to ibv_modify_qp failed with 22 Invalid argument, "
        "on dev mlx5_4:1, local GID index 3, local GID ::ffff:192.0.2.1, remote GID ::ffff:192.0.2.2"
    )
    if corroborated:
        line = "node-a:123:456 [0] NCCL WARN Call to ibv_query_qp failed with error Connection timed out errno 110 on dev 0000:03:00.3"
    observations = [("applog", line)]
    if corroborated:
        observations.append(
            (
                "syslog",
                "mlx5_core 0000:03:00.3: Health issue observed, irisc not responding, severity(3) ERROR:",
            )
        )
    for index, (source, body) in enumerate(observations):
        response = client.post(
            f"/v2/attributor/{att_id}/observation",
            json={
                "context": {"source": source, "time_interval": {"start": start.isoformat(), "end": end.isoformat()}},
                "body": json.dumps(
                    [
                        {
                            "body": body,
                            "ts": int((start + timedelta(seconds=20 + index)).timestamp() * 1e9),
                            "resources": [{"kind": "node", "hostname": "node-a"}],
                        }
                    ]
                ),
            },
        )
        assert response.status_code == 201, response.text
    response = client.get(
        f"/v2/attributor/{att_id}/attribution", params={"include_interesting": True, "include_decision_graph": True}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    expected = "Fail InfraHW" if corroborated else "Fail InfraSW"
    assert body["verdict"]["attribution_outcome"] == expected
    sites = [site for values in body["symptoms"].values() for site in values if site["type"] == "NodeSource"]
    assert any(site["node"] == "node-a" and site["status"] == "bad" for site in sites)
    assert not any(site["node"] == "node-b" and site["status"] == "bad" for site in sites)
    if corroborated:
        assert body["verdict"]["root_symptom"]["symptom"] == "Corroborated local HCA command failure"
        assert body["verdict"]["root_symptom"]["catalog_id"] == "local-hca-command-failure"
    else:
        assert any(site["node"] == "node-b" and site["status"] == "interesting" for site in sites)
        assert "mlx5_4" in body["collectors"]["bad_pairs"]
    assert "ibv_" in json.dumps(body["verdict"]["decision_graph"])
