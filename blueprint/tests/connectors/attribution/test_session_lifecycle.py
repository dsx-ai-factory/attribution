# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session ownership regressions against the end-to-end test Service."""

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from unittest.mock import Mock
from uuid import uuid4

import pytest
from nvidia_attribution_client import ObservationV2

from attribution_blueprint.connectors.attribution import AttributionConnector
from attribution_blueprint.orchestrator import AttributionOrchestrator

pytestmark = pytest.mark.integration

_XID_SYMPTOM = "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS"


@pytest.fixture
def service_url():
    url = os.environ.get("ATTRIBUTION_TEST_SERVICE_URL")
    if not url:
        pytest.skip("set ATTRIBUTION_TEST_SERVICE_URL to a disposable test Service")
    return url


@pytest.fixture
def ctx():
    return {
        "id": f"session-ownership-{uuid4()}",
        "workload_type": "slurm",
        "state": "FAILED",
        "nodes": ["node-001"],
        "start_time": datetime(2026, 5, 1, 12, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc),
    }


def _observation(source="syslog", body="NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus."):
    return {
        "context": {"source": source, "time_interval": {"start": None, "end": None}},
        "body": json.dumps(
            [
                {
                    "body": body,
                    "ts": int(datetime(2026, 5, 1, 12, 15, tzinfo=timezone.utc).timestamp()) * 1_000_000_000,
                    "resources": [{"kind": "node", "hostname": "node-001"}],
                }
            ]
        ),
    }


def _pipeline(service_url, ctx):
    eca = Mock()
    eca.discover.return_value = [ctx]
    collector = Mock(source="syslog")
    collector.collect.return_value = [_observation()]
    output = Mock()
    pipeline = AttributionOrchestrator(
        execution_context_adapters=[eca],
        log_connectors=[collector],
        metrics_connectors=[],
        attribution=AttributionConnector(service_url),
        outputs=[output],
    )
    return pipeline, output


def test_duplicate_run_preserves_active_session_and_evidence(service_url, ctx):
    pipeline, output = _pipeline(service_url, ctx)
    owner = AttributionConnector(service_url)
    attributor_id = owner.create_attributor(pipeline._build_attributor_info(ctx))
    try:
        observation_id = owner.ingest_observation(
            attributor_id, ObservationV2.model_validate(_observation())
        ).observation_id
        before = owner.get_attribution(attributor_id)
        assert before["observation_context"] == [observation_id]
        assert before["symptoms"][_XID_SYMPTOM][0]["source"]["observation_id"] == observation_id

        assert pipeline.analyze() == []

        assert pipeline.last_run_stats == {"attempted": 1, "succeeded": 0, "conflicted": 1}
        assert pipeline.last_run_summaries == [
            {
                "context_id": ctx["id"],
                "job_id": ctx["id"],
                "attempt_id": "",
                "status": "conflicted",
                "attributor_id": attributor_id,
            }
        ]
        output.write.assert_not_called()
        output.close.assert_called_once_with()
        after = owner.get_attribution(attributor_id)
        assert after["observation_context"] == before["observation_context"]
        assert after["symptoms"] == before["symptoms"]
        assert after["verdict"] == before["verdict"]

        # The original owner can keep adding evidence after the rejected run.
        next_id = owner.ingest_observation(
            attributor_id,
            ObservationV2.model_validate(_observation(source="applog", body="[rank 0] training stopped")),
        ).observation_id
        assert set(owner.get_attribution(attributor_id)["observation_context"]) == {observation_id, next_id}
    finally:
        owner.delete_attributor(attributor_id)

    # Owner cleanup makes the same workload available to subsequent runs.
    for _ in range(2):
        retry, retry_output = _pipeline(service_url, ctx)
        assert len(retry.analyze()) == 1
        assert retry.last_run_stats == {"attempted": 1, "succeeded": 1, "conflicted": 0}
        retry_output.write.assert_called_once()


def test_overlapping_blueprint_runs_preserve_the_first_runs_evidence(service_url, ctx, monkeypatch):
    first, first_output = _pipeline(service_url, ctx)
    second, second_output = _pipeline(service_url, ctx)
    evidence_ready = Event()
    resume_owner = Event()
    get_attribution = first.attribution.get_attribution
    snapshots = Future()

    def pause_after_ingestion(attributor_id):
        try:
            before = get_attribution(attributor_id)
            evidence_ready.set()
            if not resume_owner.wait(timeout=30):
                raise TimeoutError("the duplicate run did not finish")
            after = get_attribution(attributor_id)
            snapshots.set_result((before, after))
            return after
        except Exception as error:
            # Surface the original error even though analyze() contains failures.
            snapshots.set_exception(error)
            evidence_ready.set()
            raise

    monkeypatch.setattr(first.attribution, "get_attribution", pause_after_ingestion)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(first.analyze)
        try:
            assert evidence_ready.wait(timeout=30), "the first run did not reach attribution"
            assert second.analyze() == []
        finally:
            resume_owner.set()
        results = pending.result(timeout=30)

    before, after = snapshots.result(timeout=30)
    assert before["observation_context"]
    assert _XID_SYMPTOM in before["symptoms"]
    assert after["observation_context"] == before["observation_context"]
    assert after["symptoms"] == before["symptoms"]
    assert after["verdict"] == before["verdict"]
    assert len(results) == 1
    assert results[0]["observation_context"] == before["observation_context"]
    assert results[0]["symptoms"][_XID_SYMPTOM][0]["source"] == before["symptoms"][_XID_SYMPTOM][0]["source"]
    assert first.last_run_stats == {"attempted": 1, "succeeded": 1, "conflicted": 0}
    assert second.last_run_stats == {"attempted": 1, "succeeded": 0, "conflicted": 1}
    first_output.write.assert_called_once()
    second_output.write.assert_not_called()

    retry, _ = _pipeline(service_url, ctx)
    assert len(retry.analyze()) == 1
