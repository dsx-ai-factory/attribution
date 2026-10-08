# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Performance baselines for the string-based symptom runtime.

Run with:
    uv run pytest tests/test_typed_symptom_pipeline_benchmark.py \
        --benchmark-only --benchmark-json=.benchmarks/typed-symptom-baseline.json
"""

from copy import deepcopy

import pytest

from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.core import CalcBadNodesAttributionResult, attribute
from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.resources import AllJobResources
from attribution_lib.sources import Source, SourceSet
from test_support.helpers import make_node_to_ranks

# Large enough to exercise batching and per-node aggregation while keeping the
# benchmark practical in local development and presubmit runs.
EVENT_COUNT = 256
EVENT_NODE_COUNT = 32
ATTRIBUTION_NODE_COUNT = 64
# One warmup excludes DuckDB query compilation; five measured rounds keep the
# baseline stable without making the focused benchmark slow.
BENCHMARK_ROUNDS = 5
BENCHMARK_WARMUP_ROUNDS = 1


def _source(context: ObservationContext, index: int) -> Source:
    """Build deterministic benchmark evidence.

    Args:
        context: Observation context registered with the event stream.
        index: Stable event index used for evidence fields.

    Returns:
        Source evidence for one synthetic attribution event.
    """
    return Source(
        source_context=context,
        line_num=index,
        datum=f"ECC evidence {index}",
        time=context.time_interval.start,
        resources=AllJobResources,
        rule="test_rule",
    )


def _build_bad_nodes(context: ObservationContext, node_names: list[str]) -> JobNodeAttributions:
    """Build a representative per-node string symptom set.

    Args:
        context: Observation context used by all source evidence.
        node_names: Nodes to populate.

    Returns:
        Per-node attributions with GPU, system, network, and user symptoms.
    """
    symptom_ids = (
        "ECC Error",
        "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS",
        "kernel panic",
        "Connection reset by peer",
        "Python Exception",
    )
    bad_nodes = JobNodeAttributions()
    for node_index, node in enumerate(node_names):
        for symptom_index, symptom_id in enumerate(symptom_ids):
            source_index = node_index * len(symptom_ids) + symptom_index
            bad_nodes.add([node], SourceSet.new_one(symptom_id, _source(context, source_index)))
    return bad_nodes


@pytest.mark.benchmark
def test_event_storage_round_trip_baseline(benchmark, event_stream, dummy_ctx):
    stream = event_stream("typed-symptom-event-storage-benchmark")
    stream.register_observation_context(dummy_ctx)
    events = [
        (
            {
                "node": f"node-{index % EVENT_NODE_COUNT:03d}",
                "symptom": "ECC Error",
            },
            _source(dummy_ctx, index),
        )
        for index in range(EVENT_COUNT)
    ]

    def store_and_clear() -> int:
        """Store one event batch and clear it before the next round.

        Returns:
            Number of persisted events.
        """
        for payload, source in events:
            stream.add_event("Attribution.node.bad", payload, source)
        stream.flush()
        count = stream.query_raw(
            "SELECT COUNT(*) FROM events WHERE context_id=?",
            [stream.ctx_id],
        )[0][0]
        stream.execute("DELETE FROM events WHERE context_id=?", [stream.ctx_id])
        return count

    stored_count = benchmark.pedantic(
        store_and_clear,
        rounds=BENCHMARK_ROUNDS,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
        iterations=1,
    )

    assert stored_count == EVENT_COUNT


@pytest.mark.benchmark
def test_event_consumption_baseline(benchmark, event_stream, dummy_ctx, simple_job_factory):
    stream = event_stream("typed-symptom-event-consumption-benchmark")
    stream.register_observation_context(dummy_ctx)
    node_names = [f"node-{index:03d}" for index in range(EVENT_NODE_COUNT)]
    for index in range(EVENT_COUNT):
        stream.add_event(
            "Attribution.node.bad",
            {
                "node": node_names[index % EVENT_NODE_COUNT],
                "symptom": "ECC Error",
            },
            _source(dummy_ctx, index),
        )
    stream.flush()
    job_info = simple_job_factory(
        state="FAILED",
        node_to_ranks=make_node_to_ranks(node_names, 1),
    )

    def consume_without_committing():
        """Consume events inside a transaction so every round sees the same rows.

        Returns:
            Per-node attributions created from the persisted events.
        """
        stream.execute("BEGIN TRANSACTION")
        try:
            return JobNodeAttributions.from_event_stream(stream, job_info)
        finally:
            stream.execute("ROLLBACK")

    attributions = benchmark.pedantic(
        consume_without_committing,
        rounds=BENCHMARK_ROUNDS,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
        iterations=1,
    )

    assert attributions.num_nodes() == EVENT_NODE_COUNT
    assert all(list(group.symptoms()) == ["ECC Error"] for group in attributions.attributions.values())


@pytest.mark.benchmark
def test_attribute_runtime_baseline(benchmark, event_stream, dummy_ctx, simple_job_factory):
    stream = event_stream("typed-symptom-attribution-benchmark")
    node_names = [f"node-{index:03d}" for index in range(ATTRIBUTION_NODE_COUNT)]
    template = _build_bad_nodes(dummy_ctx, node_names)
    job_info = simple_job_factory(
        state="FAILED",
        node_to_ranks=make_node_to_ranks(node_names, 1),
    )

    def run_attribution(
        bad_node_result: CalcBadNodesAttributionResult,
        recorder: DecisionRecorder,
    ):
        """Run the complete post-extraction attribution path.

        Args:
            bad_node_result: Fresh per-node attribution input for one round.
            recorder: Fresh decision recorder for one round.

        Returns:
            Final attribution result for the synthetic job.
        """
        return attribute(
            event_stream=stream,
            job_info=job_info,
            bad_node_result=bad_node_result,
            extract_job_metadata_from_logfile=False,
            recorder=recorder,
        )

    def prepare_round():
        """Give each round fresh containers because refiners mutate them.

        Returns:
            Positional and keyword arguments for one benchmark round.
        """
        return (
            (
                CalcBadNodesAttributionResult(my_bad_nodes=deepcopy(template)),
                DecisionRecorder(job_info.id),
            ),
            {},
        )

    result = benchmark.pedantic(
        run_attribution,
        setup=prepare_round,
        rounds=BENCHMARK_ROUNDS,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
        iterations=1,
    )

    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert all(isinstance(symptom, str) for symptom in result.symptoms.symptoms())
