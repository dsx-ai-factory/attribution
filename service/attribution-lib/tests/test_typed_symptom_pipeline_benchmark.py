# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance baselines for the string-based symptom runtime.

Run with:
    uv run pytest tests/test_typed_symptom_pipeline_benchmark.py \
        --benchmark-only --benchmark-json=.benchmarks/typed-symptom-baseline.json

The memory case records peak Python-tracked memory in the benchmark JSON.
"""

import gc
import tracemalloc
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
MEMORY_NODE_COUNT = 512
MEMORY_SOURCES_PER_SYMPTOM = 2
# Representative GPU, system, network, and user symptoms used by both
# attribution benchmarks.
SYMPTOM_IDS = (
    "ECC Error",
    "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS",
    "kernel panic",
    "Connection reset by peer",
    "Python Exception",
)
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


def _build_bad_nodes(
    context: ObservationContext,
    node_names: list[str],
    *,
    sources_per_symptom: int = 1,
) -> JobNodeAttributions:
    """Build a representative per-node string symptom set.

    Args:
        context: Observation context used by all source evidence.
        node_names: Nodes to populate.
        sources_per_symptom: Number of evidence sources for each symptom.

    Returns:
        Per-node attributions with GPU, system, network, and user symptoms.
    """
    bad_nodes = JobNodeAttributions()
    for node_index, node in enumerate(node_names):
        for symptom_index, symptom_id in enumerate(SYMPTOM_IDS):
            source_set = SourceSet(symptom_id)
            first_source_index = (node_index * len(SYMPTOM_IDS) + symptom_index) * sources_per_symptom
            for source_offset in range(sources_per_symptom):
                source_set.add(_source(context, first_source_index + source_offset))
            bad_nodes.add([node], source_set)
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

    def clear_events():
        """Clear the prior round outside the timed operation.

        Returns:
            Empty positional and keyword arguments for the benchmark round.
        """
        stream.execute("DELETE FROM events WHERE context_id=?", [stream.ctx_id])
        return (), {}

    def store_events() -> None:
        """Store one event batch.

        Returns:
            None.
        """
        for payload, source in events:
            stream.add_event("Attribution.node.bad", payload, source)
        stream.flush()

    benchmark.pedantic(
        store_events,
        setup=clear_events,
        rounds=BENCHMARK_ROUNDS,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
        iterations=1,
    )

    stored_count = stream.query_raw(
        "SELECT COUNT(*) FROM events WHERE context_id=?",
        [stream.ctx_id],
    )[0][0]
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


@pytest.mark.benchmark
def test_attribute_peak_memory_baseline(benchmark, event_stream, dummy_ctx, simple_job_factory):
    stream = event_stream("typed-symptom-memory-benchmark")
    node_names = [f"node-{index:03d}" for index in range(MEMORY_NODE_COUNT)]
    job_info = simple_job_factory(
        state="FAILED",
        node_to_ranks=make_node_to_ranks(node_names, 1),
    )

    def measure_peak_memory():
        """Measure Python allocations while building and attributing the scale fixture.

        Returns:
            Final outcome and peak Python-tracked bytes.
        """
        tracemalloc.start()
        try:
            bad_nodes = _build_bad_nodes(
                dummy_ctx,
                node_names,
                sources_per_symptom=MEMORY_SOURCES_PER_SYMPTOM,
            )
            result = attribute(
                event_stream=stream,
                job_info=job_info,
                bad_node_result=CalcBadNodesAttributionResult(my_bad_nodes=bad_nodes),
                extract_job_metadata_from_logfile=False,
            )
            _, peak_bytes = tracemalloc.get_traced_memory()
            return result.outcome_attr.value().outcome, peak_bytes
        finally:
            tracemalloc.stop()

    gc.collect()
    outcome, peak_bytes = benchmark.pedantic(
        measure_peak_memory,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["peak_python_bytes"] = peak_bytes
    benchmark.extra_info["nodes"] = MEMORY_NODE_COUNT
    benchmark.extra_info["symptoms_per_node"] = len(SYMPTOM_IDS)
    benchmark.extra_info["sources_per_symptom"] = MEMORY_SOURCES_PER_SYMPTOM

    assert outcome == "Fail InfraHW"
    assert peak_bytes > 0
