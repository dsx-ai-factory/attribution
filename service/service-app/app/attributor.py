# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol, TypeAlias

import attribution_lib
from attribution_lib.attributor import Attributor
from attribution_lib.checkpoint_backends import MemoryCheckpointBackend
from attribution_lib.core import AttributionResult
from attribution_lib.decision_trace import DecisionDAG, build_decision_graph
from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.events import EventManager
from attribution_lib.exceptions import (
    DuplicateObservationError as LibDuplicateObservationError,
    EventStorageOutOfMemoryError as LibEventStorageOutOfMemoryError,
    MismatchedSourceAndFormatError as LibMismatchedSourceAndFormatError,
)
from attribution_lib.job_info import JobInfo
from attribution_lib.rules import CollectorRule, RuleId, RuleParameters
from loguru import logger
from pydantic import ConfigDict, Field, field_validator

from app.exceptions import (
    DAG_SHAPE_ERRORS,
    DuplicateObservationError,
    EventStorageOverflowError,
    IncompatibleObservationSourceFormatError,
)
from app.observation import Observation, ObservationId
from app.util import HashableModel


def _allow_disk_spill() -> bool:
    """Whether an attributor's per-connection DuckDB storage may spill excess
    data (beyond memory_limit) to a local .tmp directory on disk instead of
    raising. Defaults to false: this service is primarily deployed in
    containers with little or no writable disk, so failing fast with a
    catchable, isolated-to-one-request error is safer than silently writing
    to whatever ephemeral storage the container happens to have. Deployments
    where local disk can be assumed (e.g. bare metal) can opt back into
    DuckDB's own default (graceful degradation via disk spill) with
    ATTRIBUTOR_ALLOW_DISK_SPILL=true.
    """
    return os.environ.get("ATTRIBUTOR_ALLOW_DISK_SPILL", "false").strip().lower() == "true"


def _duckdb_memory_limit() -> str:
    """Per-attributor DuckDB `memory_limit`, as a DuckDB size string (e.g.
    `"128MB"`). Each attributor's dataset is one job's worth of events and
    always tiny, so 128MB is a generous default ceiling, not a tuned value
    every deployment should be stuck with -- override it with
    ATTRIBUTOR_DUCKDB_MEMORY_LIMIT.
    """
    return os.environ.get("ATTRIBUTOR_DUCKDB_MEMORY_LIMIT", "128MB").strip()


AttributorId: TypeAlias = str


@dataclass(frozen=True)
class AttributionContext:
    """Raw attribution result + metadata needed to build any versioned response."""

    parameters: "AttributionParameters"
    observation_ids: set[ObservationId]
    result: AttributionResult
    decision_graph_requested: bool
    decision_graph: Optional[DecisionDAG]


class WorkloadTypeEnum(str, Enum):
    """The types of workload identifiers we can find."""

    UNDEFINED = ""
    SLURM = "slurm"
    RUNAI = "runai"


class AttributorMetadata(HashableModel):
    """Additional metadata to help with tracing, etc"""

    cluster: str
    agent: str
    tenant: str
    ruleset: str


class AttributionParameters(HashableModel):
    """Parameters to guide attribution."""

    # Reject unknown fields so V1 clients still posting the old `nvrx_straggler_parameters` /
    # `megatron_straggler_parameters` shape get a 422 instead of silently running with lib defaults.
    model_config = ConfigDict(extra="forbid")

    rule_params: dict[RuleId, RuleParameters] = Field(default_factory=dict)

    @field_validator("rule_params")
    @classmethod
    def _validate_against_ruleset(cls, value: dict[RuleId, RuleParameters]) -> dict[RuleId, RuleParameters]:
        """Validate each rule_params entry against the registered collector rules.

        Raises ValueError for unknown rule IDs.
        """
        # TODO: Currently only allowing parameters for collector rules.
        for rule_id, rp in value.items():
            rule = attribution_lib.get_rules_engine().get_rule(rule_id)
            if not isinstance(rule, CollectorRule):
                raise ValueError(f"unknown rule id: {rule_id!r}")
            rule.parse_params(rp)
        return value


class _GuardedRecorder(DecisionRecorder):
    """Recorder that keeps decision-trace failures out of the attribution itself.

    attribution_lib records decision nodes from inside its attribution run and does not
    guard those calls, so a recording failure would cost the caller an otherwise-sound
    verdict. Failures are absorbed here and flagged, so an incomplete trace is dropped
    instead of being served as the whole picture.
    """

    def __init__(self, job_id: str) -> None:
        super().__init__(job_id)
        self.failed = False

    def record(self, *args: Any, **kwargs: Any) -> None:
        """Record one decision node, absorbing any failure."""
        self._guard(super().record, *args, **kwargs)

    def record_refiner_diff(self, *args: Any, **kwargs: Any) -> None:
        """Record one refiner's symptom and bad-node diff, absorbing any failure."""
        self._guard(super().record_refiner_diff, *args, **kwargs)

    def _guard(self, record: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """Run one recording call, flagging the trace as incomplete if it raises.

        Args:
            record: The superclass recording method to call.
            *args: Positional arguments supplied by the lib.
            **kwargs: Keyword arguments supplied by the lib.
        """
        try:
            record(*args, **kwargs)
        except DAG_SHAPE_ERRORS as e:
            self.failed = True
            logger.warning(f"Failed to record a decision node, dropping the decision graph: {e}")


def _try_build_decision_graph(
    recorder: Optional[_GuardedRecorder],
    result: AttributionResult,
) -> Optional[DecisionDAG]:
    """Assemble the recorded Decision DAG without risking the attribution itself.

    Graph assembly walks the attribution result, so a lib-side shape change can
    raise even though the verdict is sound. The graph is a diagnostic add-on and
    is dropped rather than failing the request.

    Args:
        recorder: Recorder passed to attribution, or None when no graph was requested.
        result: Completed attribution result.

    Returns:
        The Decision DAG, or None when unrequested, incompletely recorded, or not constructible.
    """
    if recorder is None:
        return None
    if recorder.failed:
        logger.warning("Decision trace is incomplete, omitting the graph from the verdict")
        return None
    try:
        return build_decision_graph(recorder, result)
    except DAG_SHAPE_ERRORS as e:
        logger.warning(f"Failed to build decision graph, omitting it from the verdict: {e}")
        return None


class AttributorInput(Protocol):
    """Contract every versioned wire ``AttributorInfo`` satisfies."""

    parameters: AttributionParameters

    def to_lib_jobinfo(self) -> JobInfo: ...
    def __hash__(self) -> int: ...


class AttributorWrapper:
    id: AttributorId
    attributor: Attributor
    parameters: AttributionParameters
    lock: threading.Lock
    last_access_time: datetime

    def __init__(self, info: AttributorInput):
        job_info = info.to_lib_jobinfo()
        self.id = str(hash(info))

        rules_engine = attribution_lib.get_rules_engine()
        # Each attributor's dataset is one job's worth of events and never
        # benefits from intra-query parallelism, so cap DuckDB's per-connection
        # host-sized defaults (threads=cpu_count, memory_limit=~80% of RAM).
        event_manager = EventManager(
            rules_engine=rules_engine,
            threads=1,
            memory_limit=_duckdb_memory_limit(),
            allow_disk_spill=_allow_disk_spill(),
        )
        event_stream = event_manager.create_stream(str(job_info.id))

        self.attributor = Attributor.new(event_stream, job_info)
        self._checkpoint_backend = MemoryCheckpointBackend()
        self._timezone = job_info.timezone
        self.parameters = info.parameters
        self.lock = threading.Lock()
        self.last_access_time = datetime.now()
        logger.info(f"Created AttributorWrapper with id: {self.id} for workload: {job_info.id}")

    @classmethod
    def new(cls, info: AttributorInput) -> "AttributorWrapper":
        """Create a new attributor."""
        logger.debug(f"Creating new AttributorWrapper (hash={hash(info)})")
        return cls(info)

    def ingest(self, observation: Observation) -> ObservationId:
        """Ingest observation into this instance and return ingest_id. Raise DuplicateObservationError if we have seen this already"""
        with self.lock:
            try:
                logger.debug(f"Ingesting observation for attributor {self.id}")
                _id = self.attributor.ingest(observation.to_lib_observation(self._timezone))
                self.last_access_time = datetime.now()
                logger.debug(f"Successfully ingested observation for attributor {self.id}, returned ingest_id: {_id}")
            except LibDuplicateObservationError as e:
                logger.warning(f"Duplicate observation error for attributor {self.id}: {e.id}")
                raise DuplicateObservationError.new(e.id).to_http_exception() from e
            except LibMismatchedSourceAndFormatError as e:
                logger.error(
                    f"Source/format mismatch error for attributor {self.id}: source={e.source}, format={e.format}"
                )
                raise IncompatibleObservationSourceFormatError.new(
                    source=e.source,
                    format=e.format,
                ).to_http_exception() from e
            except LibEventStorageOutOfMemoryError as e:
                logger.error(f"Event storage overflow for attributor {self.id}: {e}")
                raise EventStorageOverflowError.new(self.id).to_http_exception() from e
            return _id

    def attribute(self, include_decision_graph: bool = False) -> AttributionContext:
        """Run attribution and optionally build its runtime decision graph.

        Args:
            include_decision_graph: Whether to record and build the Decision DAG.

        Returns:
            Attribution result context for a versioned API response.
        """
        with self.lock:
            logger.debug(f"Starting attribution for attributor {self.id}")
            try:
                recorder = _GuardedRecorder(str(self.attributor.job_info.id)) if include_decision_graph else None
                result = self.attributor.attribute(
                    self._checkpoint_backend,
                    recorder=recorder,
                    rule_params=self.parameters.rule_params,
                    run_collectors=True,
                )
                decision_graph = _try_build_decision_graph(recorder, result)
                self.last_access_time = datetime.now()
                ctx = AttributionContext(
                    result=result,
                    parameters=self.parameters,
                    observation_ids=self.attributor.processed_ids(),
                    decision_graph_requested=include_decision_graph,
                    decision_graph=decision_graph,
                )
                logger.info(f"Attribution completed for attributor {self.id}")
                return ctx
            except LibEventStorageOutOfMemoryError as e:
                logger.error(f"Event storage overflow for attributor {self.id}: {e}")
                raise EventStorageOverflowError.new(self.id).to_http_exception() from e
            except Exception as e:
                logger.error(f"Attribution failed for attributor {self.id}: {str(e)}")
                raise
