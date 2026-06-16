# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, TypeAlias

import attribution_lib
from attribution_lib.attributor import Attributor
from attribution_lib.bundled_rules.common.straggler_types import StragglerScoreType
from attribution_lib.checkpoint_backends import MemoryCheckpointBackend
from attribution_lib.core import AttributionResult
from attribution_lib.events import EventManager
from attribution_lib.exceptions import (
    DuplicateObservationError as LibDuplicateObservationError,
    MismatchedSourceAndFormatError as LibMismatchedSourceAndFormatError,
)
from attribution_lib.job_info import JobInfo
from loguru import logger
from pydantic import Field

from app.exceptions import (
    DuplicateObservationError,
    IncompatibleObservationSourceFormatError,
)
from app.observation import Observation, ObservationId
from app.util import HashableModel

AttributorId: TypeAlias = str


@dataclass(frozen=True)
class AttributionContext:
    """Raw attribution result + metadata needed to build any versioned response."""

    parameters: "AttributionParameters"
    observation_ids: set[ObservationId]
    result: AttributionResult


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


class NVRXStragglerParameters(HashableModel):
    """Parameters to guide NVRX stragglers"""

    straggler_threshold: float = 0.7
    score_types: list[StragglerScoreType] = [StragglerScoreType.RELATIVE]


class MegatronStragglerParameters(HashableModel):
    """Parameters to guide Megatron stragglers"""

    straggler_threshold: float = 0.95  # Relative threshold (0.0-1.0) - 95%
    score_types: list[StragglerScoreType] = [StragglerScoreType.INDIVIDUAL]
    infer_threshold_from_data: bool = True  # Auto-calculate threshold from top ranks


class AttributionParameters(HashableModel):
    """Parameters to guide attribution"""

    nvrx_straggler_parameters: NVRXStragglerParameters = Field(default_factory=NVRXStragglerParameters)
    megatron_straggler_parameters: MegatronStragglerParameters = Field(default_factory=MegatronStragglerParameters)


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
        event_manager = EventManager(rules_engine=rules_engine)
        event_stream = event_manager.create_stream(str(job_info.id))

        self.attributor = Attributor.new(event_stream, job_info)
        self._checkpoint_backend = MemoryCheckpointBackend()
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
                id = self.attributor.ingest(observation.to_lib_observation())
                self.last_access_time = datetime.now()
                logger.info(f"Successfully ingested observation for attributor {self.id}, returned ingest_id: {id}")
            except LibDuplicateObservationError as e:
                logger.warning(f"Duplicate observation error for attributor {self.id}: {e.id}")
                raise DuplicateObservationError.new(e.id).to_http_exception() from e
            except LibMismatchedSourceAndFormatError as e:
                logger.error(
                    f"Source/format mismatch error for attributor {self.id}: source={e.source}, format={e.format}"
                )
                raise IncompatibleObservationSourceFormatError(source=e.source, format=e.format) from e
            return id

    def attribute(self) -> AttributionContext:
        with self.lock:
            logger.debug(f"Starting attribution for attributor {self.id}")
            try:
                result = self.attributor.attribute(self._checkpoint_backend)
                self.last_access_time = datetime.now()
                ctx = AttributionContext(
                    result=result,
                    parameters=self.parameters,
                    observation_ids=self.attributor.processed_ids(),
                )
                logger.info(f"Attribution completed for attributor {self.id}")
                return ctx
            except Exception as e:
                logger.error(f"Attribution failed for attributor {self.id}: {str(e)}")
                raise
