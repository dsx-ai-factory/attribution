# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Mapping
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

import attribution_lib as _attribution_lib
from attribution_lib.checkpoint_backends import NullCheckpointBackend
from attribution_lib.core import AttributionResult, CalcBadNodesAttributionResult, attribute, calculate_bad_nodes
from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.events import EventManager, EventStream
from attribution_lib.exceptions import (
    BadIngestedObservationStateError,
    DuplicateObservationError,
    MismatchedSourceAndFormatError,
    ReconsumeIngestedObservationError,
    UnhandledObservationSourceError,
    UnknownObservationSourceError,
)
from attribution_lib.extraction_cache import (
    CheckpointBackend,
    DataDiffResult,
    DataFingerprint,
    compute_data_fingerprint,
    load_or_update_extraction_cache,
    save_extraction_cache,
)
from attribution_lib.job_info import JobInfo
from attribution_lib.observation import (
    Context as ObservationContext,
    Observation,
)
from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
    Id as ObservationId,
    Observation as InboundObservation,
    Source as ObservationSource,
)
from attribution_lib.observation.processing import EntrySourceBase
from attribution_lib.rule_types.definitions import RuleId, RuleParameters
from attribution_lib.rules_engine import SOURCE_CONFIGS, body_fallbacks_for, process_observation_source


class ConsumedObservation:
    """Describes association of some observation data with a context."""

    context: ObservationContext
    result: CalcBadNodesAttributionResult

    def __init__(self, context: ObservationContext, result: CalcBadNodesAttributionResult):
        self.context = context
        self.result = result

    @classmethod
    def new(
        cls,
        event_stream: EventStream,
        job_info: JobInfo,
        observation: Observation,
        pre_filtered: bool = False,
    ) -> "ConsumedObservation":
        """Validate source/format and extract events into the stream.

        Extraction-only: runs extraction rules to populate the event stream,
        but does not call calculate_bad_nodes(). filter_events and
        from_event_stream are deferred to the caller (Attributor._run_attribution())
        so that scrubbers/DAG refiners see the complete cross-source picture.
        """
        context = observation.context
        _validate_source_format(context)
        result = CalcBadNodesAttributionResult()

        if context.source != ObservationSource.JOB_KEYWORDS:
            entry_source = EntrySourceBase.from_observation(
                observation, body_fallbacks=body_fallbacks_for(context.source, job_info)
            )
            config = SOURCE_CONFIGS.get(entry_source.context.source)
            if config is not None and config.requires_job_nodes and not job_info.nodes:
                pass
            else:
                source_raw_keywords = process_observation_source(
                    event_stream=event_stream,
                    observation_source=entry_source,
                    job_info=job_info,
                    rules_engine=_attribution_lib.get_rules_engine(),
                    pre_filtered=pre_filtered,
                )
                for keyword in source_raw_keywords:
                    result.raw_keywords.add(job_info.source_value(keyword, datum=None, rule="pre-filter include"))

        return cls(context, result)


class IngestedObservation:
    """Representation of injested observations"""

    ingestion_time: datetime

    # this is clumsy and I'd like a better way to represent it
    unconsumed: Optional[Observation]
    consumed: Optional[ConsumedObservation]

    def __init__(self, observation: Observation):
        self.unconsumed = observation
        self.ingestion_time = datetime.now(ZoneInfo("UTC"))
        self.consumed = None

    def is_processed(self) -> Optional[ConsumedObservation]:
        if self.consumed is not None:
            if self.unconsumed is not None:
                raise BadIngestedObservationStateError(self)
            return self.consumed
        elif self.consumed is None and self.unconsumed is None:
            raise BadIngestedObservationStateError(self)
        return None

    def consume(self, event_stream, job_info: JobInfo, pre_filtered: bool = False) -> None:
        if self.is_processed() is not None:
            raise ReconsumeIngestedObservationError(self)

        if (unconsumed := self.unconsumed) is not None:
            self.consumed = ConsumedObservation.new(event_stream, job_info, unconsumed, pre_filtered=pre_filtered)
        self.unconsumed = None

    def __repr__(self) -> str:
        return f"IngestedObservation(ingestion_time={self.ingestion_time}, unconsumed={self.unconsumed}, consumed={self.consumed})"


class Attributor:
    job_info: JobInfo

    event_stream: EventStream
    ingested_observations: dict[ObservationId, IngestedObservation]
    _raw_keywords_observation: Optional[Observation]
    _batch_raw_keywords: set[str]
    _source_fingerprints: dict[str, DataFingerprint]

    def __init__(
        self,
        event_stream: EventStream,
        job_info: JobInfo,
    ):
        """Constructor for attributor."""
        self.job_info = job_info
        self.event_stream = event_stream
        self.ingested_observations = dict()
        self._raw_keywords_observation = None
        self._batch_raw_keywords = set()
        self._source_fingerprints = {}

    @classmethod
    def new(
        cls,
        event_stream: EventStream,
        job_info: JobInfo,
    ) -> "Attributor":
        """Create a new attributor.

        Args:
            event_stream: The event stream to capture events from observations.
            job_info: The job information to use for the attributor.
        """
        return Attributor(event_stream, job_info)

    def ingest(self, observation: InboundObservation) -> ObservationId:
        """Ingest observation: validate format and extract events into the stream.

        Extraction-only — filter_events and from_event_stream run once in
        _run_attribution() so scrubbers/DAG refiners see the complete
        cross-source picture before making decisions.
        """
        new_observation = Observation.from_inbound_observation(observation)
        if new_observation.context.id in self.ingested_observations:
            raise DuplicateObservationError(new_observation.context.id)

        # Register observation context in the database immediately upon ingestion
        self.event_stream.register_observation_context(new_observation.context)

        if new_observation.context.source == ObservationSource.JOB_KEYWORDS:
            self._raw_keywords_observation = new_observation
        else:
            fingerprint = compute_data_fingerprint(
                observation.lines,
                extraction_context=observation.context,
                body_fallbacks=body_fallbacks_for(observation.context.source, self.job_info),
            )
            if fingerprint is not None:
                self._source_fingerprints[new_observation.context.source.value] = fingerprint

        # For now, eagerly process the observation
        # In future, we can do this asynchronously
        ingested = IngestedObservation(new_observation)
        ingested.consumed = ConsumedObservation.new(
            self.event_stream,
            self.job_info,
            new_observation,
            pre_filtered=new_observation.context.pre_filtered,
        )
        ingested.unconsumed = None
        self.ingested_observations[new_observation.context.id] = ingested
        return new_observation.context.id

    def processed_ids(self) -> set[ObservationId]:
        return {
            ing_obs.consumed.context.id
            for ing_obs in self.ingested_observations.values()
            if ing_obs.is_processed() and ing_obs.consumed is not None
        }

    def process_observation(self, new_obs: IngestedObservation) -> None:
        new_obs.consume(self.event_stream, self.job_info)

    def _run_attribution(
        self,
        recorder: Optional[DecisionRecorder] = None,
        rule_params: Optional[Mapping[RuleId, RuleParameters]] = None,
        run_collectors: bool = False,
    ) -> AttributionResult:
        """Run attribution once on the complete event stream.

        Calls calculate_bad_nodes (which runs filter_events + from_event_stream)
        on all accumulated events, ensuring cross-source scrub/refiner decisions
        are made with the full picture.

        Internal helper -- external callers should use attribute() which wraps
        this with extraction caching and mutation safety.
        """
        self.event_stream.flush()

        raw_keywords_entry = None
        if self._raw_keywords_observation is not None:
            raw_keywords_entry = EntrySourceBase.from_observation(self._raw_keywords_observation)

        bad_node_result = calculate_bad_nodes(
            event_stream=self.event_stream,
            job_info=self.job_info,
            raw_keywords_in=raw_keywords_entry,
        )
        for ingested_observation in self.ingested_observations.values():
            if ingested_observation.consumed is not None:
                bad_node_result.raw_keywords_update(ingested_observation.consumed.result.raw_keywords)
        for keyword in self._batch_raw_keywords:
            bad_node_result.raw_keywords.add(self.job_info.source_value(keyword, datum=None, rule="pre-filter include"))
        return attribute(
            event_stream=self.event_stream,
            job_info=self.job_info,
            bad_node_result=bad_node_result,
            extract_job_metadata_from_logfile=False,
            recorder=recorder,
            rule_params=rule_params,
            run_collectors=run_collectors,
        )

    @staticmethod
    def get_cache_key(job_info: JobInfo) -> str:
        """Derive the cache key for a job. Shared with callers that need to
        access cache metadata independently (e.g. outcome tracking)."""
        return f"{job_info.id}-{int(job_info.job_start_time.timestamp())}"

    def ingest_batch(
        self,
        observations: list[InboundObservation],
        checkpoint_backend: CheckpointBackend,
        skipped_sources: Optional[set[str]] = None,
    ) -> None:
        """Cache-aware batch ingestion. Fingerprints observations, diffs
        against cached state, extracts only changed/new sources. Replaces
        self.event_stream for a subsequent attribute() call."""
        cache_key = self.get_cache_key(self.job_info)
        job_id_str = str(self.job_info.id)
        rules_engine = _attribution_lib.get_rules_engine()

        extraction_observations = []
        for observation in observations:
            if observation.context.source == ObservationSource.JOB_KEYWORDS:
                self._raw_keywords_observation = Observation.from_inbound_observation(observation)
            else:
                extraction_observations.append(observation)

        current_fingerprints = {}
        observations_by_source: dict[str, InboundObservation] = {}
        for observation in extraction_observations:
            source_name = observation.context.source.value
            fingerprint = compute_data_fingerprint(
                observation.lines,
                extraction_context=observation.context,
                body_fallbacks=body_fallbacks_for(observation.context.source, self.job_info),
            )
            if fingerprint is not None:
                current_fingerprints[source_name] = fingerprint
            observations_by_source[source_name] = observation

        self._source_fingerprints = current_fingerprints

        cache_result = load_or_update_extraction_cache(
            checkpoint_backend, cache_key, current_fingerprints, skipped_sources=skipped_sources
        )

        if cache_result.needs_full_reextraction:
            logger.info(
                "job {}: no usable checkpoint, running full extraction on {} sources",
                job_id_str,
                len(extraction_observations),
            )
            event_manager = EventManager(rules_engine=rules_engine)
            event_stream = event_manager.create_stream(job_id_str)
            self._batch_raw_keywords = _extract_observations(
                event_stream, extraction_observations, self.job_info, rules_engine
            )
            event_stream.flush()
            checkpoint_backend.cleanup()
            save_extraction_cache(
                checkpoint_backend,
                cache_key,
                event_manager,
                job_id_str,
                current_fingerprints,
                raw_keywords=self._batch_raw_keywords,
            )
        else:
            if (
                cache_result.event_manager is None
                or cache_result.metadata is None
                or cache_result.metadata.raw_keywords is None
            ):
                raise RuntimeError("Cache reported valid but returned incomplete checkpoint state")
            event_manager = cache_result.event_manager
            self._batch_raw_keywords = set(cache_result.metadata.raw_keywords)
            incremental_observations = []
            for source_name, diff in cache_result.data_diffs.items():
                if diff.result == DataDiffResult.NEW_SOURCE:
                    logger.debug("job {}: new source '{}', will extract fully", job_id_str, source_name)
                    source_observation = observations_by_source.get(source_name)
                    if source_observation is not None:
                        incremental_observations.append(source_observation)
                elif diff.result == DataDiffResult.APPENDED:
                    source_observation = observations_by_source.get(source_name)
                    if source_observation is not None:
                        sliced_lines = source_observation.lines[diff.cached_count :]
                        if sliced_lines:
                            logger.debug(
                                "job {}: source '{}' appended {} new entries",
                                job_id_str,
                                source_name,
                                len(sliced_lines),
                            )
                            sliced_observation = InboundObservation(
                                context=source_observation.context, lines=sliced_lines
                            )
                            incremental_observations.append(sliced_observation)

            if incremental_observations:
                logger.info(
                    "job {}: checkpoint restored, incremental extraction on {} sources",
                    job_id_str,
                    len(incremental_observations),
                )
                event_stream = event_manager.create_stream(job_id_str)
                self._batch_raw_keywords.update(
                    _extract_observations(event_stream, incremental_observations, self.job_info, rules_engine)
                )
                event_stream.flush()
                save_extraction_cache(
                    checkpoint_backend,
                    cache_key,
                    event_manager,
                    job_id_str,
                    current_fingerprints,
                    raw_keywords=self._batch_raw_keywords,
                )
            else:
                logger.info("job {}: checkpoint restored, all sources unchanged — skipping extraction", job_id_str)

        self.event_stream = event_manager.create_stream(job_id_str)

    def attribute(
        self,
        checkpoint_backend: Optional[CheckpointBackend] = None,
        recorder: Optional[DecisionRecorder] = None,
        rule_params: Optional[Mapping[RuleId, RuleParameters]] = None,
        run_collectors: bool = False,
    ) -> AttributionResult:
        """Run attribution with checkpoint support.

        Saves extraction cache if fingerprints changed since last save,
        protects the event table from mutation via checkpoint/restore,
        then runs attribution.

        Works after either ingest() (streaming) or ingest_batch() (batch).
        When no backend is provided, a NullCheckpointBackend is used so
        mutation protection still applies but nothing is persisted.
        """
        if checkpoint_backend is None:
            checkpoint_backend = NullCheckpointBackend()
        cache_key = self.get_cache_key(self.job_info)
        job_id_str = str(self.job_info.id)
        event_manager = self.event_stream.mgr
        current_fingerprints = dict(self._source_fingerprints)

        needs_save = True
        metadata = checkpoint_backend.load_metadata(cache_key)
        if metadata is not None:
            current_extractor_hash = _attribution_lib.get_rules_engine().extractor_hash()
            if metadata.extractor_hash == current_extractor_hash and metadata.data_fingerprints == current_fingerprints:
                needs_save = False

        if needs_save:
            raw_keywords = set(self._batch_raw_keywords)
            for ingested_observation in self.ingested_observations.values():
                if ingested_observation.consumed is not None:
                    raw_keywords.update(ingested_observation.consumed.result.raw_keywords.symptoms())
            save_extraction_cache(
                checkpoint_backend,
                cache_key,
                event_manager,
                job_id_str,
                current_fingerprints,
                raw_keywords=raw_keywords,
            )

        ckpt_name = f"_attr_{cache_key}"
        event_manager.checkpoint(ckpt_name)
        try:
            return self._run_attribution(
                recorder=recorder,
                rule_params=rule_params,
                run_collectors=run_collectors,
            )
        finally:
            event_manager.restore(ckpt_name)
            event_manager.drop_checkpoint(ckpt_name)


def _extract_observations(
    event_stream: EventStream,
    observations_to_extract: list[InboundObservation],
    job_info: JobInfo,
    rules_engine,
) -> set[str]:
    """Run extraction rules and return their raw keyword evidence."""
    raw_keywords: set[str] = set()
    for inbound_observation in observations_to_extract:
        observation = Observation.from_inbound_observation(inbound_observation)
        event_stream.register_observation_context(observation.context)
        entry_source = EntrySourceBase.from_observation(
            observation, body_fallbacks=body_fallbacks_for(observation.context.source, job_info)
        )

        source_type = entry_source.context.source
        config = SOURCE_CONFIGS.get(source_type)
        if config is not None and config.requires_job_nodes and not job_info.nodes:
            continue

        raw_keywords.update(
            process_observation_source(
                event_stream=event_stream,
                observation_source=entry_source,
                job_info=job_info,
                rules_engine=rules_engine,
                pre_filtered=inbound_observation.context.pre_filtered,
            )
        )
    return raw_keywords


def _validate_source_format(context: ObservationContext) -> None:
    """Validate that observation source and format are compatible."""
    match context.source:
        case ObservationSource.JOB_KEYWORDS:
            if context.format is not ObservationFormat.STRING_LIST:
                raise MismatchedSourceAndFormatError(context.source, context.format)
        case ObservationSource.APPLOG:
            if context.format not in [
                ObservationFormat.RAW_LOG_LINES,
                ObservationFormat.LOKI_DICT_LIST,
                ObservationFormat.RAW_LOKI_STREAMS,
                ObservationFormat.ENTRY_TIMESERIES,
            ]:
                raise MismatchedSourceAndFormatError(context.source, context.format)
        case ObservationSource.SYSLOG:
            if context.format not in [
                ObservationFormat.LOKI_DICT_LIST,
                ObservationFormat.RAW_LOKI_STREAMS,
                ObservationFormat.ENTRY_TIMESERIES,
            ]:
                raise MismatchedSourceAndFormatError(context.source, context.format)
        case ObservationSource.HEALTH_CHECK:
            if context.format not in [
                ObservationFormat.LOKI_DICT_LIST,
                ObservationFormat.RAW_LOKI_STREAMS,
                ObservationFormat.ENTRY_TIMESERIES,
            ]:
                raise MismatchedSourceAndFormatError(context.source, context.format)
        case ObservationSource.UFMLOG:
            if context.format not in [ObservationFormat.UFMLOG, ObservationFormat.ENTRY_TIMESERIES]:
                raise MismatchedSourceAndFormatError(context.source, context.format)
        case ObservationSource.NODE_METRICS:
            if context.format not in [ObservationFormat.PROMETHEUS_METRICS, ObservationFormat.ENTRY_TIMESERIES]:
                raise MismatchedSourceAndFormatError(context.source, context.format)
        case ObservationSource.NVLINK_METRICS:
            if context.format not in [ObservationFormat.PROMETHEUS_METRICS, ObservationFormat.ENTRY_TIMESERIES]:
                raise MismatchedSourceAndFormatError(context.source, context.format)
        case ObservationSource.CSP_EVENT:
            if context.format not in [ObservationFormat.CSP_EVENT_LIST]:
                raise MismatchedSourceAndFormatError(context.source, context.format)
        case (
            ObservationSource.IB_METRICS
            | ObservationSource.ETH_METRICS
            | ObservationSource.NETWORK_SYSLOG
            | ObservationSource.JOB_INFO
        ):
            raise UnhandledObservationSourceError(context.source)
        case _:
            raise UnknownObservationSourceError(context.source)
