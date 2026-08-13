# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Exception hierarchy for the attribution library."""

from typing import Any

from pydantic import ValidationError


class AttributionError(Exception):
    """Base class for all attribution library exceptions."""

    pass


class DecisionGraphError(AttributionError):
    """Raised when invalid decision-trace data prevents graph assembly."""

    pass


class ObservationError(AttributionError):
    """Base class for observation-related errors."""

    pass


class FormatError(ObservationError):
    """Base class for format-related errors."""

    pass


class UnawareDatetimeError(FormatError):
    """Exception for when we encounter a 'naive' datetime (a datetime without a timezone."""

    pass


class MismatchedSourceAndFormatError(FormatError):
    """Base class for unsupported format errors."""

    def __init__(self, source: str, format: str) -> None:
        self.source = source
        self.format = format


class UnknownFormatError(FormatError):
    """Base class for unknown format errors."""

    def __init__(self, format: str) -> None:
        self.format = format


class UnknownObservationFormatError(UnknownFormatError):
    """Raised when observation node is unknown."""

    pass


class UnhandledObservationFormatError(UnknownFormatError):
    """Raised when observation format is not handled."""

    pass


class UnknownSourceError(ObservationError):
    """Base class for unknown source errors."""

    def __init__(self, source: Any) -> None:
        self.source = source


class UnknownObservationSourceError(UnknownSourceError):
    """Raised when observation source is unknown."""

    pass


class UnhandledObservationSourceError(UnknownSourceError):
    """Raised when observation source is not handled."""

    pass


class InvalidTimestampSourceError(ObservationError):
    """Raised when timestamp source is invalid."""

    pass


class ProcessingError(ObservationError):
    """Base class for observation processing errors."""

    pass


class UnprocessedObservationError(ProcessingError):
    """Raised when observation has not been processed."""

    pass


class DuplicateObservationError(ProcessingError):
    """Raised when duplicate observation is detected."""

    def __init__(self, id: Any) -> None:
        self.id = id


class BadIngestedObservationStateError(ProcessingError):
    """Raised when ingested observation is in an invalid state."""

    pass


class ReconsumeIngestedObservationError(ProcessingError):
    """Raised when attempting to reconsume an already consumed observation."""

    pass


class UnknownNodeExtractionSourceError(ProcessingError):
    """Raise where we try to do node extraction from a source that we don't support it on"""

    def __init__(self, source: Any) -> None:
        self.source = source


class MissingRequiredFieldError(ProcessingError):
    """Raised when required fields are missing from data."""

    def __init__(self, missing_field: str, data: dict):
        self.missing_field = missing_field
        self.data = data
        super().__init__(f"Missing required field '{missing_field}' in data: {data}")


class UnknownResourceError(AttributionError):
    """Base class for node-related errors."""

    pass


class NoNodesError(UnknownResourceError):
    """Represents that we cannot extract *any* node from a resource."""

    pass


class UnidentifiableNodeError(UnknownResourceError):
    """Raised when node cannot be extracted from data."""

    def __init__(self, body: Any) -> None:
        self.node = body


class UnknownNodeError(UnknownResourceError):
    """Raised when node is not recognized."""

    def __init__(self, node: str) -> None:
        self.node = node


class InvalidRankError(UnknownResourceError):
    """Raised when rank is not recognized."""

    def __init__(self, rank: int) -> None:
        self.rank = rank


class AttributionStateError(AttributionError):
    """Base class for attribution state errors."""

    pass


class SymptomRegistryError(AttributionError):
    """Base class for symptom registry errors."""

    pass


class UnknownSymptomError(SymptomRegistryError):
    """Raised when a symptom id is not known to the registry."""

    pass


class UninitializedSymptomRegistryError(SymptomRegistryError):
    """Raised when the symptom registry is accessed before setup_rules() has been called."""

    pass


class SymptomValidationError(SymptomRegistryError):
    """Raised when a symptom declaration cannot form a valid registry."""

    pass


class DuplicateSymptomDeclarationError(SymptomValidationError):
    """Raised when a symptom declaration is duplicated."""

    pass


class DuplicateSymptomMetadataError(SymptomValidationError):
    """Raised when symptom metadata is duplicated."""

    pass


class DuplicateAttributionError(AttributionStateError):
    """Raised when duplicate attribution is detected."""

    pass


class MismatchedNodeAttributionGroupUpdateError(AttributionStateError):
    """Raised when attempting to update node attribution group with mismatched hostnames."""

    pass


class ParseStateError(AttributionStateError):
    """Raised when a parse state method is called in an invalid order (e.g. finalize before any lines processed)."""

    pass


class MismatchedSourceSetValue(AttributionError):
    """Raised when trying to manipulate two sourcesets with different value."""

    def __init__(self, values: Any) -> None:
        self.values = values


class JobInfoError(AttributionError):
    """Base class for JobInfo errors."""

    pass


class JobInfoInvalidTimeLimitError(JobInfoError):
    """Error for when we are passed in an non-positive time_limit."""

    def __init__(self, time_limit: Any) -> None:
        self.time_limit = time_limit


class JobInfoIncompleteEndTimeError(JobInfoError):
    """Error for when we are passed in an a job that has neither interest_end_time nor job_end_time."""

    pass


class JobInfoDuplicateFieldError(JobInfoError):
    """Error for when multiple builder fields that map to the same underlying field are given."""

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name


class JobInfoUnevenNodeToRanksError(JobInfoError):
    """Error when participating nodes in node_to_ranks have rank lists of differing lengths.

    Empty rank lists (non-participating nodes) are allowed and excluded from this check.
    """

    def __init__(self, node_to_ranks: dict[str, list[int]]) -> None:
        self.node_to_ranks = node_to_ranks


class JobInfoNodeNameCollisionError(JobInfoError):
    """Error when node_to_ranks keys collide after case-normalization (e.g. 'NodeA' + 'nodea')."""

    def __init__(self, original_keys: list[str]) -> None:
        self.original_keys = original_keys


class DatabaseError(AttributionError):
    """Base error for DB-related issues"""


class UnexpectedEmptyQueryResultError(DatabaseError):
    """Error for when a query result is unexpectly empty."""

    def __init__(self, query: str, params: Any) -> None:
        self.query = query
        self.params = params


class RuleEngineError(AttributeError):
    """Blanket exception for rules engine items."""

    pass


class UninitializedRuleEngineError(RuleEngineError):
    """Raised when the rules engine is accessed before initialize() has been called."""

    pass


class UnknownRuleTypeError(RuleEngineError):
    """Error for when we have an unknown rule type"""

    pass


class RuleIngestionError(RuleEngineError):
    """Error for when we fail to ingest a rule."""

    pass


class DuplicateRuleError(RuleEngineError):
    """Error for when we have a repeated rule."""

    pass


class UndeclaredRuleError(RuleEngineError):
    """Raised when a rule name is used that was not registered or declared via declare_rule_names()."""

    pass


class UndeclaredEventTypeError(RuleEngineError):
    """Raised when an event type is used that was not registered or declared via declare_event_types()."""

    pass


class RuleApplicationError(RuleEngineError):
    """Base error for when a rule fails to be applied properly."""

    pass


class InvalidRuleParams(RuleEngineError, ValueError):
    """Raised when a rule's params payload fails validation against its ``params_model``.

    Inherits from ``ValueError`` so it integrates with Pydantic ``@field_validator``.
    """

    def __init__(self, rule_id: str, underlying: ValidationError):
        self.rule_id = rule_id
        self.underlying = underlying
        super().__init__(f"invalid params for rule {rule_id!r}: {underlying.errors()}")


class RuleAcceptsNoParamsError(RuleEngineError):
    """Raised when params are supplied for a rule that declares no ``params_model``."""

    def __init__(self, rule_id: str):
        self.rule_id = rule_id
        super().__init__(f"rule {rule_id!r} does not accept parameters")
