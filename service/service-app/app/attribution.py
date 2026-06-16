# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, Optional, Protocol, TypeAlias, Union

from attribution_lib.attr_types import NodeAttributionGroup
from attribution_lib.bundled_rules.common.megatron import MegatronStragglerItem
from attribution_lib.bundled_rules.common.nvrx import StragglerItem
from attribution_lib.bundled_rules.common.straggler_types import StragglerQualifier, StragglerScoreType
from attribution_lib.outcome import calculate_outcome
from attribution_lib.sources import (
    Source as LibSource,
    SourceSet,
    SymptomSet,
)
from pydantic import BaseModel, Field

from app.observation import ObservationId, ObservationSource

if TYPE_CHECKING:
    from app.attributor import MegatronStragglerParameters, NVRXStragglerParameters


# TODO: move under app/v1/.
class AttributionClassEnum(str, Enum):
    """The types of attributions/errors we can find. V1-only."""

    UNDEFINED = ""
    NO_ERROR = "no_error"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    USER = "user"
    ARTIFICIAL = "artificial"
    UNDETERMINED = "undetermined"

    @classmethod
    def from_lib_outcomes(cls, outcome: str) -> "AttributionClassEnum":
        match outcome:
            case "Incomplete Info":
                return cls.UNDETERMINED
            case "Fail Undetermined":
                return cls.UNDETERMINED
            case "Pass":
                return cls.NO_ERROR
            case "Fail InfraSW":
                return cls.SOFTWARE
            case "Fail User":
                return cls.USER
            case "Fail InfraHW":
                return cls.HARDWARE
            case "Artificial":
                return cls.ARTIFICIAL
        return cls.UNDEFINED


class RecommendationEnum(str, Enum):
    """The types of attributions/errors we can find."""

    UNDEFINED = ""
    DRAIN = "drain"
    NO_ERROR = "no_error"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"

    @classmethod
    def from_lib_recovery_actions(cls, recovery_actions: SymptomSet[str]) -> "RecommendationEnum":
        ## This obivously needs to be updated.
        return cls.UNKNOWN


Symptom: TypeAlias = str


class AttributionSource(BaseModel):
    """Description of provenance of attribution."""

    observation_id: ObservationId
    source: ObservationSource
    pos: Optional[int]
    line: Optional[str]
    rule: Optional[str]
    time: Optional[datetime]

    @classmethod
    def from_lib(cls, source: LibSource) -> "AttributionSource":
        return cls(
            observation_id=source.source_context.id,
            source=source.source_context.source,
            pos=source.line_num,
            line=source.datum,
            time=source.time,
            rule=source.rule,
        )


class SingleAttributionBase(BaseModel):
    """A base for the different types of attributions we can produce"""

    symptom: Symptom
    description: str
    attribution_class: AttributionClassEnum
    recommendation: RecommendationEnum
    sources: list[AttributionSource] = []


class SingleAttribution(SingleAttributionBase):
    type: Literal["SingleAttribution"] = Field(default="SingleAttribution")

    @classmethod
    def from_lib_source_set_bad(
        cls,
        source_set: SourceSet[str],
        description: str,
        attribution_class: AttributionClassEnum,
        recommendation: RecommendationEnum,
    ) -> Optional["SingleAttribution"]:
        if not source_set.has_bad():
            return None
        return SingleAttribution(
            symptom=source_set.value(),
            description=description,
            attribution_class=attribution_class,
            recommendation=recommendation,
            sources=[AttributionSource.from_lib(attr_source) for attr_source in source_set._bad],
        )

    @classmethod
    def from_lib_source_set_other(
        cls,
        source_set: SourceSet[str],
    ) -> Optional["SingleAttribution"]:
        if not source_set.has_interesting():
            return None
        return SingleAttribution(
            symptom=source_set.value(),
            description="",
            attribution_class=AttributionClassEnum.UNDETERMINED,
            recommendation=RecommendationEnum.AMBIGUOUS,
            sources=[AttributionSource.from_lib(attr_source) for attr_source in source_set._interesting],
        )


class WorkloadRecord(BaseModel):
    """A class for non-failure related observations"""

    type: Literal["WorkloadRecord"] = Field(default="WorkloadRecord")
    key: str
    val: str
    sources: list[AttributionSource]


class RankScore(NamedTuple):
    """A rank and its associated score."""

    rank: int
    score: float


class StragglerGroupProtocol(Protocol):
    """Protocol for straggler groups that can be unpacked."""

    ranks: list[Union["StragglerItem", "MegatronStragglerItem"]]
    qualifier: SourceSet[StragglerQualifier]
    score_type: SourceSet[StragglerScoreType]


class StragglerItemProtocol(Protocol):
    """Protocol for straggler items that can provide node information."""

    rank: int
    score: float

    def get_node(self) -> str:
        """Get the node identifier for this straggler item."""
        ...


class StragglerAttribution(SingleAttributionBase):
    """A node-specific attribution"""

    type: Literal["StragglerAttribution"] = Field(default="StragglerAttribution")
    symptom: Literal["Straggler"] = Field(default="Straggler")
    description: str = Field(default="Straggler Pattern")  # Dynamic description based on source
    attribution_class: AttributionClassEnum = AttributionClassEnum.UNDETERMINED
    recommendation: RecommendationEnum = RecommendationEnum.AMBIGUOUS
    rank_scores: list[RankScore] = []

    def add_straggler_item(self, item: StragglerItem | MegatronStragglerItem, additional_sources: list[LibSource]):
        self.sources += [AttributionSource.from_lib(src) for src in [item.source] + additional_sources]
        self.rank_scores.append(RankScore(rank=item.rank, score=item.score))

    @classmethod
    def unpack_lib_groups(
        cls, nvrx_straggler_parameters: "NVRXStragglerParameters", groups: list[StragglerGroupProtocol]
    ) -> dict[str, "StragglerAttribution"]:
        """Straggler attributions are stored in app_params.straggler_reports; here we produce a node-centric list of straggler attributions we have found."""
        res: dict[str, StragglerAttribution] = {}

        for group in groups:
            if group.score_type.value() not in nvrx_straggler_parameters.score_types:
                continue

            additional_sources = list(group.score_type) + list(group.qualifier)

            for _count, item in enumerate(group.ranks):
                if item.score > nvrx_straggler_parameters.straggler_threshold:
                    continue

                # Extract node identifier from straggler item
                node = item.get_node()
                if node is None:
                    # Skip items where we can't resolve the node
                    continue

                if node not in res:
                    res[node] = StragglerAttribution(description="NVRX Pattern")
                res[node].add_straggler_item(item, additional_sources)

        return res

    @classmethod
    def _should_process_megatron_group(
        cls, group: StragglerGroupProtocol, megatron_straggler_parameters: "MegatronStragglerParameters"
    ) -> bool:
        """Check if group should be processed for straggler detection."""
        if group.score_type.value() not in megatron_straggler_parameters.score_types:
            return False
        return group.qualifier.value() == StragglerQualifier.WORST

    @classmethod
    def _find_paired_top_group(
        cls,
        groups: list[StragglerGroupProtocol],
        index: int,
        megatron_straggler_parameters: "MegatronStragglerParameters",
    ) -> StragglerGroupProtocol | None:
        """Find the Top group paired with the current Bottom group.
        Megatron logs Bottom/Top groups as consecutive pairs.
        Validates that the next group has the BEST qualifier and matching score_type."""
        if index + 1 >= len(groups):
            return None

        next_group = groups[index + 1]
        if (
            next_group.score_type.value() in megatron_straggler_parameters.score_types
            and next_group.qualifier.value() == StragglerQualifier.BEST
        ):
            return next_group
        return None

    @classmethod
    def _calculate_event_threshold(
        cls,
        top_group: StragglerGroupProtocol | None,
        megatron_straggler_parameters: "MegatronStragglerParameters",
    ) -> float:
        """Calculate per-event threshold from paired Top group."""
        if top_group is None or not megatron_straggler_parameters.infer_threshold_from_data:
            return megatron_straggler_parameters.straggler_threshold

        event_top_scores = [item.score for item in top_group.ranks]
        if event_top_scores:
            event_max_throughput = max(event_top_scores)
            return event_max_throughput * megatron_straggler_parameters.straggler_threshold

        return megatron_straggler_parameters.straggler_threshold

    @classmethod
    def _process_straggler_items(
        cls,
        group: StragglerGroupProtocol,
        event_threshold: float,
        res: dict[str, "StragglerAttribution"],
    ) -> None:
        """Process straggler items and add to result dict."""
        additional_sources = list(group.score_type) + list(group.qualifier)

        for item in group.ranks:
            if item.score >= event_threshold:
                continue

            node = item.get_node()
            if node is None:
                continue

            if node not in res:
                res[node] = StragglerAttribution(description="Megatron Pattern")
            res[node].add_straggler_item(item, additional_sources)

    @classmethod
    def unpack_megatron_lib_groups(
        cls,
        megatron_straggler_parameters: "MegatronStragglerParameters",
        groups: list[StragglerGroupProtocol],
        top_scores: list[float],
    ) -> dict[str, "StragglerAttribution"]:
        """Megatron straggler attributions are stored in app_params.megatron_straggler_data.reports; here we produce a node-centric list of straggler attributions we have found."""
        res: dict[str, StragglerAttribution] = {}

        i = 0
        while i < len(groups):
            group = groups[i]

            # Skip non-Bottom groups
            if not cls._should_process_megatron_group(group, megatron_straggler_parameters):
                i += 1
                continue

            # Find the paired Top group (next in sequence)
            top_group = cls._find_paired_top_group(groups, i, megatron_straggler_parameters)

            # Skip incomplete events when threshold inference is enabled
            if top_group is None and megatron_straggler_parameters.infer_threshold_from_data:
                i += 1
                continue

            # Calculate per-event threshold from Top group
            event_threshold = cls._calculate_event_threshold(top_group, megatron_straggler_parameters)
            # Flag stragglers below threshold
            cls._process_straggler_items(group, event_threshold, res)

            i += 1

        return res


SingleAttributionTypes = Annotated[
    Union[SingleAttribution, StragglerAttribution],
    Field(discriminator="type"),
]


class NodeAttribution(BaseModel):
    """A node-specific attribution"""

    type: Literal["NodeAttribution"] = Field(default="NodeAttribution")
    attributions: list[SingleAttributionTypes]
    node: str

    @classmethod
    def from_lib_node_group(
        cls,
        node_attr: NodeAttributionGroup,
    ) -> "NodeAttribution":
        attributions = []
        for symptom, source_set in node_attr.attributions._data.items():
            if source_set.has_bad():
                tmp: SymptomSet[Symptom] = SymptomSet()
                tmp.add(source_set)
                outcome = calculate_outcome(tmp)
                att_class = AttributionClassEnum.from_lib_outcomes(outcome.value().outcome)
                if (
                    att := SingleAttribution.from_lib_source_set_bad(
                        source_set,
                        outcome.value().hw_cause,
                        att_class,
                        RecommendationEnum.UNKNOWN,
                    )
                ) is not None:
                    attributions.append(att)
            elif source_set.has_interesting():
                if (att := SingleAttribution.from_lib_source_set_other(source_set)) is not None:
                    attributions.append(att)
            else:
                continue
        assert len(attributions) > 0
        return cls(
            node=node_attr.hostname,
            attributions=attributions,
        )


class StorageAttribution(BaseModel):
    """Storage-specific attribution; details TBD"""

    type: Literal["StorageAttribution"] = Field(default="StorageAttribution")
    attributions: list[SingleAttributionTypes]


class SwitchAttribution(BaseModel):
    """Switch-specific attribution; details TBD"""

    type: Literal["SwitchAttribution"] = Field(default="SwitchAttribution")
    attributions: list[SingleAttributionTypes]


class GlobalAttribution(BaseModel):
    """Non-specific attribution (i.e. job-level)"""

    type: Literal["GlobalAttribution"] = Field(default="GlobalAttribution")
    attributions: list[SingleAttributionTypes]


AttributionTypes = Annotated[
    Union[WorkloadRecord, NodeAttribution, StorageAttribution, SwitchAttribution, GlobalAttribution],
    Field(discriminator="type"),
]
