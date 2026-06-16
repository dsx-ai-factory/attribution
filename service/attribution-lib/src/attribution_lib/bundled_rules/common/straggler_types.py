# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Shared straggler type definitions used by Megatron straggler module.

from dataclasses import dataclass, field
from enum import Enum

from attribution_lib.sources import Source


class UnknownStragglerQualifier(Exception):
    """Exception to represent unknown qualifier. Shouldn't be raised."""

    def __init__(self, string: str):
        self.string = string


class StragglerQualifier(int, Enum):
    """Type of qualifier (best/worst) found in Straggler Detector supports."""

    WORST = 0
    BEST = 1

    @classmethod
    def from_str(cls, string: str) -> "StragglerQualifier":
        match string:
            case "Worst":
                return cls.WORST
            case "Best":
                return cls.BEST
            case _:
                raise UnknownStragglerQualifier(string)


class UnknownStragglerScoreType(Exception):
    """Exception to represent unknown header parameters. Shouldn't be raised."""

    def __init__(self, string: str):
        self.string = string


class StragglerScoreType(int, Enum):
    """Type of scoring Straggler Detector supports."""

    RELATIVE = 0
    INDIVIDUAL = 1

    @classmethod
    def from_str(cls, string: str) -> "StragglerScoreType":
        match string:
            case "relative":
                return cls.RELATIVE
            case "individual":
                return cls.INDIVIDUAL
            case _:
                raise UnknownStragglerScoreType(string)


@dataclass(frozen=True, slots=True)
class StragglerRankReport:
    """A single rank flagged as a straggler with its score."""

    rank: int
    score: float


@dataclass(slots=True)
class StragglerNodeReport:
    """Per-node straggler report emitted by NVRX / Megatron straggler collectors.

    ``rank_scores`` lists each flagged rank/score pair on this node;
    ``sources`` carries the contributing per-item sources plus each containing
    group's score_type and qualifier sources, so downstream callers can render
    full provenance without re-running the collection state machine.
    """

    pattern: str
    rank_scores: list[StragglerRankReport] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
