# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import NamedTuple, Optional, TypeVar
from zoneinfo import ZoneInfo

from attribution_lib import ufmlog_common, utils
from attribution_lib.exceptions import (
    InvalidRankError,
    JobInfoDuplicateFieldError,
    JobInfoIncompleteEndTimeError,
    JobInfoInvalidTimeLimitError,
    JobInfoNodeNameCollisionError,
    JobInfoUnevenNodeToRanksError,
    UnidentifiableNodeError,
    UnknownNodeError,
    UnknownNodeExtractionSourceError,
)
from attribution_lib.hostlist import parse_hostlist
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
    Id,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.sources import Source, SourceSet

T = TypeVar("T")

NVRX_TAG = "nvrx"


def _coerce_optional_int(raw: "str | int | None") -> Optional[int]:
    """Parse a SlurmDB integer field, treating None and empty string as absent.

    SlurmDB records can carry ``""`` for unset fields; ``int("")`` would raise, so
    empty/absent inputs return None while ``0`` and numeric values are preserved.
    """
    if raw is None or raw == "":
        return None
    return int(raw)


def _coerce_tags(raw: "str | Iterable[str] | None") -> frozenset[str]:
    """Normalize a SlurmDB ``tags`` value into a set of whole tags.

    Accepts the expected list/tuple/set of tag strings, tolerates a single bare
    string (wrapping it rather than letting ``frozenset("nvrx")`` split into
    characters), and treats absent/empty input as no tags.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset((raw,)) if raw else frozenset()
    return frozenset(raw)


class JobInfo(NamedTuple):
    """Container class for parameters extracted from the job system, intended to be given to attribution engines (and not modified)"""

    id: str
    type: str
    state: str
    cluster: str
    nodes: list[str]
    node_from_rank: dict[int, str]
    slurm_node_infos: dict[str, dict[str, str]]
    job_start_time: datetime
    interest_start_time: datetime
    interest_end_time: datetime
    srun_cmd: Optional[str]
    exit_code_signal: Optional[int]
    # Job-level return code (0 = success), and free-form tags describing the job
    # (e.g. "nvrx" for fault-tolerance-managed jobs). Used by the NVRx exit-code
    # outcome rule; both default to unknown/empty so the rule stays dormant until
    # ingestion populates them. Tags let later stages key off job traits without a
    # dedicated field per framework/tool.
    exit_code: Optional[int]
    tags: frozenset[str]
    cancelled_by: Optional[str]
    failed_node: Optional[str]
    job_end_time: Optional[datetime]
    time_limit: Optional[int]
    last_tflops_time: Optional[datetime]
    job_script_content: Optional[str]
    num_gpus: int
    # Derived from node_to_ranks at build() time. Stored here so clients can read ranks-per-node in constant time .
    RANKS_PER_NODE: float
    framework: str
    timezone: ZoneInfo

    l_context: ObservationContext

    def context(self) -> ObservationContext:
        return self.l_context

    def exceeded_time_limit(self) -> bool:
        """Return whether the parent job interval exceeds its finite limit.

        The interest interval may represent one selected Slurm step and is not
        the interval governed by the job allocation's time limit.
        """
        if self.time_limit is None or self.job_end_time is None:
            return False
        return (self.job_end_time - self.job_start_time) > timedelta(minutes=self.time_limit)

    def source(self, datum: Optional[str], rule: str) -> Source:
        return Source(self.context(), None, datum or "", None, AllJobResources, rule)

    def source_value(self, value: T, datum: Optional[str], rule: str) -> SourceSet[T]:
        return SourceSet(value).add(self.source(datum, rule))

    def get_name_from_rank(self, rank: int) -> str:
        """Return the node name for a given rank. Raise InvalidRankError if rank does not exist."""
        if not self.rank_exists(rank):
            raise InvalidRankError(rank)
        return self.node_from_rank[rank]

    def verify_node(self, node: str) -> str:
        """Check that node is part of this job.

        Args:
            node: The node name to verify.

        Returns:
            The normalized node name (lowercase).

        Raises:
            UnknownNodeError: If the node is not in the job.
        """
        normalized_node = node.lower()
        if len(self.nodes) > 0 and normalized_node not in self.nodes:
            raise UnknownNodeError(node)
        return normalized_node

    def rank_exists(self, rank: int) -> bool:
        """Check if a rank exists in the job."""
        return rank in self.node_from_rank

    def ufmlog_extract_node(self, line: str) -> str:
        """Try to get a valid node out of this ufmlog line, or raise an error."""
        node = ufmlog_common.extract_ufmlog_line_node(line)
        if (actual_node := node) is not None:
            return self.verify_node(actual_node)
        raise UnidentifiableNodeError(line)

    def applog_extract_node(self, line: str) -> str:
        """Try to get a valid node out of this applog line, or raise an error."""
        node, rank = utils.applog_extract_node_rank(line)
        if (actual_node := node) is not None:
            return self.verify_node(actual_node)
        if (actual_rank := rank) is not None:
            return self.get_name_from_rank(actual_rank)
        raise UnidentifiableNodeError(line)

    def syslog_extract_node(self, line: str) -> str:
        """Try to get a valid node out of this syslog line, or raise an error."""
        node = utils.syslog_extract_node(line)
        if (actual_node := node) is not None:
            return self.verify_node(actual_node)
        raise UnidentifiableNodeError(line)

    def extract_node(self, line: str, source: ObservationSource) -> str:
        match source:
            case ObservationSource.APPLOG:
                return self.applog_extract_node(line)
            case ObservationSource.SYSLOG | ObservationSource.HEALTH_CHECK:
                return self.syslog_extract_node(line)
            case ObservationSource.UFMLOG:
                return self.ufmlog_extract_node(line)
            case _:
                raise UnknownNodeExtractionSourceError(source)

    @classmethod
    def from_slurmdb(
        cls,
        *,
        slurmdb_job: dict[str, str],
        cluster: str,
        RANKS_PER_NODE: int,
        slurm_node_infos: dict[str, dict[str, str]] = {},
        last_tflops_time: Optional[datetime] = None,
        timezone: ZoneInfo,
    ) -> "JobInfo":
        """Build a JobInfo from a slurmdb job record.

        Raises ParseError if the nodelist cannot be parsed.
        """
        dt_start_time = datetime.fromtimestamp(float(slurmdb_job["start_time"]), tz=timezone)
        dt_end_time = datetime.fromtimestamp(float(slurmdb_job["end_time"]), tz=timezone)

        nodelist_str = slurmdb_job["nodelist"]
        if nodelist_str == "None assigned":
            node_to_ranks: dict[str, list[int]] = {}
        else:
            parsed_nodes = parse_hostlist(nodelist_str)
            node_to_ranks = JobInfoBuilder.uniform_node_to_ranks(parsed_nodes, RANKS_PER_NODE)

        builder = (
            JobInfoBuilder(
                id=slurmdb_job["id"],
                job_start_time=dt_start_time,
                cluster=cluster,
                state=slurmdb_job["state"],
                framework=slurmdb_job["framework"],
                type=slurmdb_job["type"],
                timezone=timezone,
                source_name="slurmdb",
                node_to_ranks=node_to_ranks,
            )
            .num_gpus(int(slurmdb_job.get("gpus", 0)))
            .srun_cmd(slurmdb_job["srun_cmd"])
            .exit_code_signal(_coerce_optional_int(slurmdb_job.get("exit_code_signal")))
            .exit_code(_coerce_optional_int(slurmdb_job.get("exit_code")))
            .tags(_coerce_tags(slurmdb_job.get("tags")))
            .cancelled_by(slurmdb_job["cancelled_by"])
            .failed_node(slurmdb_job.get("failed_node", None))
            .slurm_node_infos(slurm_node_infos)
            .job_end_time(dt_end_time)
            .time_limit(int(slurmdb_job["time_limit"]) if slurmdb_job.get("time_limit") else None)
        )
        if last_tflops_time is not None:
            builder.last_tflops_time(last_tflops_time)

        return builder.build()


class JobInfoBuilder:
    _id: str
    _srun_cmd: Optional[str] = None
    _exit_code_signal: Optional[int] = None
    _exit_code: Optional[int] = None
    _tags: Optional[frozenset[str]] = None
    _type: str
    _cancelled_by: Optional[str] = None
    _state: str
    _failed_node: Optional[str] = None
    _cluster: str
    _slurm_node_infos: dict[str, dict[str, str]] = dict()
    _job_start_time: datetime
    _job_end_time: Optional[datetime] = None
    _interest_start_time: Optional[datetime] = None
    _interest_end_time: Optional[datetime] = None
    _time_limit: Optional[int] = None
    _last_tflops_time: Optional[datetime] = None
    _num_gpus: Optional[int] = None
    _job_script_content: Optional[str] = None
    _node_to_ranks: dict[str, list[int]]
    _framework: str
    _source_name: str
    _timezone: ZoneInfo

    def __init__(
        self,
        *,
        id: str,
        type: str,
        state: str,
        cluster: str,
        job_start_time: datetime,
        framework: str,
        source_name: str,
        timezone: ZoneInfo,
        node_to_ranks: Mapping[str, list[int]],
    ):
        """Construct a builder. ``node_to_ranks`` is the explicit, required node-info input.

        Keys (node names) are lowercased. Semantics of the mapping:
          - ``{}`` (empty): the job never received an allocation (never ran).
          - a node mapping to a non-empty rank list: a participating node carrying those ranks.
            All such lists must share the same width (raises JobInfoUnevenNodeToRanksError otherwise).
          - a node mapping to ``[]``: a member of the job that ran no collective ranks.
            It stays attributable by hostname but has no ranks.
        """
        self._id = id
        self._type = type
        self._state = state
        self._cluster = cluster
        self._job_start_time = job_start_time
        self._framework = framework
        self._source_name = source_name
        self._timezone = timezone

        copied = {k.lower(): list(v) for k, v in node_to_ranks.items()}

        if len(copied) < len(node_to_ranks):
            raise JobInfoNodeNameCollisionError(list[str](node_to_ranks.keys()))

        rank_widths = {len(ranks) for ranks in copied.values() if len(ranks) > 0}
        if len(rank_widths) > 1:
            raise JobInfoUnevenNodeToRanksError(copied)

        self._node_to_ranks = copied

    @staticmethod
    def uniform_node_to_ranks(
        nodes: Sequence[str], ranks_per_node: int, *, start_rank: int = 0
    ) -> dict[str, list[int]]:
        """Build a uniform node_to_ranks mapping.

        Args:
            nodes: The node names to assign ranks to.
            ranks_per_node: The number of ranks to assign to each node.
            start_rank: The first global rank assigned to the first node. This is useful when the
                supplied nodes are one component of a larger distributed job.

        Returns:
            A dictionary mapping node names to lists of ranks.

        Examples:
            >>> JobInfoBuilder.uniform_node_to_ranks(["node-a", "node-b"], 4)
            {'node-a': [0, 1, 2, 3], 'node-b': [4, 5, 6, 7]}
            >>> JobInfoBuilder.uniform_node_to_ranks(["node-a"], 1)
            {'node-a': [0]}
            >>> JobInfoBuilder.uniform_node_to_ranks(["node-a", "node-b"], 4, start_rank=8)
            {'node-a': [8, 9, 10, 11], 'node-b': [12, 13, 14, 15]}
            >>> JobInfoBuilder.uniform_node_to_ranks([], 8)
            {}
        """
        return {
            n: list(
                range(
                    start_rank + i * ranks_per_node,
                    start_rank + (i + 1) * ranks_per_node,
                )
            )
            for i, n in enumerate(nodes)
        }

    def srun_cmd(self, in_srun_cmd: str) -> "JobInfoBuilder":
        if self._srun_cmd is not None:
            raise JobInfoDuplicateFieldError("srun_cmd")
        if in_srun_cmd != "":  # Normalize empty strings to None
            self._srun_cmd = in_srun_cmd
        return self

    def num_gpus(self, in_num_gpus: int) -> "JobInfoBuilder":
        if self._num_gpus is not None:
            raise JobInfoDuplicateFieldError("num_gpus")
        self._num_gpus = in_num_gpus
        return self

    def exit_code_signal(self, in_exit_code_signal: Optional[int]) -> "JobInfoBuilder":
        if self._exit_code_signal is not None:
            raise JobInfoDuplicateFieldError("exit_code_signal")
        self._exit_code_signal = in_exit_code_signal
        return self

    def exit_code(self, in_exit_code: Optional[int]) -> "JobInfoBuilder":
        if self._exit_code is not None:
            raise JobInfoDuplicateFieldError("exit_code")
        self._exit_code = in_exit_code
        return self

    def tags(self, in_tags: frozenset[str]) -> "JobInfoBuilder":
        if self._tags is not None:
            raise JobInfoDuplicateFieldError("tags")
        self._tags = in_tags
        return self

    def cancelled_by(self, in_cancelled_by: str) -> "JobInfoBuilder":
        if self._cancelled_by is not None:
            raise JobInfoDuplicateFieldError("cancelled_by")
        if in_cancelled_by != "":  # Normalize empty cancelled_by to None if empty
            self._cancelled_by = in_cancelled_by
        return self

    def failed_node(self, in_failed_node: Optional[str]) -> "JobInfoBuilder":
        if self._failed_node is not None:
            raise JobInfoDuplicateFieldError("failed_node")
        self._failed_node = in_failed_node
        return self

    def slurm_node_infos(self, in_slurm_node_infos: dict[str, dict[str, str]]) -> "JobInfoBuilder":
        if len(self._slurm_node_infos) > 0:
            raise JobInfoDuplicateFieldError("slurm_node_infos")
        self._slurm_node_infos = in_slurm_node_infos
        return self

    def job_end_time(self, in_job_end_time: datetime) -> "JobInfoBuilder":
        if self._job_end_time is not None:
            raise JobInfoDuplicateFieldError("job_end_time")
        self._job_end_time = utils.check_datetime(in_job_end_time)
        return self

    def interest_start_time(self, in_interest_start_time: datetime) -> "JobInfoBuilder":
        if self._interest_start_time is not None:
            raise JobInfoDuplicateFieldError("interest_start_time")
        self._interest_start_time = utils.check_datetime(in_interest_start_time)
        return self

    def interest_end_time(self, in_interest_end_time: datetime) -> "JobInfoBuilder":
        if self._interest_end_time is not None:
            raise JobInfoDuplicateFieldError("interest_end_time")
        self._interest_end_time = utils.check_datetime(in_interest_end_time)
        return self

    def time_limit(self, in_time_limit: Optional[int]) -> "JobInfoBuilder":
        if self._time_limit is not None:
            raise JobInfoDuplicateFieldError("time_limit")
        if in_time_limit is not None and in_time_limit <= 0:
            raise JobInfoInvalidTimeLimitError(in_time_limit)
        self._time_limit = in_time_limit
        return self

    def last_tflops_time(self, in_last_tflops_time: datetime) -> "JobInfoBuilder":
        if self._last_tflops_time is not None:
            raise JobInfoDuplicateFieldError("last_tflops_time")
        self._last_tflops_time = utils.check_datetime(in_last_tflops_time)
        return self

    def job_script_content(self, in_job_script_content: str) -> "JobInfoBuilder":
        if self._job_script_content is not None:
            raise JobInfoDuplicateFieldError("job_script_content")
        self._job_script_content = in_job_script_content
        return self

    def build(self) -> JobInfo:
        if self._interest_start_time is None:
            self._interest_start_time = self._job_start_time
        if self._interest_end_time is None:
            self._interest_end_time = self._job_end_time
        if self._interest_end_time is None:
            raise JobInfoIncompleteEndTimeError

        context = ObservationContext(
            TimeInterval(self._interest_start_time, self._interest_end_time),
            AllJobResources,
            ObservationSource.JOB_INFO,
            ObservationFormat.UNDEFINED,
            self._timezone,
            Id(hash(self._source_name)),  # placeholder id
        )

        # TODO check slurm_node_infos, if given

        derived_nodes = list[str](self._node_to_ranks.keys())
        derived_rpn = 0.0
        for ranks in self._node_to_ranks.values():
            if len(ranks) > 0:
                derived_rpn = len(ranks)
                break
        derived_num_gpus = (
            self._num_gpus if self._num_gpus is not None else sum(len(ranks) for ranks in self._node_to_ranks.values())
        )

        node_from_rank: dict[int, str] = {r: n for n, ranks in self._node_to_ranks.items() for r in ranks}

        return JobInfo(
            id=self._id,
            srun_cmd=self._srun_cmd,
            exit_code_signal=self._exit_code_signal,
            exit_code=self._exit_code,
            tags=self._tags if self._tags is not None else frozenset(),
            type=self._type,
            cancelled_by=self._cancelled_by,
            state=self._state,
            failed_node=self._failed_node,
            cluster=self._cluster,
            nodes=derived_nodes,
            slurm_node_infos=self._slurm_node_infos,
            job_start_time=self._job_start_time,
            job_end_time=self._job_end_time,
            interest_start_time=self._interest_start_time,
            interest_end_time=self._interest_end_time,
            time_limit=self._time_limit,
            timezone=self._timezone,
            last_tflops_time=self._last_tflops_time,
            num_gpus=derived_num_gpus,
            RANKS_PER_NODE=derived_rpn,
            framework=self._framework,
            l_context=context,
            job_script_content=self._job_script_content,
            node_from_rank=node_from_rank,
        )
