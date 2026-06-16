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
from typing import NamedTuple, Optional, TypeVar
from zoneinfo import ZoneInfo

from attribution_lib import ufmlog_common, utils
from attribution_lib.exceptions import (
    InvalidRankError,
    JobInfoDuplicateFieldError,
    JobInfoEmptyNodesError,
    JobInfoIncompleteEndTimeError,
    JobInfoInvalidTimeLimitError,
    JobInfoNodelistAfterNodesError,
    JobInfoNodeNameCollisionError,
    JobInfoNodesAfterNodelistError,
    JobInfoNodeToRanksConflictError,
    JobInfoNoNodeInfoError,
    JobInfoUnevenNodeToRanksError,
    UnidentifiableNodeError,
    UnknownNodeError,
    UnknownNodeExtractionSourceError,
)
from attribution_lib.hostlist import ParseError, parse_hostlist
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


class JobInfo(NamedTuple):
    """Container class for parameters extracted from the job system, intended to be given to attribution engines (and not modified)"""

    id: str
    type: str
    state: str
    cluster: str
    nodes: list[str]
    slurm_node_infos: dict[str, dict[str, str]]
    job_start_time: datetime
    interest_start_time: datetime
    interest_end_time: datetime
    srun_cmd: Optional[str]
    exit_code_signal: Optional[int]
    cancelled_by: Optional[str]
    nodelist: Optional[str]
    failed_node: Optional[str]
    job_end_time: Optional[datetime]
    time_limit: Optional[int]
    last_tflops_time: Optional[datetime]
    job_script_content: Optional[str]
    num_gpus: int
    RANKS_PER_NODE: float
    framework: str
    timezone: ZoneInfo

    l_context: ObservationContext

    node_to_ranks: Optional[Mapping[str, list[int]]] = None

    def context(self) -> ObservationContext:
        return self.l_context

    def source(self, datum: Optional[str], rule: str) -> Source:
        return Source(self.context(), None, datum or "", None, AllJobResources, rule)

    def source_value(self, value: T, datum: Optional[str], rule: str) -> SourceSet[T]:
        return SourceSet(value).add(self.source(datum, rule))

    def get_name_from_rank(self, rank: int) -> str:
        """Look up rank in nodes (if it exists) and return name. Otherwise, make one up."""
        if self.node_to_ranks is not None:
            for node, ranks in self.node_to_ranks.items():
                if rank in ranks:
                    return node
            raise InvalidRankError(rank)
        try:
            return self.nodes[int(rank // self.RANKS_PER_NODE)]
        except IndexError:
            return f"node-{int(rank // self.RANKS_PER_NODE)}"

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

    def find_node(self, node: str, rank: int) -> Optional[str]:
        """If node is in our job info, return it. Otherwise, try and
        look up rank and return it. Return None if we can't find
        anything."""
        try:
            return self.verify_node(node)
        except UnknownNodeError:
            pass

        try:
            return self.verify_rank(rank)
        except InvalidRankError:
            return None

    def verify_rank(self, rank: int) -> str:
        """Check that node is part of this job.
        Raise InvalidRankException if we fail."""

        if self.node_to_ranks is not None:
            return self.get_name_from_rank(rank)
        if rank >= int(len(self.nodes) * self.RANKS_PER_NODE):
            # this rank is not in the list of nodes
            raise InvalidRankError(rank)
        return self.get_name_from_rank(rank)

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
            return self.verify_rank(actual_rank)
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
        dt_start_time = datetime.fromtimestamp(float(slurmdb_job["start_time"]), tz=timezone)
        dt_end_time = datetime.fromtimestamp(float(slurmdb_job["end_time"]), tz=timezone)

        builder = (
            JobInfoBuilder(
                id=slurmdb_job["id"],
                job_start_time=dt_start_time,
                cluster=cluster,
                state=slurmdb_job["state"],
                RANKS_PER_NODE=int(RANKS_PER_NODE),
                framework=slurmdb_job["framework"],
                type=slurmdb_job["type"],
                timezone=timezone,
                source_name="slurmdb",
            )
            .num_gpus(int(slurmdb_job.get("gpus", 0)))
            .srun_cmd(slurmdb_job["srun_cmd"])
            .exit_code_signal(int(slurmdb_job["exit_code_signal"]))
            .cancelled_by(slurmdb_job["cancelled_by"])
            .nodelist(slurmdb_job["nodelist"])
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
    _type: str
    _cancelled_by: Optional[str] = None
    _nodelist: Optional[str] = None
    _state: str
    _failed_node: Optional[str] = None
    _cluster: str
    _nodes: list[str] = list()
    _slurm_node_infos: dict[str, dict[str, str]] = dict()
    _job_start_time: datetime
    _job_end_time: Optional[datetime] = None
    _interest_start_time: Optional[datetime] = None
    _interest_end_time: Optional[datetime] = None
    _time_limit: Optional[int] = None
    _last_tflops_time: Optional[datetime] = None
    _num_gpus: Optional[int] = None
    _job_script_content: Optional[str] = None
    _RANKS_PER_NODE: Optional[float] = None
    _node_to_ranks: Optional[Mapping[str, list[int]]] = None
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
        RANKS_PER_NODE: Optional[float] = None,
    ):
        self._id = id
        self._type = type
        self._state = state
        self._cluster = cluster
        self._job_start_time = job_start_time
        self._RANKS_PER_NODE = RANKS_PER_NODE
        self._framework = framework
        self._source_name = source_name
        self._timezone = timezone

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

    def exit_code_signal(self, in_exit_code_signal: int) -> "JobInfoBuilder":
        if self._exit_code_signal is not None:
            raise JobInfoDuplicateFieldError("exit_code_signal")
        self._exit_code_signal = in_exit_code_signal
        return self

    def cancelled_by(self, in_cancelled_by: str) -> "JobInfoBuilder":
        if self._cancelled_by is not None:
            raise JobInfoDuplicateFieldError("cancelled_by")
        if in_cancelled_by != "":  # Normalize empty cancelled_by to None if empty
            self._cancelled_by = in_cancelled_by
        return self

    def nodelist(self, in_nodelist: str) -> "JobInfoBuilder":
        if self._nodelist is not None:
            raise JobInfoDuplicateFieldError("nodelist")
        if self._node_to_ranks is not None:
            raise JobInfoNodeToRanksConflictError("nodelist")
        if len(self._nodes) > 0:
            raise JobInfoNodelistAfterNodesError(in_nodelist)
        self._nodelist = in_nodelist
        return self

    def failed_node(self, in_failed_node: Optional[str]) -> "JobInfoBuilder":
        if self._failed_node is not None:
            raise JobInfoDuplicateFieldError("failed_node")
        self._failed_node = in_failed_node
        return self

    def nodes(self, in_nodes: list[str]) -> "JobInfoBuilder":
        if len(self._nodes) > 0:
            raise JobInfoDuplicateFieldError("nodes")
        if len(in_nodes) == 0:
            raise JobInfoEmptyNodesError
        if self._nodelist is not None:
            raise JobInfoNodesAfterNodelistError(in_nodes)
        if self._node_to_ranks is not None:
            raise JobInfoNodeToRanksConflictError("nodes")
        self._nodes = in_nodes
        return self

    def node_to_ranks(self, in_mapping: Mapping[str, list[int]]) -> "JobInfoBuilder":
        """Set an explicit node-to-ranks mapping; derives nodes, RANKS_PER_NODE, num_gpus at build time.

        Mutually exclusive with ``.nodes()``, ``.nodelist()``, and the ``RANKS_PER_NODE`` args.
        """
        if self._node_to_ranks is not None:
            raise JobInfoDuplicateFieldError("node_to_ranks")
        if self._RANKS_PER_NODE is not None:
            raise JobInfoNodeToRanksConflictError("RANKS_PER_NODE")
        if len(self._nodes) > 0:
            raise JobInfoNodeToRanksConflictError("nodes")
        if self._nodelist is not None:
            raise JobInfoNodeToRanksConflictError("nodelist")
        if len(in_mapping) == 0:
            raise JobInfoNoNodeInfoError
        copied = {k.lower(): list(v) for k, v in in_mapping.items()}
        if len(copied) < len(in_mapping):
            raise JobInfoNodeNameCollisionError(list(in_mapping.keys()))
        lengths = {len(ranks) for ranks in copied.values()}
        if len(lengths) > 1:
            raise JobInfoUnevenNodeToRanksError(copied)
        self._node_to_ranks = copied
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

        if self._node_to_ranks is not None:
            derived_nodes = sorted(self._node_to_ranks.keys())
            per_node_len = len(next(iter(self._node_to_ranks.values())))
            self._nodes = derived_nodes
            self._RANKS_PER_NODE = float(per_node_len)
            self._nodelist = ",".join(derived_nodes)
            if self._num_gpus is None:
                self._num_gpus = sum(len(ranks) for ranks in self._node_to_ranks.values())
        else:
            if self._nodelist:
                try:
                    self._nodes = parse_hostlist(self._nodelist)
                except ParseError:
                    pass

            if self._nodelist is None and len(self._nodes) == 0:
                raise JobInfoNoNodeInfoError

            if self._RANKS_PER_NODE is None:
                raise JobInfoNoNodeInfoError

            if self._num_gpus is None:
                self._num_gpus = int(len(self._nodes) * self._RANKS_PER_NODE)

        assert self._num_gpus is not None
        return JobInfo(
            id=self._id,
            srun_cmd=self._srun_cmd,
            exit_code_signal=self._exit_code_signal,
            type=self._type,
            cancelled_by=self._cancelled_by,
            nodelist=self._nodelist,
            state=self._state,
            failed_node=self._failed_node,
            cluster=self._cluster,
            nodes=[n.lower() for n in self._nodes],
            slurm_node_infos=self._slurm_node_infos,
            job_start_time=self._job_start_time,
            job_end_time=self._job_end_time,
            interest_start_time=self._interest_start_time,
            interest_end_time=self._interest_end_time,
            time_limit=self._time_limit,
            timezone=self._timezone,
            last_tflops_time=self._last_tflops_time,
            num_gpus=self._num_gpus,
            RANKS_PER_NODE=self._RANKS_PER_NODE,
            framework=self._framework,
            l_context=context,
            job_script_content=self._job_script_content,
            node_to_ranks=self._node_to_ranks,
        )
