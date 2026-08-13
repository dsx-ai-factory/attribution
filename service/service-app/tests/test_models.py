# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from attribution_lib.core import AttributionResult, CalcBadNodesAttributionResult
from attribution_lib.observation import Context as FullObservationContext
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format,
    Id,
    TimeInterval,
)
from attribution_lib.outcome import OutcomeAttribution
from attribution_lib.resources import AllJobResources
from attribution_lib.sources import (
    Source as LibSource,
    SourceSet,
)

from app.observation import ObservationSource
from app.v2.models import (
    AttributionResponseV2,
    ResourcelessSource,
)

_UTC = ZoneInfo("UTC")
_T = datetime(2026, 3, 22, 14, 3, 12, tzinfo=_UTC)


def _lib_source(rule: str, obs_id: int = 1, line_num: int = 0, datum: str = "raw") -> LibSource:
    ti = TimeInterval(_T, _T + timedelta(minutes=5))
    inbound = InboundContext(ti, AllJobResources, ObservationSource.APPLOG, Format.RAW_LOG_LINES, _UTC)
    ctx = FullObservationContext(**vars(inbound), id=Id(obs_id))
    return LibSource(
        source_context=ctx,
        line_num=line_num,
        datum=datum,
        time=_T,
        resources=AllJobResources,
        rule=rule,
    )


def _empty_result() -> AttributionResult:
    return AttributionResult(event_stream=MagicMock(), bad_nodes_attr=CalcBadNodesAttributionResult())


def _add_node_symptom(result: AttributionResult, node: str, symptom: str, *sources: LibSource) -> None:
    """Use JobNodeAttributions.add() - the public, lib-idiomatic entry point.
    It also runs _normalize_node (lowercases the hostname), so tests see the same
    casing behavior that the real lib produces end-to-end."""
    ss = SourceSet(symptom)
    for s in sources:
        ss.add(s)
    result.bad_nodes_attr.my_bad_nodes.add([node], ss)


def _add_job_symptom(result: AttributionResult, symptom: str, *sources: LibSource) -> None:
    ss = SourceSet(symptom)
    for s in sources:
        ss.add(s)
    result.symptoms.add(ss)


def _set_outcome(result: AttributionResult, outcome: OutcomeAttribution, *driver_sources: LibSource) -> None:
    ss = SourceSet(outcome)
    for s in driver_sources:
        ss.add(s)
    result.outcome_attr = ss


class TestV2Models:
    def test_empty_result_produces_incomplete_info_verdict(self):
        """Empty results produce the lib default Incomplete Info outcome with no root symptom, detail, or symptoms."""
        result = _empty_result()

        response = AttributionResponseV2.from_lib_attribution(set(), result)

        assert response.verdict.attribution_outcome == "Incomplete Info"
        assert response.verdict.root_symptom is None
        assert response.verdict.detail == "Not Applicable"
        assert response.symptoms == {}
        assert response.observation_context == []

    def test_root_symptom_surfaces_outcome_attribution_record(self):
        """When the lib sets root_cause_symptom, V2 surfaces it and its doc slug unchanged."""
        src = _lib_source(rule="xid_79_gpu_off_bus")
        result = _empty_result()
        _set_outcome(
            result,
            OutcomeAttribution(
                outcome="Fail InfraHW",
                hw_cause="GPU fell off the bus",
                root_cause_symptom="XID 79",
                doc="xid-79",
            ),
            src,
        )

        response = AttributionResponseV2.from_lib_attribution(set(), result)

        assert response.verdict.root_symptom is not None
        assert response.verdict.root_symptom.symptom == "XID 79"
        assert response.verdict.root_symptom.source.rule == "xid_79_gpu_off_bus"
        assert response.verdict.root_symptom.catalog_id == "xid-79"

    def test_root_symptom_carries_null_catalog_id_when_lib_has_no_doc(self):
        """A root-cause outcome the lib leaves without a doc slug surfaces catalog_id=None."""
        src = _lib_source(rule="app_error")
        result = _empty_result()
        _set_outcome(
            result,
            OutcomeAttribution(
                outcome="Fail User",
                hw_cause="Application raised an unhandled exception",
                root_cause_symptom="Python Exception",
            ),
            src,
        )

        response = AttributionResponseV2.from_lib_attribution(set(), result)

        assert response.verdict.root_symptom is not None
        assert response.verdict.root_symptom.catalog_id is None

    def test_attribution_outcome_surfaces_raw_lib_outcome_string(self):
        """V2 surfaces the lib's raw outcome string verbatim (Fail MultiCategory)."""
        src = _lib_source(rule="xid_31_mem_page_fault")
        result = _empty_result()
        _set_outcome(
            result,
            OutcomeAttribution(
                outcome="Fail MultiCategory",
                hw_cause="Not Applicable",
                root_cause_symptom="XID-31-",
            ),
            src,
        )

        response = AttributionResponseV2.from_lib_attribution(set(), result)

        assert response.verdict.attribution_outcome == "Fail MultiCategory"

    def test_attribution_outcome_surfaces_pass_verbatim(self):
        """V2 passes through losslessly-mapped outcomes like Pass too."""
        src = _lib_source(rule="job_ok")
        result = _empty_result()
        _set_outcome(
            result,
            OutcomeAttribution(
                outcome="Pass",
                hw_cause="Not Applicable",
                root_cause_symptom="",
            ),
            src,
        )

        response = AttributionResponseV2.from_lib_attribution(set(), result)

        assert response.verdict.attribution_outcome == "Pass"

    def test_symptoms_emits_node_source_for_per_node_symptom(self):
        """When a single Source is attributed to a single node, V2 emits a NodeSource."""
        src = _lib_source(rule="xid_79")
        result = _empty_result()
        _add_node_symptom(result, "gpu-node-042", "XID-79-", src)
        _add_job_symptom(result, "XID-79-", src)

        response = AttributionResponseV2.from_lib_attribution(set(), result)

        assert "XID-79-" in response.symptoms
        sites = response.symptoms["XID-79-"]
        assert len(sites) == 1
        assert sites[0].type == "NodeSource"
        assert sites[0].node == "gpu-node-042"
        assert sites[0].source.rule == "xid_79"

    def test_symptoms_emits_resourceless_source_when_no_node_binding(self):
        """When a Source is attributed to the job but not to any nodes, V2 emits a ResourcelessSource."""
        src = _lib_source(rule="slurm_state")
        result = _empty_result()
        _add_job_symptom(result, "Cancelled by Timeout", src)

        response = AttributionResponseV2.from_lib_attribution(set(), result)

        sites = response.symptoms["Cancelled by Timeout"]
        assert len(sites) == 1
        assert isinstance(sites[0], ResourcelessSource)
        assert sites[0].source.rule == "slurm_state"

    def test_peer_symptom_emits_one_node_source_per_attributed_node(self):
        """When a single Source is attributed to multiple nodes V2 emits one NodeSource per node."""
        src = _lib_source(rule="link_peer_flap")
        result = _empty_result()
        _add_node_symptom(result, "node-a", "link_flap", src)
        _add_node_symptom(result, "node-b", "link_flap", src)
        _add_job_symptom(result, "link_flap", src)

        response = AttributionResponseV2.from_lib_attribution(set(), result)

        sites = response.symptoms["link_flap"]
        assert [s.node for s in sites] == ["node-a", "node-b"]
        assert all(s.type == "NodeSource" for s in sites)
