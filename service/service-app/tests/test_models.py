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
    NodeSource,
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


def _demote_job_sources(result: AttributionResult, symptom: str, *sources: LibSource) -> None:
    """Demote selected job-level sources the way a refiner does, leaving the rest bad.

    discard_selected moves the matched sources from the bad set to the interesting
    set rather than dropping them, which is the non-purge default across the lib.
    """
    result.symptoms[symptom].discard_selected(sources)


def _demote_node_symptom(result: AttributionResult, node: str, symptom: str) -> None:
    """Demote a whole per-node symptom, the way a scrub rule clears a cascaded
    symptom off the peers of the node that actually failed."""
    result.bad_nodes_attr.my_bad_nodes.discard([node], [symptom])


def _site_key(site) -> tuple:
    """Identify a site by everything a caller would notice, so a status flip fails the check."""
    return (type(site).__name__, getattr(site, "node", None), site.status, site.source.pos)


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

    def test_sites_report_bad_status_when_interesting_evidence_is_not_requested(self):
        """Every site on the default path is attribution-driving, so it reports status bad."""
        src = _lib_source(rule="xid_79")
        result = _empty_result()
        _add_node_symptom(result, "gpu-node-042", "XID-79-", src)
        _add_job_symptom(result, "XID-79-", src)
        _add_job_symptom(result, "Cancelled by Timeout", _lib_source(rule="slurm_state", line_num=1))

        response = AttributionResponseV2.from_lib_attribution(set(), result)

        assert all(site.status == "bad" for sites in response.symptoms.values() for site in sites)

    def test_fully_demoted_symptom_is_omitted_unless_interesting_evidence_is_requested(self):
        """A symptom whose every source was demoted is hidden by default and surfaces on request."""
        src = _lib_source(rule="link_down")
        result = _empty_result()
        _add_job_symptom(result, "link down(east-west)", src)
        _demote_job_sources(result, "link down(east-west)", src)

        default = AttributionResponseV2.from_lib_attribution(set(), result)
        with_interesting = AttributionResponseV2.from_lib_attribution(set(), result, include_interesting=True)

        assert default.symptoms == {}
        sites = with_interesting.symptoms["link down(east-west)"]
        assert len(sites) == 1
        assert isinstance(sites[0], ResourcelessSource)
        assert sites[0].status == "interesting"

    def test_partially_demoted_symptom_reports_its_surviving_and_demoted_sources_separately(self):
        """Demotion is per source, so one symptom can carry both bad and interesting sites."""
        kept = _lib_source(rule="link_down", line_num=10)
        demoted = _lib_source(rule="link_down", line_num=2)
        result = _empty_result()
        _add_job_symptom(result, "link down(east-west)", kept, demoted)
        _demote_job_sources(result, "link down(east-west)", demoted)

        default = AttributionResponseV2.from_lib_attribution(set(), result)
        with_interesting = AttributionResponseV2.from_lib_attribution(set(), result, include_interesting=True)

        assert [s.source.pos for s in default.symptoms["link down(east-west)"]] == [10]
        sites = with_interesting.symptoms["link down(east-west)"]
        assert [(s.source.pos, s.status) for s in sites] == [(2, "interesting"), (10, "bad")]

    def test_demoting_a_sources_only_node_binding_keeps_its_job_level_bad_site(self):
        """A node whose only evidence was demoted is named as interesting, and the job-level
        bad site it came from is kept rather than replaced by that node binding."""
        src = _lib_source(rule="dataloader")
        result = _empty_result()
        _add_node_symptom(result, "node-b", "DataLoader Exception", src)
        _add_job_symptom(result, "DataLoader Exception", src)
        _demote_node_symptom(result, "node-b", "DataLoader Exception")

        response = AttributionResponseV2.from_lib_attribution(set(), result, include_interesting=True)

        sites = response.symptoms["DataLoader Exception"]
        assert [(type(s), s.status) for s in sites] == [
            (ResourcelessSource, "bad"),
            (NodeSource, "interesting"),
        ]
        assert sites[1].node == "node-b"

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
