# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for collector params_model plumbing.

A CollectorRule may declare a Pydantic params_model;
RuleEngine.run_collectors parses request-scoped rule_params dicts into
those models before dispatch; attribute(run_collectors=True) opts into
populating AttributionResult.collectors from the engine's digest.
"""

import pytest
from pydantic import BaseModel

import attribution_lib
from attribution_lib.core import (
    AttributionResult,
    CalcBadNodesAttributionResult,
    attribute,
)
from attribution_lib.events import EventManager
from attribution_lib.rules_engine import (
    CollectorRule,
    RuleLoader,
    RuleLocation,
    RuleParameters,
)


class _DemoParams(BaseModel):
    """Schema used by the synthetic collectors below."""

    threshold: float = 0.5
    top_n: int = 3


def _make_collector(name, processor, params_model=None):
    """Build a CollectorRule with a passthrough formatter (raw == digest)."""
    return CollectorRule(
        name=name,
        author="test",
        location=RuleLocation("test.py", 0),
        processor=processor,
        formatter=lambda raw: raw,
        params_model=params_model,
    )


class TestCollectorRuleParamsModel:
    """params_model field on CollectorRule."""

    def test_default_params_model_is_none(self):
        rule = _make_collector("R", lambda _es, _ji, _p: {})
        assert rule.params_model is None

    def test_params_model_stored_on_rule(self):
        rule = _make_collector("R", lambda _es, _ji, _p: {}, params_model=_DemoParams)
        assert rule.params_model is _DemoParams


class TestRunCollectorsParams:
    """RuleEngine.run_collectors threads rule_params to each collector's processor."""

    @pytest.fixture(autouse=True)
    def _setup(self, simple_job_factory):
        self.loader = RuleLoader()
        self.mgr = EventManager(rules_engine=self.loader.build())
        self.stream = self.mgr.create_stream("collector-params-test")
        self.job_info = simple_job_factory(state="RUNNING", nodes=["node0"])
        yield
        self.mgr.close()

    def test_passes_none_when_collector_has_no_params_model(self):
        captured: list = []

        def proc(_es, _ji, params):
            captured.append(params)
            return {}

        self.loader.register_rule(_make_collector("C.NoModel", proc))
        self.loader.build().run_collectors(self.stream, self.job_info)

        assert captured == [None]

    def test_passes_default_params_when_entry_missing(self):
        captured: list = []

        def proc(_es, _ji, params):
            captured.append(params)
            return {}

        self.loader.register_rule(_make_collector("C.Demo", proc, params_model=_DemoParams))
        self.loader.build().run_collectors(self.stream, self.job_info)

        assert captured == [_DemoParams()]

    def test_parses_raw_dict_into_params_model(self):
        captured: list = []

        def proc(_es, _ji, params):
            captured.append(params)
            return {}

        self.loader.register_rule(_make_collector("C.Demo", proc, params_model=_DemoParams))
        self.loader.build().run_collectors(
            self.stream, self.job_info, rule_params={"C.Demo": RuleParameters(params={"threshold": 0.9})}
        )

        assert captured[0].threshold == 0.9
        assert captured[0].top_n == 3  # unspecified field keeps default

    def test_omitting_rule_params_matches_legacy_digest(self):
        """run_collectors() with no rule_params produces the same digest as today."""
        self.loader.register_rule(_make_collector("C.NoModel", lambda _es, _ji, _p: {"k": 1}))

        assert self.loader.build().run_collectors(self.stream, self.job_info) == {"k": 1}
        assert self.loader.build().run_collectors(self.stream, self.job_info, rule_params=None) == {"k": 1}
        assert self.loader.build().run_collectors(self.stream, self.job_info, rule_params={}) == {"k": 1}

    def test_entries_for_unregistered_collectors_are_ignored(self):
        """Lib is permissive: caller validates RuleIds upstream; unknown keys no-op."""
        self.loader.register_rule(_make_collector("C.NoModel", lambda _es, _ji, _p: {"k": 1}))

        digest = self.loader.build().run_collectors(
            self.stream, self.job_info, rule_params={"Not.A.Real.Collector": RuleParameters(params={"foo": "bar"})}
        )

        assert digest == {"k": 1}

    def test_invalid_params_skip_collector_without_aborting_siblings(self):
        """A pydantic ValidationError on one collector logs+skips, sibling still runs."""
        sibling_called: list = []

        def sibling(_es, _ji, params):
            sibling_called.append(params)
            return {"sibling": True}

        self.loader.register_rule(
            _make_collector("C.Demo", lambda _es, _ji, _p: {"never": True}, params_model=_DemoParams)
        )
        self.loader.register_rule(_make_collector("C.Sibling", sibling))

        digest = self.loader.build().run_collectors(
            self.stream, self.job_info, rule_params={"C.Demo": RuleParameters(params={"threshold": "not a number"})}
        )

        assert digest == {"sibling": True}
        assert sibling_called == [None]

    def test_processor_exception_isolated_from_sibling_collectors(self):
        """An exception inside a collector's processor logs+skips, siblings still run."""

        def bombing(_es, _ji, _p):
            raise RuntimeError("kaboom")

        self.loader.register_rule(_make_collector("C.Bomb", bombing))
        self.loader.register_rule(_make_collector("C.Sibling", lambda _es, _ji, _p: {"sibling": True}))

        digest = self.loader.build().run_collectors(self.stream, self.job_info)

        assert digest == {"sibling": True}


class TestAttributeRunCollectors:
    """attribute(run_collectors=...) populates AttributionResult.collectors via the engine."""

    @pytest.fixture(autouse=True)
    def _setup(self, simple_job_factory):
        self._original_engine = attribution_lib._rules_engine
        self.loader = RuleLoader()
        self._publish()
        self.mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        self.stream = self.mgr.create_stream("attribute-run-collectors-test")
        self.job_info = simple_job_factory(state="FAILED", exit_code_signal=1, nodes=["node0"])
        yield
        self.mgr.close()
        attribution_lib._rules_engine = self._original_engine

    def _publish(self):
        """Install what the loader holds now as the process-global engine.

        The engine is immutable, so a test registering a collector has to
        rebuild and reinstall before attribute() will see it.
        """
        attribution_lib._rules_engine = self.loader.build()

    def _call_attribute(self, **overrides):
        self._publish()
        return attribute(
            event_stream=self.stream,
            job_info=self.job_info,
            bad_node_result=CalcBadNodesAttributionResult(),
            extract_job_metadata_from_logfile=False,
            **overrides,
        )

    def test_attribution_result_collectors_defaults_to_empty(self):
        result = AttributionResult(
            bad_nodes_attr=CalcBadNodesAttributionResult(),
            event_stream=self.stream,
        )
        assert result.collectors == {}

    def test_default_does_not_populate_collectors(self):
        self.loader.register_rule(_make_collector("Collect.Test", lambda _es, _ji, _p: {"never_run": True}))
        result = self._call_attribute()
        assert result.collectors == {}

    def test_run_collectors_true_populates_from_registered_collectors(self):
        captured: list = []

        def collector(es, ji, params):
            captured.append((es, ji, params))
            return {"emitted": True}

        self.loader.register_rule(_make_collector("Collect.Test", collector))

        result = self._call_attribute(run_collectors=True)

        assert result.collectors == {"emitted": True}
        assert len(captured) == 1
        assert captured[0][0] is self.stream
        assert captured[0][1] is self.job_info
        assert captured[0][2] is None  # no params_model → params is None

    def test_rule_params_threaded_through_to_collector(self):
        """End-to-end: caller passes rule_params, collector reads params.threshold,
        emits a digest that reflects the configured value."""

        def collector(_es, _ji, params):
            return {"threshold_seen": params.threshold}

        self.loader.register_rule(_make_collector("Collect.Test", collector, params_model=_DemoParams))

        result = self._call_attribute(
            run_collectors=True,
            rule_params={"Collect.Test": RuleParameters(params={"threshold": 0.85})},
        )

        assert result.collectors == {"threshold_seen": 0.85}

    def test_run_collectors_true_with_rule_params_none_uses_defaults(self):
        captured: list = []

        def collector(_es, _ji, params):
            captured.append(params)
            return {}

        self.loader.register_rule(_make_collector("Collect.Test", collector, params_model=_DemoParams))

        self._call_attribute(run_collectors=True)

        assert captured == [_DemoParams()]
