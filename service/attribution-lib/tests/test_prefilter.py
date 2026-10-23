# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import textwrap

import pytest

import attribution_lib
from attribution_lib.declarative.errors import RegistrationError
from attribution_lib.declarative.loader import load_toml_rules
from attribution_lib.exceptions import RuleIngestionError, UninitializedRuleEngineError
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.prefilters import PreFilterSpec, register_pre_filter
from attribution_lib.rules_engine import RuleLoader


@pytest.fixture
def active_loader():
    with RuleLoader() as loader:
        yield loader


class TestPreFilterRegistration:
    def test_register_and_get_pre_filter(self, active_loader, monkeypatch):
        spec = PreFilterSpec(
            include_terms=["error", "fatal"],
            exclude_terms=["debug"],
        )
        register_pre_filter(ObservationSource.APPLOG, spec)
        monkeypatch.setattr(attribution_lib, "_rules_engine", active_loader.build())
        result = attribution_lib.get_pre_filter(ObservationSource.APPLOG)
        assert result is not None
        assert result.include_terms == ["error", "fatal"]
        assert result.exclude_terms == ["debug"]

    def test_get_pre_filter_unregistered_source_returns_empty_spec(self, active_loader):
        register_pre_filter(ObservationSource.APPLOG, PreFilterSpec(include_terms=["error"]))
        assert active_loader.build().get_pre_filter(ObservationSource.SYSLOG) == PreFilterSpec()

    def test_get_pre_filter_raises_when_setup_rules_never_ran(self, monkeypatch):
        monkeypatch.setattr(attribution_lib, "_rules_engine", None)
        with pytest.raises(UninitializedRuleEngineError):
            attribution_lib.get_pre_filter(ObservationSource.APPLOG)

    def test_multiple_registrations_merge(self, active_loader):
        register_pre_filter(
            ObservationSource.UFMLOG,
            PreFilterSpec(include_terms=["term1"], exclude_terms=["exc1"]),
        )
        register_pre_filter(
            ObservationSource.UFMLOG,
            PreFilterSpec(include_terms=["term2"], exclude_terms=["exc2"]),
        )
        result = active_loader.build().get_pre_filter(ObservationSource.UFMLOG)
        assert result is not None
        assert "term1" in result.include_terms
        assert "term2" in result.include_terms
        assert "exc1" in result.exclude_terms
        assert "exc2" in result.exclude_terms

    def test_registration_copies_lists(self, active_loader):
        """Registration should copy, not alias, the input lists."""
        original = ["term1"]
        register_pre_filter(
            ObservationSource.APPLOG,
            PreFilterSpec(include_terms=original),
        )
        original.append("term2")
        result = active_loader.build().get_pre_filter(ObservationSource.APPLOG)
        assert "term2" not in result.include_terms

    def test_registration_requires_an_active_loader(self):
        with pytest.raises(RuleIngestionError, match="No active rule loader"):
            register_pre_filter(ObservationSource.APPLOG, PreFilterSpec())

    def test_loaders_do_not_share_pre_filters(self):
        first = RuleLoader()
        second = RuleLoader()
        first.register_pre_filter(ObservationSource.APPLOG, PreFilterSpec(include_terms=["error"]))

        assert ObservationSource.APPLOG in first.pre_filters
        assert second.pre_filters == {}


class TestBundledPrefilterRegistration:
    """Verify that setup_rules() registers prefilters for every expected source.

    These tests guard against migration mistakes where a prefilter file stops
    being loaded (e.g. a missing import, a renamed file, a bad TOML section).
    setup_class re-runs setup_rules(force=True) to install a freshly built engine.
    """

    @classmethod
    def setup_class(cls):
        import attribution_lib.core as core

        core.setup_rules(force=True)

    def test_applog_prefilter_registered(self):
        pf = attribution_lib.get_pre_filter(ObservationSource.APPLOG)
        assert pf.include_terms, "APPLOG prefilter has no include terms"
        assert pf.exclude_terms, "APPLOG prefilter has no exclude terms"
        assert "error" in pf.include_terms
        assert "NCCL INFO" in pf.exclude_terms

    def test_ufmlog_prefilter_registered(self):
        pf = attribution_lib.get_pre_filter(ObservationSource.UFMLOG)
        assert "Link went down" in pf.include_terms
        assert "node is Down" in pf.include_terms
        assert "message repeated" in pf.exclude_terms

    def test_health_check_prefilter_registered(self):
        pf = attribution_lib.get_pre_filter(ObservationSource.HEALTH_CHECK)
        assert "laso" in pf.exclude_terms
        assert "prolog" in pf.exclude_terms
        assert len(pf.callables) == 1, "epilog callable must be registered"

    def test_registered_sources_complete(self):
        """Fail if a source gains or loses a prefilter unexpectedly."""
        assert set(attribution_lib.get_rules_engine().pre_filters) == {
            ObservationSource.APPLOG,
            ObservationSource.UFMLOG,
            ObservationSource.HEALTH_CHECK,
        }


class TestTomlPrefilterLoading:
    def _load(self, toml_text: str) -> RuleLoader:
        loader = RuleLoader()
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
            f.write(textwrap.dedent(toml_text).encode())
            path = f.name
        try:
            load_toml_rules(loader, path)
        finally:
            os.unlink(path)
        return loader

    def test_include_exclude_registered(self):
        loader = self._load("""
            sources = ["APPLOG"]
            [prefilter]
            include = ["error", "fatal"]
            exclude = ["debug", "info"]
        """)
        result = loader.build().get_pre_filter(ObservationSource.APPLOG)
        assert "error" in result.include_terms
        assert "fatal" in result.include_terms
        assert "debug" in result.exclude_terms
        assert "info" in result.exclude_terms

    def test_prefilter_inherits_file_sources(self):
        loader = self._load("""
            sources = ["UFMLOG"]
            [prefilter]
            include = ["Link went down"]
        """)
        result = loader.build().get_pre_filter(ObservationSource.UFMLOG)
        assert "Link went down" in result.include_terms

    def test_prefilter_overrides_sources(self):
        loader = self._load("""
            sources = ["APPLOG"]
            [prefilter]
            sources  = ["UFMLOG"]
            include  = ["node is Down"]
        """)
        result = loader.build().get_pre_filter(ObservationSource.UFMLOG)
        assert "node is Down" in result.include_terms

    def test_prefilter_invalid_source_raises(self):
        with pytest.raises(RegistrationError):
            self._load("""
                [prefilter]
                sources = ["NO_SUCH_SOURCE"]
                include = ["x"]
            """)

    def test_prefilter_no_sources_raises(self):
        with pytest.raises(RegistrationError):
            self._load("""
                [prefilter]
                include = ["x"]
            """)

    def test_no_prefilter_section_is_fine(self):
        """A TOML file with no [prefilter] section should not register anything."""
        loader = self._load("""
            sources = ["APPLOG"]
        """)
        assert loader.pre_filters == {}


def test_failed_rule_load_rolls_back_specs(tmp_path):
    rule_file = tmp_path / "failing_rules.py"
    rule_file.write_text(
        textwrap.dedent("""
            from attribution_lib.observation.inbound import Source
            from attribution_lib.rules import MetricSpec, PreFilterSpec, register_metric_spec, register_pre_filter

            register_pre_filter(Source.APPLOG, PreFilterSpec(include_terms=["error"]))
            register_metric_spec(Source.NODE_METRICS, MetricSpec(thresholds={"m1": {}}))
            raise RuntimeError("fail after registration")
        """),
        encoding="utf-8",
    )
    loader = RuleLoader()

    with pytest.raises(RuntimeError, match="fail after registration"):
        loader.load(str(rule_file))

    assert loader.pre_filters == {}
    assert loader.metric_specs == {}
