# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import tempfile
import textwrap

import pytest

import attribution_lib
import attribution_lib.exceptions
from attribution_lib.declarative.errors import RegistrationError
from attribution_lib.declarative.loader import load_toml_rules
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rules_engine import _PRE_FILTERS, PreFilterSpec, RuleLoader, register_pre_filter


@pytest.fixture
def clean_registry():
    """Run the test against an empty registry, then restore whatever was there before."""
    saved = dict(_PRE_FILTERS)
    _PRE_FILTERS.clear()
    yield
    _PRE_FILTERS.clear()
    _PRE_FILTERS.update(saved)


@pytest.fixture
def initialized_lib(monkeypatch):
    """Stand in for a completed setup_rules() without paying to load the bundled rules."""
    monkeypatch.setattr(attribution_lib, "initialized", True)


@pytest.mark.usefixtures("clean_registry", "initialized_lib")
class TestPreFilterRegistration:
    def test_register_and_get_pre_filter(self):
        spec = PreFilterSpec(
            include_terms=["error", "fatal"],
            exclude_terms=["debug"],
        )
        register_pre_filter(ObservationSource.APPLOG, spec)
        result = attribution_lib.get_pre_filter(ObservationSource.APPLOG)
        assert result is not None
        assert result.include_terms == ["error", "fatal"]
        assert result.exclude_terms == ["debug"]

    def test_get_pre_filter_unregistered_source_returns_empty_spec(self):
        register_pre_filter(ObservationSource.APPLOG, PreFilterSpec(include_terms=["error"]))
        assert attribution_lib.get_pre_filter(ObservationSource.SYSLOG) == PreFilterSpec()

    def test_get_pre_filter_raises_when_setup_rules_never_ran(self, monkeypatch):
        monkeypatch.setattr(attribution_lib, "initialized", False)
        with pytest.raises(attribution_lib.exceptions.UninitializedRuleEngineError):
            attribution_lib.get_pre_filter(ObservationSource.APPLOG)

    def test_multiple_registrations_merge(self):
        register_pre_filter(
            ObservationSource.UFMLOG,
            PreFilterSpec(include_terms=["term1"], exclude_terms=["exc1"]),
        )
        register_pre_filter(
            ObservationSource.UFMLOG,
            PreFilterSpec(include_terms=["term2"], exclude_terms=["exc2"]),
        )
        result = attribution_lib.get_pre_filter(ObservationSource.UFMLOG)
        assert result is not None
        assert "term1" in result.include_terms
        assert "term2" in result.include_terms
        assert "exc1" in result.exclude_terms
        assert "exc2" in result.exclude_terms

    def test_registration_copies_lists(self):
        """Registration should copy, not alias, the input lists."""
        original = ["term1"]
        register_pre_filter(
            ObservationSource.APPLOG,
            PreFilterSpec(include_terms=original),
        )
        original.append("term2")
        result = attribution_lib.get_pre_filter(ObservationSource.APPLOG)
        assert "term2" not in result.include_terms


class TestBundledPrefilterRegistration:
    """Verify that setup_rules() registers prefilters for every expected source.

    These tests guard against migration mistakes where a prefilter file stops
    being loaded (e.g. a missing import, a renamed file, a bad TOML section).
    setup_class re-runs setup_rules(force=True) so unit tests in this file
    that clear _PRE_FILTERS don't affect these integration tests.
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
        from attribution_lib.rules_engine import _PRE_FILTERS

        assert set(_PRE_FILTERS) == {
            ObservationSource.APPLOG,
            ObservationSource.UFMLOG,
            ObservationSource.HEALTH_CHECK,
        }


@pytest.mark.usefixtures("clean_registry", "initialized_lib")
class TestTomlPrefilterLoading:
    def _load(self, toml_text: str) -> None:
        loader = RuleLoader()
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
            f.write(textwrap.dedent(toml_text).encode())
            path = f.name
        try:
            load_toml_rules(loader, path)
        finally:
            os.unlink(path)

    def test_include_exclude_registered(self):
        self._load("""
            sources = ["APPLOG"]
            [prefilter]
            include = ["error", "fatal"]
            exclude = ["debug", "info"]
        """)
        result = attribution_lib.get_pre_filter(ObservationSource.APPLOG)
        assert "error" in result.include_terms
        assert "fatal" in result.include_terms
        assert "debug" in result.exclude_terms
        assert "info" in result.exclude_terms

    def test_prefilter_inherits_file_sources(self):
        self._load("""
            sources = ["UFMLOG"]
            [prefilter]
            include = ["Link went down"]
        """)
        result = attribution_lib.get_pre_filter(ObservationSource.UFMLOG)
        assert "Link went down" in result.include_terms

    def test_prefilter_overrides_sources(self):
        self._load("""
            sources = ["APPLOG"]
            [prefilter]
            sources  = ["UFMLOG"]
            include  = ["node is Down"]
        """)
        result = attribution_lib.get_pre_filter(ObservationSource.UFMLOG)
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
        _PRE_FILTERS.clear()
        self._load("""
            sources = ["APPLOG"]
        """)
        assert len(_PRE_FILTERS) == 0
