"""Tests for declarative ``[const]`` tables and ``@const`` providers."""

import pytest

from attribution_lib.declarative.errors import LoadError
from attribution_lib.declarative.expressions import (
    check_consts_resolvable,
    referenced_const_names,
)
from attribution_lib.rules_engine import (
    RuleIngestionError,
    RuleLoader,
    const,
)
from tests.declarative.conftest import _with_loader


def test_provider_merges_flat_names():
    with _with_loader() as loader:

        @const("nvlink")
        def _p():
            return {"plr_a": 0.05, "plr_b": 0.01}

        assert loader.declarative_ctx.const_values == {"plr_a": 0.05, "plr_b": 0.01}
        assert _p() == {"plr_a": 0.05, "plr_b": 0.01}  # returned unchanged


def test_two_providers_merge():
    with _with_loader() as loader:

        @const("a")
        def _a():
            return {"x": 1}

        @const("b")
        def _b():
            return {"y": 2}

        assert loader.declarative_ctx.const_values == {"x": 1, "y": 2}


def test_duplicate_const_name_across_providers_is_error():
    with _with_loader():

        @const("a")
        def _a():
            return {"dup": 1}

        with pytest.raises(RuleIngestionError, match="dup"):

            @const("b")
            def _b():
                return {"dup": 2}


def test_provider_must_return_dict():
    with _with_loader():
        with pytest.raises(RuleIngestionError, match="must return a dict"):

            @const("bad")
            def _bad():
                return [("x", 1)]


def test_no_active_engine_is_error():
    with pytest.raises(RuleIngestionError, match="No active rule loader"):

        @const("orphan")
        def _p():
            return {"x": 1}


def test_fresh_engine_starts_empty():
    assert RuleLoader().declarative_ctx.const_values == {}


def test_referenced_const_names():
    assert referenced_const_names("float(value) > const.plr_x") == {"plr_x"}
    assert referenced_const_names("const.a or const.b") == {"a", "b"}
    assert referenced_const_names("float(value) > 5.0") == set()


def test_check_consts_resolvable_unknown_raises():
    with pytest.raises(LoadError, match="unknown const 'mystery'"):
        check_consts_resolvable("float(value) > const.mystery", allowed=set(), is_template=False)


def test_check_consts_resolvable_allows_known():
    # no raise
    check_consts_resolvable("float(value) > const.plr_x", allowed={"plr_x"}, is_template=False)


def test_check_consts_resolvable_template():
    with pytest.raises(LoadError, match="unknown const 'z'"):
        check_consts_resolvable("value is {const.z}", allowed=set())
