"""Lifecycle predicate + injection metadata (Plan 5c, Task 1)."""

import datetime
import types

from attribution_lib.declarative.expressions import (
    bind_functions,
    called_names_in,
    check_helpers_resolvable,
)
from attribution_lib.declarative.lifecycle import (
    LIFECYCLE_INJECTIONS,
    LIFECYCLE_PREDICATES,
    injections_of,
)


def test_injections_of_detects_declared_params():
    def a(x):
        return x

    def b(x, *, ctx):
        return x

    def c(x, *, source, ctx):
        return x

    assert injections_of(a) == frozenset()
    assert injections_of(b) == frozenset({"ctx"})
    assert injections_of(c) == frozenset({"source", "ctx"})


def test_at_end_of_job_registered():
    assert set(LIFECYCLE_PREDICATES) == {"at_end_of_job"}
    assert LIFECYCLE_INJECTIONS["at_end_of_job"] == frozenset({"source", "ctx"})


def test_at_end_of_job_true_within_window():
    # event 100s before job end -> within the 780s window -> True.
    end = datetime.datetime(2026, 1, 1, 12, 0, 0)
    src = types.SimpleNamespace(time=end - datetime.timedelta(seconds=100))
    ctx = types.SimpleNamespace(job_info=types.SimpleNamespace(job_end_time=end))
    assert LIFECYCLE_PREDICATES["at_end_of_job"](source=src, ctx=ctx) is True


def test_at_end_of_job_false_without_time_or_job():
    src = types.SimpleNamespace(time=None)
    ctx = types.SimpleNamespace(job_info=None)
    assert LIFECYCLE_PREDICATES["at_end_of_job"](source=src, ctx=ctx) is False


def test_at_end_of_job_false_outside_window():
    end = datetime.datetime(2026, 1, 1, 12, 0, 0)
    src = types.SimpleNamespace(time=end - datetime.timedelta(seconds=1000))
    ctx = types.SimpleNamespace(job_info=types.SimpleNamespace(job_end_time=end))
    assert LIFECYCLE_PREDICATES["at_end_of_job"](source=src, ctx=ctx) is False


def test_bind_functions_empty_returns_base():
    base = {"int": int}
    assert bind_functions(base, {}, source="S", ctx="C") is base


def test_bind_functions_binds_declared_params():
    seen = {}

    def helper(x, *, ctx):
        seen["ctx"] = ctx
        return x

    injecting = {"h": (helper, frozenset({"ctx"}))}
    fns = bind_functions({"int": int}, injecting, source="S", ctx="C")
    assert fns["int"] is int  # base preserved
    assert fns["h"]("v") == "v"
    assert seen["ctx"] == "C"  # ctx injected, source NOT passed (not declared)


def test_bind_functions_binds_at_end_of_job():
    end = datetime.datetime(2026, 1, 1, 12, 0, 0)
    src = types.SimpleNamespace(time=end - datetime.timedelta(seconds=100))
    ctx = types.SimpleNamespace(job_info=types.SimpleNamespace(job_end_time=end))
    injecting = {"at_end_of_job": (LIFECYCLE_PREDICATES["at_end_of_job"], LIFECYCLE_INJECTIONS["at_end_of_job"])}
    fns = bind_functions({}, injecting, source=src, ctx=ctx)
    assert fns["at_end_of_job"]() is True  # nullary call, bound source+ctx


def test_check_helpers_resolvable_allows_at_end_of_job():
    # at_end_of_job is a lifecycle built-in, always an allowed callable.
    check_helpers_resolvable("float(value) > 0 and at_end_of_job()", allowed=set(), is_template=False)


def test_called_names_in():
    assert called_names_in("float(value) > 0 and at_end_of_job()", is_template=False) == {"float", "at_end_of_job"}
    assert called_names_in("{pfx(datum, 'x')} and {int(y)}") == {"pfx", "int"}
    assert called_names_in("plain literal") == set()
