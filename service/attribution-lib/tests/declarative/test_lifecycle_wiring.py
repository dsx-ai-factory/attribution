"""Per-match ``source`` and ``ctx`` injection reaches compiled closures."""

import datetime
import textwrap
import types

import pytest

from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.context import _using_rule_loader
from attribution_lib.declarative.emit import compile_element
from attribution_lib.declarative.errors import FinalizeError
from attribution_lib.declarative.lifecycle import LIFECYCLE_INJECTIONS, LIFECYCLE_PREDICATES
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rules_engine import RuleContext, RuleLoader
from attribution_lib.sources import Source


def _src(datum, dummy_ctx, when):
    return Source(source_context=dummy_ctx, line_num=1, datum=datum, time=when, resources=AllJobResources, rule=None)


def test_leaf_when_calls_at_end_of_job(dummy_ctx):
    injecting = {"at_end_of_job": (LIFECYCLE_PREDICATES["at_end_of_job"], LIFECYCLE_INJECTIONS["at_end_of_job"])}
    emit = {"type": "bad_node", "when": "float(value) > 0 and at_end_of_job()", "symptom": "down"}
    proc = compile_element(emit, rule_name="r", injecting=injecting)

    end = datetime.datetime(2026, 1, 1, 12, 0, 0)
    ctx = RuleContext(job_info=types.SimpleNamespace(job_end_time=end), hostname="n")
    # event 100s before end -> at_end_of_job True -> emits
    near = _src("x", ctx, end - datetime.timedelta(seconds=100))
    assert len(proc({"value": "5"}, "n", near, ctx)) == 1
    # event 1000s before end -> at_end_of_job False -> no emit
    far = _src("x", ctx, end - datetime.timedelta(seconds=1000))
    assert proc({"value": "5"}, "n", far, ctx) == []
    # value not > 0 -> no emit even at end
    assert proc({"value": "0"}, "n", near, ctx) == []


def test_injecting_none_default_is_unaffected(dummy_ctx):
    emit = {"type": "bad_node", "symptom": "s"}
    proc = compile_element(emit, rule_name="r")
    ctx = RuleContext(job_info=None, hostname="n")
    src = _src("x", ctx, None)
    assert proc({}, "n", src, ctx)[0].payload["symptom"] == "s"


def _load(tmp_path, toml_text):
    loader = RuleLoader()
    with _using_rule_loader(loader):
        f = tmp_path / "rules.toml"
        f.write_text(textwrap.dedent(toml_text))
        return loader, load_toml_rules(loader, str(f))


def test_rule_using_at_end_of_job_loads_with_lifecycle_cases(tmp_path):
    toml = r"""
    sources = ["NODE_METRICS"]
    [[extractor]]
    name = "Ib"
    match_re = 'metric_value: (?P<value>[^,]+)'
    emit = { type = "bad_node", when = "float(value) > 0 and at_end_of_job()", symptom = "down", derived_rule = "ib" }
    [[extractor.cases]]
    expect = "bad"
    line = "metric_value: 5"
    at_end_of_job = true
    symptom = "down"
    [[extractor.cases]]
    expect = "none"
    line = "metric_value: 5"
    at_end_of_job = false
    """
    _, added = _load(tmp_path, toml)
    assert "Ib" in added


def test_at_end_of_job_case_false_covers_when_false(tmp_path):
    # If the only cases both declared at_end_of_job=true, the when:false side would
    # be uncovered and load must fail. This guards that coverage actually evaluates
    # the lifecycle conjunct rather than trusting labels.
    toml = r"""
    sources = ["NODE_METRICS"]
    [[extractor]]
    name = "Ib"
    match_re = 'metric_value: (?P<value>[^,]+)'
    emit = { type = "bad_node", when = "float(value) > 0 and at_end_of_job()", symptom = "down", derived_rule = "ib" }
    [[extractor.cases]]
    expect = "bad"
    line = "metric_value: 5"
    at_end_of_job = true
    symptom = "down"
    """
    with pytest.raises(FinalizeError):  # coverage: when:false side uncovered
        _load(tmp_path, toml)
