import pytest

from attribution_lib.declarative.emit import LoadError, compile_element, compile_leaf
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rules_engine import RuleContext
from attribution_lib.sources import Source


def _ctx(node="node01"):
    return RuleContext(job_info=None, hostname=node)


def _src(dummy_ctx):
    return Source(
        source_context=dummy_ctx,
        line_num=1,
        datum="whatever",
        time=None,
        resources=AllJobResources,
        rule=None,
    )


def test_compile_bad_node_literal_symptom(dummy_ctx):
    proc = compile_leaf({"type": "bad_node", "symptom": "GspRmAlloc failed"}, rule_name="r")
    events = proc({}, "node01", _src(dummy_ctx), _ctx())
    assert len(events) == 1
    assert events[0].type == "Attribution.node.bad"
    assert events[0].payload["symptom"] == "GspRmAlloc failed"
    assert events[0].payload["node"] == "node01"


def test_compile_interesting_node_literal_symptom(dummy_ctx):
    proc = compile_leaf({"type": "interesting_node", "symptom": "slow link"}, rule_name="r")
    events = proc({}, "node02", _src(dummy_ctx), _ctx())
    assert len(events) == 1
    assert events[0].type == "Attribution.node.interesting"
    assert events[0].payload["symptom"] == "slow link"
    assert events[0].payload["node"] == "node02"


def test_compile_generic_event_literal_payload(dummy_ctx):
    proc = compile_leaf(
        {"type": "generic_event", "event_type": "Some.event", "payload": {"k": "v"}},
        rule_name="r",
    )
    events = proc({}, "node01", _src(dummy_ctx), _ctx())
    assert len(events) == 1
    assert events[0].type == "Some.event"
    assert events[0].payload == {"k": "v"}


def test_compile_non_leaf_type_raises():
    with pytest.raises(LoadError):
        compile_leaf({"type": "all_of", "emits": []}, rule_name="r")


def test_compile_bad_node_templated_symptom(dummy_ctx):
    proc = compile_leaf({"type": "bad_node", "symptom": "SXID-{sxid}"}, rule_name="r")
    events = proc({"sxid": "79"}, "node01", _src(dummy_ctx), _ctx())
    assert events[0].payload["symptom"] == "SXID-79"


def test_compile_generic_event_templated_payload(dummy_ctx):
    proc = compile_leaf(
        {"type": "generic_event", "event_type": "Some.event", "payload": {"{k}": "v-{sxid}"}},
        rule_name="r",
    )
    events = proc({"k": "key", "sxid": "79"}, "node01", _src(dummy_ctx), _ctx())
    assert events[0].payload == {"key": "v-79"}


def test_derived_rule_credits_explicit_slug(dummy_ctx):
    proc = compile_leaf(
        {"type": "bad_node", "symptom": "boom", "derived_rule": "some-slug"},
        rule_name="r",
    )
    events = proc({}, "node01", _src(dummy_ctx), _ctx())
    assert events[0].source.rule == "some-slug"


def test_no_derived_rule_leaves_source_rule_unset(dummy_ctx):
    # Without derived_rule the closure leaves source.rule unset so
    # ExtractorRule.process later fills in the stanza name.
    proc = compile_leaf({"type": "bad_node", "symptom": "boom"}, rule_name="r")
    events = proc({}, "node01", _src(dummy_ctx), _ctx())
    assert not events[0].source.rule


def test_derived_rule_credits_generic_event(dummy_ctx):
    proc = compile_leaf(
        {"type": "generic_event", "event_type": "Some.event", "payload": {"k": "v"}, "derived_rule": "some-slug"},
        rule_name="r",
    )
    events = proc({}, "node01", _src(dummy_ctx), _ctx())
    assert events[0].source.rule == "some-slug"


def test_when_gates_emit(dummy_ctx):
    proc = compile_leaf(
        {"type": "interesting_node", "symptom": "hot", "when": "float(value) > 50.0"},
        rule_name="r",
    )
    src = _src(dummy_ctx)
    assert proc({"value": "60"}, "n", src, _ctx()) != []
    assert proc({"value": "10"}, "n", src, _ctx()) == []


@pytest.mark.parametrize("arm", ["branches", "events", "default"])
def test_leaf_rejects_combinator_arms(arm):
    with pytest.raises(LoadError, match="no arms"):
        compile_element({"type": "bad_node", "symptom": "x", arm: []}, rule_name="r")


def test_compile_element_rejects_unknown_type():
    with pytest.raises(LoadError, match="not supported"):
        compile_element({"type": "unknown"}, rule_name="r")


def test_generic_event_when_can_gate(dummy_ctx):
    proc = compile_leaf({"type": "generic_event", "event_type": "x", "when": "enabled"}, rule_name="r")
    assert proc({"enabled": False}, "node", _src(dummy_ctx), _ctx()) == []
