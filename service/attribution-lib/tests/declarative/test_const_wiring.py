# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The const namespace reaches compiled closures."""

import textwrap
import types

import pytest

from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.context import _using_rule_loader, const
from attribution_lib.declarative.emit import compile_element
from attribution_lib.declarative.errors import LoadError
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rule_types.definitions import RuleContext
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.sources import Source


def _src(datum, dummy_ctx):
    return Source(source_context=dummy_ctx, line_num=1, datum=datum, time=None, resources=AllJobResources, rule=None)


def test_leaf_when_reads_const(dummy_ctx):
    consts = types.SimpleNamespace(plr_x=0.05)
    emit = {"type": "bad_node", "when": "float(value) > const.plr_x", "symptom": "high"}
    proc = compile_element(emit, rule_name="r", consts=consts)
    ctx = RuleContext(job_info=None, hostname="n")
    # 0.06 > 0.05 -> emits
    assert len(proc({"value": "0.06"}, "n", _src("x", dummy_ctx), ctx)) == 1
    # 0.04 > 0.05 -> no emit
    assert proc({"value": "0.04"}, "n", _src("x", dummy_ctx), ctx) == []


def test_const_absent_default_is_unaffected(dummy_ctx):
    # A helper-free / const-free leaf still works with the default consts=None.
    emit = {"type": "bad_node", "symptom": "s"}
    proc = compile_element(emit, rule_name="r")
    ctx = RuleContext(job_info=None, hostname="n")
    assert proc({}, "n", _src("x", dummy_ctx), ctx)[0].payload["symptom"] == "s"


def _load(tmp_path, toml_text, register=None):
    loader = RuleLoader()
    with _using_rule_loader(loader):
        if register:
            register()
        f = tmp_path / "rules.toml"
        f.write_text(textwrap.dedent(toml_text))
        return loader, load_toml_rules(loader, str(f))


def test_rule_using_provider_const_loads(tmp_path):
    def register():
        @const("nvlink")
        def _p():
            return {"plr_x": 0.05}

    toml = r"""
    sources = ["NVLINK_METRICS"]
    [[extractor]]
    name = "Plr"
    match_re = 'metric_value: (?P<value>[^,]+)'
    emit = { type = "bad_node", when = "float(value) > const.plr_x", symptom = "high", derived_rule = "plr" }
    [[extractor.cases]]
    expect = "bad"
    line = "metric_value: 0.06"
    [[extractor.cases]]
    expect = "none"
    line = "metric_value: 0.04"
    """
    loader, added = _load(tmp_path, toml, register)
    assert "Plr" in added


def test_inline_const_table_loads(tmp_path):
    toml = r"""
    sources = ["NVLINK_METRICS"]
    [const]
    some_threshold = 0.05
    [[extractor]]
    name = "Plr"
    match_re = 'metric_value: (?P<value>[^,]+)'
    emit = { type = "bad_node", when = "float(value) > const.some_threshold", symptom = "high", derived_rule = "plr" }
    [[extractor.cases]]
    expect = "bad"
    line = "metric_value: 0.06"
    [[extractor.cases]]
    expect = "none"
    line = "metric_value: 0.04"
    """
    loader, added = _load(tmp_path, toml)
    assert "Plr" in added
    assert loader.declarative_ctx.const_values["some_threshold"] == 0.05


def test_unknown_const_is_load_error(tmp_path):
    toml = r"""
    sources = ["NVLINK_METRICS"]
    [[extractor]]
    name = "Bad"
    match_re = 'metric_value: (?P<value>[^,]+)'
    emit = { type = "bad_node", when = "float(value) > const.mystery", symptom = "s", derived_rule = "b" }
    [[extractor.cases]]
    expect = "bad"
    line = "metric_value: 0.06"
    [[extractor.cases]]
    expect = "none"
    line = "metric_value: 0.04"
    """
    with pytest.raises(LoadError, match="unknown const 'mystery'"):
        _load(tmp_path, toml)


def test_inline_const_duplicate_of_provider_is_error(tmp_path):
    def register():
        @const("nvlink")
        def _p():
            return {"dup": 1}

    toml = r"""
    sources = ["NVLINK_METRICS"]
    [const]
    dup = 2
    """
    with pytest.raises(LoadError, match="dup"):
        _load(tmp_path, toml, register)


def test_capture_named_const_is_rejected(tmp_path):
    toml = r"""
    sources = ["NVLINK_METRICS"]
    [[extractor]]
    name = "Bad"
    match_re = 'x(?P<const>\S+)'
    emit = { type = "bad_node", symptom = "s", derived_rule = "b" }
    [[extractor.cases]]
    expect = "bad"
    line = "x/y"
    """
    with pytest.raises(LoadError, match="reserved"):
        _load(tmp_path, toml)
