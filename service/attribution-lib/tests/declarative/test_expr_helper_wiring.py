"""A compiled closure and load-time coverage both see registered helpers.

Mirrors tests/declarative/test_emit.py: the `dummy_ctx` fixture (conftest), a real
`Source`, and symptom read from `payload["symptom"]`.
"""

import re
import textwrap

import pytest

from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.context import _using_rule_loader
from attribution_lib.declarative.coverage import check_coverage
from attribution_lib.declarative.emit import compile_element
from attribution_lib.declarative.errors import LoadError
from attribution_lib.declarative.expressions import merge_functions
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rules_engine import RuleContext, RuleLoader, expr_helper
from attribution_lib.sources import Source


def _src(datum, dummy_ctx):
    return Source(source_context=dummy_ctx, line_num=1, datum=datum, time=None, resources=AllJobResources, rule=None)


def test_leaf_symptom_calls_helper(dummy_ctx):
    helpers = {"pfx": lambda line, msg: f"{line.split(':')[0]}: {msg}" if ":" in line else msg}
    emit = {"type": "bad_node", "symptom": "{pfx(datum, 'boom')}"}
    proc = compile_element(emit, rule_name="r", helpers=helpers)
    ctx = RuleContext(job_info=None, hostname="node-1")
    events = proc({}, "node-1", _src("imex_worker: boom", dummy_ctx), ctx)
    assert events[0].payload["symptom"] == "imex_worker: boom"


def test_pick_one_branch_when_calls_helper(dummy_ctx):
    helpers = {"is_user_path": lambda p: p.startswith("/home/")}
    emit = {
        "type": "pick_one",
        "branches": [{"when": "is_user_path(path)", "type": "bad_node", "symptom": "user"}],
        "default": {"type": "bad_node", "symptom": "system"},
    }
    proc = compile_element(emit, rule_name="r", helpers=helpers)
    ctx = RuleContext(job_info=None, hostname="n")
    assert proc({"path": "/home/a"}, "n", _src("x", dummy_ctx), ctx)[0].payload["symptom"] == "user"
    assert proc({"path": "/etc/a"}, "n", _src("x", dummy_ctx), ctx)[0].payload["symptom"] == "system"


def test_coverage_evaluates_helper_when():
    # Two cases: one selects the helper branch, one falls through. Coverage must
    # eval `is_user_path(path)` -> needs the functions map.
    functions = merge_functions({"is_user_path": lambda p: p.startswith("/home/")})
    emit = {
        "type": "pick_one",
        "branches": [{"when": "is_user_path(path)", "type": "bad_node", "symptom": "user"}],
        "default": {"type": "bad_node", "symptom": "system"},
    }
    cases = [
        {"expect": "bad", "line": "[ERROR] Not enough space left in /home/a"},
        {"expect": "bad", "line": "[ERROR] Not enough space left in /var/a"},
    ]
    compiled = re.compile(r"Not enough space left in\s+(?P<path>\S+)")

    def match_groups(case):
        m = compiled.search(case["line"])
        g = m.groupdict() if m else {}
        g["datum"] = case["line"]
        return g

    # Must not raise (both pick_one sides covered, helper resolves).
    check_coverage("EnrootNotEnoughSpace", emit, cases, None, match_groups=match_groups, functions=functions)


def _load(tmp_path, toml_text, register=None):
    loader = RuleLoader()
    with _using_rule_loader(loader):
        if register:
            register()  # register helpers on this loader, mimicking a sibling .py
        f = tmp_path / "rules.toml"
        f.write_text(textwrap.dedent(toml_text))
        return loader, load_toml_rules(loader, str(f))


def test_toml_rule_using_registered_helper_loads(tmp_path):
    def register():
        @expr_helper("is_user_path")
        def _p(path):
            return path.startswith("/home/")

    # A raw string keeps `\[`, `\s`, `\S` as single backslashes; a TOML literal
    # (single-quoted) string then carries them verbatim into the regex.
    toml = r"""
    sources = ["APPLOG"]
    [[extractor]]
    name = "EnrootNotEnoughSpace"
    match_re = '\[ERROR\]\s+Not enough space left in\s+(?P<path>\S+)'
    [extractor.emit]
    type = "pick_one"
    branches = [ { when = "is_user_path(path)", type = "bad_node", symptom = "user" } ]
    default = { type = "bad_node", symptom = "system" }
    [[extractor.cases]]
    expect = "bad"
    line = "[ERROR] Not enough space left in /home/a"
    [[extractor.cases]]
    expect = "bad"
    line = "[ERROR] Not enough space left in /var/a"
    """
    loader, added = _load(tmp_path, toml, register)
    assert "EnrootNotEnoughSpace" in added


def test_toml_rule_calling_unknown_helper_is_load_error(tmp_path):
    toml = r"""
    sources = ["APPLOG"]
    [[extractor]]
    name = "Bad"
    match_re = 'x(?P<path>\S+)'
    emit = { type = "bad_node", when = "mystery(path)", symptom = "s" }
    [[extractor.cases]]
    expect = "bad"
    line = "x/home/a"
    [[extractor.cases]]
    expect = "no_match"
    line = "zzz"
    """
    with pytest.raises(LoadError, match="unknown function 'mystery'"):
        _load(tmp_path, toml)
