import re

import pytest

import attribution_lib.declarative.loader as declarative_loader
from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.errors import FinalizeError, RegistrationError, UnknownKeyError
from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rules_engine import ExtractorRule, RuleContext, RuleEngine
from attribution_lib.sources import Source


def test_load_toml_rules_is_importable():
    assert callable(load_toml_rules)


def _write(tmp_path, text):
    p = tmp_path / "r.toml"
    p.write_text(text)
    return str(p)


def test_load_registers_a_bad_node_extractor(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "SyslogGspFailure"
match_re = " NVRM: rpcRmApiAlloc_GSP: GspRmAlloc failed"
emit = { type = "bad_node", symptom = "GspRmAlloc failed" }
untested = "coverage exercised in test_cases_coverage"
"""
    engine = RuleEngine()
    added = load_toml_rules(engine, _write(tmp_path, toml))
    assert "SyslogGspFailure" in added
    rule = added["SyslogGspFailure"]
    assert isinstance(rule, ExtractorRule)
    assert rule.match_re == " NVRM: rpcRmApiAlloc_GSP: GspRmAlloc failed"
    assert rule.source_filters == frozenset({ObsSource.SYSLOG})
    # Registered on the engine under the same path @extractor_def uses.
    assert engine.extractor_rules["SyslogGspFailure"] is rule


def test_file_level_tags_land_on_the_registered_rule(tmp_path):
    toml = """
sources = ["SYSLOG"]
tags = ["external"]
[[extractor]]
name = "SyslogGspFailure"
match_re = "GspRmAlloc failed"
emit = { type = "bad_node", symptom = "GspRmAlloc failed" }
untested = "validation"
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert added["SyslogGspFailure"].tags == ("external",)


def test_stanza_tags_merge_with_file_tags_and_deduplicate(tmp_path):
    toml = """
sources = ["SYSLOG"]
tags = ["oom"]

[[extractor]]
name = "Disjoint"
match_re = "x"
emit = { type = "bad_node", symptom = "y" }
untested = "validation"
tags = ["external"]

[[extractor]]
name = "Overlapping"
match_re = "z"
emit = { type = "bad_node", symptom = "y" }
untested = "validation"
tags = ["oom", "external"]
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert added["Disjoint"].tags == ("external", "oom")
    assert added["Overlapping"].tags == ("external", "oom")


def test_untagged_stanza_in_an_untagged_file_has_no_tags(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "x"
emit = { type = "bad_node", symptom = "y" }
untested = "validation"
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert added["R"].tags == ()


def test_loaded_rule_processor_emits(tmp_path, dummy_ctx):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "SyslogGspFailure"
match_re = "GspRmAlloc failed"
emit = { type = "bad_node", symptom = "GspRmAlloc failed" }
untested = "processor behavior asserted directly below"
"""
    engine = RuleEngine()
    added = load_toml_rules(engine, _write(tmp_path, toml))
    rule = added["SyslogGspFailure"]
    assert isinstance(rule, ExtractorRule)
    src = Source(
        source_context=dummy_ctx,
        line_num=1,
        datum="GspRmAlloc failed",
        time=None,
        resources=AllJobResources,
        rule=None,
    )
    events = rule.processor({}, "node01", src, RuleContext(job_info=None, hostname="node01"))
    assert len(events) == 1
    assert events[0].type == "Attribution.node.bad"
    assert events[0].payload["symptom"] == "GspRmAlloc failed"


@pytest.mark.parametrize("reserved", ["datum", "node", "const"])
def test_reserved_capture_group_name_raises(tmp_path, reserved):
    toml = f"""
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "(?P<{reserved}>x)"
emit = {{ type = "bad_node", symptom = "y" }}
untested = "validation"
"""
    with pytest.raises(Exception, match="reserved"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_unknown_source_raises(tmp_path):
    toml = """
sources = ["NOT_A_SOURCE"]
[[extractor]]
name = "R"
match_re = "x"
emit = { type = "bad_node", symptom = "y" }
"""
    engine = RuleEngine()
    with pytest.raises(FinalizeError):
        load_toml_rules(engine, _write(tmp_path, toml))


def test_processor_stanza_unsupported(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "x"
processor = "some.module:fn"
"""
    engine = RuleEngine()
    with pytest.raises(RegistrationError):
        load_toml_rules(engine, _write(tmp_path, toml))


def test_unguarded_optional_capture_is_load_error(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "code (?P<code>\\\\w+)(?: at (?P<loc>\\\\S+))?"
emit = { type = "bad_node", symptom = "{code} {loc}" }
"""
    with pytest.raises(FinalizeError):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_capture_named_datum_is_load_error(tmp_path):
    # `datum` (the full log line) is injected into every eval namespace; a capture
    # of the same name would shadow it, so the loader rejects it.
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "(?P<datum>.*evicted.*)"
emit = { type = "bad_node", symptom = "x" }
untested = "guard rejects load before coverage"
"""
    with pytest.raises(FinalizeError, match="datum"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_guarded_optional_capture_loads(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "code (?P<code>\\\\w+)(?: at (?P<loc>\\\\S+))?"
emit = { type = "bad_node", symptom = "{code}{loc if loc else ''}" }
untested = "guard behavior asserted by load success"
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert "R" in added


def test_when_dominates_leaf_templates(tmp_path):
    # `when` establishes loc is not None, so the symptom may use loc unguarded.
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "code (?P<code>\\\\w+)(?: at (?P<loc>\\\\S+))?"
emit = { type = "bad_node", symptom = "{code} {loc}", when = "loc is not None" }
untested = "when-dominance asserted by load success"
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert "R" in added


def test_when_node_guard_loads_cleanly(tmp_path):
    # `when = "node is not None"` passes name validation (node is in static_names).
    # Before the fix, coverage evaluation crashed with a raw simpleeval NameNotDefined
    # because match_groups didn't inject `node`. node=None at coverage time means
    # when:true is never coverable via cases, so untested is required.
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "hello"
emit = { type = "bad_node", symptom = "bad", when = "node is not None" }
untested = "node is always None during coverage evaluation"
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert "R" in added


def test_engine_load_dispatches_toml(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "hello"
emit = { type = "bad_node", symptom = "hi" }
untested = "dispatch-only test"
"""
    p = _write(tmp_path, toml)
    engine = RuleEngine()
    added = engine.load(p)
    assert "R" in added


def _hash_of(tmp_path, name, symptom):
    d = tmp_path / name
    d.mkdir()
    p = d / "r.toml"
    p.write_text(f'''
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "hello"
emit = {{ type = "bad_node", symptom = "{symptom}" }}
untested = "hash test"
''')
    engine = RuleEngine()
    load_toml_rules(engine, str(p))
    return engine.extractor_hash()


def test_extractor_hash_covers_emit_symptom(tmp_path):
    h1 = _hash_of(tmp_path, "a", "GspRmAlloc failed")
    h2 = _hash_of(tmp_path, "b", "Xid 79 fell off the bus")
    assert h1 != h2


def test_extractor_hash_stable_for_identical_emit(tmp_path):
    h1 = _hash_of(tmp_path, "a", "GspRmAlloc failed")
    h2 = _hash_of(tmp_path, "b", "GspRmAlloc failed")
    assert h1 == h2


# --- load-time template name-resolution guard (Fix 1) ---------------------------


def test_symptom_typo_is_load_error(tmp_path):
    # `valeu` is neither a capture (`value`), datum, const, helper, nor lifecycle
    # predicate -> caught at LOAD instead of raising NameNotDefined at match time.
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "val (?P<value>\\\\d+)"
emit = { type = "bad_node", symptom = "bad {valeu}" }
"""
    with pytest.raises(FinalizeError, match="undeclared name 'valeu'"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_payload_value_typo_is_load_error(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "val (?P<value>\\\\d+)"
emit = { type = "generic_event", event_type = "E", payload = { amount = "{valeu}" } }
"""
    with pytest.raises(FinalizeError, match="undeclared name 'valeu'"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_catalog_project_column_typo_is_load_error(tmp_path):
    # The project leaf renders over the RECORD's columns (nice_name); a typo in a
    # column reference is caught at load.
    toml = """
sources = ["SYSLOG"]
[rows.xid]
1 = { nice_name = "XID-1" }
[[extractor]]
name = "R"
match_re = "code (?P<code>\\\\d+)"
[extractor.emit]
type = "catalog_lookup"
catalog = "xid"
key = "int(code)"
project = { type = "interesting_node", symptom = "{nice_naem}" }
not_found = "none"
"""
    with pytest.raises(FinalizeError, match="undeclared name 'nice_naem'"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_catalog_key_typo_is_load_error(tmp_path):
    # A typo in the `key` expression references an undeclared name; caught at load.
    toml = """
sources = ["SYSLOG"]
[rows.xid]
1 = { nice_name = "XID-1" }
[[extractor]]
name = "R"
match_re = "code (?P<code>\\\\d+)"
[extractor.emit]
type = "catalog_lookup"
catalog = "xid"
key = "int(coed)"
project = { type = "interesting_node", symptom = "{nice_name}" }
not_found = "none"
"""
    with pytest.raises(FinalizeError, match="undeclared name 'coed'"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_catalog_key_set_literal_loads_without_stray_brace_error(tmp_path):
    # A `key` expression containing a set/dict literal (inner braces) must not
    # produce a confusing "stray brace" LoadError -- it should load cleanly.
    # (Name validation inside set-literal expressions is a known limitation of
    # the placeholder regex and is not attempted.)
    toml = """
sources = ["SYSLOG"]
[rows.xid]
foo = { nice_name = "FOO" }
bar = { nice_name = "BAR" }
[[extractor]]
name = "R"
match_re = "code (?P<code>\\\\S+)"
untested = "set-literal key loads without stray-brace error"
[extractor.emit]
type = "catalog_lookup"
catalog = "xid"
key = "datum if datum in {'foo', 'bar'} else 'foo'"
project = { type = "interesting_node", symptom = "{nice_name}" }
not_found = "none"
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert "R" in added


def test_rows_scalar_entry_is_load_error(tmp_path):
    # A non-table entry under [rows.*] (scalar instead of inline table) is caught
    # at parse time before any rule uses the table.
    toml = """
sources = ["SYSLOG"]
[rows.xid]
1 = "not a table"
[[extractor]]
name = "R"
match_re = "code (?P<code>\\\\d+)"
[extractor.emit]
type = "catalog_lookup"
catalog = "xid"
key = "int(code)"
project = { type = "interesting_node", symptom = "{nice_name}" }
not_found = "none"
"""
    with pytest.raises(FinalizeError, match=r"\[rows\.xid\]\.1 must be a table"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_when_typo_is_load_error(tmp_path):
    # A typo in a leaf `when` expression is caught at load (not at match time).
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "val (?P<value>\\\\d+)"
emit = { type = "bad_node", symptom = "bad {value}", when = "valeu > 0" }
"""
    with pytest.raises(FinalizeError, match="undeclared name 'valeu'"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_catalog_when_typo_is_load_error(tmp_path):
    # A typo in a catalog_lookup `when` expression is caught at load.
    toml = """
sources = ["SYSLOG"]
[rows.xid]
1 = { nice_name = "XID-1" }
[[extractor]]
name = "R"
match_re = "code (?P<code>\\\\d+)"
[extractor.emit]
type = "catalog_lookup"
catalog = "xid"
key = "int(code)"
when = "codee > 0"
project = { type = "interesting_node", symptom = "{nice_name}" }
not_found = "none"
"""
    with pytest.raises(FinalizeError, match="undeclared name 'codee'"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_valid_template_names_load(tmp_path):
    # A capture, datum, and a fixed function all resolve -> loads clean.
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "val (?P<value>\\\\d+)"
emit = { type = "bad_node", symptom = "bad {value} in {datum} ({int(value)})" }
untested = "name-resolution asserted by load success"
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert "R" in added


def test_rows_provider_collision_with_catalog(tmp_path):
    engine = RuleEngine()
    engine.record_tables["x"] = {"a": {}}
    with pytest.raises(RegistrationError, match="collides with.*catalog"):
        load_toml_rules(engine, _write(tmp_path, "[catalog.x]\na = { type = 'bad_node', symptom = 'x' }"))


def test_element_catalog_row_name_typo_is_load_error(tmp_path):
    # A typo in an element catalog row's symptom template is caught at load.
    toml = """
sources = ["SYSLOG"]
[catalog.severity]
low  = { type = "bad_node", symptom = "{captuer} is bad" }
[[extractor]]
name = "R"
match_re = "sev (?P<capture>\\\\S+)"
untested = "validation"
[extractor.emit]
type = "catalog_lookup"
catalog = "severity"
key = "capture"
not_found = "none"
"""
    with pytest.raises(FinalizeError, match="undeclared name 'captuer'"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_element_catalog_row_unguarded_optional_capture_is_load_error(tmp_path):
    # An element catalog row using an optional capture without a guard is caught at load.
    toml = """
sources = ["SYSLOG"]
[catalog.severity]
low  = { type = "bad_node", symptom = "{level} is bad" }
[[extractor]]
name = "R"
match_re = "sev (?P<level>\\\\S+)?"
untested = "validation"
[extractor.emit]
type = "catalog_lookup"
catalog = "severity"
key = "level or 'low'"
not_found = "none"
"""
    with pytest.raises(FinalizeError, match="possibly-absent capture"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


@pytest.mark.parametrize(
    ("extractor", "message"),
    [
        ("emit = { type = 'bad_node', symptom = 'x' }", "missing required 'name'"),
        ("name = 'r'", "requires either"),
    ],
)
def test_loader_rejects_malformed_registration_stanzas(tmp_path, extractor, message):
    with pytest.raises(RegistrationError, match=message):
        load_toml_rules(RuleEngine(), _write(tmp_path, f"[[extractor]]\n{extractor}\n"))


@pytest.mark.parametrize(
    ("stanza", "message"),
    [
        ("name = 'r'\nemit = { type = 'bad_node', symptom = 'x' }", "missing required 'match_re'"),
        (
            "name = 'r'\nmatch_re = '['\nemit = { type = 'bad_node', symptom = 'x' }",
            "invalid 'match_re'",
        ),
        (
            "name = 'r'\nmatch_re = 'x'\nemit = { type = 'try_all', events = [{ type = 'bad_node', symptom = 'x' }] }\n"
            "[[extractor.cases]]\nline = 'no match'",
            "uncovered obligations",
        ),
    ],
)
def test_loader_wraps_emit_finalize_errors(tmp_path, stanza, message):
    with pytest.raises(FinalizeError, match=message):
        load_toml_rules(RuleEngine(), _write(tmp_path, f"[[extractor]]\n{stanza}\n"))


def test_loader_wraps_unknown_source(tmp_path):
    text = """
sources = ["UNKNOWN"]
[[extractor]]
name = "r"
match_re = "x"
emit = { type = "bad_node", symptom = "x" }
untested = "validation"
"""
    with pytest.raises(FinalizeError, match="unknown source"):
        load_toml_rules(RuleEngine(), _write(tmp_path, text))


def test_processor_missing_match_is_wrapped(tmp_path):
    engine = RuleEngine()
    engine.rule_processors["p"] = lambda groups: []
    text = "[[extractor]]\nname = 'r'\nprocessor = 'p'\n"
    with pytest.raises(FinalizeError, match="missing required 'match_re'"):
        load_toml_rules(engine, _write(tmp_path, text))


def test_processor_preserves_existing_finalize_error(tmp_path, monkeypatch):
    error = FinalizeError("already finalized")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(declarative_loader, "compile_processor", fail)
    engine = RuleEngine()
    engine.rule_processors["p"] = lambda groups: []
    text = "[[extractor]]\nname = 'r'\nprocessor = 'p'\nmatch_re = 'x'\n"
    with pytest.raises(FinalizeError) as caught:
        load_toml_rules(engine, _write(tmp_path, text))
    assert caught.value is error


def test_emit_preserves_existing_finalize_error(tmp_path, monkeypatch):
    error = FinalizeError("already finalized")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(declarative_loader, "compile_element", fail)
    text = """
[[extractor]]
name = "r"
match_re = "x"
emit = { type = "bad_node", symptom = "x" }
untested = "validation"
"""
    with pytest.raises(FinalizeError) as caught:
        load_toml_rules(RuleEngine(), _write(tmp_path, text))
    assert caught.value is error


def test_family_generated_emit_undeclared_when_name_is_load_error(tmp_path):
    # A family emit `when` referencing a name that isn't datum/const/a helper -
    # e.g. an author mistakenly expects a capture group - is caught at load time.
    toml = """
sources = ["SYSLOG"]
[rows.kind]
foo = { label = "Foo" }
[[family]]
name      = "F"
rows_from = "kind"
pattern   = "{label}"
emit      = { type = "bad_node", symptom = "bad", when = "xid is not None" }
untested  = "validation"
"""
    with pytest.raises(FinalizeError, match="undeclared name 'xid'"):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


_TAGGED_STANZA = """
sources = ["SYSLOG"]
{file_tags}
[[extractor]]
name = "R"
match_re = "x"
emit = {{ type = "bad_node", symptom = "y" }}
untested = "validation"
{stanza_tags}
"""


def test_file_level_internal_and_external_tags_are_rejected_naming_the_file(tmp_path):
    text = _TAGGED_STANZA.format(file_tags='tags = ["internal", "external"]', stanza_tags="")
    path = _write(tmp_path, text)
    with pytest.raises(RegistrationError, match=re.escape(path)):
        load_toml_rules(RuleEngine(), path)


@pytest.mark.parametrize(
    ("file_tags", "stanza_tags"),
    [
        pytest.param("", 'tags = ["internal", "external"]', id="both-on-the-stanza"),
        pytest.param('tags = ["internal"]', 'tags = ["external"]', id="one-per-layer"),
    ],
)
def test_stanza_internal_and_external_tags_are_rejected_naming_the_file(tmp_path, file_tags, stanza_tags):
    path = _write(tmp_path, _TAGGED_STANZA.format(file_tags=file_tags, stanza_tags=stanza_tags))
    with pytest.raises(FinalizeError, match=re.escape(path)):
        load_toml_rules(RuleEngine(), path)


def test_stanza_bare_string_tags_is_load_error(tmp_path):
    # A bare string is rejected as a type error rather than splatting into one tag
    # per character, which is what merging it unvalidated would do.
    path = _write(tmp_path, _TAGGED_STANZA.format(file_tags="", stanza_tags='tags = "external"'))
    with pytest.raises(FinalizeError, match=re.escape(path)):
        load_toml_rules(RuleEngine(), path)


def test_family_preserves_existing_finalize_error(tmp_path, monkeypatch):
    error = FinalizeError("already finalized")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(declarative_loader, "stamp_family", fail)
    with pytest.raises(FinalizeError) as caught:
        load_toml_rules(RuleEngine(), _write(tmp_path, '[[family]]\nname = "f"\n'))
    assert caught.value is error


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(_TAGGED_STANZA.format(file_tags='tag = ["x"]', stanza_tags=""), id="file-level"),
        pytest.param(_TAGGED_STANZA.format(file_tags="", stanza_tags='tag = ["x"]'), id="extractor-stanza"),
        pytest.param(
            """
[[extractor]]
tag = ["x"]
emit = { type = "bad_node", symptom = "y" }
""",
            id="unnamed-extractor-stanza",
        ),
        pytest.param(
            """
sources = ["SYSLOG"]
[rows.kind]
foo = { label = "Foo" }
[[family]]
name      = "F"
rows_from = "kind"
pattern   = "{label}"
emit      = { type = "bad_node", symptom = "bad" }
untested  = "validation"
tag       = ["x"]
""",
            id="family-stanza",
        ),
        pytest.param(
            """
sources = ["SYSLOG"]
[prefilter]
incld = ["x"]
""",
            id="prefilter",
        ),
        pytest.param(
            """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "x"
derived_rules = ["a"]
emit = { type = "bad_node", symptom = "y" }
untested = "validation"
""",
            id="processor-only-key-on-an-emit-stanza",
        ),
    ],
)
def test_unknown_key_at_any_declaration_level_is_rejected(tmp_path, text):
    with pytest.raises(UnknownKeyError):
        load_toml_rules(RuleEngine(), _write(tmp_path, text))


def test_non_table_extractor_is_not_reported_as_an_unknown_key(tmp_path):
    with pytest.raises(RegistrationError) as caught:
        load_toml_rules(RuleEngine(), _write(tmp_path, 'extractor = "nope"\n'))
    assert not isinstance(caught.value, UnknownKeyError)


def test_file_level_unknown_key_leaves_the_engine_untouched(tmp_path):
    text = """
sources = ["SYSLOG"]
tag = ["typo"]
[const]
threshold = 5
[[extractor]]
name = "R"
match_re = "x"
emit = { type = "bad_node", symptom = "y" }
untested = "validation"
"""
    engine = RuleEngine()
    with pytest.raises(UnknownKeyError):
        load_toml_rules(engine, _write(tmp_path, text))
    assert engine.extractor_rules == {}
    assert engine.const_values == {}
