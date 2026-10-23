# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the loader and sub-engine hierarchies."""

import pytest

from attribution_lib.declarative.context import DeclarativeContext
from attribution_lib.exceptions import DuplicateRuleError, UnknownRuleTypeError
from attribution_lib.rule_types.bases import (
    _RULE_TYPE_REGISTRY,
    LoaderBase,
    OrderedLoader,
    SubEngineBase,
    rule_type,
)
from attribution_lib.rule_types.collector import CollectorLoader
from attribution_lib.rule_types.definitions import (
    CollectorContext,
    CollectorRule,
    ExecutionContext,
    ExtractorAccumulatorRule,
    ExtractorRule,
    FilterContext,
    FilterPriority,
    FilterRule,
    RefinerContext,
    RefinerRule,
    Rule,
    RuleLocation,
)
from attribution_lib.rule_types.extractor import ExtractorLoader
from attribution_lib.rule_types.filter import FilterLoader
from attribution_lib.rule_types.refiner import RefinerLoader
from attribution_lib.rules_engine import RuleLoader


class _DummyEngine(SubEngineBase):
    pass


class _DummyLoader(LoaderBase):
    engine_class = _DummyEngine


class _OrderedDummyEngine(SubEngineBase):
    pass


class _OrderedDummyLoader(OrderedLoader):
    engine_class = _OrderedDummyEngine

    def register(self, rule):
        super().register(rule)
        self._add_to_order(rule.priority, rule.name, rule.processor)


class _TestRule(Rule):
    def process(self, *args, **kwargs):
        return None


def _filter_rule(name, priority=FilterPriority.STANDARD, line=1):
    return FilterRule(
        name=name,
        author="test",
        location=RuleLocation("f.py", line),
        processor=lambda ctx: None,
        priority=priority,
    )


def test_loader_base_register_and_snapshot():
    loader = _DummyLoader()
    loader.register(_filter_rule("R1"))
    assert "R1" in loader.rules

    snap = loader.snapshot()
    loader.rules.clear()
    loader.restore(snap)
    assert "R1" in loader.rules


def test_loader_base_snapshot_is_isolated_from_later_registration():
    """A snapshot must not observe rules registered after it was taken."""
    loader = _DummyLoader()
    loader.register(_filter_rule("Before"))
    snap = loader.snapshot()
    loader.register(_filter_rule("After"))

    loader.restore(snap)
    assert set(loader.rules) == {"Before"}


def test_declarative_context_restore_does_not_alias_snapshot():
    context = DeclarativeContext()
    snapshot = context.snapshot()

    context.restore(snapshot)
    for field_name in snapshot:
        getattr(context, field_name)["after_restore"] = object()

    assert all("after_restore" not in saved for saved in snapshot.values())


def test_loader_base_duplicate_raises():
    loader = _DummyLoader()
    loader.register(_filter_rule("Dup", line=1))
    with pytest.raises(DuplicateRuleError, match="first at f.py:1"):
        loader.register(_filter_rule("Dup", line=2))


def test_sub_engine_base_defaults_are_inert():
    engine = _DummyEngine()
    assert engine.extractor_hash_contribution() == ""
    assert engine.attribution_hash_contribution() == b""
    assert engine.extra_rules() == {}
    assert engine.extra_rule_names() == set()
    assert engine.handles(ExecutionContext()) is False


def test_coordinators_do_not_expose_rule_type_specific_helpers():
    loader = RuleLoader()
    engine = loader.build()

    assert hasattr(loader, "extractor_rules") is False
    for name in (
        "rules_of_type",
        "extractor_rules",
        "extractor_accumulator_rules",
        "filter_rules",
        "refiner_rules",
        "collector_rules",
        "parse_collector_params",
    ):
        assert hasattr(engine, name) is False


def test_ordered_loader_holds_only_the_append_order():
    """The loader accumulates; sorting and phase bucketing belong to the engine."""
    loader = _OrderedDummyLoader()
    loader.register(_filter_rule("B", priority=FilterPriority.LATE))
    loader.register(_filter_rule("A", priority=FilterPriority.EARLY))

    assert [entry[2] for entry in loader._execution_order] == ["B", "A"]
    assert hasattr(loader, "_by_phase") is False
    assert hasattr(loader, "_ensure_sorted") is False


def test_ordered_loader_snapshot_round_trips_the_order():
    loader = _OrderedDummyLoader()
    loader.register(_filter_rule("A"))
    snap = loader.snapshot()
    loader.register(_filter_rule("B"))

    loader.restore(snap)
    assert [entry[2] for entry in loader._execution_order] == ["A"]
    assert set(loader.rules) == {"A"}


def test_rule_type_decorator_registers_entry():
    before = len(_RULE_TYPE_REGISTRY)

    @rule_type(_TestRule)
    class _TestLoader(LoaderBase):
        engine_class = _DummyEngine

    try:
        assert len(_RULE_TYPE_REGISTRY) == before + 1
        entry = _RULE_TYPE_REGISTRY[-1]
        assert entry.rule_types == (_TestRule,)
        assert entry.loader_class is _TestLoader
        assert entry.engine_class is _DummyEngine
    finally:
        # Restore the registry; other tests iterate it.
        del _RULE_TYPE_REGISTRY[-1]


def test_rule_type_decorator_rejects_an_already_registered_type():
    before = len(_RULE_TYPE_REGISTRY)

    with pytest.raises(DuplicateRuleError, match="FilterRule"):

        @rule_type(FilterRule)
        class _DuplicateFilterLoader(LoaderBase):
            engine_class = _DummyEngine

    assert len(_RULE_TYPE_REGISTRY) == before


# --- Concrete loader and engine pairs ---------------------------------------


def _collector_rule(name, processor, formatter=lambda raw: raw, params_model=None):
    return CollectorRule(
        name=name,
        author="test",
        location=RuleLocation("f.py", 1),
        processor=processor,
        formatter=formatter,
        params_model=params_model,
    )


def _extractor_rule(name, match_re=r"boom", sources=frozenset()):
    return ExtractorRule(
        name=name,
        author="test",
        location=RuleLocation("f.py", 1),
        match_re=match_re,
        processor=lambda groups, hostname, source, context: [],
        source_filters=sources,
    )


def test_filter_engine_runs_in_priority_order():
    loader = FilterLoader()
    log: list[str] = []
    for name, priority in (("late", FilterPriority.LATE), ("early", FilterPriority.EARLY)):
        loader.register(
            FilterRule(
                name=name,
                author="test",
                location=RuleLocation("f.py", 1),
                processor=(lambda n: lambda ctx: log.append(n))(name),
                priority=priority,
            )
        )
    loader.build().execute(FilterContext())
    assert log == ["early", "late"]


def test_filter_engine_same_priority_keeps_registration_order():
    loader = FilterLoader()
    log: list[str] = []
    for name in ("a", "b", "c"):
        loader.register(
            FilterRule(
                name=name,
                author="test",
                location=RuleLocation("f.py", 1),
                processor=(lambda n: lambda ctx: log.append(n))(name),
            )
        )
    loader.build().execute(FilterContext())
    assert log == ["a", "b", "c"]


def test_filter_engine_phase_selects_one_priority():
    loader = FilterLoader()
    log: list[str] = []
    for name, priority in (("mid", FilterPriority.MID), ("std", FilterPriority.STANDARD)):
        loader.register(
            FilterRule(
                name=name,
                author="test",
                location=RuleLocation("f.py", 1),
                processor=(lambda n: lambda ctx: log.append(n))(name),
                priority=priority,
            )
        )
    loader.build().execute(FilterContext(phase=FilterPriority.MID))
    assert log == ["mid"]


@pytest.mark.parametrize(
    ("phase_window", "expected"),
    [
        ((FilterPriority.SCRUB, FilterPriority.FINAL), ["scrub-start", "scrub-late"]),
        ((FilterPriority.FINAL, FilterPriority.POST_OUTCOME), ["final"]),
        ((FilterPriority.POST_OUTCOME, None), ["post-outcome"]),
    ],
)
def test_refiner_engine_runs_explicit_priority_window(phase_window, expected):
    loader = RefinerLoader()
    log: list[str] = []
    for name, priority in (
        ("before-scrub", FilterPriority.SCRUB - 1),
        ("scrub-start", FilterPriority.SCRUB),
        ("scrub-late", FilterPriority.FINAL - 1),
        ("final", FilterPriority.FINAL),
        ("post-outcome", FilterPriority.POST_OUTCOME),
    ):
        loader.register(
            RefinerRule(
                name=name,
                author="test",
                location=RuleLocation("f.py", 1),
                processor=(lambda n: lambda ctx: log.append(n))(name),
                priority=priority,
            )
        )

    loader.build().execute(RefinerContext(phase_window=phase_window))

    assert log == expected


def test_refiner_engine_default_window_excludes_post_outcome():
    loader = RefinerLoader()
    log: list[str] = []
    for name, priority in (
        ("before-scrub", FilterPriority.SCRUB - 1),
        ("scrub-start", FilterPriority.SCRUB),
        ("scrub-late", FilterPriority.FINAL - 1),
        ("final", FilterPriority.FINAL),
        ("post-outcome", FilterPriority.POST_OUTCOME),
    ):
        loader.register(
            RefinerRule(
                name=name,
                author="test",
                location=RuleLocation("f.py", 1),
                processor=(lambda n: lambda ctx: log.append(n))(name),
                priority=priority,
            )
        )

    loader.build().execute(RefinerContext())

    assert log == ["before-scrub", "scrub-start", "scrub-late", "final"]


def test_engines_ignore_contexts_they_do_not_handle():
    """Each engine acts only on its own context type."""
    log: list[str] = []
    floader = FilterLoader()
    floader.register(
        FilterRule(
            name="f",
            author="test",
            location=RuleLocation("f.py", 1),
            processor=lambda ctx: log.append("filter ran"),
        )
    )
    engine = floader.build()
    engine.execute(CollectorContext(event_stream=None, job_info=None))
    assert log == []


def test_handles_reports_the_engines_context_type():
    assert FilterLoader().build().handles(FilterContext()) is True
    assert FilterLoader().build().handles(RefinerContext()) is False
    assert RefinerLoader().build().handles(RefinerContext()) is True
    assert CollectorLoader().build().handles(CollectorContext(event_stream=None, job_info=None)) is True


def test_rule_engine_dispatches_matching_context_to_sub_engine():
    log: list[str] = []
    loader = RuleLoader()
    loader.register_rule(
        FilterRule(
            name="coordinator-filter",
            author="test",
            location=RuleLocation("f.py", 1),
            processor=lambda ctx: log.append(type(ctx).__name__),
        )
    )

    loader.build().execute(FilterContext())

    assert log == ["FilterContext"]


def test_rule_engine_rejects_unclaimed_context():
    with pytest.raises(UnknownRuleTypeError, match="ExecutionContext"):
        RuleLoader().build().execute(ExecutionContext())


def test_collector_engine_accumulates_into_the_context():
    loader = CollectorLoader()
    loader.register(_collector_rule("c1", lambda stream, job, params: {"a": 1}))
    loader.register(_collector_rule("c2", lambda stream, job, params: {"b": 2}))
    ctx = CollectorContext(event_stream=None, job_info=None)
    loader.build().execute(ctx)
    assert ctx.result == {"a": 1, "b": 2}


def test_collector_engine_keeps_first_value_on_duplicate_key():
    loader = CollectorLoader()
    loader.register(_collector_rule("first", lambda stream, job, params: {"k": "kept"}))
    loader.register(_collector_rule("second", lambda stream, job, params: {"k": "dropped"}))
    ctx = CollectorContext(event_stream=None, job_info=None)
    loader.build().execute(ctx)
    assert ctx.result == {"k": "kept"}


def test_collector_engine_survives_a_failing_collector():
    def _boom(stream, job, params):
        raise RuntimeError("boom")

    loader = CollectorLoader()
    loader.register(_collector_rule("bad", _boom))
    loader.register(_collector_rule("good", lambda stream, job, params: {"ok": 1}))
    ctx = CollectorContext(event_stream=None, job_info=None)
    loader.build().execute(ctx)
    assert ctx.result == {"ok": 1}


def test_extractor_loader_does_not_carry_a_matcher():
    """The matcher is an execution concern; the loader has no business holding one."""
    assert hasattr(ExtractorLoader(), "matcher") is False


def test_extractor_loader_routes_accumulators_to_their_own_store():
    loader = ExtractorLoader()
    loader.register(_extractor_rule("E1"))
    loader.register(
        ExtractorAccumulatorRule(
            name="A1",
            author="test",
            location=RuleLocation("f.py", 1),
            match_re=r"start",
            factory=lambda: None,
        )
    )
    assert set(loader.extra_rules()) == {"A1"}
    engine = loader.build()
    assert set(engine.rules) == {"E1"}
    assert set(engine.extractor_accumulator_rules) == {"A1"}
    assert set(engine.extra_rules()) == {"A1"}
    assert engine.extra_rule_names() == {"A1"}


def test_extractor_loader_rejects_a_duplicate_across_both_stores():
    loader = ExtractorLoader()
    loader.register(_extractor_rule("Shared"))
    assert loader.find_by_name("Shared") is not None
    with pytest.raises(DuplicateRuleError):
        loader.register(_extractor_rule("Shared"))


def test_extractor_hash_contribution_tracks_rule_identity():
    a = ExtractorLoader()
    a.register(_extractor_rule("E1", match_re=r"one"))
    b = ExtractorLoader()
    b.register(_extractor_rule("E1", match_re=r"two"))
    assert a.build().extractor_hash_contribution() != b.build().extractor_hash_contribution()


def test_attribution_hash_contribution_tracks_priority():
    a = FilterLoader()
    a.register(_filter_rule("F1", priority=FilterPriority.EARLY))
    b = FilterLoader()
    b.register(_filter_rule("F1", priority=FilterPriority.LATE))
    assert a.build().attribution_hash_contribution() != b.build().attribution_hash_contribution()
