# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-filter and metric specifications declared by Python rule files.

The specifications themselves are plain values. Registration delegates to the
active ``RuleLoader``; runtime lookup belongs to the built ``RuleEngine``.
"""

import re
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, Union

from attribution_lib.declarative.context import _require_rule_loader
from attribution_lib.observation.inbound import Source as ObservationSource

if TYPE_CHECKING:
    from attribution_lib.job_info import JobInfo
    from attribution_lib.observation.processing import EntryProtocol


PreFilterCallable: TypeAlias = Callable[["EntryProtocol", "JobInfo"], Union[bool, tuple[bool, list[str]]]]


@dataclass
class PreFilterSpec:
    """Pre-filter specification registered by rules for a given source type."""

    include_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    callables: list[PreFilterCallable] = field(default_factory=list)


@dataclass
class MetricSpec:
    """Metric collection specification registered by rules."""

    thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)


def exclude_re(*patterns: str) -> PreFilterCallable:
    """Build a pre-filter callable that skips entries matching a regex with job fields filled in.

    Each pattern may hold ``{field}`` placeholders. A field is any ``JobInfo``
    attribute, plus ``job_number``: the leading digits of ``JobInfo.id``, so a
    step or restart record such as ``1234.1`` gives ``1234``, and an id with no
    leading digits gives ``""``. Values are regex escaped before they are filled
    in. Write a literal regex brace as ``{{`` and ``}}``, for example ``\\d{{4}}``.

    Args:
        patterns: Regexes, searched anywhere in the entry body.

    Returns:
        A callable for ``PreFilterSpec.callables`` that returns True when any pattern matches.

    Raises:
        ValueError: If a pattern names an unknown field.
        re.error: If a pattern is not a valid regex.
    """
    from attribution_lib.job_info import JobInfo  # local: job_info imports the package root

    known_fields = frozenset(JobInfo._fields) | {"job_number"}
    used_fields: set[str] = set()
    for pattern in patterns:
        for _, field_name, _, _ in string.Formatter().parse(pattern):
            if field_name is not None and field_name not in known_fields:
                raise ValueError(f"unknown job field {field_name!r} in pattern {pattern!r}")
            if field_name is not None:
                used_fields.add(field_name)
        re.compile(pattern.format_map({name: "0" for name in known_fields}))

    # Cache key: the job attributes the patterns read. job_number derives from id.
    key_fields = sorted({"id" if name == "job_number" else name for name in used_fields})
    # One-slot cache: a job's entries arrive together, and a dict keyed by job would grow for the process lifetime.
    last_compiled: dict[tuple[str, ...], list[re.Pattern[str]]] = {}

    def _skip(entry: "EntryProtocol", job_info: "JobInfo") -> bool:
        """Skip the entry when any filled-in pattern matches its body.

        Args:
            entry: Observation entry to check.
            job_info: Job under attribution, whose fields fill the patterns.

        Returns:
            True if the entry must be skipped.
        """
        key = tuple(str(getattr(job_info, name)) for name in key_fields)
        compiled = last_compiled.get(key)
        if compiled is None:
            number = re.match(r"\d+", job_info.id)
            fields = {**job_info._asdict(), "job_number": number.group() if number is not None else ""}
            escaped = {name: re.escape(str(value)) for name, value in fields.items()}
            compiled = [re.compile(pattern.format_map(escaped)) for pattern in patterns]
            last_compiled.clear()
            last_compiled[key] = compiled
        body = entry.body()
        return any(pattern.search(body) is not None for pattern in compiled)

    return _skip


def register_pre_filter(source: ObservationSource, spec: PreFilterSpec) -> None:
    """Register a pre-filter specification on the active loader."""
    loader = _require_rule_loader("registering a pre-filter")
    loader.register_pre_filter(source, spec)


def register_metric_spec(source: ObservationSource, spec: MetricSpec) -> None:
    """Register a metric specification on the active loader."""
    loader = _require_rule_loader("registering a metric specification")
    loader.register_metric_spec(source, spec)
