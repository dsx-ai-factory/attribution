# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Sized
from dataclasses import asdict
from datetime import datetime
from typing import Any, Generic, NamedTuple, Optional, Self, TypeVar

from attribution_lib.exceptions import MismatchedSourceSetValue
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.resources.resource_set import ResourceSet, validate_resource_set
from attribution_lib.utils import log_discard_decorator


def json_helper(obj: Any) -> Any:
    try:
        return obj.to_dict()
    except AttributeError:
        pass

    return asdict(obj)


def recursive_asdict(obj: Any) -> Any:
    # For dataclass instance (not the class itself)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: recursive_asdict(v) for k, v in dataclasses.asdict(obj).items()}
    # For namedtuple instance
    elif hasattr(obj, "_asdict"):
        return {k: recursive_asdict(v) for k, v in obj._asdict().items()}
    # For list/tuple of objects
    elif isinstance(obj, (list, tuple)):
        return [recursive_asdict(v) for v in obj]
    # For dict of objects
    elif isinstance(obj, dict):
        return {k: recursive_asdict(v) for k, v in obj.items()}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    # Otherwise
    else:
        return obj


Ts = TypeVar("Ts")


def set_if_none(lval: Optional[Ts], rval: Optional[Ts]) -> Optional[Ts]:
    return rval if lval is None else lval


class Source(NamedTuple):
    """Stores information about where an attribution came from for eventual consumption by caller/debugger."""

    source_context: ObservationContext
    line_num: Optional[int]
    datum: str
    time: Optional[datetime]
    resources: ResourceSet
    rule: Optional[str]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Reconstruct Source from dict."""
        source_context = ObservationContext.from_dict(data["source_context"])
        time = datetime.fromisoformat(data["time"]) if isinstance(data["time"], str) else data["time"]

        return cls(
            source_context=source_context,
            line_num=data.get("line_num"),
            datum=data["datum"],
            time=time,
            resources=validate_resource_set(data.get("resources") or "AllJobResources"),
            rule=data.get("rule"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = recursive_asdict(self)
        return result

    def derived(
        self,
        *,
        rule: Optional[str] = None,
        time: Optional[datetime] = None,
        datum: Optional[str] = None,
        resources: Optional[ResourceSet] = None,
    ) -> "Source":
        the_time = time if time is not None else self.time
        the_rule = rule if rule is not None else self.rule
        the_datum = datum if datum is not None else self.datum
        the_resources = resources if resources is not None else self.resources
        return Source(self.source_context, self.line_num, the_datum, the_time, the_resources, the_rule)


T = TypeVar("T")


class SourceSet(Sized, Generic[T]):
    """A collection of of sources that have the same value"""

    _value: T
    _bad: set[Source]
    _interesting: set[Source]

    def __init__(self, value: T):
        self._value = value
        self._bad = set()
        self._interesting = set()

    def __len__(self) -> int:
        return len(self._bad)

    def __iter__(self) -> Iterator[Source]:
        return iter(self._bad)

    def value(self) -> T:
        return self._value

    def one_source(self) -> Optional[Source]:
        try:
            return next(iter(self))
        except StopIteration:
            return None

    def all_sources(self) -> Iterator[Source]:
        """Return bad and interesting sources for this value.

        This is intended for provenance/debug views that need to preserve
        demoted sources after a discard operation.
        """
        yield from self._bad
        yield from self._interesting

    def is_void(self) -> bool:
        return not self.has_bad() and not self.has_interesting()

    def has_bad(self) -> bool:
        return len(self._bad) > 0

    def has_interesting(self) -> bool:
        return len(self._interesting) > 0

    def discard_all(self, purge: bool = False) -> None:
        if not purge:
            self._interesting.update(self._bad)
        self._bad.clear()

    def discard_by_rule(self, rules: frozenset[str], purge: bool = False) -> None:
        bads = frozenset(self._bad)
        for v in bads:
            if (rule := v.rule) is not None:
                if rule in rules:
                    self._bad.discard(v)
                    if not purge:
                        self._interesting.add(v)

    def discard_selected(self, sources: Iterable[Source], purge: bool = False) -> set[Source]:
        """Discard selected bad sources and return those that matched.

        By default, matched sources remain available as interesting evidence.
        Sources that are only interesting or are absent are unchanged. The
        return value contains exactly the sources that were bad and discarded.
        """
        matched_sources = self._bad.intersection(sources)
        if not purge:
            self._interesting.update(matched_sources)
        self._bad.difference_update(matched_sources)
        return matched_sources

    def value_mapped_copy(self, func: Callable[[T], Any]) -> "SourceSet":
        new = SourceSet(func(self.value()))
        new._bad.update(self._bad)
        new._interesting.update(self._interesting)
        return new

    def update(self, other: "SourceSet") -> "SourceSet":
        if not isinstance(other, SourceSet):
            raise NotImplementedError()
        if self._value != other._value:
            raise MismatchedSourceSetValue((self._value, other._value))
        self._bad.update(other._bad)
        self._interesting.update(other._interesting)
        return self

    def add(self, src: Source) -> "SourceSet":
        self._bad.add(src)
        return self

    def add_interesting(self, src: Source) -> "SourceSet":
        self._interesting.add(src)
        return self

    @classmethod
    def new_one(cls, value: T, src: Source) -> "SourceSet[T]":
        return cls(value).add(src)

    @classmethod
    def from_dict(cls, data: dict, value_type: Callable[[Any], T]) -> "SourceSet[T]":
        """Reconstruct SourceSet from dict.

        Args:
            data: Dict with _value, _bad, _interesting keys
            value_type: Optional type hint for the value (for enum reconstruction)
        """
        value = value_type(data.get("_value"))
        ss = SourceSet(value)

        for source_data in data.get("_bad", []):
            if isinstance(source_data, dict):
                ss._bad.add(Source.from_dict(source_data))

        for source_data in data.get("_interesting", []):
            if isinstance(source_data, dict):
                ss._interesting.add(Source.from_dict(source_data))

        return ss


class SymptomSet(Sized, Generic[T]):
    """Maintains a mapping from a unique symptom/value to all sources that have produced it.
    Expose set-like functionality."""

    _data: dict[T, SourceSet[T]]

    def __init__(self) -> None:
        self._data = {}

    def __len__(self) -> int:
        return sum((1 for v in self._data.values() if v.has_bad()))

    def is_void(self) -> bool:
        return len(self._data) == 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._data)})"

    def __contains__(self, key: T) -> bool:
        return key in self._data and self._data[key].has_bad()

    def __getitem__(self, key: T) -> SourceSet[T]:
        return self._data[key]

    def has_bad(self) -> bool:
        return any(x.has_bad() for x in self._data.values())

    def has_interesting(self) -> bool:
        return any(x.has_interesting() for x in self._data.values())

    def symptoms(self) -> Iterator[T]:
        return (k for k, v in self._data.items() if v.has_bad())

    def items(self) -> Iterator[tuple[T, SourceSet[T]]]:
        return ((k, v) for k, v in self._data.items() if v.has_bad())

    def values(self) -> Iterator[SourceSet[T]]:
        return (v for v in self._data.values() if v.has_bad())

    def all_items(self) -> Iterator[tuple[T, SourceSet[T]]]:
        """Return all symptom entries, including interesting-only entries.

        Normal attribution consumers should continue using ``items()``. This
        accessor is for provenance/debug consumers that need demoted symptoms.
        """
        return iter(self._data.items())

    def update(self, other: "SymptomSet[T]") -> "SymptomSet[T]":
        if not isinstance(other, SymptomSet):
            raise NotImplementedError()
        for symptom, sources in other._data.items():  # using _data to avoid filtering out interesting
            self._data.setdefault(symptom, SourceSet(symptom)).update(sources)
        return self

    def add(self, source_set: SourceSet[T]) -> "SymptomSet[T]":
        self._data.setdefault(source_set.value(), SourceSet(source_set.value())).update(source_set)
        return self

    @log_discard_decorator
    def discard(self, v: T, purge: bool = False, context: Optional[dict[str, Any]] = None) -> None:
        """Discard a single symptom from the set.

        Args:
            v: A single symptom to discard.
            purge: Whether to purge (True) or mark as interesting (False).
            context: Optional context for logging.
        """
        if v in self._data:
            self._data[v].discard_all(purge)
            if self._data[v].is_void():
                del self._data[v]

    @log_discard_decorator
    def discard_seq(self, symptoms: list[T], purge: bool = False, context: Optional[dict[str, Any]] = None) -> None:
        """Discard a sequence of symptoms from the set.

        Args:
            symptoms: A list of symptoms to discard.
            purge: Whether to purge (True) or mark as interesting (False).
            context: Optional context for logging.
        """
        for symptom in symptoms:
            if symptom in self._data:
                self._data[symptom].discard_all(purge)
                if self._data[symptom].is_void():
                    del self._data[symptom]

    def discard_by_rule(self, rules: frozenset[str], purge: bool = False) -> None:
        for k, v in list(self._data.items()):
            v.discard_by_rule(rules, purge)
            if v.is_void():
                del self._data[k]

    def discard_all(self, purge: bool = False) -> None:
        for key, item in list(self._data.items()):
            item.discard_all(purge)
            if item.is_void():
                del self._data[key]
