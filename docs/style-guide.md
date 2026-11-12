# Python Style Guide

This guide covers how we write Python, whether the code is written by a person or a coding agent. It builds on [PEP 8](https://peps.python.org/pep-0008/) and on recurring feedback from past code reviews.

Rules come in two groups.

1. **Standard rules** are backed by a PEP, the Python typing docs, an established style guide, or a linter rule. Each one links its source.
2. **Team conventions** are choices we made where no external standard decides. Each one gives its reasoning and an example.

## Standard rules

### S1. Catch narrowly and fail loudly

Catch only the exceptions you expect, around the fewest lines that can raise them. Never write a bare `except:` or `except Exception:` unless you re-raise, or you are in a boundary handler ([S17](#s17-log-handled-errors-with-their-traceback)), including a script's top level, that logs the error. Don't hide a failure behind a sentinel return value or a silent fallback.

```python
# Yes
try:
    payload = json.loads(raw)
except json.JSONDecodeError as e:
    raise EventParseError(f"bad event body: {e}") from e
```

```python
# No
try:
    payload = json.loads(raw)
    process(payload)
except Exception:
    return None
```

Source: [PEP 8](https://peps.python.org/pep-0008/#programming-recommendations)

### S2. Annotate types with modern syntax

- Annotate every public function and method
- Use built-in generics (`list`, `dict`), over `typing.List`
- Write optional types as `X | None`, not `Optional[X]`
- Avoid `Any`; if a parameter really accepts anything, use `object`
- Take arguments as abstract types from `collections.abc` (`Iterable`, `Sequence`, `Mapping`) and return concrete types (`list`, `dict`)
    - Import those abstract types from `collections.abc`, not `typing`.

```python
# Yes
from collections.abc import Iterable
def find_node(name: str, nodes: Iterable[Node]) -> Node | None: ...

# No
from typing import Any, List, Optional
def find_node(name: Any, nodes: List[Node]) -> Optional[Node]: ...
```

Source: [Python typing best practices](https://typing.python.org/en/latest/reference/best_practices.html)

### S3. Check for None explicitly

Use `is None` and `is not None` to test for `None`. Use plain truthiness to test a sequence or collection for emptiness (`if items:`). An `Iterable` may be a generator, which is always truthy, so materialize it first, for example with `list(...)`. Never let a truthiness test treat a valid falsy value such as `0` or `""` as missing.

```python
# Yes
if timeout is None:
    timeout = DEFAULT_TIMEOUT_S

# No
if not timeout:  # also replaces a valid timeout of 0
    timeout = DEFAULT_TIMEOUT_S
```

Source: [PEP 8](https://peps.python.org/pep-0008/#programming-recommendations)

### S4. Write docstrings, and comments that explain why

Every public module, class, function and method has a docstring. Write docstring summaries either imperatively ("Return the ...") or descriptively ("Returns the ..."), and keep one style within a module. Comments explain why the code does something, not what it does. If a comment needs a paragraph to explain what a block does, extract the block into a well-named function instead. Update comments when the code changes.

```python
# Yes
# The scheduler reports end time in the cluster's local zone, not UTC.
end = to_utc(job.end_time, cluster.tz)

# No
# convert end time to utc
end = to_utc(job.end_time, cluster.tz)
```

Source: [Google 3.8](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)

### S5. Name things precisely and consistently

Names are descriptive and precise. Use the domain's terms and the names already in the codebase. Don't introduce a second word for an existing concept, or reuse one word for two concepts.

```python
# Yes
def failed_ranks(job: Job) -> list[int]: ...

# No
def get_data(j) -> list: ...
```

Source: [Google 3.16](https://google.github.io/styleguide/pyguide.html#s3.16-naming)

### S6. Build only what is needed or explicitly planned

Write the simplest code that does today's job. Scaffolding for planned work is fine as part of a PR series, or with a `# TODO(<issue link>): <what's next>` comment. Otherwise, don't add a base class with no abstract methods, a wrapper that only forwards to a library, a parameter no caller passes, or a fallback that preserves behavior nothing depends on.

```python
# Yes
class SingleAttribution(BaseModel): ...

# No
class SingleAttribution(BaseModel, abc.ABC): ...  # no abstract methods
class ResourceAttribution(SingleAttribution):
    pass
```

Sources: [Fowler, YAGNI](https://martinfowler.com/bliki/Yagni.html), [Google Python Style Guide | TODO comments](https://google.github.io/styleguide/pyguide.html#312-todo-comments)

### S7. Keep one source of truth

Before adding a helper, constant or abstraction, look for an existing one and extend it. Two abstractions that are almost the same cost more to maintain than one. The same applies to docs: link to the canonical page rather than copying it.

Source: [The Pragmatic Programmer | Tip 15](https://pragprog.com/tips/)

### S8. Name magic values

Give magic values a name. This covers numbers and also strings that carry meaning, such as a wildcard or sentinel like `"*"`. The name says what the value means and keeps every use in sync.

```python
# Yes
MAX_RETRIES = 3
ANY_XID = "*"
if attempts > MAX_RETRIES: ...
if xid == ANY_XID: ...

# No
if attempts > 3: ...
if xid == "*": ...
```

Source: ruff [PLR2004](https://docs.astral.sh/ruff/rules/magic-value-comparison/)

### S9. Keep functions short

Prefer small, focused functions. Past about 40 lines, or more than three levels of nested control flow, look for pieces to extract into named helpers. Run `ruff check --select C901` to flag functions that may be too complex.

Source: [Google 3.18](https://google.github.io/styleguide/pyguide.html#s3.18-function-length)

### S10. Use explicit interfaces, not attribute probing

When code treats several types alike, give them a shared method, base class or `Protocol`. A `Protocol` is for static type checking; where a runtime type check is needed, use `isinstance` with a concrete class or ABC. Don't branch on `hasattr` or `getattr` probes, and don't read another object's `_private` attributes; add a public method instead. Tests may access protected members of the module under test.

```python
# Yes
class HasNodes(Protocol):
    def node_names(self) -> list[str]: ...

# No
if hasattr(source, "_nodes"):
    names = [n.name for n in source._nodes]
```

Source: [PEP 544](https://peps.python.org/pep-0544/)

### S11. Use typed structures, not primitives

When data has structure, give it a type: a `NamedTuple`, `dataclass`, enum or `TypedDict`. Avoid long tuples, deeply nested dicts, and plain strings that stand for a fixed set of values. For an enum whose values are strings, use `StrEnum`, not `Enum` or `(str, Enum)`, so members behave as their string value everywhere, including comparisons, `str()` and f-strings. With `(str, Enum)`, `str(m)` and `f"{m}"` give `"Cls.MEMBER"`.

```python
# Yes
class Severity(StrEnum):
    WARNING = "warning"
    FATAL = "fatal"

@dataclass(frozen=True)
class NodeFailure:
    node: str
    severity: Severity

# No
failure = ("node-12", "fatal", 3, None)
```

Sources: [Primitive Obsession](https://docs.python.org/3/library/collections.html#collections.namedtuple), [enum.StrEnum](https://docs.python.org/3/library/enum.html#enum.StrEnum)

### S12. Use underscores in long numeric literals

A numeric literal is long when it has 5 or more digits. Unit conversion factors are the exception: write them as powers, such as `10**9` for nanoseconds per second.

```python
# Yes
TIMEOUT_MS = 120_000
NS_PER_S = 10**9

# No
TIMEOUT_MS = 120000
NS_PER_S = 1_000_000_000
```

Source: [PEP 515](https://peps.python.org/pep-0515/)

### S13. Test every behavior change

Every behavior change comes with a test that fails without it. Use `pytest.mark.parametrize` for input variations, with readable ids. Cover `None`, empty, zero and malformed inputs where they apply.

Source: [pytest, parametrizing tests](https://docs.pytest.org/en/stable/example/parametrize.html#different-options-for-test-ids)

### S14. Name tests for the behavior they check

Each test checks one behavior, and its name states the scenario and the expected outcome, so a failing test name says what broke. Long names are fine for tests, but a name that only mirrors the method under test is not.

```python
# Yes
def test_lockout_after_three_invalid_logins(): ...
def test_low_balance_purchase_sends_warning_email(): ...

# No
def test_authenticate(): ...  # which behavior?
def test_process_transaction(): ...  # checks display, email and balance at once
```

Source: [Google Testing Blog | Writing Descriptive Test Names](https://testing.googleblog.com/2014/10/testing-on-toilet-writing-descriptive.html)

### S15. Keep complex logic out of tests

Keep complex logic out of tests. Write expected outputs as literals, inline or in parametrize cases; computing them can repeat the bug under test. Building inputs with simple code, such as an f-string, is fine. Put case variations in `pytest.mark.parametrize` ([S13](#s13-test-every-behavior-change)), not in a loop.

```python
# Yes
assert photos_url("https://example.com") == "https://example.com/u/0/photos"

# No
base = "https://example.com/"
assert photos_url(base) == base + "/u/0/photos"  # double slash goes unnoticed
```

Source: [Google Testing Blog | Don't Put Logic in Tests](https://testing.googleblog.com/2014/07/testing-on-toilet-dont-put-logic-in.html)

### S16. Don't use assert for runtime checks

Don't use `assert` outside tests, including `assert x is not None` to narrow a type. Raise an exception instead, because Python skips asserts when run with `-O`. In tests, `assert` is expected.

```python
# Yes
if rank >= world_size:
    raise RankOutOfRangeError(f"rank {rank} is outside world size {world_size}")
if job.end_time is None:
    raise IncompleteJobError("job has no end time")

# No
assert rank < world_size  # vanishes under python -O
assert job.end_time is not None  # only narrows the type
```

Source: [Google 2.4.4](https://google.github.io/styleguide/pyguide.html#s2.4.4-decision)

### S17. Log handled errors with their traceback

In a boundary handler, the place that catches an error and stops it, such as an entry point or a request or job handler, log it once with `logger.exception(...)` so the traceback is kept, and don't repeat the exception text in the message. Log a condition the code recovers from at the level the repo conventions give ([R7](repo-conventions.md#r7-the-log-level-matches-the-consequence)), with its cause and without a traceback.

```python
# Yes
except RuleLoadError:
    logger.exception("failed to load rules from {}", path)

# No
except RuleLoadError as e:
    logger.error(f"failed to load rules: {e}")  # traceback lost
```

Source: ruff [TRY400](https://docs.astral.sh/ruff/rules/error-instead-of-exception/)

### S18. Return early instead of nesting

Handle edge cases first with a guard clause that returns, raises or continues, so the main path stays at the lowest indentation. Don't put an `else` after a branch that already returns.

```python
# Yes
def failed_ranks(job: Job) -> list[int]:
    if job.exit_code == 0:
        return []
    return [r.rank for r in job.ranks if r.failed]

# No
def failed_ranks(job: Job) -> list[int]:
    if job.exit_code != 0:
        return [r.rank for r in job.ranks if r.failed]
    else:
        return []
```

Source: ruff [RET505](https://docs.astral.sh/ruff/rules/superfluous-else-return/)

### S19. Make parameters keyword-only in long signatures

When a function takes any `bool` parameter or three or more of the same type, put a bare `*` before them so callers must pass each one by name.

```python
# Yes
def attribute(job: Job, *, rules: RuleSet, strict: bool = False) -> Result: ...
attribute(job, rules=rules, strict=True)

# No
def attribute(job: Job, rules: RuleSet, strict: bool = False) -> Result: ...
attribute(job, rules, True)  # what does True mean?
```

Source: [PEP 3102](https://peps.python.org/pep-3102/)

## Team conventions

### T1. Raise domain-specific exceptions

Prefer the component's own exceptions. Define exception classes for its failure modes, derived from one base class, and raise those instead of `ValueError` or `RuntimeError`.

```python
# Yes
class AttributionError(Exception): ...
class RuleLoadError(AttributionError): ...

try:
    rules = load_rules(path)
except RuleLoadError:  # a bug's ValueError still surfaces
    ...

# No
raise ValueError("bad rule file")  # callers must catch every ValueError
```

**Why:** Callers can catch our failures without also catching unrelated built-in errors raised by bugs, and the class name documents what went wrong.

### T2. Validate on construction

Data classes check their invariants in `__post_init__`, so no instance can be invalid, even one built directly, and code that uses them doesn't repeat the checks. Use `from_*` constructors only to parse raw input, and have them call the class so the checks still run. Avoid classes whose fields are all optional.

```python
# Yes
@dataclass(frozen=True)
class TimeWindow:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise InvalidWindowError(f"end {self.end} is before start {self.start}")

    @classmethod
    def from_iso(cls, start: str, end: str) -> "TimeWindow":  # parsing only
        return cls(datetime.fromisoformat(start), datetime.fromisoformat(end))

# No
@dataclass
class TimeWindow:
    start: datetime | None = None
    end: datetime | None = None  # every caller re-checks both
```

**Why:** An object that can't exist in an invalid state removes repeated `None` and range checks from every caller. Background: [Parse, don't validate](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/).

### T3. Avoid “utils”

A `utils` or `helpers` module holds only generic code with no domain knowledge. Everything else goes in a module named for what it does.

```text
# Yes
time_windows.py  # overlap(), clamp()
xid.py           # parse_xid(), XID_CODES

# No
utils.py         # overlap(), parse_xid(), load_config()
```

**Why:** A module named `utils` collects unrelated code, grows without an owner, and its name says nothing about its contents. The wemake linter's [WPS100](https://wemake-python-styleguide.readthedocs.io/en/latest/pages/usage/violations/naming.html) rule bans `utils` and `helpers` as module names outright.

### T4. Keep test-only code out of production modules

No `reset()` hooks or `if testing:` branches in production code. Tests use fixtures, `monkeypatch` or mocks.

```python
# Yes
@pytest.fixture
def cache() -> Cache:
    return Cache()  # each test gets a fresh one

# No
class Cache:
    def reset(self) -> None:  # only tests call this
        self._entries.clear()
```

**Why:** Test hooks are dead code in production, and they blur which methods are the real API.
