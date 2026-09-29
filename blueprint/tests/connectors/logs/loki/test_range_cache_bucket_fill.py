"""Filling fetched entries into buckets must stay faithful and sub-quadratic.

`LokiRangeCache.query` used to rescan the whole fetched list once per bucket,
so the work for one missing range grew with the square of the span. Grouping
each entry once must store byte-for-byte what the rescan stored, including the
order inside a bucket, which `iter_partitions` exposes to callers unsorted.

A bucket is identified by the absolute instant it starts at, so the grid must
be stepped in absolute time. Both DST transitions are covered here because
wall-clock stepping breaks them in opposite ways: a spring-forward repeats
instants and runs the starts backwards, a fall-back skips an hour of them.
"""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from attribution_blueprint.connectors.logs.loki.client import datetime_to_nanoseconds
from attribution_blueprint.connectors.logs.loki.range_cache import LokiRangeCache

QUERY = '{job="syslog"}'
CONFIG = {"tenant": "fact"}
BUCKET = LokiRangeCache.BUCKET

# Windows spanning a real transition of a real zone, as `datetime` positional
# fields. The 2026-11-01 02:00 fall-back repeats the local hour 01:00..01:59;
# the 2026-03-08 02:00 spring-forward skips 02:00..02:59 so those local times
# never happen.
DST_WINDOWS = [
    (ZoneInfo("America/New_York"), (2026, 11, 1, 0, 30)),
    (ZoneInfo("America/Los_Angeles"), (2026, 3, 8, 1, 40)),
]
DST_IDS = ["dst-fallback", "dst-spring-forward"]

# The one bucket pre-filled by test_grouping_matches_the_rescan_across_several_missing_ranges.
MIDDLE_INDEX = 6


def timestamp(item):
    """Return one entry's Unix nanosecond timestamp.

    Args:
        item: Entry built by `entries_at` or by a test directly.

    Returns:
        The entry's `ts` field.
    """
    return item["ts"]


class RecordingStore:
    """Range cache store that keeps every stored bucket payload."""

    def __init__(self):
        """Start with nothing stored."""
        self.puts = {}

    def get(self, key):
        """Miss every read so a query fetches its whole window.

        Args:
            key: Bucket key being read.

        Returns:
            `None`, always.
        """
        return None

    def put(self, key, value, *, ttl_seconds):
        """Record one stored bucket payload under `key`.

        Args:
            key: Bucket key being written.
            value: Payload the cache stored.
            ttl_seconds: Requested lifetime, unused by this store.
        """
        self.puts[key] = value


def rescan_reference(buckets, range_start, range_end, values):
    """Group `values` exactly the way the pre-fix nested rescan did.

    The membership test is the origin/main body of the bucket fill loop, kept
    verbatim so the single-pass grouping is compared against the behaviour it
    replaced rather than against a restatement of it. Buckets are keyed and
    bounded by absolute instant rather than by comparing the datetimes
    directly, because a fall-back gives two buckets one local time and an aware
    datetime compares and hashes on its local fields alone when the zone
    matches, which would silently merge that pair inside the oracle.

    Args:
        buckets: `(bucket_start, bucket_end)` pairs the fill would iterate.
        range_start: Start of the fetched range, inclusive.
        range_end: End of the fetched range, exclusive.
        values: Entries the fetch returned.

    Returns:
        The entries each stored bucket would hold, keyed by bucket index and
        in the order the rescan produced them.
    """
    stored = {}
    range_start_ns = datetime_to_nanoseconds(range_start)
    range_end_ns = datetime_to_nanoseconds(range_end)
    for index, (bucket_start, bucket_end) in enumerate(buckets):
        start_ns = datetime_to_nanoseconds(bucket_start)
        end_ns = datetime_to_nanoseconds(bucket_end)
        if start_ns < range_start_ns or end_ns > range_end_ns:
            continue
        bucket_values = [item for item in values if start_ns <= timestamp(item) < end_ns]
        stored[index] = list(bucket_values)
    return stored


def run_query(cache, store, start, end, values, now):
    """Drive the real `query` and return what it stored, keyed by bucket start.

    Args:
        cache: Cache under test.
        store: The `RecordingStore` backing `cache`.
        start: Requested window start, inclusive.
        end: Requested window end, exclusive.
        values: Entries every fetch returns.
        now: Wall clock the query runs at, deciding which buckets are hot.

    Returns:
        `(buckets, fetches, stored)`: the bucket list the query used, the
        `(range_start, range_end)` pairs it fetched, and the stored payload of
        every bucket that reached the store, keyed by bucket index.
    """
    fetches = []

    def fetch(query, range_start, range_end):
        """Record one requested range and return the same entries for it.

        Args:
            query: Loki selector, unused.
            range_start: Start of the missing range, inclusive.
            range_end: End of the missing range, exclusive.

        Returns:
            A fresh copy of `values`.
        """
        fetches.append((range_start, range_end))
        return list(values)

    cache.query(
        query=QUERY,
        config=CONFIG,
        start=start,
        end=end,
        fetch=fetch,
        timestamp=timestamp,
        now=now,
    )
    buckets = cache._buckets(start, end)
    stored = {}
    for index, (bucket_start, _bucket_end) in enumerate(buckets):
        key = cache._bucket_key(QUERY, CONFIG, bucket_start)
        if key in store.puts:
            stored[index] = store.puts[key]
    return buckets, fetches, stored


def entries_at(offsets, origin, per_offset=1):
    """Build entries whose timestamps sit at `origin + offset`, with stable ids.

    Args:
        offsets: Offsets from `origin`, one group of entries each.
        origin: Datetime the offsets are measured from.
        per_offset: How many entries to build at each offset, so several
            entries can share one timestamp.

    Returns:
        Entries with a `ts` field in Unix nanoseconds and a unique `id`.
    """
    values = []
    for index, offset in enumerate(offsets):
        for repeat in range(per_offset):
            values.append({"ts": datetime_to_nanoseconds(origin + offset), "id": f"{index}-{repeat}"})
    return values


@pytest.mark.parametrize(
    ("zone", "start_fields"),
    [(UTC, (2026, 3, 4, 12, 3, 17)), *DST_WINDOWS],
    ids=["utc", *DST_IDS],
)
def test_single_pass_grouping_stores_what_the_rescan_stored(zone, start_fields):
    """Every stored bucket matches the rescan, contents and order alike.

    Args:
        zone: Time zone the requested window is expressed in. Both DST
            transitions are covered, since wall-clock stepping used to break
            them in opposite directions.
        start_fields: `datetime` positional fields for the window start.
    """
    start = datetime(*start_fields, tzinfo=zone)
    end = start + timedelta(hours=3)
    cache = LokiRangeCache(RecordingStore())
    origin = cache._align(start)
    offsets = [
        -BUCKET,  # before every bucket
        timedelta(0),  # exactly the first bucket_start
        BUCKET - timedelta(microseconds=1),  # last instant of the first bucket
        BUCKET,  # exactly a bucket_end, so it belongs to the next bucket
        BUCKET + timedelta(seconds=137),
        timedelta(hours=1, minutes=12, seconds=4),
        timedelta(hours=3),  # exactly the requested end
        timedelta(hours=9),  # past every bucket
    ]
    # Repeats give several entries identical timestamps, so a reordering inside
    # one bucket cannot hide behind a stable sort downstream.
    values = entries_at(offsets, origin, per_offset=3)

    store = RecordingStore()
    cache = LokiRangeCache(store)
    now = end + timedelta(days=1)
    buckets, fetches, stored = run_query(cache, store, start, end, values, now)

    assert len(fetches) == 1
    expected = rescan_reference(buckets, fetches[0][0], fetches[0][1], values)
    assert stored == expected
    for index, items in expected.items():
        assert [item["id"] for item in stored[index]] == [item["id"] for item in items]
    assert any(items == [] for items in expected.values()), "empty buckets must still be stored"
    assert any(items != [] for items in expected.values())


@pytest.mark.parametrize(("zone", "start_fields"), DST_WINDOWS, ids=DST_IDS)
def test_every_fetched_entry_reaches_exactly_one_bucket_across_a_transition(zone, start_fields):
    """A DST transition must not drop, duplicate, or reorder a cached entry.

    Buckets used to be stepped by adding `self._bucket` to a local datetime,
    which advances the wall clock while `_align` and `_bucket_key` identify a
    bucket by the absolute instant it starts at. Across the spring-forward the
    nonexistent 02:00 hour resolved at the pre-transition offset, so bucket
    starts named instants the real 03:00 hour named again: the two shared one
    aligned cache key and the one stored last, empty, overwrote the filled one,
    and the starts ran backwards in real time in between. Across the fall-back
    it skipped the repeated hour instead, leaving those instants in no bucket.

    Entries are laid down every minute of real time rather than of wall-clock
    time, so the transition hour is covered whichever way the clock moves.

    Args:
        zone: Time zone whose transition the window spans.
        start_fields: `datetime` positional fields for the window start.
    """
    start = datetime(*start_fields, tzinfo=zone)
    end = start + timedelta(hours=3)
    now = end + timedelta(days=1)
    store = RecordingStore()
    cache = LokiRangeCache(store)
    minute_ns = 60 * 1_000_000_000
    first_ns = datetime_to_nanoseconds(cache._align(start))
    values = [{"ts": ns, "id": str(ns)} for ns in range(first_ns, datetime_to_nanoseconds(end), minute_ns)]

    buckets, fetches, stored = run_query(cache, store, start, end, values, now)

    assert len(fetches) == 1, fetches
    starts_ns = [datetime_to_nanoseconds(bucket_start) for bucket_start, _bucket_end in buckets]
    assert starts_ns == sorted(starts_ns), "bucket starts must increase in real time"
    assert len(set(starts_ns)) == len(starts_ns), "no two buckets may start at one instant"
    keys = {cache._bucket_key(QUERY, CONFIG, bucket_start) for bucket_start, _bucket_end in buckets}
    assert len(keys) == len(buckets), "each bucket must own its cache key"
    assert stored == rescan_reference(buckets, fetches[0][0], fetches[0][1], values)
    placed = [item["id"] for payload in stored.values() for item in payload]
    assert len(set(placed)) == len(placed), "no entry may be stored in two buckets"
    assert set(placed) == {item["id"] for item in values}, "no fetched entry may be lost"


@pytest.mark.parametrize(("zone", "start_fields"), DST_WINDOWS, ids=DST_IDS)
def test_hot_boundary_follows_each_buckets_real_end_across_a_transition(zone, start_fields):
    """A bucket stops being hot exactly one buffer after it really ends.

    The hot check used to add the bucket width to the bucket start, which
    advances the wall clock, so either side of a fall-back it placed the end an
    hour from where it is. An hour late, a settled bucket is never cached and
    is refetched forever; an hour early, a bucket still collecting entries is
    cached incomplete and every later read serves that short copy. `now` is
    given in UTC so the boundary is compared in absolute time.

    Args:
        zone: Time zone whose transition the window spans.
        start_fields: `datetime` positional fields for the window start.
    """
    start = datetime(*start_fields, tzinfo=zone)
    cache = LokiRangeCache(RecordingStore())
    buckets = cache._buckets(start, start + timedelta(hours=3))
    assert len(buckets) > 0
    for bucket_start, bucket_end in buckets:
        settled = datetime.fromtimestamp(bucket_end.timestamp() + cache._hot_buffer.total_seconds(), tz=UTC)
        assert cache._is_hot(bucket_start, settled), bucket_start
        assert not cache._is_hot(bucket_start, settled + timedelta(seconds=1)), bucket_start


def test_grouping_matches_the_rescan_across_several_missing_ranges():
    """A store holding a middle bucket splits the window into two ranges."""
    start = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    now = end + timedelta(days=1)

    class PrefilledStore(RecordingStore):
        def __init__(self, hit_key):
            super().__init__()
            self.hit_key = hit_key

        def get(self, key):
            return [] if key == self.hit_key else None

    probe = LokiRangeCache(RecordingStore())
    middle = probe._align(start) + MIDDLE_INDEX * BUCKET
    store = PrefilledStore(probe._bucket_key(QUERY, CONFIG, middle))
    cache = LokiRangeCache(store)
    origin = cache._align(start)
    values = entries_at([timedelta(minutes=m) for m in range(0, 60, 3)], origin, per_offset=2)

    buckets, fetches, stored = run_query(cache, store, start, end, values, now)

    assert len(fetches) == 2, fetches
    expected = {}
    for range_start, range_end in fetches:
        expected.update(rescan_reference(buckets, range_start, range_end, values))
    assert stored == expected
    assert MIDDLE_INDEX not in stored, "a cached bucket is never refilled"


def test_empty_fetch_still_stores_every_bucket_the_rescan_stored():
    """No entries at all must leave the same empty buckets behind."""
    start = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)
    end = start + timedelta(minutes=25)
    store = RecordingStore()
    cache = LokiRangeCache(store)
    now = end + timedelta(days=1)

    buckets, fetches, stored = run_query(cache, store, start, end, [], now)

    expected = rescan_reference(buckets, fetches[0][0], fetches[0][1], [])
    assert stored == expected
    assert list(stored.values()) == [[] for _ in buckets]


def test_bucket_fill_cost_does_not_grow_with_the_span():
    """The fill reads each entry's timestamp once, however many buckets there are.

    The pre-fix rescan read every entry once per bucket, so this count grew
    with the span while the entries stayed the same. Counting extractor calls
    keeps the check deterministic instead of timing it.
    """
    start = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)
    probe = LokiRangeCache(RecordingStore())
    origin = probe._align(start)
    values = entries_at([timedelta(seconds=5 * n) for n in range(120)], origin)

    counts = {}
    for hours in (2, 12, 24):
        calls = 0

        def counting_timestamp(item):
            nonlocal calls
            calls += 1
            return item["ts"]

        end = start + timedelta(hours=hours)
        store = RecordingStore()
        cache = LokiRangeCache(store)
        cache.query(
            query=QUERY,
            config=CONFIG,
            start=start,
            end=end,
            fetch=lambda query, range_start, range_end: list(values),
            timestamp=counting_timestamp,
            now=end + timedelta(days=1),
        )
        assert len(cache._buckets(start, end)) == hours * 12
        counts[hours] = calls

    assert counts[2] == counts[12] == counts[24], counts
    assert counts[24] < len(values) * 12, counts
