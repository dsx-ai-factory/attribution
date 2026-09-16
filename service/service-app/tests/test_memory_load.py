# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Load-shaped memory smoke test.

tests/test_memory_smoke.py proves that a *single* attributor's object graph
is released after delete/expiry. It does not prove the service survives
production-like concurrent traffic -- FastAPI dispatches sync route handlers
onto a real threadpool (this is how uvicorn serves this app, since
app/v1/api.py and app/v2/api.py register `def` handlers, not `async def`),
and production has been OOM-killed, so we need to actually exercise that
threadpool with realistic-sized observations and check both:

  1. Peak memory while N attributors are concurrently live+active (this is
     a capacity number, not a pass/fail -- there's no bound today on how
     many attributors can be live at once).
  2. That memory is actually released once load subsides (a pass/fail check
     -- if this fails under concurrency but tests/test_memory_smoke.py
     passes single-threaded, that points at a race in AttributorManager's
     locking, not a plain dangling reference).
"""

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

from tests.conftest import make_up_attributor_info, post_attributor

_APPLOG_BODY = (Path(__file__).parent / "testdata" / "applog.json").read_text()

_CONCURRENCY = 16
_LIFECYCLES = 80

_process = psutil.Process()


def _current_rss_mb() -> float:
    return _process.memory_info().rss / (1024 * 1024)


def _run_lifecycle(client, i: int, refs: list, refs_lock: threading.Lock):
    """Mimic one real client: create an attributor, post a realistic-sized
    observation, attribute, then delete -- the same sequence a real caller
    driving app/v1/api.py performs."""
    info = make_up_attributor_info(id=f"load-{i}")
    att_id = post_attributor(client, info)

    resp = client.post(
        f"/v1/attributor/{att_id}/observation",
        content=(
            '{"context": {"time_interval": {"start": "'
            + info.workload.job_start_time.isoformat()
            + '", "end": "'
            + info.workload.job_end_time.isoformat()
            + '"}, "resources": "AllJobResources", "source": "applog", '
            '"format": "raw_loki_streams"}, "body": ' + _json_escape(_APPLOG_BODY) + "}"
        ),
    )
    assert resp.status_code == 201, resp.text

    resp = client.get(f"/v1/attributor/{att_id}/attribution")
    assert resp.status_code == 200, resp.text

    wrapper = client.app.state.aim.attributors[att_id]
    event_manager = wrapper.attributor.event_stream.mgr
    with refs_lock:
        refs.append((weakref.ref(wrapper), weakref.ref(event_manager)))

    resp = client.delete(f"/v1/attributor/{att_id}")
    assert resp.status_code == 204, resp.text


def _json_escape(s: str) -> str:
    import json

    return json.dumps(s)


def test_concurrent_load_releases_memory(app, client):
    """Drive realistic concurrent traffic through the full create -> ingest
    -> attribute -> delete lifecycle and confirm memory comes back down."""
    refs: list = []
    refs_lock = threading.Lock()

    rss_samples = []
    stop_sampling = threading.Event()

    def sample_rss():
        while not stop_sampling.is_set():
            rss_samples.append(_current_rss_mb())
            time.sleep(0.1)

    gc.collect()
    baseline_rss = _current_rss_mb()

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()

    try:
        with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
            futures = [pool.submit(_run_lifecycle, client, i, refs, refs_lock) for i in range(_LIFECYCLES)]
            for f in futures:
                f.result()
    finally:
        stop_sampling.set()
        sampler.join(timeout=5)

    peak_rss = max(rss_samples) if rss_samples else baseline_rss

    gc.collect()
    time.sleep(0.2)  # let native allocators (DuckDB) settle after close()
    gc.collect()
    final_rss = _current_rss_mb()

    dead_wrappers = sum(1 for wrapper_ref, _ in refs if wrapper_ref() is None)
    dead_event_managers = sum(1 for _, mgr_ref in refs if mgr_ref() is None)

    print(
        f"\nconcurrency={_CONCURRENCY} lifecycles={_LIFECYCLES} "
        f"baseline={baseline_rss:.1f}MB peak={peak_rss:.1f}MB "
        f"(+{peak_rss - baseline_rss:.1f}MB, ~{(peak_rss - baseline_rss) / _CONCURRENCY:.2f}MB/concurrent-attributor) "
        f"final={final_rss:.1f}MB (+{final_rss - baseline_rss:.1f}MB after cleanup)"
    )

    assert len(refs) == _LIFECYCLES, "not all lifecycles completed"
    assert dead_wrappers == _LIFECYCLES, (
        f"only {dead_wrappers}/{_LIFECYCLES} AttributorWrapper instances were collected after "
        "concurrent delete -- suggests a race in AttributorManager, not a plain dangling reference"
    )
    assert dead_event_managers == _LIFECYCLES, (
        f"only {dead_event_managers}/{_LIFECYCLES} EventManager instances were collected after concurrent delete"
    )

    # Generous, noise-tolerant bound: post-cleanup RSS should be nowhere near
    # the peak. A true leak would leave final_rss close to (or above) peak_rss.
    growth = final_rss - baseline_rss
    peak_growth = peak_rss - baseline_rss
    assert growth < 0.5 * peak_growth + 20, (
        f"RSS did not come back down after cleanup: baseline={baseline_rss:.1f}MB "
        f"peak={peak_rss:.1f}MB final={final_rss:.1f}MB -- looks like a real leak, not GC lag"
    )
