# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests guarding against dangling-reference leaks.

Each attributor pins its own AttributionLib EventManager (and the DuckDB
in-memory connection it owns). These tests prove that removing an attributor
-- whether via explicit DELETE or via expiry cleanup -- actually releases
that whole object graph instead of relying on nothing more than "the dict
entry is gone".
"""

import gc
import weakref
from datetime import datetime, timedelta

from tests.conftest import make_up_attributor_info, post_attributor


def _wrapper_and_refs(app, att_id):
    wrapper = app.state.aim.attributors[att_id]
    event_manager = wrapper.attributor.event_stream.mgr
    return weakref.ref(wrapper), weakref.ref(event_manager)


def test_deleted_attributors_are_garbage_collected(app, client):
    """Explicit DELETE must release the AttributorWrapper and its EventManager."""
    refs = []

    for i in range(20):
        info = make_up_attributor_info(id=str(700000 + i))
        att_id = post_attributor(client, info)

        refs.append(_wrapper_and_refs(app, att_id))

        resp = client.delete(f"/v1/attributor/{att_id}")
        assert resp.status_code == 204

    gc.collect()

    dead_wrappers = sum(1 for wrapper_ref, _ in refs if wrapper_ref() is None)
    dead_event_managers = sum(1 for _, mgr_ref in refs if mgr_ref() is None)

    assert dead_wrappers == len(refs), (
        f"only {dead_wrappers}/{len(refs)} AttributorWrapper instances were collected after delete"
    )
    assert dead_event_managers == len(refs), (
        f"only {dead_event_managers}/{len(refs)} EventManager instances were collected after delete"
    )


def test_expired_attributors_are_garbage_collected(app, client):
    """Background expiry cleanup must release memory the same way explicit DELETE does."""
    refs = []

    for i in range(20):
        info = make_up_attributor_info(id=str(710000 + i))
        att_id = post_attributor(client, info)

        wrapper_ref, mgr_ref = _wrapper_and_refs(app, att_id)
        refs.append((wrapper_ref, mgr_ref))

        # Backdate so the next cleanup pass considers this attributor expired.
        wrapper_ref().last_access_time = datetime.now() - timedelta(hours=100)

    app.state.aim.cleanup_expired_attributors()
    assert len(app.state.aim.attributors) == 0

    gc.collect()

    dead_wrappers = sum(1 for wrapper_ref, _ in refs if wrapper_ref() is None)
    dead_event_managers = sum(1 for _, mgr_ref in refs if mgr_ref() is None)

    assert dead_wrappers == len(refs), (
        f"only {dead_wrappers}/{len(refs)} AttributorWrapper instances were collected after expiry"
    )
    assert dead_event_managers == len(refs), (
        f"only {dead_event_managers}/{len(refs)} EventManager instances were collected after expiry"
    )
