# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.kubernetes.daemon.health import HealthState


def test_persistence_failures_affect_readiness_until_recovery():
    health = HealthState()
    health.update(initial_list_complete=True)
    assert health.ready()

    health.update(persistence_failed=True)
    assert not health.ready()
    health.update(persistence_failed=False)
    assert health.ready()
