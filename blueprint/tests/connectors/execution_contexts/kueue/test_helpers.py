# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.connectors.execution_contexts.kueue.helpers import (
    controller_owner,
)
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import (
    OwnerIdentity,
    WorkloadIdentity,
)


def test_controller_owner_requires_literal_true():
    def resource(controller_marker):
        owner = {"kind": "Job", "name": "trainer", "uid": "job-uid"}
        if controller_marker != "missing":
            owner["controller"] = controller_marker
        return {"metadata": {"ownerReferences": [owner]}}

    assert controller_owner(resource("missing")) is None
    assert controller_owner(resource(None)) is None
    assert controller_owner(resource(False)) is None
    assert controller_owner(resource(True))["uid"] == "job-uid"


def test_legacy_placement_identity_imports_remain_available():
    assert WorkloadIdentity("training", "workload-uid").uid == "workload-uid"
    assert OwnerIdentity("batch/v1", "Job", "training", "job-uid").uid == "job-uid"
