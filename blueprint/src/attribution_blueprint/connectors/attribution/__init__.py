# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Service's `Observation` is re-exported under a distinct name as to not conflict
# with the one in `attribution_blueprint.types`. The former is the shape the service
# accepts, the latter is the shape connectors emit.
from nvidia_attribution_client import (
    AttributorInfoV2,
    Observation as ServiceObservation,
)

from attribution_blueprint.connectors.attribution.client import AttributionConnector

__all__ = [
    "AttributionConnector",
    "AttributorInfoV2",
    "ServiceObservation",
]
