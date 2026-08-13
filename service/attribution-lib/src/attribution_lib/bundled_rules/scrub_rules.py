# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from attribution_lib.rules_engine import FilterPriority, RefinerContext, refiner_def


# _priority() from refiner_rules.py is not available here (exec-context only).
# _SCRUB_ORDER in refiner_rules.py has one entry: "ScrubBadNodes" at FilterPriority.SCRUB + 0.
# Use + 5 to slot after it within the SCRUB band (step size is 10).
_LUSTRE_SCRUB_PRIORITY = FilterPriority.SCRUB + 5


@refiner_def(
    name="LustreErrorCleanup",
    author="NVIDIA Attribution team",
    priority=_LUSTRE_SCRUB_PRIORITY,
    derived_rules=["LustreErrorCleanup"],
)
def lustre_error_cleanup(context: RefinerContext):
    """Discard LustreError from per-node attributions when it is the sole symptom
    and Lustre Client Evicted is absent. Also discard DataLoader Exception from
    any node that has LustreError (co-occurrence cascade)."""
    if context.my_bad_nodes is None:
        return
    nodes = list(context.my_bad_nodes.attributions.keys())
    for node in nodes:
        errors = context.my_bad_nodes.attributions[node]
        if "LustreError" not in errors:
            continue
        if len(errors) == 1:  # "Lustre Client Evicted" cannot co-exist when len == 1; check is implicit
            errors.discard("LustreError")
        # NOTE: the original core.py code had "Dataloader Exception" (lowercase d), which
        # was a silent no-op. This uses the correct casing and is intentionally live.
        errors.discard("DataLoader Exception")
        errors.discard("(non-fatal) DataLoader Exception")
