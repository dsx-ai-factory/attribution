# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common helper functions for filter and refiner rule files.
"""

from attribution_lib.events import DemotionTypes
from attribution_lib.rule_types.definitions import EventTriple
from attribution_lib.sources import Source


def _bad(node: str | None, symptom: str, source: Source, demotion: dict | str | None = None) -> EventTriple:
    """Create a bad node EventTriple.

    Args:
        node: Node hostname.
        symptom: Symptom description.
        source: Source attribution.
        demotion: Either a dict of {DemotionType: rule_name} entries,
            a string (shorthand for {AFTER_MFU_DEMOTION: value}), or None.
    """
    payload: dict = {"node": node, "symptom": symptom}
    if isinstance(demotion, dict):
        payload.update(demotion)
    elif demotion is not None:
        payload[DemotionTypes.AFTER_MFU_DEMOTION] = demotion
    return EventTriple(type="Attribution.node.bad", payload=payload, source=source)


def _interesting(node: str | None, symptom: str, source: Source, demotion: dict | str | None = None) -> EventTriple:
    """Create an interesting node EventTriple.

    Args:
        node: Node hostname.
        symptom: Symptom description.
        source: Source attribution.
        demotion: Either a dict of {DemotionType: rule_name} entries,
            a string (shorthand for {AFTER_MFU_DEMOTION: value}), or None.
    """
    payload: dict = {"node": node, "symptom": symptom}
    if isinstance(demotion, dict):
        payload.update(demotion)
    elif demotion is not None:
        payload[DemotionTypes.AFTER_MFU_DEMOTION] = demotion
    return EventTriple(type="Attribution.node.interesting", payload=payload, source=source)


def _other_event(event_type: str, payload: dict, source: Source, rule: str) -> EventTriple:
    """Create a generic EventTriple with a derived rule name."""
    return EventTriple(type=event_type, payload=payload, source=source.derived(rule=rule))
