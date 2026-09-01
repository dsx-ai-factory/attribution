# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from typing import Optional

from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import EventStream
from attribution_lib.sources import (
    SourceSet,
    SymptomSet,
)


@dataclass(kw_only=True)
class XidParserInfo:
    job_disruption: str
    category: str
    code: Optional[str] = None
    subcode: Optional[str] = None
    port_down_reason_code: Optional[str] = None
    pci_id: Optional[str] = None


class SyslogAttributionParams:
    @classmethod
    def collect_xid_pci_mapping(cls, event_stream: EventStream) -> dict[str, str]:
        """Build dict of xid/pci mapping from event stream; mimics logic formerly found in check()"""
        typ_id = event_stream.mgr.types.get_id("XID_parser.pci_mapping")

        rows = event_stream.query_raw(
            "SELECT payload FROM events WHERE context_id=? AND event_type=? AND status=1",
            [event_stream.ctx_id, typ_id],
        )
        res: dict[str, str] = {}
        for (payload_json,) in rows:
            payload = json.loads(payload_json) if payload_json else {}
            res.update(payload)
        return res

    @classmethod
    def collect_xidparser_actions(cls, event_stream: EventStream) -> tuple[SymptomSet[str], SymptomSet[str]]:
        """Collect xidparser actions and apply some filtering logic to them. Mimics logic formerly found in syslog.check()."""
        typ_id = event_stream.mgr.types.get_id("XID_parser.actions")

        recovery_actions: SymptomSet[str] = SymptomSet()
        investigatory_actions: SymptomSet[str] = SymptomSet()

        for eid, evt_type, payload, source in query_events_with_source(event_stream, typ_id):
            if source is not None and (recovery := payload.get("recovery_action")) is not None:
                recovery_actions.add(SourceSet(recovery).add(source))
                if (investigatory := payload.get("investigatory_action")) is not None:
                    investigatory_actions.add(SourceSet(investigatory).add(source))

        if len(recovery_actions) > 1:
            recovery_actions.discard("IGNORE")
        if len(investigatory_actions) > 1:
            investigatory_actions.discard("IGNORE")

        return recovery_actions, investigatory_actions
