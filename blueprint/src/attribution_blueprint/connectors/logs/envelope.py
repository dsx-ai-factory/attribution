# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json


def _as_str(value: object) -> str:
    """Envelope fields come from arbitrary JSON, so only a str is usable downstream."""
    return value if isinstance(value, str) else ""


def extract_body_and_node(payload: str, node: str | None) -> tuple[str, str | None]:
    """Pull the log line, and a hostname when the record had none, out of a JSON-encoded log payload.

    Handles OTel (`body` + `attributes.hostname`), `message`, and CloudEvents
    (`data.message` + `data.hostname`) envelopes; anything else is returned verbatim.
    """
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload, node
    if not isinstance(parsed, dict):
        return payload, node

    attributes = parsed.get("attributes")
    if node is None and isinstance(attributes, dict):
        node = _as_str(attributes.get("hostname")) or None
    body = _as_str(parsed.get("body")) or _as_str(parsed.get("message"))
    if body == "":
        data = parsed.get("data")
        if isinstance(data, dict):
            body = _as_str(data.get("message"))
            if node is None:
                node = _as_str(data.get("hostname")) or None
    if body == "":
        body = payload
    return body, node
