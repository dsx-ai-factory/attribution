# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Build the `requests` session the query clients share."""

from __future__ import annotations

import hashlib
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from attribution_blueprint.connectors.auth.protocol import Auth

DEFAULT_RETRIES = 2
RETRYABLE_HTTP_STATUS = (429, 500, 502, 503, 504)


def apply_auth(session: requests.Session, auth: Auth | None) -> None:
    """Install `auth`'s headers, client certificate, and CA bundle on `session`."""
    if auth is None:
        return
    kwargs = auth.request_kwargs()
    if "headers" in kwargs:
        session.headers.update(kwargs["headers"])
    if "cert" in kwargs:
        session.cert = kwargs["cert"]
    if "verify" in kwargs:
        session.verify = kwargs["verify"]


def build_session(auth: Auth | None, *, retries: int = DEFAULT_RETRIES) -> requests.Session:
    """Return a session carrying `auth` that retries transient POST failures.

    Connection failures and HTTP 429/5xx responses are retried `retries` times
    with exponential backoff, and the last response is returned so the caller's
    `raise_for_status` reports it. Read timeouts are not retried.
    """
    if retries < 0:
        raise ValueError(f"retries must not be negative, got {retries}")
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=0,
        status=retries,
        backoff_factor=1.0,
        status_forcelist=RETRYABLE_HTTP_STATUS,
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    apply_auth(session, auth)
    return session


def credential_identity(auth: Auth | None) -> str:
    """Non-secret digest of the auth material, used to scope cache entries per tenant."""
    if auth is None:
        return ""
    material = json.dumps(auth.request_kwargs(), sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()


__all__ = ["DEFAULT_RETRIES", "RETRYABLE_HTTP_STATUS", "apply_auth", "build_session", "credential_identity"]
