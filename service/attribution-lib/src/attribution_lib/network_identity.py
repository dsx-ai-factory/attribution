# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read logged source hostnames and GID mappings without checking allocation."""

import re
from dataclasses import dataclass
from ipaddress import ip_address

from attribution_lib.utils import NCCL_HOST_PID_TID_REGEX, get_hostname

_LOCAL_GID = re.compile(r"\blocalGid\s+(?P<ip>[^\s,]+)")
_HOST_PREFIX = re.compile(r"^(?P<node>[A-Za-z][A-Za-z0-9_.-]*):\s")


def normalize_network_hostname(value: str) -> str | None:
    """Return a lowercase short hostname, excluding unresolved IP addresses.

    Args:
        value: An observed or mapped endpoint name.
    Returns:
        A normalized hostname, or None for an address or invalid name.
    """
    try:
        ip_address(value)
        return None
    except ValueError:
        name = value.rstrip(".").split(".", 1)[0].lower()
        return name if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) else None


@dataclass(frozen=True)
class NetworkSourceIdentity:
    """Keep the logged source and local-GID mapping, including disagreement."""

    log_hostname: str | None
    local_ip: str | None
    mapped_hostname: str | None

    @property
    def conflict(self) -> bool:
        return (
            self.log_hostname is not None
            and self.mapped_hostname is not None
            and self.log_hostname != self.mapped_hostname
        )

    @property
    def hostname(self) -> str | None:
        """Prefer the explicit emitter; a differing GID mapping remains evidence."""
        return self.log_hostname if self.log_hostname is not None else self.mapped_hostname


def network_source_identity(line: str, *, prefix_hostname: str | None = None) -> NetworkSourceIdentity:
    """Collect explicit NCCL source and local-GID mapping evidence without I/O.

    Rank and allocation are deliberately absent. A CUDA device index or an
    implicit rank zero cannot identify the endpoint. Mapping uses the caller's
    existing scoped snapshot; the peer address never identifies the source.

    Args:
        line: One network application-log line.
        prefix_hostname: A caller-verified hostname, accepted only if it is
            explicitly present as the leading hostname prefix. This never
            substitutes a node obtained from an implicit rank zero.
    Returns:
        Source evidence retaining both names when they disagree.
    """
    match = NCCL_HOST_PID_TID_REGEX.search(line)
    observed = normalize_network_hostname(match.group("node")) if match is not None else None
    prefix = _HOST_PREFIX.match(line)
    if observed is None and prefix is not None and prefix_hostname is not None:
        candidate = normalize_network_hostname(prefix.group("node"))
        if candidate == normalize_network_hostname(prefix_hostname):
            observed = candidate
    gid = _LOCAL_GID.search(line)
    local_ip = None
    mapped = None
    if gid is not None:
        try:
            address = ip_address(gid.group("ip"))
            local_ip = str(getattr(address, "ipv4_mapped", None) or address)
        except ValueError:
            pass
        if local_ip is not None:
            mapped = normalize_network_hostname(get_hostname(local_ip))
    return NetworkSourceIdentity(observed, local_ip, mapped)
