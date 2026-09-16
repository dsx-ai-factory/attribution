# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

from loguru import logger

from attribution_lib.nodename_utils import is_ipv4, is_ipv6

if TYPE_CHECKING:
    pass

import attribution_lib
from attribution_lib.exceptions import UnawareDatetimeError

_hostname_mapping_overlay: ContextVar[Mapping[str, str] | None] = ContextVar(
    "attribution_hostname_mapping_overlay",
    default=None,
)


def check_datetime(dt: datetime) -> datetime:
    if not aware_datetime(dt):
        raise UnawareDatetimeError()
    return dt


def aware_datetime(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


def convert_hostname_to_short_format(hostname: str) -> str:
    """Convert long hostnames to short format.

    Examples:
        host-a1b2c3d4.example.internal -> host-a1b2c3d4
        simple-hostname -> simple-hostname
        192.168.1.1 -> 192.168.1.1 (IPv4 addresses are returned unchanged)
        2001:db8::1 -> 2001:db8::1 (IPv6 addresses are returned unchanged)
    """
    # First check if it's an IP address (IPv4 or IPv6) - if so, return as-is
    if is_ipv4(hostname) or is_ipv6(hostname):
        return hostname

    # Split by dots and take the first part for hostnames
    # example: host-a1b2c3d4.example.internal -> host-a1b2c3d4
    if "." in hostname:
        return hostname.split(".")[0]
    return hostname


def matches_ignored_hostname_prefix(peer: str, prefixes: Optional[Iterable[str]] = None) -> bool:
    """Return True if *peer* should be skipped by ignored hostname prefixes.

    Hostnames are compared in short form (first DNS label), so
    ``scanner-001.example.com`` matches prefix ``scanner-``. IP addresses are
    compared in full; they are not split on dots.

    Args:
        peer: Hostname or IP after node resolution, or the raw peer string
            from a log line.
        prefixes: Prefixes to test. Defaults to
            ``attribution_lib.ignored_hostname_prefixes``.

    Returns:
        True if the peer matches an ignored prefix and should be excluded.
    """
    if prefixes is None:
        prefixes = attribution_lib.ignored_hostname_prefixes
    if not prefixes:
        return False
    identity = convert_hostname_to_short_format(peer)
    return any(identity.startswith(prefix) for prefix in prefixes)


@contextmanager
def hostname_mapping(mapping: Mapping[str, str]) -> Iterator[None]:
    """Provide an I/O-free hostname mapping scoped to the current execution context.

    The overlay takes precedence over the process-wide mapping configured by
    :func:`attribution_lib.core.setup_rules`. It is copied on entry and fully
    replaced by nested contexts, so concurrent jobs cannot observe or mutate
    one another's mappings.

    Args:
        mapping: Node identifier overrides scoped to attribution performed
            inside this context.

    Yields:
        None. The previous context is restored on exit, including exceptions.
    """
    token = _hostname_mapping_overlay.set(dict(mapping))
    try:
        yield
    finally:
        _hostname_mapping_overlay.reset(token)


def get_hostname(ip_or_hostname: str) -> str:
    """Resolve a node identifier without consulting network services.

    IP addresses are resolved only through the node mapping configured by the
    caller. Unmapped IP addresses are returned unchanged. Hostnames are already
    node identifiers, so they are normalized to their short form without a
    forward or reverse DNS lookup.

    Args:
        ip_or_hostname: Either an IP address (e.g. "192.168.1.1")
                       or a hostname (e.g. "host-a1b2c3d4.example.internal")
    Returns:
        A mapped node name, normalized hostname, or the original unmapped IP.
    """
    if is_ipv4(ip_or_hostname) or is_ipv6(ip_or_hostname):
        overlay = _hostname_mapping_overlay.get()
        if overlay is not None and ip_or_hostname in overlay:
            return overlay[ip_or_hostname]
        return attribution_lib.ip_to_nodenames.get(ip_or_hostname, ip_or_hostname)
    return convert_hostname_to_short_format(ip_or_hostname)


RANK_REGEX = re.compile(r"^\[rank\s*:?\s*(?P<rank>\d+)\]")


def applog_extract_node_rank(line: str) -> tuple[Optional[str], Optional[int]]:
    """Extract & return node name and rank from line. Return both, filling in None for those we could not identify"""
    rank = None
    node = None
    if (
        line.startswith("terminate called")
        or line.startswith("wandb")
        or "Writing temporary files back into original files" in line
        # TODO: remove "Writing temporary files back into original files"
    ):
        # special case when the error is not associated with a node in log file, e.g.
        #
        # terminate called after throwing an instance of 'c10::DistBackendError'
        #   what():  [PG 13 Rank 6] NCCL watchdog thread terminated with exception: CUDA error: device-side assert triggered
        # CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
        #
        rank = 0
    elif line.startswith("[rank"):
        # extract 4 from
        # [rank4]:   File \"/usr/lib/python3.10/runpy.py\", line 196, in _run_module_as_main\n",
        # rank = int(line.split(":")[0].replace("[rank", "").replace("]", ""))
        if (match := RANK_REGEX.match(line)) is not None:
            rank = int(match.group("rank"))
    else:
        rank_or_node = line.split(":")[0].strip()
        try:
            if " " in rank_or_node:
                # if it's not a valid node name
                rank = 0
            else:
                rank = int(rank_or_node)
        except ValueError:
            # node name
            node = rank_or_node
    return node, rank


SYSLOG_PATTERN = re.compile(  # this pattern came from ADI code
    r"(?P<datetime>"
    r"(?P<month>\w{3})\s+"
    r"(?P<day>\d{1,2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r")\s+"
    r"(?P<hostname>[\w.-]+)\s+"
    r"(?P<program>[\w()./-]+)(\[(?P<pid>\d+)\])?:\s+"
    r"(?P<message>.*)"
)


def syslog_extract_node(line: str) -> Optional[str]:
    """Extract & return node name from line, or return None."""
    if (match := SYSLOG_PATTERN.match(line)) is not None:
        log_data = match.groupdict()
        return log_data.get("hostname")
    return None


def method_cache(func: Callable[..., Any]) -> Callable[..., Any]:
    """This is an unusual (dangerous?) cacheing method that ignores the args passed in; it's intended to be used on immutable items/args. The first time a method is called on an instance, it's kept for the lifetime for that instance"""

    cache_name = f"_{getattr(func, '__name__', 'unknown')}_cache"

    def wrapper(self: Any, *args: Any) -> Any:
        if not hasattr(self, cache_name):
            setattr(self, cache_name, func(self, *args))
        cache = getattr(self, cache_name)
        return cache

    return wrapper


def passthrough(*args: Any, **kwargs: Any) -> None:
    pass


def add_hook_to_method(cls: type, method_name: str, hook_func: Callable[..., Any]) -> None:
    if hasattr(cls, method_name):
        orig_method = getattr(cls, method_name)
    else:
        orig_method = passthrough

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        hook_func(self, *args, **kwargs)
        return orig_method(self, *args, **kwargs)

    setattr(cls, method_name, wrapped)


class ObserverHook(NamedTuple):
    cls: Any
    method_name: str
    func: Callable[..., None]


def setup_observer_hooks(hooks: Iterable[ObserverHook]) -> None:
    for hook in hooks:
        add_hook_to_method(hook.cls, hook.method_name, hook.func)


def _format_discard_log(removed: set[Any], before: set[Any], after: set[Any], context: dict[str, Any]) -> str:
    msg = f"Symptoms Discarded: {list(removed)}"
    context_fields = [
        ("Node Name", context.get("node_name")),
        ("Rule", context.get("rule")),
        ("Job ID", context.get("job_id")),
        ("Discard Reason", context.get("discard_reason")),
    ]
    for label, value in context_fields:
        if value:
            msg += f" | {label}: {value}"
    return msg


def log_discard_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper for the discard method of SymptomSet.
    Logs every discard action, with:
    1. the symptoms actually removed (computed from before/after),
    2. optional context (node_name, rule, job_id, discard_reason),
    3. the before/after state of the set (useful for debugging).
    Ensures all discard actions are traceable for debugging and analysis.
    """

    def wrapped(self: Any, symptom: Any, *args: Any, **kwargs: Any) -> Any:
        before = set(self.symptoms())
        result = func(self, symptom, *args, **kwargs)
        after = set(self.symptoms())
        context = kwargs.get("context", {})
        removed = before - after
        if removed:
            msg = _format_discard_log(removed, before, after, context)
            logger.debug(msg)
            # Also emit via stdlib logging so pytest caplog can capture it.
            logging.getLogger(__name__).debug(msg)
        return result

    return wrapped


def normalize_pci_id(pci_id: str) -> str:
    """Normalize PCI ID to a consistent format for comparison
    Args:
        pci_id: PCI ID in two encountered formats:
        - Syslog: "0000:c2:00" (domain:bus:device)
        - Four-field PCI address: "00000000:C2:00.0" (domain:bus:device.function)
    Returns:
        Normalized PCI ID in format "domain:bus:device" (e.g., "0000:c2:00")
    """
    normalized = pci_id.lower().strip()
    parts = normalized.split(":")

    if len(parts) == 2:
        return f"0000:{normalized}"
    elif len(parts) == 3:
        domain, bus, device_func = parts
        domain = domain.zfill(4)[-4:]  # Standardize to 4-digit domain
        device = device_func.split(".")[0]  # Remove function suffix
        return f"{domain}:{bus}:{device}"
    else:
        return normalized


def is_event_at_end_of_job(event_time: datetime, job_end_time: datetime) -> bool:
    """Check if an event occurred near the end of a job.

    Checks if the absolute time difference between the event and job end time
    is within 780 seconds (600 seconds watchdog timeout + 180 seconds buffer).

    Args:
        event_time: When the event occurred
        job_end_time: When the job ended

    Returns:
        True if the time difference between job_end_time and event_time
        is <= 780 seconds (watchdog_timeout + buffer_time), False otherwise
    """

    watchdog_timeout = 600  # default watchdog timeout in seconds
    buffer_time = 180  # buffer time in seconds when slurm cancelled the job after watchdog timeout
    return (job_end_time - event_time).total_seconds() <= (watchdog_timeout + buffer_time)


def extract_node_name_from_hostname(hostname: str) -> str:
    """Extract a node name by removing a versioned cluster prefix.

    - "cluster-m3n4o5p6-v3-host-q7r8s9t0" -> "host-q7r8s9t0"
    - "host-q7r8s9t0" -> "host-q7r8s9t0" (no change if no prefix)
    """
    match = re.search(r"-v\d+-(.+)$", hostname)
    return match.group(1) if match else hostname
