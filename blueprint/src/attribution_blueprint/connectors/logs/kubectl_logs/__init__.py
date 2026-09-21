# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Read application logs (pod stdout) from a Kubernetes cluster by shelling
to ``kubectl logs``.

Pairs with the kueue ExecutionContextAdapter: that adapter stamps a list of
pod names onto each context (`pod_names`), and this connector turns each
pod into one APPLOG Observation. The connector deliberately does NOT
re-derive the per-owner pod selector — the ECA already knows whether a
Workload is owned by a batch/v1 Job or a JobSet and chose the right label;
re-doing that resolution here would duplicate logic and add round-trips.

Kueue's own pod stdout retention is bounded by the container runtime
(typically ``/var/log/containers/*.log`` rotated to a few hundred MiB) and
by whether the pod itself still exists — once Kueue preemption, JobSet
retention, or admin ``kubectl delete pod`` removes the pod object,
``kubectl logs`` returns ``NotFound``. The connector treats that case as a
per-pod warn-and-skip; the same pod-GC limitation documented for the kueue
ECA's node attribution applies here.
"""

from __future__ import annotations

import re
import subprocess
from datetime import timezone

from loguru import logger

from attribution_blueprint.connectors.logs import entries_from_text, log_observation
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.types import (
    ExecutionContext,
    Format,
    Observation,
    Source,
    assert_aware,
)

_FORMAT_FOR_SOURCE: dict[Source, Format] = {
    Source.APPLOG: Format.ENTRY_TIMESERIES,
    Source.SYSLOG: Format.RAW_LOG,
}

DEFAULT_MAX_BYTES_PER_POD = 16 * 1024 * 1024  # 16 MiB — caps a single pod
DEFAULT_TIMEOUT_SECONDS = 60

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")
_SIZE_SUFFIXES: dict[str, int] = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
}


def _parse_size(value: int | str) -> int:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TypeError(f"size must be int or str, got {type(value).__name__}")
    m = _SIZE_RE.match(value)
    if not m:
        raise ValueError(f"unparseable size: {value!r}")
    number, suffix = m.group(1), m.group(2).lower()
    if suffix not in _SIZE_SUFFIXES:
        raise ValueError(f"unknown size suffix {suffix!r} in {value!r}")
    return int(float(number) * _SIZE_SUFFIXES[suffix])


def _tail_bytes(body: str, cap: int) -> str:
    """Keep only the trailing `cap` bytes of `body`. Crash symptoms
    (CUDA errors, NCCL aborts, OOM kills) typically land near the end of
    a training log, so tailing is the right truncation direction.
    """
    raw = body.encode("utf-8")
    if len(raw) <= cap:
        return body
    return raw[-cap:].decode("utf-8", errors="replace")


class KubectlLogsConnector:
    """Collector that emits one APPLOG (or SYSLOG) Observation per pod by
    shelling ``kubectl logs``.

    Reads ``context["namespace"]`` and ``context["pod_names"]`` (a list
    populated by the kueue ECA). Pods listed there but missing from the
    cluster (e.g. gc'd before query) produce a warning and are skipped.
    """

    def __init__(
        self,
        *,
        kubectl_path: str = "kubectl",
        kubeconfig: str | None = None,
        context: str | None = None,
        container: str | None = None,
        timestamps: bool = True,
        previous: bool = False,
        since_context_start: bool = False,
        max_bytes_per_pod: int | str | None = DEFAULT_MAX_BYTES_PER_POD,
        source: Source = Source.APPLOG,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self._kubectl_path = kubectl_path
        self._kubeconfig = kubeconfig
        self._context = context
        self._container = container
        self._timestamps = timestamps
        self._previous = previous
        self._since_context_start = since_context_start
        self._max_bytes = _parse_size(max_bytes_per_pod) if max_bytes_per_pod is not None else None
        self._source = source
        self._timeout_seconds = timeout_seconds

    @property
    def source(self) -> Source:
        return self._source

    # --- public protocol ---------------------------------------------------

    def collect(self, context: ExecutionContext) -> list[Observation]:
        # Naive datetimes would break --since-time rendering, and the
        # service rejects naive observations anyway. Fail at the
        # boundary, same discipline as the local_file connector.
        assert_aware(context["start_time"], "start_time")
        assert_aware(context["end_time"], "end_time")

        pod_names = context.get("pod_names") or []
        namespace = context.get("namespace") or ""
        if not pod_names:
            return []
        if not namespace:
            logger.warning(
                f"kubectl_logs: context {context.get('id')} has no namespace; "
                f"refusing to issue `kubectl logs` without -n (would target the default namespace)"
            )
            return []

        observations: list[Observation] = []
        for pod in pod_names:
            body = self._read_pod_logs(pod, namespace, context)
            if not body:
                continue
            if self._max_bytes is not None:
                body = _tail_bytes(body, self._max_bytes)
            observations.append(self._observation(body, context))
        return observations

    def _observation(self, body: str, context: ExecutionContext) -> Observation:
        fmt = _FORMAT_FOR_SOURCE[self._source]
        if fmt == Format.ENTRY_TIMESERIES:
            return log_observation(source=self._source, entries=entries_from_text(body), context=context)
        return log_observation(source=self._source, format=fmt, body=body, context=context)

    # --- private kubectl plumbing -----------------------------------------

    def _build_argv(self, pod: str, namespace: str, context: ExecutionContext) -> list[str]:
        argv: list[str] = [self._kubectl_path]
        if self._kubeconfig:
            argv += ["--kubeconfig", self._kubeconfig]
        if self._context:
            argv += ["--context", self._context]
        argv += ["logs", pod, "-n", namespace]
        if self._container:
            argv += ["-c", self._container]
        else:
            argv += ["--all-containers=true"]
        if self._timestamps:
            argv += ["--timestamps=true"]
        if self._previous:
            argv += ["--previous"]
        if self._since_context_start:
            # kubectl rejects fractional seconds on --since-time; serialize
            # to whole-second RFC3339 with a trailing Z.
            start = context["start_time"].astimezone(timezone.utc).replace(microsecond=0)
            argv += [f"--since-time={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"]
        return argv

    def _read_pod_logs(self, pod: str, namespace: str, context: ExecutionContext) -> str:
        argv = self._build_argv(pod, namespace, context)
        try:
            result = subprocess.run(
                argv,
                check=True,
                text=True,
                capture_output=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"kubectl_logs: `kubectl logs {pod} -n {namespace}` exited {e.returncode}: "
                f"{(e.stderr or '').strip() or '<no stderr>'}"
            )
            return ""
        except subprocess.TimeoutExpired:
            logger.warning(
                f"kubectl_logs: `kubectl logs {pod} -n {namespace}` timed out after {self._timeout_seconds}s"
            )
            return ""
        return result.stdout or ""


# --- MCA registration -----------------------------------------------------

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.mca_registry import _PARAM


def _build(spec: dict, ctx: dict) -> KubectlLogsConnector:
    from attribution_blueprint.mca import _coerce_int

    source: Source = ctx["source"]

    def _opt_bool(key: str, default: bool) -> bool:
        if key not in spec:
            return default
        v = spec[key]
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.lower() in ("true", "yes", "1"):
                return True
            if v.lower() in ("false", "no", "0", ""):
                return False
        raise MCAError(f"log.{source.name.lower()}.kubectl_logs.{key} must be a bool, got {v!r}")

    timeout = spec.get("timeout_seconds")
    if timeout is None:
        timeout = DEFAULT_TIMEOUT_SECONDS
    else:
        timeout = _coerce_int(timeout, "log.kubectl_logs.timeout_seconds")
        if timeout <= 0:
            raise MCAError(f"log.kubectl_logs.timeout_seconds must be positive, got {timeout}")

    return KubectlLogsConnector(
        kubectl_path=spec.get("kubectl_path", "kubectl"),
        kubeconfig=spec.get("kubeconfig"),
        context=spec.get("context"),
        container=spec.get("container"),
        timestamps=_opt_bool("timestamps", True),
        previous=_opt_bool("previous", False),
        since_context_start=_opt_bool("since_context_start", False),
        max_bytes_per_pod=spec.get("max_bytes_per_pod", DEFAULT_MAX_BYTES_PER_POD),
        source=source,
        timeout_seconds=timeout,
    )


_registry.register_component(
    framework="log",
    name="kubectl_logs",
    schema={
        "kubectl_path": _PARAM,
        "kubeconfig": _PARAM,
        "context": _PARAM,
        "container": _PARAM,
        "timestamps": _PARAM,
        "previous": _PARAM,
        "since_context_start": _PARAM,
        "max_bytes_per_pod": _PARAM,
        "timeout_seconds": _PARAM,
    },
    build=_build,
)


__all__ = ["KubectlLogsConnector", "DEFAULT_MAX_BYTES_PER_POD"]
