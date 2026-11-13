# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read application logs (pod stdout) from a Kubernetes cluster through the
Kubernetes API.

The same job as ``kubectl_logs``, without the ``kubectl`` binary: it reads
each pod's logs with the Python ``kubernetes`` client the Blueprint already
uses for discovery, so it works wherever the Blueprint runs, including inside a
cluster with a service account and no extra tools in the image. It emits the
same Observations as ``kubectl_logs`` for the same logs.

Pairs with the Kueue and KAI execution-context adapters: they stamp a list of
pod names onto each context (``pod_names``), and this connector turns each pod
into one APPLOG Observation. A pod that is gone (deleted, or garbage-collected
before the query) is a per-pod warn-and-skip, as with ``kubectl_logs``.

The log is read per container, init containers first, and joined in that order
with no separator, which is what ``kubectl logs --all-containers`` prints.

Needs ``get`` on ``pods`` and ``pods/log`` in the workload's namespace.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timezone

from loguru import logger

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.logs import log_observation
from attribution_blueprint.connectors.logs.kubectl_logs import (
    _ENTRIES_FOR_SOURCE,
    DEFAULT_MAX_BYTES_PER_POD,
    DEFAULT_TIMEOUT_SECONDS,
    _parse_size,
    _tail_bytes,
)
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.kubernetes.client import KubernetesClient, KubernetesClientError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.types import ExecutionContext, Observation, Source, assert_aware


def _container_names(pod: dict) -> list[str]:
    """Every container in log order: init, regular, then ephemeral."""
    spec = pod.get("spec") or {}
    names: list[str] = []
    for key in ("initContainers", "containers", "ephemeralContainers"):
        for container in spec.get(key) or []:
            name = container.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _positive_size(value: int | str | None) -> int | None:
    """A byte cap from an int or a size string such as "16MB"; None means no cap.

    Raises `ValueError` for an unparseable or non-positive value. Zero would
    keep nothing here but the whole log under `kubectl_logs`, and a negative cap
    has no meaning, so neither is accepted.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"size must be an integer or a size string, got {value!r}")
    try:
        size = _parse_size(value)
    except TypeError as error:
        raise ValueError(str(error)) from error
    if size <= 0:
        raise ValueError(f"size must be positive, got {value!r}")
    return size


class KubernetesLogsConnector:
    """Collector that emits one APPLOG (or SYSLOG) Observation per pod, read
    through the Kubernetes API.

    Reads ``context["namespace"]`` and ``context["pod_names"]``;
    ``context["pod_nodes"]`` (pod -> node) stamps each entry with the node it
    came from. A pod listed there but missing, or one this identity may not
    read, produces a warning and is skipped.
    """

    def __init__(
        self,
        *,
        client: KubernetesClient | None = None,
        kubeconfig: str | None = None,
        context: str | None = None,
        container: str | None = None,
        timestamps: bool = True,
        previous: bool = False,
        since_context_start: bool = False,
        max_bytes_per_pod: int | str | None = DEFAULT_MAX_BYTES_PER_POD,
        source: Source = Source.APPLOG,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        now: Callable[[], datetime] | None = None,
    ):
        """Configure the connector.

        Args:
            client: API client to read through; built from `kubeconfig`, `context`
                and `timeout_seconds` when omitted.
            kubeconfig: Optional kubeconfig path.
            context: Optional kubeconfig context name.
            container: One container to read; every container when omitted.
            timestamps: Ask the API to prefix each line with a timestamp.
            previous: Read the previous container instance.
            since_context_start: Read only from the context's start.
            max_bytes_per_pod: Trailing bytes to keep per container and per pod,
                as an int or a size string; `None` keeps everything. Must be
                positive.
            source: Observation source to emit.
            timeout_seconds: Timeout for each API request.
            now: Clock used to measure the since-start window; for tests.

        Raises:
            ValueError: If `max_bytes_per_pod` is unparseable or not positive.
        """
        self._client = client or KubernetesClient(
            kubeconfig=kubeconfig, context=context, timeout_seconds=timeout_seconds
        )
        self._container = container
        self._timestamps = timestamps
        self._previous = previous
        self._since_context_start = since_context_start
        self._max_bytes = _positive_size(max_bytes_per_pod)
        self._source = source
        self._now = now or (lambda: datetime.now(timezone.utc))

    @property
    def source(self) -> Source:
        """The observation source this connector emits."""
        return self._source

    # --- public protocol ---------------------------------------------------

    def collect(self, context: ExecutionContext) -> list[Observation]:
        """One Observation per pod in `context["pod_names"]` that has readable logs."""
        # Naive datetimes would break the since-seconds arithmetic, and the
        # service rejects naive observations anyway. Fail at the boundary,
        # same discipline as the other log connectors.
        assert_aware(context["start_time"], "start_time")
        assert_aware(context["end_time"], "end_time")

        pod_names = context.get("pod_names") or []
        namespace = context.get("namespace") or ""
        if not pod_names:
            return []
        if not namespace:
            logger.warning(
                f"kubernetes_logs: context {context.get('id')} has no namespace; "
                f"refusing to read pod logs without one (would target the default namespace)"
            )
            return []

        pod_nodes: dict[str, str] = context.get("pod_nodes") or {}
        observations: list[Observation] = []
        for pod in pod_names:
            body = self._read_pod(pod, namespace, context)
            if not body:
                continue
            if self._max_bytes is not None:
                body = _tail_bytes(body, self._max_bytes)

            entries = _ENTRIES_FOR_SOURCE[self._source](body, context, pod_nodes.get(pod))
            observations.append(log_observation(source=self._source, entries=entries, context=context))
        return observations

    # --- private API plumbing ---------------------------------------------

    def _since_seconds(self, context: ExecutionContext) -> int:
        """Seconds back from now to the workload's start. The log API takes a
        relative time only and measures it from each request, so this is
        evaluated for every read, never once per collection: a slow earlier read
        would otherwise shift the window forward for the pods after it. It is
        never below 1, because 0 is not a valid window."""
        elapsed = (self._now() - context["start_time"]).total_seconds()
        return max(1, math.ceil(elapsed))

    def _containers(self, pod: str, namespace: str) -> list[str]:
        """The containers to read: the configured one, else every container of the pod."""
        if self._container:
            return [self._container]
        manifest = self._client.get_one("Pod", pod, namespace=namespace, api_version="v1")
        return _container_names(manifest)

    def _read_pod(self, pod: str, namespace: str, context: ExecutionContext) -> str:
        """One pod's log, its containers' logs joined in order, or "" when the pod
        cannot be read. A container that fails is skipped without losing the rest.

        With a cap, the joined log is trimmed to its tail after each container, so
        memory stays near twice the cap however many containers the pod has."""
        try:
            containers = self._containers(pod, namespace)
        except KubernetesClientError as error:
            logger.warning(f"kubernetes_logs: could not read pod {pod} in {namespace}: {error}")
            return ""
        if not containers:
            logger.warning(f"kubernetes_logs: pod {pod} in {namespace} lists no containers; skipping")
            return ""

        joined = ""
        for container in containers:
            try:
                part = self._client.read_pod_log(
                    pod,
                    namespace=namespace,
                    container=container,
                    previous=self._previous,
                    timestamps=self._timestamps,
                    since_seconds=self._since_seconds(context) if self._since_context_start else None,
                    max_bytes=self._max_bytes,
                )
                joined += part
                if self._max_bytes is not None:
                    joined = _tail_bytes(joined, self._max_bytes)
            except KubernetesClientError as error:
                # One container that never started (HTTP 400) or has no
                # previous instance must not lose the others' logs.
                logger.warning(f"kubernetes_logs: could not read {namespace}/{pod} container {container}: {error}")
        return joined


# --- MCA registration -----------------------------------------------------


def _build(spec: dict, ctx: dict) -> KubernetesLogsConnector:
    """Build the connector from its configuration spec, naming the field in every error."""
    from attribution_blueprint.mca import _coerce_int

    source: Source = ctx["source"]

    def _opt_bool(key: str, default: bool) -> bool:
        """A boolean option, accepting TOML booleans and the strings true/false/yes/no/1/0."""
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
        raise MCAError(f"log.{source.name.lower()}.kubernetes_logs.{key} must be a bool, got {v!r}")

    timeout = spec.get("timeout_seconds")
    if timeout is None:
        timeout = DEFAULT_TIMEOUT_SECONDS
    else:
        timeout = _coerce_int(timeout, "log.kubernetes_logs.timeout_seconds")
        if timeout <= 0:
            raise MCAError(f"log.kubernetes_logs.timeout_seconds must be positive, got {timeout}")

    try:
        max_bytes = _positive_size(spec.get("max_bytes_per_pod", DEFAULT_MAX_BYTES_PER_POD))
    except ValueError as error:
        raise MCAError(f"log.{source.name.lower()}.kubernetes_logs.max_bytes_per_pod: {error}") from error

    return KubernetesLogsConnector(
        kubeconfig=spec.get("kubeconfig"),
        context=spec.get("context"),
        container=spec.get("container"),
        timestamps=_opt_bool("timestamps", True),
        previous=_opt_bool("previous", False),
        since_context_start=_opt_bool("since_context_start", False),
        max_bytes_per_pod=max_bytes,
        source=source,
        timeout_seconds=timeout,
    )


_registry.register_component(
    framework="log",
    name="kubernetes_logs",
    schema={
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


__all__ = ["KubernetesLogsConnector"]
