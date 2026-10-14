# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Protocol

from attribution_blueprint.kubernetes.daemon.resolvers import normalize_resource

_ATTEMPT_METADATA_KEYS = frozenset(
    {
        "batch.kubernetes.io/job-completion-index",
        "jobset.sigs.k8s.io/job-index",
        "jobset.sigs.k8s.io/replicatedjob-name",
        "jobset.sigs.k8s.io/restart-attempt",
    }
)


class ResourceWatchSource(Protocol):
    name: str
    api_version: str
    kind: str

    def list_resources(self) -> tuple[list[dict], str]: ...

    def stream(self, resource_version: str, timeout_seconds: int) -> Iterable[dict]: ...

    def selected_status(self, resource: dict) -> dict: ...


@dataclass(frozen=True)
class KubernetesResourceWatchSource:
    """A typed list/watch source with kind-specific status selection."""

    name: str
    api_version: str
    kind: str
    list_callable: Callable
    list_kwargs: dict

    def list_resources(self) -> tuple[list[dict], str]:
        listing = self.list_callable(**self.list_kwargs)
        if isinstance(listing, dict):
            metadata = listing.get("metadata") or {}
            items = listing.get("items") or []
            resource_version = metadata.get("resourceVersion")
        else:
            metadata = listing.metadata
            items = listing.items
            resource_version = metadata.resource_version
        if not resource_version:
            raise ValueError(f"{self.name} list response has no resourceVersion")
        return [self._normalize(item) for item in items], str(resource_version)

    def stream(self, resource_version: str, timeout_seconds: int) -> Iterable[dict]:
        from kubernetes.watch import Watch

        watcher = Watch()
        for event in watcher.stream(
            self.list_callable,
            **self.list_kwargs,
            resource_version=resource_version,
            timeout_seconds=timeout_seconds,
            allow_watch_bookmarks=True,
        ):
            event_type = event.get("type")
            yield {
                "type": event_type,
                "object": self._normalize(event.get("object")),
            }

    def selected_status(self, resource: dict) -> dict:
        return selected_status(self.kind, resource)

    def _normalize(self, resource) -> dict:
        if isinstance(resource, dict):
            normalized = resource
        else:
            normalized = normalize_resource(resource.to_dict())
        return {
            **normalized,
            "apiVersion": normalized.get("apiVersion") or self.api_version,
            "kind": normalized.get("kind") or self.kind,
        }


def pod_watch_source(core_api, namespace: str) -> KubernetesResourceWatchSource:
    return KubernetesResourceWatchSource(
        name="pods",
        api_version="v1",
        kind="Pod",
        list_callable=core_api.list_namespaced_pod,
        list_kwargs={"namespace": namespace},
    )


def job_watch_source(batch_api, namespace: str) -> KubernetesResourceWatchSource:
    return KubernetesResourceWatchSource(
        name="jobs",
        api_version="batch/v1",
        kind="Job",
        list_callable=batch_api.list_namespaced_job,
        list_kwargs={"namespace": namespace},
    )


def custom_watch_source(
    custom_api,
    *,
    name: str,
    group: str,
    version: str,
    namespace: str,
    plural: str,
    kind: str,
) -> KubernetesResourceWatchSource:
    return KubernetesResourceWatchSource(
        name=name,
        api_version=f"{group}/{version}",
        kind=kind,
        list_callable=partial(
            custom_api.list_namespaced_custom_object,
            group,
            version,
            namespace,
            plural,
        ),
        list_kwargs={},
    )


def selected_status(kind: str, resource: dict) -> dict:
    status = resource.get("status") or {}
    if kind == "Pod":
        return _pod_status(status)
    if kind == "Job":
        selected = _select(
            status,
            "active",
            "completedIndexes",
            "completionTime",
            "conditions",
            "failed",
            "failedIndexes",
            "startTime",
            "succeeded",
        )
        return _normalize_conditions(selected)
    if kind == "JobSet":
        selected = _select(
            status,
            "conditions",
            "replicatedJobsStatus",
            "restarts",
            "terminalState",
        )
        return _normalize_conditions(selected)
    if kind == "Workload":
        selected = _select(
            status,
            "admission",
            "conditions",
            "requeueState",
        )
        return _normalize_conditions(selected)
    if kind == "PodGroup":
        selected = _select(
            status,
            "conditions",
            "failed",
            "pending",
            "phase",
            "resourcesStatus",
            "running",
            "schedulingConditions",
            "succeeded",
        )
        selected = _normalize_conditions(selected)
        last_start = ((resource.get("metadata") or {}).get("annotations") or {}).get(
            "kai.scheduler/last-start-timestamp"
        )
        if last_start:
            selected["lastStartTimestamp"] = last_start
        return selected
    return {}


def selected_metadata(resource: dict) -> dict | None:
    metadata = resource.get("metadata") or {}
    selected = {}
    for field in ("labels", "annotations"):
        values = metadata.get(field) or {}
        retained = {key: values[key] for key in sorted(_ATTEMPT_METADATA_KEYS & values.keys())}
        if retained:
            selected[field] = retained
    return selected or None


def source_timestamp(status: dict) -> str | None:
    timestamps: list[str] = []

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and (
                    key.endswith("Time")
                    or key.endswith("Timestamp")
                    or key in {"startedAt", "finishedAt"}
                ):
                    timestamps.append(item)
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(status)
    return max(timestamps) if timestamps else None


def _pod_status(status: dict) -> dict:
    selected = _select(
        status,
        "conditions",
        "containerStatuses",
        "ephemeralContainerStatuses",
        "initContainerStatuses",
        "message",
        "phase",
        "reason",
        "startTime",
    )
    selected = _normalize_conditions(selected)
    for key in ("containerStatuses", "initContainerStatuses", "ephemeralContainerStatuses"):
        if key in selected:
            selected[key] = [_container_status(item) for item in selected[key]]
    return selected


def _container_status(status: dict) -> dict:
    selected = _select(
        status,
        "containerID",
        "lastState",
        "name",
        "ready",
        "restartCount",
        "started",
        "state",
    )
    for field in ("state", "lastState"):
        if field in selected:
            selected[field] = _container_state(selected[field])
    return selected


def _normalize_conditions(selected: dict) -> dict:
    if "conditions" not in selected:
        return selected
    return {
        **selected,
        "conditions": [
            _select(
                condition,
                "type",
                "status",
                "reason",
                "message",
                "observedGeneration",
                "transitionID",
                "lastTransitionTime",
            )
            for condition in selected["conditions"]
        ],
    }


def _container_state(state: dict) -> dict:
    selected = {}
    running = state.get("running")
    if running:
        selected["running"] = _select(running, "startedAt")
    terminated = state.get("terminated")
    if terminated:
        selected["terminated"] = _select(
            terminated,
            "containerID",
            "exitCode",
            "finishedAt",
            "message",
            "reason",
            "signal",
            "startedAt",
        )
    waiting = state.get("waiting")
    if waiting:
        selected["waiting"] = _select(waiting, "message", "reason")
    return selected


def _select(value: dict, *fields: str) -> dict:
    selected = {}
    for field in fields:
        snake_case = _snake_case(field)
        if field in value:
            item = value[field]
        elif snake_case in value:
            item = value[snake_case]
        else:
            continue
        if item is not None:
            selected[field] = item
    return selected


def _snake_case(value: str) -> str:
    words = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words).lower()
