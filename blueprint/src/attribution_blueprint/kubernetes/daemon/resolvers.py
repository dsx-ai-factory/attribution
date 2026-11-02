# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol

from attribution_blueprint.kubernetes.daemon.models import ResourceIdentity


class ResolutionPending(RuntimeError):
    pass


class UnsupportedOwner(RuntimeError):
    pass


class StaleOwnerReference(RuntimeError):
    """A name lookup returned an object with a different UID."""


class OwnershipResolver(Protocol):
    def resolve(self, pod: dict) -> ResourceIdentity: ...


class AssociatedResourceResolver(Protocol):
    def resolve(self, pod: dict, owner: ResourceIdentity) -> tuple[ResourceIdentity, ...]: ...


class KubernetesOwnershipResolver:
    """Resolve a Pod's UID-verified Job or JobSet owner without a scheduler."""

    def __init__(self, batch_api, custom_api, *, jobset_version: str = "v1alpha2"):
        self.batch_api = batch_api
        self.custom_api = custom_api
        self.jobset_version = jobset_version

    def resolve(self, pod: dict) -> ResourceIdentity:
        from kubernetes.client.exceptions import ApiException

        try:
            return self._resolve(pod)
        except ApiException as error:
            if error.status == 404:
                raise ResolutionPending("owner is not visible yet") from error
            raise

    def _resolve(self, pod: dict) -> ResourceIdentity:
        metadata = pod.get("metadata") or {}
        namespace = metadata.get("namespace")
        pod_owner = _controller_owner(pod, "Job")
        if not namespace or not pod_owner or not pod_owner.get("name"):
            raise UnsupportedOwner("Pod is not controlled by a batch/v1 Job")

        job = self.batch_api.read_namespaced_job(pod_owner["name"], namespace).to_dict()
        job = normalize_resource(job, api_version="batch/v1", kind="Job")
        _verify_uid("Job", namespace, pod_owner, job)

        jobset_ref = _controller_owner(job, "JobSet")
        if not jobset_ref or not jobset_ref.get("name"):
            return _identity(job, namespace)

        jobset = self.custom_api.get_namespaced_custom_object(
            "jobset.x-k8s.io",
            self.jobset_version,
            namespace,
            "jobsets",
            jobset_ref["name"],
        )
        _verify_uid("JobSet", namespace, jobset_ref, jobset)
        return _identity(jobset, namespace, api_version=f"jobset.x-k8s.io/{self.jobset_version}", kind="JobSet")


def _controller_owner(resource: dict, kind: str | None = None) -> dict | None:
    owners = [
        owner
        for owner in (resource.get("metadata") or {}).get("ownerReferences") or []
        if owner.get("controller") is True and (kind is None or owner.get("kind") == kind)
    ]
    return owners[0] if len(owners) == 1 else None


def _verify_uid(kind: str, namespace: str, reference: dict, resource: dict) -> None:
    expected_uid = reference.get("uid")
    actual_uid = (resource.get("metadata") or {}).get("uid")
    if not expected_uid or actual_uid != expected_uid:
        raise StaleOwnerReference(
            f"{kind} {namespace}/{reference.get('name')} uid mismatch "
            f"(reference uid={expected_uid!r}, fetched uid={actual_uid!r})"
        )


def _identity(
    resource: dict,
    namespace: str,
    *,
    api_version: str | None = None,
    kind: str | None = None,
) -> ResourceIdentity:
    metadata = resource.get("metadata") or {}
    identity = ResourceIdentity(
        api_version=resource.get("apiVersion") or api_version or "",
        kind=resource.get("kind") or kind or "",
        namespace=metadata.get("namespace") or namespace,
        name=metadata.get("name") or "",
        uid=metadata.get("uid") or "",
    )
    if not all((identity.api_version, identity.kind, identity.namespace, identity.name, identity.uid)):
        raise ResolutionPending("owner object is not fully visible yet")
    return identity


def normalize_resource(resource: dict, *, api_version: str | None = None, kind: str | None = None) -> dict:
    """Normalize Kubernetes SDK snake_case dictionaries to API JSON keys."""
    from datetime import datetime

    def jsonable(value):
        if isinstance(value, datetime):
            return value.isoformat().replace("+00:00", "Z")
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [jsonable(item) for item in value]
        return value

    metadata = resource.get("metadata") or {}
    status = resource.get("status") or {}
    container_statuses = status.get("container_statuses") or status.get("containerStatuses") or []
    normalized = {
        **resource,
        "apiVersion": resource.get("api_version") or resource.get("apiVersion") or api_version,
        "kind": resource.get("kind") or kind,
        "metadata": {
            "name": metadata.get("name"),
            "namespace": metadata.get("namespace"),
            "uid": metadata.get("uid"),
            "labels": metadata.get("labels") or {},
            "annotations": metadata.get("annotations") or {},
            "resourceVersion": metadata.get("resource_version") or metadata.get("resourceVersion"),
            "ownerReferences": [
                {
                    "apiVersion": owner.get("api_version") or owner.get("apiVersion"),
                    "kind": owner.get("kind"),
                    "name": owner.get("name"),
                    "uid": owner.get("uid"),
                    "controller": owner.get("controller"),
                }
                for owner in metadata.get("owner_references") or metadata.get("ownerReferences") or []
            ],
        },
        "spec": {
            **(resource.get("spec") or {}),
            "nodeName": (resource.get("spec") or {}).get("node_name")
            or (resource.get("spec") or {}).get("nodeName"),
        },
        "status": {
            **status,
            "conditions": [
                {
                    **condition,
                    "lastTransitionTime": condition.get("last_transition_time")
                    or condition.get("lastTransitionTime"),
                }
                for condition in status.get("conditions") or []
            ],
            "containerStatuses": [_normalize_container_status(item) for item in container_statuses],
        },
    }
    return jsonable(normalized)


def _normalize_container_status(container_status: dict) -> dict:
    state = container_status.get("state") or {}
    terminated = state.get("terminated")
    if not terminated:
        return {**container_status, "state": state}
    return {
        **container_status,
        "state": {
            **state,
            "terminated": {
                **terminated,
                "exitCode": terminated.get("exit_code")
                if "exit_code" in terminated
                else terminated.get("exitCode"),
            },
        },
    }
