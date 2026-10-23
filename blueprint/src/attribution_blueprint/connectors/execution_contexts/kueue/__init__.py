# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.connectors.execution_contexts.kueue.connector import (
    KueueExecutionContext,
    KueueExecutionContextAdapter,
)
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM


def _build_prometheus_metrics(spec: dict):
    from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.prometheus_resolver import (
        PrometheusPlacementResolver,
    )
    from attribution_blueprint.mca import _coerce_float, _coerce_int

    endpoint = spec.get("endpoint")
    if not endpoint:
        raise MCAError("eca.kueue.prometheus_metrics requires endpoint")

    auth = None
    auth_spec = spec.get("auth")
    if auth_spec is not None:
        auth_type = auth_spec.get("type")
        if auth_type == "token":
            env_name = auth_spec.get("token_env")
            if not env_name:
                raise MCAError("eca.kueue.prometheus_metrics.auth type=token requires token_env")
            if env_name not in os.environ:
                raise MCAError(f"auth.token_env references unset environment variable: {env_name}")
            auth = TokenAuth(token=os.environ[env_name], org_id=auth_spec.get("org_id"))
        elif auth_type == "mtls":
            if not auth_spec.get("cert_path") or not auth_spec.get("key_path"):
                raise MCAError("eca.kueue.prometheus_metrics.auth type=mtls requires cert_path and key_path")
            auth = MTLSAuth(
                cert_path=auth_spec["cert_path"],
                key_path=auth_spec["key_path"],
                ca_cert_path=auth_spec.get("ca_cert_path"),
                org_id=auth_spec.get("org_id"),
            )
        else:
            raise MCAError(f"unknown Prometheus placement auth type: {auth_type!r}")

    timeout = _coerce_float(spec.get("timeout_seconds", 60.0), "eca.kueue.prometheus_metrics.timeout_seconds")
    padding = _coerce_int(
        spec.get("lookback_padding_seconds", 300),
        "eca.kueue.prometheus_metrics.lookback_padding_seconds",
    )
    batch_size = _coerce_int(spec.get("batch_size", 20), "eca.kueue.prometheus_metrics.batch_size")
    try:
        return PrometheusPlacementResolver(
            endpoint=endpoint,
            auth=auth,
            timeout_seconds=timeout,
            lookback_padding_seconds=padding,
            cluster=spec.get("cluster"),
            cluster_label=spec.get("cluster_label"),
            pod_info_metric=spec.get("pod_info_metric", "kube_pod_info"),
            pod_labels_metric=spec.get("pod_labels_metric", "kube_pod_labels"),
            pod_controller_uid_label=spec.get(
                "pod_controller_uid_label",
                "label_batch_kubernetes_io_controller_uid",
            ),
            batch_size=batch_size,
        )
    except ValueError as error:
        raise MCAError(f"invalid eca.kueue.prometheus_metrics configuration: {error}") from error


def _build_placement_recorder(spec: dict):
    from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder_client import (
        RecorderPlacementClient,
    )
    from attribution_blueprint.mca import _coerce_float

    endpoint = spec.get("endpoint")
    if not endpoint:
        raise MCAError("eca.kueue.placement_recorder requires endpoint")

    auth = None
    auth_spec = spec.get("auth")
    if auth_spec is not None:
        auth_type = auth_spec.get("type")
        if auth_type == "token":
            env_name = auth_spec.get("token_env")
            if not env_name:
                raise MCAError("eca.kueue.placement_recorder.auth type=token requires token_env")
            if env_name not in os.environ:
                raise MCAError(f"auth.token_env references unset environment variable: {env_name}")
            auth = TokenAuth(token=os.environ[env_name], org_id=auth_spec.get("org_id"))
        elif auth_type == "mtls":
            if not auth_spec.get("cert_path") or not auth_spec.get("key_path"):
                raise MCAError("eca.kueue.placement_recorder.auth type=mtls requires cert_path and key_path")
            auth = MTLSAuth(
                cert_path=auth_spec["cert_path"],
                key_path=auth_spec["key_path"],
                ca_cert_path=auth_spec.get("ca_cert_path"),
                org_id=auth_spec.get("org_id"),
            )
        else:
            raise MCAError(f"unknown placement recorder auth type: {auth_type!r}")

    timeout = _coerce_float(spec.get("timeout_seconds", 10.0), "eca.kueue.placement_recorder.timeout_seconds")
    try:
        return RecorderPlacementClient(endpoint, auth=auth, timeout_seconds=timeout)
    except ValueError as error:
        raise MCAError(f"invalid eca.kueue.placement_recorder configuration: {error}") from error


def _build(spec: dict, ctx: dict) -> KueueExecutionContextAdapter:
    from attribution_blueprint.mca import _coerce_int

    # `spec.get(..., 60)` returns 60 only when the key is absent; a TOML
    # literal `null` arrives as Python None and must fall back to the same
    # default. Otherwise None would propagate to the Kubernetes client as a
    # request timeout of `None`, silently disabling the timeout entirely.
    timeout = spec.get("timeout_seconds")
    if timeout is None:
        timeout = 60
    else:
        timeout = _coerce_int(timeout, "eca.kueue.timeout_seconds")
        if timeout <= 0:
            raise MCAError(f"eca.kueue.timeout_seconds must be positive, got {timeout}")
    placement_records_enabled = spec.get("placement_records_enabled", False)
    if isinstance(placement_records_enabled, str):
        normalized = placement_records_enabled.lower()
        if normalized in {"true", "yes", "1"}:
            placement_records_enabled = True
        elif normalized in {"false", "no", "0", ""}:
            placement_records_enabled = False
        else:
            raise MCAError(
                "eca.kueue.placement_records_enabled must be a bool, "
                f"got {placement_records_enabled!r}"
            )
    elif not isinstance(placement_records_enabled, bool):
        raise MCAError(
            "eca.kueue.placement_records_enabled must be a bool, "
            f"got {placement_records_enabled!r}"
        )
    prometheus_spec = spec.get("prometheus_metrics")
    prometheus_placement_resolver = _build_prometheus_metrics(prometheus_spec) if prometheus_spec is not None else None
    recorder_spec = spec.get("placement_recorder")
    if placement_records_enabled and recorder_spec is None:
        raise MCAError("eca.kueue.placement_records_enabled requires eca.kueue.placement_recorder")
    # A recorder installation watches exactly one namespace (see the Helm
    # chart's `--namespace={{ .Release.Namespace }}`). Without a configured
    # namespace, discovery could span namespaces the recorder doesn't watch,
    # and its "no matching records" response is indistinguishable from "this
    # namespace isn't covered at all".
    if placement_records_enabled and not spec.get("namespace"):
        raise MCAError("eca.kueue.placement_records_enabled requires eca.kueue.namespace")
    placement_reader = _build_placement_recorder(recorder_spec) if placement_records_enabled else None
    event_max_age = spec.get("event_max_age_seconds")
    if event_max_age is None:
        event_max_age = 3600
    else:
        event_max_age = _coerce_int(event_max_age, "eca.kueue.event_max_age_seconds")
        if event_max_age <= 0:
            raise MCAError(f"eca.kueue.event_max_age_seconds must be positive, got {event_max_age}")
    kwargs = dict(
        kubeconfig=spec.get("kubeconfig"),
        context=spec.get("context"),
        namespace=spec.get("namespace"),
        timeout_seconds=timeout,
        placement_records_enabled=placement_records_enabled,
        placement_reader=placement_reader,
        event_max_age_seconds=event_max_age,
        prometheus_placement_resolver=prometheus_placement_resolver,
    )
    workload_api_version = spec.get("workload_api_version")
    if workload_api_version is not None:
        if not isinstance(workload_api_version, str) or not workload_api_version:
            raise MCAError(
                f"eca.kueue.workload_api_version must be a non-empty string, got {workload_api_version!r}"
            )
        kwargs["workload_api_version"] = workload_api_version
    return KueueExecutionContextAdapter(**kwargs)


_registry.register_component(
    framework="eca",
    name="kueue",
    schema={
        "ids": _PARAM,
        "start_time": _PARAM,
        "end_time": _PARAM,
        "nodelist": _PARAM,
        # Optional path to a kubeconfig file. When unset, falls back to its
        # default discovery (KUBECONFIG env var, ~/.kube/config, or an
        # in-cluster service-account token when running inside a pod).
        "kubeconfig": _PARAM,
        # Optional context name within the resolved kubeconfig. When unset,
        # uses the kubeconfig's current-context.
        "context": _PARAM,
        # Optional namespace scope. The empty string (or unset) queries every
        # namespace. Set this to limit blast radius on multi-tenant clusters
        # or to allow bare `<name>` --ids values.
        "namespace": _PARAM,
        "timeout_seconds": _PARAM,
        # Kueue Workload CRD `<group>/<version>` to read.
        "workload_api_version": _PARAM,
        "placement_records_enabled": _PARAM,
        "placement_recorder": {
            "endpoint": _PARAM,
            "timeout_seconds": _PARAM,
            "auth": {
                "type": _PARAM,
                "cert_path": _PARAM,
                "key_path": _PARAM,
                "ca_cert_path": _PARAM,
                "token_env": _PARAM,
                "org_id": _PARAM,
            },
        },
        # Max age of a Workload's end_time before Event-based recovery is
        # skipped entirely.
        "event_max_age_seconds": _PARAM,
        "prometheus_metrics": {
            "endpoint": _PARAM,
            "cluster": _PARAM,
            "cluster_label": _PARAM,
            "timeout_seconds": _PARAM,
            "lookback_padding_seconds": _PARAM,
            "batch_size": _PARAM,
            "pod_info_metric": _PARAM,
            "pod_labels_metric": _PARAM,
            "pod_controller_uid_label": _PARAM,
            "auth": {
                "type": _PARAM,
                "cert_path": _PARAM,
                "key_path": _PARAM,
                "ca_cert_path": _PARAM,
                "token_env": _PARAM,
                "org_id": _PARAM,
            },
        },
    },
    build=_build,
)


__all__ = ["KueueExecutionContext", "KueueExecutionContextAdapter"]
