# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only `kubectl get ... -o json` runner shared by Kubernetes adapters."""

from __future__ import annotations

import json
import subprocess


class KubectlError(RuntimeError):
    """A kubectl query failed."""


class KubectlNotFoundError(KubectlError):
    """kubectl reported that the requested object does not exist."""


class KubectlClient:
    """Runs `kubectl get` queries and returns the decoded JSON.

    `kubectl_path` is the binary to invoke. `kubeconfig` and `context` map to
    `--kubeconfig` and `--context`; both are optional, and when unset kubectl
    falls back to its default discovery (KUBECONFIG, ~/.kube/config, or the
    in-cluster service account).

    `get_one` returns `None` only for NotFound, and every other failure
    (Forbidden, timeout, malformed output) raises `KubectlError`, so a failed
    read is never mistaken for an empty result. `get_one` requires a JSON
    object and `list` requires an object with an `items` list; any other
    shape raises.
    """

    def __init__(
        self,
        kubectl_path: str = "kubectl",
        kubeconfig: str | None = None,
        context: str | None = None,
        timeout_seconds: float = 60,
    ):
        self.kubectl_path = kubectl_path
        self.kubeconfig = kubeconfig
        self.context = context
        self.timeout_seconds = timeout_seconds

    def _base(self) -> list[str]:
        cmd = [self.kubectl_path]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        if self.context:
            cmd += ["--context", self.context]
        return cmd

    def _run_json(self, args: list[str]) -> object:
        cmd = self._base() + args + ["-o", "json"]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                timeout=self.timeout_seconds,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            error_cls = KubectlNotFoundError if "Error from server (NotFound)" in stderr else KubectlError
            raise error_cls(f"kubectl query failed with exit {e.returncode}: {stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise KubectlError(f"kubectl query timed out after {self.timeout_seconds}s") from e
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise KubectlError(f"kubectl query returned invalid JSON: {e}") from e

    def get_one(self, kind: str, name: str, *, namespace: str) -> dict | None:
        args = ["get", kind, name]
        if namespace:
            args += ["-n", namespace]
        try:
            payload = self._run_json(args)
        except KubectlNotFoundError:
            return None
        if not isinstance(payload, dict):
            raise KubectlError(f"kubectl query returned an unexpected response shape: {args!r}")
        return payload

    def list(
        self,
        kind: str,
        *,
        namespace: str | None,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ) -> list[dict]:
        args = ["get", kind]
        if field_selector:
            args += ["--field-selector", field_selector]
        if label_selector:
            args += ["-l", label_selector]
        if namespace:
            args += ["-n", namespace]
        else:
            args += ["-A"]
        payload = self._run_json(args)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise KubectlError(f"kubectl query returned an unexpected response shape: {args!r}")
        return payload["items"]
