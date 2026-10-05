# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only `kubectl get ... -o json` runner shared by Kubernetes adapters."""

from __future__ import annotations

import json
import subprocess

from loguru import logger


class KubectlClient:
    """Runs `kubectl get` queries and returns the decoded JSON.

    `kubectl_path` is the binary to invoke. `kubeconfig` and `context` map to
    `--kubeconfig` and `--context`; both are optional, and when unset kubectl
    falls back to its default discovery (KUBECONFIG, ~/.kube/config, or the
    in-cluster service account).

    A failed, timed-out, or malformed query logs a warning and yields
    `None`/`[]` so one bad lookup cannot sink a whole discovery pass. Pass
    `required=True` to `run_json` to raise instead.
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

    def run_json(self, args: list[str], *, required: bool = False) -> dict | None:
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
            if required:
                raise RuntimeError(
                    f"required kubectl query failed with exit {e.returncode}: {(e.stderr or '').strip()}"
                ) from e
            logger.warning(f"kubectl exited {e.returncode}: {(e.stderr or '').strip() or '<no stderr>'}")
            return None
        except subprocess.TimeoutExpired:
            if required:
                raise RuntimeError(f"required kubectl query timed out after {self.timeout_seconds}s")
            logger.warning(f"kubectl timed out after {self.timeout_seconds}s: {cmd!r}")
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            if required:
                raise RuntimeError(f"required kubectl query returned invalid JSON: {e}") from e
            logger.warning(f"kubectl returned invalid JSON: {e}")
            return None

    def get_one(self, kind: str, name: str, *, namespace: str) -> dict | None:
        args = ["get", kind, name]
        if namespace:
            args += ["-n", namespace]
        return self.run_json(args)

    def list(self, kind: str, *, namespace: str | None, field_selector: str | None = None) -> list[dict]:
        args = ["get", kind]
        if field_selector:
            args += ["--field-selector", field_selector]
        if namespace:
            args += ["-n", namespace]
        else:
            args += ["-A"]
        payload = self.run_json(args)
        if not payload:
            return []
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        return items
