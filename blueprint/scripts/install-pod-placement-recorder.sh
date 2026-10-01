#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

set -euo pipefail

namespace="${1:-default}"
if (( $# > 0 )); then
  shift
fi
release="${RELEASE_NAME:-attribution}"
chart="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/deploy/pod-placement-recorder"

command -v kubectl >/dev/null || { echo "kubectl is required" >&2; exit 1; }
command -v helm >/dev/null || { echo "helm is required" >&2; exit 1; }

context="$(kubectl config current-context)"
echo "Installing Kueue Pod placement recorder into namespace '$namespace' on context '$context'."
if [[ "${YES:-}" != "1" ]]; then
  read -r -p "Continue? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || exit 1
fi

kubectl api-resources --api-group=kueue.x-k8s.io | grep -q workloads \
  || { echo "Kueue Workload API is not installed" >&2; exit 1; }

kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install "$release" "$chart" --namespace "$namespace" --wait "$@"

echo
echo "Recorder ready. Configure the Blueprint ECA with:"
cat <<EOF
[eca.kueue]
namespace = "$namespace"
placement_records_enabled = true

[eca.kueue.placement_recorder]
endpoint = "http://$release-pod-placement-recorder.$namespace.svc:8080"
EOF
