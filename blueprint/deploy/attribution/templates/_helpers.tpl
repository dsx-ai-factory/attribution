# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

{{- define "attribution.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end }}

{{- define "attribution.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "attribution.labels" -}}
{{ include "attribution.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end }}

{{/*
The image of one component: its own repository and tag where set, else the
shared image. Called as (list $ "service"), (list $ "runner") or (list $ "dashboard").
*/}}
{{- define "attribution.componentImage" -}}
{{- $root := index . 0 -}}
{{- $name := index . 1 -}}
{{- $component := index $root.Values $name -}}
{{- $repository := default $root.Values.image.repository $component.image.repository -}}
{{- $tag := default $root.Values.image.tag $component.image.tag -}}
{{- if not $repository -}}
{{- fail (printf "an image repository is required for %s: set image.repository or %s.image.repository" $name $name) -}}
{{- end -}}
{{- if not $tag -}}
{{- fail (printf "an image tag is required for %s: set image.tag or %s.image.tag" $name $name) -}}
{{- end -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end }}

{{/* "true" when the pod gets a persistent volume (see persistence.enabled). */}}
{{- define "attribution.persistence" -}}
{{- if kindIs "invalid" .Values.persistence.enabled -}}
{{- if or .Values.runner.enabled .Values.dashboard.enabled -}}true{{- else -}}false{{- end -}}
{{- else -}}
{{- ternary "true" "false" (eq (toString .Values.persistence.enabled) "true") -}}
{{- end -}}
{{- end }}

{{/* "true" when the runner reads a complete, user-provided Blueprint TOML file. */}}
{{- define "attribution.existingConfig" -}}
{{- if or .Values.config.existing.configMap.name .Values.config.existing.secret.name -}}true{{- else -}}false{{- end -}}
{{- end }}

{{/* The address the Service listens on: all addresses only when its API is exposed. */}}
{{- define "attribution.serviceBindAddress" -}}
{{- if .Values.service.api.enabled -}}0.0.0.0{{- else -}}127.0.0.1{{- end -}}
{{- end }}

{{/* The Service URL the runner calls and the generated configuration names. */}}
{{- define "attribution.serviceUrl" -}}
{{- if .Values.runner.serviceUrl -}}
{{- .Values.runner.serviceUrl | trimSuffix "/" -}}
{{- else if .Values.service.enabled -}}
{{- printf "http://127.0.0.1:%d" (int .Values.service.port) -}}
{{- else -}}
{{- fail "runner.serviceUrl is required when service.enabled is false" -}}
{{- end -}}
{{- end }}

{{/* A Service probe: its timing values plus /v2/healthz, over HTTP when the API is exposed, else in the container. */}}
{{- define "attribution.serviceProbe" -}}
{{- $root := index . 0 -}}
{{- toYaml (index . 1) }}
{{- if $root.Values.service.api.enabled }}
httpGet:
  path: /v2/healthz
  port: api
{{- else if $root.Values.service.probes.command }}
exec:
  command:
{{ toYaml $root.Values.service.probes.command | indent 4 }}
{{- else }}
exec:
  command:
    - /opt/blueprint/.venv/bin/python
    - -c
    - {{ printf "import urllib.request; urllib.request.urlopen('http://127.0.0.1:%d/v2/healthz', timeout=3)" (int $root.Values.service.port) | toJson }}
{{- end }}
{{- end }}

{{/* Fails the render on settings that cannot work together. */}}
{{- define "attribution.validate" -}}
{{- if not (or .Values.service.enabled .Values.runner.enabled .Values.dashboard.enabled) -}}
{{- fail "enable at least one of service, runner and dashboard" -}}
{{- end -}}
{{- if .Values.runner.enabled -}}
{{- if not .Values.namespaces -}}
{{- fail "namespaces is required when the runner is enabled: list the namespaces whose workloads are attributed" -}}
{{- end -}}
{{- if le (int .Values.runner.initialLookbackSeconds) (int .Values.runner.settleSeconds) -}}
{{- fail "runner.initialLookbackSeconds must be larger than runner.settleSeconds, or a pass window is empty" -}}
{{- end -}}
{{- if and (not .Values.service.enabled) (not .Values.runner.serviceUrl) -}}
{{- fail "runner.serviceUrl is required when service.enabled is false" -}}
{{- end -}}
{{- else -}}
{{- if eq (include "attribution.existingConfig" .) "true" -}}
{{- fail "config.existing has no effect when the runner is disabled: remove it or enable the runner" -}}
{{- end -}}
{{- end -}}
{{- if and .Values.config.existing.configMap.name .Values.config.existing.secret.name -}}
{{- fail "set only one of config.existing.configMap and config.existing.secret" -}}
{{- end -}}
{{- if and (eq (include "attribution.existingConfig" .) "true") .Values.extraConfig -}}
{{- fail "extraConfig is not used with config.existing: put those settings in your own file, which is the complete configuration" -}}
{{- end -}}
{{- if and .Values.config.existing.configMap.name (not .Values.config.existing.configMap.key) -}}
{{- fail "config.existing.configMap.key is required" -}}
{{- end -}}
{{- if and .Values.config.existing.secret.name (not .Values.config.existing.secret.key) -}}
{{- fail "config.existing.secret.key is required" -}}
{{- end -}}
{{- if and (or .Values.runner.enabled .Values.dashboard.enabled) (not (regexMatch "^/data(/[^/]+)+$" .Values.dashboard.databasePath)) -}}
{{- fail "dashboard.databasePath must be a file path under /data, the only path the runner and the dashboard share" -}}
{{- end -}}
{{- if and .Values.service.enabled .Values.dashboard.enabled (eq (int .Values.service.port) (int .Values.dashboard.port)) -}}
{{- fail "service.port and dashboard.port must differ: both containers share the pod's network namespace" -}}
{{- end -}}
{{- if and .Values.persistence.existingClaim (eq (include "attribution.persistence" .) "false") -}}
{{- fail "persistence.existingClaim has no effect when the volume is off: remove it, or leave persistence.enabled unset" -}}
{{- end -}}
{{- if and .Values.service.api.enabled (not .Values.service.enabled) -}}
{{- fail "service.api.enabled needs service.enabled" -}}
{{- end -}}
{{- if and .Values.service.api.enabled .Values.networkPolicy.enabled (not .Values.networkPolicy.service.from) -}}
{{- fail "networkPolicy.service.from must name who may call the API when service.api.enabled and networkPolicy.enabled are both true, or nothing can reach it" -}}
{{- end -}}
{{- if and .Values.ingress.enabled (not .Values.dashboard.enabled) -}}
{{- fail "ingress.enabled needs dashboard.enabled" -}}
{{- end -}}
{{- if and .Values.ingress.enabled (not .Values.ingress.hosts) -}}
{{- fail "ingress.hosts is required when ingress.enabled is true" -}}
{{- end -}}
{{- if and .Values.ingress.enabled .Values.networkPolicy.enabled (not .Values.networkPolicy.dashboard.from) -}}
{{- fail "networkPolicy.dashboard.from must name the ingress controller's namespace when ingress.enabled and networkPolicy.enabled are both true, or the Ingress cannot reach the dashboard" -}}
{{- end -}}
{{- end }}
