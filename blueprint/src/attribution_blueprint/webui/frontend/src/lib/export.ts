// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Decimal from 'decimal.js'
import type { RunDetail, RunSummary } from '@/interfaces/attribution'
import { attemptGpuHours, nodeGpuHours } from './metric'
import { summarizeNode } from './node-summary'

/** Per-run elapsed hours, rounded to 4dp at the export boundary for clean output. */
function exportElapsed(run: RunSummary): number {
  return new Decimal(run.elapsedH).toDecimalPlaces(4).toNumber()
}

function exportAttemptGpuHours(run: RunSummary): number {
  return attemptGpuHours(run).toDecimalPlaces(4).toNumber()
}

function exportNodeGpuHours(run: RunSummary, nodeId: string): number {
  return nodeGpuHours(run, nodeId).toDecimalPlaces(4).toNumber()
}

export function downloadBlob(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

const SAFE_KEY = /^[A-Za-z_][A-Za-z0-9_-]*$/
const NEEDS_QUOTE =
  /[\n:#&*!|>'"%@`{}[\],]|^\s|\s$|^(true|false|null|yes|no|on|off|~)$|^-?\d/i

function yamlString(value: string): string {
  if (value === '') return '""'
  if (NEEDS_QUOTE.test(value)) {
    return `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n')}"`
  }
  return value
}

function yamlScalar(value: unknown): string {
  if (value === null || value === undefined) return 'null'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'null'
  if (typeof value === 'string') return yamlString(value)
  return yamlString(String(value))
}

export function toYaml(value: unknown, indent = 0): string {
  const pad = '  '.repeat(indent)

  if (Array.isArray(value)) {
    if (value.length === 0) return `${pad}[]`
    return value
      .map((item) => {
        if (item && typeof item === 'object' && !Array.isArray(item)) {
          const entries = Object.entries(item as Record<string, unknown>)
          if (entries.length === 0) return `${pad}- {}`
          const lines = entries.map(([k, v], i) => {
            const key = SAFE_KEY.test(k) ? k : `"${k}"`
            const linePad = i === 0 ? `${pad}- ` : `${pad}  `
            if (v === null || v === undefined) return `${linePad}${key}: null`
            if (Array.isArray(v) && v.length === 0) return `${linePad}${key}: []`
            if (v && typeof v === 'object') {
              return `${linePad}${key}:\n${toYaml(v, indent + 2)}`
            }
            return `${linePad}${key}: ${yamlScalar(v)}`
          })
          return lines.join('\n')
        }
        return `${pad}- ${yamlScalar(item)}`
      })
      .join('\n')
  }

  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
    if (entries.length === 0) return `${pad}{}`
    return entries
      .map(([k, v]) => {
        const key = SAFE_KEY.test(k) ? k : `"${k}"`
        if (v === null || v === undefined) return `${pad}${key}: null`
        if (Array.isArray(v)) {
          if (v.length === 0) return `${pad}${key}: []`
          return `${pad}${key}:\n${toYaml(v, indent + 1)}`
        }
        if (typeof v === 'object') {
          return `${pad}${key}:\n${toYaml(v, indent + 1)}`
        }
        return `${pad}${key}: ${yamlScalar(v)}`
      })
      .join('\n')
  }

  return `${pad}${yamlScalar(value)}`
}

export function verdictExportData(run: RunSummary) {
  return {
    attempt_uid: run.attemptUid,
    execution_context: run.jobId,
    workload: run.workload && run.workload !== '—' ? run.workload : null,
    attempt: run.attempt,
    user: run.user,
    status: run.status,
    outcome: run.outcome,
    verdict_label: run.verdictLabel,
    detail: run.detail,
    window: {
      start_utc: run.startTime,
      end_utc: run.endTime,
      elapsed_hours: exportElapsed(run),
    },
    gpu_count: run.gpuCount,
    gpus_per_node: run.gpusPerNode,
    gpu_hours: exportAttemptGpuHours(run),
    root_symptom: {
      symptom: run.rootSymptom.symptom,
      rule: run.rootSymptom.rule,
      time_utc: run.rootSymptom.time,
      line: run.rootSymptom.line,
    },
    symptoms: run.symptoms.map((s) => ({ name: s.name, sites: s.sites })),
    nodes: run.nodeIds,
    responsible_nodes: run.responsibleNodes,
    pipeline: run.pipeline,
    service: run.service,
  }
}

export function logToText(run: RunDetail): string {
  return run.log
    .map((l) => {
      const ts = l.t
      const level =
        l.level === 'symptom' ? 'SYMPTOM'
        : l.level === 'error' ? 'ERROR  '
        : l.level === 'debug' ? 'DEBUG  '
        :                       'INFO   '
      return `${ts} ${level} ${l.text}`
    })
    .join('\n')
}

// The node's page's own summary of its runs (summarizeNode), for machines:
// symptoms most attributed first, as the page lists them.
export function nodeExportData(nodeId: string, attempts: RunSummary[]) {
  const { responsible, symptoms, gpuHours } = summarizeNode(attempts, nodeId)
  return {
    node_id: nodeId,
    attempts_impacted: responsible.length,
    distinct_symptoms: symptoms.length,
    gpu_hours_observed: gpuHours.toDecimalPlaces(2).toNumber(),
    symptoms: symptoms.map(([name, count]) => ({ name, count })),
    attempts: responsible.map((r) => ({
      attempt_uid: r.attemptUid,
      execution_context: r.jobId,
      attempt: r.attempt,
      user: r.user,
      outcome: r.outcome,
      end_time_utc: r.endTime,
      elapsed_hours: exportElapsed(r),
      gpu_hours_observed: exportNodeGpuHours(r, nodeId),
      root_symptom: r.rootSymptom.symptom,
      rule: r.rootSymptom.rule,
    })),
  }
}

const CSV_COLUMNS: { header: string; pick: (r: RunSummary) => string | number }[] = [
  { header: 'end_time_utc',    pick: (r) => r.endTime },
  { header: 'execution_context', pick: (r) => r.jobId },
  { header: 'workload',        pick: (r) => (r.workload && r.workload !== '—' ? r.workload : '') },
  { header: 'attempt',         pick: (r) => r.attempt },
  { header: 'attempt_uid',     pick: (r) => r.attemptUid },
  { header: 'user',            pick: (r) => r.user },
  { header: 'outcome',         pick: (r) => r.outcome },
  { header: 'verdict_label',   pick: (r) => r.verdictLabel },
  { header: 'root_symptom',    pick: (r) => r.rootSymptom.symptom },
  { header: 'rule',            pick: (r) => r.rootSymptom.rule },
  { header: 'nodes',           pick: (r) => r.nodes },
  { header: 'gpu_count',       pick: (r) => r.gpuCount },
  { header: 'gpus_per_node',   pick: (r) => r.gpusPerNode },
  { header: 'responsible_nodes', pick: (r) => r.responsibleNodes.join('|') },
  { header: 'elapsed_hours',   pick: (r) => exportElapsed(r) },
  { header: 'gpu_hours',       pick: (r) => exportAttemptGpuHours(r) },
  { header: 'pipeline',        pick: (r) => r.pipeline },
  { header: 'service',         pick: (r) => r.service },
]

function csvCell(value: string | number): string {
  const s = String(value)
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return `"${s.replace(/"/g, '""')}"`
  }
  return s
}

export function runsToCSV(runs: RunSummary[]): string {
  const header = CSV_COLUMNS.map((c) => c.header).join(',')
  const rows = runs.map((r) => CSV_COLUMNS.map((c) => csvCell(c.pick(r))).join(','))
  return [header, ...rows].join('\n')
}

export function runsExportData(runs: RunSummary[]) {
  return runs.map(verdictExportData)
}

export function slug(input: string, max = 60): string {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, max) || 'export'
}

export function utcDateStamp(): string {
  const d = new Date()
  return `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, '0')}${String(d.getUTCDate()).padStart(2, '0')}-${String(d.getUTCHours()).padStart(2, '0')}${String(d.getUTCMinutes()).padStart(2, '0')}`
}
