// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Decimal from 'decimal.js'
import type { RunSummary } from '@/interfaces/attribution'
import { nodeGpuHours } from '@/lib/metric'

export interface NodeSummary {
  /** The attempts that ran on the node or named it responsible. */
  attempts: RunSummary[]
  /** Those that named it responsible. */
  responsible: RunSummary[]
  /** Each root symptom those were attributed to and how often, most often first; a tie keeps the runs' order. */
  symptoms: [string, number][]
  /** The symptom that stands out, if one does. */
  leading?: [string, number]
  /** The GPU-hours the node spent in its attempts: none in an attempt that only named it. */
  gpuHours: Decimal
}

// What a node's runs say of it, the same wherever it is said (its page, its
// hover card). A symptom is named as the node's own only when one stands out:
// recorded, and attributed more often than any other. At a tie the runs'
// order would pick one of the tied at random, so none is named.
export function summarizeNode(runs: RunSummary[], nodeId: string): NodeSummary {
  const attempts = runs.filter((r) => r.nodeIds.includes(nodeId) || r.responsibleNodes.includes(nodeId))
  const responsible = attempts.filter((r) => r.responsibleNodes.includes(nodeId))
  const counts = new Map<string, number>()
  responsible.forEach((r) => counts.set(r.rootSymptom.symptom, (counts.get(r.rootSymptom.symptom) ?? 0) + 1))
  const symptoms = [...counts].sort(([, a], [, b]) => b - a)
  const [top, runnerUp] = symptoms
  const leading = top !== undefined && top[0] !== '' && (runnerUp === undefined || runnerUp[1] < top[1]) ? top : undefined
  const gpuHours = attempts.reduce((sum, r) => sum.plus(nodeGpuHours(r, nodeId)), new Decimal(0))
  return { attempts, responsible, symptoms, leading, gpuHours }
}
