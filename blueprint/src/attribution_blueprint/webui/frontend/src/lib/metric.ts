// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Decimal from 'decimal.js'
import type { RunSummary } from '@/interfaces/attribution'

/** gpuCount x elapsedH for one attempt. */
export function attemptGpuHours(run: RunSummary): Decimal {
  return new Decimal(run.gpuCount).times(run.elapsedH)
}

/**
 * A node's GPU-hours in one attempt, as the node's own page counts them: the
 * attempt's spread evenly over the nodes it lists (those it ran on); none for
 * a node it does not list, such as a responsible node the attempt did not run
 * on. Failure concentration charges a failure's GPU-hours to the nodes it
 * blames instead, split evenly (the server's node ledger).
 */
export function nodeGpuHours(run: RunSummary, nodeId: string): Decimal {
  return run.nodeIds.includes(nodeId) ? attemptGpuHours(run).div(run.nodes) : new Decimal(0)
}

/** Sum attempt GPU-hours as an exact Decimal (no float accumulation). */
export function gpuHours(runs: RunSummary[]): Decimal {
  return runs.reduce((acc, r) => acc.plus(attemptGpuHours(r)), new Decimal(0))
}

/** Sum elapsedH as an exact Decimal (per-run elapsed sum; no node multiplier). */
export function elapsedHours(runs: RunSummary[]): Decimal {
  return runs.reduce((acc, r) => acc.plus(new Decimal(r.elapsedH)), new Decimal(0))
}
