import Decimal from 'decimal.js'
import type { Run } from '@/interfaces/attribution'

/** gpuCount x elapsedH for one attempt. */
export function attemptGpuHours(run: Run): Decimal {
  return new Decimal(run.gpuCount).times(run.elapsedH)
}

/** Attempt GPU-hours distributed over listed nodes for one observed node. */
export function nodeGpuHours(run: Run): Decimal {
  const billedNodes = run.nodes > 0 ? run.nodes : 1
  return attemptGpuHours(run).div(billedNodes)
}

/** Sum attempt GPU-hours as an exact Decimal (no float accumulation). */
export function gpuHours(runs: Run[]): Decimal {
  return runs.reduce((acc, r) => acc.plus(attemptGpuHours(r)), new Decimal(0))
}

/** Sum elapsedH as an exact Decimal (per-run elapsed sum; no node multiplier). */
export function elapsedHours(runs: Run[]): Decimal {
  return runs.reduce((acc, r) => acc.plus(new Decimal(r.elapsedH)), new Decimal(0))
}
