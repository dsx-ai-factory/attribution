import type { OutcomeKey } from '@/interfaces/attribution'

export type ParetoAxis = 'symptom' | 'node' | 'user' | 'rule'
export type ParetoMetric = 'count' | 'nodeHours'
export type RunStatus = 'failed' | 'all'
export type DateWindowMode = 'custom'

export interface FleetFilter {
  outcome?: OutcomeKey
  symptom?: string
  node?: string
  rule?: string
  user?: string
  status: RunStatus
  q?: string
  axis: ParetoAxis
  metric: ParetoMetric
  window?: DateWindowMode
  from?: string // ISO date YYYY-MM-DD
  to?: string   // ISO date YYYY-MM-DD
}
