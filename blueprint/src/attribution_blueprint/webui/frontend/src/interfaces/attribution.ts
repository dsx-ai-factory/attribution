// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type OutcomeKey = 'hardware' | 'software' | 'multi' | 'user' | 'undetermined' | 'pass'

export interface Outcome {
  id: OutcomeKey
  label: string
}

export type LogLevel = 'info' | 'debug' | 'error' | 'symptom'

export interface LogLine {
  t: string
  level: LogLevel
  text: string
}

export interface RootSymptom {
  symptom: string
  rule: string
  time: string
  line: string
}

export interface SymptomEntry {
  name: string
  sites: number
}

export interface PersonaDiagnose {
  hasPersona: boolean
  title: string
  summary: string
  impact: string
  action: string
  supportDetails: string[]
  category?: string | null
  severity?: string | null
  catalogId?: string | null
}

export type AttributionSection<T> =
  | { state: 'available'; data: T }
  | { state: 'empty'; data: T }
  | { state: 'absent' | 'null' | 'invalid'; data: null }

// WebUI detail fields preserve Service V2 evidence and nullability. Observation
// IDs are decoded from the original JSON number tokens as strings, since Service
// integers can exceed JavaScript's exact range. Source positions stay numeric.
export type ObservationSource =
  | ''
  | 'slurm_info'
  | 'job_keywords'
  | 'applog'
  | 'syslog'
  | 'health_check'
  | 'node_metrics'
  | 'nvlink_metrics'
  | 'ib_metrics'
  | 'eth_metrics'
  | 'network_syslog'
  | 'ufmlog'
  | 'csp_event'

export interface AttributionSource {
  observation_id: string
  source: ObservationSource
  pos: number | null
  line: string | null
  rule: string | null
  time: string | null
}

export interface SourcedSymptom {
  symptom: string
  source: AttributionSource
  catalog_id?: string | null
}

export type SymptomSite = {
  status: 'bad' | 'interesting'
  source: AttributionSource
} & (
  | { type: 'NodeSource'; node: string }
  | { type: 'ResourcelessSource' }
)

export type DecisionNodeType =
  | 'UNKNOWN'
  | 'EXTRACTION'
  | 'FILTER'
  | 'SCRUB'
  | 'REFINER'
  | 'CASCADE_PRUNE'
  | 'OUTCOME'

export type DecisionAction = 'UNKNOWN' | 'CREATED' | 'MATCHED' | 'REMOVED' | 'PRUNED'

export interface DecisionNode {
  node_id: string
  node_type: DecisionNodeType
  hostname: string | null
  symptom: string
  rule_name: string | null
  action: DecisionAction
  reason?: string | null
  datum?: string | null
  line_num?: number | null
  timestamp?: string | null
  obs_source?: string | null
}

export interface DecisionEdge {
  parent_id: string
  child_id: string
  edge_type: string
}

export interface DecisionGraph {
  nodes: DecisionNode[]
  edges: DecisionEdge[]
}

export interface RunSummary {
  decisionGraphState: AttributionSection<DecisionGraph>['state']
  id: string
  jobId: string
  workload: string
  attempt: number
  attemptUid: string
  user: string
  status: 'FAILED' | 'COMPLETED'
  nodes: number
  nodeIds: string[]
  elapsedH: number
  gpusPerNode: number
  gpuCount: number
  endTime: string
  startTime: string
  outcome: OutcomeKey
  verdictLabel: string
  terminalStates?: string[]
  failureDomains?: string[]
  cancellationActors?: string[]
  uncertaintyType?: string | null
  evidenceStatus?: string | null
  // Why evidence was partial or missing.
  ingestionReports?: { connector_reports: object[]; ingestion_reports: object[] } | null
  detail: string
  rootSymptom: RootSymptom
  symptoms: SymptomEntry[]
  errorPairs?: string
  responsibleNodes: string[]
  pipeline: string
  service: string
  sibling?: string
  diagnose?: PersonaDiagnose
  // Legacy payloads may include this; tenant views and exports must ignore it.
  operatorDiagnose?: PersonaDiagnose
}

export interface RunDetail extends RunSummary {
  log: LogLine[]
  attribution: {
    rootSymptom: AttributionSection<SourcedSymptom>
    symptomSites: AttributionSection<Record<string, SymptomSite[]>>
    // Missing and null graphs both mean no graph is available to display.
    decisionGraph: AttributionSection<DecisionGraph>
  }
}

// Daily buckets cover only the outcomes the day-by-day charts plot. 'undetermined'
// is intentionally omitted (kept in sync with the daily SQL in webui/db.py); window
// totals come from metrics.windowedRuns, not from summing these buckets.
export interface DailyBucket {
  date: string
  hw: number
  sw: number
  mix: number
  usr: number
  pass: number
  nodeHours: number
}

export interface DailyBreakdownSeries {
  symptom: string
  data: number[]   // failed-attempt count per day, aligned to DailyBreakdown.days
  total: number    // sum over the window
}

export interface DailyBreakdown {
  days: string[]                  // YYYY-MM-DD spine (same rule as buildDaily)
  series: DailyBreakdownSeries[]  // top-N by total desc, then a folded 'Other' last (if any)
  total: number                   // total failed attempts for this outcome in the window
}

export interface NodeLedgerEntry {
  node: string
  count: number
  outcomes: Partial<Record<OutcomeKey, number>>
  symptoms: Record<string, number>
  nodeHours: number
}

export interface SymptomLedgerEntry {
  symptom: string
  count: number
  nodeHours: number
  outcomes: Partial<Record<OutcomeKey, number>>
}

export interface RuleLedgerEntry {
  rule: string
  count: number
  nodeHours: number
  outcomes: Partial<Record<OutcomeKey, number>>
}

export interface UserLedgerEntry {
  user: string
  count: number
  nodeHours: number
  outcomes: Partial<Record<OutcomeKey, number>>
}

export interface OutcomeLedgerEntry {
  outcome: OutcomeKey
  count: number
  nodeHours: number
}

export interface SymptomMover {
  symptom: string
  count: number              // total FAILED runs with this symptom in the window
  dominantOutcome: OutcomeKey
  spark: number[]            // per-bucket failed-run counts across the window
  score: number              // signed movement; sign = direction, |score| = magnitude
}
