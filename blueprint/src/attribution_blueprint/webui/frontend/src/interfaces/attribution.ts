export type OutcomeKey = 'hardware' | 'software' | 'multi' | 'user' | 'undetermined' | 'pass'

export interface Outcome {
  id: OutcomeKey
  label: string
  short: string
  color: string
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

export interface Run {
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
  detail: string
  rootSymptom: RootSymptom
  symptoms: SymptomEntry[]
  responsibleNodes: string[]
  pipeline: string
  service: string
  log: LogLine[]
  sibling?: string
  diagnose?: PersonaDiagnose
  operatorDiagnose?: PersonaDiagnose
}

// Daily buckets cover only the outcomes the day-by-day charts plot. 'undetermined'
// is intentionally omitted (kept in sync with the daily SQL in webui/db.py); window
// totals come from metrics.totalRuns, not from summing these buckets.
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
