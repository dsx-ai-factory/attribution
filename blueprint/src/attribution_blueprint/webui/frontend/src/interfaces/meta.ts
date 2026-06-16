import type {
  DailyBreakdown,
  DailyBucket,
  NodeLedgerEntry,
  OutcomeLedgerEntry,
  Run,
  RuleLedgerEntry,
  SymptomLedgerEntry,
  SymptomMover,
  UserLedgerEntry,
} from '@/interfaces/attribution'

/** Server-injected bootstrap blob. Mirrors bootstrap.build_payload (Python). */
export interface BootstrapPayload {
  runs: Run[]
  meta: Meta
}

export interface Meta {
  versions: { schema: string | null; pipeline: string | null; service: string | null }
  dateBounds: { min: string | null; max: string | null }
  totalRuns: number
  returnedRuns: number
}

/** Mirrors the future /api/v1 query params so hook signatures don't change later. */
export interface FleetQuery {
  from?: string
  to?: string
  outcome?: string
  q?: string
  page?: number
  size?: number
  sort?: string
  order?: 'asc' | 'desc'
}

export interface DashboardMetrics {
  totalRuns: number
  gpuHoursRaw: number
  gpuCount: number
  hwFailures: number
  swFailures: number
  mixFailures: number
  combinedFailures: number
  hwMtbi: number
  combinedMtbi: number
}

export interface DashboardPayload {
  meta: Meta
  metrics: DashboardMetrics
  daily: DailyBucket[]
  hwBreakdown: DailyBreakdown
  swBreakdown: DailyBreakdown
  ledgers: {
    symptom: SymptomLedgerEntry[]
    node: NodeLedgerEntry[]
    user: UserLedgerEntry[]
    rule: RuleLedgerEntry[]
    outcome: OutcomeLedgerEntry[]
  }
  movers: SymptomMover[]
}

export interface RunsPagePayload {
  runs: Run[]
  total: number
  page: number
  pageSize: number
}
