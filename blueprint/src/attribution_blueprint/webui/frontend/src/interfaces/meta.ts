// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DailyBreakdown,
  DailyBucket,
  NodeLedgerEntry,
  OutcomeLedgerEntry,
  RunDetail,
  RunSummary,
  RuleLedgerEntry,
  SymptomLedgerEntry,
  SymptomMover,
  UserLedgerEntry,
} from '@/interfaces/attribution'

/** Server-injected bootstrap blob. Mirrors bootstrap.build_payload (Python). */
export interface BootstrapPayload {
  runs: RunDetail[]
  meta: Meta
}

export interface DateBounds {
  min: string | null
  max: string | null
}

export interface Meta {
  versions: { schema: string | null; pipeline: string | null; service: string | null }
  dateBounds: DateBounds
  /** Runs in the caller's query scope, which is not purely the date window. */
  windowedRuns: number
  /** Every run in the database. Unlike windowedRuns, no query scope narrows it. */
  dbTotalRuns: number
  /** File name of the database being served, or null when the server did not name one. */
  database: string | null
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
  windowedRuns: number
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
  /** How many categories each concentration ledger has in all; a ledger holds only those the fleet view ranks. */
  categoryCounts: {
    symptom: number
    node: number
    user: number
    rule: number
  }
  /** How many categories tie for largest in each concentration ledger, by count and by GPU-hours, the uncut ledger's. */
  largestTies: {
    symptom: { count: number; nodeHours: number }
    node: { count: number; nodeHours: number }
    user: { count: number; nodeHours: number }
    rule: { count: number; nodeHours: number }
  }
  movers: SymptomMover[]
}

export interface RunsPagePayload {
  runs: RunSummary[]
  total: number
  page: number
  pageSize: number
}
