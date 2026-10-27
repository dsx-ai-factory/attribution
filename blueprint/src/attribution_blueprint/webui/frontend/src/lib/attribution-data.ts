// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Decimal from 'decimal.js'
import type {
  OutcomeKey,
  Outcome,
  RunSummary,
  DailyBucket,
  DailyBreakdown,
  DailyBreakdownSeries,
  SymptomMover,
  SymptomSite,
} from '@/interfaces/attribution'

// Labels only: an outcome's color lives in src/styles/globals.css and is
// reached through data-outcome (markup) or lib/tokens outcomeColor (canvas).
export const OUTCOMES: Record<OutcomeKey, Outcome> = {
  hardware: { id: 'hardware', label: 'Hardware Fault' },
  software: { id: 'software', label: 'System Software Fault' },
  multi:    { id: 'multi',    label: 'Ambiguous Fault' },
  user:     { id: 'user',     label: 'Application Fault' },
  undetermined: { id: 'undetermined', label: 'Unattributed' },
  pass:     { id: 'pass',     label: 'Pass' },
}

// The order outcomes are listed in wherever they appear together.
export const OUTCOME_ORDER: readonly OutcomeKey[] = ['hardware', 'software', 'multi', 'user', 'undetermined', 'pass']

// The one wording of a symptom site's responsibility state.
export const SITE_STATUS_LABELS: Record<SymptomSite['status'], string> = {
  bad: 'Responsible',
  interesting: 'Supporting evidence',
}

function attemptGpuHours(run: RunSummary): Decimal {
  return new Decimal(run.gpuCount).times(run.elapsedH)
}

// ─── Movers (in-window symptom movement) ──────────────────────────────────

const OUTCOME_PRIORITY: OutcomeKey[] = ['hardware', 'multi', 'software', 'user', 'undetermined', 'pass']

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

// Equal-width buckets across [from, to]; last bucket includes `to` (the
// window's end is inclusive). NOT calendar-aligned. The hour/day switch
// sits at ~1.5 days, so only the 24h preset is hourly; 7d/30d and date-picker
// (>= 1 day) ranges are daily.
export function bucketCount(from: Date, to: Date): number {
  const spanH = (to.getTime() - from.getTime()) / 3600000
  return spanH <= 36 ? clamp(Math.round(spanH), 6, 36) : clamp(Math.round(spanH / 24), 3, 60)
}

// Deterministic argmax over a symptom's outcome counts; ties resolve by
// OUTCOME_PRIORITY so the card color never flips on insertion order.
export function dominantOutcome(outcomes: Partial<Record<OutcomeKey, number>>): OutcomeKey {
  let best: OutcomeKey = 'hardware'
  let bestN = -1
  for (const k of OUTCOME_PRIORITY) {
    const n = outcomes[k] ?? 0
    if (n > bestN) { bestN = n; best = k }
  }
  return best
}

// Guarded movement: late-half minus early-half, damped by sqrt(total). Returns
// 0 ("did not move materially") so noise can't earn a slot. The |delta| >= 2
// guard ignores +/-1 jitter. New (0 -> >=2) and gone (>=2 -> 0) symptoms count
// even below the 3-run floor, since an emerging or cleared failure mode is worth
// surfacing at low volume; anything mid-range needs total >= 3 to be a real trend.
export function movementScore(spark: number[]): number {
  if (spark.length === 0) return 0
  const total = spark.reduce((a, b) => a + b, 0)
  const mid = Math.floor(spark.length / 2)
  const early = spark.slice(0, mid).reduce((a, b) => a + b, 0)
  const late = spark.slice(mid).reduce((a, b) => a + b, 0)
  const delta = late - early
  if (Math.abs(delta) < 2) return 0
  const isAppearance = early === 0 || late === 0 // new symptom or fully cleared one
  if (total < (isAppearance ? 2 : 3)) return 0
  return delta / Math.sqrt(total)
}

// Top failure symptoms ranked by in-window movement (not count). Pass/Clean runs
// are excluded first so a mixed-outcome symptom isn't dropped or mis-colored.
export function buildMovers(runs: RunSummary[], from: Date, to: Date, limit = 5): SymptomMover[] {
  const spanMs = to.getTime() - from.getTime()
  if (spanMs <= 0) return []
  const count = bucketCount(from, to)
  const fromT = from.getTime()
  const bucketMs = spanMs / count

  const acc = new Map<
    string,
    { count: number; outcomes: Partial<Record<OutcomeKey, number>>; spark: number[] }
  >()
  for (const r of runs) {
    if (r.outcome === 'pass') continue
    const s = r.rootSymptom.symptom
    let m = acc.get(s)
    if (!m) {
      m = { count: 0, outcomes: {}, spark: new Array(count).fill(0) }
      acc.set(s, m)
    }
    m.count += 1
    m.outcomes[r.outcome] = (m.outcomes[r.outcome] ?? 0) + 1
    const i = clamp(Math.floor((new Date(r.endTime).getTime() - fromT) / bucketMs), 0, count - 1)
    m.spark[i] += 1
  }

  return [...acc.entries()]
    .map(([symptom, m]) => ({
      symptom,
      count: m.count,
      dominantOutcome: dominantOutcome(m.outcomes),
      spark: m.spark,
      score: movementScore(m.spark),
    }))
    .filter((mv) => mv.score !== 0)
    .sort((a, b) => Math.abs(b.score) - Math.abs(a.score) || b.count - a.count)
    .slice(0, limit)
}

// Inclusive UTC day spine across [from, to], one 'YYYY-MM-DD' per calendar day.
// Shared by buildDaily and buildDailyByOutcome so their day axes are provably identical.
export function dailySpine(from: Date, to: Date): string[] {
  const dayMs = 86400000
  const fromDay = Date.UTC(from.getUTCFullYear(), from.getUTCMonth(), from.getUTCDate())
  const toDay   = Date.UTC(to.getUTCFullYear(),   to.getUTCMonth(),   to.getUTCDate())
  const days: string[] = []
  for (let t = fromDay; t <= toDay; t += dayMs) {
    days.push(new Date(t).toISOString().slice(0, 10))
  }
  return days
}

export function buildDaily(runs: RunSummary[], from: Date, to: Date): DailyBucket[] {
  const days: DailyBucket[] = dailySpine(from, to).map((date) => ({
    date, hw: 0, sw: 0, mix: 0, usr: 0, pass: 0, nodeHours: 0,
  }))
  const indexByDate = new Map(days.map((d, i) => [d.date, i]))
  const hours = days.map(() => new Decimal(0))
  runs.forEach((r) => {
    const i = indexByDate.get(r.endTime.slice(0, 10))
    if (i === undefined) return
    const bucket = days[i]
    if (r.outcome === 'hardware') bucket.hw += 1
    else if (r.outcome === 'software') bucket.sw += 1
    else if (r.outcome === 'multi') bucket.mix += 1
    else if (r.outcome === 'user') bucket.usr += 1
    else if (r.outcome === 'pass') bucket.pass += 1
    hours[i] = hours[i].plus(attemptGpuHours(r))
  })
  days.forEach((d, i) => { d.nodeHours = hours[i].toNumber() })
  return days
}

// Per-day stack of failed attempts for ONE outcome, split by root symptom.
// Top-N symptoms by window total are kept; the rest fold into a trailing 'Other'.
export function buildDailyByOutcome(
  runs: RunSummary[],
  from: Date,
  to: Date,
  outcome: OutcomeKey,
  topN = 7,
): DailyBreakdown {
  const days = dailySpine(from, to)
  const indexByDate = new Map(days.map((d, i) => [d, i]))
  const bySymptom = new Map<string, number[]>()
  let total = 0
  for (const r of runs) {
    if (r.outcome !== outcome) continue
    const i = indexByDate.get(r.endTime.slice(0, 10))
    if (i === undefined) continue
    const s = r.rootSymptom.symptom
    let arr = bySymptom.get(s)
    if (!arr) { arr = new Array(days.length).fill(0); bySymptom.set(s, arr) }
    arr[i] += 1
    total += 1
  }
  const ranked = [...bySymptom.entries()]
    .map(([symptom, data]) => ({ symptom, data, total: data.reduce((a, b) => a + b, 0) }))
    .sort((a, b) => b.total - a.total || a.symptom.localeCompare(b.symptom))

  const series: DailyBreakdownSeries[] = ranked.slice(0, topN)
  const rest = ranked.slice(topN)
  if (rest.length > 0) {
    const otherData = new Array(days.length).fill(0)
    let otherTotal = 0
    for (const r of rest) {
      r.data.forEach((v, i) => { otherData[i] += v })
      otherTotal += r.total
    }
    series.push({ symptom: 'Other', data: otherData, total: otherTotal })
  }
  return { days, series, total }
}

// The symptom a breakdown chart should highlight (dimming its siblings): the active
// `symptom` filter, but only when THIS chart actually plots it. Outcome-agnostic by
// design — symptom filters set from Pareto / Movers / a deep link carry no outcome, so
// gating on outcome would silently never highlight. The synthetic 'Other' bucket is not
// a real symptom, so it never highlights.
export function breakdownHighlight(
  breakdown: DailyBreakdown,
  symptom: string | undefined,
): string | undefined {
  if (!symptom || symptom === 'Other') return undefined
  return breakdown.series.some((s) => s.symptom === symptom) ? symptom : undefined
}
