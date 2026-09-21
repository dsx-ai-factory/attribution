import Decimal from 'decimal.js'
import type {
  OutcomeKey,
  Outcome,
  RunSummary,
  DailyBucket,
  DailyBreakdown,
  DailyBreakdownSeries,
  NodeLedgerEntry,
  SymptomLedgerEntry,
  RuleLedgerEntry,
  OutcomeLedgerEntry,
  SymptomMover,
} from '@/interfaces/attribution'

export const OUTCOMES: Record<OutcomeKey, Outcome> = {
  // Hex literals mirror --outcome-* / --color-*-400 tokens in src/styles/globals.css.
  // Keep in sync if a token changes.
  hardware: { id: 'hardware', label: 'Hardware Fault', short: 'HW', color: '#ec6358' },
  software: { id: 'software', label: 'System Software Fault', short: 'SW', color: '#3a96d9' },
  multi:    { id: 'multi',    label: 'Ambiguous Fault', short: 'AMB', color: '#a1a1a1' },
  user:     { id: 'user',     label: 'Application Fault', short: 'APP', color: '#c58420' },
  undetermined: { id: 'undetermined', label: 'Unattributed', short: 'UNA', color: '#a05ab4' },
  pass:     { id: 'pass',     label: 'Pass', short: 'OK', color: '#76b900' },
}

// ─── Ledger builders ──────────────────────────────────────────────────────
// Pure derivations over a run set (e.g. window-filtered). The single data
// boundary is the dashboard API; views fetch only the slices they need.

function attemptGpuHours(run: RunSummary): Decimal {
  return new Decimal(run.gpuCount).times(run.elapsedH)
}

function nodeGpuHours(run: RunSummary): Decimal {
  const billedNodes = run.nodes > 0 ? run.nodes : 1
  return attemptGpuHours(run).div(billedNodes)
}

export function buildSymptomLedger(runs: RunSummary[]): SymptomLedgerEntry[] {
  const m: Record<string, SymptomLedgerEntry> = {}
  const hours: Record<string, Decimal> = {}
  runs.forEach((r) => {
    const s = r.rootSymptom.symptom
    if (!m[s]) {
      m[s] = { symptom: s, count: 0, nodeHours: 0, outcomes: {} }
      hours[s] = new Decimal(0)
    }
    m[s].count += 1
    hours[s] = hours[s].plus(attemptGpuHours(r))
    m[s].outcomes[r.outcome] = (m[s].outcomes[r.outcome] || 0) + 1
  })
  Object.keys(m).forEach((s) => { m[s].nodeHours = hours[s].toNumber() })
  return Object.values(m).sort((a, b) => b.count - a.count)
}

export function buildRuleLedger(runs: RunSummary[]): RuleLedgerEntry[] {
  const m: Record<string, RuleLedgerEntry> = {}
  const hours: Record<string, Decimal> = {}
  runs.forEach((r) => {
    const s = r.rootSymptom.rule
    if (!m[s]) {
      m[s] = { rule: s, count: 0, nodeHours: 0, outcomes: {} }
      hours[s] = new Decimal(0)
    }
    m[s].count += 1
    hours[s] = hours[s].plus(attemptGpuHours(r))
    m[s].outcomes[r.outcome] = (m[s].outcomes[r.outcome] || 0) + 1
  })
  Object.keys(m).forEach((s) => { m[s].nodeHours = hours[s].toNumber() })
  return Object.values(m).sort((a, b) => b.count - a.count)
}

export function buildNodeLedger(runs: RunSummary[]): NodeLedgerEntry[] {
  const m: Record<string, NodeLedgerEntry> = {}
  const hours: Record<string, Decimal> = {}
  runs.forEach((r) => {
    r.responsibleNodes.forEach((n) => {
      if (!m[n]) {
        m[n] = { node: n, count: 0, outcomes: {}, symptoms: {}, nodeHours: 0 }
        hours[n] = new Decimal(0)
      }
      m[n].count += 1
      m[n].outcomes[r.outcome] = (m[n].outcomes[r.outcome] || 0) + 1
      m[n].symptoms[r.rootSymptom.symptom] = (m[n].symptoms[r.rootSymptom.symptom] || 0) + 1
      hours[n] = hours[n].plus(nodeGpuHours(r))
    })
  })
  Object.keys(m).forEach((n) => { m[n].nodeHours = hours[n].toNumber() })
  return Object.values(m).sort((a, b) => b.count - a.count)
}

export function buildOutcomeLedger(runs: RunSummary[]): OutcomeLedgerEntry[] {
  const m: Partial<Record<OutcomeKey, OutcomeLedgerEntry>> = {}
  const hours: Partial<Record<OutcomeKey, Decimal>> = {}
  runs.forEach((r) => {
    if (!m[r.outcome]) {
      m[r.outcome] = { outcome: r.outcome, count: 0, nodeHours: 0 }
      hours[r.outcome] = new Decimal(0)
    }
    m[r.outcome]!.count += 1
    hours[r.outcome] = hours[r.outcome]!.plus(attemptGpuHours(r))
  })
  ;(Object.keys(m) as OutcomeKey[]).forEach((k) => { m[k]!.nodeHours = hours[k]!.toNumber() })
  return Object.values(m).sort((a, b) => b.count - a.count) as OutcomeLedgerEntry[]
}

// ─── Movers (in-window symptom movement) ──────────────────────────────────

const OUTCOME_PRIORITY: OutcomeKey[] = ['hardware', 'multi', 'software', 'user', 'undetermined', 'pass']

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

// Equal-width buckets across [from, to]; last bucket includes `to` (matches
// filterRunsByDate's inclusive end). NOT calendar-aligned. The hour/day switch
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

// Categorical palette for symptom stacks, readable on the #121212 card surface.
// Assigned by series index; stable because buildDailyByOutcome orders series
// deterministically. 'Other' uses OTHER_COLOR rather than a palette slot.
export const SYMPTOM_PALETTE: string[] = [
  '#3a96d9', '#ec6358', '#c58420', '#76b900', '#a78bfa', '#22d3ee', '#f472b6',
]
export const OTHER_COLOR = '#52525b'

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

export function filterRunsByDate(runs: RunSummary[], from: Date, to: Date): RunSummary[] {
  const fromT = from.getTime()
  const toT = to.getTime()
  return runs.filter((r) => {
    const t = new Date(r.endTime).getTime()
    return t >= fromT && t <= toT
  })
}

/**
 * Anchor "now" to the latest endTime in the supplied runs. Served data is
 * fixed in time; a real-clock anchor would yield zero rows for "24h". For an
 * empty set there is nothing to anchor to, so fall back to the wall clock.
 */
export function nowAnchor(runs: RunSummary[]): Date {
  if (runs.length === 0) return new Date()
  const t = runs.reduce((acc, r) => Math.max(acc, new Date(r.endTime).getTime()), 0)
  return new Date(t || Date.now())
}
