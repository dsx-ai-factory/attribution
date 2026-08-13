import type { OutcomeKey, Run } from '@/interfaces/attribution'
import type { FleetFilter, ParetoAxis, ParetoMetric } from '@/interfaces/filter'

const OUTCOMES_LIST: OutcomeKey[] = ['hardware', 'software', 'multi', 'user', 'undetermined', 'pass']
const AXIS_LIST: ParetoAxis[] = ['symptom', 'node', 'user', 'rule']
const METRIC_LIST: ParetoMetric[] = ['count', 'nodeHours']

export const DEFAULT_FILTER: FleetFilter = {
  status: 'failed',
  axis: 'symptom',
  metric: 'count',
}

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/

// Round-trip through Date because V8 rolls over impossible days (2026-02-30
// parses as 2026-03-02) instead of rejecting them.
function isValidIsoDay(s: string): boolean {
  if (!ISO_DATE.test(s)) return false
  const ms = Date.parse(`${s}T00:00:00Z`)
  return !Number.isNaN(ms) && new Date(ms).toISOString().slice(0, 10) === s
}

/**
 * Validate the from/to pair from the URL. Both must be present, well-formed
 * ISO dates, and ordered from <= to; otherwise the pair is dropped so the view
 * falls back to the default window instead of rendering an invalid range.
 */
function parseDateRange(params: URLSearchParams): { from?: string; to?: string } {
  const from = params.get('from')
  const to = params.get('to')
  if (!from || !to) return {}
  if (!isValidIsoDay(from) || !isValidIsoDay(to)) return {}
  if (from > to) return {}
  return { from, to }
}

export function parseFilter(params: URLSearchParams): FleetFilter {
  const outcome = params.get('outcome')
  const axis = params.get('axis')
  const metric = params.get('metric')
  const status = params.get('status')
  const windowPreset = params.get('window')
  const { from, to } = parseDateRange(params)
  return {
    outcome: outcome && OUTCOMES_LIST.includes(outcome as OutcomeKey)
      ? (outcome as OutcomeKey) : undefined,
    symptom: params.get('symptom') || undefined,
    node:    params.get('node')    || undefined,
    rule:    params.get('rule')    || undefined,
    user:    params.get('user')    || undefined,
    status:  status === 'all' ? 'all' : 'failed',
    q:       params.get('q')       || undefined,
    axis:    axis && AXIS_LIST.includes(axis as ParetoAxis)
      ? (axis as ParetoAxis) : 'symptom',
    metric:  metric && METRIC_LIST.includes(metric as ParetoMetric)
      ? (metric as ParetoMetric) : 'count',
    window:  windowPreset === 'custom' ? 'custom' : undefined,
    from,
    to,
  }
}

/**
 * Produce a URLSearchParams that contains only non-default values, so URLs stay clean.
 */
export function serializeFilter(f: FleetFilter): URLSearchParams {
  const sp = new URLSearchParams()
  if (f.outcome) sp.set('outcome', f.outcome)
  if (f.symptom) sp.set('symptom', f.symptom)
  if (f.node)    sp.set('node', f.node)
  if (f.rule)    sp.set('rule', f.rule)
  if (f.user)    sp.set('user', f.user)
  // Emit status explicitly so spec URLs (?outcome=hardware&status=failed) appear verbatim.
  sp.set('status', f.status)
  if (f.q)       sp.set('q', f.q)
  if (f.axis !== 'symptom') sp.set('axis', f.axis)
  if (f.metric !== 'count') sp.set('metric', f.metric)
  if (f.window === 'custom') sp.set('window', 'custom')
  if (f.from)    sp.set('from', f.from)
  if (f.to)      sp.set('to', f.to)
  return sp
}

function matchesSearch(r: Run, q: string): boolean {
  const needle = q.toLowerCase()
  return (
    r.jobId.toLowerCase().includes(needle) ||
    r.workload.toLowerCase().includes(needle) ||
    r.user.toLowerCase().includes(needle) ||
    r.rootSymptom.symptom.toLowerCase().includes(needle) ||
    r.rootSymptom.rule.toLowerCase().includes(needle) ||
    r.outcome.toLowerCase().includes(needle) ||
    r.attemptUid.toLowerCase().includes(needle) ||
    r.detail.toLowerCase().includes(needle) ||
    r.responsibleNodes.some((n) => n.toLowerCase().includes(needle))
  )
}

export function applyFilter(runs: Run[], f: FleetFilter): Run[] {
  return runs.filter((r) => {
    if (f.status === 'failed' && r.outcome === 'pass') return false
    if (f.outcome && r.outcome !== f.outcome) return false
    if (f.symptom && r.rootSymptom.symptom !== f.symptom) return false
    if (f.node && !r.responsibleNodes.includes(f.node)) return false
    if (f.rule && r.rootSymptom.rule !== f.rule) return false
    if (f.user && r.user !== f.user) return false
    if (f.q && !matchesSearch(r, f.q)) return false
    return true
  })
}

/** Set/clear a single dimension filter, returning the new filter. */
export function withDimension<K extends keyof FleetFilter>(
  f: FleetFilter,
  key: K,
  value: FleetFilter[K] | undefined,
): FleetFilter {
  if (value === undefined || value === '') {
    const next = { ...f }
    delete next[key]
    return next
  }
  return { ...f, [key]: value }
}
