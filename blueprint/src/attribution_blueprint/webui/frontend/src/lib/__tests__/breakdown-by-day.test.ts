import { describe, it, expect } from 'vitest'
import { buildDailyByOutcome, dailySpine, buildDaily, breakdownHighlight } from '@/lib/attribution-data'
import type { Run, OutcomeKey } from '@/interfaces/attribution'

const base: Run = {
  id: '1', jobId: 'team/x', workload: 'team/x', attempt: 1, attemptUid: 'u', user: 'alice',
  status: 'FAILED', nodes: 1, nodeIds: ['n1'], elapsedH: 1.5, gpusPerNode: 8, gpuCount: 8,
  endTime: '2026-05-26T00:00:00Z', startTime: '2026-05-26T00:00:00Z',
  outcome: 'hardware', verdictLabel: 'Hardware', detail: '',
  rootSymptom: { symptom: 'GpuFell', rule: 'gpu_fell_off_bus', time: 'T', line: 'x' },
  symptoms: [], responsibleNodes: [], pipeline: '0.1.0', service: 'v2', log: [],
}

function mk(symptom: string, outcome: OutcomeKey, endTime: string): Run {
  return { ...base, outcome, endTime, rootSymptom: { ...base.rootSymptom, symptom } }
}

const from = new Date('2026-05-26T00:00:00Z')
const to = new Date('2026-05-29T23:59:59Z') // 4 calendar days: 26,27,28,29

describe('dailySpine', () => {
  it('produces inclusive UTC day strings matching buildDaily', () => {
    expect(dailySpine(from, to)).toEqual(['2026-05-26', '2026-05-27', '2026-05-28', '2026-05-29'])
    expect(dailySpine(from, to)).toEqual(buildDaily([], from, to).map((d) => d.date))
  })
})

describe('buildDailyByOutcome', () => {
  it('buckets failed attempts of one outcome by day and symptom', () => {
    const runs: Run[] = [
      mk('GpuFell', 'hardware', '2026-05-26T01:00:00Z'),
      mk('GpuFell', 'hardware', '2026-05-26T02:00:00Z'),
      mk('EccDoubleBit', 'hardware', '2026-05-28T01:00:00Z'),
      mk('NcclWatchdog', 'software', '2026-05-27T01:00:00Z'), // wrong outcome, excluded
    ]
    const r = buildDailyByOutcome(runs, from, to, 'hardware')
    expect(r.days).toEqual(['2026-05-26', '2026-05-27', '2026-05-28', '2026-05-29'])
    expect(r.total).toBe(3)
    const gpu = r.series.find((s) => s.symptom === 'GpuFell')!
    const ecc = r.series.find((s) => s.symptom === 'EccDoubleBit')!
    expect(gpu.data).toEqual([2, 0, 0, 0])
    expect(gpu.total).toBe(2)
    expect(ecc.data).toEqual([0, 0, 1, 0])
    expect(r.series.map((s) => s.symptom)).toEqual(['GpuFell', 'EccDoubleBit'])
  })

  it('only counts the requested outcome', () => {
    const runs: Run[] = [mk('GpuFell', 'hardware', '2026-05-26T01:00:00Z')]
    expect(buildDailyByOutcome(runs, from, to, 'software').total).toBe(0)
    expect(buildDailyByOutcome(runs, from, to, 'software').series).toEqual([])
  })

  it('folds symptoms beyond topN into a single trailing Other series', () => {
    const runs: Run[] = [
      mk('A', 'hardware', '2026-05-26T01:00:00Z'),
      mk('A', 'hardware', '2026-05-26T02:00:00Z'),
      mk('A', 'hardware', '2026-05-27T02:00:00Z'), // A total 3
      mk('B', 'hardware', '2026-05-26T03:00:00Z'),
      mk('B', 'hardware', '2026-05-28T03:00:00Z'), // B total 2
      mk('C', 'hardware', '2026-05-27T03:00:00Z'), // C total 1
      mk('D', 'hardware', '2026-05-29T03:00:00Z'), // D total 1
    ]
    const r = buildDailyByOutcome(runs, from, to, 'hardware', 2)
    expect(r.series.map((s) => s.symptom)).toEqual(['A', 'B', 'Other'])
    const other = r.series[2]
    expect(other.data).toEqual([0, 1, 0, 1]) // C on day27(idx1) + D on day29(idx3)
    expect(other.total).toBe(2)
  })

  it('breaks total ties by symptom name ascending', () => {
    const runs: Run[] = [
      mk('Zeta', 'hardware', '2026-05-26T01:00:00Z'),
      mk('Alpha', 'hardware', '2026-05-27T01:00:00Z'),
    ]
    const r = buildDailyByOutcome(runs, from, to, 'hardware')
    expect(r.series.map((s) => s.symptom)).toEqual(['Alpha', 'Zeta'])
  })

  it('returns an empty series with a full day spine for no matching runs', () => {
    const r = buildDailyByOutcome([], from, to, 'hardware')
    expect(r.series).toEqual([])
    expect(r.total).toBe(0)
    expect(r.days.length).toBe(4)
  })
})

describe('breakdownHighlight', () => {
  // A hardware breakdown plotting GpuFell + EccDoubleBit.
  const hw = buildDailyByOutcome(
    [
      mk('GpuFell', 'hardware', '2026-05-26T01:00:00Z'),
      mk('EccDoubleBit', 'hardware', '2026-05-27T01:00:00Z'),
    ],
    from,
    to,
    'hardware',
  )

  it('highlights the active symptom when this chart plots it — no outcome needed', () => {
    // The regression: a symptom filter set without an outcome (as Pareto/Movers do)
    // must still highlight. breakdownHighlight takes no outcome at all.
    expect(breakdownHighlight(hw, 'GpuFell')).toBe('GpuFell')
  })

  it('does not highlight a symptom this chart does not plot', () => {
    expect(breakdownHighlight(hw, 'NcclWatchdog')).toBeUndefined()
  })

  it('returns undefined for no active symptom or the synthetic Other bucket', () => {
    expect(breakdownHighlight(hw, undefined)).toBeUndefined()
    expect(breakdownHighlight(hw, 'Other')).toBeUndefined()
  })
})
