import { describe, it, expect } from 'vitest'
import { buildMovers, bucketCount, movementScore, dominantOutcome } from '@/lib/attribution-data'
import type { RunSummary, OutcomeKey } from '@/interfaces/attribution'

const base: RunSummary = {
  id: '1', jobId: 'team/x', workload: 'team/x', attempt: 1, attemptUid: 'u', user: 'alice',
  status: 'FAILED', nodes: 1, nodeIds: ['n1'], elapsedH: 1.5, gpusPerNode: 8, gpuCount: 8,
  endTime: '2026-05-26T00:00:00Z', startTime: '2026-05-26T00:00:00Z',
  outcome: 'hardware', verdictLabel: 'Hardware', detail: '',
  decisionGraphState: 'absent',
  rootSymptom: { symptom: 'GpuFell', rule: 'gpu_fell_off_bus', time: 'T', line: 'x' },
  symptoms: [], responsibleNodes: [], pipeline: '0.1.0', service: 'v2',
}

function mk(symptom: string, outcome: OutcomeKey, endTime: string): RunSummary {
  return { ...base, outcome, endTime, rootSymptom: { ...base.rootSymptom, symptom } }
}

describe('movementScore', () => {
  it('surfaces a NEW symptom at low count (0 -> 2 appearance)', () => {
    expect(movementScore([0, 0, 2])).toBeGreaterThan(0)
  })
  it('surfaces a GONE symptom at low count (2 -> 0 cleared)', () => {
    expect(movementScore([2, 0, 0])).toBeLessThan(0)
  })
  it('still ignores a single-run appearance (0 -> 1 noise)', () => {
    expect(movementScore([0, 1])).toBe(0)
  })
  it('returns 0 when the half-window delta < 2', () => {
    expect(movementScore([2, 2, 2, 2])).toBe(0) // early 4, late 4, delta 0
  })
  it('is positive for a rising series', () => {
    expect(movementScore([0, 0, 0, 3, 3, 3])).toBeGreaterThan(0)
  })
  it('is negative for a falling series', () => {
    expect(movementScore([3, 3, 3, 0, 0, 0])).toBeLessThan(0)
  })
  it('returns 0 for an empty series', () => {
    expect(movementScore([])).toBe(0)
  })
  it('returns 0 at the count boundary when there is no movement (total 3, delta 0)', () => {
    expect(movementScore([1, 1, 1])).toBe(0)
  })
})

describe('bucketCount', () => {
  const d = (iso: string) => new Date(iso)
  it('uses hourly resolution for a 24h window', () => {
    expect(bucketCount(d('2026-06-01T00:00:00Z'), d('2026-06-02T00:00:00Z'))).toBe(24)
  })
  it('uses 7 buckets for a 7d window', () => {
    expect(bucketCount(d('2026-05-26T00:00:00Z'), d('2026-06-02T00:00:00Z'))).toBe(7)
  })
  it('uses 30 buckets for a 30d window', () => {
    expect(bucketCount(d('2026-05-03T00:00:00Z'), d('2026-06-02T00:00:00Z'))).toBe(30)
  })
})

describe('dominantOutcome', () => {
  it('picks the argmax outcome', () => {
    expect(dominantOutcome({ hardware: 2, software: 3 })).toBe('software')
  })
  it('breaks ties by severity priority (hardware > software)', () => {
    expect(dominantOutcome({ hardware: 2, software: 2 })).toBe('hardware')
  })
  it('defaults to hardware for an empty outcome map', () => {
    expect(dominantOutcome({})).toBe('hardware')
  })
})

describe('buildMovers', () => {
  const from = new Date('2026-05-26T00:00:00Z')
  const to = new Date('2026-06-02T00:00:00Z') // 7d -> 7 daily buckets

  it('surfaces movers by |movement|, excludes flat-but-frequent and pass runs', () => {
    const runs: RunSummary[] = [
      mk('Flat', 'software', '2026-05-26T01:00:00Z'),
      mk('Flat', 'software', '2026-05-27T01:00:00Z'),
      mk('Flat', 'software', '2026-05-29T01:00:00Z'),
      mk('Flat', 'software', '2026-05-31T01:00:00Z'),
      mk('Flat', 'software', '2026-06-01T01:00:00Z'),
      mk('Rise', 'hardware', '2026-05-31T01:00:00Z'),
      mk('Rise', 'hardware', '2026-06-01T01:00:00Z'),
      mk('Rise', 'hardware', '2026-06-01T05:00:00Z'),
      mk('Clean', 'pass', '2026-06-01T02:00:00Z'),
    ]
    const movers = buildMovers(runs, from, to)
    expect(movers.map((m) => m.symptom)).toEqual(['Rise'])
    expect(movers[0].count).toBe(3)
    expect(movers[0].dominantOutcome).toBe('hardware')
    expect(movers[0].spark.length).toBe(7)
    expect(movers[0].score).toBeGreaterThan(0)
  })

  it('returns [] when nothing moves materially', () => {
    expect(buildMovers([mk('Flat', 'software', '2026-05-27T01:00:00Z')], from, to)).toEqual([])
  })
  it('surfaces a NEW symptom that appears at low count (0 -> 2)', () => {
    const runs: RunSummary[] = [
      mk('NewSym', 'software', '2026-06-01T01:00:00Z'),
      mk('NewSym', 'software', '2026-06-01T05:00:00Z'),
    ]
    const movers = buildMovers(runs, from, to)
    expect(movers.map((m) => m.symptom)).toEqual(['NewSym'])
    expect(movers[0].count).toBe(2)
    expect(movers[0].score).toBeGreaterThan(0)
  })
  it('returns [] for empty input', () => {
    expect(buildMovers([], from, to)).toEqual([])
  })
})
