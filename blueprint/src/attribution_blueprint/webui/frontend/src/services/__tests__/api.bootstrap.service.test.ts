// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest'
import { ApiBootstrapService } from '@/services/api.bootstrap.service'

const blob = {
  runs: [
    { id: 'a', jobId: 'a', workload: '—', attempt: 1, attemptUid: 'a', user: 'u', status: 'FAILED',
      nodes: 1, nodeIds: ['n1'], elapsedH: 1, gpusPerNode: 8, gpuCount: 8, endTime: '2026-05-02T00:00:00Z',
      startTime: '2026-05-01T23:00:00Z', outcome: 'hardware', verdictLabel: '', detail: '',
      rootSymptom: { symptom: 'GpuFell', rule: 'r', time: '', line: '' }, symptoms: [],
      responsibleNodes: ['n1'], pipeline: '', service: '', log: [] },
    { id: 'b', jobId: 'b', workload: '—', attempt: 1, attemptUid: 'b', user: 'u', status: 'COMPLETED',
      nodes: 1, nodeIds: ['n2'], elapsedH: 1, gpusPerNode: 8, gpuCount: 8, endTime: '2026-05-02T00:00:00Z',
      startTime: '2026-05-01T23:00:00Z', outcome: 'pass', verdictLabel: '', detail: '',
      rootSymptom: { symptom: 'Clean', rule: 'r', time: '', line: '' }, symptoms: [],
      responsibleNodes: [], pipeline: '', service: '', log: [] },
  ],
  meta: { versions: { schema: '1', pipeline: null, service: null },
          dateBounds: { min: '2026-05-02', max: '2026-05-02' }, totalRuns: 2, returnedRuns: 2 },
}

describe('ApiBootstrapService', () => {
  beforeEach(() => {
    document.body.innerHTML =
      `<script type="application/json" id="attr-bootstrap">${JSON.stringify(blob)}</script>`
  })

  it('returns all runs and applies the failed-only filter', async () => {
    const ds = new ApiBootstrapService()
    const all = await ds.getRuns({ status: 'all', axis: 'symptom', metric: 'count' })
    expect(all).toHaveLength(2)
    const failed = await ds.getRuns({ status: 'failed', axis: 'symptom', metric: 'count' })
    expect(failed.map((r) => r.id)).toEqual(['a'])
  })

  it('getAllRuns returns the FULL set (incl. pass) regardless of filter', async () => {
    const ds = new ApiBootstrapService()
    const all = await ds.getAllRuns()
    expect(all.map((r) => r.id)).toEqual(['a', 'b'])
    // The pass run must be present — getAllRuns never applies a filter.
    expect(all.some((r) => r.outcome === 'pass')).toBe(true)
  })

  it('resolves a single run and meta', async () => {
    const ds = new ApiBootstrapService()
    expect((await ds.getRun('a'))?.id).toBe('a')
    expect((await ds.getMeta()).totalRuns).toBe(2)
  })
})
