// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { describe, it, expect, beforeEach } from 'vitest'
import type { BootstrapPayload } from '@/interfaces/meta'
import { ApiBootstrapService } from '@/services/api.bootstrap.service'

const blob: BootstrapPayload = {
  runs: [
    { id: 'a', jobId: 'a', workload: '—', attempt: 1, attemptUid: 'a', user: 'u', status: 'FAILED',
      nodes: 1, nodeIds: ['n1'], elapsedH: 1, gpusPerNode: 8, gpuCount: 8, endTime: '2026-05-02T00:00:00Z',
      startTime: '2026-05-01T23:00:00Z', outcome: 'hardware', verdictLabel: '', detail: '',
      decisionGraphState: 'absent',
      rootSymptom: { symptom: 'GpuFell', rule: 'r', time: '', line: '' }, symptoms: [],
      responsibleNodes: ['n1'], pipeline: '', service: '', log: [], attribution: {
        rootSymptom: { state: 'absent', data: null },
        symptomSites: { state: 'absent', data: null },
        decisionGraph: { state: 'absent', data: null },
      } },
    { id: 'b', jobId: 'b', workload: '—', attempt: 1, attemptUid: 'b', user: 'u', status: 'COMPLETED',
      nodes: 1, nodeIds: ['n2'], elapsedH: 1, gpusPerNode: 8, gpuCount: 8, endTime: '2026-05-02T00:00:00Z',
      startTime: '2026-05-01T23:00:00Z', outcome: 'pass', verdictLabel: '', detail: '',
      decisionGraphState: 'absent',
      rootSymptom: { symptom: 'Clean', rule: 'r', time: '', line: '' }, symptoms: [],
      responsibleNodes: [], pipeline: '', service: '', log: [], attribution: {
        rootSymptom: { state: 'null', data: null },
        symptomSites: { state: 'empty', data: {} },
        decisionGraph: { state: 'absent', data: null },
      } },
  ],
  meta: { versions: { schema: '1', pipeline: null, service: null },
          dateBounds: { min: '2026-05-02', max: '2026-05-02' }, windowedRuns: 2, dbTotalRuns: 2,
          returnedRuns: 2, database: null },
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
    expect(await ds.getRun('a')).toEqual(blob.runs[0])
    expect((await ds.getMeta()).windowedRuns).toBe(2)
  })

  it('keeps source observation IDs exact when parsing the injected run details', async () => {
    document.body.innerHTML = `<script type="application/json" id="attr-bootstrap">{
      "runs": [{
        "id": "precise",
        "gpuCount": 8,
        "elapsedH": 0.125,
        "attribution": {
          "rootSymptom": {
            "state": "available",
            "data": { "source": { "observation_id": "9223372036854775807", "pos": 0 } }
          },
          "symptomSites": {
            "state": "available",
            "data": { "fault": [{ "source": { "observation_id": "-6139371663235533333", "pos": null } }] }
          }
        }
      }],
      "meta": ${JSON.stringify(blob.meta)}
    }</script>`

    const detail = await new ApiBootstrapService().getRun('precise')

    expect(detail?.attribution.rootSymptom.data?.source.observation_id).toBe('9223372036854775807')
    expect(detail?.attribution.symptomSites.data?.fault[0].source.observation_id).toBe('-6139371663235533333')
    expect(detail?.attribution.rootSymptom.data?.source.pos).toBe(0)
    expect(detail?.attribution.symptomSites.data?.fault[0].source.pos).toBeNull()
    expect(detail?.gpuCount).toBe(8)
    expect(detail?.elapsedH).toBe(0.125)
  })
})
