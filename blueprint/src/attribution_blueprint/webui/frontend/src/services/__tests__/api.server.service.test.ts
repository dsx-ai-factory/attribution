// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { RunDetail, RunSummary } from '@/interfaces/attribution'
import { ApiServerService } from '@/services/api.server.service'
import { HttpError } from '@/services/http-client.service'

const { httpGetMock } = vi.hoisted(() => ({ httpGetMock: vi.fn() }))

vi.mock('@/services/http-client.service', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/services/http-client.service')>(),
  httpGet: httpGetMock,
}))

const run = (id: string): RunSummary => ({
  id,
  jobId: id,
  workload: id,
  attempt: 1,
  attemptUid: id,
  user: 'alice',
  status: 'FAILED',
  nodes: 1,
  nodeIds: ['node-a'],
  elapsedH: 1,
  gpusPerNode: 8,
  gpuCount: 8,
  endTime: '2026-05-01T01:00:00Z',
  startTime: '2026-05-01T00:00:00Z',
  outcome: 'hardware',
  verdictLabel: 'Hardware',
  detail: '',
  decisionGraphState: 'absent',
  rootSymptom: { symptom: 'Xid79', rule: 'xid', time: '', line: '' },
  symptoms: [],
  responsibleNodes: ['node-a'],
  pipeline: '1',
  service: '2',
})

beforeEach(() => {
  httpGetMock.mockReset()
})

describe('ApiServerService', () => {
  it('loads every page of the unfiltered run collection', async () => {
    httpGetMock
      .mockResolvedValueOnce({ runs: [run('a'), run('b')], total: 3 })
      .mockResolvedValueOnce({ runs: [run('c')], total: 3 })

    await expect(new ApiServerService().getAllRuns()).resolves.toEqual([
      run('a'),
      run('b'),
      run('c'),
    ])
    expect(httpGetMock).toHaveBeenNthCalledWith(1, '/runs', {
      page: 0,
      pageSize: 100,
      status: 'all',
    })
    expect(httpGetMock).toHaveBeenNthCalledWith(2, '/runs', {
      page: 1,
      pageSize: 100,
      status: 'all',
    })
  })

  it('stops paging if the server returns an empty page', async () => {
    httpGetMock.mockResolvedValue({ runs: [], total: 5 })

    await expect(new ApiServerService().getAllRuns()).resolves.toEqual([])
    expect(httpGetMock).toHaveBeenCalledTimes(1)
  })

  it('forwards fleet filters to the paginated endpoint', async () => {
    httpGetMock.mockResolvedValue({ runs: [run('a')], total: 1 })
    const filter = {
      from: '2026-05-01',
      to: '2026-05-02',
      outcome: 'hardware' as const,
      q: 'job',
      status: 'failed' as const,
      symptom: 'Xid79',
      node: 'node-a',
      rule: 'xid',
      user: 'alice',
      axis: 'symptom' as const,
      metric: 'count' as const,
    }

    await expect(new ApiServerService().getRuns(filter)).resolves.toEqual([run('a')])
    expect(httpGetMock).toHaveBeenCalledWith('/runs', {
      from: '2026-05-01',
      to: '2026-05-02',
      outcome: 'hardware',
      q: 'job',
      status: 'failed',
      symptom: 'Xid79',
      node: 'node-a',
      rule: 'xid',
      user: 'alice',
      pageSize: 10,
    })
  })

  it('encodes run IDs and exposes metadata', async () => {
    const detail: RunDetail = {
      ...run('a/b'),
      log: [],
      attribution: {
        rootSymptom: { state: 'absent', data: null },
        symptomSites: { state: 'empty', data: {} },
        decisionGraph: { state: 'null', data: null },
      },
    }
    httpGetMock.mockResolvedValueOnce(detail).mockResolvedValueOnce({ windowedRuns: 1 })
    const api = new ApiServerService()

    await expect(api.getRun('a/b')).resolves.toEqual(detail)
    await expect(api.getMeta()).resolves.toEqual({ windowedRuns: 1 })
    expect(httpGetMock).toHaveBeenNthCalledWith(1, '/runs/a%2Fb')
    expect(httpGetMock).toHaveBeenNthCalledWith(2, '/meta')
  })

  it('returns undefined when the requested run does not exist', async () => {
    httpGetMock.mockRejectedValue(new HttpError(404, 'Run not found'))

    await expect(new ApiServerService().getRun('missing')).resolves.toBeUndefined()
  })

  it.each([new HttpError(500, 'Service unavailable'), new TypeError('Failed to fetch')])(
    'preserves other request failures: %s',
    async (error) => {
      httpGetMock.mockRejectedValue(error)

      await expect(new ApiServerService().getRun('run')).rejects.toBe(error)
    },
  )
})
