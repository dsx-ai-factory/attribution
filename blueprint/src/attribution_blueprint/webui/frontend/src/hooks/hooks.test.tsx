// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { renderHook } from '@testing-library/react'
import type { RunSummary } from '@/interfaces/attribution'
import { useDashboard } from '@/hooks/use-dashboard'
import { useMeta } from '@/hooks/use-meta'
import { useNodeRuns } from '@/hooks/use-node-runs'
import { useRuns } from '@/hooks/use-runs'
import { useRunsTable } from '@/hooks/use-runs-table'
import { useNav } from '@/hooks/use-nav'
import { useECharts } from '@/hooks/use-echarts'
import { useSymptomDocumentation } from '@/hooks/use-symptom-documentation'

const mocks = vi.hoisted(() => ({
  useSWR: vi.fn(),
  httpGet: vi.fn(),
  useApi: vi.fn(),
  useNavigate: vi.fn(),
  echartsInit: vi.fn(),
}))

vi.mock('swr', () => ({ default: mocks.useSWR }))
vi.mock('@/services/http-client.service', () => ({ httpGet: mocks.httpGet }))
vi.mock('@/services/api-context', () => ({ useApi: mocks.useApi }))
vi.mock('react-router-dom', () => ({ useNavigate: mocks.useNavigate }))
vi.mock('echarts', () => ({ init: mocks.echartsInit }))

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
  vi.clearAllMocks()
  vi.stubEnv('VITE_SYMPTOM_DOCS_URL', 'https://reference.example/api/docs/symptoms')
  mocks.useSWR.mockReturnValue({
    data: undefined,
    isLoading: false,
    error: undefined,
    mutate: vi.fn(),
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('data hooks', () => {
  it('requests the reference API directly and shares its unmodified response by URL', async () => {
    const catalog = { symptoms: { 'recorded-id': { id: 'recorded-id', title: 'Title', doc_url: 'https://docs.example/original-path/' } } }
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(catalog)))
    vi.stubGlobal('fetch', fetch)
    renderHook(() => useSymptomDocumentation())
    const [key, fetcher, options] = mocks.useSWR.mock.calls[0]
    expect(key).toBe('https://reference.example/api/docs/symptoms')
    await expect(fetcher(key)).resolves.toEqual(catalog)
    expect(fetch).toHaveBeenCalledWith(key, expect.objectContaining({ headers: { Accept: 'application/json' }, redirect: 'error', signal: expect.any(AbortSignal) }))
    expect(options).toMatchObject({ revalidateOnFocus: false, shouldRetryOnError: false })
    expect(mocks.httpGet).not.toHaveBeenCalled()
  })

  it('disables documentation requests without a recorded ID', () => {
    renderHook(() => useSymptomDocumentation(false))
    expect(mocks.useSWR.mock.calls[0][0]).toBeNull()
  })

  it.each([undefined, '', '/api/docs/symptoms', 'not a URL'])('disables documentation requests for absent or invalid configuration %s', (url) => {
    vi.stubEnv('VITE_SYMPTOM_DOCS_URL', url)
    renderHook(() => useSymptomDocumentation())
    expect(mocks.useSWR.mock.calls[0][0]).toBeNull()
  })

  it('allows an explicitly configured HTTP endpoint in a built local preview', () => {
    vi.stubEnv('DEV', false)
    vi.stubEnv('VITE_SYMPTOM_DOCS_URL', 'http://reference.example/api/docs/symptoms')
    expect(window.location.hostname).toBe('localhost')
    renderHook(() => useSymptomDocumentation())
    expect(mocks.useSWR.mock.calls[0][0]).toBe('http://reference.example/api/docs/symptoms')
  })

  it('surfaces reference API failures through SWR without a fallback catalog', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(new Response('', { status: 503 })).mockRejectedValueOnce(new DOMException('Timed out', 'TimeoutError'))
    vi.stubGlobal('fetch', fetch)
    renderHook(() => useSymptomDocumentation())
    const [key, fetcher] = mocks.useSWR.mock.calls[0]
    await expect(fetcher(key)).rejects.toThrow('503')
    await expect(fetcher(key)).rejects.toThrow('Timed out')
    expect(mocks.httpGet).not.toHaveBeenCalled()
  })

  it('builds the dashboard cache key and request', async () => {
    mocks.useSWR.mockReturnValue({
      data: { metrics: { windowedRuns: 2 } },
      isLoading: false,
      error: undefined,
    })

    const { result } = renderHook(() => useDashboard({
      from: '2026-05-01',
      to: '2026-05-02',
      outcome: 'hardware',
    }))
    const [key, fetcher] = mocks.useSWR.mock.calls[0]
    mocks.httpGet.mockResolvedValue({ metrics: { windowedRuns: 2 } })

    expect(key).toEqual(['dashboard', '2026-05-01', '2026-05-02', 'hardware'])
    await expect(fetcher()).resolves.toEqual({ metrics: { windowedRuns: 2 } })
    expect(mocks.httpGet).toHaveBeenCalledWith('/dashboard', {
      from: '2026-05-01',
      to: '2026-05-02',
      outcome: 'hardware',
    })
    expect(result.current.dashboard).toEqual({ metrics: { windowedRuns: 2 } })
    expect(result.current.error).toBeNull()
  })

  it('reads the backend it is bound to from /meta under one static key', async () => {
    mocks.useSWR.mockReturnValue({ data: { database: 'demo.db' }, isLoading: false, error: undefined })
    mocks.httpGet.mockResolvedValue({ database: 'demo.db' })

    const { result } = renderHook(() => useMeta())
    const [key, fetcher] = mocks.useSWR.mock.calls[0]

    expect(key).toBe('meta')
    await expect(fetcher()).resolves.toEqual({ database: 'demo.db' })
    expect(mocks.httpGet).toHaveBeenCalledWith('/meta')
    expect(result.current.meta).toEqual({ database: 'demo.db' })
  })

  it('disables node requests without an ID and paginates enabled requests', async () => {
    renderHook(() => useNodeRuns('', false))
    expect(mocks.useSWR.mock.calls[0][0]).toBeNull()

    mocks.useSWR.mockClear()
    renderHook(() => useNodeRuns('node-a'))
    const [key, fetcher] = mocks.useSWR.mock.calls[0]
    mocks.httpGet
      .mockResolvedValueOnce({ runs: [run('a')], total: 2 })
      .mockResolvedValueOnce({ runs: [run('b')], total: 2 })

    expect(key).toEqual(['node-runs', 'node-a'])
    await expect(fetcher()).resolves.toEqual([run('a'), run('b')])
    expect(mocks.httpGet).toHaveBeenNthCalledWith(2, '/runs', {
      node: 'node-a',
      status: 'all',
      page: 1,
      pageSize: 100,
    })
  })

  it('combines runs and metadata through the shared runs cache', async () => {
    const api = {
      getAllRuns: vi.fn().mockResolvedValue([run('a')]),
      getMeta: vi.fn().mockResolvedValue({ windowedRuns: 1 }),
    }
    mocks.useApi.mockReturnValue(api)
    const mutate = vi.fn()
    mocks.useSWR.mockReturnValue({
      data: { runs: [run('a')], meta: { windowedRuns: 1 } },
      isLoading: false,
      error: undefined,
      mutate,
    })

    const { result } = renderHook(() => useRuns())
    const [key, fetcher] = mocks.useSWR.mock.calls[0]

    expect(key).toBe('runs')
    await expect(fetcher()).resolves.toEqual({ runs: [run('a')], meta: { windowedRuns: 1 } })
    expect(result.current).toMatchObject({
      runs: [run('a')],
      meta: { windowedRuns: 1 },
      isLoading: false,
      mutate,
    })
  })

  it('loads one page per requested outcome and drops empty groups', async () => {
    const filter = {
      status: 'failed' as const,
      axis: 'symptom' as const,
      metric: 'count' as const,
      symptom: 'Xid79',
    }
    renderHook(() => useRunsTable({
      filter,
      outcomes: ['hardware', 'software'],
      pageByOutcome: { hardware: 2 },
      pageSize: 25,
    }))
    const [, fetcher] = mocks.useSWR.mock.calls[0]
    mocks.httpGet
      .mockResolvedValueOnce({ runs: [run('a')], total: 3, page: 2, pageSize: 25 })
      .mockResolvedValueOnce({ runs: [], total: 0, page: 0, pageSize: 25 })

    await expect(fetcher()).resolves.toEqual([{
      outcome: 'hardware',
      runs: [run('a')],
      total: 3,
      page: 2,
      pageSize: 25,
    }])
    expect(mocks.httpGet).toHaveBeenNthCalledWith(1, '/runs', expect.objectContaining({
      outcome: 'hardware',
      page: 2,
      pageSize: 25,
      symptom: 'Xid79',
    }))
  })
})

describe('navigation and chart hooks', () => {
  it('navigates through named routes', () => {
    const navigate = vi.fn()
    mocks.useNavigate.mockReturnValue(navigate)
    const { result } = renderHook(() => useNav())

    result.current('node', { id: 'node/a' })

    expect(navigate).toHaveBeenCalledWith('/node/node%2Fa')
  })

  it('initializes, updates, dispatches, resizes, and disposes a chart', () => {
    const handlers = new Map<string, (event: unknown) => void>()
    const chart = {
      resize: vi.fn(),
      setOption: vi.fn(),
      dispose: vi.fn(),
      on: vi.fn((name: string, callback: (event: unknown) => void) => handlers.set(name, callback)),
    }
    mocks.echartsInit.mockReturnValue(chart)
    const observe = vi.fn()
    const disconnect = vi.fn()
    vi.stubGlobal('ResizeObserver', class {
      observe = observe
      disconnect = disconnect
    })
    const ref = { current: document.createElement('div') }
    const firstClick = vi.fn()
    const secondClick = vi.fn()
    const mouseover = vi.fn()
    const mouseout = vi.fn()

    const { rerender, unmount } = renderHook(
      ({ option, click }) => useECharts(ref, option, { click, mouseover, mouseout }),
      { initialProps: { option: { title: { text: 'one' } }, click: firstClick } },
    )

    expect(mocks.echartsInit).toHaveBeenCalledWith(ref.current)
    expect(chart.setOption).toHaveBeenCalledWith({ title: { text: 'one' } }, { notMerge: false })
    expect(observe).toHaveBeenCalledWith(ref.current)
    handlers.get('click')?.({ name: 'first' })
    expect(firstClick).toHaveBeenCalledWith({ name: 'first' })

    rerender({ option: { title: { text: 'two' } }, click: secondClick })
    handlers.get('click')?.({ name: 'second' })
    handlers.get('mouseover')?.({ name: 'over' })
    handlers.get('mouseout')?.({ name: 'out' })
    expect(secondClick).toHaveBeenCalledWith({ name: 'second' })
    expect(mouseover).toHaveBeenCalledWith({ name: 'over' })
    expect(mouseout).toHaveBeenCalledWith({ name: 'out' })

    window.dispatchEvent(new Event('resize'))
    expect(chart.resize).toHaveBeenCalled()
    unmount()
    expect(disconnect).toHaveBeenCalled()
    expect(chart.dispose).toHaveBeenCalled()
  })
})
