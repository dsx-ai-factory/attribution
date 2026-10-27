// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import type { RunSummary } from '@/interfaces/attribution'
import { useDashboard } from '@/hooks/use-dashboard'
import { useMeta } from '@/hooks/use-meta'
import { useNodeRuns } from '@/hooks/use-node-runs'
import { useRuns } from '@/hooks/use-runs'
import { useRunsTable } from '@/hooks/use-runs-table'
import { useNav } from '@/hooks/use-nav'
import { useDocumentTitle } from '@/hooks/use-document-title'
import { useECharts } from '@/hooks/use-echarts'
import { useSymptomDocumentation } from '@/hooks/use-symptom-documentation'
import { useCopyToClipboard } from '@/hooks/use-copy-to-clipboard'

const mocks = vi.hoisted(() => ({
  useSWR: vi.fn(),
  httpGet: vi.fn(),
  useApi: vi.fn(),
  useNavigate: vi.fn(),
  echartsInit: vi.fn(),
  registerTheme: vi.fn(),
  chartTheme: vi.fn(() => ({ textStyle: { fontFamily: 'token(--app-font-sans)' } })),
}))

vi.mock('swr', () => ({ default: mocks.useSWR }))
vi.mock('@/services/http-client.service', () => ({ httpGet: mocks.httpGet }))
vi.mock('@/services/api-context', () => ({ useApi: mocks.useApi }))
vi.mock('react-router-dom', () => ({ useNavigate: mocks.useNavigate }))
vi.mock('echarts', () => ({ init: mocks.echartsInit, registerTheme: mocks.registerTheme }))
vi.mock('@/lib/tokens', () => ({ chartTheme: mocks.chartTheme }))

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
    const mutate = vi.fn()
    mocks.useSWR.mockReturnValue({
      data: { metrics: { windowedRuns: 2 } },
      isLoading: false,
      error: undefined,
      mutate,
    })

    const { result } = renderHook(() => useDashboard({
      from: '2026-05-01',
      to: '2026-05-02',
      outcome: 'hardware',
    }))
    const [key, fetcher, options] = mocks.useSWR.mock.calls[0]
    mocks.httpGet.mockResolvedValue({ metrics: { windowedRuns: 2 } })

    expect(key).toEqual(['dashboard', '2026-05-01', '2026-05-02', 'hardware'])
    // A view's dashboard is its own: nothing from the previous view stands in while it loads.
    expect(options).toBeUndefined()
    await expect(fetcher()).resolves.toEqual({ metrics: { windowedRuns: 2 } })
    expect(mocks.httpGet).toHaveBeenCalledWith('/dashboard', {
      from: '2026-05-01',
      to: '2026-05-02',
      outcome: 'hardware',
    })
    expect(result.current.dashboard).toEqual({ metrics: { windowedRuns: 2 } })
    expect(result.current.error).toBeNull()
    // The page's Refresh re-fetches this view through the key's own mutate.
    expect(result.current.mutate).toBe(mutate)
  })

  it('reads the backend it is bound to from /meta under one static key', async () => {
    const mutate = vi.fn()
    mocks.useSWR.mockReturnValue({ data: { database: 'demo.db' }, isLoading: false, error: undefined, mutate })
    mocks.httpGet.mockResolvedValue({ database: 'demo.db' })

    const { result } = renderHook(() => useMeta())
    const [key, fetcher] = mocks.useSWR.mock.calls[0]

    expect(key).toBe('meta')
    await expect(fetcher()).resolves.toEqual({ database: 'demo.db' })
    expect(mocks.httpGet).toHaveBeenCalledWith('/meta')
    expect(result.current.meta).toEqual({ database: 'demo.db' })
    expect(result.current.mutate).toBe(mutate)
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
    const mutate = vi.fn()
    mocks.useSWR.mockReturnValue({ data: undefined, isLoading: true, error: undefined, mutate })
    const { result } = renderHook(() => useRunsTable({
      filter,
      outcomes: ['hardware', 'software'],
      pageByOutcome: { hardware: 2 },
      pageSize: 25,
    }))
    expect(result.current.mutate).toBe(mutate)
    const [, fetcher, options] = mocks.useSWR.mock.calls[0]
    // A view's pages are its own: nothing from the previous view stands in while they load.
    expect(options).toBeUndefined()
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

  it('titles the tab with the shown record first and follows it as it changes', () => {
    const { rerender } = renderHook(({ subject }: { subject?: string }) => useDocumentTitle('Verdict', subject), {
      initialProps: {},
    })
    expect(document.title).toBe('Verdict · Attribution')

    rerender({ subject: 'team/job' })
    expect(document.title).toBe('team/job · Verdict · Attribution')
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

    // Every chart paints from the one token theme, registered before the first chart.
    expect(mocks.registerTheme).toHaveBeenCalledWith('attribution', { textStyle: { fontFamily: 'token(--app-font-sans)' } })
    expect(mocks.echartsInit).toHaveBeenCalledWith(ref.current, 'attribution')
    expect(chart.setOption).toHaveBeenCalledWith({ title: { text: 'one' } }, { notMerge: true })
    expect(observe).toHaveBeenCalledWith(ref.current)
    handlers.get('click')?.({ name: 'first' })
    expect(firstClick).toHaveBeenCalledWith({ name: 'first' })

    rerender({ option: { title: { text: 'two' } }, click: secondClick })
    // A new option replaces the chart's last one whole, never merges into it.
    expect(chart.setOption).toHaveBeenLastCalledWith({ title: { text: 'two' } }, { notMerge: true })
    expect(chart.setOption).toHaveBeenCalledTimes(2)
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

    // A second chart reuses the registered theme.
    renderHook(() => useECharts(ref, { title: { text: 'three' } })).unmount()
    expect(mocks.echartsInit).toHaveBeenLastCalledWith(ref.current, 'attribution')
    expect(mocks.registerTheme).toHaveBeenCalledTimes(1)
    expect(mocks.chartTheme).toHaveBeenCalledTimes(1)
  })
})

describe('copy state', () => {
  afterEach(() => {
    vi.useRealTimers()
    Reflect.deleteProperty(navigator, 'clipboard')
  })

  const stubWriteText = (writeText: () => Promise<void>) =>
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })

  it('keeps a repeated copy confirmed for the full delay after the last one', async () => {
    vi.useFakeTimers()
    stubWriteText(() => Promise.resolve())
    const { result } = renderHook(() => useCopyToClipboard(1000))

    await act(async () => { await result.current.copy('first') })
    expect(result.current.copied).toBe(true)
    await act(async () => { vi.advanceTimersByTime(600) })
    await act(async () => { await result.current.copy('second') })
    await act(async () => { vi.advanceTimersByTime(600) })
    expect(result.current.copied).toBe(true)
    await act(async () => { vi.advanceTimersByTime(400) })
    expect(result.current.copied).toBe(false)
  })

  it('announces a copy that failed without confirming it, then resets', async () => {
    vi.useFakeTimers()
    // jsdom has no execCommand either, so the legacy fallback fails too.
    stubWriteText(() => Promise.reject(new Error('denied')))
    const { result } = renderHook(() => useCopyToClipboard(1000))
    expect(result.current.announcement).toBe('')

    await act(async () => { await result.current.copy('node-a') })
    expect(result.current.copied).toBe(false)
    expect(result.current.announcement).toBe('Copy failed')
    await act(async () => { vi.advanceTimersByTime(1000) })
    expect(result.current.announcement).toBe('')

    // The next copy that lands is confirmed as usual.
    stubWriteText(() => Promise.resolve())
    await act(async () => { await result.current.copy('node-a') })
    expect([result.current.copied, result.current.announcement]).toEqual([true, 'Copied'])
  })

  it('starts no timer for a copy that lands after the control unmounted', async () => {
    vi.useFakeTimers()
    let finishWrite = () => {}
    stubWriteText(() => new Promise<void>((resolve) => { finishWrite = resolve }))
    const { result, unmount } = renderHook(() => useCopyToClipboard(1000))

    let pending: Promise<boolean> = Promise.resolve(false)
    act(() => { pending = result.current.copy('node-a') })
    unmount()
    finishWrite()

    await expect(pending).resolves.toBe(true)
    expect(vi.getTimerCount()).toBe(0)
  })
})
