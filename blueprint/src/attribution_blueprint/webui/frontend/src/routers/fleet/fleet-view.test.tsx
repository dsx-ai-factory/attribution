// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { FleetFilter } from '@/interfaces/filter'
import { FleetView } from '@/routers/fleet'

const mocks = vi.hoisted(() => ({
  searchParams: new URLSearchParams(),
  setSearchParams: vi.fn(),
  useDashboard: vi.fn(),
  useRunsTable: vi.fn(),
  useMeta: vi.fn(),
  refetchDashboard: vi.fn(),
  refetchRuns: vi.fn(),
  refetchMeta: vi.fn(),
  datePickerProps: vi.fn(),
  kpiProps: vi.fn(),
  paretoProps: vi.fn(),
  breakdownProps: vi.fn(),
  moversProps: vi.fn(),
  runsProps: vi.fn(),
  resizeCallbacks: [] as (() => void)[],
  resizeObserved: [] as Element[],
}))

vi.mock('react-router-dom', async (importOriginal) => ({
  ...await importOriginal<typeof import('react-router-dom')>(),
  useSearchParams: () => [mocks.searchParams, mocks.setSearchParams],
}))
vi.mock('@/hooks/use-dashboard', () => ({ useDashboard: mocks.useDashboard }))
vi.mock('@/hooks/use-runs-table', () => ({ useRunsTable: mocks.useRunsTable }))
vi.mock('@/hooks/use-meta', () => ({ useMeta: mocks.useMeta }))
vi.mock('@/components/ui/toggle-group', () => ({
  ToggleGroup: ({ children, onValueChange }: {
    children: ReactNode
    onValueChange: (value: string) => void
  }) => (
    <div>
      {children}
      <button onClick={() => onValueChange('7d')}>Choose 7d</button>
      <button onClick={() => onValueChange('custom')}>Choose custom</button>
      <button onClick={() => onValueChange('')}>Reapply current</button>
    </div>
  ),
  ToggleGroupItem: ({ children }: { children: ReactNode }) => <span>{children}</span>,
}))
vi.mock('@/routers/fleet/components/date-picker', () => ({
  DatePicker: (props: { label: string; value?: Date; onChange: (date: Date | undefined) => void }) => {
    mocks.datePickerProps(props)
    return <button onClick={() => props.onChange(new Date('2026-05-02T00:00:00Z'))}>Set date</button>
  },
}))
vi.mock('@/routers/fleet/components/kpi-strip', () => ({
  KpiStrip: (props: {
    setFilter: (next: FleetFilter, options?: { revealRuns?: boolean }) => void
    hwFailures: number
  }) => {
    mocks.kpiProps(props)
    return (
      <button onClick={() => props.setFilter({
        status: 'failed',
        outcome: 'hardware',
        axis: 'symptom',
        metric: 'count',
      }, { revealRuns: true })}>
        KPI filter
      </button>
    )
  },
}))
vi.mock('@/routers/fleet/components/pareto', () => ({
  ParetoCard: (props: object) => {
    mocks.paretoProps(props)
    return <div>Pareto content</div>
  },
}))
vi.mock('@/routers/fleet/components/breakdown-by-day', () => ({
  BreakdownByDay: (props: object) => {
    mocks.breakdownProps(props)
    return <div>Breakdown content</div>
  },
}))
vi.mock('@/routers/fleet/components/movers-strip', () => ({
  MoversStrip: (props: object) => {
    mocks.moversProps(props)
    return <div>Movers content</div>
  },
}))
vi.mock('@/routers/fleet/components/runs-section', () => ({
  RunsSection: (props: {
    setPageSize: (size: number) => void
    setGroupPage: (outcome: 'hardware', page: number) => void
  }) => {
    mocks.runsProps(props)
    // The real section is this focusable region: the page scrolls to and focuses it by id.
    return (
      <section id="runs-section" aria-label="Runs" tabIndex={-1}>
        <button onClick={() => props.setPageSize(25)}>Set page size</button>
        <button onClick={() => props.setGroupPage('hardware', 1)}>Set hardware page</button>
      </section>
    )
  },
}))

const dashboard = {
  metrics: {
    hwMtbi: 10,
    combinedMtbi: 8,
    gpuHoursRaw: 80,
    hwFailures: 2,
    swFailures: 1,
    mixFailures: 0,
    combinedFailures: 3,
    gpuCount: 16,
    windowedRuns: 4,
  },
  daily: [{ day: '2026-05-01', hardware: 2 }],
  hwBreakdown: { days: ['2026-05-01'], series: [], total: 2 },
  swBreakdown: { days: ['2026-05-01'], series: [], total: 1 },
  ledgers: {
    symptom: [{ symptom: 'Xid79', count: 2, nodeHours: 8, outcomes: { hardware: 2 } }],
    node: [],
    user: [],
    rule: [],
    outcome: [{ outcome: 'hardware', count: 2, nodeHours: 8 }],
  },
  movers: [{ symptom: 'Xid79', count: 2, dominantOutcome: 'hardware', spark: [0, 2], score: 2 }],
  meta: {
    versions: { schema: '2' },
    dateBounds: { min: '2026-05-01', max: '2026-05-03' },
    dbTotalRuns: 1204,
  },
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.searchParams = new URLSearchParams()
  Element.prototype.scrollIntoView = vi.fn()
  mocks.resizeCallbacks = []
  mocks.resizeObserved = []
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback: () => void) { mocks.resizeCallbacks.push(callback) }
    observe(element: Element) { mocks.resizeObserved.push(element) }
    disconnect() {}
  })
  mocks.useDashboard.mockReturnValue({ dashboard, isLoading: false, error: null, mutate: mocks.refetchDashboard })
  mocks.useRunsTable.mockReturnValue({
    groups: [],
    total: 4,
    visibleRuns: [],
    isLoading: false,
    error: null,
    mutate: mocks.refetchRuns,
  })
  mocks.useMeta.mockReturnValue({ meta: null, mutate: mocks.refetchMeta })
})

afterEach(cleanup)

const scrollIntoView = () => vi.mocked(Element.prototype.scrollIntoView)
const loaded = { isLoading: false, error: null, mutate: mocks.refetchDashboard }
const runsLoaded = { groups: [], total: 4, visibleRuns: [], isLoading: false, error: null, mutate: mocks.refetchRuns }

describe('FleetView', () => {
  it('tells the page how tall its sticky window band is while it is mounted', () => {
    const root = document.documentElement
    const { unmount } = render(<FleetView />)
    const band = screen.getByRole('button', { name: 'Refresh' }).closest('.sticky')!
    expect(mocks.resizeObserved).toEqual([band])
    expect(root.style.getPropertyValue('--page-sticky-height')).toBe('0px')
    // Its own controls are stuck chrome: focusing one never scrolls the page (globals.css).
    expect(band.hasAttribute('data-stuck-chrome')).toBe(true)

    // The bar wraps onto a second line on a narrow window: the height follows.
    Object.defineProperty(band, 'offsetHeight', { configurable: true, value: 90 })
    act(() => mocks.resizeCallbacks.forEach((callback) => callback()))
    expect(root.style.getPropertyValue('--page-sticky-height')).toBe('90px')

    unmount()
    expect(root.style.getPropertyValue('--page-sticky-height')).toBe('')
  })

  it('composes dashboard data and synchronizes filter controls with the URL', () => {
    render(<FleetView />)

    expect(mocks.useDashboard).toHaveBeenCalledWith({
      from: undefined,
      to: undefined,
      outcome: undefined,
    })
    expect(mocks.kpiProps).toHaveBeenCalledWith(expect.objectContaining({ hwFailures: 2 }))
    expect(mocks.paretoProps).toHaveBeenCalledWith(expect.objectContaining({
      symptomLedger: dashboard.ledgers.symptom,
    }))
    // The root page is titled by itself, for assistive tech and the tab alone:
    // no visible title, no trail above it, and no footer restating versions.
    expect(screen.getByRole('heading', { level: 1, name: 'Fleet' }).className).toBe('sr-only')
    expect(document.title).toBe('Fleet · Attribution')
    expect(screen.queryByRole('navigation')).toBeNull()
    expect(screen.queryByText(/Attribution v0\.3\.0|schema/)).toBeNull()
    // The window controls lead on screen: a bordered bar on the card surface,
    // its controls on the card inset, stuck under the app bar in a band of page
    // background that takes over the page's gutter and top padding.
    const band = screen.getByRole('button', { name: 'Refresh' }).closest('.sticky')!
    expect([...band.classList]).toEqual(expect.arrayContaining(['top-(--app-bar-height)', 'bg-background-secondary', '-mx-page-x', 'px-page-x', '-mt-page-top', 'pt-page-top']))
    expect([...band.firstElementChild!.classList]).toEqual(expect.arrayContaining(['rounded-md', 'border', 'bg-background', 'px-card-x', 'py-1.5']))
    expect(band.firstElementChild!.contains(screen.getByRole('button', { name: 'Choose 7d' }))).toBe(true)
    // The server reported movers, so the section is there.
    expect(mocks.moversProps).toHaveBeenCalledWith(expect.objectContaining({ movers: dashboard.movers, windowLabel: 'all time' }))

    fireEvent.click(screen.getByRole('button', { name: 'Reapply current' }))
    expect(mocks.setSearchParams).toHaveBeenCalledWith(
      new URLSearchParams('status=all'),
      { replace: false },
    )

    mocks.setSearchParams.mockClear()
    fireEvent.click(screen.getByRole('button', { name: 'Choose 7d' }))
    const presetParams = mocks.setSearchParams.mock.calls[0][0] as URLSearchParams
    expect(presetParams.has('from')).toBe(true)
    expect(presetParams.has('to')).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'KPI filter' }))
    const kpiCall = mocks.setSearchParams.mock.calls.at(-1)
    const kpiParams = kpiCall?.[0] as URLSearchParams
    expect(kpiParams.get('outcome')).toBe('hardware')
    expect(kpiParams.get('status')).toBe('failed')
    expect(kpiParams.has('axis')).toBe(false)
    expect(kpiParams.has('metric')).toBe(false)
    expect(kpiCall?.[1]).toEqual({ replace: false })
  })

  it('handles custom dates and table pagination callbacks', () => {
    mocks.searchParams = new URLSearchParams(
      'window=custom&from=2026-05-01&to=2026-05-03&status=all',
    )
    render(<FleetView />)

    // The window is named once, by the two labelled date fields.
    const pickers = Object.fromEntries(
      mocks.datePickerProps.mock.calls.slice(-2).map(([props]) => [props.label, props]),
    )
    expect(pickers.From.value.toISOString().slice(0, 10)).toBe('2026-05-01')
    expect(pickers.To.value.toISOString().slice(0, 10)).toBe('2026-05-03')
    expect(pickers.From.maxDate).toBe(pickers.To.value)
    expect(pickers.To.minDate).toBe(pickers.From.value)
    const dateButtons = screen.getAllByRole('button', { name: 'Set date' })
    fireEvent.click(dateButtons[0])
    fireEvent.click(dateButtons[1])
    fireEvent.click(screen.getByRole('button', { name: 'Choose custom' }))
    fireEvent.click(screen.getByRole('button', { name: 'Set page size' }))
    fireEvent.click(screen.getByRole('button', { name: 'Set hardware page' }))

    expect(mocks.setSearchParams).toHaveBeenCalledTimes(2)
    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({
      pageSize: 25,
      pageByOutcome: { hardware: 1 },
    }))
    expect(mocks.useRunsTable).toHaveBeenLastCalledWith(expect.objectContaining({
      outcomes: ['hardware', 'software', 'multi', 'user', 'undetermined', 'pass'],
      pageSize: 25,
      pageByOutcome: { hardware: 1 },
    }))
  })

  it('starts every run group on page 1 again once the filter or the page size changes', () => {
    const { rerender } = render(<FleetView />)
    fireEvent.click(screen.getByRole('button', { name: 'Set hardware page' }))
    expect(mocks.useRunsTable).toHaveBeenLastCalledWith(expect.objectContaining({
      pageByOutcome: { hardware: 1 },
    }))

    mocks.useRunsTable.mockClear()
    mocks.searchParams = new URLSearchParams('status=all&symptom=Xid79')
    rerender(<FleetView />)
    // Every render, not only the last: fetching the old page for the new filter
    // requests a page that view may not have.
    expect(mocks.useRunsTable).toHaveBeenCalled()
    for (const [args] of mocks.useRunsTable.mock.calls) expect(args.pageByOutcome).toEqual({})
    expect(mocks.useRunsTable).toHaveBeenLastCalledWith(expect.objectContaining({
      filter: expect.objectContaining({ symptom: 'Xid79' }),
    }))
    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({ pageByOutcome: {} }))

    // Returning to the first view does not bring its old pages back.
    mocks.useRunsTable.mockClear()
    mocks.searchParams = new URLSearchParams()
    rerender(<FleetView />)
    expect(mocks.useRunsTable).toHaveBeenCalled()
    for (const [args] of mocks.useRunsTable.mock.calls) expect(args.pageByOutcome).toEqual({})
    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({ pageByOutcome: {} }))

    fireEvent.click(screen.getByRole('button', { name: 'Set hardware page' }))
    mocks.useRunsTable.mockClear()
    fireEvent.click(screen.getByRole('button', { name: 'Set page size' }))
    expect(mocks.useRunsTable).toHaveBeenCalled()
    for (const [args] of mocks.useRunsTable.mock.calls) expect(args.pageByOutcome).toEqual({})
    expect(mocks.useRunsTable).toHaveBeenLastCalledWith(expect.objectContaining({ pageSize: 25 }))
  })

  it('refreshes by re-fetching the dashboard, the run pages and the backend meta, busy until all answer', async () => {
    const answers: Array<() => void> = []
    const pending = () => new Promise<undefined>((resolve) => answers.push(() => resolve(undefined)))
    mocks.refetchDashboard.mockImplementation(pending)
    mocks.refetchRuns.mockImplementation(pending)
    mocks.refetchMeta.mockImplementation(pending)
    render(<FleetView />)

    const refresh = screen.getByRole('button', { name: 'Refresh' })
    // The page's one primary action, the green default button.
    expect([...document.querySelectorAll('[data-variant="default"]')]).toEqual([refresh])
    refresh.focus()
    fireEvent.click(refresh)
    // A real revalidation, not a no-op URL push.
    expect(mocks.setSearchParams).not.toHaveBeenCalled()
    expect(mocks.refetchDashboard).toHaveBeenCalledTimes(1)
    expect(mocks.refetchRuns).toHaveBeenCalledTimes(1)
    expect(mocks.refetchMeta).toHaveBeenCalledTimes(1)
    // Busy, it is unavailable but keeps keyboard focus (a disabled button would
    // drop it to the page), and pressing it again starts nothing.
    expect(refresh.getAttribute('aria-busy')).toBe('true')
    expect(refresh.getAttribute('aria-disabled')).toBe('true')
    expect((refresh as HTMLButtonElement).disabled).toBe(false)
    expect(document.activeElement).toBe(refresh)
    fireEvent.click(refresh)
    expect(mocks.refetchDashboard).toHaveBeenCalledTimes(1)

    await act(async () => { answers[0](); answers[1]() })
    expect(refresh.getAttribute('aria-busy')).toBe('true')
    await act(async () => { answers[2]() })
    expect(refresh.hasAttribute('aria-busy')).toBe(false)
    expect(refresh.hasAttribute('aria-disabled')).toBe(false)
    expect(document.activeElement).toBe(refresh)
  })

  it('leaves Movers out when the server reports none', () => {
    mocks.useDashboard.mockReturnValue({
      dashboard: { ...dashboard, movers: [] },
      isLoading: false,
      error: null,
      mutate: mocks.refetchDashboard,
    })
    render(<FleetView />)

    expect(screen.queryByText('Movers content')).toBeNull()
    expect(mocks.moversProps).not.toHaveBeenCalled()
  })

  it('names the database date range and jumps to it when the window excludes every row', () => {
    mocks.searchParams = new URLSearchParams('window=custom&from=2020-01-01&to=2020-01-02')
    const emptyWindow = { ...dashboard, metrics: { ...dashboard.metrics, windowedRuns: 0 } }
    mocks.useDashboard.mockReturnValue({ dashboard: emptyWindow, ...loaded })
    const { rerender } = render(<FleetView />)

    // Inside the always-mounted polite region, not merely somewhere on the page:
    // moving the notice out of it silently costs the announcement.
    expect(
      within(screen.getByRole('status')).getByText('Workloads exist outside the selected dates'),
    ).toBeTruthy()
    expect(screen.queryByText('No runs listed in this view')).toBeNull()
    expect(screen.getByText(
      'Workloads finished between 2026-05-01 and 2026-05-03. Nothing in the selected date range has an attribution result.',
    )).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'View 2026-05-01 to 2026-05-03' }))
    const params = mocks.setSearchParams.mock.calls[0][0] as URLSearchParams
    expect(params.get('from')).toBe('2026-05-01')
    expect(params.get('to')).toBe('2026-05-03')
    expect(params.get('window')).toBe('custom')
    // The action's own button goes away, so focus moves to the run list at once.
    const runs = screen.getByRole('region', { name: 'Runs' })
    expect(document.activeElement).toBe(runs)
    // The scroll waits for the new window's page: from the click it would aim
    // at the empty window's short layout, which then grows past it.
    expect(scrollIntoView()).not.toHaveBeenCalled()

    // The router commits the new window; its dashboard and runs are still on their way.
    mocks.searchParams = params
    mocks.useDashboard.mockReturnValue({ dashboard: emptyWindow, ...loaded, isLoading: true })
    mocks.useRunsTable.mockReturnValue({ ...runsLoaded, isLoading: true })
    rerender(<FleetView />)
    expect(scrollIntoView()).not.toHaveBeenCalled()
    mocks.useDashboard.mockReturnValue({ dashboard, ...loaded })
    rerender(<FleetView />)
    expect(scrollIntoView()).not.toHaveBeenCalled()

    // With both on screen, the run list comes into view, once.
    mocks.useRunsTable.mockReturnValue(runsLoaded)
    rerender(<FleetView />)
    expect(scrollIntoView()).toHaveBeenCalledTimes(1)
    expect(scrollIntoView()).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' })
    expect(scrollIntoView().mock.contexts[0]).toBe(runs)
    expect(document.activeElement).toBe(runs)
    // A later reload of the same view (another page of a group) leaves the reader where they are.
    mocks.useRunsTable.mockReturnValue({ ...runsLoaded, isLoading: true })
    rerender(<FleetView />)
    mocks.useRunsTable.mockReturnValue(runsLoaded)
    rerender(<FleetView />)
    expect(scrollIntoView()).toHaveBeenCalledTimes(1)
  })

  it('brings the runs into view for a filter toggle only if its view is the one that loads', () => {
    const { rerender } = render(<FleetView />)

    // A KPI toggle asks for the run list; focus stays with the toggle, so the page moves none.
    fireEvent.click(screen.getByRole('button', { name: 'KPI filter' }))
    const requested = mocks.setSearchParams.mock.calls[0][0] as URLSearchParams
    expect(requested.get('outcome')).toBe('hardware')
    expect(document.activeElement).not.toBe(screen.getByRole('region', { name: 'Runs' }))
    rerender(<FleetView />)
    expect(scrollIntoView()).not.toHaveBeenCalled()

    // The reader moves on before that view lands: the request goes with the view it asked for.
    mocks.searchParams = new URLSearchParams('status=all&symptom=Xid79')
    rerender(<FleetView />)
    mocks.searchParams = requested
    rerender(<FleetView />)
    expect(scrollIntoView()).not.toHaveBeenCalled()

    // Asked again from that view, loaded already, the runs come into view at once.
    fireEvent.click(screen.getByRole('button', { name: 'KPI filter' }))
    expect(scrollIntoView()).toHaveBeenCalledTimes(1)
  })

  it('shows each view only its own data, keeping what the same window or filter already said', () => {
    const groups = [{ outcome: 'hardware', runs: [], total: 12, page: 0, pageSize: 10 }]
    const sectionsLoading = (loading: boolean) => {
      for (const props of [mocks.kpiProps, mocks.paretoProps, mocks.breakdownProps]) {
        expect(props).toHaveBeenLastCalledWith(expect.objectContaining({ loading }))
      }
    }
    // Nothing has arrived: every section is told so, and no count is named.
    mocks.useDashboard.mockReturnValue({ dashboard: null, ...loaded, isLoading: true })
    mocks.useRunsTable.mockReturnValue({ ...runsLoaded, total: 0, isLoading: true })
    const { rerender } = render(<FleetView />)
    sectionsLoading(true)
    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({ loading: true, groups: [], matchingRuns: 0 }))
    expect(screen.queryByText(/runs in database/)).toBeNull()

    mocks.useDashboard.mockReturnValue({ dashboard, ...loaded })
    mocks.useRunsTable.mockReturnValue({ ...runsLoaded, groups, total: 12 })
    rerender(<FleetView />)
    sectionsLoading(false)
    expect(mocks.paretoProps).toHaveBeenLastCalledWith(expect.objectContaining({ ledgersLoading: false }))
    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({ loading: false, groups, matchingRuns: 12 }))

    // Another page of the same filter loads (the hook has nothing of it yet):
    // the groups that filter lists and their totals stay, only their rows wait.
    mocks.useRunsTable.mockReturnValue({ ...runsLoaded, total: 0, isLoading: true })
    fireEvent.click(screen.getByRole('button', { name: 'Set hardware page' }))
    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({
      loading: true, groups, matchingRuns: 12, pageByOutcome: { hardware: 1 },
    }))
    // The page still names no count until the fetch is back.
    expect(screen.queryByText(/runs in database/)).toBeNull()
    // So does another page size of it.
    fireEvent.click(screen.getByRole('button', { name: 'Set page size' }))
    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({ loading: true, groups, pageSize: 25 }))

    // Another outcome of the same window: the server narrows only the
    // concentration ledgers to it, so the window's facts (the KPI, the
    // breakdowns, the outcome split) stay and only those ledgers wait. A new
    // filter's run groups are unknown until its pages arrive.
    mocks.searchParams = new URLSearchParams('status=failed&outcome=hardware')
    mocks.useDashboard.mockReturnValue({ dashboard: null, ...loaded, isLoading: true })
    rerender(<FleetView />)
    sectionsLoading(false)
    expect(mocks.kpiProps).toHaveBeenLastCalledWith(expect.objectContaining({ hwFailures: 2 }))
    expect(mocks.breakdownProps).toHaveBeenLastCalledWith(expect.objectContaining({ hardware: dashboard.hwBreakdown }))
    expect(mocks.paretoProps).toHaveBeenLastCalledWith(expect.objectContaining({
      ledgersLoading: true, outcomeLedger: dashboard.ledgers.outcome, symptomLedger: [],
    }))
    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({ loading: true, groups: [], matchingRuns: 0 }))
    expect(screen.queryByText(/runs in database/)).toBeNull()

    // A new window's dashboard is its own: nothing of the previous window stands in.
    mocks.searchParams = new URLSearchParams('window=custom&from=2026-05-01&to=2026-05-02&status=failed&outcome=hardware')
    rerender(<FleetView />)
    sectionsLoading(true)
    expect(mocks.kpiProps).toHaveBeenLastCalledWith(expect.objectContaining({ hwFailures: 0 }))
    expect(mocks.paretoProps).toHaveBeenLastCalledWith(expect.objectContaining({ outcomeLedger: [] }))
  })

  it('shows no notice when the database reports no date bounds', () => {
    mocks.searchParams = new URLSearchParams('window=custom&from=2020-01-01&to=2020-01-02')
    mocks.useDashboard.mockReturnValue({
      dashboard: {
        ...dashboard,
        metrics: { ...dashboard.metrics, windowedRuns: 0 },
        meta: { ...dashboard.meta, dateBounds: { min: null, max: null } },
      },
      isLoading: false,
      error: null,
    })
    render(<FleetView />)

    expect(screen.queryByText('Workloads exist outside the selected dates')).toBeNull()
    expect(screen.queryByText('No runs listed in this view')).toBeNull()
  })

  it('points at the run list, not the dates, when the window holds runs the table does not list', () => {
    mocks.searchParams = new URLSearchParams(
      'window=custom&from=2026-05-02&to=2026-05-10'
      + '&outcome=hardware&symptom=Xid79&node=gpu-01&rule=xid-79&user=alice&q=job-100',
    )
    mocks.useRunsTable.mockReturnValue({
      groups: [],
      total: 0,
      visibleRuns: [],
      isLoading: false,
      error: null,
    })
    const { rerender } = render(<FleetView />)

    // Inside the always-mounted polite region, not merely somewhere on the page:
    // moving the notice out of it silently costs the announcement.
    expect(
      within(screen.getByRole('status')).getByText('No runs listed in this view'),
    ).toBeTruthy()
    expect(screen.queryByText('Workloads exist outside the selected dates')).toBeNull()
    expect(screen.getByText(
      'This view contains runs, but none of them appear in the run list below.',
    )).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Clear run filters' }))
    const params = mocks.setSearchParams.mock.calls[0][0] as URLSearchParams
    expect(params.get('status')).toBe('all')
    expect(params.has('outcome')).toBe(false)
    expect(params.has('symptom')).toBe(false)
    expect(params.has('node')).toBe(false)
    expect(params.has('rule')).toBe(false)
    expect(params.has('user')).toBe(false)
    expect(params.has('q')).toBe(false)
    // The dates are the other notice's business: this action must leave the window intact.
    expect(params.get('from')).toBe('2026-05-02')
    expect(params.get('to')).toBe('2026-05-10')
    expect(document.activeElement).toBe(screen.getByRole('region', { name: 'Runs' }))
    // The run list comes into view once the unfiltered runs are listed, not before.
    expect(scrollIntoView()).not.toHaveBeenCalled()
    mocks.searchParams = params
    mocks.useRunsTable.mockReturnValue({ ...runsLoaded, isLoading: true })
    rerender(<FleetView />)
    expect(scrollIntoView()).not.toHaveBeenCalled()
    mocks.useRunsTable.mockReturnValue(runsLoaded)
    rerender(<FleetView />)
    expect(scrollIntoView()).toHaveBeenCalledTimes(1)
  })

  it('says the database is empty rather than blaming the dates or the filters', () => {
    mocks.useDashboard.mockReturnValue({ dashboard: null, isLoading: true, error: null })
    mocks.useRunsTable.mockReturnValue({
      groups: [],
      total: 0,
      visibleRuns: [],
      isLoading: false,
      error: null,
    })
    const { rerender } = render(<FleetView />)

    expect(screen.queryByText('This database has no attribution runs')).toBeNull()

    mocks.useDashboard.mockReturnValue({
      dashboard: {
        ...dashboard,
        metrics: { ...dashboard.metrics, windowedRuns: 0 },
        meta: { ...dashboard.meta, dbTotalRuns: 0, dateBounds: { min: null, max: null } },
      },
      isLoading: false,
      error: null,
    })
    rerender(<FleetView />)

    const region = screen.getByRole('status')
    expect(within(region).getByText('This database has no attribution runs')).toBeTruthy()
    expect(screen.queryByText('Workloads exist outside the selected dates')).toBeNull()
    expect(screen.queryByText('No runs listed in this view')).toBeNull()
    // Nothing is left to loosen, so this notice carries no action to take.
    expect(within(region).queryByRole('button')).toBeNull()
    expect(mocks.runsProps).toHaveBeenLastCalledWith(
      expect.objectContaining({ databaseEmpty: true }),
    )
  })

  it('counts the listed runs against the whole database, populated, filtered down, or empty', () => {
    const { rerender } = render(<FleetView />)

    expect(screen.getByText('4 of 1,204 runs in database')).toBeTruthy()

    mocks.useRunsTable.mockReturnValue({
      groups: [],
      total: 0,
      visibleRuns: [],
      isLoading: false,
      error: null,
    })
    rerender(<FleetView />)

    expect(screen.getByText('0 of 1,204 runs in database')).toBeTruthy()

    mocks.useDashboard.mockReturnValue({
      dashboard: {
        ...dashboard,
        metrics: { ...dashboard.metrics, windowedRuns: 0 },
        meta: { ...dashboard.meta, dbTotalRuns: 0, dateBounds: { min: null, max: null } },
      },
      isLoading: false,
      error: null,
    })
    rerender(<FleetView />)

    expect(screen.getByText('0 of 0 runs in database')).toBeTruthy()
  })

  it('names no counts while a fetch is in flight and says so when one fails', () => {
    mocks.useDashboard.mockReturnValue({ dashboard: null, isLoading: true, error: null })
    const { rerender } = render(<FleetView />)

    expect(screen.queryByText(/runs in database/)).toBeNull()
    expect(screen.queryByText('Run counts unavailable')).toBeNull()

    mocks.useDashboard.mockReturnValue({ dashboard, isLoading: false, error: null })
    mocks.useRunsTable.mockReturnValue({
      groups: [],
      total: 0,
      visibleRuns: [],
      isLoading: true,
      error: null,
    })
    rerender(<FleetView />)

    expect(screen.queryByText(/runs in database/)).toBeNull()

    mocks.useDashboard.mockReturnValue({
      dashboard: null,
      isLoading: false,
      error: new Error('dashboard fetch failed'),
    })
    mocks.useRunsTable.mockReturnValue({
      groups: [],
      total: 0,
      visibleRuns: [],
      isLoading: false,
      error: null,
    })
    rerender(<FleetView />)

    expect(screen.getByText('Run counts unavailable')).toBeTruthy()
    expect(screen.queryByText(/runs in database/)).toBeNull()
  })

  it('tells the dashboard sections a failed request apart from a window with no failures', () => {
    const sections = [mocks.kpiProps, mocks.paretoProps, mocks.breakdownProps]
    const failure = new Error('dashboard fetch failed')
    // Nothing of this window came back: every dashboard section is told it
    // failed, so none states zeros or "no failures" for it.
    mocks.useDashboard.mockReturnValue({ dashboard: null, ...loaded, error: failure })
    const { rerender } = render(<FleetView />)
    for (const props of sections) expect(props).toHaveBeenLastCalledWith(expect.objectContaining({ loading: false, failed: true }))
    expect(mocks.paretoProps).toHaveBeenLastCalledWith(expect.objectContaining({ ledgersLoading: false, ledgersFailed: true }))

    // While SWR retries it, the request is loading again, not failed.
    mocks.useDashboard.mockReturnValue({ dashboard: null, ...loaded, isLoading: true, error: failure })
    rerender(<FleetView />)
    for (const props of sections) expect(props).toHaveBeenLastCalledWith(expect.objectContaining({ loading: true, failed: false }))

    // A failed revalidation keeps the dashboard it fetched: nothing failed to show.
    mocks.useDashboard.mockReturnValue({ dashboard, ...loaded, error: failure })
    rerender(<FleetView />)
    for (const props of sections) expect(props).toHaveBeenLastCalledWith(expect.objectContaining({ loading: false, failed: false }))
    expect(mocks.paretoProps).toHaveBeenLastCalledWith(expect.objectContaining({ ledgersFailed: false, symptomLedger: dashboard.ledgers.symptom }))

    // Another outcome of the same window fails: the window's facts stay, and
    // only the concentration ledgers, which the outcome narrows, failed.
    mocks.searchParams = new URLSearchParams('status=all&outcome=hardware')
    mocks.useDashboard.mockReturnValue({ dashboard: null, ...loaded, error: failure })
    rerender(<FleetView />)
    for (const props of sections) expect(props).toHaveBeenLastCalledWith(expect.objectContaining({ loading: false, failed: false }))
    expect(mocks.kpiProps).toHaveBeenLastCalledWith(expect.objectContaining({ hwFailures: 2 }))
    expect(mocks.paretoProps).toHaveBeenLastCalledWith(expect.objectContaining({
      ledgersFailed: true, outcomeLedger: dashboard.ledgers.outcome, symptomLedger: [],
    }))
  })

  it('keeps both notices hidden while the runs fetch is in flight and when it fails', () => {
    mocks.searchParams = new URLSearchParams(
      'window=custom&from=2026-05-02&to=2026-05-10&outcome=hardware',
    )
    mocks.useRunsTable.mockReturnValue({
      groups: [],
      total: 0,
      visibleRuns: [],
      isLoading: false,
      error: new Error('runs fetch failed'),
    })
    const { rerender } = render(<FleetView />)

    expect(screen.queryByText('Workloads exist outside the selected dates')).toBeNull()
    expect(screen.queryByText('No runs listed in this view')).toBeNull()
    // Nor does the section read the failure as an empty view.
    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({ failed: true, matchingRuns: 0 }))

    mocks.useRunsTable.mockReturnValue({
      groups: [],
      total: 0,
      visibleRuns: [],
      isLoading: true,
      error: null,
    })
    rerender(<FleetView />)

    expect(screen.queryByText('Workloads exist outside the selected dates')).toBeNull()
    expect(screen.queryByText('No runs listed in this view')).toBeNull()

    // A failed revalidation keeps showing the runs already fetched.
    const groups = [{ outcome: 'hardware', runs: [], total: 3, page: 0, pageSize: 10 }]
    mocks.useRunsTable.mockReturnValue({
      groups,
      total: 3,
      visibleRuns: [],
      isLoading: false,
      error: new Error('runs revalidation failed'),
    })
    rerender(<FleetView />)

    expect(mocks.runsProps).toHaveBeenLastCalledWith(expect.objectContaining({ failed: false, groups, matchingRuns: 3 }))
  })
})
