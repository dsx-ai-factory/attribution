// @vitest-environment jsdom
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { FleetFilter } from '@/interfaces/filter'
import { FleetView } from '@/routers/fleet'

const mocks = vi.hoisted(() => ({
  searchParams: new URLSearchParams(),
  setSearchParams: vi.fn(),
  useDashboard: vi.fn(),
  useRunsTable: vi.fn(),
  kpiProps: vi.fn(),
  paretoProps: vi.fn(),
  breakdownProps: vi.fn(),
  moversProps: vi.fn(),
  runsProps: vi.fn(),
}))

vi.mock('react-router-dom', async (importOriginal) => ({
  ...await importOriginal<typeof import('react-router-dom')>(),
  useSearchParams: () => [mocks.searchParams, mocks.setSearchParams],
}))
vi.mock('@/hooks/use-dashboard', () => ({ useDashboard: mocks.useDashboard }))
vi.mock('@/hooks/use-runs-table', () => ({ useRunsTable: mocks.useRunsTable }))
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
  DatePicker: ({ onChange }: { onChange: (date: Date | undefined) => void }) => (
    <button onClick={() => onChange(new Date('2026-05-02T00:00:00Z'))}>Set date</button>
  ),
}))
vi.mock('@/routers/fleet/components/kpi-strip', () => ({
  KpiStrip: (props: { setFilter: (next: FleetFilter) => void; hwFailures: number }) => {
    mocks.kpiProps(props)
    return (
      <button onClick={() => props.setFilter({
        status: 'failed',
        outcome: 'hardware',
        axis: 'symptom',
        metric: 'count',
      })}>
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
    return (
      <div>
        <button onClick={() => props.setPageSize(25)}>Set page size</button>
        <button onClick={() => props.setGroupPage('hardware', 1)}>Set hardware page</button>
      </div>
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
  movers: [{ symptom: 'Xid79', previous: 1, current: 2, delta: 1 }],
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
  mocks.useDashboard.mockReturnValue({ dashboard, isLoading: false, error: null })
  mocks.useRunsTable.mockReturnValue({
    groups: [],
    total: 4,
    visibleRuns: [],
    isLoading: false,
    error: null,
  })
})

afterEach(cleanup)

describe('FleetView', () => {
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
    expect(screen.getByText(/schema 2/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
    expect(mocks.setSearchParams).toHaveBeenCalledTimes(1)

    mocks.setSearchParams.mockClear()
    fireEvent.click(screen.getByRole('button', { name: 'Reapply current' }))
    expect(mocks.setSearchParams).toHaveBeenCalledWith(
      new URLSearchParams('status=failed'),
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

    expect(screen.getByText(/window 2026-05-01 → 2026-05-03/)).toBeTruthy()
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

  it('names the database date range and jumps to it when the window excludes every row', () => {
    mocks.searchParams = new URLSearchParams('window=custom&from=2020-01-01&to=2020-01-02')
    mocks.useDashboard.mockReturnValue({
      dashboard: { ...dashboard, metrics: { ...dashboard.metrics, windowedRuns: 0 } },
      isLoading: false,
      error: null,
    })
    render(<FleetView />)

    // Inside the always-mounted polite region, not merely somewhere on the page:
    // moving the notice out of it silently costs the announcement.
    expect(
      within(screen.getByRole('status')).getByText('Workloads exist outside the selected dates'),
    ).toBeTruthy()
    expect(screen.queryByText('No runs listed in this view')).toBeNull()
    expect(screen.getByText(
      'Workloads finished between 2026-05-01 and 2026-05-03. Nothing in the selected date range has an attribution result.',
    )).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /2026-05-01 → 2026-05-03/ }))
    const params = mocks.setSearchParams.mock.calls[0][0] as URLSearchParams
    expect(params.get('from')).toBe('2026-05-01')
    expect(params.get('to')).toBe('2026-05-03')
    expect(params.get('window')).toBe('custom')
    expect(document.activeElement).toBe(screen.getByRole('region', { name: 'Runs' }))
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
    render(<FleetView />)

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
  })
})
