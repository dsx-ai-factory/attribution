// @vitest-environment jsdom
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { Run } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import { ParetoCard } from '@/routers/fleet/components/pareto'
import { RunsSection } from '@/routers/fleet/components/runs-section'
import { RunsToolbar } from '@/routers/fleet/components/runs-toolbar'

const mocks = vi.hoisted(() => ({
  useECharts: vi.fn(),
  nav: vi.fn(),
  downloadBlob: vi.fn(),
}))

vi.mock('@/hooks/use-echarts', () => ({
  ECHART_COMMON: {},
  useECharts: mocks.useECharts,
}))
vi.mock('@/hooks/use-nav', () => ({ useNav: () => mocks.nav }))
vi.mock('@/components/def-hover', () => ({
  DefHover: ({ children, term }: { children?: ReactNode; term: string }) => <span>{children ?? term}</span>,
}))
vi.mock('@/components/job-hover-card', () => ({
  JobHoverCard: ({ children }: { children: ReactNode }) => <>{children}</>,
}))
vi.mock('@/components/node-hover-card', () => ({
  NodeHoverCard: ({ children }: { children: ReactNode }) => <>{children}</>,
}))
vi.mock('@/components/time-hover', () => ({
  TimeHover: ({ date }: { date: string }) => <time>{date}</time>,
}))
vi.mock('@/components/export-menu', () => ({
  ExportMenu: ({ items, triggerLabel }: {
    items: Array<{ label: string; onSelect: () => void }>
    triggerLabel?: ReactNode
  }) => (
    <div>
      <span>{triggerLabel}</span>
      {items.map((item) => <button key={item.label} onClick={item.onSelect}>{item.label}</button>)}
    </div>
  ),
}))
vi.mock('@/lib/export', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/lib/export')>(),
  downloadBlob: mocks.downloadBlob,
  utcDateStamp: () => '20260501-0102',
}))

const filter: FleetFilter = {
  status: 'failed',
  axis: 'symptom',
  metric: 'count',
}

const run = (overrides: Partial<Run> = {}): Run => ({
  id: 'run-1',
  jobId: 'team/job',
  workload: 'team/job',
  attempt: 1,
  attemptUid: 'attempt-1',
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
  rootSymptom: { symptom: 'Xid79', rule: 'xid', time: '', line: '' },
  symptoms: [],
  responsibleNodes: ['node-a', 'node-b', 'node-c', 'node-d'],
  pipeline: '1',
  service: '2',
  log: [],
  ...overrides,
})

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(cleanup)

describe('RunsToolbar', () => {
  it('updates search, status, active chips, and visible exports', () => {
    const setFilter = vi.fn()
    const activeFilter: FleetFilter = {
      ...filter,
      outcome: 'hardware',
      symptom: 'Xid79',
      node: 'node-a',
      rule: 'xid',
      user: 'alice',
      q: 'team',
    }
    render(<RunsToolbar runs={12} filteredRuns={[run()]} filter={activeFilter} setFilter={setFilter} />)

    fireEvent.change(screen.getByPlaceholderText(/Filter by execution context/), {
      target: { value: 'new query' },
    })
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: 'Clear Symptom filter' }))
    fireEvent.click(screen.getByRole('button', { name: 'Visible rows (JSON)' }))

    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ q: 'new query' }))
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ status: 'all' }))
    expect(setFilter.mock.calls[2][0]).not.toHaveProperty('symptom')
    expect(mocks.downloadBlob).toHaveBeenCalledWith(
      'runs-20260501-0102.json',
      expect.any(String),
      'application/json',
    )
    expect(screen.getByText('Export visible (1)')).toBeTruthy()

    // Hiding passing runs is a filter like any other, so it gets a clearable chip too.
    expect(screen.getByText('Passing runs hidden')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Clear Status filter' }))
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ status: 'all' }))
  })

  it('releases the conflicting control when Pass and hide-passing would contradict', () => {
    const setFilter = vi.fn()
    const { rerender } = render(
      <RunsToolbar runs={12} filteredRuns={[run()]} filter={filter} setFilter={setFilter} />,
    )

    // Picking Pass while passing runs are hidden: the outcome wins, hiding is released.
    fireEvent.click(screen.getByText('Pass'))
    expect(setFilter).toHaveBeenCalledWith(
      expect.objectContaining({ outcome: 'pass', status: 'all' }),
    )

    // Hiding passing runs while Pass is selected: the checkbox wins, the outcome is cleared.
    setFilter.mockClear()
    rerender(
      <RunsToolbar
        runs={12}
        filteredRuns={[run()]}
        filter={{ ...filter, status: 'all', outcome: 'pass' }}
        setFilter={setFilter}
      />,
    )
    fireEvent.click(screen.getByRole('checkbox'))
    expect(setFilter).toHaveBeenCalledWith(
      expect.objectContaining({ status: 'failed', outcome: undefined }),
    )
  })
})

describe('RunsSection', () => {
  it('renders grouped rows, node navigation, collapse, and pagination', () => {
    const setPageSize = vi.fn()
    const setGroupPage = vi.fn()
    const props = {
      groups: [{ outcome: 'hardware' as const, runs: [run()], total: 12, page: 0, pageSize: 10 }],
      matchingRuns: 12,
      visibleRuns: [run()],
      filter,
      setFilter: vi.fn(),
      pageByOutcome: {},
      pageSize: 10,
      setPageSize,
      setGroupPage,
    }
    render(<RunsSection {...props} />)

    fireEvent.click(screen.getByText('team/job').closest('tr') as HTMLElement)
    fireEvent.click(screen.getByText(/^node-a/))
    fireEvent.click(screen.getByRole('button', { name: 'Next Hardware Fault page' }))
    fireEvent.change(screen.getByRole('combobox', { name: 'Hardware Fault records per page' }), {
      target: { value: '25' },
    })

    expect(mocks.nav).toHaveBeenCalledWith('verdict', { id: 'run-1' })
    expect(mocks.nav).toHaveBeenCalledWith('node', { id: 'node-a' })
    expect(setGroupPage).toHaveBeenCalledWith('hardware', 1)
    expect(setPageSize).toHaveBeenCalledWith(25)
    expect(screen.getByText('...')).toBeTruthy()

    const groupRow = screen.getAllByText('Hardware Fault').find((element) => element.closest('tr'))?.closest('tr')
    fireEvent.click(groupRow as HTMLElement)
    expect(screen.queryByText('team/job')).toBeNull()
  })

  it('renders loading and empty states', () => {
    const common = {
      groups: [],
      matchingRuns: 0,
      visibleRuns: [],
      filter,
      setFilter: vi.fn(),
      pageByOutcome: {},
      pageSize: 10,
      setPageSize: vi.fn(),
      setGroupPage: vi.fn(),
    }
    const { rerender } = render(<RunsSection {...common} loading />)
    expect(document.querySelectorAll('[class*="animate-pulse"]')).toHaveLength(6)

    rerender(<RunsSection {...common} />)
    expect(screen.getByText('No runs in this view')).toBeTruthy()
    expect(screen.getByText(
      'Try to uncheck "Hide passing runs".',
    )).toBeTruthy()
    expect(screen.queryByText(/filter chips above/)).toBeNull()
    expect(screen.queryByText(/widen the date range/)).toBeNull()

    rerender(
      <RunsSection
        {...common}
        filter={{ ...filter, node: 'node-1', from: '2026-05-01', to: '2026-05-03' }}
      />,
    )
    expect(screen.getByText(
      'Try to clear the active filter chips above, or uncheck "Hide passing runs", or '
      + 'widen the date range.',
    )).toBeTruthy()
  })

  it('drops the loosen-a-filter advice when the database itself is empty', () => {
    render(
      <RunsSection
        groups={[]}
        matchingRuns={0}
        visibleRuns={[]}
        filter={{ ...filter, node: 'node-1', from: '2026-05-01', to: '2026-05-03' }}
        setFilter={vi.fn()}
        pageByOutcome={{}}
        pageSize={10}
        setPageSize={vi.fn()}
        setGroupPage={vi.fn()}
        databaseEmpty
      />,
    )

    expect(screen.getByText('No runs in this database')).toBeTruthy()
    expect(screen.queryByText(/^Try to/)).toBeNull()
  })
})

describe('ParetoCard', () => {
  it('builds chart options and applies category and outcome filters', () => {
    const setFilter = vi.fn()
    render(<ParetoCard
      filter={filter}
      setFilter={setFilter}
      symptomLedger={[
        { symptom: '<Xid79>', count: 8, nodeHours: 10, outcomes: { hardware: 8 } },
        { symptom: 'Timeout', count: 2, nodeHours: 3, outcomes: { software: 2 } },
      ]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[
        { outcome: 'hardware', count: 8, nodeHours: 10 },
        { outcome: 'software', count: 2, nodeHours: 3 },
      ]}
      runs={[]}
    />)

    expect(screen.getByText('<Xid79>')).toBeTruthy()
    expect(screen.getByText(/Vital few/)).toBeTruthy()
    const paretoOption = mocks.useECharts.mock.calls[0][1]
    const tooltip = paretoOption.tooltip.formatter
    expect(tooltip([])).toBe('')
    expect(tooltip([{
      seriesIndex: 0,
      axisValue: '<Xid79>',
      data: { value: 8, dominant: 'hardware', cumPct: 80 },
    }])).toContain('&lt;Xid79&gt;')

    const paretoEvents = mocks.useECharts.mock.calls[0][2]
    paretoEvents.click({ seriesIndex: 1, data: { name: '<Xid79>' } })
    expect(setFilter).not.toHaveBeenCalled()
    paretoEvents.click({ seriesIndex: 0, data: { name: '<Xid79>' } })
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ symptom: '<Xid79>' }))

    const donutEvents = mocks.useECharts.mock.calls[1][2]
    donutEvents.click({ data: { __key: 'hardware' } })
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ outcome: 'hardware' }))
  })

  it('covers node empty state and user fallback aggregation', () => {
    const setFilter = vi.fn()
    const { rerender } = render(<ParetoCard
      filter={{ ...filter, axis: 'node' }}
      setFilter={setFilter}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[]}
      runs={[]}
    />)
    expect(screen.getAllByText('No responsible nodes for this filter.')).toHaveLength(2)

    rerender(<ParetoCard
      filter={{ ...filter, axis: 'user', metric: 'nodeHours', user: 'alice', outcome: 'hardware' }}
      setFilter={setFilter}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[]}
      runs={[run()]}
    />)
    expect(screen.getByText('alice')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'clear pareto filter' }))
    expect(setFilter).toHaveBeenCalledWith({
      status: 'failed',
      axis: 'user',
      metric: 'nodeHours',
    })
  })
})
