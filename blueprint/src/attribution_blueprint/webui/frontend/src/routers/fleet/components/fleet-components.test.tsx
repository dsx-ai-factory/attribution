// @vitest-environment jsdom
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { FleetFilter } from '@/interfaces/filter'
import { DatePicker } from '@/routers/fleet/components/date-picker'
import { BreakdownByDay } from '@/routers/fleet/components/breakdown-by-day'
import { MoversStrip } from '@/routers/fleet/components/movers-strip'
import { KpiStrip } from '@/routers/fleet/components/kpi-strip'
import {
  DeltaPill,
  FailureLink,
  MtbiSparkline,
  scrollToRuns,
} from '@/routers/fleet/components/mtbi-primitives'

const mocks = vi.hoisted(() => ({
  calendarProps: vi.fn(),
  useECharts: vi.fn(),
}))

vi.mock('@/components/ui/calendar', () => ({
  Calendar: (props: { onSelect: (date: Date | undefined) => void }) => {
    mocks.calendarProps(props)
    return <button onClick={() => props.onSelect(new Date(2026, 4, 3))}>Choose May 3</button>
  },
}))
vi.mock('@/components/ui/popover', () => ({
  Popover: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  PopoverTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  PopoverContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}))
vi.mock('@/components/ui/hover-card', () => ({
  HoverCard: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  HoverCardTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  HoverCardContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}))
vi.mock('@/components/def-hover', () => ({
  DefHover: ({ children, term }: { children?: ReactNode; term: string }) => <span>{children ?? term}</span>,
}))
vi.mock('@/hooks/use-echarts', () => ({
  ECHART_COMMON: {},
  useECharts: mocks.useECharts,
}))

const filter: FleetFilter = {
  status: 'failed',
  axis: 'symptom',
  metric: 'count',
}

beforeEach(() => {
  vi.clearAllMocks()
  document.body.innerHTML = '<div id="runs-section"></div>'
  Element.prototype.scrollIntoView = vi.fn()
})

afterEach(cleanup)

describe('date and metric primitives', () => {
  it('bridges selected days between UTC and the local calendar', () => {
    const onChange = vi.fn()
    render(<DatePicker
      value={new Date('2026-05-01T00:00:00Z')}
      minDate={new Date('2026-04-01T00:00:00Z')}
      maxDate={new Date('2026-06-01T00:00:00Z')}
      onChange={onChange}
    />)

    expect(screen.getByText('2026-05-01')).toBeTruthy()
    const props = mocks.calendarProps.mock.calls[0][0]
    expect(props.disabled).toHaveLength(2)
    fireEvent.click(screen.getByRole('button', { name: 'Choose May 3' }))
    expect(onChange.mock.calls[0][0].toISOString()).toBe('2026-05-03T00:00:00.000Z')
  })

  it('renders empty and populated sparklines plus delta directions', () => {
    const { container, rerender } = render(<MtbiSparkline values={[]} />)
    expect(container.innerHTML).toBe('')

    rerender(<><MtbiSparkline values={[2, 2, 4]} /><DeltaPill now={12} prior={10} /><DeltaPill now={8} prior={10} /><DeltaPill now={1} prior={0} /></>)
    expect(container.querySelector('path')?.getAttribute('d')).toContain('M0.0')
    expect(screen.getByText('▲ 20%')).toBeTruthy()
    expect(screen.getByText('▼ 20%')).toBeTruthy()
    expect(screen.getByText('▲ 0%')).toBeTruthy()
  })

  it('applies a failure outcome and scrolls to the run table', () => {
    const setFilter = vi.fn()
    const onClick = vi.fn()
    render(<FailureLink outcome="hardware" filter={filter} setFilter={setFilter} onClick={onClick}>3 HW</FailureLink>)

    fireEvent.click(screen.getByText('3 HW'))
    scrollToRuns()

    expect(onClick).toHaveBeenCalled()
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({
      status: 'failed',
      outcome: 'hardware',
    }))
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled()
  })
})

describe('fleet summary components', () => {
  it('renders the movers empty state and toggles a symptom', () => {
    const setFilter = vi.fn()
    const { rerender } = render(<MoversStrip movers={[]} filter={filter} setFilter={setFilter} />)
    expect(screen.getByText('No symptoms moved materially in this window.')).toBeTruthy()

    rerender(<MoversStrip
      movers={[{
        symptom: 'Xid79',
        count: 3,
        dominantOutcome: 'hardware',
        spark: [1, 3],
        score: 2,
      }]}
      filter={filter}
      setFilter={setFilter}
    />)
    fireEvent.keyDown(screen.getByRole('button'), { key: 'Enter' })
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ symptom: 'Xid79' }))
    expect(screen.getByTitle('Rising within window')).toBeTruthy()
  })

  it('renders KPI loading and populated states', () => {
    const setFilter = vi.fn()
    const common = {
      filter,
      setFilter,
      hwMtbi: 200,
      combinedMtbi: 100,
      gpuHoursRaw: 1_250,
      daily: [{ date: '2026-05-01', hw: 1, sw: 1, mix: 0, usr: 0, pass: 1, nodeHours: 4 }],
      hwFailures: 1,
      swFailures: 1,
      mixFailures: 0,
      combinedFailures: 2,
      range: { from: new Date('2026-05-01T00:00:00Z'), to: new Date('2026-05-02T00:00:00Z') },
      gpuCount: 8,
      runCount: 3,
    }
    const { container, rerender } = render(<KpiStrip {...common} loading />)
    expect(container.querySelector('[aria-busy="true"]')).not.toBeNull()

    rerender(<KpiStrip {...common} />)
    expect(screen.getByText('1.25')).toBeTruthy()
    expect(screen.getByText('8 GPUs · 3 runs')).toBeTruthy()
    expect(screen.getByText(/2026-05-01/)).toBeTruthy()
  })

  it('renders empty breakdowns and forwards chart symptom clicks', () => {
    const setFilter = vi.fn()
    render(<BreakdownByDay
      hardware={{
        days: ['2026-05-01'],
        series: [{ symptom: 'Xid79', data: [1], total: 1 }],
        total: 1,
      }}
      software={{ days: ['2026-05-01'], series: [], total: 0 }}
      filter={filter}
      setFilter={setFilter}
    />)

    expect(screen.getByText('No System Software Fault results in this window')).toBeTruthy()
    const events = mocks.useECharts.mock.calls[0][2]
    events.click({ seriesName: 'Other' })
    expect(setFilter).not.toHaveBeenCalled()
    events.click({ seriesName: 'Xid79' })
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ symptom: 'Xid79' }))
  })
})
