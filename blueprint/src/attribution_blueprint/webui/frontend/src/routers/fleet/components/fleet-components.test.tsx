// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { DailyBreakdown } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import { pressedStat } from '@/lib/utils'
import { DatePicker } from '@/routers/fleet/components/date-picker'
import { BreakdownByDay } from '@/routers/fleet/components/breakdown-by-day'
import { MoversStrip } from '@/routers/fleet/components/movers-strip'
import { KpiStrip } from '@/routers/fleet/components/kpi-strip'
import {
  FailureFilterButton,
  MtbiSparkline,
} from '@/routers/fleet/components/mtbi-primitives'

const mocks = vi.hoisted(() => ({
  calendarProps: vi.fn(),
  dispatchAction: vi.fn(),
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
vi.mock('@/hooks/use-echarts', () => ({
  useECharts: mocks.useECharts,
}))
// jsdom loads no stylesheet, so the real readers would throw; these name the token they were asked for.
vi.mock('@/lib/tokens', () => ({
  readColor: (name: string) => `token(${name})`,
  outcomeColor: (outcome: string) => `token(--outcome-${outcome})`,
}))

const filter: FleetFilter = {
  status: 'failed',
  axis: 'symptom',
  metric: 'count',
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.useECharts.mockReturnValue({ current: { dispatchAction: mocks.dispatchAction } })
  // A run list the toggles could scroll to: they must leave that to the page, which waits for the new rows.
  document.body.innerHTML = '<div id="runs-section"></div>'
  Element.prototype.scrollIntoView = vi.fn()
})

afterEach(cleanup)

describe('date and metric primitives', () => {
  it('bridges selected days between UTC and the local calendar', () => {
    const onChange = vi.fn()
    const { rerender } = render(<DatePicker
      label="From"
      value={new Date('2026-05-01T00:00:00Z')}
      minDate={new Date('2026-04-01T00:00:00Z')}
      maxDate={new Date('2026-06-01T00:00:00Z')}
      onChange={onChange}
    />)

    // The label names the trigger together with the day it shows.
    expect(screen.getByRole('button', { name: 'From 2026-05-01' })).toBeTruthy()
    const props = mocks.calendarProps.mock.calls[0][0]
    expect(props.disabled).toHaveLength(2)
    fireEvent.click(screen.getByRole('button', { name: 'Choose May 3' }))
    expect(onChange.mock.calls[0][0].toISOString()).toBe('2026-05-03T00:00:00.000Z')

    rerender(<DatePicker label="To" value={undefined} onChange={onChange} />)
    expect(screen.getByRole('button', { name: 'To Pick a date' })).toBeTruthy()
  })

  it('renders empty sparklines, neutral trends, and outcome-toned trends', () => {
    const { container, rerender } = render(<MtbiSparkline values={[]} />)
    expect(container.innerHTML).toBe('')
    // One day is not a trend.
    rerender(<MtbiSparkline values={[3]} />)
    expect(container.innerHTML).toBe('')

    rerender(<><MtbiSparkline values={[2, 2, 4]} /><MtbiSparkline values={[1, 3]} outcome="undetermined" /></>)
    const [neutral, toned] = container.querySelectorAll('svg')
    // One 60×20 box, stretched to the caller's size: the line spans its width, 3px in from top and bottom.
    expect(neutral.getAttribute('viewBox')).toBe('0 0 60 20')
    expect(neutral.getAttribute('preserveAspectRatio')).toBe('none')
    expect(neutral.querySelector('path')?.getAttribute('d')).toBe('M0.0,17.0 L30.0,17.0 L60.0,3.0')
    expect(neutral.hasAttribute('data-outcome')).toBe(false)
    expect(neutral.querySelector('path')?.getAttribute('stroke')).toBe('var(--foreground-secondary)')
    expect(neutral.querySelector('circle')?.getAttribute('fill')).toBe('var(--foreground-secondary)')
    // An outcome never travels as a color string: the SVG names it and paints with its tone.
    expect(toned.getAttribute('data-outcome')).toBe('undetermined')
    expect(toned.querySelector('path')?.getAttribute('stroke')).toBe('var(--tone)')
  })

  it('toggles a failure outcome filter and asks for the run list when it applies one', () => {
    const setFilter = vi.fn()
    const { rerender } = render(<FailureFilterButton outcome="hardware" count={3} filter={{ ...filter, status: 'all' }} setFilter={setFilter} />)

    // Named by what pressing does, keeping the visible count and the full outcome label.
    const button = screen.getByRole('button', { name: 'Filter runs: 3 Hardware Fault' })
    expect(button.textContent).toBe('3Hardware Fault')
    expect(button.querySelector('[data-slot="outcome-dot"]')?.getAttribute('data-outcome')).toBe('hardware')
    expect(button.getAttribute('aria-pressed')).toBe('false')
    // It sets the outcome alone: an outcome's runs are all failed ones, so the
    // status the reader chose stays as it was.
    fireEvent.click(button)
    expect(setFilter).toHaveBeenLastCalledWith(
      { status: 'all', outcome: 'hardware', axis: 'symptom', metric: 'count' },
      { revealRuns: true },
    )

    // Pressed, it releases only the outcome and stays in place: the filter is
    // the one it pressed, and pressing again gives the same outcome again.
    rerender(<FailureFilterButton outcome="hardware" count={3} filter={setFilter.mock.lastCall![0]} setFilter={setFilter} />)
    expect(button.getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(button)
    expect(setFilter).toHaveBeenLastCalledWith({ status: 'all', axis: 'symptom', metric: 'count' })
    rerender(<FailureFilterButton outcome="hardware" count={3} filter={setFilter.mock.lastCall![0]} setFilter={setFilter} />)
    expect(button.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(button)
    expect(setFilter).toHaveBeenLastCalledWith(
      { status: 'all', outcome: 'hardware', axis: 'symptom', metric: 'count' },
      { revealRuns: true },
    )
    // With passing runs hidden, pressing and releasing leaves them hidden.
    rerender(<FailureFilterButton outcome="software" count={1} filter={filter} setFilter={setFilter} />)
    fireEvent.click(button)
    expect(setFilter).toHaveBeenLastCalledWith({ ...filter, outcome: 'software' }, { revealRuns: true })
    rerender(<FailureFilterButton outcome="software" count={1} filter={setFilter.mock.lastCall![0]} setFilter={setFilter} />)
    fireEvent.click(button)
    expect(setFilter).toHaveBeenLastCalledWith(filter)
    // The page brings the runs into view once they are listed; the toggle never scrolls on its own.
    expect(Element.prototype.scrollIntoView).not.toHaveBeenCalled()
  })
})

describe('fleet summary components', () => {
  it('lists movers as symptom filter toggles with their trend', () => {
    const setFilter = vi.fn()
    const mover = { symptom: 'Xid79', count: 3, dominantOutcome: 'hardware' as const, spark: [1, 3], score: 2 }
    const { rerender } = render(<MoversStrip movers={[mover]} windowLabel="7d" filter={filter} setFilter={setFilter} />)

    const section = screen.getByRole('region', { name: 'Movers' })
    expect(within(section).getByText('7d')).toBeTruthy()
    const button = within(section).getByRole('button', { name: /^Xid79/ })
    expect(button.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(button)
    expect(setFilter).toHaveBeenLastCalledWith(expect.objectContaining({ symptom: 'Xid79' }))
    // Rising failures are the worsening trend color, marked by a named icon rather than a glyph;
    // the outcome is named in full beside its dot, and the sparkline takes the dominant outcome's tone.
    expect(within(button).getByRole('img', { name: 'Rising within window' }).classList.contains('text-trend-worse')).toBe(true)
    expect(within(button).getByText('Hardware Fault')).toBeTruthy()
    expect(button.querySelector('[data-slot="outcome-dot"]')?.getAttribute('data-outcome')).toBe('hardware')
    expect(button.querySelector('svg[data-outcome]')?.getAttribute('data-outcome')).toBe('hardware')
    expect(button.textContent).toContain('3 runs')

    rerender(<MoversStrip
      movers={[{ ...mover, count: 1, dominantOutcome: 'software', spark: [3, 1], score: -2 }]}
      windowLabel="7d"
      filter={{ ...filter, symptom: 'Xid79' }}
      setFilter={setFilter}
    />)
    expect(screen.getByRole('img', { name: 'Falling within window' }).classList.contains('text-trend-better')).toBe(true)
    const pressed = screen.getByRole('button', { name: /^Xid79/ })
    expect(pressed.getAttribute('aria-pressed')).toBe('true')
    // Pressed in the one pressed-stat recipe the KPI stats use.
    expect(pressed.hasAttribute('data-pressed')).toBe(true)
    expect(pressed.className.split(' ')).toEqual(expect.arrayContaining(pressedStat.split(' ')))
    expect(pressed.textContent).toContain('1 run')
    expect(pressed.textContent).not.toContain('1 runs')
    fireEvent.click(pressed)
    expect(setFilter.mock.lastCall![0]).not.toHaveProperty('symptom')
  })

  it('renders KPI loading and populated states', () => {
    const setFilter = vi.fn()
    const common = {
      filter,
      setFilter,
      hwMtbi: 200,
      combinedMtbi: 100,
      gpuHoursRaw: 1_250,
      daily: [
        { date: '2026-05-01', hw: 1, sw: 1, mix: 0, usr: 0, pass: 1, nodeHours: 4 },
        { date: '2026-05-02', hw: 0, sw: 0, mix: 0, usr: 0, pass: 0, nodeHours: 0 },
      ],
      hwFailures: 1,
      swFailures: 1,
      mixFailures: 0,
      combinedFailures: 2,
      range: { from: new Date('2026-05-01T00:00:00Z'), to: new Date('2026-05-02T00:00:00Z') },
      gpuCount: 8,
      runCount: 3,
    }
    const { container, rerender } = render(<KpiStrip {...common} loading />)
    // Loading, the stats keep their labels and filter toggles (one may be the
    // control just pressed); only the figures wait, and none is claimed.
    expect(container.querySelector('[data-slot="card"]')?.getAttribute('aria-busy')).toBe('true')
    expect(screen.getByRole('button', { name: 'Filter runs: Combined MTBI' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Filter runs: HW MTBI' })).toBeTruthy()
    expect(within(screen.getByRole('group', { name: 'Failures' })).getAllByRole('button').map((button) => button.getAttribute('aria-label')))
      .toEqual(['Filter runs: Hardware Fault', 'Filter runs: System Software Fault', 'Filter runs: Ambiguous Fault'])
    expect(container.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(3 * 2)
    expect(container.querySelector('.num')).toBeNull()
    // (Hints' definitions are hidden descriptions, not text on the card.)
    const shown = container.cloneNode(true) as HTMLElement
    shown.querySelectorAll('[hidden]').forEach((node) => node.remove())
    expect(shown.textContent).toBe('Combined MTBI HW MTBI FailuresHardware FaultSystem Software FaultAmbiguous FaultGPU-hours ')

    rerender(<KpiStrip {...common} />)
    expect(container.querySelector('[data-slot="card"]')?.hasAttribute('aria-busy')).toBe(false)
    expect(screen.getByText('1.25')).toBeTruthy()
    expect(screen.getByText('8 GPUs · 3 runs')).toBeTruthy()
    expect(screen.getByText('1250 GPU-hours ÷ 2 failures ÷ 1000')).toBeTruthy()
    expect(screen.getByText('1250 GPU-hours ÷ 1 failure ÷ 1000')).toBeTruthy()
    // Both MTBI trends are neutral lines, not brand green (lucide icons are the hints' marks, not trends).
    const strokes = [...container.querySelectorAll('svg:not(.lucide) path')].map((path) => path.getAttribute('stroke'))
    expect(strokes).toEqual(['var(--foreground-secondary)', 'var(--foreground-secondary)'])
    // Each trend ends its value's line, on the value's baseline, out of the
    // accessibility tree; the caption follows that line, held to the stat's foot.
    for (const [value, caption] of [['100.00', '1250 GPU-hours ÷ 2 failures ÷ 1000'], ['200.00', '1250 GPU-hours ÷ 1 failure ÷ 1000']]) {
      const valueLine = screen.getByText(value).closest('p')!.parentElement!
      expect([...valueLine.classList]).toEqual(expect.arrayContaining(['flex', 'items-baseline', 'justify-between']))
      expect(valueLine.lastElementChild!.getAttribute('aria-hidden')).toBe('true')
      expect(valueLine.lastElementChild!.querySelector('svg path')).not.toBeNull()
      expect(valueLine.nextElementSibling!.textContent).toBe(caption)
      expect(valueLine.nextElementSibling!.classList.contains('mt-auto')).toBe(true)
    }
    // Only the derived metrics carry a definition hint; plain labels do not.
    expect(screen.getAllByRole('button', { name: /^About / }).map((hint) => hint.getAttribute('aria-label')))
      .toEqual(['About Combined MTBI', 'About HW MTBI', 'About GPU-hours'])
    // Failures name every outcome in full, each a filter toggle.
    expect(within(screen.getByRole('group', { name: 'Failures' })).getAllByRole('button').map((button) => button.getAttribute('aria-label')))
      .toEqual(['Filter runs: 1 Hardware Fault', 'Filter runs: 1 System Software Fault', 'Filter runs: 0 Ambiguous Fault'])
    expect(screen.queryByText(/^(HW|SW|MIX)$/)).toBeNull()

    // One GPU and one run are counted in the singular.
    rerender(<KpiStrip {...common} gpuCount={1} runCount={1} />)
    expect(screen.getByText('1 GPU · 1 run')).toBeTruthy()

    // The window's dashboard failed: its figures fall back to zeros, and the
    // strip states none of them. The toggles stay; every value is the shared
    // dash read as "Not available", every caption says the figures could not
    // be loaded, and no count, zero or "no failures" is claimed.
    const nothing = { hwMtbi: 0, combinedMtbi: 0, gpuHoursRaw: 0, daily: [], hwFailures: 0, swFailures: 0, mixFailures: 0, combinedFailures: 0, gpuCount: 0, runCount: 0 }
    rerender(<KpiStrip {...common} {...nothing} failed />)
    expect(container.querySelector('[data-slot="card"]')?.hasAttribute('aria-busy')).toBe(false)
    expect(container.querySelector('[data-slot="skeleton"]')).toBeNull()
    expect(screen.getByRole('button', { name: 'Filter runs: Combined MTBI', description: 'Not available Figures could not be loaded' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Filter runs: HW MTBI', description: 'Not available Figures could not be loaded' })).toBeTruthy()
    expect(within(screen.getByRole('group', { name: 'Failures' })).getAllByRole('button').map((button) => button.getAttribute('aria-label')))
      .toEqual(['Filter runs: Hardware Fault', 'Filter runs: System Software Fault', 'Filter runs: Ambiguous Fault'])
    const gpuHoursStat = screen.getByRole('button', { name: 'About GPU-hours' }).closest('p')!.parentElement!.cloneNode(true) as HTMLElement
    gpuHoursStat.querySelectorAll('[hidden]').forEach((node) => node.remove())
    expect(gpuHoursStat.textContent).toBe('GPU-hours —Not availableFigures could not be loaded')
    expect(screen.getAllByText('Figures could not be loaded')).toHaveLength(3)
    expect(container.querySelector('.num')).toBeNull()
    expect(screen.queryByText(/No (infrastructure|hardware) failures|0 GPUs|0 runs/)).toBeNull()
  })

  it('makes each MTBI stat one filter toggle beside, never around, its definition hint', () => {
    const setFilter = vi.fn()
    const common = {
      setFilter,
      hwMtbi: 200,
      combinedMtbi: 100,
      gpuHoursRaw: 1_250,
      daily: [],
      hwFailures: 0,
      swFailures: 0,
      mixFailures: 0,
      combinedFailures: 0,
      range: { from: new Date('2026-05-01T00:00:00Z'), to: new Date('2026-05-02T00:00:00Z') },
      gpuCount: 8,
      runCount: 3,
    }
    const { rerender } = render(<KpiStrip {...common} filter={{ ...filter, status: 'all' }} />)

    const combined = screen.getByRole('button', { name: 'Filter runs: Combined MTBI' })
    const hw = screen.getByRole('button', { name: 'Filter runs: HW MTBI' })
    // The hint is a sibling control, so Enter on it cannot also filter.
    expect(combined.contains(screen.getByRole('button', { name: 'About Combined MTBI' }))).toBe(false)
    // The value and its derivation describe the toggle; with no failures there
    // is no MTBI: the value slot holds the shared muted dash, read as "Not
    // available", never a mono "n/a", and the caption says why.
    expect(screen.getByRole('button', { name: 'Filter runs: Combined MTBI', description: 'Not available No infrastructure failures in this window' })).toBe(combined)
    expect(screen.getByRole('button', { name: 'Filter runs: HW MTBI', description: 'Not available No hardware failures in this window' })).toBe(hw)
    const unavailable = document.getElementById(combined.getAttribute('aria-describedby')!)!.firstElementChild!
    expect(unavailable.textContent).toBe('—Not available')
    expect(unavailable.querySelector('.num')).toBeNull()
    expect(unavailable.querySelector('.text-xl > .text-foreground-muted > [aria-hidden="true"]')?.textContent).toBe('—')
    expect(screen.queryByText('n/a')).toBeNull()

    expect(combined.getAttribute('aria-pressed')).toBe('false')
    // Applying a filter asks the page for the run list; the page scrolls once the new rows are there.
    fireEvent.click(combined)
    expect(setFilter).toHaveBeenLastCalledWith({ status: 'failed', axis: 'symptom', metric: 'count' }, { revealRuns: true })
    // HW MTBI sets its outcome alone, as a Failures row does: the hardware
    // runs are all failed ones, so the status stays the reader's.
    fireEvent.click(hw)
    expect(setFilter).toHaveBeenLastCalledWith({ status: 'all', outcome: 'hardware', axis: 'symptom', metric: 'count' }, { revealRuns: true })
    expect(Element.prototype.scrollIntoView).not.toHaveBeenCalled()
    // Pressed, released and pressed again, it only ever moves the outcome:
    // releasing leaves no status behind that would press Combined MTBI or
    // hide passing runs, and the next press gives the same filter again.
    rerender(<KpiStrip {...common} filter={setFilter.mock.lastCall![0]} />)
    expect([hw.getAttribute('aria-pressed'), combined.getAttribute('aria-pressed')]).toEqual(['true', 'false'])
    fireEvent.click(hw)
    expect(setFilter).toHaveBeenLastCalledWith({ status: 'all', axis: 'symptom', metric: 'count' })
    rerender(<KpiStrip {...common} filter={setFilter.mock.lastCall![0]} />)
    expect([hw.getAttribute('aria-pressed'), combined.getAttribute('aria-pressed')]).toEqual(['false', 'false'])
    fireEvent.click(hw)
    expect(setFilter).toHaveBeenLastCalledWith({ status: 'all', outcome: 'hardware', axis: 'symptom', metric: 'count' }, { revealRuns: true })

    // Pressed, each stat shows it on the whole stat and releases its own filter.
    rerender(<KpiStrip {...common} filter={{ ...filter, status: 'failed' }} />)
    expect(combined.getAttribute('aria-pressed')).toBe('true')
    expect(combined.closest('[data-pressed]')).not.toBeNull()
    expect(combined.closest('[data-pressed]')!.className.split(' ')).toEqual(expect.arrayContaining(pressedStat.split(' ')))
    expect(hw.closest('[data-pressed]')).toBeNull()
    // Releasing asks for nothing: the list stays where the reader is.
    fireEvent.click(combined)
    expect(setFilter).toHaveBeenLastCalledWith({ status: 'all', axis: 'symptom', metric: 'count' })

    // Pressed over passing runs already hidden, releasing keeps them hidden.
    rerender(<KpiStrip {...common} filter={{ ...filter, outcome: 'hardware' }} />)
    expect(hw.getAttribute('aria-pressed')).toBe('true')
    expect(combined.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(hw)
    expect(setFilter).toHaveBeenLastCalledWith({ status: 'failed', axis: 'symptom', metric: 'count' })

    // Another dimension narrows a stat's list and stays: pressed shows what the
    // press set, so pressing again releases it rather than setting it again.
    const narrowed = { symptom: 'GpuFell', node: 'node-a', rule: 'xid' }
    rerender(<KpiStrip {...common} filter={{ ...filter, ...narrowed, status: 'all' }} />)
    expect(combined.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(combined)
    expect(setFilter).toHaveBeenLastCalledWith({ ...narrowed, status: 'failed', axis: 'symptom', metric: 'count' }, { revealRuns: true })
    rerender(<KpiStrip {...common} filter={setFilter.mock.lastCall![0]} />)
    expect(combined.getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(combined)
    expect(setFilter).toHaveBeenLastCalledWith({ ...narrowed, status: 'all', axis: 'symptom', metric: 'count' })
    rerender(<KpiStrip {...common} filter={{ ...filter, ...narrowed, outcome: 'hardware' }} />)
    expect([hw.getAttribute('aria-pressed'), combined.getAttribute('aria-pressed')]).toEqual(['true', 'false'])
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

    // The chart card is a named section; a pane with no failures beside a charted
    // one says so in a line of muted text, not a blank chart or a second empty state.
    const section = screen.getByRole('region', { name: 'Failures by day' })
    const paneEmpty = within(section).getByText('No System Software Fault results.')
    expect(paneEmpty.tagName).toBe('P')
    expect(paneEmpty.classList.contains('text-foreground-muted')).toBe(true)
    expect(section.querySelector('[data-slot="empty"]')).toBeNull()
    const events = mocks.useECharts.mock.calls[0][2]
    events.click({ seriesName: 'Other' })
    expect(setFilter).not.toHaveBeenCalled()
    events.click({ seriesName: 'Xid79' })
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ symptom: 'Xid79' }))
  })

  it('keeps its pane titles while the breakdowns load, and calls nothing empty', () => {
    const none = { days: [], series: [], total: 0 }
    render(<BreakdownByDay hardware={none} software={none} filter={filter} setFilter={vi.fn()} loading />)

    const section = screen.getByRole('region', { name: 'Failures by day' })
    expect(section.querySelector('[data-slot="empty"]')).toBeNull()
    expect(within(section).queryByText(/No .* results/)).toBeNull()
    expect(within(section).getAllByRole('heading', { level: 3 }).map((title) => title.textContent))
      .toEqual(['Hardware Fault', 'System Software Fault'])
    expect(section.querySelector('[aria-busy="true"]')?.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(2 * 2)
    expect(mocks.useECharts).not.toHaveBeenCalled()
  })

  it('stands one empty state for both panes when neither outcome failed', () => {
    const none = { days: [], series: [], total: 0 }
    render(<BreakdownByDay hardware={none} software={none} filter={{ ...filter, from: '2026-05-01', to: '2026-05-03' }} setFilter={vi.fn()} />)

    const section = screen.getByRole('region', { name: 'Failures by day' })
    expect(within(section).queryByRole('heading', { level: 3 })).toBeNull()
    const empty = section.querySelector<HTMLElement>('[data-slot="empty"]')!
    expect(empty.getAttribute('data-size')).toBe('sm')
    expect(within(empty).getByText('No failures to chart').getAttribute('data-slot')).toBe('empty-title')
    // Hardware and system software failures together are the KPI's infrastructure failures.
    expect(within(empty).getByText('No infrastructure failures in this window.').getAttribute('data-slot')).toBe('empty-description')
    expect(mocks.useECharts).not.toHaveBeenCalled()
  })

  it('says the breakdowns could not be loaded rather than charting no failures', () => {
    const none = { days: [], series: [], total: 0 }
    render(<BreakdownByDay hardware={none} software={none} filter={filter} setFilter={vi.fn()} failed />)

    // As the run list does: the section's one empty state names the failure and the way out.
    const section = screen.getByRole('region', { name: 'Failures by day' })
    const empty = section.querySelector<HTMLElement>('[data-slot="empty"]')!
    expect(within(empty).getByText('Failures by day could not be loaded').getAttribute('data-slot')).toBe('empty-title')
    expect(within(empty).getByText('Refresh to try again.').getAttribute('data-slot')).toBe('empty-description')
    expect(within(section).queryByText(/No (failures|infrastructure|.* results)/)).toBeNull()
    expect(within(section).queryByRole('heading', { level: 3 })).toBeNull()
    expect(mocks.useECharts).not.toHaveBeenCalled()
  })

  it('paints each pane in its outcome color and filters symptoms from the legend', () => {
    const setFilter = vi.fn()
    render(<BreakdownByDay
      hardware={{
        days: ['2026-05-01', '2026-05-02'],
        series: [
          { symptom: 'Xid79', data: [1, 1], total: 2 },
          { symptom: 'Xid48', data: [0, 1], total: 1 },
          { symptom: 'Other', data: [1, 0], total: 1 },
        ],
        total: 4,
      }}
      software={{ days: ['2026-05-01', '2026-05-02'], series: [], total: 0 }}
      filter={{ ...filter, symptom: 'Xid48' }}
      setFilter={setFilter}
    />)

    // The segment color comes from the pane's outcome, never from a symptom's rank.
    const option = mocks.useECharts.mock.calls[0][1]
    const symptomSeries = option.series.slice(0, 3)
    const totals = option.series[3]
    expect(option.series).toHaveLength(4)
    expect(symptomSeries.map((series: { itemStyle: object }) => series.itemStyle)).toEqual([
      { color: 'token(--outcome-hardware)', borderColor: 'token(--chart-surface)', borderWidth: 1, opacity: 0.25 },
      { color: 'token(--outcome-hardware)', borderColor: 'token(--chart-surface)', borderWidth: 1, opacity: 1 },
      { color: 'token(--outcome-hardware)', borderColor: 'token(--chart-surface)', borderWidth: 1, opacity: 0.25 },
    ])
    // Each day's total caps its stack from a silent layer of its own, which no
    // selected or hovered symptom dims: the symptoms carry no labels.
    expect(symptomSeries.map((series: { label?: unknown }) => series.label)).toEqual([undefined, undefined, undefined])
    expect(totals).toMatchObject({ type: 'bar', stack: 'sym', silent: true, label: { position: 'top' }, blur: { label: { opacity: 1 } } })
    expect(totals).not.toHaveProperty('itemStyle')
    expect(totals.data).toEqual([
      { value: 0, label: { show: true, formatter: '2' } },
      { value: 0, label: { show: true, formatter: '2' } },
    ])
    expect(option).not.toHaveProperty('legend')
    // Chart chrome comes from the registered token theme: the option sets no color, font or tooltip surface of its own.
    expect(JSON.stringify(option)).not.toMatch(/#[0-9a-f]{3,8}\b|rgba?\(|JetBrains|font(Size|Family|Weight)|backgroundColor|textStyle/i)
    // Recorded names are escaped in the tooltip's HTML, and empty segments are left out.
    const tooltip = option.tooltip.formatter([
      { axisValue: '05-02', seriesName: '<img src=x onerror=alert(1)>', value: 2 },
      { axisValue: '05-02', seriesName: 'Xid48', value: 0 },
    ])
    expect(tooltip).toContain('&lt;img src=x onerror=alert(1)&gt;: <span class="num text-foreground">2</span>')
    expect(tooltip).not.toMatch(/<img|Xid48/)
    expect(tooltip).toContain('Total: <span class="num text-foreground">2</span>')
    expect(option.tooltip.formatter([{ axisValue: '05-02', seriesName: 'Xid48', value: 0 }])).toBe('')
    // The pane header: its outcome dot, its title, and its total beside the title, outside its name.
    const title = screen.getByRole('heading', { level: 3, name: 'Hardware Fault' })
    const header = title.parentElement!
    expect(header.querySelector('[data-slot="outcome-dot"]')?.getAttribute('data-outcome')).toBe('hardware')
    expect(title.nextElementSibling?.textContent).toBe('4')

    const legend = within(screen.getByRole('list', { name: 'Hardware Fault root symptoms' }))
    const xid79 = legend.getByRole('button', { name: 'Xid79 2' })
    expect(xid79.getAttribute('aria-pressed')).toBe('false')
    expect(legend.getByRole('button', { name: 'Xid48 1' }).getAttribute('aria-pressed')).toBe('true')
    // Other folds unnamed symptoms together, so it is listed but cannot filter.
    expect(legend.getByText('Other')).toBeTruthy()
    expect(legend.queryByRole('button', { name: /Other/ })).toBeNull()

    fireEvent.click(xid79)
    expect(setFilter).toHaveBeenLastCalledWith(expect.objectContaining({ symptom: 'Xid79' }))
    fireEvent.click(legend.getByRole('button', { name: 'Xid48 1' }))
    expect(setFilter.mock.lastCall![0]).not.toHaveProperty('symptom')

    // A legend entry and its segments point at each other.
    fireEvent.focus(xid79)
    expect(mocks.dispatchAction).toHaveBeenLastCalledWith({ type: 'highlight', seriesName: 'Xid79' })
    fireEvent.blur(xid79)
    expect(mocks.dispatchAction).toHaveBeenLastCalledWith({ type: 'downplay', seriesName: 'Xid79' })
    fireEvent.mouseEnter(xid79)
    expect(mocks.dispatchAction).toHaveBeenLastCalledWith({ type: 'highlight', seriesName: 'Xid79' })
    fireEvent.mouseLeave(xid79)
    expect(mocks.dispatchAction).toHaveBeenLastCalledWith({ type: 'downplay', seriesName: 'Xid79' })
    const events = mocks.useECharts.mock.calls[0][2]
    act(() => events.mouseover({ seriesName: 'Xid79' }))
    expect(xid79.hasAttribute('data-highlighted')).toBe(true)
    act(() => events.mouseout({}))
    expect(xid79.hasAttribute('data-highlighted')).toBe(false)
  })

  it('gives a chart updated in place, as by Refresh, only the series its new data holds', () => {
    const none = { days: [], series: [], total: 0 }
    const hardware = (dayCount: number, series: DailyBreakdown['series']): DailyBreakdown => ({
      days: Array.from({ length: dayCount }, (_, index) => `2026-05-${String(index + 1).padStart(2, '0')}`),
      series,
      total: series.reduce((sum, { total }) => sum + total, 0),
    })
    const pane = (breakdown: DailyBreakdown) => <BreakdownByDay hardware={breakdown} software={none} filter={filter} setFilter={vi.fn()} />
    // The option the chart is given: useECharts applies each one whole, never merged into the last (hooks test).
    const option = () => mocks.useECharts.mock.lastCall![1] as { series: { name?: string; id?: string; data: unknown[] }[] }
    const layers = () => option().series.map((series) => series.name ?? series.id)
    const totals = () => option().series.at(-1)!.data.map((day) => (day as { label: { formatter: string } }).label.formatter)
    const { rerender } = render(pane(hardware(2, [
      { symptom: 'Xid79', data: [1, 1], total: 2 },
      { symptom: 'Xid48', data: [1, 0], total: 1 },
    ])))
    expect(layers()).toEqual(['Xid79', 'Xid48', 'day-totals'])
    expect(totals()).toEqual(['2', '1'])

    // A symptom the new data drops is gone, and each total is its day's remaining stack.
    rerender(pane(hardware(2, [{ symptom: 'Xid79', data: [1, 1], total: 2 }])))
    expect(layers()).toEqual(['Xid79', 'day-totals'])
    expect(totals()).toEqual(['1', '1'])

    // A new symptom takes its place in the data's order, under the totals that cap the stacks.
    rerender(pane(hardware(2, [
      { symptom: 'Xid13', data: [2, 3], total: 5 },
      { symptom: 'Xid79', data: [1, 1], total: 2 },
    ])))
    expect(layers()).toEqual(['Xid13', 'Xid79', 'day-totals'])
    expect(totals()).toEqual(['3', '4'])

    // Past ten days no day carries a total.
    rerender(pane(hardware(13, [{ symptom: 'Xid79', data: Array(13).fill(1), total: 13 }])))
    expect(layers()).toEqual(['Xid79'])
    // Each update is the same chart given a new option, not a new chart.
    expect(mocks.useECharts.mock.calls.every(([ref]) => ref === mocks.useECharts.mock.calls[0][0])).toBe(true)
  })
})
