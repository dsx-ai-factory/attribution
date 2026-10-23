// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactElement, ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { RunSummary } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import type { DashboardPayload } from '@/interfaces/meta'
import { OUTCOMES } from '@/lib/attribution-data'
import { DEFINITIONS } from '@/lib/definitions'
import { ParetoCard } from '@/routers/fleet/components/pareto'
import { RunsSection } from '@/routers/fleet/components/runs-section'
import { ActiveFilters, RunsToolbar } from '@/routers/fleet/components/runs-toolbar'

const mocks = vi.hoisted(() => ({
  useECharts: vi.fn(),
  nav: vi.fn(),
  downloadBlob: vi.fn(),
}))

vi.mock('@/hooks/use-echarts', () => ({
  useECharts: mocks.useECharts,
}))
// jsdom loads no stylesheet, so the real readers would throw; these name the token they were asked for.
vi.mock('@/lib/tokens', () => ({
  readColor: (name: string) => `token(${name})`,
  outcomeColor: (outcome: string) => `token(--outcome-${outcome})`,
  chartFonts: () => ({ sans: 'token(--app-font-sans)', mono: 'token(--app-font-mono)' }),
}))
vi.mock('@/hooks/use-nav', () => ({ useNav: () => mocks.nav }))
vi.mock('@/components/node-hover-card', () => ({
  NodeHoverCard: ({ children }: { children: ReactNode }) => <>{children}</>,
}))
vi.mock('@/components/time-hover', () => ({
  // Marks the table cells' times, whose invalid value is the cells' dash (TimeHover's own test).
  TimeHover: ({ date, cell }: { date: string; cell?: boolean }) => <time dateTime={date} data-cell={cell || undefined}>{date}</time>,
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

const run = (overrides: Partial<RunSummary> = {}): RunSummary => ({
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
  decisionGraphState: 'absent',
  rootSymptom: { symptom: 'Xid79', rule: 'xid', time: '', line: '' },
  symptoms: [],
  responsibleNodes: ['node-a', 'node-b', 'node-c', 'node-d'],
  pipeline: '1',
  service: '2',
  ...overrides,
})

// Text a reader sees: a hint's hidden definition is its button's description, not visible text.
function visibleText(element: Element) {
  const copy = element.cloneNode(true) as Element
  copy.querySelectorAll('[hidden]').forEach((node) => node.remove())
  return copy.textContent?.replace(/\s+/g, ' ')
}

// Links resolve their href from the router; the Radix Select measures its content.
const renderInRouter = (ui: ReactElement) => render(ui, { wrapper: MemoryRouter })

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  Element.prototype.scrollIntoView = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('RunsToolbar', () => {
  it('updates status, active filters, and visible exports', () => {
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
    renderInRouter(
      <>
        <RunsToolbar filteredRuns={[run()]} filter={activeFilter} setFilter={setFilter} />
        <ActiveFilters filter={activeFilter} setFilter={setFilter} />
      </>,
    )

    fireEvent.click(screen.getByRole('checkbox', { name: 'Hide passing runs' }))
    // A chip's clear button is a mark beside its value, as a copy button is:
    // on the value's first line, centered on its lowercase.
    const clear = screen.getByRole('button', { name: 'Clear Symptom filter' })
    expect([...clear.classList]).toEqual(expect.arrayContaining(['relative', 'top-(--mark-offset)']))
    expect(clear.parentElement!.classList.contains('items-start')).toBe(true)
    // A long value wraps; its label keeps its width rather than breaking with it.
    expect(screen.getByText('Symptom:').classList.contains('shrink-0')).toBe(true)
    fireEvent.click(clear)
    fireEvent.click(screen.getByRole('button', { name: 'Download as JSON' }))

    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ status: 'all' }))
    expect(setFilter.mock.calls[1][0]).not.toHaveProperty('symptom')
    expect(mocks.downloadBlob).toHaveBeenCalledWith(
      'runs-20260501-0102.json',
      expect.any(String),
      'application/json',
    )
    expect(screen.getByText('Export visible (1)')).toBeTruthy()

    // Each filter reads "Label: value" in one list named by its visible heading.
    const active = screen.getByRole('list', { name: 'Active filters' })
    expect(within(active).getAllByRole('listitem').map((item) => item.textContent)).toEqual([
      'Outcome:Hardware Fault',
      'Symptom:Xid79',
      'Responsible node:node-a',
      'Rule:xid',
      'User:alice',
      'Search:team',
      'Status:Passing runs hidden',
    ])
    // The bar is a 28px row like the table's: each chip a 20px badge, its clear
    // button 16px, and the chip does not clip the button's focus ring.
    expect([...active.parentElement!.classList]).toEqual(expect.arrayContaining(['px-card-x', 'py-1']))
    const status = screen.getByRole('button', { name: 'Clear Status filter' })
    expect([...status.classList]).toEqual(expect.arrayContaining(['size-4', "[&_svg:not([class*='size-'])]:size-3"]))
    expect(status.classList.contains('size-5')).toBe(false)
    const chip = status.closest<HTMLElement>('[data-slot="badge"]')!
    expect([...chip.classList]).toEqual(expect.arrayContaining(['min-h-5', 'py-px', 'overflow-visible']))
    expect(chip.classList.contains('overflow-hidden')).toBe(false)
    // Hiding passing runs is a filter like any other, so it gets a clearable chip too.
    fireEvent.click(status)
    expect(setFilter).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'all' }))
    // Each clear button drops its own filter and keeps the others.
    const cleared = { Outcome: 'outcome', 'Responsible node': 'node', Rule: 'rule', User: 'user', Search: 'q' } as const
    for (const [label, key] of Object.entries(cleared)) {
      fireEvent.click(screen.getByRole('button', { name: `Clear ${label} filter` }))
      const next = setFilter.mock.lastCall![0]
      expect(next[key], label).toBeUndefined()
      expect(next.symptom, label).toBe('Xid79')
    }
  })

  it('shows every keystroke at once and writes the search once typing pauses', () => {
    vi.useFakeTimers()
    const setFilter = vi.fn()
    const { rerender } = renderInRouter(<RunsToolbar filteredRuns={[]} filter={filter} setFilter={setFilter} />)

    // Named for everything it searches; the placeholder says it searches more than one field.
    const search = screen.getByRole<HTMLInputElement>('textbox', {
      name: 'Filter runs by execution context, workload, user, symptom, rule, node, attempt UID or detail',
    })
    expect(search.placeholder).toBe('Filter by context, user, symptom…')
    expect(search.name).toBe('q')
    // 288px wide, never wider than the header row a narrow screen leaves it.
    expect([...search.closest('[data-slot="input-group"]')!.classList]).toEqual(expect.arrayContaining(['w-72', 'max-w-full']))
    fireEvent.change(search, { target: { value: 'ex' } })
    fireEvent.change(search, { target: { value: 'example' } })
    // The field holds what is typed while the URL has not caught up.
    expect(search.value).toBe('example')
    expect(setFilter).not.toHaveBeenCalled()
    act(() => { vi.advanceTimersByTime(300) })
    expect(setFilter).toHaveBeenCalledTimes(1)
    expect(setFilter).toHaveBeenLastCalledWith(expect.objectContaining({ q: 'example' }))

    // The search comes back through the URL; what was typed meanwhile stays.
    fireEvent.change(search, { target: { value: 'example-' } })
    rerender(<RunsToolbar filteredRuns={[]} filter={{ ...filter, q: 'example' }} setFilter={setFilter} />)
    expect(search.value).toBe('example-')
    // A search cleared elsewhere (its filter chip) empties the field, and nothing is written back.
    rerender(<RunsToolbar filteredRuns={[]} filter={filter} setFilter={setFilter} />)
    expect(search.value).toBe('')
    act(() => { vi.advanceTimersByTime(300) })
    expect(setFilter).toHaveBeenCalledTimes(1)
  })

  it('shows no active filters row when nothing narrows the list', () => {
    renderInRouter(<ActiveFilters filter={{ ...filter, status: 'all' }} setFilter={vi.fn()} />)
    expect(screen.queryByRole('list', { name: 'Active filters' })).toBeNull()
  })

  it('releases the conflicting control when Pass and hide-passing would contradict', () => {
    const setFilter = vi.fn()
    const { rerender } = renderInRouter(
      <RunsToolbar filteredRuns={[run()]} filter={filter} setFilter={setFilter} />,
    )

    const outcome = screen.getByRole('combobox', { name: 'Outcome' })
    expect(outcome.textContent).toBe('OutcomeAll')
    fireEvent.keyDown(outcome, { key: 'ArrowDown' })
    // Each outcome choice is marked by its own tone, through data-outcome only.
    for (const [key, { label }] of Object.entries(OUTCOMES)) {
      const dot = screen.getByRole('option', { name: label }).querySelector('[data-slot="outcome-dot"]')
      expect(dot?.getAttribute('data-outcome'), key).toBe(key)
    }
    // All has no dot: its label starts on the outcomes' label column, 24px in
    // (the 8px inset, the 8px dot and its 8px gap); the trigger shows no gap.
    const all = screen.getByRole('option', { name: 'All' })
    expect(all.querySelector('[data-slot="outcome-dot"]')).toBeNull()
    expect(all.classList.contains('pl-6')).toBe(true)
    expect(all.classList.contains('pl-2')).toBe(false)
    expect(screen.getByRole('option', { name: 'Pass' }).classList.contains('pl-2')).toBe(true)

    // Picking Pass while passing runs are hidden: the outcome wins, hiding is released.
    fireEvent.click(screen.getByRole('option', { name: 'Pass' }))
    expect(setFilter).toHaveBeenCalledWith(
      expect.objectContaining({ outcome: 'pass', status: 'all' }),
    )

    // Hiding passing runs while Pass is selected: the checkbox wins, the outcome is cleared.
    setFilter.mockClear()
    rerender(
      <RunsToolbar
        filteredRuns={[run()]}
        filter={{ ...filter, status: 'all', outcome: 'pass' }}
        setFilter={setFilter}
      />,
    )
    expect(screen.getByRole('combobox', { name: 'Outcome' }).textContent).toBe('OutcomePass')
    fireEvent.click(screen.getByRole('checkbox', { name: 'Hide passing runs' }))
    expect(setFilter).toHaveBeenCalledWith(
      expect.objectContaining({ status: 'failed', outcome: undefined }),
    )
  })
})

describe('RunsSection', () => {
  it('renders one header row, grouped rows, links, node overflow, collapse, and pagination', () => {
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
    const { rerender } = renderInRouter(<RunsSection {...props} />)

    // The section is the Runs region the page scrolls to and focuses, showing
    // that focus with the one 3px ring instead of the browser outline.
    const region = screen.getByRole('region', { name: 'Runs' })
    expect(region.id).toBe('runs-section')
    expect(region.tabIndex).toBe(-1)
    expect([...region.classList]).toEqual(expect.arrayContaining([
      'outline-none', 'focus-visible:ring-[3px]', 'focus-visible:ring-border-active/80',
    ]))
    // Scrolled to, it lands under the sticky bars through the page's scroll
    // padding (globals.css), which a margin of its own would add to.
    expect(region.className).not.toMatch(/scroll-m/)
    expect(within(region).getByRole('heading', { level: 2, name: 'Runs' }).nextElementSibling?.textContent).toBe('12')

    // One header row for every group, with no Outcome column (the group names it) and
    // no Attempt column (the adapter records a constant there).
    const columns = [
      'End time', 'Execution context', 'User', 'GPUs', 'GPU-hours', 'Root symptom', 'Responsible nodes', 'Attribution logic',
    ]
    expect(screen.getAllByRole('row').filter((row) => within(row).queryAllByRole('columnheader').length > 0)).toHaveLength(1)
    expect(screen.getAllByRole('columnheader')).toHaveLength(columns.length)
    columns.forEach((name, index) => expect(screen.getAllByRole('columnheader')[index]).toBe(screen.getByRole('columnheader', { name })))
    // The header is named by its label alone; the definition is its hint's
    // description, not text read on every move into the column.
    const { def, formula } = DEFINITIONS['gpu-hours']
    const gpuHours = screen.getByRole('columnheader', { name: 'GPU-hours' })
    expect(within(gpuHours).getByRole('button', { name: 'About GPU-hours', description: `${def} Formula: ${formula}` })).toBeTruthy()

    // The execution context is the row's link; a click elsewhere on the row goes there too.
    expect(screen.getByRole('link', { name: 'team/job' }).getAttribute('href')).toBe('/verdict/run-1')
    // Recorded identifiers wrap in their columns, so the table fits its card
    // whatever their length: the context and each responsible node as inline
    // links, the user and the root symptom as text, the nodes' list onto
    // further lines.
    const context = screen.getByRole('link', { name: 'team/job' })
    expect(context.getAttribute('data-slot')).toBe('link')
    expect(context.closest('td')!.classList.contains('whitespace-normal')).toBe(true)
    const contextRow = context.closest('tr')!
    expect([...contextRow.children].map((cell) => cell.classList.contains('whitespace-normal'))).toEqual([false, true, true, false, false, true, true, false])
    expect(contextRow.querySelector('ul')!.classList.contains('flex-wrap')).toBe(true)
    const node = screen.getByRole('link', { name: 'node-a' })
    expect(node.getAttribute('data-slot')).toBe('link')
    expect(node.className).not.toMatch(/(^|\s)(h-\d|shrink-0|whitespace-nowrap)(\s|$)/)
    fireEvent.click(screen.getByText('alice').closest('tr') as HTMLElement)
    expect(mocks.nav).toHaveBeenCalledWith('verdict', { id: 'run-1' })
    mocks.nav.mockClear()
    fireEvent.click(screen.getByRole('link', { name: 'team/job' }))
    expect(mocks.nav).not.toHaveBeenCalled()

    // Three responsible nodes inline as links; the rest behind a +N badge, a
    // button named for the nodes it holds, that leaves the row where it is.
    expect(screen.getAllByRole('link', { name: /^node-/ }).map((link) => link.getAttribute('href')))
      .toEqual(['/node/node-a', '/node/node-b', '/node/node-c'])
    const more = screen.getByRole('button', { name: '1 more responsible node: node-d' })
    expect(more.textContent).toBe('+1')
    expect(more.getAttribute('data-variant')).toBe('zinc-subtle')
    expect(more.getAttribute('type')).toBe('button')
    fireEvent.click(more)
    expect(mocks.nav).not.toHaveBeenCalled()
    expect(screen.queryByText('...')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Next Hardware Fault page' }))
    expect(setGroupPage).toHaveBeenCalledWith('hardware', 1)
    const pageSize = screen.getByRole('combobox', { name: 'Hardware Fault records per page' })
    fireEvent.keyDown(pageSize, { key: 'ArrowDown' })
    fireEvent.click(screen.getByRole('option', { name: '25 / page' }))
    expect(setPageSize).toHaveBeenCalledWith(25)
    // A size that fits the whole group keeps its pager, so the select just
    // used, and the keyboard focus on it, stays; there is no page to step to.
    pageSize.focus()
    rerender(<RunsSection {...props} pageSize={25} />)
    expect(document.activeElement).toBe(pageSize)
    expect(screen.getByRole('group', { name: 'Hardware Fault pages' }).textContent).toContain('1–12 of 12 runs')
    expect(screen.getByRole('button', { name: 'Next Hardware Fault page' }).getAttribute('aria-disabled')).toBe('true')
    rerender(<RunsSection {...props} />)

    // The outcome is named in full once, by the group's disclosure button and its tone; rows carry no outcome badge.
    expect(document.querySelector('tbody [data-slot="badge"][data-outcome]')).toBeNull()
    expect(screen.queryByText('HW')).toBeNull()
    const group = screen.getByRole('button', { name: 'Hardware Fault 12' })
    expect(group.querySelector('[data-slot="outcome-dot"]')?.getAttribute('data-outcome')).toBe('hardware')
    expect(group.getAttribute('aria-expanded')).toBe('true')
    const rows = document.getElementById(group.getAttribute('aria-controls')!)
    expect(rows?.contains(screen.getByRole('link', { name: 'team/job' }))).toBe(true)
    fireEvent.click(group)
    expect(group.getAttribute('aria-expanded')).toBe('false')
    expect(rows?.hidden).toBe(true)
    expect(screen.queryByRole('link', { name: 'team/job' })).toBeNull()
  })

  it('shows a dash for absent values and no pagination for a group on one page', () => {
    renderInRouter(
      <RunsSection
        groups={[{ outcome: 'software', runs: [run({ outcome: 'software', user: '', responsibleNodes: [], rootSymptom: { symptom: '', rule: '', time: '', line: '' } })], total: 1, page: 0, pageSize: 10 }]}
        matchingRuns={1}
        visibleRuns={[]}
        filter={filter}
        setFilter={vi.fn()}
        pageByOutcome={{}}
        pageSize={10}
        setPageSize={vi.fn()}
        setGroupPage={vi.fn()}
      />,
    )

    const row = screen.getByRole('link', { name: 'team/job' }).closest('tr') as HTMLElement
    // User, root symptom and responsible nodes left empty all read as the one muted dash.
    expect(within(row).getAllByText('None')).toHaveLength(3)
    // The end time is a table cell's, so one that does not parse would be the
    // same dash, read as "Unavailable".
    expect(within(row).getAllByRole('cell')[0].querySelector('time')!.getAttribute('data-cell')).toBe('true')
    expect(screen.queryByRole('group', { name: /pages/ })).toBeNull()
    // A recorded identifier may break after its separators before anywhere else.
    expect(screen.getByRole('link', { name: 'team/job' }).innerHTML).toBe('<span>team/<span aria-hidden="true" class="select-none">\u200b</span>job</span>')
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
    const { rerender } = renderInRouter(<RunsSection {...common} loading />)
    expect(document.querySelectorAll('[class*="animate-pulse"]')).toHaveLength(6)
    // No count is claimed while the rows are loading: the title runs straight into the toolbar.
    expect(screen.getByRole('heading', { level: 2, name: 'Runs' }).nextElementSibling?.getAttribute('data-slot')).toBe('card-action')

    // Another page of the same filter loading: the groups that filter lists
    // stay, with their totals and pagination (so the control pressed keeps
    // focus), and only their rows wait.
    rerender(
      <RunsSection
        {...common}
        groups={[{ outcome: 'hardware', runs: [run()], total: 12, page: 0, pageSize: 10 }]}
        matchingRuns={12}
        pageByOutcome={{ hardware: 1 }}
        loading
      />,
    )
    const group = screen.getByRole('button', { name: 'Hardware Fault 12' })
    const rows = document.getElementById(group.getAttribute('aria-controls')!)!
    expect(rows.getAttribute('aria-busy')).toBe('true')
    expect(within(rows).queryByRole('link')).toBeNull()
    // The second page of 12 runs holds 2.
    expect(rows.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(2)
    expect(within(rows).getByRole('button', { name: 'Previous Hardware Fault page' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 2, name: 'Runs' }).nextElementSibling?.getAttribute('data-slot')).toBe('card-action')

    rerender(<RunsSection {...common} />)
    expect(screen.getByText('No runs in this view')).toBeTruthy()
    expect(screen.getByText(
      'Try to uncheck "Hide passing runs".',
    )).toBeTruthy()
    expect(screen.queryByText(/filters above/)).toBeNull()
    expect(screen.queryByText(/widen the date range/)).toBeNull()

    rerender(
      <RunsSection
        {...common}
        filter={{ ...filter, node: 'node-1', from: '2026-05-01', to: '2026-05-03' }}
      />,
    )
    expect(screen.getByText(
      'Try to clear the active filters above, or uncheck "Hide passing runs", or '
      + 'widen the date range.',
    )).toBeTruthy()
  })

  it('says the runs could not be loaded instead of calling a failed fetch an empty view', () => {
    renderInRouter(
      <RunsSection
        groups={[]}
        matchingRuns={0}
        visibleRuns={[]}
        filter={{ ...filter, node: 'node-1' }}
        setFilter={vi.fn()}
        pageByOutcome={{}}
        pageSize={10}
        setPageSize={vi.fn()}
        setGroupPage={vi.fn()}
        failed
      />,
    )

    expect(screen.getByText('Runs could not be loaded').getAttribute('data-slot')).toBe('empty-title')
    expect(screen.getByText('Refresh to try again.').getAttribute('data-slot')).toBe('empty-description')
    // No count is stated and no filter is blamed for what the fetch did not return.
    expect(within(screen.getByRole('region', { name: 'Runs' })).queryByText('0')).toBeNull()
    expect(screen.queryByText('No runs in this view')).toBeNull()
    expect(screen.queryByText(/^Try to/)).toBeNull()
  })

  it('drops the loosen-a-filter advice when the database itself is empty', () => {
    renderInRouter(
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

    expect(screen.getByText('No runs in this database').getAttribute('data-slot')).toBe('empty-title')
    expect(screen.getByText('Nothing has been attributed yet.').getAttribute('data-slot')).toBe('empty-description')
    expect(screen.queryByText(/^Try to/)).toBeNull()
  })
})

// How many categories tie for largest, as the server counts them over the
// uncut ledgers: one in every grouping and measure unless given.
function tiesOf(given: Partial<DashboardPayload['largestTies']> = {}): DashboardPayload['largestTies'] {
  const one = { count: 1, nodeHours: 1 }
  return { symptom: one, node: one, user: one, rule: one, ...given }
}

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
      categoryCounts={{ symptom: 2, node: 0, user: 0, rule: 0 }}
      largestTies={tiesOf()}
    />)

    const card = screen.getByRole('region', { name: 'Failure concentration' })
    // What the chart shows is the hint beside the title, read as its description.
    expect(within(card).getByRole('button', { name: 'About Failure concentration', description: 'Failed attempts ranked by root symptom, the ten largest; the line is their running share of all failed attempts.' })).toBeTruthy()
    // The toggles carry their choice alone, with no hint inside an item.
    expect(within(screen.getByRole('group', { name: 'Group by' })).getAllByRole('radio').map((item) => item.textContent))
      .toEqual(['Root symptom', 'Responsible node', 'User', 'Rule'])
    expect(within(screen.getByRole('group', { name: 'Measure' })).getAllByRole('radio').map((item) => item.textContent))
      .toEqual(['Count', 'GPU-hours'])
    // Each group's name keeps to one line and to its group's first row, as the group wraps on a narrow screen.
    for (const name of ['Group by', 'Measure']) {
      expect([...screen.getByText(name).classList]).toEqual(expect.arrayContaining(['flex', 'h-7', 'shrink-0', 'items-center']))
    }
    expect(within(card).queryAllByRole('button', { name: /^About / }).map((hint) => hint.getAttribute('aria-label')))
      .toEqual(['About Failure concentration'])
    // Nothing is selected, so there is nothing to clear: the control holds its
    // place at the end of the header, unavailable but focusable, and does nothing.
    const clear = within(card).getByRole('button', { name: 'Clear selection' })
    expect(clear.getAttribute('aria-disabled')).toBe('true')
    expect(clear.parentElement!.lastElementChild).toBe(clear)
    fireEvent.click(clear)
    expect(setFilter).not.toHaveBeenCalled()
    // '<Xid79>' alone reaches 80%, so it is the whole vital few; the summary
    // says so in plain words, with the share it reaches and the largest's value and share.
    expect(visibleText(card.querySelector('[data-slot="card-footer"] p')!))
      .toBe('Top 1 of 2 root symptoms accounts for 80% of failed attempts · Largest: <Xid79>, 8 failed attempts (80%)')
    // It fills its row beside the Ranked list button, and takes a row of its own before it would have under 16rem.
    expect([...card.querySelector('[data-slot="card-footer"] p')!.classList]).toEqual(expect.arrayContaining(['min-w-0', 'grow', 'basis-64']))
    const paretoOption = mocks.useECharts.mock.calls[0][1]
    // Chart chrome comes from the registered token theme: the option sets no color, font or tooltip surface of its own.
    expect(JSON.stringify(paretoOption)).not.toMatch(/#[0-9a-f]{3,8}\b|rgba?\(|JetBrains|font(Size|Family|Weight)|backgroundColor|textStyle/i)
    const tooltip = paretoOption.tooltip.formatter
    expect(tooltip([])).toBe('')
    const barTooltip = tooltip([{
      seriesIndex: 0,
      axisValue: '<Xid79>',
      data: { value: 8, dominant: 'hardware', running: 8 },
    }])
    expect(barTooltip).toContain('&lt;Xid79&gt;')
    // Only the swatch carries the outcome, from its data-outcome tone; tooltip text stays neutral.
    expect(barTooltip).toContain('<span data-outcome="hardware"')
    expect(barTooltip).not.toMatch(/#[0-9a-f]{6}|token\(/i)
    // Tooltips are DOM: text takes the app's neutral classes and values read as numbers.
    expect(barTooltip).toContain('Failed attempts: <span class="num text-foreground">8</span>')
    expect(barTooltip).toContain('Share: <span class="num text-foreground">80%</span>')
    expect(barTooltip).toContain('Running share: <span class="num text-foreground">80%</span>')
    // Room above the plot for the axis names.
    expect(paretoOption.grid.top).toBeGreaterThanOrEqual(28)

    // A bar takes its dominant outcome's color; the running share and 80% mark are neutral.
    expect(paretoOption.series).toHaveLength(2)
    const [bars, cumulative] = paretoOption.series
    expect(bars.data.map((bar: { itemStyle: { color: string } }) => bar.itemStyle.color))
      .toEqual(['token(--outcome-hardware)', 'token(--outcome-software)'])
    expect(cumulative.lineStyle.color).toBe('token(--foreground-secondary)')
    expect(cumulative.itemStyle.color).toBe('token(--foreground-secondary)')
    expect(cumulative.symbol).toBe('none')
    expect(cumulative.markLine.data[0].lineStyle.color).toBe('token(--foreground-muted)')
    // The mark is drawn in place: every option replaces the chart and rebuilds
    // it, so a sweep would replay on each selection and change of width.
    expect(cumulative.markLine.animation).toBe(false)
    // The running share can never cross a bar value: no series draws value
    // labels; the tooltip, the axis and the Ranked list give the values.
    expect(paretoOption.series.map((series: { label?: unknown }) => series.label)).toEqual([undefined, undefined])
    // Counts step by whole numbers; every category is named, cut only to the width of its band.
    expect(paretoOption.yAxis[0].minInterval).toBe(1)
    // The axis picks its own round maximum rather than the largest value, and
    // its ticks read at most five characters, so the plot keeps its inset.
    expect(paretoOption.yAxis[0]).not.toHaveProperty('max')
    expect(paretoOption.grid).toMatchObject({ left: 48, right: 56, outerBoundsMode: 'none' })
    const tick = paretoOption.yAxis[0].axisLabel.formatter
    expect([0, 0.3, 2.5, 25, 900, 1000, 2500, 12000, 1_500_000].map(tick))
      .toEqual(['0', '0.3', '2.5', '25', '900', '1k', '2.5k', '12k', '1.5M'])
    expect(paretoOption.xAxis.axisLabel).toMatchObject({ interval: 0, overflow: 'truncate', ellipsis: '…' })
    expect(paretoOption.xAxis.axisLabel.width).toBeGreaterThanOrEqual(0)

    const donutOption = mocks.useECharts.mock.calls[1][1]
    const [donut] = donutOption.series
    expect(donut.data.map((slice: { itemStyle: { color: string } }) => slice.itemStyle.color))
      .toEqual(['token(--outcome-hardware)', 'token(--outcome-software)'])
    expect(donut.itemStyle.borderColor).toBe('token(--chart-surface)')
    // The center total is the chart's title, not a label of the slices, which
    // take a slice's dimmed opacity once another outcome is selected: a 20px
    // mono number (the KPI size) over a sentence-case muted caption.
    expect(donut.label).toEqual({ show: false })
    expect(donutOption.title).toMatchObject({ text: '{num|10}\n{lbl|failed attempts}', left: 'center', top: 'middle' })
    expect(donutOption.title.textStyle).toEqual({
      fontWeight: 400,
      rich: {
        num: { color: 'token(--foreground)', fontFamily: 'token(--app-font-mono)', fontSize: 20, fontWeight: 500, lineHeight: 28 },
        lbl: { color: 'token(--foreground-muted)', fontSize: 12, lineHeight: 16 },
      },
    })
    expect(JSON.stringify({ ...donutOption, series: [], title: undefined })).not.toMatch(/#[0-9a-f]{3,8}\b|rgba?\(|JetBrains|backgroundColor|textStyle/i)
    const donutTooltip = donutOption.tooltip.formatter({ name: 'Hardware Fault', value: 8, data: { __key: 'hardware' } })
    expect(donutTooltip).toContain('<span data-outcome="hardware"')
    // Its share is the legend's, of the exact total.
    expect(donutTooltip).toContain('Share: <span class="num text-foreground">80%</span> of total')
    expect([...document.querySelectorAll('[data-slot="outcome-dot"]')].map((dot) => dot.getAttribute('data-outcome')))
      .toEqual(['hardware', 'software'])

    const paretoEvents = mocks.useECharts.mock.calls[0][2]
    paretoEvents.click({ seriesIndex: 1, data: { name: '<Xid79>' } })
    expect(setFilter).not.toHaveBeenCalled()
    paretoEvents.click({ seriesIndex: 0, data: { name: '<Xid79>' } })
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ symptom: '<Xid79>' }))

    const donutEvents = mocks.useECharts.mock.calls[1][2]
    donutEvents.click({ data: { __key: 'hardware' } })
    expect(setFilter).toHaveBeenCalledWith(expect.objectContaining({ outcome: 'hardware' }))

    // The legend sits under the donut as one filter toggle per outcome, the donut's keyboard path.
    const outcomes = within(screen.getByRole('group', { name: 'Outcomes' }))
    const hardware = outcomes.getByRole('button', { name: 'Hardware Fault 8 80%' })
    expect(hardware.getAttribute('aria-pressed')).toBe('false')
    // A label with no room beside the numbers wraps rather than being cut; a
    // row reads from its first line, its dot centred on that 20px line.
    expect([...hardware.classList]).toEqual(expect.arrayContaining(['h-auto', 'min-h-6', 'py-0.5', 'items-start', 'text-left', 'whitespace-normal']))
    expect(hardware.querySelector('.truncate')).toBeNull()
    expect([...hardware.querySelector('[data-slot="outcome-dot"]')!.parentElement!.classList]).toEqual(['flex', 'h-5', 'items-center'])
    fireEvent.click(outcomes.getByRole('button', { name: 'System Software Fault 2 20%' }))
    expect(setFilter).toHaveBeenLastCalledWith(expect.objectContaining({ outcome: 'software' }))
  })

  it('covers empty states, the user axis and clearing a selection', () => {
    const setFilter = vi.fn()
    const { rerender } = render(<ParetoCard
      filter={{ ...filter, axis: 'node' }}
      setFilter={setFilter}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[]}
    />)
    // No failure at all: one empty state, titled and described, stands for the chart, the donut and the summary.
    const empty = screen.getByRole('status')
    expect(empty.getAttribute('data-slot')).toBe('empty')
    expect(within(empty).getByText('No failures to rank').getAttribute('data-slot')).toBe('empty-title')
    expect(within(empty).getByText('No failed attempts for this filter.').getAttribute('data-slot')).toBe('empty-description')
    expect(screen.queryByText(/Largest:/)).toBeNull()
    expect(screen.queryByRole('button', { name: 'Ranked list' })).toBeNull()
    expect(mocks.useECharts).not.toHaveBeenCalled()

    // Failures that named no responsible node: the node chart says so once; the donut still splits them.
    rerender(<ParetoCard
      filter={{ ...filter, axis: 'node' }}
      setFilter={setFilter}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'software', count: 2, nodeHours: 3 }]}
    />)
    expect(screen.getAllByText('No responsible nodes for this filter.')).toHaveLength(1)
    // The chart pane is a sub-block beside the donut, so its empty state is a line of muted text.
    const paneEmpty = screen.getByRole('status')
    expect(paneEmpty.textContent).toBe('No responsible nodes for this filter.')
    expect(paneEmpty.tagName).toBe('P')
    expect(paneEmpty.classList.contains('text-foreground-muted')).toBe(true)
    expect(document.querySelector('[data-slot="empty"]')).toBeNull()
    expect(screen.getByRole('button', { name: 'System Software Fault 2 100%' })).toBeTruthy()
    expect(screen.queryByText(/Largest:/)).toBeNull()

    rerender(<ParetoCard
      filter={{ ...filter, axis: 'user', metric: 'nodeHours', user: 'alice', outcome: 'hardware' }}
      setFilter={setFilter}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={[{ user: 'alice', count: 1, nodeHours: 8, outcomes: { hardware: 1 } }]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 1, nodeHours: 8 }]}
      categoryCounts={{ symptom: 1, node: 0, user: 1, rule: 1 }}
      largestTies={tiesOf()}
    />)
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('The only user is alice: 8.0 GPU-hours (100%)')
    // GPU-hours read to a tenth wherever this card gives one, whole or not.
    expect(screen.getByRole('button', { name: 'Hardware Fault 8.0 100%' })).toBeTruthy()
    const clear = screen.getByRole('button', { name: 'Clear selection' })
    expect(clear.hasAttribute('aria-disabled')).toBe(false)
    fireEvent.click(clear)
    expect(setFilter).toHaveBeenCalledWith({
      status: 'failed',
      axis: 'user',
      metric: 'nodeHours',
    })
  })
  it('gives each category name the width of its band, and steps counts by whole numbers', () => {
    // jsdom has no layout: a chart 904px wide leaves an 800px plot between the value axes.
    const width = vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(904)
    const ledger = [
      { symptom: 'XID-171-UNCORRECTABLE_DRAM_ERROR', count: 3, nodeHours: 10.5, outcomes: { hardware: 3 } },
      { symptom: 'ibv_open_device failed', count: 1, nodeHours: 3.25, outcomes: { software: 1 } },
    ]
    const card = (metric: FleetFilter['metric']) => (
      <ParetoCard
        filter={{ ...filter, metric }}
        setFilter={vi.fn()}
        symptomLedger={ledger}
        nodeLedger={[]}
        userLedger={[]}
        ruleLedger={[]}
        outcomeLedger={[{ outcome: 'hardware', count: 3, nodeHours: 10.5 }, { outcome: 'software', count: 1, nodeHours: 3.25 }]}
      />
    )
    const { rerender } = render(card('count'))
    const pareto = () => mocks.useECharts.mock.calls.filter(([, option]) => 'xAxis' in option).at(-1)![1]

    // Two categories share the plot: 400px bands, less the 8px kept between names.
    expect(pareto().xAxis.axisLabel.width).toBe(392)
    expect(pareto().yAxis[0].minInterval).toBe(1)
    // GPU-hours are not whole: their axis steps by no less than the tenth the
    // values are given to, and its top tick is a round number of its own, never
    // the largest value as a raw float.
    rerender(card('nodeHours'))
    expect(pareto().yAxis[0].minInterval).toBe(0.1)
    expect(pareto().yAxis[0]).not.toHaveProperty('max')
    expect(pareto().xAxis.axisLabel.width).toBe(392)
    expect(pareto().tooltip.formatter([{ seriesIndex: 0, axisValue: 'x', data: { value: 7.666666656732559, dominant: 'hardware', running: 7.666666656732559 } }]))
      .toContain('GPU-hours: <span class="num text-foreground">7.7</span>')
    width.mockRestore()
  })

  it('names a tie for largest as a tie, never one of the tied as the largest alone', () => {
    render(<ParetoCard
      filter={filter}
      setFilter={vi.fn()}
      symptomLedger={[
        { symptom: 'Xid171', count: 1, nodeHours: 16, outcomes: { hardware: 1 } },
        { symptom: 'Xid48', count: 1, nodeHours: 16, outcomes: { hardware: 1 } },
        { symptom: 'ibv_open_device failed', count: 1, nodeHours: 16, outcomes: { software: 1 } },
        { symptom: 'job failed', count: 1, nodeHours: 16, outcomes: { user: 1 } },
      ]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[
        { outcome: 'hardware', count: 2, nodeHours: 32 },
        { outcome: 'software', count: 1, nodeHours: 16 },
        { outcome: 'user', count: 1, nodeHours: 16 },
      ]}
      categoryCounts={{ symptom: 4, node: 0, user: 0, rule: 0 }}
      largestTies={tiesOf({ symptom: { count: 4, nodeHours: 4 } })}
    />)
    // Spread evenly, it takes all four to reach 80%, and together they are all of it.
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('All 4 root symptoms account for 100% of failed attempts · Largest: Xid171 and 3 more, 1 failed attempt each (25% each)')
  })

  it('counts a tie past the ten it ranks, as the server counts it', () => {
    // Twenty-five users failed once each: the ledger holds ten of them, all
    // tied, and the server counts all twenty-five.
    const userLedger = Array.from({ length: 10 }, (_, index) => ({
      user: `user-${String(index).padStart(2, '0')}`, count: 1, nodeHours: 2, outcomes: { hardware: 1 },
    }))
    render(<ParetoCard
      filter={{ ...filter, axis: 'user' }}
      setFilter={vi.fn()}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={userLedger}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 25, nodeHours: 50 }]}
      categoryCounts={{ symptom: 1, node: 1, user: 25, rule: 1 }}
      largestTies={tiesOf({ user: { count: 25, nodeHours: 25 } })}
    />)
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('Top 10 of 25 users account for 40% of failed attempts · Largest: user-00 and 24 more, 1 failed attempt each (4% each)')
  })

  it('ties only values that are equal, not two shown alike', () => {
    // 16.04 and 15.96 GPU-hours both read 16.0, but their shares of 32 read
    // 50% and 50%, of 31 they would not: only an exact tie is named as one.
    render(<ParetoCard
      filter={{ ...filter, metric: 'nodeHours' }}
      setFilter={vi.fn()}
      symptomLedger={[
        { symptom: 'Xid48', count: 1, nodeHours: 16.04, outcomes: { hardware: 1 } },
        { symptom: 'Xid79', count: 1, nodeHours: 15.96, outcomes: { hardware: 1 } },
      ]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 2, nodeHours: 32 }]}
      categoryCounts={{ symptom: 2, node: 0, user: 0, rule: 0 }}
      largestTies={tiesOf({ symptom: { count: 2, nodeHours: 1 } })}
    />)
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('Both root symptoms account for 100% of GPU-hours of failed attempts · Largest: Xid48, 16.0 GPU-hours (50%)')
  })

  it('calls a share all of it when what is left shows as nothing', () => {
    // One user holds every failure, but the server keeps each sum to a
    // millionth: its two outcomes' 1.0000006 GPU-hours read 1.000001 each, the
    // user's 2.0000012 reads 2.000001, a millionth short of their total.
    render(<ParetoCard
      filter={{ ...filter, axis: 'user', metric: 'nodeHours' }}
      setFilter={vi.fn()}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={[{ user: 'alice', count: 2, nodeHours: 2.000001, outcomes: { hardware: 1, software: 1 } }]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 1, nodeHours: 1.000001 }, { outcome: 'software', count: 1, nodeHours: 1.000001 }]}
      categoryCounts={{ symptom: 1, node: 1, user: 1, rule: 1 }}
      largestTies={tiesOf()}
    />)
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('The only user is alice: 2.0 GPU-hours (100%)')
    fireEvent.click(screen.getByRole('button', { name: 'Ranked list' }))
    expect(within(screen.getByRole('table')).getAllByRole('row').slice(1).map((row) => within(row).getAllByRole('cell').map((cell) => cell.textContent)))
      .toEqual([['1', 'alice', '2.0', '100%', '100%']])
  })

  it('cuts the vital few where the running share reads 80%, as the Ranked list shows it', () => {
    // Five symptoms of 1/3 GPU-hour each, kept to a millionth: four reach
    // 79.99995% of the 1.666667 total, which the Ranked list shows as 80%.
    const symptomLedger = Array.from({ length: 5 }, (_, index) => ({ symptom: `s${index}`, count: 1, nodeHours: 0.333333, outcomes: { hardware: 1 } }))
    render(<ParetoCard
      filter={{ ...filter, metric: 'nodeHours' }}
      setFilter={vi.fn()}
      symptomLedger={symptomLedger}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 5, nodeHours: 1.666667 }]}
      categoryCounts={{ symptom: 5, node: 0, user: 0, rule: 0 }}
      largestTies={tiesOf({ symptom: { count: 5, nodeHours: 5 } })}
    />)
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('Top 4 of 5 root symptoms account for 80% of GPU-hours of failed attempts · Largest: s0 and 4 more, 0.3 GPU-hours each (20% each)')
  })

  it.each([
    // A running share shown as none of it (0.049 of 0.06, shown as 0.0) is no cut.
    { values: [0.049, 0.011], summary: 'Both root symptoms account for 100% of GPU-hours of failed attempts · Largest: a, 0.0 GPU-hours (0%)' },
    // One leaving 0.0 shown over (0.151 of 0.2) is all of it, cut there.
    { values: [0.151, 0.049], summary: 'Top 1 of 2 root symptoms accounts for 100% of GPU-hours of failed attempts · Largest: a, 0.2 GPU-hours (100%)' },
  ])('cuts a tiny total as its shares are shown: $values', ({ values, summary }) => {
    const symptomLedger = values.map((nodeHours, index) => ({ symptom: 'ab'[index], count: 1, nodeHours, outcomes: { hardware: 1 } }))
    render(<ParetoCard
      filter={{ ...filter, metric: 'nodeHours' }}
      setFilter={vi.fn()}
      symptomLedger={symptomLedger}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 2, nodeHours: values[0] + values[1] }]}
      categoryCounts={{ symptom: 2, node: 0, user: 0, rule: 0 }}
      largestTies={tiesOf({ symptom: { count: 2, nodeHours: 1 } })}
    />)
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!)).toBe(summary)
  })

  it('speaks of failed attempts when Pass is chosen, and of one attempt as one', () => {
    const { rerender } = render(<ParetoCard
      filter={{ ...filter, outcome: 'pass' }}
      setFilter={vi.fn()}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 1, nodeHours: 2 }]}
      categoryCounts={{ symptom: 0, node: 0, user: 0, rule: 0 }}
      largestTies={tiesOf()}
    />)
    // A Pass run is no failure, so there is none to rank, and the card still speaks of failures.
    expect(within(screen.getByRole('region', { name: 'Failure concentration' })).getByRole('button', {
      name: 'About Failure concentration',
      description: 'Failed attempts ranked by root symptom, the ten largest; the line is their running share of all failed attempts.',
    })).toBeTruthy()
    rerender(<ParetoCard
      filter={{ ...filter, axis: 'node', outcome: 'multi' }}
      setFilter={vi.fn()}
      symptomLedger={[]}
      nodeLedger={[{ node: 'node-a', count: 1, nodeHours: 2, outcomes: { multi: 1 }, symptoms: { Mixed: 1 } }]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'multi', count: 1, nodeHours: 2 }]}
      categoryCounts={{ symptom: 1, node: 1, user: 1, rule: 1 }}
      largestTies={tiesOf()}
    />)
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('The only responsible node is node-a: responsible in 1 of 1 Ambiguous Fault attempt (100%)')
  })

  it('says whose failures its figures are of when an outcome is chosen', () => {
    render(<ParetoCard
      filter={{ ...filter, outcome: 'hardware' }}
      setFilter={vi.fn()}
      symptomLedger={[
        { symptom: 'Xid79', count: 2, nodeHours: 6, outcomes: { hardware: 2 } },
        { symptom: 'Xid48', count: 1, nodeHours: 2, outcomes: { hardware: 1 } },
        { symptom: 'Xid31', count: 1, nodeHours: 1, outcomes: { hardware: 1 } },
      ]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 4, nodeHours: 9 }, { outcome: 'software', count: 5, nodeHours: 7 }]}
      categoryCounts={{ symptom: 3, node: 0, user: 0, rule: 0 }}
      largestTies={tiesOf()}
    />)
    const card = screen.getByRole('region', { name: 'Failure concentration' })
    expect(within(card).getByRole('button', {
      name: 'About Failure concentration',
      description: 'Hardware Fault attempts ranked by root symptom, the ten largest; the line is their running share of all Hardware Fault attempts.',
    })).toBeTruthy()
    expect(visibleText(card.querySelector('[data-slot="card-footer"] p')!))
      .toBe('All 3 root symptoms account for 100% of Hardware Fault attempts · Largest: Xid79, 2 Hardware Fault attempts (50%)')
  })

  it('measures every share against all the failures in scope, not the ten it ranks', () => {
    // Twelve root symptoms, the server's ledger holding the ten largest: the
    // ten do not reach 80%, so the summary says what the ten do reach, of all
    // the window's failed attempts and of all twelve symptoms.
    const symptomLedger = Array.from({ length: 10 }, (_, index) => ({
      symptom: `s${String(index).padStart(2, '0')}`, count: 10 - Math.floor(index / 5), nodeHours: 1, outcomes: { hardware: 10 - Math.floor(index / 5) },
    }))
    render(<ParetoCard
      filter={filter}
      setFilter={vi.fn()}
      symptomLedger={symptomLedger}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 120, nodeHours: 12 }]}
      categoryCounts={{ symptom: 12, node: 0, user: 0, rule: 0 }}
      largestTies={tiesOf({ symptom: { count: 5, nodeHours: 12 } })}
    />)
    // The ten hold 5 × 10 + 5 × 9 = 95 of the 120 failed attempts.
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('Top 10 of 12 root symptoms account for 79% of failed attempts · Largest: s00 and 4 more, 10 failed attempts each (8% each)')
    const pareto = mocks.useECharts.mock.calls.filter(([, option]) => 'xAxis' in option).at(-1)![1]
    // The running share ends where the ten end, short of 100%.
    expect(pareto.series[1].data.at(-1)).toBeCloseTo(95 / 120 * 100, 9)
  })

  it('draws a running share by responsible node only in GPU-hours, where node shares add up', () => {
    // Two failed attempts blamed both nodes and a third blamed none: the nodes'
    // counts add up to four of three failed attempts.
    const { rerender } = render(<ParetoCard
      filter={{ ...filter, axis: 'node' }}
      setFilter={vi.fn()}
      symptomLedger={[]}
      nodeLedger={[
        { node: 'node-a', count: 2, nodeHours: 4, outcomes: { hardware: 2 }, symptoms: { Xid79: 2 } },
        { node: 'node-b', count: 2, nodeHours: 4, outcomes: { hardware: 2 }, symptoms: { Xid79: 2 } },
      ]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 3, nodeHours: 9 }]}
      categoryCounts={{ symptom: 1, node: 2, user: 1, rule: 1 }}
      largestTies={tiesOf({ node: { count: 2, nodeHours: 2 } })}
    />)
    const card = screen.getByRole('region', { name: 'Failure concentration' })
    expect(within(card).getByRole('button', {
      name: 'About Failure concentration',
      description: 'Failed attempts ranked by responsible node, the ten largest. An attempt can name several responsible nodes, or none, so their counts do not add up and no running share is drawn.',
    })).toBeTruthy()
    // Each node's share is of all failed attempts; no line, mark or share axis joins them.
    const pareto = mocks.useECharts.mock.calls.filter(([, option]) => 'xAxis' in option).at(-1)![1]
    expect(pareto.series).toHaveLength(1)
    expect(pareto.yAxis).toHaveLength(1)
    expect(pareto.tooltip.formatter([{ seriesIndex: 0, axisValue: 'node-a', data: { value: 2, dominant: 'hardware' } }]))
      .not.toContain('Running share')
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('Largest: node-a and 1 more, each responsible in 2 of 3 failed attempts (67%)')
    fireEvent.click(screen.getByRole('button', { name: 'Ranked list' }))
    const table = screen.getByRole('table', { name: 'Failures ranked by responsible node' })
    expect(within(table).getAllByRole('columnheader').map((header) => header.textContent))
      .toEqual(['Rank', 'Responsible node', 'Failed attempts', 'Share'])
    expect(within(table).getAllByRole('row').slice(1).map((row) => within(row).getAllByRole('cell').map((cell) => cell.textContent)))
      .toEqual([['1', 'node-a', '2', '67%'], ['2', 'node-b', '2', '67%']])

    // In GPU-hours each node has only its own share of an attempt's, so the
    // nodes' shares add up: the line, the vital few and the running column return.
    rerender(<ParetoCard
      filter={{ ...filter, axis: 'node', metric: 'nodeHours' }}
      setFilter={vi.fn()}
      symptomLedger={[]}
      nodeLedger={[
        { node: 'node-a', count: 2, nodeHours: 4, outcomes: { hardware: 2 }, symptoms: { Xid79: 2 } },
        { node: 'node-b', count: 2, nodeHours: 4, outcomes: { hardware: 2 }, symptoms: { Xid79: 2 } },
      ]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 3, nodeHours: 9 }]}
      categoryCounts={{ symptom: 1, node: 2, user: 1, rule: 1 }}
      largestTies={tiesOf({ node: { count: 2, nodeHours: 2 } })}
    />)
    expect(within(card).getByRole('button', {
      name: 'About Failure concentration',
      description: 'GPU-hours of failed attempts ranked by responsible node, each with an even share of the GPU-hours of every attempt that names it, the ten largest; the line is their running share of all GPU-hours of failed attempts.',
    })).toBeTruthy()
    expect(mocks.useECharts.mock.calls.filter(([, option]) => 'xAxis' in option).at(-1)![1].series).toHaveLength(2)
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('Both responsible nodes account for 89% of GPU-hours of failed attempts · Largest: node-a and 1 more, 4.0 GPU-hours each (44% each)')
    expect(within(table).getAllByRole('columnheader').map((header) => header.textContent))
      .toEqual(['Rank', 'Responsible node', 'GPU-hours', 'Share', 'Running share'])
  })

  it('lists large values grouped in thousands and gives the donut center in thousands or millions', () => {
    const card = (hardware: number, software: number) => (
      <ParetoCard
        filter={{ ...filter, metric: 'nodeHours' }}
        setFilter={vi.fn()}
        symptomLedger={[
          { symptom: 'Xid79', count: 3, nodeHours: hardware, outcomes: { hardware: 3 } },
          { symptom: 'Timeout', count: 1, nodeHours: software, outcomes: { software: 1 } },
        ]}
        nodeLedger={[]}
        userLedger={[]}
        ruleLedger={[]}
        outcomeLedger={[{ outcome: 'hardware', count: 3, nodeHours: hardware }, { outcome: 'software', count: 1, nodeHours: software }]}
        categoryCounts={{ symptom: 2, node: 0, user: 0, rule: 0 }}
        largestTies={tiesOf()}
      />
    )
    const donut = () => mocks.useECharts.mock.calls.filter(([, option]) => option.series?.[0]?.type === 'pie').at(-1)![1]
    const center = () => donut().title.text.split('\n')[0]
    const { rerender } = render(card(833_333.33, 300_000))

    // A seven-digit total: the center, in a hole six mono characters wide,
    // reads it in millions; the legend, the Ranked list, the summary and
    // both tooltips give each value exactly, grouped in thousands.
    expect(center()).toBe('{num|1.1M}')
    const outcomes = within(screen.getByRole('group', { name: 'Outcomes' }))
    expect(outcomes.getByRole('button', { name: 'Hardware Fault 833,333.3 74%' })).toBeTruthy()
    expect(outcomes.getByRole('button', { name: 'System Software Fault 300,000.0 26%' })).toBeTruthy()
    expect(visibleText(document.querySelector('[data-slot="card-footer"] p')!))
      .toBe('Both root symptoms account for 100% of GPU-hours of failed attempts · Largest: Xid79, 833,333.3 GPU-hours (74%)')
    fireEvent.click(screen.getByRole('button', { name: 'Ranked list' }))
    const ranked = within(screen.getByRole('table', { name: 'Failures ranked by root symptom' }))
    expect(ranked.getAllByRole('row').slice(1).map((row) => within(row).getAllByRole('cell')[2].textContent))
      .toEqual(['833,333.3', '300,000.0'])
    expect(donut().tooltip.formatter({ name: 'Hardware Fault', value: 833_333.33, data: { __key: 'hardware' } }))
      .toContain('GPU-hours: <span class="num text-foreground">833,333.3</span>')
    const pareto = mocks.useECharts.mock.calls.filter(([, option]) => 'xAxis' in option).at(-1)![1]
    expect(pareto.tooltip.formatter([{ seriesIndex: 0, axisValue: 'Xid79', data: { value: 833_333.33, dominant: 'hardware', running: 833_333.33 } }]))
      .toContain('GPU-hours: <span class="num text-foreground">833,333.3</span>')

    // Below a thousand the center is exact; a unit is taken once the value
    // rounds to a thousand of the unit below, never as 1000.0 of that unit.
    for (const [hardware, text] of [[999.94, '999.9'], [999.96, '1.0k'], [113_333.3, '113.3k'], [999_949, '999.9k'], [999_950, '1.0M']] as const) {
      rerender(card(hardware, 0))
      expect(center()).toBe(`{num|${text}}`)
    }
  })

  it('keeps its header and choices while the dashboard loads, and calls nothing empty', () => {
    const setFilter = vi.fn()
    render(<ParetoCard
      filter={{ ...filter, outcome: 'hardware' }}
      setFilter={setFilter}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[]}
      loading
    />)

    const card = screen.getByRole('region', { name: 'Failure concentration' })
    expect(within(card).queryByText('No failures to rank')).toBeNull()
    expect(within(card).queryByRole('status')).toBeNull()
    expect(card.querySelector('[aria-busy="true"] [data-slot="skeleton"]')).not.toBeNull()
    expect(mocks.useECharts).not.toHaveBeenCalled()
    // The choices stay usable; their names are not form labels.
    expect(within(screen.getByRole('group', { name: 'Group by' })).getAllByRole('radio')).toHaveLength(4)
    expect(within(screen.getByRole('group', { name: 'Measure' })).getAllByRole('radio')).toHaveLength(2)
    expect(card.querySelector('label')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Clear selection' }))
    expect(setFilter).toHaveBeenLastCalledWith(filter)
    cleanup()

    // Another outcome of the same window: the outcome split is the window's, so
    // the donut and its legend (the control just pressed) stay; only the bars,
    // the summary and the ranked list wait for the new ledgers.
    render(<ParetoCard
      filter={{ ...filter, outcome: 'hardware' }}
      setFilter={setFilter}
      symptomLedger={[]}
      nodeLedger={[]}
      userLedger={[]}
      ruleLedger={[]}
      outcomeLedger={[{ outcome: 'hardware', count: 3, nodeHours: 9 }, { outcome: 'software', count: 1, nodeHours: 2 }]}
      ledgersLoading
    />)
    const outcomes = within(screen.getByRole('group', { name: 'Outcomes' }))
    expect(outcomes.getByRole('button', { name: 'Hardware Fault 3 75%' }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByRole('region', { name: 'Failure concentration' }).querySelector('[aria-busy="true"] [data-slot="skeleton"]')).not.toBeNull()
    expect(screen.queryByText('No failures to rank')).toBeNull()
    expect(screen.queryByText(/for this filter/)).toBeNull()
    expect(screen.queryByText(/Largest:/)).toBeNull()
    expect(screen.queryByRole('button', { name: 'Ranked list' })).toBeNull()
    expect(mocks.useECharts.mock.calls.every(([, option]) => !('xAxis' in option))).toBe(true)
  })

  it('says what could not be loaded instead of ranking nothing', () => {
    const setFilter = vi.fn()
    const ledgers = { symptomLedger: [], nodeLedger: [], userLedger: [], ruleLedger: [] }
    const { rerender } = render(<ParetoCard filter={filter} setFilter={setFilter} {...ledgers} outcomeLedger={[]} failed ledgersFailed />)

    // The window's dashboard failed: one empty state names the failure and the
    // way out, as the run list's does; the header and its choices stay.
    const card = screen.getByRole('region', { name: 'Failure concentration' })
    const empty = card.querySelector<HTMLElement>('[data-slot="empty"]')!
    expect(within(empty).getByText('Failure concentration could not be loaded').getAttribute('data-slot')).toBe('empty-title')
    expect(within(empty).getByText('Refresh to try again.').getAttribute('data-slot')).toBe('empty-description')
    expect(within(card).queryByText(/No failures|No failed attempts|for this filter/)).toBeNull()
    expect(within(card).queryByRole('status')).toBeNull()
    expect(within(screen.getByRole('group', { name: 'Group by' })).getAllByRole('radio')).toHaveLength(4)
    expect(mocks.useECharts).not.toHaveBeenCalled()

    // Only this outcome's ledgers failed: the window's outcome split stays in
    // the donut and its legend, the bars' pane says what could not be loaded,
    // and no summary or ranked list is offered.
    rerender(<ParetoCard
      filter={{ ...filter, outcome: 'hardware' }}
      setFilter={setFilter}
      {...ledgers}
      outcomeLedger={[{ outcome: 'hardware', count: 3, nodeHours: 9 }, { outcome: 'software', count: 1, nodeHours: 2 }]}
      ledgersFailed
    />)
    expect(screen.getByText('Failures by root symptom could not be loaded. Refresh to try again.').tagName).toBe('P')
    expect(within(screen.getByRole('group', { name: 'Outcomes' })).getByRole('button', { name: 'Hardware Fault 3 75%' }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.queryByText(/for this filter|No failures/)).toBeNull()
    expect(screen.queryByText(/Largest:/)).toBeNull()
    expect(mocks.useECharts.mock.calls.every(([, option]) => !('xAxis' in option))).toBe(true)
  })

  it('offers the bars as a ranked list, the keyboard path to the same filter', () => {
    const setFilter = vi.fn()
    const ledger = [
      { symptom: 'Xid79', count: 3, nodeHours: 10, outcomes: { hardware: 3 } },
      { symptom: 'Timeout', count: 1, nodeHours: 3, outcomes: { software: 1 } },
    ]
    const card = (selected?: string) => (
      <ParetoCard
        filter={{ ...filter, symptom: selected }}
        setFilter={setFilter}
        symptomLedger={ledger}
        nodeLedger={[]}
        userLedger={[]}
        ruleLedger={[]}
        outcomeLedger={[{ outcome: 'hardware', count: 3, nodeHours: 10 }, { outcome: 'software', count: 1, nodeHours: 3 }]}
        categoryCounts={{ symptom: 2, node: 0, user: 0, rule: 0 }}
        largestTies={tiesOf()}
      />
    )
    const { rerender } = render(card())

    const trigger = screen.getByRole('button', { name: 'Ranked list' })
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByRole('table')).toBeNull()
    fireEvent.click(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')

    // Same order and numbers as the bars: rank, category, value, share, running share.
    const table = screen.getByRole('table', { name: 'Failures ranked by root symptom' })
    expect(within(table).getAllByRole('columnheader').map((header) => header.textContent))
      .toEqual(['Rank', 'Root symptom', 'Failed attempts', 'Share', 'Running share'])
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map((row) => within(row).getAllByRole('cell').map((cell) => cell.textContent)))
      .toEqual([['1', 'Xid79', '3', '75%', '75%'], ['2', 'Timeout', '1', '25%', '100%']])

    // Each category is a focusable toggle bound to the bar's filter.
    const xid79 = within(table).getByRole('button', { name: 'Xid79' })
    xid79.focus()
    expect(document.activeElement).toBe(xid79)
    expect(xid79.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(xid79)
    expect(setFilter).toHaveBeenLastCalledWith(expect.objectContaining({ symptom: 'Xid79' }))

    rerender(card('Xid79'))
    const pressed = within(screen.getByRole('table')).getByRole('button', { name: 'Xid79' })
    expect(pressed.getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(pressed)
    expect(setFilter.mock.lastCall![0]).not.toHaveProperty('symptom')
  })
})
