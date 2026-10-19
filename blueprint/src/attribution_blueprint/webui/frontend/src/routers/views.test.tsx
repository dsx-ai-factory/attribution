// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { RunDetail } from '@/interfaces/attribution'
import { NodeView } from '@/routers/node'
import { VerdictView } from '@/routers/verdict'

const mocks = vi.hoisted(() => ({
  useParams: vi.fn(),
  useNodeRuns: vi.fn(),
  useSWR: vi.fn(),
  useApi: vi.fn(),
  nav: vi.fn(),
  downloadBlob: vi.fn(),
  clipboardWrite: vi.fn(),
  symptomDocumentation: vi.fn(),
}))

vi.mock('react-router-dom', async (importOriginal) => ({
  ...await importOriginal<typeof import('react-router-dom')>(),
  useParams: mocks.useParams,
}))
vi.mock('swr', () => ({ default: mocks.useSWR }))
vi.mock('@/hooks/use-node-runs', () => ({ useNodeRuns: mocks.useNodeRuns }))
vi.mock('@/hooks/use-symptom-documentation', () => ({ useSymptomDocumentation: mocks.symptomDocumentation }))
vi.mock('@/hooks/use-nav', () => ({ useNav: () => mocks.nav }))
vi.mock('@/services/api-context', () => ({ useApi: mocks.useApi }))
vi.mock('@/lib/export', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/lib/export')>(),
  downloadBlob: mocks.downloadBlob,
  utcDateStamp: () => '20260501-0102',
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
vi.mock('@/components/node-hover-card', () => ({
  NodeHoverCard: ({ children }: { children: ReactNode }) => <>{children}</>,
}))
vi.mock('@/components/time-hover', () => ({
  // Marks the table cells' times, whose invalid value is the cells' dash (TimeHover's own test).
  TimeHover: ({ date, cell }: { date: string; cell?: boolean }) => <time dateTime={date} data-cell={cell || undefined}>{date}</time>,
}))

const run = (overrides: Partial<RunDetail> = {}): RunDetail => ({
  id: 'run-1',
  jobId: 'team/job',
  workload: 'team/job',
  attempt: 2,
  attemptUid: 'attempt-2',
  user: 'alice',
  status: 'FAILED',
  nodes: 2,
  nodeIds: ['node-a', 'node-b'],
  elapsedH: 1.25,
  gpusPerNode: 8,
  gpuCount: 12,
  endTime: '2026-05-01T01:15:00Z',
  startTime: '2026-05-01T00:00:00Z',
  outcome: 'hardware',
  verdictLabel: 'Hardware failure',
  detail: 'GPU fell off the bus.',
  decisionGraphState: 'absent',
  rootSymptom: {
    symptom: 'Xid79',
    rule: 'xid',
    time: '2026-05-01T00:30:00Z',
    line: 'Xid 79',
  },
  symptoms: [{ name: 'Xid79', sites: 1 }],
  responsibleNodes: ['node-a'],
  pipeline: '1',
  service: '2',
  log: [{ t: '2026-05-01T00:30:00Z', level: 'symptom', text: 'Xid 79' }],
  diagnose: {
    hasPersona: true,
    title: 'GPU unavailable',
    summary: 'The GPU stopped responding.',
    impact: 'The attempt failed.',
    action: 'Retry elsewhere.',
    supportDetails: ['run ID'],
    catalogId: 'xid-79',
  },
  operatorDiagnose: {
    hasPersona: true,
    title: 'Inspect GPU',
    summary: 'The GPU is unavailable.',
    impact: 'Node health is degraded.',
    action: 'Drain the node.',
    supportDetails: ['run ID', 'node ID'],
    catalogId: 'xid-79',
  },
  attribution: {
    rootSymptom: { state: 'absent', data: null },
    symptomSites: { state: 'absent', data: null },
    decisionGraph: { state: 'absent', data: null },
  },
  ...overrides,
})

beforeEach(() => {
  vi.clearAllMocks()
  vi.useFakeTimers()
  mocks.symptomDocumentation.mockReturnValue({ data: undefined })
  mocks.useParams.mockReturnValue({ id: 'node-a' })
  mocks.useNodeRuns.mockReturnValue({ runs: [run()], isLoading: false, error: null })
  mocks.useSWR.mockReturnValue({ data: run(), isLoading: false, error: undefined })
  mocks.useApi.mockReturnValue({ getRun: vi.fn().mockResolvedValue(run()) })
  mocks.clipboardWrite.mockResolvedValue(undefined)
  // The matched lines watch their own size; jsdom has no layout to observe.
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText: mocks.clipboardWrite },
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('NodeView', () => {
  const rootSymptom = (symptom: string) => ({ symptom, rule: 'rule', time: '2026-05-01T00:30:00Z', line: symptom })
  // The value of a page-header fact, found by its term.
  const metaValue = (term: string) => screen.getByText(term, { selector: 'dt' }).nextElementSibling?.textContent
  // A section card's count sits right after its heading.
  const sectionCount = (title: string) => screen.getByRole('heading', { level: 2, name: title }).nextElementSibling?.textContent

  it('renders node evidence, copies actions, exports formats, and navigates', () => {
    mocks.useNodeRuns.mockReturnValue({
      runs: [
        run(),
        run({ id: 'run-2', jobId: 'team/other', outcome: 'software', responsibleNodes: ['node-b'], rootSymptom: rootSymptom('NcclTimeout') }),
      ],
      isLoading: false,
      error: null,
    })
    render(<MemoryRouter><NodeView /></MemoryRouter>)

    const title = screen.getByRole('heading', { level: 1, name: 'node-a' })
    expect(document.title).toBe('node-a · Node · Attribution')
    expect(screen.getByRole('link', { name: 'Fleet' }).getAttribute('href')).toBe('/')
    // Blame is red, beside the node ID.
    expect(within(title.parentElement!).getByText('Responsible').getAttribute('data-variant')).toBe('red-subtle')
    expect(metaValue('Attempts')).toBe('2')
    expect(metaValue('Responsible in')).toBe('1')
    expect(metaValue('Distinct symptoms')).toBe('1')
    expect(metaValue('GPU-hours on node')).toBe('15.00')

    // An informational notice: no alert role, no actions, the icon in the blame color.
    const alert = screen.getByText('Responsible node').closest<HTMLElement>('[data-slot="alert"]')!
    expect(alert.getAttribute('role')).toBeNull()
    expect(within(alert).queryByRole('button')).toBeNull()
    expect(alert.firstElementChild?.tagName.toLowerCase()).toBe('svg')
    expect(alert.className).toContain('*:[svg]:text-destructive-400')
    expect(alert.querySelector('[data-slot="alert-description"]')?.textContent)
      .toBe('Xid79 attributed to this node. Consider escalating to the cloud operator with the evidence.')

    const symptoms = within(screen.getByRole('region', { name: 'Symptoms on this node' }))
    expect(sectionCount('Symptoms on this node')).toBe('1')
    expect(symptoms.getAllByRole('row').slice(1).map((row) => row.textContent)).toEqual(['Xid791'])

    const attempts = within(screen.getByRole('region', { name: 'Attempts' }))
    expect(sectionCount('Attempts')).toBe('2')
    const headers = ['End time', 'Execution context', 'User', 'Outcome', 'Responsible', 'GPU-hours on node', 'Root symptom']
    expect(attempts.getAllByRole('columnheader')).toHaveLength(headers.length)
    headers.forEach((name, index) => {
      expect(attempts.getByRole<HTMLTableCellElement>('columnheader', { name }).cellIndex).toBe(index)
    })
    const [, blamed, cleared] = attempts.getAllByRole('row')
    expect(within(blamed).getByRole('link', { name: 'team/job' }).getAttribute('href')).toBe('/verdict/run-1')
    // As on Fleet, the execution context wraps in its column as an inline
    // link, and the user and the root symptom as text.
    expect(within(blamed).getByRole('link', { name: 'team/job' }).getAttribute('data-slot')).toBe('link')
    expect([...blamed.children].map((cell) => cell.classList.contains('whitespace-normal'))).toEqual([false, true, true, false, false, false, true])
    expect(within(blamed).getByText('Hardware Fault').getAttribute('data-outcome')).toBe('hardware')
    expect(within(blamed).getByText('Responsible').getAttribute('data-variant')).toBe('red-subtle')
    expect(within(blamed).getByText('7.50')).toBeTruthy()
    expect(within(blamed).getByText('Xid79')).toBeTruthy()
    expect(within(cleared).getByText('System Software Fault').getAttribute('data-outcome')).toBe('software')
    expect(within(cleared).queryByText('Responsible')).toBeNull()
    expect(within(cleared).getByText('None')).toBeTruthy()
    expect(within(cleared).getByText('NcclTimeout')).toBeTruthy()

    // Copying the evidence is an outline action, as it is on the verdict: the
    // same action never changes weight from page to page, so this page has no
    // green primary.
    expect(screen.getByRole('button', { name: 'Copy evidence' }).getAttribute('data-variant')).toBe('outline')
    expect(document.querySelectorAll('[data-variant="default"]')).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: 'Copy node ID' }))
    fireEvent.click(screen.getByRole('button', { name: 'Copy evidence' }))
    expect(mocks.clipboardWrite.mock.calls).toEqual([
      ['node-a'],
      [expect.stringContaining('*Attribution node escalation: `node-a`*')],
    ])

    expect(screen.getByText('Export')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Download as JSON' }))
    fireEvent.click(screen.getByRole('button', { name: 'Download as YAML' }))
    fireEvent.click(screen.getByRole('button', { name: 'Download as Markdown' }))
    expect(mocks.downloadBlob.mock.calls.map(([name, , type]) => [name, type])).toEqual([
      ['node-node-a-20260501-0102.json', 'application/json'],
      ['node-node-a-20260501-0102.yaml', 'application/yaml'],
      ['node-node-a-20260501-0102.md', 'text/markdown'],
    ])

    // The execution context link navigates by itself; a click elsewhere in the row opens the verdict.
    fireEvent.click(within(blamed).getByRole('link', { name: 'team/job' }))
    expect(mocks.nav).not.toHaveBeenCalled()
    fireEvent.click(within(blamed).getByText('alice'))
    expect(mocks.nav).toHaveBeenCalledWith('verdict', { id: 'run-1' })
  })

  it('names the most attributed recorded symptom in the notice, and only one that stands out', () => {
    mocks.useNodeRuns.mockReturnValue({
      runs: [
        run({ id: 'run-1', rootSymptom: rootSymptom('Xid48') }),
        run({ id: 'run-2', rootSymptom: rootSymptom('Xid79') }),
        run({ id: 'run-3', rootSymptom: rootSymptom('Xid79') }),
      ],
      isLoading: false,
      error: null,
    })
    const { rerender } = render(<MemoryRouter><NodeView /></MemoryRouter>)
    const description = () => document.querySelector('[data-slot="alert-description"]')?.textContent
    expect(description()).toBe('Recurring Xid79 attributed to this node. Consider escalating to the cloud operator with the evidence.')
    const symptoms = within(screen.getByRole('region', { name: 'Symptoms on this node' }))
    expect(symptoms.getAllByRole('row').slice(1).map((row) => row.textContent)).toEqual(['Xid792', 'Xid481'])

    // With no symptom attributed more often than the others, none is named as the node's: they are counted.
    mocks.useNodeRuns.mockReturnValue({
      runs: [
        run({ id: 'run-1', rootSymptom: rootSymptom('Xid48') }),
        run({ id: 'run-2', rootSymptom: rootSymptom('Xid79') }),
        run({ id: 'run-3', rootSymptom: rootSymptom('NcclTimeout') }),
      ],
      isLoading: false,
      error: null,
    })
    rerender(<MemoryRouter><NodeView /></MemoryRouter>)
    expect(description()).toBe('3 symptoms attributed to this node. Consider escalating to the cloud operator with the evidence.')
    // Nor is the one recorded symptom named when the unrecorded one is attributed as often.
    mocks.useNodeRuns.mockReturnValue({
      runs: [run({ id: 'run-1', rootSymptom: rootSymptom('') }), run({ id: 'run-2', rootSymptom: rootSymptom('Xid79') })],
      isLoading: false,
      error: null,
    })
    rerender(<MemoryRouter><NodeView /></MemoryRouter>)
    expect(description()).toBe('2 symptoms attributed to this node. Consider escalating to the cloud operator with the evidence.')

    // An unrecorded root symptom reads as none and is never named.
    mocks.useNodeRuns.mockReturnValue({ runs: [run({ rootSymptom: rootSymptom('') })], isLoading: false, error: null })
    rerender(<MemoryRouter><NodeView /></MemoryRouter>)
    expect(description()).toBe('Consider escalating to the cloud operator with the evidence.')
    expect(within(screen.getByRole('region', { name: 'Symptoms on this node' })).getByText('None')).toBeTruthy()
    expect(within(screen.getByRole('region', { name: 'Attempts' })).getAllByText('None')).toHaveLength(1)

    // An unrecorded user is the same muted, "None"-read dash as every empty table value.
    mocks.useNodeRuns.mockReturnValue({ runs: [run({ user: '' })], isLoading: false, error: null })
    rerender(<MemoryRouter><NodeView /></MemoryRouter>)
    const attempt = within(screen.getByRole('region', { name: 'Attempts' })).getAllByRole('row')[1]
    expect(within(attempt).getAllByRole('cell')[2].textContent).toBe('—None')

    // The end time is a table cell's, so one that does not parse is the same
    // dash, read as "Unavailable"; a recorded identifier may break after its
    // separators before anywhere else.
    mocks.useNodeRuns.mockReturnValue({
      runs: [run({ rootSymptom: rootSymptom('XID-171-UNCORRECTABLE_DRAM_ERROR') })],
      isLoading: false,
      error: null,
    })
    rerender(<MemoryRouter><NodeView /></MemoryRouter>)
    const cells = within(within(screen.getByRole('region', { name: 'Attempts' })).getAllByRole('row')[1]).getAllByRole('cell')
    expect(cells[0].querySelector('time')!.getAttribute('data-cell')).toBe('true')
    const BREAK = '<span aria-hidden="true" class="select-none">\u200b</span>'
    expect(cells[1].querySelector('a')!.innerHTML).toBe(`<span>team/${BREAK}job</span>`)
    expect(cells[6].innerHTML).toBe(`<span>XID-171-UNCORRECTABLE_${BREAK}DRAM_${BREAK}ERROR</span>`)
  })

  it('states nothing about the node until its runs arrive', () => {
    mocks.useNodeRuns.mockReturnValue({ runs: [], isLoading: true, error: null })
    const { rerender } = render(<MemoryRouter><NodeView /></MemoryRouter>)
    // The route names the node, so its tab does too while the runs load.
    expect(document.title).toBe('node-a · Node · Attribution')
    expect(screen.getAllByRole('table').map((table) => table.getAttribute('aria-busy'))).toEqual(['true', 'true'])
    // Only the figures wait: the head keeps its facts' labels, with no value under any, and holds
    // the places of its two actions, so it keeps its height when the runs arrive.
    expect(document.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(2 + 4 + 3 * 2 + 3 * 7)
    const facts = ['Attempts', 'Responsible in', 'Distinct symptoms', 'GPU-hours on node']
    expect(facts.map(metaValue)).toEqual(['', '', '', ''])
    expect(screen.getByText('Attempts', { selector: 'dt' }).closest('dl')!.getAttribute('aria-busy')).toBe('true')
    expect(sectionCount('Attempts')).toBeUndefined()
    expect(screen.queryByRole('button', { name: 'Copy evidence' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Copy node ID' })).toBeTruthy()

    mocks.useNodeRuns.mockReturnValue({ runs: [], isLoading: false, error: new Error('unreachable') })
    rerender(<MemoryRouter><NodeView /></MemoryRouter>)
    expect(screen.getAllByText('Runs for this node could not be loaded')).toHaveLength(2)
    expect(screen.queryByText('Attempts', { selector: 'dt' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Copy evidence' })).toBeNull()
    expect(document.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(0)

    mocks.useNodeRuns.mockReturnValue({ runs: [], isLoading: false, error: null })
    rerender(<MemoryRouter><NodeView /></MemoryRouter>)
    expect(screen.getAllByRole('table').map((table) => table.getAttribute('aria-busy'))).toEqual([null, null])
    expect(within(screen.getByRole('region', { name: 'Symptoms on this node' })).getByText('No attributed symptoms')).toBeTruthy()
    expect(within(screen.getByRole('region', { name: 'Attempts' })).getByText('No recorded attempts')).toBeTruthy()
    expect(metaValue('Attempts')).toBe('0')
    expect(screen.getByText('Attempts', { selector: 'dt' }).closest('dl')!.getAttribute('aria-busy')).toBeNull()
    expect(sectionCount('Attempts')).toBe('0')
    expect(screen.queryByText('Responsible node')).toBeNull()
    expect(screen.getByRole('button', { name: 'Copy evidence' })).toBeTruthy()
  })
})

describe('VerdictView', () => {
  it('places the API document in the symptom table without a duplicate on the verdict root', () => {
    mocks.useParams.mockReturnValue({ id: 'run-1' })
    const detail = run()
    detail.attribution.rootSymptom = { state: 'available', data: {
      symptom: 'Xid79', catalog_id: 'xid-79',
      source: { observation_id: '0', source: 'syslog', pos: 0, line: null, rule: null, time: null },
    } }
    detail.attribution.symptomSites = { state: 'available', data: {
      Xid79: [{ type: 'NodeSource', node: 'node-a', status: 'bad', source: detail.attribution.rootSymptom.data.source }],
    } }
    mocks.useSWR.mockReturnValue({ data: detail })
    mocks.symptomDocumentation.mockReturnValue({ data: { symptoms: {
      'xid-79': { id: 'xid-79', title: 'XID 79', doc_url: 'https://docs.example/xid-79/' },
      Xid79: { id: 'xid-79', title: 'XID 79', doc_url: 'https://docs.example/xid-79/' },
    } } })
    const { rerender } = render(<MemoryRouter><VerdictView /></MemoryRouter>)
    const docs = screen.getAllByRole('link', { name: /Symptom documentation/ })
    expect(docs).toHaveLength(1)
    expect(docs[0].getAttribute('href')).toBe('https://docs.example/xid-79/')
    expect(within(screen.getByRole('table', { name: 'Symptoms' })).getByRole('link', { name: /Symptom documentation/ })).toBe(docs[0])

    detail.attribution.rootSymptom.data.catalog_id = null
    rerender(<MemoryRouter><VerdictView /></MemoryRouter>)
    expect(screen.getByRole('link', { name: /Symptom documentation/ }).getAttribute('href')).toBe('https://docs.example/xid-79/')

    detail.attribution.rootSymptom.data.catalog_id = 'xid-79'
    mocks.symptomDocumentation.mockReturnValue({ data: { symptoms: {
      'xid-79': { id: 'xid-79', title: 'XID 79', doc_url: 'https://docs.example/xid-79/' },
    } } })
    rerender(<MemoryRouter><VerdictView /></MemoryRouter>)
    expect(screen.queryByRole('link', { name: /Symptom documentation/ })).toBeNull()
  })

  it('lists each recorded node once without conflating responsibility with attempt membership', () => {
    mocks.useSWR.mockReturnValue({
      data: run({ nodeIds: ['group/node one', 'shared-node', 'shared-node'], responsibleNodes: ['shared-node', 'attributed-node', 'attributed-node'], gpuCount: 96 }),
      isLoading: false,
      error: undefined,
    })
    render(<MemoryRouter><VerdictView /></MemoryRouter>)

    const nodes = within(screen.getByRole('region', { name: 'Nodes' }))
    expect(nodes.getByRole('link', { name: 'group/node one' }).getAttribute('href')).toBe('/node/group%2Fnode%20one')
    expect(nodes.getAllByRole('link')).toHaveLength(3)
    expect(nodes.getByRole('link', { name: 'shared-node Responsible' }).getAttribute('href')).toBe('/node/shared-node')
    expect(nodes.getByRole('link', { name: 'attributed-node Not listed in attempt nodes Responsible' }).getAttribute('href')).toBe('/node/attributed-node')
    // Responsibility is blame, so it is the red badge and never the brand green.
    expect(nodes.getAllByText('Responsible').map((badge) => badge.getAttribute('data-variant'))).toEqual(['red-subtle', 'red-subtle'])
    // The icon and the badge are marks in the name's running text: centered on
    // its lowercase (its x-height), not on the line box, whose middle that
    // text sits below, and keeping out of the line's height.
    const row = nodes.getByRole('link', { name: 'shared-node Responsible' }).closest('li')!
    for (const mark of [row.querySelector('.lucide-server')!, within(row).getByText('Responsible')]) {
      expect([...mark.classList].filter((name) => /^(align-|-?my-)/.test(name))).toEqual(['align-middle', '-my-1'])
    }
    // The row is the node's link; its hover underline marks the name alone.
    const link = nodes.getByRole('link', { name: 'attributed-node Not listed in attempt nodes Responsible' })
    expect(link.classList.contains('hover:underline')).toBe(false)
    expect(within(link).getByText('attributed-node').closest('span')!.classList.contains('group-hover/link:underline')).toBe(true)
    expect(nodes.getByRole('paragraph').textContent).toBe('2 in attempt · 2 responsible')
    // The summary is the section's count, beside its title as Symptoms' is, not a second line under it.
    expect(nodes.getByRole('heading', { level: 2, name: 'Nodes' }).nextElementSibling).toBe(nodes.getByRole('paragraph'))
    // The GPU count is said once, in the page facts, not again beside the nodes.
    expect(nodes.queryByText(/GPUs/)).toBeNull()
    expect(screen.getByText('GPUs', { selector: 'dt' }).nextElementSibling?.textContent).toBe('96')
  })

  it('shows a Nodes section with no recorded node as one empty state', () => {
    mocks.useSWR.mockReturnValue({ data: run({ nodeIds: [], responsibleNodes: [] }), isLoading: false, error: undefined })
    render(<MemoryRouter><VerdictView /></MemoryRouter>)

    const nodes = screen.getByRole('region', { name: 'Nodes' })
    expect(within(nodes).queryByRole('list')).toBeNull()
    const empty = nodes.querySelector<HTMLElement>('[data-slot="empty"]')!
    expect(empty.getAttribute('data-size')).toBe('sm')
    expect(within(empty).getByText('No recorded nodes').getAttribute('data-slot')).toBe('empty-title')
    expect(within(empty).getByText('No nodes recorded for this attempt.').getAttribute('data-slot')).toBe('empty-description')
  })

  it('shows the evidence status row when recorded', () => {
    mocks.useSWR.mockReturnValue({ data: run({ evidenceStatus: 'partial' }), isLoading: false, error: undefined })
    render(<MemoryRouter><VerdictView /></MemoryRouter>)

    const verdict = within(screen.getByRole('region', { name: 'Verdict' }))
    expect(verdict.getByText('Evidence').closest('dt')?.nextElementSibling?.textContent)
      .toBe('Partial')
  })

  it('renders loading, missing and failed states', () => {
    // Every state keeps the way back to Fleet where the loaded page has it.
    const expectWayBack = () => {
      const breadcrumb = within(screen.getByRole('navigation', { name: 'breadcrumb' }))
      expect(breadcrumb.getByRole('link', { name: 'Fleet' }).getAttribute('href')).toBe('/')
      expect(breadcrumb.getByText('Verdict').getAttribute('aria-current')).toBe('page')
    }
    mocks.useParams.mockReturnValue({ id: 'run-1' })
    mocks.useSWR.mockReturnValue({ data: undefined, isLoading: true, error: undefined })
    const { rerender } = render(<MemoryRouter><VerdictView /></MemoryRouter>)
    // Loading holds the page's shape with skeletons, as node and fleet do: the
    // head, the answer card and the matched lines in their row, no title yet,
    // and the loading named once.
    const loading = screen.getByRole('status')
    expect(loading.textContent).toBe('Loading execution context…')
    expect(loading.classList.contains('sr-only')).toBe(true)
    expect(document.querySelector('[data-slot="empty"]')).toBeNull()
    expect(screen.queryByRole('heading')).toBeNull()
    // The head is the loaded page's own PageHeader under the one loading rule:
    // a placeholder title and actions, and the facts' labels over placeholder values.
    const head = document.querySelector('header')!
    const strip = head.querySelector('dl')!
    expect(strip.getAttribute('aria-busy')).toBe('true')
    expect(within(strip).getAllByRole('term')).toHaveLength(5)
    for (const label of ['User', 'Status', 'GPUs', 'GPU-hours', 'Window']) {
      expect(within(strip).getByText(label, { selector: 'dt' }).nextElementSibling?.textContent).toBe('')
    }
    expect(strip.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(5)
    expect(head.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(5 + 1 + 3)
    const [answer, lines] = document.querySelectorAll<HTMLElement>('[data-slot="card"]')
    expect(answer.parentElement).toBe(lines.parentElement)
    expect(answer.parentElement!.className).toContain('lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(min-content,1fr)]')
    expect(answer.querySelector('[data-slot="card-header"] [data-slot="card-action"] [data-slot="skeleton"]')).not.toBeNull()
    expect(answer.querySelectorAll('[data-slot="card-content"] [data-slot="skeleton"]')).toHaveLength(4)
    expect(lines.className).toContain('lg:order-first')
    expect(lines.className).toContain('lg:col-span-2')
    expect(lines.querySelector('[data-slot="card-content"] [data-slot="skeleton"]')!.className).toContain('min-h-60')
    expectWayBack()
    // Without a record there is no execution context to name the tab by.
    expect(document.title).toBe('Verdict · Attribution')

    mocks.useSWR.mockReturnValue({ data: undefined, isLoading: false, error: undefined })
    rerender(<MemoryRouter><VerdictView /></MemoryRouter>)
    expect(screen.getByText('Execution context not found')).toBeTruthy()
    expect(document.querySelector('[data-slot="skeleton"]')).toBeNull()
    expectWayBack()

    // A failed request is not a missing record: it says so and can be retried.
    const mutate = vi.fn()
    mocks.useSWR.mockReturnValue({ data: undefined, isLoading: false, error: new Error('503'), mutate })
    rerender(<MemoryRouter><VerdictView /></MemoryRouter>)
    expect(screen.getByRole('alert').textContent).toBe('Execution context could not be loaded')
    expect(screen.queryByText('Execution context not found')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(mutate).toHaveBeenCalledTimes(1)
    expectWayBack()
    expect(document.title).toBe('Verdict · Attribution')
  })

  it.each(['absent', 'null'] as const)('renders verdict evidence and actions with a %s graph', async (state) => {
    mocks.useParams.mockReturnValue({ id: 'run-1' })
    const detail = run()
    detail.attribution.decisionGraph = { state, data: null }
    detail.detail = 'XID 79 GPU_FELL_OFF_BUS: GPU fell off the bus.'
    mocks.useSWR.mockReturnValue({ data: detail, isLoading: false, error: undefined })
    render(<MemoryRouter><VerdictView /></MemoryRouter>)

    // The page is named by its execution context; the hard-coded attempt number is not shown.
    expect(screen.getByRole('heading', { level: 1 })).toBe(screen.getByRole('heading', { name: 'team/job' }))
    expect(document.title).toBe('team/job · Verdict · Attribution')
    expect(screen.queryByText(/attempt 2/i)).toBeNull()
    expect(screen.getByRole('link', { name: 'Fleet' }).getAttribute('href')).toBe('/')
    const meta = (label: string) => screen.getByText(label, { selector: 'dt' }).nextElementSibling
    expect(meta('User')?.textContent).toBe('alice')
    expect(meta('Status')?.textContent).toBe('Failed')
    expect(meta('GPUs')?.textContent).toBe('12')
    expect(meta('GPU-hours')?.textContent).toBe('15.00')
    expect([...meta('Window')!.querySelectorAll('time')].map((time) => time.getAttribute('datetime'))).toEqual([detail.startTime, detail.endTime])

    // The answer is said once: the outcome titles the verdict card and the root symptom opens its body.
    const verdict = within(screen.getByRole('region', { name: 'Verdict' }))
    const title = verdict.getByRole('heading', { level: 2, name: 'Verdict: Hardware failure' })
    expect(title.closest('[data-slot="card-header"]')).toBeTruthy()
    expect(screen.getAllByText('Hardware failure')).toHaveLength(1)
    expect(screen.getByText('Hardware failure').getAttribute('data-outcome')).toBe('hardware')
    // The badge is the title itself, beside no text, so it takes no mark offset.
    expect([...screen.getByText('Hardware failure').classList].filter((name) => /^-?top-/.test(name))).toEqual(['top-0'])
    const rootSymptom = verdict.getByText('Xid79', { selector: 'p' })
    expect(rootSymptom.textContent).toBe('Root symptom: Xid79')
    expect(rootSymptom.className).toContain('font-medium')
    expect(rootSymptom.parentElement!.firstElementChild).toBe(rootSymptom)
    // The attribution logic is the answer card's own action and the page's one
    // primary action: the green default button, unavailable without a graph.
    const graphAction = verdict.getByRole('button', { name: 'View attribution logic' })
    expect(graphAction.closest('[data-slot="card-action"]')).toBeTruthy()
    expect([graphAction.getAttribute('data-variant'), graphAction.getAttribute('data-size')]).toEqual(['default', 'default'])
    expect([...document.querySelectorAll('[data-variant="default"]')]).toEqual([graphAction])
    expect(graphAction.getAttribute('aria-disabled')).toBe('true')
    // The service detail is a sentence, not a recorded identifier: it wraps
    // between its words, with no break of its own inside a token.
    expect(verdict.getByText('Service detail', { selector: 'dt' }).nextElementSibling?.innerHTML).toBe('XID 79 GPU_FELL_OFF_BUS: GPU fell off the bus.')
    expect(verdict.getByText('Rule', { selector: 'dt' }).nextElementSibling?.textContent).toBe('xid')
    expect(verdict.getByText('Root symptom time', { selector: 'dt' }).nextElementSibling?.querySelector('time')?.getAttribute('datetime')).toBe(detail.rootSymptom.time)
    // Key-value times are no table cells: an invalid one keeps the word.
    expect(screen.getByRole('region', { name: 'Verdict' }).querySelector('time[data-cell]')).toBeNull()
    expect(verdict.getByText('Pipeline', { selector: 'dt' }).nextElementSibling?.textContent).toBe('1')
    expect(verdict.getByText('Service', { selector: 'dt' }).nextElementSibling?.textContent).toBe('2')
    expect(verdict.getByRole('button', { name: 'Copy attempt UID' })).toBeTruthy()
    // The facts are one column on one label axis: a single list, one pair per row, in reading order.
    const facts = verdict.getByText('Service detail', { selector: 'dt' }).closest('dl')!
    expect([...facts.querySelectorAll('dt')].map((term) => term.textContent))
      .toEqual(['Service detail', 'Rule', 'Root symptom time', 'Attempt UID', 'Pipeline', 'Service'])
    expect(facts.previousElementSibling).toBe(rootSymptom)
    expect(verdict.getByText('Tenant guidance')).toBeTruthy()
    expect(verdict.queryByText('Cloud operator guidance')).toBeNull()
    expect(verdict.queryByText('Drain the node.')).toBeNull()
    expect(verdict.getByText('xid-79').getAttribute('data-variant')).toBe('zinc-subtle')
    expect(verdict.getByText('Retry elsewhere.')).toBeTruthy()
    expect(verdict.getByText('Include when asking for help', { selector: 'dt' }).nextElementSibling?.textContent).toBe('run ID')

    // Matched lines are the recorded lines only, in their own section counted in its header:
    // the root line is marked, no level words or line numbers are added.
    const matched = within(screen.getByRole('region', { name: 'Matched lines' }))
    const excerpt = matched.getByText('Root symptom').closest<HTMLElement>('[data-slot="log-excerpt"]')!
    // The root line's message, verbatim.
    expect(excerpt.querySelector('[data-root="true"] code')?.textContent).toBe('Xid 79')
    expect(within(excerpt).getByText('Root symptom').getAttribute('data-variant')).toBe('zinc-subtle')
    expect(excerpt.querySelector('time')?.getAttribute('datetime')).toBe(detail.log[0].t)
    expect(excerpt.textContent).not.toMatch(/ERROR|INFO|DEBUG|SYMP|◆/)
    // The count alone: each line's time carries its own UTC mark, as every time does.
    expect(matched.getByRole('heading', { level: 2, name: 'Matched lines' }).nextElementSibling?.textContent).toBe('1')

    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy evidence' })) })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy link' })) })
    expect(mocks.clipboardWrite).toHaveBeenCalledTimes(2)
    // The verdict by the page's name for it (its badge's label), not a code in capitals.
    expect(mocks.clipboardWrite.mock.calls[0][0]).toMatch(/^\*Attribution escalation: Hardware failure · Xid79\*/)
    expect(mocks.clipboardWrite.mock.calls[0][0]).toContain('**Include on escalation:**\n\nrun ID')
    expect(mocks.clipboardWrite.mock.calls[0][0]).not.toContain('Drain the node.')
    expect(mocks.clipboardWrite.mock.calls[0][0]).not.toContain('node ID')
    expect(mocks.clipboardWrite.mock.calls[1][0]).toBe(window.location.href)

    expect(screen.getByText('Export')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Download as JSON' }))
    fireEvent.click(screen.getByRole('button', { name: 'Download as YAML' }))
    fireEvent.click(screen.getByRole('button', { name: 'Download as log' }))
    fireEvent.click(screen.getByRole('button', { name: 'Download as Markdown' }))
    expect(mocks.downloadBlob.mock.calls.map(([name, , mime]) => [name, mime])).toEqual([
      ['verdict-team-job-att2-20260501-0102.json', 'application/json'],
      ['verdict-team-job-att2-20260501-0102.yaml', 'application/yaml'],
      ['verdict-team-job-att2-20260501-0102.log', 'text/plain'],
      ['verdict-team-job-att2-20260501-0102.md', 'text/markdown'],
    ])
    expect(mocks.downloadBlob.mock.calls[3][1]).toBe(mocks.clipboardWrite.mock.calls[0][0])

    // Page actions cover the whole page; there is no shortcut for only the first responsible node.
    expect(screen.queryByRole('button', { name: /Other runs/ })).toBeNull()
    expect(mocks.nav).not.toHaveBeenCalled()
  })

  it('displays and exports the same multiline tenant instructions without rendering HTML', async () => {
    const action = '1. Check your application settings.\n2. Retry only your workload.\n3. Confirm the failed operation completes.'
    const support = 'Include the NVIDIA Attribution diagnosis (verdict, symptom chain), job ID and available logs; you can ask for help with incomplete evidence.'
    const markup = '<img src=x onerror=alert(1)>'
    mocks.useSWR.mockReturnValue({
      data: run({ diagnose: { ...run().diagnose!, action, supportDetails: [support, markup] } }),
      isLoading: false,
      error: undefined,
    })
    render(<MemoryRouter><VerdictView /></MemoryRouter>)

    const verdict = within(screen.getByRole('region', { name: 'Verdict' }))
    const actionValue = verdict.getByText('Action', { selector: 'dt' }).nextElementSibling!
    expect(actionValue.textContent).toBe(action)
    expect(actionValue.firstElementChild?.classList.contains('whitespace-pre-wrap')).toBe(true)
    const supportValue = verdict.getByText('Include when asking for help', { selector: 'dt' }).nextElementSibling!
    expect(supportValue.textContent).toBe(`${support}\n\n${markup}`)
    expect(supportValue.querySelector('img')).toBeNull()
    expect(verdict.queryByText('Drain the node.')).toBeNull()

    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy evidence' })) })
    fireEvent.click(screen.getByRole('button', { name: 'Download as Markdown' }))
    const copied = mocks.clipboardWrite.mock.calls[0][0]
    expect(copied).toContain(`**Recommended action:**\n\n${action}`)
    expect(copied).toContain(support)
    expect(copied).not.toContain('Cloud operator guidance')
    expect(mocks.downloadBlob.mock.calls[0][1]).toBe(copied)
    fireEvent.click(screen.getByRole('button', { name: 'Download as JSON' }))
    fireEvent.click(screen.getByRole('button', { name: 'Download as YAML' }))
    const json = mocks.downloadBlob.mock.calls[1][1]
    const yaml = mocks.downloadBlob.mock.calls[2][1]
    expect(JSON.parse(json).tenant_guidance).toEqual({
      summary: run().diagnose!.summary,
      impact: run().diagnose!.impact,
      action: actionValue.textContent,
      support_details: [support, markup],
    })
    expect(yaml).toContain(`  action: ${JSON.stringify(action)}`)
    expect(yaml).toContain(`    - ${JSON.stringify(support)}`)
    expect(yaml).toContain(`    - ${JSON.stringify(markup)}`)
    for (const serialized of [json, yaml]) {
      expect(serialized).not.toContain('Drain the node.')
      expect(serialized).not.toContain('node ID')
      expect(serialized).not.toContain('operatorDiagnose')
    }
  })

  it.each([undefined, { ...run().diagnose!, hasPersona: false }])(
    'does not display operator guidance when tenant guidance is unavailable', (diagnose) => {
      mocks.useSWR.mockReturnValue({ data: run({ diagnose }), isLoading: false, error: undefined })
      render(<MemoryRouter><VerdictView /></MemoryRouter>)
      expect(screen.queryByText('Tenant guidance')).toBeNull()
      expect(screen.queryByText('Cloud operator guidance')).toBeNull()
      expect(screen.queryByText('Drain the node.')).toBeNull()
      expect(screen.queryByText('Retry elsewhere.')).toBeNull()
      fireEvent.click(screen.getByRole('button', { name: 'Download as JSON' }))
      fireEvent.click(screen.getByRole('button', { name: 'Download as YAML' }))
      expect(JSON.parse(mocks.downloadBlob.mock.calls[0][1]).tenant_guidance).toBeNull()
      expect(mocks.downloadBlob.mock.calls[1][1]).toContain('tenant_guidance: null')
      for (const [, content] of mocks.downloadBlob.mock.calls) {
        expect(content).not.toContain('Drain the node.')
        expect(content).not.toContain('Retry elsewhere.')
      }
    },
  )

  it('orders the sections and keeps the raw payloads folded but copyable', async () => {
    mocks.useParams.mockReturnValue({ id: 'run-1' })
    render(<MemoryRouter><VerdictView /></MemoryRouter>)

    expect(screen.getAllByRole('region').map((region) => region.getAttribute('aria-label') ?? region.querySelector('h2')?.textContent))
      .toEqual(['Verdict', 'Matched lines', 'Nodes', 'Symptoms', 'Payloads'])
    // The answer and its lines share one grid row from 1024px: the lines take the
    // main two thirds on the left and fill their section, the answer the last third;
    // stacked, the answer stays first, as it is in reading order.
    const answer = screen.getByRole('region', { name: 'Verdict' })
    const matched = screen.getByRole('region', { name: 'Matched lines' })
    expect(answer.parentElement).toBe(matched.parentElement)
    // The answer's third is never narrower than what in the card cannot wrap:
    // its header, which does not wrap there, and the facts' timestamp. Every
    // other recorded value wraps anywhere, so none of them widens the answer.
    expect(answer.parentElement!.className.split(' ')).toEqual(expect.arrayContaining(['grid', 'grid-cols-1', 'lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(min-content,1fr)]']))
    expect(answer.className.split(' ').filter((name) => /^(\w+:)*(col-span|order)-/.test(name))).toEqual([])
    expect(matched.className.split(' ')).toEqual(expect.arrayContaining(['lg:order-first', 'lg:col-span-2']))
    expect(answer.querySelector('[data-slot="card-header"]')!.classList.contains('lg:flex-nowrap')).toBe(true)
    const bodies = [...answer.querySelectorAll('[data-slot="card-content"]')]
    expect(bodies).toHaveLength(2)
    for (const body of bodies) expect(body.className).not.toMatch(/contain-/)
    expect(answer.querySelector('[data-slot="card-content"] > p')!.classList.contains('wrap-anywhere')).toBe(true)
    const values = [...answer.querySelectorAll('dd')]
    expect(values.length).toBeGreaterThan(6)
    for (const value of values) expect(value.classList.contains('wrap-anywhere')).toBe(true)
    expect(within(answer).getByRole('button', { name: 'Copy attempt UID' }).previousElementSibling!.classList.contains('wrap-anywhere')).toBe(true)
    // The values' column takes its least width from them, so the root symptom
    // time, a stamp that never breaks (TimeHover), sets the list's.
    expect(within(answer).getByText('Root symptom time', { selector: 'dt' }).nextElementSibling!.querySelector('time')).not.toBeNull()
    expect(answer.querySelector('dl')!.classList.contains('sm:grid-cols-[112px_minmax(min-content,1fr)]')).toBe(true)
    // Under 640px a label stands over its value, which has the card's width, so the stamp fits a phone.
    expect(answer.querySelector('dl')!.classList.contains('grid-cols-1')).toBe(true)
    const well = matched.querySelector<HTMLElement>('[data-slot="log-excerpt"]')!
    expect(well.parentElement!.className.split(' ')).toEqual(expect.arrayContaining(['flex', 'flex-1', 'flex-col', 'p-0']))
    expect(well.className.split(' ')).toEqual(expect.arrayContaining(['min-h-60', 'flex-1', 'contain-size']))

    const payloads = within(screen.getByRole('region', { name: 'Payloads' }))
    const copy = payloads.getByRole('button', { name: 'Copy Result summary as JSON' })
    expect(payloads.getByRole('button', { name: 'Copy Scheduler summary as JSON' })).toBeTruthy()
    // Runs recorded before ingestion reports were saved get no third card.
    expect(payloads.queryByRole('button', { name: 'Copy Ingestion reports as JSON' })).toBeNull()
    expect(payloads.queryByRole('region', { name: 'Result summary' })).toBeNull()
    // Folded, a block is its header alone inside one border: the header draws no divider above the bottom edge.
    const header = copy.closest<HTMLElement>('[data-slot="code-block-header"]')!
    expect([...header.classList].filter((name) => /^([\w-]+:)*border(-[trblxy])?$/.test(name))).toEqual([])
    await act(async () => { fireEvent.click(copy) })
    const copied = JSON.parse(mocks.clipboardWrite.mock.calls[0][0] as string)
    expect(Object.keys(copied)).toEqual(['attempt_uid', 'outcome', 'root_symptom', 'symptoms', 'nodes', 'responsible_nodes', 'elapsed_hours', 'pipeline', 'service'])
    expect(copied.attempt_uid).toBe('attempt-2')

    const trigger = payloads.getByRole('button', { name: 'Result summary JSON' })
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    const body = payloads.getByRole('region', { name: 'Result summary' })
    expect(JSON.parse(body.textContent!)).toEqual(copied)
    expect(body.classList.contains('border-t')).toBe(true)
    // Open, a payload's body is as tall as its JSON up to 20rem, then scrolls: never a fixed height.
    expect(body.classList.contains('max-h-80')).toBe(true)
    expect([...body.classList].filter((name) => /^h-/.test(name))).toEqual([])

    // The scheduler summary never shows the adapter's constant attempt number.
    await act(async () => { fireEvent.click(payloads.getByRole('button', { name: 'Copy Scheduler summary as JSON' })) })
    const scheduler = JSON.parse(mocks.clipboardWrite.mock.calls[1][0] as string)
    expect(Object.keys(scheduler)).toEqual(['execution_context', 'user', 'status', 'nodes', 'start_time', 'end_time'])
  })

  it('shows the ingestion reports when the run recorded them', () => {
    mocks.useSWR.mockReturnValue({ data: run({ ingestionReports: { connector_reports: [], ingestion_reports: [{ source: 'syslog', status: 'error' }] } }), isLoading: false, error: undefined })
    render(<MemoryRouter><VerdictView /></MemoryRouter>)
    expect(within(screen.getByRole('region', { name: 'Payloads' })).getByRole('button', { name: 'Copy Ingestion reports as JSON' })).toBeTruthy()
  })

  it.each(['hardware', 'software', 'multi', 'user', 'undetermined', 'pass'] as const)('always marks the root line of a %s verdict, the same way', (outcome) => {
    mocks.useParams.mockReturnValue({ id: 'run-1' })
    mocks.useSWR.mockReturnValue({
      data: run({
        outcome,
        log: [
          { t: '2026-05-01T00:29:00Z', level: 'error', text: 'Earlier line.' },
          { t: '2026-05-01T00:30:00Z', level: 'symptom', text: 'Job completed with exit code 0.' },
        ],
      }),
      isLoading: false,
      error: undefined,
    })
    render(<MemoryRouter><VerdictView /></MemoryRouter>)

    // The root line is the line to read, not a blame: whatever the outcome it
    // takes the accent tint and a neutral label, and nothing on it is red.
    const excerpt = screen.getByText('Root symptom').closest<HTMLElement>('[data-slot="log-excerpt"]')!
    const roots = excerpt.querySelectorAll<HTMLElement>('[data-root="true"]')
    expect(roots).toHaveLength(1)
    expect([...roots[0].querySelectorAll('code')].map((code) => code.textContent)).toEqual(['Job completed with exit code 0.'])
    expect(roots[0].className).toContain('data-[root=true]:bg-primary-500/10')
    expect(within(roots[0]).getByText('Root symptom').getAttribute('data-variant')).toBe('zinc-subtle')
    expect(excerpt.innerHTML).not.toMatch(/destructive|red-subtle/)
    expect(within(excerpt).getAllByText('Root symptom')).toHaveLength(1)
  })

  it('renders the verdict card for a run without a detail, root symptom or matched lines', () => {
    mocks.useParams.mockReturnValue({ id: 'run-1' })
    mocks.useSWR.mockReturnValue({
      data: run({
        detail: '',
        rootSymptom: { symptom: '', rule: '', time: '2026-05-01T00:30:00Z', line: '' },
        log: [],
        user: '',
        pipeline: '',
        service: '',
        diagnose: undefined,
        operatorDiagnose: undefined,
      }),
      isLoading: false,
      error: undefined,
    })
    render(<MemoryRouter><VerdictView /></MemoryRouter>)

    const verdict = within(screen.getByRole('region', { name: 'Verdict' }))
    expect(verdict.getByRole('heading', { level: 2, name: 'Verdict: Hardware failure' })).toBeTruthy()
    const rootSymptom = verdict.getByText('Root symptom not recorded')
    expect(rootSymptom.tagName).toBe('P')
    expect(rootSymptom.className).toContain('text-foreground-muted')
    expect(verdict.queryByText('Service detail')).toBeNull()
    expect([...rootSymptom.nextElementSibling!.querySelectorAll('dt')].map((term) => term.textContent))
      .toEqual(['Rule', 'Root symptom time', 'Attempt UID', 'Pipeline', 'Service'])
    expect(verdict.getByText('Rule', { selector: 'dt' }).nextElementSibling?.textContent).toBe('Not recorded')
    expect(verdict.getByText('Pipeline', { selector: 'dt' }).nextElementSibling?.textContent).toBe('Not recorded')
    expect(verdict.getByText('Service', { selector: 'dt' }).nextElementSibling?.textContent).toBe('Not recorded')
    expect(screen.getByText('User', { selector: 'dt' }).nextElementSibling?.textContent).toBe('Not recorded')
    expect(verdict.queryByText('Cloud operator guidance')).toBeNull()
    expect(verdict.queryByText('Error pairs')).toBeNull()
    const matched = within(screen.getByRole('region', { name: 'Matched lines' }))
    expect(matched.getByText('No matched lines were recorded.')).toBeTruthy()
    expect(matched.getByRole('heading', { level: 2, name: 'Matched lines' }).nextElementSibling).toBeNull()
  })

  it('renders error pairs in the verdict details', () => {
    const firstPair = "{'src_node': 'node-a', 'dst_node': 'node-b'}"
    const secondPair = "{'src_node': 'node-c', 'dst_node': 'node-d'}"
    mocks.useParams.mockReturnValue({ id: 'run-1' })
    mocks.useSWR.mockReturnValue({
      data: run({ errorPairs: `${firstPair}; ${secondPair}` }),
      isLoading: false,
      error: undefined,
    })

    render(<MemoryRouter><VerdictView /></MemoryRouter>)

    // The pairs are the last row of the one facts list, not a row of their own spanning columns.
    const term = screen.getByText('Error pairs', { selector: 'dt' })
    expect([...term.closest('dl')!.querySelectorAll('dt')].at(-1)).toBe(term)
    expect(term.closest('dl')!.querySelectorAll('dt')).toHaveLength(7)
    const pairRows = within(screen.getByRole('list', { name: 'error pairs' })).getAllByRole('listitem')
    expect(pairRows).toHaveLength(2)
    // Read as recorded: an identifier's break points (hidden zero-width spaces) are no text.
    expect(pairRows.map((row) => row.textContent!.replaceAll('\u200b', ''))).toEqual([firstPair, secondPair])
  })
})
