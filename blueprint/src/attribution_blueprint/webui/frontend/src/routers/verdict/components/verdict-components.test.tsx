// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { RunDetail, SymptomSite } from '@/interfaces/attribution'
import { SymptomsSection } from '@/routers/verdict/components/symptom-sites'

const mocks = vi.hoisted(() => ({
  symptomDocumentation: vi.fn(),
}))

vi.mock('@/hooks/use-symptom-documentation', () => ({ useSymptomDocumentation: mocks.symptomDocumentation }))

const run = (overrides: Partial<RunDetail> = {}): RunDetail => ({
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
  verdictLabel: 'Hardware failure',
  detail: 'GPU unavailable.',
  decisionGraphState: 'absent',
  rootSymptom: {
    symptom: 'Xid79',
    rule: 'xid',
    time: '2026-05-01T00:03:00Z',
    line: 'Xid 79',
  },
  symptoms: [],
  responsibleNodes: ['node-a'],
  pipeline: '1',
  service: '2',
  log: [
    { t: '2026-05-01T00:01:00Z', level: 'info', text: 'started' },
    { t: '2026-05-01T00:02:00Z', level: 'debug', text: 'checking' },
    { t: '2026-05-01T00:03:00Z', level: 'symptom', text: 'Xid 79' },
    { t: '2026-05-01T00:04:00Z', level: 'error', text: 'stopped' },
  ],
  attribution: {
    rootSymptom: { state: 'absent', data: null },
    symptomSites: { state: 'absent', data: null },
    decisionGraph: { state: 'absent', data: null },
  },
  ...overrides,
})

beforeEach(() => {
  mocks.symptomDocumentation.mockReturnValue({ data: undefined })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('SymptomsSection', () => {
  const source: SymptomSite['source'] = {
    observation_id: '0',
    source: 'syslog',
    pos: 0,
    line: 'first line\n\n  preserved indentation\n',
    rule: 'recorded-rule',
    time: '2026-05-01T00:03:00+00:00',
  }

  const evidenceSites = (count: number): SymptomSite[] => Array.from({ length: count }, (_, index) => ({
    type: 'NodeSource',
    node: `node-${index + 1}`,
    status: index % 3 === 0 ? 'bad' : 'interesting',
    source: { ...source, pos: index, line: `Recorded evidence ${index + 1}` },
  }))

  const runWithEvidence = (sites: SymptomSite[], id = 'run-1'): RunDetail => run({
    id,
    attribution: {
      ...run().attribution,
      symptomSites: { state: 'available', data: { 'Recorded symptom': sites } },
    },
  })

  it('links each exact API symptom key beside its disclosure without toggling evidence', () => {
    const reference = { id: 'canonical-id', title: 'Published symptom', doc_url: 'https://docs.example/original/path/' }
    mocks.symptomDocumentation.mockReturnValue({ data: { symptoms: {
      'Recorded symptom': reference,
      'canonical-id': reference,
    } } })
    const detail = run()
    detail.attribution.rootSymptom = { state: 'available', data: {
      symptom: 'Recorded symptom', catalog_id: reference.id, source,
    } }
    detail.attribution.symptomSites = { state: 'available', data: {
      'Recorded symptom': [{ type: 'NodeSource', node: 'node-a', status: 'bad', source }],
      'Unmapped symptom': [{ type: 'NodeSource', node: 'node-b', status: 'bad', source }],
    } }
    const { rerender } = render(<SymptomsSection run={detail} />)
    const trigger = screen.getByRole('button', { name: 'Recorded symptom 1 site' })
    const doc = within(trigger.closest('tr')!).getByRole('link', { name: /Symptom documentation/ })
    expect(doc.getAttribute('href')).toBe(reference.doc_url)
    expect(doc.getAttribute('target')).toBe('_blank')
    expect(trigger.contains(doc)).toBe(false)
    expect(doc.parentElement).toBe(trigger.parentElement)
    // A name that wraps keeps the toggle's chevron, centered in its box of the
    // name's first line, and the link, a mark centered on the name's lowercase, there.
    expect(trigger.parentElement!.classList.contains('items-start')).toBe(true)
    expect(trigger.classList.contains('items-start')).toBe(true)
    expect([...trigger.firstElementChild!.classList]).toEqual(expect.arrayContaining(['flex', 'h-5', 'items-center']))
    expect([...doc.classList]).toEqual(expect.arrayContaining(['h-5', 'relative', 'top-(--mark-offset)']))
    doc.addEventListener('click', (event) => event.preventDefault())
    fireEvent.click(doc)
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(trigger)
    fireEvent.click(doc)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getAllByRole('link', { name: /Symptom documentation/ })).toHaveLength(1)

    mocks.symptomDocumentation.mockReturnValue({ data: { symptoms: { 'canonical-id': reference } } })
    rerender(<SymptomsSection run={detail} />)
    expect(screen.queryByRole('link', { name: /Symptom documentation/ })).toBeNull()
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
  })

  it.each(['0', '9007199254740993', '-6139371663235533333'])('preserves observation ID %s without conflating source evidence', (observationId) => {
    const recordedSource = { ...source, observation_id: observationId }
    const detail = run({
      symptoms: [{ name: 'Summary only', sites: 99 }],
      attribution: {
        ...run().attribution,
        symptomSites: {
          state: 'available',
          data: {
            'Recorded symptom': [
              { type: 'NodeSource', node: 'node-a', status: 'bad', source: recordedSource },
              { type: 'NodeSource', node: 'node-b', status: 'interesting', source: { ...recordedSource, line: 'supporting line' } },
              { type: 'ResourcelessSource', status: 'bad', source: { ...recordedSource, pos: null, time: null, line: null, rule: null, source: '' } },
            ],
          },
        },
      },
    })
    const original = structuredClone(detail)
    render(<SymptomsSection run={detail} />)

    const trigger = screen.getByRole('button', { name: 'Recorded symptom 3 sites' })
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText('Summary only')).toBeNull()
    expect(screen.queryByRole('article')).toBeNull()
    fireEvent.click(trigger)

    const finding = within(screen.getByRole('article', { name: 'node-a: Responsible' }))
    expect(finding.getByText('recorded-rule')).toBeTruthy()
    // The node's icon keeps to the first line of a name that wraps, a mark in a
    // box of that line centered on its lowercase.
    const siteName = finding.getByText('node-a').closest('span.break-words')!
    expect(siteName.parentElement!.classList.contains('items-start')).toBe(true)
    expect([...siteName.previousElementSibling!.querySelector('svg')!.classList]).toEqual(expect.arrayContaining(['inline', 'align-middle', '-my-1']))
    expect(finding.getByText('syslog')).toBeTruthy()
    expect(finding.getByText('Observation').nextElementSibling?.textContent).toBe(observationId)
    expect(finding.getByText('Source position').nextElementSibling?.textContent).toBe('0')
    // The four facts stand on the table's own columns: its column widths
    // across the row's full width, each fact inset as the cell above it is.
    // The widths are set once, on the table, and both read them from there.
    const table = screen.getByRole('table')
    expect(table.className.split(' ')).toEqual(expect.arrayContaining([
      '[--col-symptom:38%]', '[--col-scope:19%]', '[--col-source:13%]', '[--col-status:23%]', '[--col-sites:7%]',
    ]))
    expect([...table.querySelectorAll('col')].map((col) => col.className))
      .toEqual(['w-(--col-symptom)', 'w-(--col-scope)', 'w-(--col-source)', 'w-(--col-status)', 'w-(--col-sites)'])
    const facts = finding.getByText('Rule').closest('dl')!
    expect(table.contains(facts)).toBe(true)
    expect(facts.className.split(' ')).toEqual(expect.arrayContaining(['-mx-card-x', 'grid', 'grid-cols-[var(--col-symptom)_var(--col-scope)_var(--col-source)_minmax(0,1fr)]']))
    // Each fact is inset as the cell above it: 12px, and the card's inset at the row's ends.
    expect([...facts.children].map((fact) => ['px-3', 'first:pl-card-x', 'last:pr-card-x'].every((c) => fact.classList.contains(c)))).toEqual([true, true, true, true])
    expect(screen.getByRole('region', { name: 'Recorded symptom evidence' }).className.split(' ')).toEqual(expect.arrayContaining(['-mx-card-x', 'px-card-x']))
    // The recorded time is itself the hover card's trigger, keeping the recorded string.
    const recordedTime = finding.getByText('Recorded time').nextElementSibling?.querySelector('time')
    expect(recordedTime?.getAttribute('datetime')).toBe(source.time)
    expect(recordedTime?.getAttribute('data-slot')).toBe('hover-card-trigger')
    expect(recordedTime?.textContent).toBe('2026-05-01 00:03:00 UTC')
    const log = finding.getByRole('region', { name: 'Source evidence' })
    expect([...log.querySelectorAll('pre')].map(line => line.textContent).join('')).toBe(source.line)
    // The recorded line is a machine-output block whose copy action is always shown.
    expect(log.closest('[data-slot="code-block"]')).not.toBeNull()
    expect(finding.getByRole('button', { name: 'Copy source evidence' })).toBeTruthy()
    expect(within(screen.getByRole('article', { name: 'node-b: Supporting evidence' })).getByText('supporting line')).toBeTruthy()
    const job = within(screen.getByRole('article', { name: 'Job-level evidence: Responsible' }))
    expect(job.getByText('Source position').nextElementSibling?.textContent).toBe('Not recorded')
    expect(job.getByText('No source line recorded.')).toBeTruthy()
    expect(screen.queryByText(/pruned/i)).toBeNull()
    expect(detail).toEqual(original)

    fireEvent.click(trigger)
    expect(screen.queryByRole('article')).toBeNull()
  })

  it.each([
    ['absent', 'Symptom-site evidence was not included in this run.'],
    ['null', 'Symptom-site evidence is unavailable for this run.'],
    ['invalid', 'Stored symptom-site evidence could not be read.'],
    ['empty', 'No symptom sites were recorded for this run.'],
  ] as const)('keeps the %s evidence state local to the symptoms card', (state, message) => {
    const detail = run({ symptoms: [{ name: 'Known symptom', sites: 2 }] })
    detail.attribution.symptomSites = state === 'empty'
      ? { state, data: {} }
      : { state, data: null }
    render(<><div>{detail.detail}</div><SymptomsSection run={detail} /></>)

    // Beside the recorded summary, why its sites are missing is a muted note.
    expect(screen.getByText(message).tagName).toBe('P')
    expect(screen.getByText(message).classList.contains('text-foreground-muted')).toBe(true)
    expect(screen.getByText('GPU unavailable.')).toBeTruthy()
    const summary = screen.getByRole('list', { name: 'Recorded symptom summary' })
    expect(within(summary).getByText('Known symptom')).toBeTruthy()
    // The summary keeps the table's rhythm: flush 28px rows on the card's inset,
    // every row alike, and the note below them inset as card content is.
    expect(summary.parentElement!.classList.contains('p-0')).toBe(true)
    expect([...within(summary).getByRole('listitem').classList]).toEqual(expect.arrayContaining(['px-card-x', 'py-1']))
    expect(within(summary).getByRole('listitem').className).not.toMatch(/first:/)
    expect([...screen.getByText(message).classList]).toEqual(expect.arrayContaining(['border-t', 'px-card-x', 'py-card-y']))
    expect(screen.queryByRole('button')).toBeNull()
    expect(document.querySelector('[data-slot="empty"]')).toBeNull()

    // With no summary either, the section has nothing to show: one empty state, the state its description.
    cleanup()
    const bare = run({ symptoms: [] })
    bare.attribution.symptomSites = detail.attribution.symptomSites
    render(<SymptomsSection run={bare} />)
    const section = screen.getByRole('region', { name: 'Symptoms' })
    expect(within(section).queryByRole('list')).toBeNull()
    const empty = section.querySelector<HTMLElement>('[data-slot="empty"]')!
    expect(empty.getAttribute('data-size')).toBe('sm')
    expect(within(empty).getByText('No symptoms to show').getAttribute('data-slot')).toBe('empty-title')
    expect(within(empty).getByText(message).getAttribute('data-slot')).toBe('empty-description')
  })

  it('counts recorded sites separately from distinct nodes in a mixed-scope summary', () => {
    const detail = run()
    detail.attribution.symptomSites = {
      state: 'available',
      data: {
        'Mixed symptom': [
          { type: 'NodeSource', node: 'node-a', status: 'bad', source },
          { type: 'NodeSource', node: 'node-a', status: 'interesting', source: { ...source, source: 'slurm_info' } },
          { type: 'NodeSource', node: '', status: 'bad', source },
          { type: 'ResourcelessSource', status: 'interesting', source },
        ],
      },
    }
    const { rerender } = render(<SymptomsSection run={detail} />)

    const trigger = screen.getByRole('button', { name: 'Mixed symptom 4 sites' })
    // A count and its noun are one phrase in one text node.
    expect(screen.getByText('1 node')).toBeTruthy()
    expect(screen.getByText('Job-level evidence')).toBeTruthy()
    expect(screen.getByText('Node not recorded')).toBeTruthy()
    // Blame is the red badge; supporting evidence stays neutral. Each count shares its label's text node.
    expect(screen.getByText('2 Responsible').getAttribute('data-variant')).toBe('red-subtle')
    expect(screen.getByText('2 Supporting evidence').getAttribute('data-variant')).toBe('zinc-subtle')
    expect(screen.getByText('2 sources')).toBeTruthy()
    expect(screen.queryByRole('article')).toBeNull()

    fireEvent.click(trigger)
    for (const node of ['node-a', 'Node not recorded']) {
      const site = within(screen.getByRole('article', { name: `${node}: Responsible` }))
      expect(site.getByText(node)).toBeTruthy()
      expect(site.getByText('syslog').getAttribute('data-slot')).toBe('badge')
      expect(site.getByText('syslog').getAttribute('data-variant')).toBe('zinc-subtle')
      expect(site.getByText('Responsible').getAttribute('data-slot')).toBe('badge')
      expect(site.getByText('Responsible').getAttribute('data-variant')).toBe('red-subtle')
      expect(site.getByText('Responsible').textContent).toBe('Responsible')
      expect(site.queryByText('Supporting evidence')).toBeNull()
    }
    for (const [label, sourceName] of [['node-a', 'slurm_info'], ['Job-level evidence', 'syslog']]) {
      const site = within(screen.getByRole('article', { name: `${label}: Supporting evidence` }))
      expect(site.getByText(label)).toBeTruthy()
      // The source name is the whole of its badge, whether or not it has a
      // separator to break after.
      const sourceBadge = site.getByText(sourceName).closest('[data-slot="badge"]')!
      expect(sourceBadge.textContent!.replaceAll('\u200b', '')).toBe(sourceName)
      expect(sourceBadge.getAttribute('data-variant')).toBe('zinc-subtle')
      expect(site.queryByText('Supporting evidence')).toBeNull()
      expect(site.queryByText('Responsible')).toBeNull()
    }
    expect(screen.queryByText('Bad')).toBeNull()
    expect(screen.getAllByRole('article')).toHaveLength(4)

    const sites = detail.attribution.symptomSites.data!['Mixed symptom']
    for (const [status, label, otherLabel, variant] of [
      ['bad', 'Responsible', 'Supporting evidence', 'red-subtle'],
      ['interesting', 'Supporting evidence', 'Responsible', 'zinc-subtle'],
    ] as const) {
      rerender(<SymptomsSection run={runWithEvidence(sites.filter((site) => site.status === status))} />)
      const row = within(screen.getByRole('button', { name: 'Recorded symptom 2 sites' }).closest('tr')!)
      expect(row.getByText(`2 ${label}`).getAttribute('data-variant')).toBe(variant)
      expect(row.queryByText(new RegExp(otherLabel))).toBeNull()
    }
  })

  it('pages every recorded site in order without deduplicating IDs, changing totals, or fetching more data', () => {
    const sites = evidenceSites(21)
    const detail = runWithEvidence(sites)
    const original = structuredClone(detail)
    const fetch = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Evidence is already recorded in the run'))
    try {
      render(<SymptomsSection run={detail} />)
      const trigger = screen.getByRole('button', { name: 'Recorded symptom 21 sites' })
      const summary = within(trigger.closest('tr')!)
      fireEvent.click(trigger)

      const first = screen.getByRole('button', { name: 'First evidence page' })
      const previous = screen.getByRole('button', { name: 'Previous evidence page' })
      const next = screen.getByRole('button', { name: 'Next evidence page' })
      const last = screen.getByRole('button', { name: 'Last evidence page' })
      const pages = screen.getByRole('group', { name: 'Recorded symptom evidence pages' })
      const displayedLines = () => screen.getAllByRole('article').map((article) => article.querySelector('pre')?.textContent)
      const allLines: (string | null | undefined)[] = []
      const expectFullSummary = () => {
        expect(summary.getByText('21 nodes')).toBeTruthy()
        expect(summary.getByText('7 Responsible')).toBeTruthy()
        expect(summary.getByText('14 Supporting evidence')).toBeTruthy()
        expect(summary.getByRole('cell', { name: '21' })).toBeTruthy()
        expect(trigger.getAttribute('aria-expanded')).toBe('true')
      }

      expect(first.getAttribute('aria-disabled')).toBe('true')
      expect(previous.getAttribute('aria-disabled')).toBe('true')
      expect(next.getAttribute('aria-disabled')).toBe(null)
      expect(last.getAttribute('aria-disabled')).toBe(null)
      expect(pages.textContent).toContain('1–10 of 21 sites')
      expect(screen.getAllByRole('article')).toHaveLength(10)
      allLines.push(...displayedLines())
      expectFullSummary()

      fireEvent.click(next)
      expect(pages.textContent).toContain('11–20 of 21 sites')
      expect(screen.getAllByRole('article')).toHaveLength(10)
      allLines.push(...displayedLines())
      expectFullSummary()

      fireEvent.click(next)
      expect(pages.textContent).toContain('21–21 of 21 sites')
      expect(screen.getAllByRole('article')).toHaveLength(1)
      allLines.push(...displayedLines())
      expect(allLines).toEqual(sites.map((site) => site.source.line))
      const finalSite = within(screen.getByRole('article', { name: 'node-21: Supporting evidence' }))
      expect(finalSite.getByText('node-21')).toBeTruthy()
      expect(finalSite.getByText('syslog')).toBeTruthy()
      expect(finalSite.getByText('Observation').nextElementSibling?.textContent).toBe(source.observation_id)
      expect(finalSite.getByText('Source position').nextElementSibling?.textContent).toBe('20')
      expect(first.getAttribute('aria-disabled')).toBe(null)
      expect(previous.getAttribute('aria-disabled')).toBe(null)
      expect(next.getAttribute('aria-disabled')).toBe('true')
      expect(last.getAttribute('aria-disabled')).toBe('true')
      // Past the end there is nowhere to go: the page stays.
      fireEvent.click(next)
      expect(pages.textContent).toContain('21–21 of 21 sites')
      expectFullSummary()

      fireEvent.click(previous)
      expect(displayedLines()).toEqual(sites.slice(10, 20).map((site) => site.source.line))
      fireEvent.click(first)
      expect(displayedLines()).toEqual(sites.slice(0, 10).map((site) => site.source.line))
      fireEvent.click(last)
      expect(displayedLines()).toEqual([sites[20].source.line])
      expectFullSummary()
      expect(fetch).not.toHaveBeenCalled()
      expect(detail).toEqual(original)
    } finally {
      fetch.mockRestore()
    }
  })

  it('starts at the first evidence page after collapsing and reopening a symptom', () => {
    render(<SymptomsSection run={runWithEvidence(evidenceSites(21))} />)
    const trigger = screen.getByRole('button', { name: 'Recorded symptom 21 sites' })
    fireEvent.click(trigger)
    fireEvent.click(screen.getByRole('button', { name: 'Last evidence page' }))
    expect(screen.getByRole('article', { name: 'node-21: Supporting evidence' })).toBeTruthy()

    fireEvent.click(trigger)
    expect(screen.queryByRole('article')).toBeNull()
    expect(screen.queryByRole('group', { name: 'Recorded symptom evidence pages' })).toBeNull()
    fireEvent.click(trigger)
    expect(screen.getAllByRole('article')).toHaveLength(10)
    expect(screen.getByRole('article', { name: 'node-1: Responsible' })).toBeTruthy()
    expect(screen.queryByRole('article', { name: 'node-21: Supporting evidence' })).toBeNull()
    expect(screen.getByRole('button', { name: 'First evidence page' }).getAttribute('aria-disabled')).toBe('true')
  })

  it('keeps the current page valid when recorded sites shrink and does not restore a stale page when they grow', () => {
    const sites = evidenceSites(21)
    const { rerender } = render(<SymptomsSection run={runWithEvidence(sites)} />)
    fireEvent.click(screen.getByRole('button', { name: 'Recorded symptom 21 sites' }))
    fireEvent.click(screen.getByRole('button', { name: 'Last evidence page' }))

    rerender(<SymptomsSection run={runWithEvidence(sites.slice(0, 11))} />)
    expect(screen.getByRole('group', { name: 'Recorded symptom evidence pages' }).textContent).toContain('11–11 of 11 sites')
    expect(screen.getAllByRole('article')).toHaveLength(1)
    expect(screen.getByRole('article', { name: 'node-11: Supporting evidence' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Next evidence page' }).getAttribute('aria-disabled')).toBe('true')

    rerender(<SymptomsSection run={runWithEvidence(sites)} />)
    expect(screen.getByRole('group', { name: 'Recorded symptom evidence pages' }).textContent).toContain('11–20 of 21 sites')
    expect(screen.getAllByRole('article')).toHaveLength(10)
    expect(screen.getByRole('article', { name: 'node-11: Supporting evidence' })).toBeTruthy()
    expect(screen.queryByRole('article', { name: 'node-21: Supporting evidence' })).toBeNull()

    rerender(<SymptomsSection run={runWithEvidence(sites.slice(0, 10))} />)
    expect(screen.queryByRole('group', { name: 'Recorded symptom evidence pages' })).toBeNull()
    expect(screen.getAllByRole('article')).toHaveLength(10)
    expect(screen.getByRole('article', { name: 'node-1: Responsible' })).toBeTruthy()
    rerender(<SymptomsSection run={runWithEvidence(sites)} />)
    expect(screen.getByRole('group', { name: 'Recorded symptom evidence pages' }).textContent).toContain('1–10 of 21 sites')
    expect(screen.getByRole('button', { name: 'First evidence page' }).getAttribute('aria-disabled')).toBe('true')
  })

  it('closes evidence when navigating to another run with the same symptom name', () => {
    const sites = evidenceSites(21)
    const { rerender } = render(<SymptomsSection run={runWithEvidence(sites)} />)
    fireEvent.click(screen.getByRole('button', { name: 'Recorded symptom 21 sites' }))
    fireEvent.click(screen.getByRole('button', { name: 'Last evidence page' }))

    rerender(<SymptomsSection run={runWithEvidence(sites, 'run-2')} />)
    const trigger = screen.getByRole('button', { name: 'Recorded symptom 21 sites' })
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByRole('article')).toBeNull()
    fireEvent.click(trigger)
    expect(screen.getAllByRole('article')).toHaveLength(10)
    expect(screen.getByRole('article', { name: 'node-1: Responsible' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'First evidence page' }).getAttribute('aria-disabled')).toBe('true')
  })

  it('keeps single-site context visible when collapsed and distinguishes empty evidence', () => {
    const detail = run()
    detail.attribution.symptomSites = {
      state: 'available',
      data: {
        'Supporting symptom': [{ type: 'NodeSource', node: '', status: 'interesting', source: { ...source, line: '', source: '' } }],
        'Responsible symptom': [{ type: 'NodeSource', node: 'node-a', status: 'bad', source }],
        'No recorded sites': [],
      },
    }
    render(<SymptomsSection run={detail} />)

    // The table takes its name from the section title itself, not a copy of it.
    const table = screen.getByRole('table', { name: 'Symptoms' })
    expect(document.getElementById(table.getAttribute('aria-labelledby')!)!.closest('h2'))
      .toBe(within(screen.getByRole('region', { name: 'Symptoms' })).getByRole('heading', { level: 2 }))
    expect(table.hasAttribute('aria-label')).toBe(false)
    // The distinct count sits beside the title, as every section count does, and stays out of its name.
    expect(within(screen.getByRole('region', { name: 'Symptoms' })).getByRole('heading', { level: 2 }).nextElementSibling?.textContent)
      .toBe('3 distinct')
    expect(screen.getByText('Node not recorded')).toBeTruthy()
    expect(screen.getByText('Source not recorded')).toBeTruthy()
    expect(screen.getByRole('columnheader', { name: 'Status' })).toBeTruthy()
    const supportingRow = within(screen.getByRole('button', { name: 'Supporting symptom 1 site' }).closest('tr')!)
    const responsibleRow = within(screen.getByRole('button', { name: 'Responsible symptom 1 site' }).closest('tr')!)
    const emptyRow = within(screen.getByRole('button', { name: 'No recorded sites 0 sites' }).closest('tr')!)
    expect(supportingRow.getByText('1 Supporting evidence')).toBeTruthy()
    expect(supportingRow.queryByText(/Responsible/)).toBeNull()
    expect(responsibleRow.getByText('1 Responsible')).toBeTruthy()
    expect(responsibleRow.queryByText(/Supporting evidence/)).toBeNull()
    // A symptom with no sites leaves its cells as the muted dash every table uses, read as "None".
    expect(emptyRow.getAllByRole('cell')[3].textContent).toBe('—None')
    expect(emptyRow.queryByText(/Responsible|Supporting evidence/)).toBeNull()
    expect(screen.queryByText('Source line is empty.')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Supporting symptom 1 site' }))
    expect(screen.getByText('Source line is empty.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'No recorded sites 0 sites' }))
    expect(screen.getByText('No sites recorded for this symptom.')).toBeTruthy()
  })

  it('expands from any summary cell while keeping selection and log interaction independent', async () => {
    const detail = run()
    detail.attribution.symptomSites = {
      state: 'available',
      data: { 'Recorded symptom': [{ type: 'NodeSource', node: 'node-a', status: 'bad', source }] },
    }
    render(<SymptomsSection run={detail} />)
    const trigger = screen.getByRole('button', { name: 'Recorded symptom 1 site' })
    const row = within(trigger.closest('tr')!)
    const scope = row.getByRole('cell', { name: 'node-a' })

    fireEvent.click(scope)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    fireEvent.click(screen.getByRole('region', { name: 'Source evidence' }))
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy source evidence' })) })
    expect(writeText).toHaveBeenCalledWith(source.line)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')

    expect(row.queryByRole('link')).toBeNull()

    fireEvent.click(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(row.getByRole('cell', { name: '1' }))
    expect(trigger.getAttribute('aria-expanded')).toBe('true')

    const selection = window.getSelection()!
    const range = document.createRange()
    range.selectNodeContents(scope)
    selection.addRange(range)
    fireEvent.click(scope)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    selection.removeAllRanges()
    fireEvent.click(scope)
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
  })
})
