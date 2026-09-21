// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { RunDetail, SymptomSite } from '@/interfaces/attribution'
import { LogHero } from '@/routers/verdict/components/log-hero'
import { NodeBubble } from '@/routers/verdict/components/node-bubble'
import { SymptomsSection } from '@/routers/verdict/components/symptom-sites'

const mocks = vi.hoisted(() => ({
  editorProps: vi.fn(),
  loaderConfig: vi.fn(),
  defineTheme: vi.fn(),
  setTheme: vi.fn(),
  symptomDocumentation: vi.fn(),
}))

vi.mock('@/hooks/use-symptom-documentation', () => ({ useSymptomDocumentation: mocks.symptomDocumentation }))
vi.mock('@monaco-editor/react', () => ({
  default: (props: { value: string; onMount: unknown }) => {
    mocks.editorProps(props)
    return <pre>{props.value}</pre>
  },
  loader: { config: mocks.loaderConfig },
}))
vi.mock('monaco-editor', () => ({
  editor: {
    defineTheme: mocks.defineTheme,
    setTheme: mocks.setTheme,
    ScrollType: { Immediate: 1 },
  },
  Range: class {
    startLineNumber: number
    startColumn: number
    endLineNumber: number
    endColumn: number

    constructor(
      startLineNumber: number,
      startColumn: number,
      endLineNumber: number,
      endColumn: number,
    ) {
      this.startLineNumber = startLineNumber
      this.startColumn = startColumn
      this.endLineNumber = endLineNumber
      this.endColumn = endColumn
    }
  },
}))

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

describe('LogHero', () => {
  it('formats log levels and installs Monaco decorations', () => {
    render(<LogHero run={run()} />)
    const props = mocks.editorProps.mock.calls[0][0]
    const createDecorationsCollection = vi.fn()
    const revealLineInCenter = vi.fn()
    const editor = { createDecorationsCollection, revealLineInCenter }
    const monaco = {
      editor: {
        defineTheme: mocks.defineTheme,
        setTheme: mocks.setTheme,
        ScrollType: { Immediate: 1 },
      },
      Range: class {
        startLineNumber: number
        startColumn: number
        endLineNumber: number
        endColumn: number

        constructor(
          startLineNumber: number,
          startColumn: number,
          endLineNumber: number,
          endColumn: number,
        ) {
          this.startLineNumber = startLineNumber
          this.startColumn = startColumn
          this.endLineNumber = endLineNumber
          this.endColumn = endColumn
        }
      },
    }

    props.onMount(editor, monaco)

    expect(props.value).toContain('INFO    started')
    expect(props.value).toContain('DEBUG   checking')
    expect(props.value).toContain('◆ SYMP  Xid 79')
    expect(props.value).toContain('ERROR   stopped')
    expect(createDecorationsCollection.mock.calls[0][0]).toHaveLength(3)
    expect(revealLineInCenter).toHaveBeenCalledWith(3, 1)
    expect(screen.getByText('GPU unavailable.')).toBeTruthy()
    expect(screen.getByText(/matched at 00:03:00/)).toBeTruthy()
  })

  it('renders an unavailable attribution time', () => {
    render(<LogHero run={run({ rootSymptom: { symptom: 'Unknown', rule: 'fallback', time: 'invalid', line: '' } })} />)

    expect(screen.getByText(/matched at Unavailable/)).toBeTruthy()
  })
})

describe('NodeBubble', () => {
  it('renders responsible and ordinary node tones', () => {
    const { rerender } = render(<NodeBubble node="node-a" responsible className="custom" />)
    expect(screen.getByText('node-a').className).toContain('bg-white/5')
    expect(screen.getByText('node-a').className).toContain('custom')

    rerender(<NodeBubble node="node-b" />)
    expect(screen.getByText('node-b').className).toContain('hover:text-primary')
  })
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

    const finding = within(screen.getByRole('article', { name: 'node-a: Bad' }))
    expect(finding.getByText('recorded-rule')).toBeTruthy()
    expect(finding.getByText('syslog')).toBeTruthy()
    expect(finding.getByText('Observation').nextElementSibling?.textContent).toBe(observationId)
    expect(finding.getByText('Source position').nextElementSibling?.textContent).toBe('0')
    const recordedTime = finding.getByText('Recorded time').nextElementSibling
    expect(recordedTime?.querySelector('time')?.getAttribute('datetime')).toBe(source.time)
    expect(recordedTime?.querySelector('[data-slot="hover-card-trigger"]')?.textContent).toBe('2026-05-01 00:03:00UTC')
    const log = finding.getByRole('region', { name: 'Source evidence' })
    expect([...log.querySelectorAll('pre')].map(line => line.textContent).join('')).toBe(source.line)
    expect(within(screen.getByRole('article', { name: 'node-b: Supporting evidence' })).getByText('supporting line')).toBeTruthy()
    const job = within(screen.getByRole('article', { name: 'Job-level evidence: Bad' }))
    expect(job.getByText('Source position').nextElementSibling?.textContent).toBe('Not recorded')
    expect(job.getByText('No source line recorded.')).toBeTruthy()
    expect(screen.queryByText(/pruned|responsible/i)).toBeNull()
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

    expect(screen.getByText(message)).toBeTruthy()
    expect(screen.getByText('GPU unavailable.')).toBeTruthy()
    expect(within(screen.getByRole('list', { name: 'Recorded symptom summary' })).getByText('Known symptom')).toBeTruthy()
    expect(screen.queryByRole('button')).toBeNull()
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
    render(<SymptomsSection run={detail} />)

    const trigger = screen.getByRole('button', { name: 'Mixed symptom 4 sites' })
    expect(screen.getByText('node').textContent).toBe('1 node')
    expect(screen.getByText('Job-level evidence')).toBeTruthy()
    expect(screen.getByText('Node not recorded')).toBeTruthy()
    expect(screen.getByText('Bad').textContent).toBe('2 Bad')
    expect(screen.getByText('Supporting evidence').textContent).toBe('2 Supporting evidence')
    expect(screen.getByText('sources').textContent).toBe('2 sources')
    expect(screen.queryByRole('article')).toBeNull()

    fireEvent.click(trigger)
    expect(within(screen.getByRole('article', { name: 'node-a: Bad' })).getByText('syslog')).toBeTruthy()
    expect(within(screen.getByRole('article', { name: 'node-a: Supporting evidence' })).getByText('slurm_info')).toBeTruthy()
    expect(screen.getAllByRole('article')).toHaveLength(4)
  })

  it('keeps single-site context visible when collapsed and distinguishes empty evidence', () => {
    const detail = run()
    detail.attribution.symptomSites = {
      state: 'available',
      data: {
        'Supporting symptom': [{ type: 'NodeSource', node: '', status: 'interesting', source: { ...source, line: '', source: '' } }],
        'No recorded sites': [],
      },
    }
    render(<SymptomsSection run={detail} />)

    expect(screen.getByText('Node not recorded')).toBeTruthy()
    expect(screen.getByText('Source not recorded')).toBeTruthy()
    expect(screen.getByText('Supporting evidence')).toBeTruthy()
    expect(screen.queryByText('Source line is empty.')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Supporting symptom 1 site' }))
    expect(screen.getByText('Source line is empty.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'No recorded sites 0 sites' }))
    expect(screen.getByText('No sites recorded for this symptom.')).toBeTruthy()
  })

  it('expands from any summary cell while keeping selection and log interaction independent', () => {
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
