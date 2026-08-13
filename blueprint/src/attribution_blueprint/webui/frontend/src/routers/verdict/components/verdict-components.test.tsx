// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import type { Run } from '@/interfaces/attribution'
import { LogHero } from '@/routers/verdict/components/log-hero'
import { NodeBubble } from '@/routers/verdict/components/node-bubble'

const mocks = vi.hoisted(() => ({
  editorProps: vi.fn(),
  loaderConfig: vi.fn(),
  defineTheme: vi.fn(),
  setTheme: vi.fn(),
}))

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
  verdictLabel: 'Hardware failure',
  detail: 'GPU unavailable.',
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
  ...overrides,
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
