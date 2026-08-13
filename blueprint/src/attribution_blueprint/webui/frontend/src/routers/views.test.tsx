// @vitest-environment jsdom
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { Run } from '@/interfaces/attribution'
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
}))

vi.mock('react-router-dom', async (importOriginal) => ({
  ...await importOriginal<typeof import('react-router-dom')>(),
  useParams: mocks.useParams,
}))
vi.mock('swr', () => ({ default: mocks.useSWR }))
vi.mock('@/hooks/use-node-runs', () => ({ useNodeRuns: mocks.useNodeRuns }))
vi.mock('@/hooks/use-nav', () => ({ useNav: () => mocks.nav }))
vi.mock('@/services/api-context', () => ({ useApi: mocks.useApi }))
vi.mock('@/lib/export', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/lib/export')>(),
  downloadBlob: mocks.downloadBlob,
  utcDateStamp: () => '20260501-0102',
}))
vi.mock('@/components/def-hover', () => ({
  DefHover: ({ children, term }: { children?: ReactNode; term: string }) => <span>{children ?? term}</span>,
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
vi.mock('@/components/job-hover-card', () => ({
  JobHoverCard: ({ children }: { children: ReactNode }) => <>{children}</>,
}))
vi.mock('@/components/node-hover-card', () => ({
  NodeHoverCard: ({ children }: { children: ReactNode }) => <>{children}</>,
}))
vi.mock('@/components/time-hover', () => ({
  TimeHover: ({ date }: { date: string }) => <time>{date}</time>,
}))
vi.mock('@/components/ui/hover-card', () => ({
  HoverCard: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  HoverCardTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  HoverCardContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}))
vi.mock('@/routers/verdict/components/log-hero', () => ({
  LogHero: ({ run }: { run: Run }) => <div>Log for {run.id}</div>,
}))

const run = (overrides: Partial<Run> = {}): Run => ({
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
  ...overrides,
})

beforeEach(() => {
  vi.clearAllMocks()
  vi.useFakeTimers()
  mocks.useParams.mockReturnValue({ id: 'node-a' })
  mocks.useNodeRuns.mockReturnValue({ runs: [run()], isLoading: false, error: null })
  mocks.useSWR.mockReturnValue({ data: run(), isLoading: false, error: undefined })
  mocks.useApi.mockReturnValue({ getRun: vi.fn().mockResolvedValue(run()) })
  mocks.clipboardWrite.mockResolvedValue(undefined)
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText: mocks.clipboardWrite },
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('NodeView', () => {
  it('renders node evidence, copies actions, exports formats, and navigates', () => {
    render(<NodeView />)

    expect(screen.getAllByText('node-a').length).toBeGreaterThan(0)
    expect(screen.getByText('Responsible node')).toBeTruthy()
    expect(screen.getAllByText('Xid79')).toHaveLength(3)

    fireEvent.click(screen.getByRole('button', { name: 'Copy node ID' }))
    fireEvent.click(screen.getByRole('button', { name: 'Copy evidence (markdown)' }))
    expect(mocks.clipboardWrite).toHaveBeenCalledTimes(2)

    fireEvent.click(screen.getByRole('button', { name: 'Node evidence (JSON)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Node evidence (YAML)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Node evidence (markdown)' }))
    expect(mocks.downloadBlob).toHaveBeenCalledTimes(3)

    fireEvent.click(screen.getByText('Attribution'))
    fireEvent.click(screen.getByText('team/job'))
    expect(mocks.nav).toHaveBeenCalledWith('fleet')
    expect(mocks.nav).toHaveBeenCalledWith('verdict', { id: 'run-1' })
  })

  it('renders loading and empty evidence states', () => {
    mocks.useNodeRuns.mockReturnValue({ runs: [], isLoading: true, error: null })
    const { rerender } = render(<NodeView />)
    expect(screen.getByText('Loading attributed symptoms...')).toBeTruthy()

    mocks.useNodeRuns.mockReturnValue({ runs: [], isLoading: false, error: null })
    rerender(<NodeView />)
    expect(screen.getByText('No attributed symptoms in window.')).toBeTruthy()
  })
})

describe('VerdictView', () => {
  it('renders loading and missing states', () => {
    mocks.useParams.mockReturnValue({ id: 'run-1' })
    mocks.useSWR.mockReturnValue({ data: undefined, isLoading: true, error: undefined })
    const { rerender } = render(<VerdictView />)
    expect(screen.getByText('Loading execution context...')).toBeTruthy()

    mocks.useSWR.mockReturnValue({ data: undefined, isLoading: false, error: undefined })
    rerender(<VerdictView />)
    expect(screen.getByText('Execution context not found')).toBeTruthy()
  })

  it('renders verdict evidence and executes actions', () => {
    mocks.useParams.mockReturnValue({ id: 'run-1' })
    render(<VerdictView />)

    expect(screen.getByRole('heading', { name: 'Attempt 2 · team/job' })).toBeTruthy()
    expect(screen.getByText('Log for run-1')).toBeTruthy()
    expect(screen.getByText('Cloud operator guidance')).toBeTruthy()
    expect(screen.getByText('Drain the node.')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Copy evidence (markdown)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Copy local link' }))
    expect(mocks.clipboardWrite).toHaveBeenCalledTimes(2)

    fireEvent.click(screen.getByRole('button', { name: 'Verdict + evidence (JSON)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Verdict + evidence (YAML)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Raw log (text)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Evidence (markdown)' }))
    expect(mocks.downloadBlob).toHaveBeenCalledTimes(4)

    fireEvent.click(screen.getByRole('button', { name: 'Other runs on node-a' }))
    expect(mocks.nav).toHaveBeenCalledWith('node', { id: 'node-a' })
  })
})
