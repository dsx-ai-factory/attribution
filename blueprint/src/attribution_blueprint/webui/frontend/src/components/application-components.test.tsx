// @vitest-environment jsdom
import type { HTMLAttributes, ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { Run } from '@/interfaces/attribution'
import { OutcomeChip, VerdictPill } from '@/components/chip'
import { DefHover } from '@/components/def-hover'
import { ExportMenu } from '@/components/export-menu'
import { JobHoverCard } from '@/components/job-hover-card'
import { LinkTo } from '@/components/link-to'
import { NodeHoverCard } from '@/components/node-hover-card'
import { TimeHover } from '@/components/time-hover'
import { TopBar } from '@/components/top-bar'

const mocks = vi.hoisted(() => ({
  nav: vi.fn(),
  useNodeRuns: vi.fn(),
}))

vi.mock('@/hooks/use-nav', () => ({ useNav: () => mocks.nav }))
vi.mock('@/hooks/use-node-runs', () => ({ useNodeRuns: mocks.useNodeRuns }))
vi.mock('@/components/ui/hover-card', () => ({
  HoverCard: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  HoverCardTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  HoverCardContent: ({ children, ...props }: HTMLAttributes<HTMLDivElement>) => (
    <div {...props}>{children}</div>
  ),
}))
vi.mock('@/components/ui/popover', () => ({
  Popover: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  PopoverTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  PopoverContent: ({ children, ...props }: HTMLAttributes<HTMLDivElement>) => (
    <div {...props}>{children}</div>
  ),
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
  detail: '',
  rootSymptom: { symptom: 'Xid79', rule: 'xid', time: '', line: '' },
  symptoms: [],
  responsibleNodes: ['node-a'],
  pipeline: '1',
  service: '2',
  log: [],
  ...overrides,
})

beforeEach(() => {
  vi.clearAllMocks()
  mocks.useNodeRuns.mockReturnValue({
    runs: [run()],
    isLoading: false,
    error: null,
  })
})

afterEach(cleanup)

describe('outcome and definition components', () => {
  it('renders outcome labels and custom verdict labels', () => {
    render(<><OutcomeChip outcome="hardware" /><VerdictPill outcome="software" label="Custom" /></>)

    expect(screen.getByText('HW')).toBeTruthy()
    expect(screen.getByText('Custom')).toBeTruthy()
  })

  it('supports keyboard disclosure for definitions', () => {
    render(<DefHover term="combined-mtbi" />)
    const trigger = screen.getByRole('button')

    fireEvent.focus(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    fireEvent.keyDown(trigger, { key: 'Escape' })
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    fireEvent.keyDown(trigger, { key: 'Enter' })
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    fireEvent.blur(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
  })
})

describe('navigation and export components', () => {
  it('runs an export action from the menu', () => {
    const onSelect = vi.fn()
    render(<ExportMenu items={[{ label: 'JSON', hint: 'machine-readable', onSelect }]} />)

    fireEvent.click(screen.getByRole('button', { name: 'JSON machine-readable' }))

    expect(onSelect).toHaveBeenCalledTimes(1)
  })

  it('renders named links with encoded route parameters', () => {
    render(
      <MemoryRouter>
        <LinkTo name="verdict" params={{ id: 'team/job' }}>Open verdict</LinkTo>
      </MemoryRouter>,
    )

    expect(screen.getByRole('link').getAttribute('href')).toBe('/verdict/team%2Fjob')
  })

  it('opens a linkable job card by click and keyboard', () => {
    render(<JobHoverCard run={run()} linkable>Job</JobHoverCard>)
    const panel = screen.getByText('Click to open verdict →').parentElement
    expect(panel).not.toBeNull()

    fireEvent.click(panel as HTMLElement)
    fireEvent.keyDown(panel as HTMLElement, { key: 'Enter' })

    expect(mocks.nav).toHaveBeenCalledTimes(2)
    expect(mocks.nav).toHaveBeenCalledWith('verdict', { id: 'run-1' })
    expect(screen.getByText('· 1 responsible')).toBeTruthy()
  })
})

describe('node, time, and shell components', () => {
  it('summarizes attempts and attributed symptoms for a node', () => {
    mocks.useNodeRuns.mockReturnValue({
      runs: [
        run(),
        run({ id: 'run-2', responsibleNodes: ['node-b'], rootSymptom: { symptom: 'Xid48', rule: 'xid', time: '', line: '' } }),
      ],
      isLoading: false,
      error: null,
    })

    render(<NodeHoverCard nodeId="node-a">node-a</NodeHoverCard>)

    expect(screen.getByText('Xid79')).toBeTruthy()
    expect(screen.getByText('15.00h')).toBeTruthy()
    expect(screen.getByText('Click to open node detail →')).toBeTruthy()
  })

  it('renders valid UTC times, custom children, and invalid input', () => {
    const { rerender } = render(<TimeHover date="2026-05-01T01:02:03Z" />)
    expect(screen.getAllByText('2026-05-01 01:02:03').length).toBeGreaterThan(0)
    expect(screen.getAllByText('UTC').length).toBeGreaterThan(0)

    rerender(<TimeHover date="2026-05-01T01:02:03Z" omitSuffix>custom time</TimeHover>)
    expect(screen.getByText('custom time')).toBeTruthy()

    rerender(<TimeHover date="invalid" />)
    expect(screen.getByText('Unavailable')).toBeTruthy()
  })

  it('renders the active route in the top bar', () => {
    const { rerender } = render(<TopBar name="node" />)

    expect(screen.getByAltText('Attribution')).toBeTruthy()
    expect(screen.getByText('NVIDIA Attribution · v0.3.0 · node')).toBeTruthy()
    // A server that names no database leaves the slot out rather than reading 'db · null'.
    expect(screen.queryByText(/^db · /)).toBeNull()

    // An empty name is as unavailable as no name at all; rendering it would leave a bare 'db ·'.
    rerender(<TopBar name="node" database="" />)
    expect(screen.queryByText(/^db · /)).toBeNull()

    rerender(<TopBar name="node" database="demo.db" />)
    expect(screen.getByText('db · demo.db')).toBeTruthy()
  })
})
