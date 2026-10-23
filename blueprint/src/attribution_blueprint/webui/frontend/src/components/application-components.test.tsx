// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HTMLAttributes, ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Link2 } from 'lucide-react'
import type { RunSummary } from '@/interfaces/attribution'
import type { Meta } from '@/interfaces/meta'
import { OutcomeBadge, OutcomeDot } from '@/components/outcome'
import { CopyButton } from '@/components/copy-button'
import { ExportMenu } from '@/components/export-menu'
import { Field, Identifier } from '@/components/field'
import { InfoHint } from '@/components/info-hint'
import { LogExcerpt, type LogExcerptLine } from '@/components/log-excerpt'
import { NodeHoverCard } from '@/components/node-hover-card'
import { PageHeader } from '@/components/page-header'
import { SectionCard } from '@/components/section-card'
import { SiteStatusBadge } from '@/components/site-status-badge'
import { TablePagination } from '@/components/table-pagination'
import { TimeHover } from '@/components/time-hover'
import { TopBar } from '@/components/top-bar'
import { Badge } from '@/components/ui/badge'
import { Copyable } from '@/components/ui/copyable'
import { OUTCOMES } from '@/lib/attribution-data'
import { DEFINITIONS } from '@/lib/definitions'
import { shouldActivateRow } from '@/lib/row-activation'
import { cn, floatingEdge, markInRow, markInText } from '@/lib/utils'
import { navigateService } from '@/services/navigate.service'
import globalsCss from '@/styles/globals.css?raw'

const mocks = vi.hoisted(() => ({
  useNodeRuns: vi.fn(),
}))

vi.mock('@/hooks/use-node-runs', () => ({ useNodeRuns: mocks.useNodeRuns }))
vi.mock('@/components/ui/hover-card', () => ({
  HoverCard: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  HoverCardTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  HoverCardContent: ({ children, ...props }: HTMLAttributes<HTMLDivElement>) => (
    <div {...props}>{children}</div>
  ),
}))

const run = (overrides: Partial<RunSummary> = {}): RunSummary => ({
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
  decisionGraphState: 'absent',
  rootSymptom: { symptom: 'Xid79', rule: 'xid', time: '', line: '' },
  symptoms: [],
  responsibleNodes: ['node-a'],
  pipeline: '1',
  service: '2',
  ...overrides,
})

// jsdom has no clipboard; each test decides whether the async API resolves and
// whether the legacy execCommand fallback then succeeds.
function stubClipboard(writeText: () => Promise<void>, execCommand: () => boolean) {
  const write = vi.fn(writeText)
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: write } })
  Object.defineProperty(document, 'execCommand', { configurable: true, value: vi.fn(execCommand) })
  return write
}

beforeEach(() => {
  vi.clearAllMocks()
  // Radix measures popper content; jsdom has no layout.
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  Element.prototype.scrollIntoView = vi.fn()
  mocks.useNodeRuns.mockReturnValue({
    runs: [run()],
    isLoading: false,
    error: null,
  })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
  Reflect.deleteProperty(navigator, 'clipboard')
  Reflect.deleteProperty(document, 'execCommand')
})

describe('outcome and definition components', () => {
  it('renders an outcome as a neutral outline badge whose bordered dot carries the tone', () => {
    render(<><OutcomeBadge outcome="hardware" /><OutcomeBadge outcome="software" label="Custom" /></>)

    const hardware = screen.getByText('Hardware Fault')
    expect(hardware.getAttribute('data-variant')).toBe('outline')
    expect(hardware.getAttribute('data-outcome')).toBe('hardware')
    const dot = hardware.querySelector('[data-slot="outcome-dot"]')!
    expect(dot.getAttribute('data-outcome')).toBe('hardware')
    expect(dot.getAttribute('aria-hidden')).toBe('true')
    expect([...dot.classList]).toEqual(expect.arrayContaining(['border', 'border-(--tone-edge)', 'bg-(--tone)']))
    // The label is neutral text; no class may give it an outcome color.
    expect(hardware.className).not.toMatch(/tone|outcome/)

    const custom = screen.getByText('Custom')
    expect(custom.getAttribute('data-outcome')).toBe('software')
    expect(custom.querySelector('[data-slot="outcome-dot"]')!.getAttribute('data-outcome')).toBe('software')
    expect(screen.queryByText('HW')).toBeNull()
  })

  it('explains a metric from its own info button, by keyboard and after a hover delay', () => {
    vi.useFakeTimers()
    render(<p>Combined MTBI <InfoHint term="combined-mtbi" /></p>)
    const hint = screen.getByRole('button', { name: 'About Combined MTBI' })
    const { def, formula } = DEFINITIONS['combined-mtbi']
    // The definition is the button's description before anything opens.
    const description = () => document.getElementById(hint.getAttribute('aria-describedby')!)?.textContent
    expect(description()).toBe(`${def} Formula: ${formula}`)
    expect(screen.getByRole('button', { name: 'About Combined MTBI', description: `${def} Formula: ${formula}` })).toBe(hint)
    // Hidden, not sr-only, so it is no part of the text around the hint.
    expect(document.getElementById(hint.getAttribute('aria-describedby')!)!.hidden).toBe(true)
    expect(screen.queryByRole('tooltip')).toBeNull()

    // A pointer passing over does not pop it; resting on it for 300ms does.
    fireEvent.pointerMove(hint, { pointerType: 'mouse' })
    act(() => { vi.advanceTimersByTime(299) })
    expect(screen.queryByRole('tooltip')).toBeNull()
    act(() => { vi.advanceTimersByTime(1) })
    expect(screen.getByRole('tooltip').textContent).toBe(`${def}${formula}`)
    fireEvent.keyDown(hint, { key: 'Escape' })
    expect(screen.queryByRole('tooltip')).toBeNull()

    // Focus opens it at once, and the description does not move to the tooltip.
    fireEvent.focus(hint)
    expect(screen.getByRole('tooltip').textContent).toBe(`${def}${formula}`)
    expect(description()).toBe(`${def} Formula: ${formula}`)
    fireEvent.keyDown(hint, { key: 'Escape' })
    expect(screen.queryByRole('tooltip')).toBeNull()
  })
})

describe('navigation and export components', () => {
  it('opens the export menu from its trigger and runs the chosen download', () => {
    const onSelect = vi.fn()
    render(<ExportMenu triggerLabel="Export" items={[{ label: 'Download as JSON', onSelect }, { label: 'Download as Markdown', onSelect: vi.fn() }]} />)

    const trigger = screen.getByRole('button', { name: 'Export' })
    expect(trigger.getAttribute('data-variant')).toBe('outline')
    // Both icons are marked, so the house Button trims the padding on each side.
    expect([...trigger.querySelectorAll('svg')].map((icon) => icon.getAttribute('data-icon'))).toEqual(['inline-start', 'inline-end'])
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false })
    const menu = screen.getByRole('menu')
    // Each item names only the file type, on one line; the panel is as wide as
    // the longest, and never narrower than its trigger.
    expect(within(menu).getAllByRole('menuitem').map((item) => item.textContent)).toEqual(['Download as JSON', 'Download as Markdown'])
    expect(within(menu).getAllByRole('menuitem').every((item) => item.classList.contains('whitespace-nowrap'))).toBe(true)
    expect(menu.className.split(' ')).toEqual(expect.arrayContaining(['w-auto', 'min-w-(--radix-dropdown-menu-trigger-width)']))
    expect(menu.className).not.toMatch(/(^| )(w-\(--radix|min-w-32)/)
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Download as JSON' }))

    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('copies text built at click time and confirms only a copy that happened', async () => {
    vi.useFakeTimers()
    const writeText = stubClipboard(() => Promise.resolve(), () => false)
    const getText = vi.fn(() => '# Verdict')
    render(<CopyButton getText={getText} label="Copy evidence" tooltip="Copies a Markdown summary" />)

    const button = screen.getByRole('button', { name: 'Copy evidence' })
    expect(button.getAttribute('data-variant')).toBe('outline')
    expect(button.getAttribute('data-size')).toBe('default')
    const mark = () => button.querySelector('svg')?.getAttribute('data-icon')
    expect(mark()).toBe('inline-start')
    expect(getText).not.toHaveBeenCalled()
    await act(async () => { fireEvent.click(button) })
    expect(writeText).toHaveBeenCalledWith('# Verdict')
    expect(button.querySelector('.lucide-check')).not.toBeNull()
    // The check keeps the icon's mark, so the padding does not change with it.
    expect(mark()).toBe('inline-start')
    expect(screen.getByText('Copied').getAttribute('aria-live')).toBe('polite')
    // The name, and so the width, never changes.
    expect(screen.getByRole('button', { name: 'Copy evidence' })).toBe(button)
    await act(async () => { vi.advanceTimersByTime(2000) })
    expect(button.querySelector('.lucide-copy')).not.toBeNull()
    expect(screen.queryByText('Copied')).toBeNull()

    // When neither clipboard path copies, the button keeps offering to copy
    // and says, in the same polite region and under the same name, that the
    // copy failed.
    stubClipboard(() => Promise.reject(new Error('denied')), () => false)
    await act(async () => { fireEvent.click(button) })
    expect(button.querySelector('.lucide-check')).toBeNull()
    expect(screen.queryByText('Copied')).toBeNull()
    expect(screen.getByText('Copy failed').getAttribute('aria-live')).toBe('polite')
    expect(screen.getByRole('button', { name: 'Copy evidence' })).toBe(button)
  })

  it('labels every copy button, as the page actions beside it are, with a tooltip only to say more', () => {
    render(<><CopyButton getText={() => window.location.href} label="Copy link" icon={Link2} tooltip="Only opens on this Attribution instance" /><CopyButton getText={() => ''} label="Copy node evidence" /></>)

    // A labelled button needs no tooltip to say what it does.
    fireEvent.focus(screen.getByRole('button', { name: 'Copy node evidence' }))
    expect(screen.queryByRole('tooltip')).toBeNull()

    // Its own icon takes the copy mark's place, marked inline-start like it.
    const button = screen.getByRole('button', { name: 'Copy link' })
    expect(button.getAttribute('data-size')).toBe('default')
    expect(button.textContent).toBe('Copy link')
    expect(button.querySelector('.lucide-link-2')?.getAttribute('data-icon')).toBe('inline-start')
    fireEvent.focus(button)
    expect(screen.getByRole('tooltip').textContent).toBe('Only opens on this Attribution instance')
  })
})

describe('page structure components', () => {
  it('heads a page with its breadcrumb, one title, page actions and a stat strip of facts', () => {
    const { rerender } = render(
      <MemoryRouter>
        <PageHeader
          breadcrumb={[{ label: 'Fleet', to: navigateService.getPath('fleet') }, { label: 'Verdict' }]}
          title="demo/xid-cascade"
          titleAddon={<span>Addon</span>}
          actions={<button type="button">Export</button>}
          meta={[
            { key: 'user', label: 'User', value: 'demo-user' },
            { key: 'gpus', label: 'GPUs', value: <span className="num">16</span> },
          ]}
        />
      </MemoryRouter>,
    )

    const title = screen.getByRole('heading', { level: 1, name: 'demo/xid-cascade' })
    // The compact page title: 16px at 500 on its 24px line, untracked like the house base heading.
    expect([...title.classList]).toEqual(expect.arrayContaining(['inline', 'font-medium', 'break-words']))
    expect([...title.parentElement!.classList]).toEqual(expect.arrayContaining(['text-base', 'leading-6']))
    expect(title.className).not.toMatch(/tracking-/)
    // Its addon shares the title's line, centered on the title's lowercase text, not on the line box.
    const addon = screen.getByText('Addon').parentElement!
    expect(addon.parentElement).toBe(title.parentElement)
    expect([...addon.classList]).toEqual(expect.arrayContaining(['inline-flex', 'align-middle', '-my-1']))
    // The 8px after the title is the title's own, so an addon that starts a line of its own starts at the title's edge.
    expect(title.classList.contains('mr-2')).toBe(true)
    expect([...addon.classList].filter((name) => /^m[lx]-/.test(name))).toEqual([])
    const breadcrumb = within(screen.getByRole('navigation', { name: 'breadcrumb' }))
    expect(breadcrumb.getByRole('link', { name: 'Fleet' }).getAttribute('href')).toBe('/')
    expect(breadcrumb.getByRole('link', { name: 'Fleet' }).getAttribute('data-slot')).toBe('link')
    expect(breadcrumb.getByText('Verdict').getAttribute('aria-current')).toBe('page')
    expect(screen.getByText('Addon')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Export' })).toBeTruthy()
    expect(screen.getAllByRole('term').map((term) => term.textContent)).toEqual(['User', 'GPUs'])
    expect(screen.getAllByRole('definition').map((value) => value.textContent)).toEqual(['demo-user', '16'])
    // A stat strip: each fact a 12px label over its value, split by hairlines,
    // every line flush with the title when it wraps: each fact draws its
    // hairline in the 24px gap before it, and the strip clips it where a line starts.
    const strip = screen.getAllByRole('term')[0].closest('dl')!
    expect([...strip.classList]).toEqual(expect.arrayContaining(['flex', 'flex-wrap', 'gap-x-6']))
    // Its clip leaves 5px, room for an info hint's focus ring, which stands 2px out of its label's line.
    expect([...strip.parentElement!.classList]).toEqual(['-m-[5px]', 'overflow-hidden', 'p-[5px]'])
    const facts = [...strip.children]
    expect(facts.map((fact) => [...fact.children].map((part) => part.tagName))).toEqual([['DT', 'DD'], ['DT', 'DD']])
    expect([...facts[0].classList]).toEqual(expect.arrayContaining(['flex-col', 'relative', 'before:-left-3', 'before:w-px', 'before:bg-border']))
    expect(facts[0].className).not.toMatch(/(^|\s)(px|pl|ml)-/)
    expect([...screen.getAllByRole('term')[0].classList]).toEqual(expect.arrayContaining(['text-xs', 'text-foreground-muted']))
    expect(screen.getAllByRole('definition')[0].classList.contains('text-foreground')).toBe(true)
    expect(strip.getAttribute('aria-busy')).toBeNull()

    // Loading, the labels stay and a placeholder on each value's 20px line stands in for the value.
    rerender(
      <MemoryRouter>
        <PageHeader
          breadcrumb={[{ label: 'Fleet', to: navigateService.getPath('fleet') }, { label: 'Verdict' }]}
          title="demo/xid-cascade"
          meta={[
            { key: 'user', label: 'User', value: 'demo-user' },
            { key: 'gpus', label: 'GPUs', value: <span className="num">16</span> },
          ]}
          metaLoading
        />
      </MemoryRouter>,
    )
    expect(screen.getAllByRole('term').map((term) => term.textContent)).toEqual(['User', 'GPUs'])
    expect(screen.getAllByRole('definition').map((value) => value.textContent)).toEqual(['', ''])
    expect(screen.getAllByRole('definition').map((value) => value.firstElementChild?.getAttribute('data-slot'))).toEqual(['skeleton', 'skeleton'])
    expect([...screen.getAllByRole('definition')[0].firstElementChild!.classList]).toEqual(expect.arrayContaining(['h-5']))
    expect(screen.getAllByRole('term')[0].closest('dl')!.getAttribute('aria-busy')).toBe('true')

    // A title the record gives is a placeholder on the title's line until it arrives, never an empty heading.
    rerender(<MemoryRouter><PageHeader breadcrumb={[{ label: 'Fleet' }, { label: 'Verdict' }]} /></MemoryRouter>)
    expect(screen.queryByRole('heading')).toBeNull()
    const titleLine = document.querySelector('header')!.children[1]
    expect(titleLine.querySelector('[data-slot="skeleton"]')!.className.split(' ')).toEqual(expect.arrayContaining(['h-6', 'w-64']))

    rerender(<MemoryRouter><PageHeader breadcrumb={[{ label: 'Fleet' }]} title="Fleet" /></MemoryRouter>)
    expect(screen.queryByRole('term')).toBeNull()
    expect(document.querySelector('a')).toBeNull()
    expect(screen.getByText('Fleet', { selector: '[data-slot="breadcrumb-page"]' }).getAttribute('aria-current')).toBe('page')
  })

  it('makes each section card a region named by its title alone', () => {
    render(
      <>
        <SectionCard title="Runs" count="10" action={<button type="button">Export</button>} flush>
          <table aria-label="Runs table" />
        </SectionCard>
        <SectionCard title="Nodes" count="2 in attempt · 2 responsible" description="Every node the attempt names.">
          <ul aria-label="Nodes" />
        </SectionCard>
        <SectionCard title="Matched lines" flush fill>
          <div data-testid="well" />
        </SectionCard>
      </>,
    )

    const runs = screen.getByRole('region', { name: 'Runs' })
    expect(within(runs).getByRole('heading', { level: 2, name: 'Runs' })).toBeTruthy()
    expect(within(runs).getByText('10').classList.contains('tabular-nums')).toBe(true)
    expect(within(runs).getByRole('button', { name: 'Export' }).closest('[data-slot="card-action"]')).not.toBeNull()
    expect(within(runs).getByRole('table').closest('[data-slot="card-content"]')?.classList.contains('p-0')).toBe(true)
    expect(runs.querySelector('[data-slot="card-header"]')?.classList.contains('border-b')).toBe(true)

    const nodes = screen.getByRole('region', { name: 'Nodes' })
    // A summary of counts is the count beside the title, one line, as its own paragraph; prose goes below.
    const heading = within(nodes).getByRole('heading', { level: 2, name: 'Nodes' })
    const summary = within(nodes).getByText('2 in attempt · 2 responsible')
    expect(summary.tagName).toBe('P')
    expect(heading.nextElementSibling).toBe(summary)
    // What the section shows is a hint in the header line, not a second line under it.
    const hint = within(nodes).getByRole('button', { name: 'About Nodes', description: 'Every node the attempt names.' })
    expect(hint.closest('[data-slot="card-header"]')).toBe(heading.parentElement)
    expect(summary.nextElementSibling?.contains(hint)).toBe(true)
    expect([...nodes.querySelector('[data-slot="card-content"]')!.classList]).toEqual(expect.arrayContaining(['px-card-x', 'py-card-y']))

    // A filling body takes the rest of a card stretched in its grid row, as a column its content grows into; others keep their height.
    const body = (name: string) => [...screen.getByRole('region', { name }).querySelector('[data-slot="card-content"]')!.classList]
    expect(body('Matched lines')).toEqual(expect.arrayContaining(['p-0', 'flex', 'flex-1', 'flex-col']))
    expect(screen.getByTestId('well').parentElement?.getAttribute('data-slot')).toBe('card-content')
    expect(body('Runs')).not.toContain('flex-1')
    expect(body('Nodes')).not.toContain('flex-1')
  })

  it('shows a section with nothing to show as one titled and described empty state', () => {
    const { rerender } = render(
      <SectionCard title="Nodes" empty={{ title: 'No recorded nodes', description: 'No nodes recorded for this attempt.' }}>
        <ul aria-label="Nodes" />
      </SectionCard>,
    )

    const section = screen.getByRole('region', { name: 'Nodes' })
    // The empty state stands in for the body on a flush body, so its own inset is the only one.
    expect(within(section).queryByRole('list')).toBeNull()
    const empty = section.querySelector<HTMLElement>('[data-slot="empty"]')!
    expect(empty.getAttribute('data-size')).toBe('sm')
    expect(empty.closest('[data-slot="card-content"]')?.classList.contains('p-0')).toBe(true)
    expect(within(empty).getByText('No recorded nodes').getAttribute('data-slot')).toBe('empty-title')
    expect(within(empty).getByText('No nodes recorded for this attempt.').getAttribute('data-slot')).toBe('empty-description')
    // Only a section that empties in place announces it.
    expect(within(section).queryByRole('status')).toBeNull()

    rerender(
      <SectionCard title="Failure concentration" empty={{ title: 'No failures to rank', description: 'No failed attempts for this filter.', status: true }} />,
    )
    expect(screen.getByRole('status').textContent).toBe('No failures to rankNo failed attempts for this filter.')
  })

  it('stacks a field as a term over its value', () => {
    render(<dl><Field label="Rule">known-xid-event</Field></dl>)

    const term = screen.getByRole('term')
    expect(term.textContent).toBe('Rule')
    expect([...term.classList]).toEqual(expect.arrayContaining(['text-xs', 'text-foreground-muted']))
    expect(screen.getByRole('definition').textContent).toBe('known-xid-event')
  })

  it('lets a recorded identifier break after its separators before anywhere else', () => {
    const { container } = render(<a href="#run"><Identifier>{'XID-171-UNCORRECTABLE_DRAM_ERROR on gpu01.rack:3/job-x'}</Identifier></a>)
    const text = container.querySelector('a')!
    // A break may fall after '_', '.', ':' and '/', after which no line breaks
    // on its own; a '-' is left as it is, as lines already break after it.
    // Each break is a zero-width space hidden from assistive tech and from a
    // copy, never a <wbr>, which Chrome names as a space. The parts and their
    // breaks are one inline element.
    const BREAK = '<span aria-hidden="true" class="select-none">\u200b</span>'
    expect(text.innerHTML).toBe(`<span>XID-171-UNCORRECTABLE_${BREAK}DRAM_${BREAK}ERROR on gpu01.${BREAK}rack:${BREAK}3/${BREAK}job-x</span>`)
    // It is named and found as recorded: the breaks add nothing to either.
    expect(screen.getByRole('link', { name: 'XID-171-UNCORRECTABLE_DRAM_ERROR on gpu01.rack:3/job-x' })).toBe(text)
    expect(screen.getByText('XID-171-UNCORRECTABLE_DRAM_ERROR on gpu01.rack:3/job-x')).toBe(text.firstElementChild)
    // In a flex parent, such as a filter chip, it is one item beside the
    // label, never an item per part set apart by the gap.
    const { container: chip } = render(<Badge size="wrap" className="gap-1"><span>Rule:</span><Identifier>{'gpu_fell_off_bus'}</Identifier></Badge>)
    const badge = chip.querySelector('[data-slot="badge"]')!
    expect([...badge.childNodes].map((node) => node.nodeName)).toEqual(['SPAN', 'SPAN'])
    expect(badge.lastElementChild!.textContent!.replaceAll('\u200b', '')).toBe('gpu_fell_off_bus')
    // Nothing to break after, nothing added.
    const { container: plain } = render(<p><Identifier>{'GpuFell'}</Identifier></p>)
    expect(plain.querySelector('p')!.innerHTML).toBe('GpuFell')
  })

  it('pages a table with named controls, an announced range and a page-size select', () => {
    const onPageChange = vi.fn()
    const onPageSizeChange = vi.fn()
    const props = {
      'aria-label': 'Hardware Fault pages',
      label: 'Hardware Fault',
      itemLabel: 'runs',
      pageSize: 10,
      total: 25,
      controlsId: 'hardware-runs',
      onPageChange,
    }
    const { rerender } = render(<TablePagination {...props} page={0} pageSizeOptions={[10, 25, 50]} onPageSizeChange={onPageSizeChange} />)

    const pages = within(screen.getByRole('group', { name: 'Hardware Fault pages' }))
    const range = pages.getByText('1–10 of 25 runs')
    expect(range.getAttribute('aria-live')).toBe('polite')
    const button = (verb: string) => pages.getByRole('button', { name: `${verb} Hardware Fault page` }) as HTMLButtonElement
    // The steps past an end are unavailable but never disabled, so a button
    // that reaches an end keeps the keyboard focus it was pressed with.
    const unavailable = () => ['First', 'Previous', 'Next', 'Last'].map((verb) => button(verb).getAttribute('aria-disabled') === 'true')
    expect(unavailable()).toEqual([true, true, false, false])
    expect(['First', 'Previous', 'Next', 'Last'].some((verb) => button(verb).disabled)).toBe(false)
    expect(button('Next').getAttribute('aria-controls')).toBe('hardware-runs')
    expect(button('Next').getAttribute('data-size')).toBe('icon')
    fireEvent.click(button('First'))
    fireEvent.click(button('Previous'))
    expect(onPageChange).not.toHaveBeenCalled()
    fireEvent.click(button('Next'))
    fireEvent.click(button('Last'))
    expect(onPageChange.mock.calls).toEqual([[1], [2]])

    button('Last').focus()
    rerender(<TablePagination {...props} page={2} pageSizeOptions={[10, 25, 50]} onPageSizeChange={onPageSizeChange} />)
    expect(range.textContent).toBe('21–25 of 25 runs')
    expect(unavailable()).toEqual([false, false, true, true])
    expect(document.activeElement).toBe(button('Last'))
    fireEvent.click(button('Next'))
    fireEvent.click(button('Last'))
    fireEvent.click(button('First'))
    fireEvent.click(button('Previous'))
    expect(onPageChange.mock.calls.slice(2)).toEqual([[0], [1]])

    const size = pages.getByRole('combobox', { name: 'Hardware Fault records per page' })
    expect(size.textContent).toBe('10 / page')
    fireEvent.keyDown(size, { key: 'ArrowDown' })
    fireEvent.click(screen.getByRole('option', { name: '25 / page' }))
    expect(onPageSizeChange).toHaveBeenCalledWith(25)

    rerender(<TablePagination {...props} total={0} page={0} />)
    expect(range.textContent).toBe('0–0 of 0 runs')
    expect(screen.queryByRole('combobox')).toBeNull()
  })

  it('activates a row only for clicks meant for the row itself', () => {
    const activate = vi.fn()
    render(
      <table>
        <tbody>
          <tr onClick={(event) => { if (shouldActivateRow(event)) activate() }}>
            <td>team/job</td>
            <td>
              <button type="button">Open graph</button>
              <a href="#node-a">node-a</a>
              <span role="menuitem">Export</span>
            </td>
            <td>{createPortal(<div>Portaled hint</div>, document.body)}</td>
          </tr>
        </tbody>
      </table>,
    )

    fireEvent.click(screen.getByText('team/job'))
    expect(activate).toHaveBeenCalledTimes(1)
    for (const target of [screen.getByRole('button', { name: 'Open graph' }), screen.getByText('node-a'), screen.getByText('Export'), screen.getByText('Portaled hint')]) {
      fireEvent.click(target)
    }
    expect(activate).toHaveBeenCalledTimes(1)

    // Ending a text selection inside the row is not a click on the row.
    const selection = window.getSelection()!
    selection.selectAllChildren(screen.getByText('team/job'))
    fireEvent.click(screen.getByText('team/job'))
    expect(activate).toHaveBeenCalledTimes(1)
    selection.removeAllRanges()
    fireEvent.click(screen.getByText('team/job'))
    expect(activate).toHaveBeenCalledTimes(2)
  })
})

describe('log excerpt', () => {
  const excerpt = () => document.querySelector<HTMLElement>('[data-slot="log-excerpt"]')!

  it('shows recorded lines verbatim, with their time and the root symptom mark only', () => {
    const lines: LogExcerptLine[] = [
      { key: '0', time: '2026-05-01T00:30:00Z', message: 'NVRM: Xid (PCI:0000:bf:00): 48,' },
      { key: '1', time: '2026-05-01T00:30:01+00:00', message: 'NVRM: Xid (PCI:0000:bf:00): 171,', emphasis: true },
      { key: '2', time: 'not recorded', message: '  indented\tFAILED' },
    ]
    render(<LogExcerpt label="Matched lines" emptyLabel="No matched lines were recorded." lines={lines} />)

    // No level words, line numbers or glyphs the record does not hold.
    expect(excerpt().textContent).not.toMatch(/ERROR|INFO|DEBUG|◆/)
    const rows = [...excerpt().children] as HTMLElement[]
    expect(rows.map((row) => row.getAttribute('data-root'))).toEqual([null, 'true', null])
    const times = rows.map((row) => row.querySelector('time'))
    expect(times.map((time) => time?.getAttribute('datetime'))).toEqual(['2026-05-01T00:30:00Z', '2026-05-01T00:30:01+00:00', undefined])
    // Each time is the app's one timestamp: the full stamp in UTC with its mark,
    // and, as the house stamp is, no tab stop of its own.
    expect(times.map((time) => time?.textContent)).toEqual(['2026-05-01 00:30:00 UTC', '2026-05-01 00:30:01 UTC', undefined])
    expect(times.map((time) => time?.hasAttribute('tabindex'))).toEqual([false, false, undefined])
    expect(within(rows[2]).getByText('Unavailable').classList.contains('sr-only')).toBe(true)
    expect(rows[2].querySelector('code')?.textContent).toBe('  indented\tFAILED')
    // The rows share the well's columns, so the times line up however wide the
    // stamp; under 640px a line's time sits above its message.
    expect(excerpt().className.split(' ')).toEqual(expect.arrayContaining(['grid', 'grid-cols-1', 'sm:grid-cols-[auto_minmax(0,1fr)]']))
    expect(rows.every((row) => row.className.split(' ').includes('grid-cols-subgrid'))).toBe(true)
    // The root line takes the NVIDIA accent (a light tint and a 2px edge),
    // never red, and a neutral label at the row's far end.
    expect(rows[1].className).toContain('data-[root=true]:bg-primary-500/10')
    expect(rows[1].className).toContain('data-[root=true]:shadow-[inset_2px_0_0_var(--color-primary-500)]')
    expect(rows[1].className).not.toMatch(/destructive|red-/)
    const label = within(rows[1]).getByText('Root symptom')
    expect(label.getAttribute('data-variant')).toBe('zinc-subtle')
    expect(label.classList.contains('font-mono')).toBe(false)
    // The message and the label share the row's message column: the message
    // wraps before the label, which stays level with its first line at the
    // row's end, and moves under the message, still at the end, where the
    // message would have under 16 characters beside it.
    const message = label.previousElementSibling!
    expect(message.tagName).toBe('CODE')
    expect(message.textContent).toBe('NVRM: Xid (PCI:0000:bf:00): 171,')
    expect(label.parentElement!.className.split(' ')).toEqual(expect.arrayContaining(['flex', 'flex-wrap', 'items-start']))
    expect(message.className.split(' ')).toEqual(expect.arrayContaining(['font-mono', 'min-w-0', 'grow', 'basis-[16ch]', 'break-words']))
    expect(label.classList.contains('ml-auto')).toBe(true)
    // Read whole, then the label; a copy of the line holds its time and message, not the label.
    expect(rows[1].querySelectorAll('code')).toHaveLength(1)
    expect(label.classList.contains('select-none')).toBe(true)
    expect(label.closest('[aria-hidden="true"]')).toBeNull()
    expect(rows.map((row) => row.children.length)).toEqual([2, 2, 2])
    expect(within(rows[0]).queryByText('Root symptom')).toBeNull()
    // Short enough to show whole: neither a scrolling group nor a tab stop.
    expect(screen.queryByRole('group')).toBeNull()
    expect(excerpt().hasAttribute('tabindex')).toBe(false)
  })

  it('scrolls within itself once the lines outgrow its height, as a named group centered on the root line', () => {
    // The well is sized by the card it fills, not by its lines, so it re-measures whenever its own size changes.
    const observed: Element[] = []
    let resize = () => {}
    vi.stubGlobal('ResizeObserver', class {
      constructor(callback: () => void) { resize = callback }
      observe(element: Element) { observed.push(element) }
      disconnect() {}
    })
    // jsdom has no layout: 20 rows of 20px (plus 8px padding above and below) in a well at its 240px minimum.
    let wellHeight = 240
    const inWell = (element: Element) => element.matches('[data-slot="log-excerpt"]')
    vi.spyOn(Element.prototype, 'scrollHeight', 'get').mockImplementation(function (this: Element) { return inWell(this) ? 20 * 20 + 16 : 0 })
    vi.spyOn(Element.prototype, 'clientHeight', 'get').mockImplementation(function (this: Element) { return inWell(this) ? wellHeight : 0 })
    vi.spyOn(HTMLElement.prototype, 'offsetTop', 'get').mockImplementation(function (this: HTMLElement) { return 8 + 20 * [...this.parentElement!.children].indexOf(this) })
    vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(20)
    const scrollTop = vi.spyOn(Element.prototype, 'scrollTop', 'set')
    const lines = Array.from({ length: 20 }, (_, index): LogExcerptLine => ({ key: String(index), time: '2026-05-01T00:30:00Z', message: `line ${index}`, emphasis: index === 14 }))
    const excerptOf = (next: typeof lines) => <LogExcerpt lines={next} label="Matched lines" emptyLabel="No matched lines were recorded." />
    const { rerender } = render(excerptOf(lines))

    // A focusable group, not a landmark: the section holding the lines is the region of that name.
    const group = screen.getByRole('group', { name: 'Matched lines' })
    expect(group).toBe(excerpt())
    expect(group.tabIndex).toBe(0)
    expect(screen.queryByRole('region')).toBeNull()
    expect(observed).toEqual([excerpt()])
    // It fills the height its column gives it, at least 240px, whatever the lines.
    expect(excerpt().className.split(' ')).toEqual(expect.arrayContaining(['min-h-60', 'flex-1', 'contain-size', 'overflow-y-auto']))
    expect(excerpt().className).not.toMatch(/max-h-/)
    // The root row starts at 8 + 14 × 20 = 288; centered in the 240px well: 288 − (240 − 20) / 2.
    expect(scrollTop).toHaveBeenLastCalledWith(178)

    // A parent re-render passing equal lines in a new array leaves the scroll alone.
    scrollTop.mockClear()
    rerender(excerptOf(lines.map((line) => ({ ...line }))))
    expect(scrollTop).not.toHaveBeenCalled()
    // Other lines are centered afresh: the root row now starts at 8 + 10 × 20 = 208.
    const other = lines.map((line) => ({ ...line, emphasis: line.key === '10' }))
    rerender(excerptOf(other))
    expect(scrollTop).toHaveBeenLastCalledWith(98)

    // Until the reader takes the well in hand, the root line stays centered as the well
    // changes size (the card beside it reflowing): 208 − (280 − 20) / 2. A scroll event
    // alone is not the reader: a layout change that clamps the scroll fires one too.
    fireEvent.scroll(excerpt())
    wellHeight = 280
    act(() => { resize() })
    expect(scrollTop).toHaveBeenLastCalledWith(78)

    // Once the reader moves the lines (wheel, touch, pointer or key), a resize re-measures but leaves their scroll alone.
    for (const [index, input] of [fireEvent.wheel, fireEvent.touchStart, fireEvent.pointerDown, fireEvent.keyDown].entries()) {
      // Each input on fresh lines, which follow their root line again.
      scrollTop.mockClear()
      rerender(excerptOf(other.map((line) => ({ ...line, message: `${line.message} (${index})` }))))
      expect(scrollTop).toHaveBeenCalled()
      input(excerpt())
      scrollTop.mockClear()
      wellHeight = wellHeight === 240 ? 280 : 240
      act(() => { resize() })
      expect(scrollTop).not.toHaveBeenCalled()
      expect(screen.getByRole('group', { name: 'Matched lines' })).toBe(excerpt())
    }

    // A resize that lets every line show (the card beside it grew) turns the group back into plain lines.
    wellHeight = 20 * 20 + 16
    act(() => { resize() })
    expect(screen.queryByRole('group')).toBeNull()
    expect(excerpt().hasAttribute('tabindex')).toBe(false)
    // One that makes them scroll again centers the root line afresh.
    wellHeight = 240
    act(() => { resize() })
    expect(screen.getByRole('group', { name: 'Matched lines' })).toBe(excerpt())
    expect(scrollTop).toHaveBeenLastCalledWith(98)
    // A window resize alone is not a size change of the well.
    scrollTop.mockClear()
    wellHeight = 20 * 20 + 16
    act(() => { window.dispatchEvent(new Event('resize')) })
    expect(scrollTop).not.toHaveBeenCalled()
    expect(screen.getByRole('group', { name: 'Matched lines' })).toBe(excerpt())
  })

  it('says so when no line was recorded', () => {
    render(<LogExcerpt lines={[]} label="Matched lines" emptyLabel="No matched lines were recorded." />)

    // Flush in its card body, the sentence takes the body's own inset.
    const empty = screen.getByText('No matched lines were recorded.')
    expect(empty.tagName).toBe('P')
    expect(empty.className.split(' ')).toEqual(expect.arrayContaining(['px-card-x', 'py-card-y', 'text-foreground-muted']))
    expect(document.querySelector('[data-slot="log-excerpt"]')).toBeNull()
  })
})

describe('node, time, and shell components', () => {
  it('summarizes a node on its card once its runs arrive, and nothing before', () => {
    mocks.useNodeRuns.mockReturnValue({
      runs: [
        run(),
        run({ id: 'run-2', responsibleNodes: ['node-b'], rootSymptom: { symptom: 'Xid48', rule: 'xid', time: '', line: '' } }),
      ],
      isLoading: false,
      error: null,
    })

    const { rerender } = render(<NodeHoverCard nodeId="node-a"><a href="#node-a">node-a</a></NodeHoverCard>)

    // The caller's own link is the trigger, and runs are only requested once the card opens.
    expect(screen.getByRole('link', { name: 'node-a' })).toBeTruthy()
    expect(mocks.useNodeRuns).toHaveBeenCalledWith('node-a', false)
    const facts = () => Object.fromEntries(screen.getAllByRole('term').map((term) => [term.textContent, term.nextElementSibling?.textContent]))
    expect(facts()).toEqual({
      Attempts: '2',
      'Responsible in': '1',
      'Distinct symptoms': '1',
      'GPU-hours on node': '15.00',
      'Top symptom': 'Xid79',
    })
    expect(screen.getByText('15.00').classList.contains('num')).toBe(true)

    // Two symptoms attributed to it once each: as on the node page, none is
    // named its top symptom, since run order would pick one of the tied.
    mocks.useNodeRuns.mockReturnValue({
      runs: [run(), run({ id: 'run-2', rootSymptom: { symptom: 'Xid48', rule: 'xid', time: '', line: '' } })],
      isLoading: false,
      error: null,
    })
    rerender(<NodeHoverCard nodeId="node-a"><a href="#node-a">node-a</a></NodeHoverCard>)
    expect(facts()).toEqual({ Attempts: '2', 'Responsible in': '2', 'Distinct symptoms': '2', 'GPU-hours on node': '15.00' })

    // An attempt that named the node but did not run on it counts, yet adds none of its GPU-hours.
    mocks.useNodeRuns.mockReturnValue({ runs: [run(), run({ id: 'run-3', nodes: 1, nodeIds: ['node-b'] })], isLoading: false, error: null })
    rerender(<NodeHoverCard nodeId="node-a"><a href="#node-a">node-a</a></NodeHoverCard>)
    expect(facts()).toEqual({ Attempts: '2', 'Responsible in': '2', 'Distinct symptoms': '1', 'GPU-hours on node': '7.50', 'Top symptom': 'Xid79' })

    mocks.useNodeRuns.mockReturnValue({ runs: [], isLoading: false, error: null })
    rerender(<NodeHoverCard nodeId="node-a"><a href="#node-a">node-a</a></NodeHoverCard>)
    expect(facts()).toEqual({ Attempts: '0', 'Responsible in': '0', 'Distinct symptoms': '0', 'GPU-hours on node': '0.00' })

    // Loading or failing, the card never reads as a node with no attempts.
    mocks.useNodeRuns.mockReturnValue({ runs: [], isLoading: true, error: null })
    rerender(<NodeHoverCard nodeId="node-a"><a href="#node-a">node-a</a></NodeHoverCard>)
    expect(screen.queryByRole('term')).toBeNull()
    expect(document.querySelector('[aria-busy="true"]')?.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(3)

    mocks.useNodeRuns.mockReturnValue({ runs: [], isLoading: false, error: new Error('503') })
    rerender(<NodeHoverCard nodeId="node-a"><a href="#node-a">node-a</a></NodeHoverCard>)
    expect(screen.queryByRole('term')).toBeNull()
    expect(screen.getByText('Runs for this node could not be loaded.')).toBeTruthy()

    // As on the node page, a retry in flight is loading, and a failed
    // revalidation keeps summarizing the runs already fetched.
    mocks.useNodeRuns.mockReturnValue({ runs: [], isLoading: true, error: new Error('503') })
    rerender(<NodeHoverCard nodeId="node-a"><a href="#node-a">node-a</a></NodeHoverCard>)
    expect(screen.queryByText('Runs for this node could not be loaded.')).toBeNull()
    expect(document.querySelector('[aria-busy="true"]')?.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(3)
    mocks.useNodeRuns.mockReturnValue({ runs: [run()], isLoading: false, error: new Error('503') })
    rerender(<NodeHoverCard nodeId="node-a"><a href="#node-a">node-a</a></NodeHoverCard>)
    expect(screen.queryByText('Runs for this node could not be loaded.')).toBeNull()
    expect(facts()).toMatchObject({ Attempts: '1', 'Responsible in': '1' })
  })

  it('renders a recorded time in UTC, keeps the raw value, and offers the local reading', () => {
    render(<TimeHover date="2026-05-01T01:02:03Z" />)
    const time = document.querySelector<HTMLElement>('time[datetime="2026-05-01T01:02:03Z"]')!
    expect(time.textContent).toBe('2026-05-01 01:02:03 UTC')
    expect(time.querySelector('.num')?.textContent).toBe('2026-05-01 01:02:03')
    // A stamp is read whole: it never breaks across lines.
    expect(time.classList.contains('text-nowrap')).toBe(true)
    // A preview trigger, not a link: no underline at rest or on hover.
    expect(time.className).not.toMatch(/underline|decoration/)
    expect(time.tabIndex).toBe(0)
    time.focus()
    expect(document.activeElement).toBe(time)
    // The local reading is the trigger's description, from a hidden source: it
    // is read once, as the description, never again as text around the time.
    const description = document.getElementById(time.getAttribute('aria-describedby')!)!
    expect(description.hidden).toBe(true)
    expect(description.classList.contains('sr-only')).toBe(false)
    expect(description.textContent).toContain('Local time:')
    expect(description.textContent).toContain(Intl.DateTimeFormat().resolvedOptions().timeZone)
    expect(screen.getByRole('time', { description: description.textContent! })).toBe(time)
    // The reading takes the hover card's own 12px inset (the primitive's
    // test), with no inset of its own, so every hover card reads alike.
    const stamp = screen.getAllByText('2026-05-01 01:02:03').find((element) => !element.closest('time'))!
    const reading = stamp.closest('.grid')!
    expect(reading.parentElement!.className).not.toMatch(/(^|\s)p[xytrbl]?-/)
    // Under 640px each zone stands over its reading, so a long zone name never
    // outruns a narrow card; a stamp never breaks, a zone name may.
    expect([...reading.classList]).toEqual(expect.arrayContaining(['grid-cols-1', 'sm:grid-cols-[auto_auto]']))
    expect([...reading.children].map((cell) => cell.textContent)).toEqual(['UTC', '2026-05-01 01:02:03', expect.any(String), expect.any(String)])
    expect(stamp.classList.contains('text-nowrap')).toBe(true)
    expect(reading.children[2].classList.contains('wrap-anywhere')).toBe(true)
    cleanup()
    render(<table><tbody><tr><td><TimeHover date="2026-05-01T01:02:03Z" /></td></tr></tbody></table>)
    const cell = screen.getByRole('cell')
    expect(cell.textContent).toContain('Local time:')
    expect(screen.getByRole('cell', { name: (name) => name.startsWith('2026-05-01 01:02:03 UTC') && !name.includes('Local time') })).toBe(cell)
    cleanup()

    // Every time reads the same, wherever it is: no caller picks a format,
    // leaves out the UTC mark or restyles the trigger.
    // @ts-expect-error: no format to pick.
    const { container, rerender } = render(<TimeHover date="2026-05-01T01:02:03Z" format="MM-DD HH:mm" />)
    expect(container.querySelector('time')?.textContent).toBe('2026-05-01 01:02:03 UTC')
    // @ts-expect-error: no mark to leave out.
    rerender(<TimeHover date="2026-05-01T01:02:03Z" omitSuffix />)
    expect(container.querySelector('time')?.textContent).toBe('2026-05-01 01:02:03 UTC')
    // @ts-expect-error: no class to restyle it with.
    rerender(<TimeHover date="2026-05-01T01:02:03Z" className="underline" />)
    expect(container.querySelector('time')?.classList.contains('underline')).toBe(false)
    // A time recorded without a zone is UTC, as the server reads it, whatever
    // the reader's zone: read here in UTC+8, where a local reading would differ.
    vi.stubEnv('TZ', 'Asia/Shanghai')
    try {
      expect(new Date(2026, 4, 1).getTimezoneOffset()).toBe(-8 * 60)
      rerender(<TimeHover date="2026-05-01T01:02:03" />)
      expect(container.querySelector('time')?.textContent).toBe('2026-05-01 01:02:03 UTC')
    } finally {
      vi.unstubAllEnvs()
    }

    // A time that does not parse follows the empty-value rule of its slot: in
    // a key-value row it is the muted word; in a table cell the shared muted
    // dash every empty cell has, read as "Unavailable".
    rerender(<TimeHover date="invalid" />)
    const unavailable = screen.getByText('Unavailable')
    expect(unavailable.classList.contains('text-foreground-muted')).toBe(true)
    expect(unavailable.hasAttribute('tabindex')).toBe(false)
    expect(unavailable.hasAttribute('aria-describedby')).toBe(false)
    expect(unavailable.querySelector('[aria-hidden="true"]')).toBeNull()
    expect(container.querySelector('time')).toBeNull()
    rerender(<TimeHover date="invalid" cell />)
    const dash = container.firstElementChild!
    expect(dash.textContent).toBe('—Unavailable')
    expect(dash.classList.contains('text-foreground-muted')).toBe(true)
    expect(dash.querySelector('[aria-hidden="true"]')?.textContent).toBe('—')
    expect(dash.querySelector('.sr-only')?.textContent).toBe('Unavailable')
    expect(container.querySelector('time')).toBeNull()
    // A time that parses reads the same in either slot.
    rerender(<TimeHover date="2026-05-01T01:02:03Z" cell />)
    expect(container.querySelector('time')?.textContent).toBe('2026-05-01 01:02:03 UTC')
    // A line's time in a list of lines reads the same, keeps its local reading,
    // and takes no tab stop of its own.
    rerender(<TimeHover date="2026-05-01T01:02:03Z" cell focusable={false} />)
    const line = container.querySelector('time')!
    expect(line.textContent).toBe('2026-05-01 01:02:03 UTC')
    expect(line.hasAttribute('tabindex')).toBe(false)
    expect(document.getElementById(line.getAttribute('aria-describedby')!)?.textContent).toContain('Local time:')
  })

  it('names the backend in the top bar only as /meta reports it', () => {
    const meta = (versions: Partial<Meta['versions']>, database: string | null = 'demo.db'): Meta => ({
      versions: { schema: '3', pipeline: null, service: null, ...versions },
      dateBounds: { min: null, max: null },
      windowedRuns: 0,
      dbTotalRuns: 0,
      database,
      returnedRuns: 0,
    })
    const bar = () => screen.getByRole('banner')
    const { rerender } = render(<MemoryRouter initialEntries={['/node/a']}><TopBar meta={null} /></MemoryRouter>)

    // The brand appears once, as the way home; its mark is decoration beside the name.
    expect(within(bar()).getByRole('link', { name: 'NVIDIA Attribution' }).getAttribute('href')).toBe('/')
    expect(bar().querySelector('img')?.getAttribute('alt')).toBe('')
    // Before /meta answers there is nothing to name.
    expect(bar().textContent).toBe('NVIDIA Attribution')
    // It stays stuck at the top, so its own controls never scroll the page when focused (globals.css),
    // as tall as the one token the page's scroll padding and sticky band start from.
    expect(bar().hasAttribute('data-stuck-chrome')).toBe(true)
    expect([...bar().classList]).toEqual(expect.arrayContaining(['sticky', 'top-0', 'h-(--app-bar-height)']))

    rerender(<MemoryRouter><TopBar meta={meta({})} /></MemoryRouter>)
    expect(bar().textContent).toBe('NVIDIA AttributionDatabase demo.db · schema 3')
    expect(within(bar()).getByText('3').classList.contains('num')).toBe(true)

    // A version the server states is added; one it leaves null is left out, never read as 'null'.
    rerender(<MemoryRouter><TopBar meta={meta({ pipeline: '0.1.0', service: 'AttributionResponseV2' })} /></MemoryRouter>)
    expect(bar().textContent).toBe(
      'NVIDIA AttributionDatabase demo.db · schema 3 · pipeline 0.1.0 · service AttributionResponseV2',
    )
    // The bar's height is fixed, so the facts keep one line, cut short where
    // the screen is narrow and whole on hover.
    const facts = within(bar()).getByText(/demo\.db/).closest('p')!
    expect(facts.classList.contains('truncate')).toBe(true)
    expect(facts.parentElement!.getAttribute('title')).toBe('demo.db · schema 3 · pipeline 0.1.0 · service AttributionResponseV2')

    // An unnamed database is as unavailable as no name at all.
    rerender(<MemoryRouter><TopBar meta={meta({ schema: null }, '')} /></MemoryRouter>)
    expect(bar().textContent).toBe('NVIDIA Attribution')
    // No health or ingest state: the server reports none.
    expect(bar().textContent).not.toMatch(/healthy|ingesting|v0\.3\.0/)
  })
})

describe('design tokens', () => {
  // The value a custom property is declared with in globals.css, e.g. 'var(--color-teal-500)'.
  const declaration = (name: string) =>
    new RegExp(`^\\s*${name}:\\s*([^;]+);`, 'm').exec(globalsCss)?.[1]

  it('gives every outcome a fill, an edge and a data-outcome tone from the palette in globals.css', () => {
    for (const outcome of Object.keys(OUTCOMES)) {
      const tone = new RegExp(`\\[data-outcome="${outcome}"\\]\\s*\\{([^}]*)\\}`).exec(globalsCss)?.[1]
      expect(tone, outcome).toContain(`--tone: var(--outcome-${outcome});`)
      expect(tone, outcome).toContain(`--tone-edge: var(--outcome-${outcome}-edge);`)
      for (const token of [`--outcome-${outcome}`, `--outcome-${outcome}-edge`]) {
        const step = /^var\((--color-[a-z]+-\d+)\)$/.exec(declaration(token) ?? '')?.[1]
        // A step missing from globals.css would silently resolve to Tailwind's default ladder.
        expect(step && declaration(step), token).toBeTruthy()
      }
    }
    // Outcome colors never become utilities; data-outcome is the only way to paint one.
    expect(globalsCss).not.toMatch(/--color-outcome-/)
    // Nor do the retired abbreviated tokens, their tinted backgrounds or their utilities survive.
    expect(globalsCss).not.toMatch(/--outcome-(hw|sw|mix|usr|und|primary-glow)\b|--outcome-[a-z]+-bg\b|--color-(hw|sw|mix|usr|pass|primary-glow)\b/)
  })

  it('defines the NVIDIA accent fill once, for the primary Button and every selected choice', () => {
    // The primary Button's fill; a selected segment or day is that fill edged
    // by a 1px inset primary-600 ring. Retuning the accent is an edit here.
    const utility = (name: string) =>
      new RegExp(`@utility ${name} \\{\\s*@apply ([^;]+);\\s*\\}`).exec(globalsCss)?.[1]
    expect(utility('primary-fill')).toBe('bg-primary-700/60 text-white')
    expect(utility('primary-selected')).toBe('primary-fill inset-ring inset-ring-primary-600')
  })

  it('scales the reading size once, at the root, to the compact 13px on a 20px line', () => {
    // Every text-sm primitive (buttons, inputs, selects, tables, menus, card titles) follows this one token.
    expect(declaration('--text-sm')).toBe('0.8125rem')
    expect(declaration('--text-sm--line-height')).toBe('1.25rem')
  })

  it('centers every mark beside text on the text\'s lowercase, by one offset', () => {
    // NVIDIA Sans lowercase sits 1.2px below the middle of a 13px text's line
    // box: declared once, and read by every mark set in a row beside text.
    expect(declaration('--mark-offset')).toBe('1.2px')
    expect(markInRow).toBe('relative top-(--mark-offset)')
    // In running text a mark centers on the x-height itself, keeps out of the
    // line's height, and takes no offset, nor do the marks it holds.
    expect(markInText).toBe('align-middle -my-1 [--mark-offset:0px]')
    render(
      <>
        <p>Combined MTBI <InfoHint term="combined-mtbi" /></p>
        <OutcomeDot outcome="hardware" />
        <p>demo-node-a <SiteStatusBadge status="bad" className={markInText} /></p>
        <Copyable text="demo-attempt-3" label="Copy attempt UID" />
      </>,
    )
    // The info hint is a mark in its label's running text, with no nudge of its own.
    const hint = screen.getByRole('button', { name: 'About Combined MTBI' })
    expect([...hint.classList].filter((name) => /^(relative|-?top-|align-|-?my-)/.test(name))).toEqual(['align-middle', '-my-1'])
    // A status dot always sits beside its label, so it is lowered as a mark in a row.
    const dot = document.querySelector('[data-slot="outcome-dot"]')!
    expect([...dot.classList]).toEqual(expect.arrayContaining(['relative', 'top-(--mark-offset)']))
    // Every badge is a mark beside text, placed by the rule itself; in running
    // text its offset is reset, so it centers on the x-height alone.
    const badge = screen.getByText('Responsible')
    expect([...badge.classList]).toEqual(expect.arrayContaining(['relative', 'top-(--mark-offset)', 'align-middle', '-my-1', '[--mark-offset:0px]']))
    expect(badge.classList.contains('align-top')).toBe(false)
    // A copy button beside its value keeps to the value's first line, centered on its lowercase.
    const copy = screen.getByRole('button', { name: 'Copy attempt UID' })
    expect([...copy.classList]).toEqual(expect.arrayContaining(['relative', 'top-(--mark-offset)']))
    expect(copy.parentElement!.classList.contains('items-start')).toBe(true)
  })

  it('keeps floating layers off the window edge by the page gutter', () => {
    // Radix takes the distance in pixels: the page-x spacing at the 16px root.
    expect(declaration('--spacing-page-x')).toBe('1.5rem')
    expect(floatingEdge).toBe(1.5 * 16)
  })

  it('sets the page and card spacing once, as named values every inset reads', () => {
    // Pages 24px from the sides, 16px under the app bar and 48px above their
    // end, 16px between a page's blocks; cards 12px apart, content 16px by
    // 12px in. The 38px header (its divider included) leaves a 28px control
    // 4.5px above and below, room for its 3px focus ring inside the card's
    // clipping edge.
    const spacing = Object.fromEntries(
      [...globalsCss.matchAll(/^\s*--spacing-([a-z-]+):\s*([^;]+);/gm)].map(([, name, value]) => [name, value]),
    )
    expect(spacing).toEqual({
      'page-x': '1.5rem',
      'page-top': '1rem',
      'page-bottom': '3rem',
      'page-gap': '1rem',
      'card-gap': '0.75rem',
      'card-x': '1rem',
      'card-y': '0.75rem',
      'card-header': '2.375rem',
    })
    // cn() knows every one of them as spacing, so a caller's value replaces a
    // primitive's instead of both landing on the element.
    for (const name of Object.keys(spacing)) {
      expect(cn(`p-${name}`, 'p-0'), name).toBe('p-0')
      expect(cn('py-1', `py-${name}`), name).toBe(`py-${name}`)
    }
  })

  it('keeps what is brought into view clear of the sticky bars at the top of the page', () => {
    // Keyboard focus and scrollIntoView stop one card gap under the 48px app
    // bar and a page's own sticky band, whose height the page sets while it
    // is mounted, as a card sits under the one before it. The bar's height is
    // one token, which the bar itself and the band under it take too.
    expect(declaration('--app-bar-height')).toBe('3rem')
    expect(declaration('--stuck-chrome')).toBe('calc(var(--app-bar-height) + var(--page-sticky-height) + --theme(--spacing-card-gap inline))')
    expect(declaration('scroll-padding-top')).toBe('var(--stuck-chrome)')
    expect(declaration('--page-sticky-height')).toBe('0px')
    // A control in those bars gives the band back, so focusing it never scrolls the page.
    expect(/\[data-stuck-chrome\] \*\s*\{\s*scroll-margin-top: calc\(-1 \* var\(--stuck-chrome\)\);/.test(globalsCss)).toBe(true)
  })

  it('names NVIDIA Sans without shipping it, falling back to Arial on its metrics', () => {
    // NVIDIA Sans may not be distributed with open source software: text is
    // set in it only where it is installed, and no face loads a font file.
    expect(declaration('--app-font-sans')).toMatch(/^'NVIDIA Sans', 'NVIDIA Sans Fallback', /)
    const faces = [...globalsCss.matchAll(/@font-face\s*\{([^}]*)\}/g)].map(([, face]) => face)
    expect(faces).toHaveLength(1)
    const [fallback] = faces
    expect(fallback).toContain('font-family: "NVIDIA Sans Fallback";')
    expect(fallback).toContain('font-weight: 400 500;')
    const sources = /src: ([^;]+);/.exec(fallback)?.[1].split(', ')
    expect(sources?.[0]).toBe('local("Arial")')
    expect(sources?.every((source) => /^local\("[^"]+"\)$/.test(source))).toBe(true)
    // NVIDIA Sans's ascent, descent and line gap (980, 280 and 0 per 1000), so
    // text and the marks beside it sit as they do in NVIDIA Sans; Arial keeps
    // its own size, its x-height close to NVIDIA Sans's.
    for (const metric of ['ascent-override: 98%;', 'descent-override: 28%;', 'line-gap-override: 0%;']) {
      expect(fallback).toContain(metric)
    }
    expect(fallback).not.toContain('size-adjust')
  })

  describe('readers', () => {
    let tokens: typeof import('@/lib/tokens')

    // jsdom has neither a canvas nor CSS.supports: each color value paints a fixed pixel instead.
    const stubCanvas = (pixels: Record<string, number[]>) => {
      vi.stubGlobal('CSS', { supports: (_property: string, value: string) => value in pixels })
      const context = {
        fillStyle: '',
        fillRect: vi.fn(),
        getImageData: () => ({ data: Uint8ClampedArray.from(pixels[context.fillStyle]) }),
      }
      return vi
        .spyOn(HTMLCanvasElement.prototype, 'getContext')
        .mockReturnValue(context as unknown as CanvasRenderingContext2D)
    }
    const defineTokens = (values: Record<string, string>) => {
      for (const [name, value] of Object.entries(values)) {
        document.documentElement.style.setProperty(name, value)
      }
    }

    beforeEach(async () => {
      // A fresh module per test, so no color cached by one test leaks into the next.
      vi.resetModules()
      tokens = await import('@/lib/tokens')
    })

    afterEach(() => {
      document.documentElement.removeAttribute('style')
      vi.unstubAllGlobals()
      vi.restoreAllMocks()
    })

    it('throws instead of falling back when a token is not defined', () => {
      // No stylesheet is loaded in jsdom, so nothing resolves.
      expect(() => tokens.readColor('--foreground')).toThrow('CSS token --foreground is not defined')
      expect(() => tokens.outcomeColor('undetermined')).toThrow('CSS token --outcome-undetermined is not defined')
      expect(() => tokens.readFont('--app-font-sans')).toThrow('CSS token --app-font-sans is not defined')
      expect(() => tokens.chartTheme()).toThrow('is not defined')
    })

    it('resolves colors to hex through a 1×1 canvas, once per token', () => {
      defineTokens({
        '--outcome-hardware': '#008471',
        '--foreground-muted': 'oklch(0.71 0.01 286)',
        '--translucent': 'rgb(255 255 255 / 4%)',
        '--radius': '0.25rem',
      })
      const getContext = stubCanvas({
        '#008471': [0, 132, 113, 255],
        'oklch(0.71 0.01 286)': [161, 161, 168, 255],
        'rgb(255 255 255 / 4%)': [255, 255, 255, 10],
      })

      expect(tokens.outcomeColor('hardware')).toBe('#008471')
      expect(tokens.readColor('--foreground-muted')).toBe('#a1a1a8')
      expect(tokens.readColor('--translucent')).toBe('#ffffff0a')

      getContext.mockClear()
      expect(tokens.readColor('--outcome-hardware')).toBe('#008471')
      expect(getContext).not.toHaveBeenCalled()

      expect(() => tokens.readColor('--radius')).toThrow('CSS token --radius is not a color: 0.25rem')
    })

    it('builds the chart theme from tokens: sans chrome, mono numbers, no category palette', () => {
      const sans = "'NVIDIA Sans', sans-serif"
      const mono = "'JetBrains Mono Variable', monospace"
      defineTokens({
        '--app-font-sans': sans,
        '--app-font-mono': mono,
        '--foreground': '#ffffff',
        '--foreground-muted': 'oklch(0.71 0.01 286)',
        '--background': '#121212',
        '--border': 'oklch(0.27 0.01 286)',
        '--chart-grid': 'oklch(0.27 0.01 286)',
      })
      stubCanvas({
        '#ffffff': [255, 255, 255, 255],
        'oklch(0.71 0.01 286)': [161, 161, 168, 255],
        '#121212': [18, 18, 18, 255],
        'oklch(0.27 0.01 286)': [38, 38, 43, 255],
      })

      const theme = tokens.chartTheme()
      expect(tokens.chartFonts()).toEqual({ sans, mono })
      const label = { color: '#a1a1a8', fontFamily: sans, fontSize: 12 }

      expect(theme.textStyle).toEqual(label)
      expect(theme.categoryAxis.axisLabel).toEqual(label)
      expect(theme.categoryAxis.axisLine.lineStyle.color).toBe('#26262b')
      expect(theme.valueAxis.axisLabel).toEqual({ ...label, fontFamily: mono })
      expect(theme.valueAxis.splitLine.lineStyle.color).toBe('#26262b')
      expect(theme.bar.label).toEqual({ color: '#ffffff', fontFamily: mono, fontSize: 12 })
      expect(theme.tooltip).toMatchObject({
        backgroundColor: '#121212',
        borderColor: '#26262b',
        textStyle: { color: '#ffffff', fontFamily: sans, fontSize: 12 },
      })
      // A tooltip stays inside its chart, never wider than it, so it keeps
      // clear of the window's edge wherever the chart sits in the page.
      expect(theme.tooltip).toMatchObject({ confine: true, extraCssText: 'max-width: 100%; white-space: normal; overflow-wrap: anywhere;' })
      expect(theme.tooltip).not.toHaveProperty('appendTo')
      expect(theme).not.toHaveProperty('color')
    })
  })
})
