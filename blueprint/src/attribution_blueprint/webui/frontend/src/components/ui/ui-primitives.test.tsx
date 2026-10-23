// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ComponentType } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { SearchIcon } from 'lucide-react'
import { Alert, AlertAction, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardAction,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Calendar } from '@/components/ui/calendar'
import { Checkbox } from '@/components/ui/checkbox'
import { CheckboxLabel } from '@/components/ui/checkbox-label'
import {
  CodeBlock,
  CodeBlockActions,
  CodeBlockBody,
  CodeBlockCopyButton,
  CodeBlockHeader,
  CodeBlockTitle,
} from '@/components/ui/code-block'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Copyable } from '@/components/ui/copyable'
import { DescriptionList } from '@/components/ui/description-list'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@/components/ui/empty'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { Input } from '@/components/ui/input'
import { InputGroup, InputGroupAddon, InputGroupInput } from '@/components/ui/input-group'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { RouterLink } from '@/components/ui/router-link'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableMessage,
  TableRow,
} from '@/components/ui/table'
import { Toggle } from '@/components/ui/toggle'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { floatingEdge } from '@/lib/utils'

// jsdom has no clipboard; each test decides whether the async API resolves or
// rejects and whether the legacy execCommand fallback then succeeds.
function stubClipboard({ writeText, execCommand }: { writeText: () => Promise<void>; execCommand: () => boolean }) {
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: vi.fn(writeText) } })
  Object.defineProperty(document, 'execCommand', { configurable: true, value: vi.fn(execCommand) })
  return { writeText: navigator.clipboard.writeText, execCommand: document.execCommand }
}

beforeEach(() => {
  // Radix measures popper content; jsdom has no layout.
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  Element.prototype.scrollIntoView = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
  Reflect.deleteProperty(navigator, 'clipboard')
  Reflect.deleteProperty(document, 'execCommand')
})

describe('structural UI primitives', () => {
  it('renders alert, badge, button, and card variants', () => {
    const onSave = vi.fn()
    render(
      <>
        <Alert variant="destructive" role="alert">
          <svg data-testid="alert-icon" />
          <AlertTitle>Problem</AlertTitle>
          <AlertDescription>Details</AlertDescription>
          <AlertAction>Retry</AlertAction>
        </Alert>
        <Alert>
          <AlertTitle>Notice</AlertTitle>
        </Alert>
        <Badge variant="outline">Status</Badge>
        <Badge asChild><button type="button">More nodes</button></Badge>
        <Button loading onClick={onSave}>Save</Button>
        <Button asChild variant="ghost"><a href="/open">Open</a></Button>
        <Card>
          <CardHeader>
            <CardTitle>Title</CardTitle>
            <CardAction>Action</CardAction>
          </CardHeader>
          <CardContent>Content</CardContent>
          <CardFooter>Footer</CardFooter>
        </Card>
      </>,
    )

    // The caller decides whether an alert is announced; a notice without a role stays silent.
    expect(screen.getByRole('alert').getAttribute('data-slot')).toBe('alert')
    expect(screen.getByText('Notice').closest('[data-slot="alert"]')!.hasAttribute('role')).toBe(false)
    // The action takes the column after the text, never an absolute corner over it.
    expect([...screen.getByText('Retry').classList]).toEqual(expect.arrayContaining(['col-start-2', 'group-has-[>svg]/alert:col-start-3']))
    expect(screen.getByText('Retry').className).not.toMatch(/absolute/)
    // No variant means the neutral outline badge, never the house green fill.
    expect(screen.getByRole('button', { name: 'More nodes' }).getAttribute('data-variant')).toBe('outline')
    // Busy, a button is unavailable yet keeps its place in the focus order, and
    // swallows activation (disabled would drop keyboard focus to the page).
    const save = screen.getByRole('button', { name: 'Save' })
    expect(save.hasAttribute('disabled')).toBe(false)
    expect(save.getAttribute('aria-disabled')).toBe('true')
    expect(save.getAttribute('aria-busy')).toBe('true')
    save.focus()
    expect(document.activeElement).toBe(save)
    fireEvent.click(save)
    expect(onSave).not.toHaveBeenCalled()
    // The spinner is hidden, so it neither renames the button nor announces itself.
    expect(save.querySelector('svg')!.getAttribute('aria-hidden')).toBe('true')
    // It sets no size of its own, so the button sizes it as the icon it stands in for and keeps its width.
    expect(save.querySelector('svg')!.getAttribute('class')).not.toMatch(/(^|\s)size-/)
    expect(save.className).toContain("[&_svg:not([class*='size-'])]:size-3.5")
    expect(screen.getByRole('link', { name: 'Open' }).getAttribute('data-slot')).toBe('button')
    expect(screen.getByText('Content').getAttribute('data-slot')).toBe('card-content')
    // The header is one line of the card-header height with its action at the
    // right end, the 13px title at 500, and header and content take the card
    // inset (the spacing tokens in globals.css). A header or footer that wraps
    // keeps 4px above and below each line, room for a control's focus ring.
    const header = screen.getByText('Title').closest('[data-slot="card-header"]')!
    expect([...header.classList]).toEqual(expect.arrayContaining(['flex', 'min-h-card-header', 'flex-wrap', 'items-center', 'px-card-x', 'py-1']))
    expect([...screen.getByText('Title').classList]).toEqual(expect.arrayContaining(['text-sm', 'font-medium']))
    expect(screen.getByText('Action').classList.contains('ml-auto')).toBe(true)
    expect([...screen.getByText('Content').classList]).toEqual(expect.arrayContaining(['px-card-x', 'py-card-y']))
    expect([...screen.getByText('Footer').classList]).toEqual(expect.arrayContaining(['min-h-card-header', 'px-card-x', 'py-1']))
  })

  it('defaults buttons to a neutral outline that never submits and keeps the link variant for anchors', () => {
    const onSubmit = vi.fn((event: { preventDefault: () => void }) => event.preventDefault())
    render(
      <form onSubmit={onSubmit}>
        <Button>Plain</Button>
        <Button variant="default">Refresh</Button>
        <Button asChild variant="link" size="sm"><a href="/runs">Runs</a></Button>
      </form>,
    )

    const plain = screen.getByRole('button', { name: 'Plain' })
    expect(plain.getAttribute('type')).toBe('button')
    expect(plain.getAttribute('data-variant')).toBe('outline')
    // The compact control: 28px, with 14px icons.
    expect([...plain.classList]).toEqual(expect.arrayContaining(['h-7', "[&_svg:not([class*='size-'])]:size-3.5"]))
    expect(plain.className).not.toMatch(/primary/)
    // The green default, a page's one primary action, is asked for by name, at 500,
    // pressing to primary-600/80 so its white label stays AA. Its fill is the
    // shared primary-fill (globals.css), written nowhere else.
    const refresh = screen.getByRole('button', { name: 'Refresh' })
    expect([...refresh.classList]).toEqual(expect.arrayContaining([
      'border-primary-600', 'primary-fill', 'font-medium',
      'not-disabled:not-aria-disabled:hover:bg-primary-600/70', 'not-disabled:not-aria-disabled:active:bg-primary-600/80',
    ]))
    expect(refresh.className).not.toMatch(/bg-primary-700|text-white/)
    fireEvent.click(plain)
    expect(onSubmit).not.toHaveBeenCalled()
    const link = screen.getByRole('link', { name: 'Runs' })
    expect(link.getAttribute('data-variant')).toBe('link')
    // A 24px control has room for the outer ring there.
    expect(link.className).not.toContain('ring-inset')
    // Hover states must match an <a>: the house enabled: prefix never does.
    expect(link.className).toContain('not-disabled:not-aria-disabled:hover:underline')
    expect(link.className).not.toMatch(/(^|\s)underline(\s|$)/)
  })

  it('renders a card as a section with a heading title and a flush body', () => {
    render(
      <Card asChild>
        <section aria-label="Runs">
          <CardHeader className="border-b">
            <CardTitle asChild><h2>Runs</h2></CardTitle>
          </CardHeader>
          <CardContent flush>Table</CardContent>
        </section>
      </Card>,
    )

    const region = screen.getByRole('region', { name: 'Runs' })
    expect(region.getAttribute('data-slot')).toBe('card')
    expect(screen.getByRole('heading', { level: 2, name: 'Runs' }).getAttribute('data-slot')).toBe('card-title')
    // A flush body has no padding of its own, whatever the card inset is.
    const flushBody = [...screen.getByText('Table').classList]
    expect(flushBody).toContain('p-0')
    expect(flushBody.filter((name) => /^p[xytrbl]?-/.test(name))).toEqual(['p-0'])
  })

  it('keeps red for blame and neutral tones for everything else in badges', () => {
    render(
      <>
        <Badge variant="red-subtle">Responsible</Badge>
        <Badge variant="zinc-subtle" size="wrap">a-very-long-recorded-source-name</Badge>
        <Badge variant="dashed">Pruned</Badge>
      </>,
    )

    const blame = screen.getByText('Responsible')
    expect(blame.getAttribute('data-variant')).toBe('red-subtle')
    // 20px, so a badge row of a table stays 28px: set inline, it keeps to the
    // line box's top, where its 12px baseline cannot push the 13px line taller.
    // A badge is a mark beside text wherever it sits, lowered onto that text's
    // lowercase by the rule.
    expect([...blame.classList]).toEqual(expect.arrayContaining(['bg-destructive-900', 'text-destructive-400', 'h-5', 'px-1.5', 'text-xs', 'align-top', 'relative', 'top-(--mark-offset)']))
    // A mark it holds is beside its own label, wherever the badge is: the badge
    // declares its label's offset for what it holds and lowers its icons by it.
    expect([...blame.classList]).toEqual(expect.arrayContaining(['*:[--mark-offset:0.93px]', '[&>svg]:relative', '[&>svg]:top-(--mark-offset)']))
    const source = screen.getByText('a-very-long-recorded-source-name')
    // A wrapping badge is 20px too while its text fits on one line; wrapped,
    // it keeps what it holds to the first line.
    expect([...source.classList]).toEqual(expect.arrayContaining(['bg-neutral-800', 'text-neutral-200', 'whitespace-normal', 'wrap-anywhere', 'min-h-5', 'py-px', 'items-start']))
    expect(source.classList.contains('whitespace-nowrap')).toBe(false)
    expect(source.classList.contains('items-center')).toBe(false)
    expect([...screen.getByText('Pruned').classList]).toEqual(expect.arrayContaining(['border-dashed', 'text-foreground-muted']))
  })

  it('renders breadcrumb and table semantics', () => {
    render(
      <>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>Home</BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem><BreadcrumbPage>Current</BreadcrumbPage></BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <Table layout="fixed">
          <TableHeader><TableRow><TableHead>Job</TableHead></TableRow></TableHeader>
          <TableBody>
            <TableRow><TableCell>job-1</TableCell></TableRow>
            <TableRow><TableCell wrap>a-long-recorded-value</TableCell></TableRow>
            <TableRow><TableCell className="py-card-y">detail</TableCell></TableRow>
          </TableBody>
        </Table>
      </>,
    )

    expect(screen.getByRole('navigation').getAttribute('aria-label')).toBe('breadcrumb')
    // The chevron is a mark between the names, lowered onto their lowercase by the rule.
    expect([...document.querySelector('[data-slot="breadcrumb-separator"]')!.classList]).toEqual(expect.arrayContaining(['relative', 'top-(--mark-offset)']))
    const table = screen.getByRole('table')
    expect(table.classList.contains('table-fixed')).toBe(true)
    // 28px rows: a 12px header at 500 and 13px cells, each inset 12px a side
    // (24px between neighbouring cells' text), the first and last on the card's
    // inset so a flush table lines up with its card header.
    expect([...screen.getByRole('columnheader', { name: 'Job' }).classList]).toEqual(expect.arrayContaining(['h-7', 'px-3', 'first:pl-card-x', 'last:pr-card-x', 'text-xs', 'font-medium']))
    expect([...screen.getByText('job-1').classList]).toEqual(expect.arrayContaining(['px-3', 'py-1', 'first:pl-card-x', 'last:pr-card-x']))
    // A named spacing value replaces the cell's own padding instead of joining it.
    expect([...screen.getByText('detail').classList]).toContain('py-card-y')
    expect([...screen.getByText('detail').classList]).not.toContain('py-1')
    // Every table reads its values in the secondary tier, whichever page it is on.
    expect(table.classList.contains('text-foreground-secondary')).toBe(true)
    // A table wider than its card scrolls sideways within its own container,
    // which keeps 1px under the last row for a control's focus ring, and is a
    // size container, so a message row keeps to the part in view.
    expect([...table.parentElement!.classList]).toEqual(expect.arrayContaining(['overflow-x-auto', 'pb-px', '@container']))
    // Headers are muted and never take the row hover.
    expect(screen.getByRole('columnheader', { name: 'Job' }).classList.contains('text-foreground-muted')).toBe(true)
    expect(table.querySelector('thead')!.className).toContain('[&_tr]:hover:bg-transparent')
    expect(screen.getByText('job-1').classList.contains('whitespace-nowrap')).toBe(true)
    const wrapped = screen.getByText('a-long-recorded-value')
    // It may break anywhere, so no unbroken run of it sets the column's least width.
    expect([...wrapped.classList]).toEqual(expect.arrayContaining(['whitespace-normal', 'wrap-anywhere']))
    expect(wrapped.classList.contains('whitespace-nowrap')).toBe(false)
  })

  it('keeps a table\'s message in view, across every column, however far the table scrolls', () => {
    render(
      <Table className="min-w-[880px]">
        <TableBody><TableMessage colSpan={4}>No runs in this view</TableMessage></TableBody>
      </Table>,
    )
    const message = screen.getByText('No runs in this view')
    expect(message.closest('td')!.getAttribute('colspan')).toBe('4')
    // Stuck at the card's inset, as wide as the container less that inset each side.
    expect([...message.classList]).toEqual(expect.arrayContaining(['sticky', 'left-card-x', 'w-[calc(100cqw-2*var(--spacing-card-x))]']))
    expect(message.closest('tr')!.classList.contains('hover:bg-transparent')).toBe(true)
  })

  it('pairs description terms with their values on one label axis', () => {
    const { container } = render(
      <DescriptionList
        labelWidth="sm"
        items={[
          { key: 'rule', label: 'Rule', value: 'known-xid-event' },
          { key: 'uid', label: 'Attempt UID', value: 'demo-attempt-3' },
        ]}
      />,
    )

    const list = container.querySelector('dl')!
    expect(list.classList.contains('sm:grid-cols-[96px_minmax(min-content,1fr)]')).toBe(true)
    // Under 640px each label stands over its value, the pairs 8px apart.
    expect([...list.classList]).toEqual(expect.arrayContaining(['grid-cols-1', 'gap-y-0.5', 'sm:gap-y-1.5']))
    expect([...list.querySelectorAll('dt')].map((term) => term.classList.contains('mt-1.5'))).toEqual([false, true])
    // A value wraps anywhere, so only a value that cannot wrap sets the values' least width.
    for (const value of list.querySelectorAll('dd')) {
      expect(value.classList.contains('wrap-anywhere')).toBe(true)
    }
    // A 12px term and a body-size value share each row's first baseline.
    expect(list.classList.contains('items-baseline')).toBe(true)
    expect([...list.querySelector('dt')!.classList]).toEqual(expect.arrayContaining(['text-xs', 'text-foreground-muted']))
    expect([...list.querySelectorAll('dt')].map((term) => [term.textContent, term.nextElementSibling?.tagName, term.nextElementSibling?.textContent]))
      .toEqual([['Rule', 'DD', 'known-xid-event'], ['Attempt UID', 'DD', 'demo-attempt-3']])
  })

  it('renders empty states, skeletons and labelled inputs', () => {
    const { container } = render(
      <>
        <Empty size="sm">
          <EmptyHeader>
            <EmptyTitle>No runs in this view</EmptyTitle>
            <EmptyDescription>Widen the date range.</EmptyDescription>
          </EmptyHeader>
        </Empty>
        <Skeleton height="4" width="1/2" />
        <Label htmlFor="from">From</Label>
        <Input id="from" placeholder="2026-09-21" />
        <InputGroup className="w-72">
          <InputGroupAddon><SearchIcon data-testid="search" /></InputGroupAddon>
          <InputGroupInput aria-label="Filter runs" />
        </InputGroup>
      </>,
    )

    const compact = screen.getByText('No runs in this view').closest('[data-slot="empty"]')!
    expect(compact.getAttribute('data-size')).toBe('sm')
    // Compact: 20px above and below, the description a 12px muted caption.
    expect([...compact.classList]).toEqual(expect.arrayContaining(['data-[size=sm]:py-5']))
    expect([...screen.getByText('Widen the date range.').classList]).toEqual(expect.arrayContaining(['group-data-[size=sm]/empty:text-xs', 'group-data-[size=sm]/empty:text-foreground-muted']))
    expect([...container.querySelector('[data-slot="skeleton"]')!.classList]).toEqual(expect.arrayContaining(['h-4', 'w-1/2', 'animate-pulse']))
    expect(screen.getByLabelText('From').getAttribute('placeholder')).toBe('2026-09-21')
    expect(screen.getByText('From').classList.contains('text-foreground-muted')).toBe(true)

    const filter = screen.getByRole('textbox', { name: 'Filter runs' })
    expect(filter.getAttribute('data-slot')).toBe('input-group-control')
    // The group draws the ring; the bare input inside draws none.
    expect(filter.className).toContain('focus-visible:ring-0')
    expect(filter.className).not.toContain('focus-visible:ring-[3px]')
    // The addon leads the input, which gives up its own left inset beside it.
    const addon = screen.getByTestId('search').parentElement!
    expect([...addon.classList]).toEqual(expect.arrayContaining(['order-first', 'pl-2']))
    expect(filter.parentElement!.className).toContain('has-[>[data-slot=input-group-addon]]:[&>input]:pl-1.5')
    // A click on the addon icon focuses the input, as a label would.
    fireEvent.click(screen.getByTestId('search'))
    expect(document.activeElement).toBe(filter)
  })
})

describe('interactive UI primitives', () => {
  it('renders calendar, collapsible and overlay triggers', () => {
    const { container } = render(
      <>
        <Calendar mode="single" selected={new Date(2026, 4, 1)} defaultMonth={new Date(2026, 4, 1)} />
        <Collapsible defaultOpen>
          <CollapsibleTrigger>Toggle details</CollapsibleTrigger>
          <CollapsibleContent>Details</CollapsibleContent>
        </Collapsible>
        <HoverCard><HoverCardTrigger>Hover</HoverCardTrigger><HoverCardContent size="auto">Hover content</HoverCardContent></HoverCard>
        <Popover><PopoverTrigger>Open menu</PopoverTrigger><PopoverContent>Menu content</PopoverContent></Popover>
      </>,
    )

    expect(container.querySelector('[data-slot="calendar"]')).not.toBeNull()
    // A selected day reads as the same kind of choice as a selected segment:
    // the one primary-selected recipe (the green default fill, white text and
    // a 1px inset primary-600 ring), for a single day and a range's ends alike.
    const selected = container.querySelector('[data-selected-single="true"]')!
    expect(selected.textContent).toBe('1')
    expect(selected.className.split(' ')).toEqual(expect.arrayContaining([
      'data-[selected-single=true]:primary-selected',
      'data-[range-start=true]:primary-selected',
      'data-[range-end=true]:primary-selected',
    ]))
    expect(selected.className).not.toMatch(/bg-primary-700|text-white|inset-ring-primary/)
    expect(selected.className).not.toContain('primary-foreground')
    // Days of the neighbouring months step back to the muted tone, so the month's edge shows.
    const day = (date: Date) => container.querySelector(`[data-day="${date.toLocaleDateString()}"]`)!
    expect(day(new Date(2026, 3, 30)).classList.contains('text-foreground-muted')).toBe(true)
    expect(day(new Date(2026, 5, 1)).classList.contains('text-foreground-muted')).toBe(true)
    expect(day(new Date(2026, 4, 2)).classList.contains('text-foreground-muted')).toBe(false)
    expect(screen.getByText('Details')).toBeTruthy()
    expect(screen.getByText('Hover')).toBeTruthy()
    expect(screen.getByText('Open menu')).toBeTruthy()
  })

  it('sizes open hover cards from their presets and leaves a popover to its content', () => {
    const ownerWheel = vi.fn()
    render(
      <>
        <HoverCard open><HoverCardTrigger>Node</HoverCardTrigger><HoverCardContent size="lg">Node summary</HoverCardContent></HoverCard>
        <HoverCard open><HoverCardTrigger>Time</HoverCardTrigger><HoverCardContent size="auto">Time detail</HoverCardContent></HoverCard>
        <div onWheel={ownerWheel}>
          <Popover open><PopoverTrigger>Date</PopoverTrigger><PopoverContent>Calendar</PopoverContent></Popover>
        </div>
      </>,
    )

    const card = screen.getByText('Node summary')
    // Every hover card is inset 12px, as the house DateParser's card is.
    expect([...card.classList]).toEqual(expect.arrayContaining(['w-96', 'p-3', 'bg-background', 'ring-foreground/10']))
    expect([...screen.getByText('Time detail').classList]).toEqual(expect.arrayContaining(['p-3']))
    // No card is wider than the room the window leaves it inside the page's gutter.
    expect(card.classList.contains('max-w-(--radix-hover-card-content-available-width)')).toBe(true)
    expect([...screen.getByText('Time detail').classList]).toEqual(expect.arrayContaining(['w-auto', 'max-w-[min(40rem,var(--radix-hover-card-content-available-width))]']))
    // The calendar inside pads itself, so the panel has no inset of its own.
    expect(screen.getByText('Calendar').className).not.toMatch(/(^|\s)(p|px|py)-(\d|page-|card-)/)
    // React events bubble out of the portal to the owner; the panel keeps its wheel to itself.
    fireEvent.wheel(screen.getByText('Calendar'))
    expect(ownerWheel).not.toHaveBeenCalled()
  })

  it('keeps every floating layer the page gutter from the window edge, no wider than the room left', async () => {
    // Radix takes the collision padding as a prop and reflects it nowhere, so
    // the primitives are imported afresh against layers that record it.
    const padding: Record<string, unknown> = {}
    vi.resetModules()
    vi.doMock('radix-ui', async (importOriginal) => {
      const radix = await importOriginal<typeof import('radix-ui')>()
      const recorded = <P extends { collisionPadding?: unknown }>(name: string, Layer: ComponentType<P>) =>
        function Recorded(props: P) {
          padding[name] = props.collisionPadding
          return <Layer {...props} />
        }
      return {
        ...radix,
        Tooltip: { ...radix.Tooltip, Content: recorded('tooltip', radix.Tooltip.Content) },
        HoverCard: { ...radix.HoverCard, Content: recorded('hover card', radix.HoverCard.Content) },
        Popover: { ...radix.Popover, Content: recorded('popover', radix.Popover.Content) },
        DropdownMenu: { ...radix.DropdownMenu, Content: recorded('menu', radix.DropdownMenu.Content) },
        Select: { ...radix.Select, Content: recorded('select', radix.Select.Content) },
      }
    })
    try {
      const tooltip = await import('@/components/ui/tooltip')
      const hoverCard = await import('@/components/ui/hover-card')
      const popover = await import('@/components/ui/popover')
      const menu = await import('@/components/ui/dropdown-menu')
      const select = await import('@/components/ui/select')
      render(
        <>
          <tooltip.Tooltip open><tooltip.TooltipTrigger>Hint</tooltip.TooltipTrigger><tooltip.TooltipContent>Tip</tooltip.TooltipContent></tooltip.Tooltip>
          <hoverCard.HoverCard open><hoverCard.HoverCardTrigger>Node</hoverCard.HoverCardTrigger><hoverCard.HoverCardContent size="lg">Card</hoverCard.HoverCardContent></hoverCard.HoverCard>
          <popover.Popover open><popover.PopoverTrigger>Date</popover.PopoverTrigger><popover.PopoverContent>Panel</popover.PopoverContent></popover.Popover>
          <menu.DropdownMenu open modal={false}><menu.DropdownMenuTrigger>Export</menu.DropdownMenuTrigger><menu.DropdownMenuContent><menu.DropdownMenuItem>JSON</menu.DropdownMenuItem></menu.DropdownMenuContent></menu.DropdownMenu>
          <select.Select open value="a"><select.SelectTrigger aria-label="Pick"><select.SelectValue /></select.SelectTrigger><select.SelectContent><select.SelectItem value="a">Apple</select.SelectItem></select.SelectContent></select.Select>
        </>,
      )

      expect(padding).toEqual({ tooltip: floatingEdge, 'hover card': floatingEdge, popover: floatingEdge, menu: floatingEdge, select: floatingEdge })
      // Each is capped to the room Radix leaves it within that padding, a fixed width included.
      const layer = (slot: string) => document.querySelector<HTMLElement>(`[data-slot="${slot}"]`)!
      expect(layer('tooltip-content').classList.contains('max-w-[min(var(--container-xs),var(--radix-tooltip-content-available-width))]')).toBe(true)
      expect([...layer('hover-card-content').classList]).toEqual(expect.arrayContaining(['w-96', 'max-w-(--radix-hover-card-content-available-width)']))
      expect([...layer('popover-content').classList]).toEqual(expect.arrayContaining(['w-72', 'max-w-(--radix-popover-content-available-width)']))
      expect(layer('dropdown-menu-content').classList.contains('max-w-(--radix-dropdown-menu-content-available-width)')).toBe(true)
      expect(layer('select-content').classList.contains('max-w-(--radix-select-content-available-width)')).toBe(true)
      // A list opens under its trigger, where the padding holds, not over it.
      expect(document.querySelector('[data-slot="select-content"] [data-position]')!.getAttribute('data-position')).toBe('popper')
    } finally {
      vi.doUnmock('radix-ui')
      vi.resetModules()
    }
  })

  it('emits toggle state changes and renders tooltip wiring', () => {
    const onPressedChange = vi.fn()
    const onValueChange = vi.fn()
    // No provider wraps these: every Tooltip carries its own.
    render(
      <>
        <Toggle onPressedChange={onPressedChange}>Single</Toggle>
        <ToggleGroup type="single" onValueChange={onValueChange}>
          <ToggleGroupItem value="one">One</ToggleGroupItem>
        </ToggleGroup>
        <Tooltip><TooltipTrigger>Help</TooltipTrigger><TooltipContent>Help text</TooltipContent></Tooltip>
        <Tooltip open><TooltipTrigger>Webhook</TooltipTrigger><TooltipContent>https://hooks.example/abc</TooltipContent></Tooltip>
      </>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Single' }))
    fireEvent.click(screen.getByRole('radio', { name: 'One' }))

    expect(onPressedChange).toHaveBeenCalledWith(true)
    expect(onValueChange).toHaveBeenCalledWith('one')
    // Selection takes the NVIDIA accent: a pressed content toggle a green tint,
    // green text and an inset ring; a selected segment the green button fill.
    expect([...screen.getByRole('button', { name: 'Single' }).classList]).toEqual(expect.arrayContaining([
      'aria-pressed:bg-primary-500/10', 'aria-pressed:text-primary-300', 'aria-pressed:inset-ring', 'aria-pressed:inset-ring-primary-500/60',
    ]))
    expect([...screen.getByRole('radio', { name: 'One' }).classList]).toContain('data-[state=on]:primary-selected')
    expect(screen.getByRole('radio', { name: 'One' }).className).not.toMatch(/data-\[state=on\]:(bg|text|inset-ring)-/)
    // Text is 400 like every button: the fill, the inset ring and the text color mark the selection, not weight.
    for (const item of [screen.getByRole('button', { name: 'Single' }), screen.getByRole('radio', { name: 'One' })]) {
      expect(item.className).not.toMatch(/font-(medium|semibold|bold)/)
      expect(item.className).toContain('text-foreground-secondary')
    }
    // Each has one form, so neither takes a variant or size: a content toggle is
    // the borderless 24px one, a segment the outline 28px one sharing its
    // neighbours' borders from 640px; under it the segments stand apart, each
    // with its own border and corners, and the group wraps rather than run off.
    const single = screen.getByRole('button', { name: 'Single' })
    expect([...single.classList]).toEqual(expect.arrayContaining(['h-6', 'bg-transparent']))
    expect(single.classList.contains('border')).toBe(false)
    const segment = screen.getByRole('radio', { name: 'One' })
    expect([...segment.classList]).toEqual(expect.arrayContaining(['h-7', 'border', 'rounded-md', 'sm:rounded-none', 'sm:border-l-0', 'sm:first:border-l', 'sm:first:rounded-l-md', 'sm:last:rounded-r-md']))
    expect([...segment.parentElement!.classList]).toEqual(expect.arrayContaining(['flex-wrap', 'gap-1', 'max-w-full', 'sm:flex-nowrap', 'sm:gap-0']))
    expect([single, segment, segment.parentElement!].map((element) => element.hasAttribute('data-variant') || element.hasAttribute('data-size'))).toEqual([false, false, false])
    expect(screen.getByText('Help')).toBeTruthy()
    expect(screen.getByRole('tooltip').textContent).toBe('https://hooks.example/abc')
    const content = document.querySelector('[data-slot="tooltip-content"]')!
    // A long value wraps within the house 320px width, or the room a narrow window leaves, rather than running on.
    expect([...content.classList]).toEqual(expect.arrayContaining(['max-w-[min(var(--container-xs),var(--radix-tooltip-content-available-width))]', 'break-words', 'bg-background', 'ring-1']))
  })

  it('opens a dropdown menu from the pointer and runs the chosen item', () => {
    const onSelect = vi.fn()
    render(
      <DropdownMenu>
        <DropdownMenuTrigger asChild><Button>Export</Button></DropdownMenuTrigger>
        <DropdownMenuContent className="w-56">
          <DropdownMenuItem onSelect={onSelect}>JSON</DropdownMenuItem>
          <DropdownMenuItem>YAML</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>,
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Export' }), { button: 0, ctrlKey: false })
    const menu = screen.getByRole('menu')
    // The caller's width replaces the trigger-width default.
    expect(menu.className).toContain('w-56')
    expect(menu.className).not.toContain('w-(--radix-dropdown-menu-trigger-width)')
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'JSON' }))
    expect(onSelect).toHaveBeenCalledTimes(1)
  })

  it('opens a viewport dialog whose close button carries the given name', () => {
    const outerSubmit = vi.fn()
    const innerSubmit = vi.fn((event: { preventDefault: () => void }) => event.preventDefault())
    render(
      <form onSubmit={outerSubmit}>
        <Dialog>
          <DialogTrigger asChild><Button>View attribution logic</Button></DialogTrigger>
          <DialogContent closeLabel="Close attribution logic">
            <DialogHeader>
              <DialogTitle>Attribution logic</DialogTitle>
              <DialogDescription>demo/xid-cascade</DialogDescription>
            </DialogHeader>
            <form aria-label="Inspect" onSubmit={innerSubmit} />
          </DialogContent>
        </Dialog>
      </form>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'View attribution logic' }))
    const dialog = screen.getByRole('dialog', { name: 'Attribution logic' })
    // A form inside the portaled dialog never submits the form the dialog was opened from.
    fireEvent.submit(within(dialog).getByRole('form', { name: 'Inspect' }))
    expect(innerSubmit).toHaveBeenCalledTimes(1)
    expect(outerSubmit).not.toHaveBeenCalled()
    // The one dialog here is the near-full-screen panel, centered in the
    // overlay and laid out by its caller edge to edge (no inset of its own).
    // It is a focus target only by script, so it draws no outline of its own.
    expect([...dialog.classList]).toEqual(expect.arrayContaining(['h-[min(880px,92dvh)]', 'max-w-[1440px]', 'flex-col', 'motion-reduce:animate-none', 'outline-none']))
    expect(dialog.className).not.toMatch(/(^|\s)(p|px|py|gap)-(\d|page-|card-)/)
    expect(dialog.parentElement!.classList.contains('items-center')).toBe(true)
    expect([...within(dialog).getByRole('button', { name: 'Close attribution logic' }).classList]).toEqual(expect.arrayContaining(['top-2.5', 'right-2.5']))
    expect(screen.getByText('Attribution logic').classList.contains('text-base')).toBe(true)
    fireEvent.click(within(dialog).getByRole('button', { name: 'Close attribution logic' }))
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('opens a select and lists its options', () => {
    const onValueChange = vi.fn()
    render(
      <Select defaultOpen defaultValue="10" onValueChange={onValueChange}>
        <SelectTrigger aria-label="Runs records per page"><SelectValue /></SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value="10">10 / page</SelectItem>
            <SelectItem value="25">25 / page</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>,
    )

    // An open select hides the rest of the page from assistive technology, its own trigger included.
    const trigger = screen.getByRole('combobox', { name: 'Runs records per page', hidden: true })
    expect([...trigger.classList]).toEqual(expect.arrayContaining(['h-7', 'focus-visible:ring-border-active/80', 'not-disabled:hover:bg-background-active']))
    fireEvent.click(screen.getByRole('option', { name: '25 / page' }))
    expect(onValueChange).toHaveBeenCalledWith('25')
  })

  it('toggles a labelled checkbox from its text', () => {
    const onCheckedChange = vi.fn()
    const { rerender } = render(<CheckboxLabel checked={false} onCheckedChange={onCheckedChange}>Hide passing runs</CheckboxLabel>)

    const checkbox = screen.getByRole('checkbox', { name: 'Hide passing runs' })
    // Unchecked, the box is only its border, so the border must read against the card.
    expect(checkbox.classList.contains('border-neutral-500')).toBe(true)
    fireEvent.click(screen.getByText('Hide passing runs'))
    expect(onCheckedChange).toHaveBeenCalledWith(true)

    rerender(<CheckboxLabel checked onCheckedChange={onCheckedChange}>Hide passing runs</CheckboxLabel>)
    expect(checkbox.getAttribute('aria-checked')).toBe('true')
    // Checked, the box takes the NVIDIA accent with a white check.
    expect([...checkbox.classList]).toEqual(expect.arrayContaining(['data-checked:bg-primary-600', 'data-checked:border-primary-600', 'data-checked:text-white']))
    fireEvent.click(checkbox)
    expect(onCheckedChange).toHaveBeenLastCalledWith(false)

    render(<Checkbox aria-label="Standalone" defaultChecked />)
    expect(screen.getByRole('checkbox', { name: 'Standalone' }).getAttribute('data-state')).toBe('checked')
  })
})

describe('router links', () => {
  function Location() {
    return <output>{useLocation().pathname}</output>
  }

  it('navigates in the app on a plain click unless the caller cancels it', () => {
    const cancel = vi.fn((event: { preventDefault: () => void }) => event.preventDefault())
    render(
      <MemoryRouter initialEntries={['/']}>
        <RouterLink to="/verdict/team%2Fjob">team/job</RouterLink>
        <RouterLink to="/node/node-a" onClick={cancel}>node-a</RouterLink>
        <HoverCard>
          <HoverCardTrigger asChild><RouterLink to="/" variant="foreground" weight="medium">Fleet</RouterLink></HoverCardTrigger>
        </HoverCard>
        <Location />
      </MemoryRouter>,
    )

    const verdict = screen.getByRole('link', { name: 'team/job' })
    expect(verdict.getAttribute('href')).toBe('/verdict/team%2Fjob')
    // Neutral and never underlined at rest; hover turns it green and underlines it.
    expect([...verdict.classList]).toEqual(expect.arrayContaining(['text-foreground-secondary', 'hover:text-primary-400', 'hover:underline']))
    expect(verdict.className).not.toMatch(/(^|\s)(underline|text-primary-\d+)(\s|$)|ring-offset/)

    fireEvent.click(screen.getByRole('link', { name: 'node-a' }))
    expect(cancel).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('status').textContent).toBe('/')
    fireEvent.click(verdict)
    expect(screen.getByRole('status').textContent).toBe('/verdict/team%2Fjob')

    // A Slot wrapper cannot rename the link's slot.
    const fleet = screen.getByRole('link', { name: 'Fleet' })
    expect(fleet.getAttribute('data-slot')).toBe('link')
    // It is the trigger itself (Radix marks its state on it), yet keeps its own slot.
    expect(fleet.getAttribute('data-state')).toBe('closed')
    expect([...fleet.classList]).toEqual(expect.arrayContaining(['text-foreground', 'font-medium', 'hover:text-foreground/80']))
  })
})

describe('copy controls', () => {
  it('confirms a copied value without renaming its button, then resets', async () => {
    vi.useFakeTimers()
    const { writeText } = stubClipboard({ writeText: () => Promise.resolve(), execCommand: () => false })
    render(<Copyable text="node-a" label="Copy node ID" />)

    const button = screen.getByRole('button', { name: 'Copy node ID' })
    expect(screen.getByText('node-a').classList.contains('wrap-anywhere')).toBe(true)
    await act(async () => { fireEvent.click(button) })

    expect(writeText).toHaveBeenCalledWith('node-a')
    expect(button.querySelector('.lucide-check')).not.toBeNull()
    expect(screen.getByText('Copied').getAttribute('aria-live')).toBe('polite')
    expect(screen.getByRole('button', { name: 'Copy node ID' })).toBe(button)

    await act(async () => { vi.advanceTimersByTime(1000) })
    expect(button.querySelector('.lucide-copy')).not.toBeNull()
    expect(screen.queryByText('Copied')).toBeNull()
  })

  it('falls back to the legacy copy command and confirms only a copy that happened', async () => {
    const { execCommand } = stubClipboard({ writeText: () => Promise.reject(new Error('insecure context')), execCommand: () => true })
    const { unmount } = render(
      <CodeBlock code={'{\n  "outcome": "hardware"\n}'}>
        <CodeBlockHeader>
          <CodeBlockTitle id="payload-title">Result summary</CodeBlockTitle>
          <CodeBlockActions><CodeBlockCopyButton label="Copy Result summary as JSON" /></CodeBlockActions>
        </CodeBlockHeader>
        <CodeBlockBody labelledBy="payload-title" className="h-80" />
      </CodeBlock>,
    )

    const region = screen.getByRole('region', { name: 'Result summary' })
    expect(region.classList.contains('h-80')).toBe(true)
    // The one divider is the body's, so a block folded to its header draws none.
    expect(region.classList.contains('border-t')).toBe(true)
    expect(region.parentElement!.querySelector('[data-slot="code-block-header"]')!.className).not.toMatch(/border/)
    const pre = region.querySelector('pre')!
    expect(pre.textContent).toBe('{\n  "outcome": "hardware"\n}')
    expect(pre.classList.contains('font-mono')).toBe(true)
    expect(pre.classList.contains('whitespace-pre-wrap')).toBe(false)
    const copy = screen.getByRole('button', { name: 'Copy Result summary as JSON' })
    // Always shown: never faded out until hover.
    expect(copy.className).not.toMatch(/opacity-0/)
    // Browsers move focus into a field they select (jsdom does not), so the
    // fallback takes it from the button; the button gets it back.
    const select = vi.spyOn(HTMLTextAreaElement.prototype, 'select').mockImplementation(function (this: HTMLTextAreaElement) { this.focus() })
    copy.focus()
    await act(async () => { fireEvent.click(copy) })
    expect(select).toHaveBeenCalledTimes(1)
    expect(document.activeElement).toBe(copy)
    select.mockRestore()
    expect(execCommand).toHaveBeenCalledWith('copy')
    expect(copy.querySelector('.lucide-check')).not.toBeNull()
    expect(screen.getByText('Copied')).toBeTruthy()

    // When neither path copies, the button keeps offering to copy.
    unmount()
    stubClipboard({ writeText: () => Promise.reject(new Error('denied')), execCommand: () => { throw new Error('unsupported') } })
    render(
      <CodeBlock code="FAILED">
        <CodeBlockHeader>
          <CodeBlockTitle id="source-title">Source evidence</CodeBlockTitle>
          <CodeBlockActions><CodeBlockCopyButton label="Copy source evidence" /></CodeBlockActions>
        </CodeBlockHeader>
        <CodeBlockBody labelledBy="source-title" wrap className="max-h-48" />
      </CodeBlock>,
    )
    const source = screen.getByRole('button', { name: 'Copy source evidence' })
    await act(async () => { fireEvent.click(source) })
    expect(source.querySelector('.lucide-check')).toBeNull()
    expect(screen.queryByText('Copied')).toBeNull()
    // It says the copy failed, in the same polite region, under the same name.
    expect(screen.getByText('Copy failed').getAttribute('aria-live')).toBe('polite')
    expect(screen.getByRole('button', { name: 'Copy source evidence' })).toBe(source)
    expect([...screen.getByRole('region', { name: 'Source evidence' }).querySelector('pre')!.classList])
      .toEqual(expect.arrayContaining(['whitespace-pre-wrap', 'break-words']))
    // The textarea used by the fallback does not linger in the page.
    expect(document.querySelector('textarea')).toBeNull()
  })

  it('refuses code block parts outside a code block', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(() => render(<CodeBlockBody labelledBy="missing" />)).toThrow('CodeBlock parts must be rendered inside a CodeBlock')
  })
})
