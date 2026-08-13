// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { Alert, AlertAction, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import {
  Breadcrumb,
  BreadcrumbEllipsis,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Calendar } from '@/components/ui/calendar'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Toggle } from '@/components/ui/toggle'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'

afterEach(cleanup)

describe('structural UI primitives', () => {
  it('renders alert, badge, button, and card variants', () => {
    render(
      <>
        <Alert variant="destructive">
          <AlertTitle>Problem</AlertTitle>
          <AlertDescription>Details</AlertDescription>
          <AlertAction>Retry</AlertAction>
        </Alert>
        <Badge variant="outline">Status</Badge>
        <Badge asChild><a href="/status">Linked status</a></Badge>
        <Button loading>Save</Button>
        <Button asChild variant="ghost"><a href="/open">Open</a></Button>
        <Card size="sm">
          <CardHeader>
            <CardTitle>Title</CardTitle>
            <CardDescription>Description</CardDescription>
            <CardAction>Action</CardAction>
          </CardHeader>
          <CardContent>Content</CardContent>
          <CardFooter>Footer</CardFooter>
        </Card>
      </>,
    )

    expect(screen.getByRole('alert').getAttribute('data-slot')).toBe('alert')
    expect(screen.getByRole('button', { name: 'Save' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('link', { name: 'Open' }).getAttribute('data-slot')).toBe('button')
    expect(screen.getByText('Content').getAttribute('data-slot')).toBe('card-content')
  })

  it('renders breadcrumb and table semantics', () => {
    render(
      <>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem><BreadcrumbLink href="/">Home</BreadcrumbLink></BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem><BreadcrumbPage>Current</BreadcrumbPage></BreadcrumbItem>
            <BreadcrumbEllipsis />
          </BreadcrumbList>
        </Breadcrumb>
        <Table>
          <TableCaption>Runs</TableCaption>
          <TableHeader><TableRow><TableHead>Job</TableHead></TableRow></TableHeader>
          <TableBody><TableRow><TableCell>job-1</TableCell></TableRow></TableBody>
          <TableFooter><TableRow><TableCell>Total 1</TableCell></TableRow></TableFooter>
        </Table>
      </>,
    )

    expect(screen.getByRole('navigation').getAttribute('aria-label')).toBe('breadcrumb')
    expect(screen.getByRole('table')).toBeTruthy()
    expect(screen.getByText('More')).toBeTruthy()
  })
})

describe('interactive UI primitives', () => {
  it('renders calendar, collapsible, overlay triggers, and scroll area', () => {
    const { container } = render(
      <>
        <Calendar mode="single" selected={new Date(2026, 4, 1)} />
        <Collapsible defaultOpen>
          <CollapsibleTrigger>Toggle details</CollapsibleTrigger>
          <CollapsibleContent>Details</CollapsibleContent>
        </Collapsible>
        <HoverCard><HoverCardTrigger>Hover</HoverCardTrigger><HoverCardContent>Hover content</HoverCardContent></HoverCard>
        <Popover><PopoverTrigger>Open menu</PopoverTrigger><PopoverContent>Menu content</PopoverContent></Popover>
        <ScrollArea className="h-10"><div>Scrollable</div><ScrollBar orientation="horizontal" /></ScrollArea>
        <Separator orientation="vertical" />
      </>,
    )

    expect(container.querySelector('[data-slot="calendar"]')).not.toBeNull()
    expect(screen.getByText('Details')).toBeTruthy()
    expect(screen.getByText('Scrollable')).toBeTruthy()
    expect(screen.getByText('Hover')).toBeTruthy()
    expect(screen.getByText('Open menu')).toBeTruthy()
  })

  it('emits toggle state changes and renders tooltip wiring', () => {
    const onPressedChange = vi.fn()
    const onValueChange = vi.fn()
    render(
      <TooltipProvider>
        <Toggle onPressedChange={onPressedChange}>Single</Toggle>
        <ToggleGroup type="single" onValueChange={onValueChange}>
          <ToggleGroupItem value="one">One</ToggleGroupItem>
        </ToggleGroup>
        <Tooltip><TooltipTrigger>Help</TooltipTrigger><TooltipContent>Help text</TooltipContent></Tooltip>
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Single' }))
    fireEvent.click(screen.getByRole('radio', { name: 'One' }))

    expect(onPressedChange).toHaveBeenCalledWith(true)
    expect(onValueChange).toHaveBeenCalledWith('one')
    expect(screen.getByText('Help')).toBeTruthy()
  })
})
