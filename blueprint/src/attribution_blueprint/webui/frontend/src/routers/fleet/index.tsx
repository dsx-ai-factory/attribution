// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ArrowRight, Info, RotateCw } from 'lucide-react'
import { OUTCOME_ORDER } from '@/lib/attribution-data'
import { useDashboard } from '@/hooks/use-dashboard'
import { useMeta } from '@/hooks/use-meta'
import { useRunsTable, type PageByOutcome, type RunsTableGroup } from '@/hooks/use-runs-table'
import { useDocumentTitle } from '@/hooks/use-document-title'
import { useStickyBand } from '@/hooks/use-sticky-band'
import { parseFilter, serializeFilter } from '@/lib/filter-state'
import { boundsOutsideWindow, detectPreset, isoDate, presetRange, windowLabel, type WindowPreset } from '@/lib/date-window'
import type { OutcomeKey } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import type { DashboardPayload } from '@/interfaces/meta'
import { Button } from '@/components/ui/button'
import { Alert, AlertAction, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { DatePicker } from '@/routers/fleet/components/date-picker'
import { KpiStrip } from '@/routers/fleet/components/kpi-strip'
import { ParetoCard } from '@/routers/fleet/components/pareto'
import { BreakdownByDay } from '@/routers/fleet/components/breakdown-by-day'
import { MoversStrip } from '@/routers/fleet/components/movers-strip'
import { RunsSection } from '@/routers/fleet/components/runs-section'
import type { FilterChangeOptions } from '@/routers/fleet/components/mtbi-primitives'

const DEFAULT_PAGE_SIZE = 10

// Group pages chosen for one table view, identified by its filter-and-page-size key.
interface GroupPages {
  key: string
  pages: PageByOutcome
}

// Pages chosen under another view are stale: every group of a new view starts on page 1.
function pagesForView(groupPages: GroupPages | null, key: string): PageByOutcome {
  return groupPages?.key === key ? groupPages.pages : {}
}

// The dashboard last loaded for one date window, identified by its key.
interface WindowDashboard {
  key: string
  dashboard: DashboardPayload
}

// The run groups last listed for one filter, identified by its key.
interface FilterGroups {
  key: string
  groups: RunsTableGroup[]
}

// Which groups a filter lists and their totals: what changes between two
// listings of one filter only when its runs do, not when another page is shown.
function groupTotals(groups: RunsTableGroup[]): string {
  return groups.map((group) => `${group.outcome}:${group.total}`).join(',')
}

// A view is its URL filter with the entries in one order, so bookmarked or
// hand-written URLs that carry the same params in another order compare equal.
function canonicalView(params: URLSearchParams): string {
  return [...params.entries()].sort().toString()
}

// A request, made by an action that changes what the run list shows, to bring
// the list into view: the view it was made in and the view it asks for.
interface RunsReveal {
  from: string
  to: string
}

// Brings the Runs section to the top of the viewport; the page's scroll
// padding keeps it clear of the sticky bars.
function scrollToRuns() {
  document.getElementById('runs-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

// Moves focus to the Runs region without scrolling, for an action whose own
// button goes away once it applies.
function focusRuns() {
  document.getElementById('runs-section')?.focus({ preventScroll: true })
}

// A notice about why the run list is empty, with the one action that widens it.
// Informational, so it carries no role of its own: the page's polite status
// region around it announces it.
function NoResultsNotice({
  title,
  detail,
  actionLabel,
  onAction,
}: {
  title: string
  detail: string
  actionLabel?: ReactNode
  onAction?: () => void
}) {
  return (
    <Alert>
      <Info />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{detail}</AlertDescription>
      {actionLabel !== undefined && onAction !== undefined && (
        <AlertAction>
          <Button onClick={onAction}>{actionLabel}</Button>
        </AlertAction>
      )}
    </Alert>
  )
}

export function FleetView() {
  useDocumentTitle('Fleet')
  const [searchParams, setSearchParams] = useSearchParams()
  const filter = useMemo(() => parseFilter(searchParams), [searchParams])
  const view = canonicalView(searchParams)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const [groupPages, setGroupPages] = useState<GroupPages | null>(null)

  const anchor = useMemo(() => {
    const to = filter.to
    if (to) return new Date(`${to}T23:59:59.999Z`)
    return new Date()
  }, [filter.to])

  // null means "all time" — no date params sent to backend, which returns all rows
  const range = useMemo<{ from: Date; to: Date } | null>(() => {
    if (filter.from && filter.to)
      return { from: new Date(filter.from), to: new Date(`${filter.to}T23:59:59.999Z`) }
    return null
  }, [filter.from, filter.to])

  // Presets describe windows ending now, so detection and preset clicks anchor
  // to the current time, never to the selected To date.
  const preset = filter.window === 'custom' ? 'custom' : detectPreset(filter, new Date())
  const effectiveFilter = useMemo(
    () => ({
      ...filter,
      from: range ? isoDate(range.from) : undefined,
      to:   range ? isoDate(range.to)   : undefined,
    }),
    [filter, range],
  )
  const {
    dashboard,
    isLoading: dashboardLoading,
    error: dashboardError,
    mutate: refetchDashboard,
  } = useDashboard({
    from: effectiveFilter.from,
    to: effectiveFilter.to,
    outcome: effectiveFilter.outcome,
  })
  const { mutate: refetchMeta } = useMeta()
  const [refreshing, setRefreshing] = useState(false)
  const [runsReveal, setRunsReveal] = useState<RunsReveal | null>(null)
  const revealedRuns = useRef<RunsReveal | null>(null)
  const windowBand = useRef<HTMLDivElement>(null)
  useStickyBand(windowBand)

  const setFilter = useCallback(
    (next: FleetFilter, { revealRuns = false }: FilterChangeOptions = {}) => {
      const sp = serializeFilter(next)
      const nextView = canonicalView(sp)
      if (revealRuns) setRunsReveal({ from: view, to: nextView })
      // Skip no-op writes so re-clicking a preset that resolves to the same
      // window doesn't push duplicate history entries.
      if (nextView === view) return
      setSearchParams(sp, { replace: false })
    },
    [view, setSearchParams],
  )

  const setPreset = useCallback(
    (p: WindowPreset) => {
      if (p === 'all') {
        setFilter({ ...filter, window: undefined, from: undefined, to: undefined })
        return
      }
      if (p === 'custom') {
        const fallback = isoDate(anchor)
        setFilter({ ...filter, window: 'custom', from: range ? isoDate(range.from) : fallback, to: range ? isoDate(range.to) : fallback })
        return
      }
      const { from, to } = presetRange(p, new Date())
      setFilter({ ...filter, window: undefined, from: isoDate(from), to: isoDate(to) })
    },
    [filter, range, anchor, setFilter],
  )

  const setFrom = useCallback(
    (d: Date | undefined) => {
      if (!d) return
      setFilter({ ...filter, window: 'custom', from: isoDate(d), to: filter.to ?? isoDate(range?.to ?? anchor) })
    },
    [filter, range, anchor, setFilter],
  )

  const setTo = useCallback(
    (d: Date | undefined) => {
      if (!d) return
      setFilter({ ...filter, window: 'custom', from: filter.from ?? isoDate(range?.from ?? anchor), to: isoDate(d) })
    },
    [filter, range, anchor, setFilter],
  )

  const runsFilterKey = JSON.stringify({
    from: effectiveFilter.from,
    to: effectiveFilter.to,
    status: effectiveFilter.status,
    outcome: effectiveFilter.outcome,
    symptom: effectiveFilter.symptom,
    node: effectiveFilter.node,
    rule: effectiveFilter.rule,
    user: effectiveFilter.user,
    q: effectiveFilter.q,
  })
  const tableFilterKey = JSON.stringify({ filter: runsFilterKey, pageSize })
  // Derived during render, so no render ever pairs a new view with the old view's pages.
  const pageByOutcome = pagesForView(groupPages, tableFilterKey)
  // The old view's pages are dropped in the render that leaves it (React's
  // adjust-state-while-rendering), so returning to that view starts on page 1 too.
  if (groupPages !== null && groupPages.key !== tableFilterKey) setGroupPages(null)

  const tableOutcomes = useMemo(() => {
    if (effectiveFilter.outcome) return [effectiveFilter.outcome]
    return OUTCOME_ORDER.filter((outcome) => effectiveFilter.status === 'all' || outcome !== 'pass')
  }, [effectiveFilter.outcome, effectiveFilter.status])
  const table = useRunsTable({
    filter: effectiveFilter,
    outcomes: tableOutcomes,
    pageByOutcome,
    pageSize,
  })
  const setGroupPage = useCallback((outcome: OutcomeKey, page: number) => {
    setGroupPages((prev) => ({
      key: tableFilterKey,
      pages: { ...pagesForView(prev, tableFilterKey), [outcome]: page },
    }))
  }, [tableFilterKey])

  // A view's data is its own: nothing from another view stands in while it
  // loads. The one thing that carries over is what another page of the same
  // filter already said: the groups it lists and their totals, which belong to
  // the filter, not to the page shown. While another page or page size of that
  // filter loads, those groups stay listed (with their pagination, so the
  // control pressed keeps focus) and only their rows wait; a new filter's
  // groups are unknown until its pages arrive. The counts this page states
  // still wait for the fetch, as they always have.
  const [filterGroups, setFilterGroups] = useState<FilterGroups | null>(null)
  if (
    !table.isLoading &&
    table.error === null &&
    (filterGroups === null || filterGroups.key !== runsFilterKey || groupTotals(filterGroups.groups) !== groupTotals(table.groups))
  ) {
    setFilterGroups({ key: runsFilterKey, groups: table.groups })
  }
  const knownGroups = !table.isLoading
    ? table.groups
    : filterGroups?.key === runsFilterKey ? filterGroups.groups : []
  const matchingRuns = !table.isLoading
    ? table.total
    : knownGroups.reduce((sum, group) => sum + group.total, 0)

  // The router commits the view a request asks for later, as a transition, so
  // the request waits in the view it was made in. Reaching the view asked for
  // leaves it waiting there alone; any other view drops it.
  if (runsReveal !== null && view !== runsReveal.from) {
    setRunsReveal(view === runsReveal.to ? { from: view, to: view } : null)
  }
  // Everything above Runs has its final height only once the view asked for
  // is on screen with its own data. A scroll started at the click aims at the
  // old layout, which then grows and leaves Runs below the fold.
  const viewLoaded = !dashboardLoading && !table.isLoading
  useEffect(() => {
    if (runsReveal === null || runsReveal === revealedRuns.current) return
    if (view !== runsReveal.to || !viewLoaded) return
    revealedRuns.current = runsReveal
    scrollToRuns()
  }, [runsReveal, view, viewLoaded])

  // The same holds for the dashboard. The server narrows only its concentration
  // ledgers (by root symptom, responsible node, user and rule) to the outcome
  // chosen (webui db.dashboard_data: ledger_outcome); the rest (the KPI
  // figures, days, breakdowns, the outcome split, movers) belong to the date
  // window, and this page shows them for every outcome at once (the failure
  // toggles, the donut). While another outcome of the same window loads, those
  // window facts stay (so a failure toggle or donut entry just pressed keeps
  // focus) and only the ledgers wait; a new window has no facts until its
  // dashboard arrives.
  const windowKey = JSON.stringify([effectiveFilter.from, effectiveFilter.to])
  const [windowDashboard, setWindowDashboard] = useState<WindowDashboard | null>(null)
  if (
    dashboard !== null &&
    !dashboardLoading &&
    (windowDashboard === null || windowDashboard.key !== windowKey || windowDashboard.dashboard !== dashboard)
  ) {
    setWindowDashboard({ key: windowKey, dashboard })
  }
  const windowFacts = dashboard ?? (windowDashboard?.key === windowKey ? windowDashboard.dashboard : null)
  const windowLoading = windowFacts === null && dashboardLoading
  // A request that failed with nothing of its own to show says so where its
  // figures would be, never as zeros or "no failures": the window's facts once
  // no dashboard of this window is at hand, the concentration ledgers once
  // this outcome's did not arrive. A failed revalidation keeps what it fetched.
  const windowFailed = windowFacts === null && !dashboardLoading && dashboardError !== null
  const ledgersFailed = dashboard === null && !dashboardLoading && dashboardError !== null

  const metrics = windowFacts?.metrics
  const daily = windowFacts?.daily ?? []
  const hwBreakdown = windowFacts?.hwBreakdown ?? { days: [], series: [], total: 0 }
  const swBreakdown = windowFacts?.swBreakdown ?? { days: [], series: [], total: 0 }
  const outcomeLedger = windowFacts?.ledgers.outcome ?? []
  const movers = windowFacts?.movers ?? []
  const symptomLedger = dashboard?.ledgers.symptom ?? []
  const nodeLedger = dashboard?.ledgers.node ?? []
  const userLedger = dashboard?.ledgers.user ?? []
  const ruleLedger = dashboard?.ledgers.rule ?? []
  const categoryCounts = dashboard?.categoryCounts
  const largestTies = dashboard?.largestTies
  const dbTotalRuns = dashboard?.meta.dbTotalRuns ?? 0
  const outsideBounds = boundsOutsideWindow(windowFacts?.meta?.dateBounds ?? null, filter)
  const showDateNotice = metrics?.windowedRuns === 0 && outsideBounds !== null
  const showFilterNotice =
    metrics !== undefined &&
    metrics.windowedRuns > 0 &&
    !table.isLoading &&
    table.error === null &&
    table.total === 0
  const databaseEmpty =
    dashboard !== null && !dashboardLoading && dashboardError === null && dbTotalRuns === 0
  const runCountLabel =
    dashboardError !== null || table.error !== null
      ? 'Run counts unavailable'
      : dashboard !== null && !dashboardLoading && !table.isLoading
        ? `${table.total.toLocaleString()} of ${dbTotalRuns.toLocaleString()} runs in database`
        : null

  // Re-reads everything this page shows from the server (its dashboard, its
  // run pages and the backend meta) for the current view; the button stays
  // busy until all three have answered, so an unchanged result still reads as
  // a refresh that happened. SWR revalidation settles rather than rejects, and
  // keeps showing the previous data while it runs.
  const refresh = async () => {
    setRefreshing(true)
    await Promise.all([refetchDashboard(), table.mutate(), refetchMeta()])
    setRefreshing(false)
  }

  const jumpToBounds = useCallback(() => {
    if (outsideBounds === null) return
    setFilter(
      { ...filter, window: 'custom', from: outsideBounds.min, to: outsideBounds.max },
      { revealRuns: true },
    )
    focusRuns()
  }, [filter, outsideBounds, setFilter])

  const clearRunFilters = useCallback(() => {
    setFilter(
      {
        ...filter,
        outcome: undefined,
        symptom: undefined,
        node: undefined,
        rule: undefined,
        user: undefined,
        q: undefined,
        status: 'all',
      },
      { revealRuns: true },
    )
    focusRuns()
  }, [filter, setFilter])

  return (
    <div className="flex flex-col gap-page-gap">
      {/* The page's name for assistive tech (the tab has its own); on screen
          the window bar leads, right under the app bar. Out of the flow, so it
          takes no gap of the stack. */}
      <h1 className="sr-only">Fleet</h1>

      {/* The window for everything below, held in reach under the app bar: a
          compact bordered bar on the card surface, 42px, its 28px controls
          6px from its top and bottom and on the card inset at its sides, so
          the first control lines up with the cards' content. It sticks in a
          band of page background that takes over the page's top padding and
          side gutters, so stuck it stays as far under the app bar as at rest
          and the cards scroll out of sight behind it. The page's scroll
          padding follows its height, so nothing brought into view lands
          behind it. */}
      <div ref={windowBand} data-stuck-chrome className="sticky top-(--app-bar-height) z-20 -mx-page-x -mt-page-top bg-background-secondary px-page-x pt-page-top">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-md border bg-background px-card-x py-1.5">
          <ToggleGroup
            type="single"
            aria-label="Date window"
            value={preset}
            // Re-clicking the pressed preset emits '' (Radix deselect); re-apply
            // it instead so clicking 30d always re-anchors the window to now.
            onValueChange={(v) => setPreset((v || preset) as WindowPreset)}
          >
            <ToggleGroupItem value="all">All</ToggleGroupItem>
            <ToggleGroupItem value="24h">24h</ToggleGroupItem>
            <ToggleGroupItem value="7d">7d</ToggleGroupItem>
            <ToggleGroupItem value="30d">30d</ToggleGroupItem>
            <ToggleGroupItem value="custom">Custom</ToggleGroupItem>
          </ToggleGroup>
          <DatePicker label="From" value={range?.from} onChange={setFrom} maxDate={range?.to} />
          <DatePicker label="To" value={range?.to} onChange={setTo} minDate={range?.from} />
          <div className="ml-auto flex items-center gap-4">
            {runCountLabel !== null && (
              <span className="text-sm text-foreground-muted tabular-nums">{runCountLabel}</span>
            )}
            {/* The page's one primary action, in the green default. */}
            <Button variant="default" loading={refreshing} onClick={() => void refresh()}>
              {!refreshing && <RotateCw data-icon="inline-start" />}
              Refresh
            </Button>
          </div>
        </div>
      </div>

      {/* Notices and cards form one card stack, as on every page. */}
      <div className="flex flex-col gap-card-gap">
        <div role="status" aria-live="polite" className="contents">
          {databaseEmpty && (
            <NoResultsNotice
              title="This database has no attribution runs"
              detail="Nothing has been attributed yet. See blueprint/docs/quickstart.md to run your first attribution, then refresh."
            />
          )}
          {showDateNotice && outsideBounds !== null && (
            <NoResultsNotice
              title="Workloads exist outside the selected dates"
              detail={`Workloads finished between ${outsideBounds.min} and ${outsideBounds.max}. Nothing in the selected date range has an attribution result.`}
              actionLabel={
                <>
                  View {outsideBounds.min}{' '}
                  <ArrowRight aria-hidden="true" className="size-3.5 text-foreground-muted" />
                  <span className="sr-only">to</span> {outsideBounds.max}
                </>
              }
              onAction={jumpToBounds}
            />
          )}
          {showFilterNotice && (
            <NoResultsNotice
              title="No runs listed in this view"
              detail="This view contains runs, but none of them appear in the run list below."
              actionLabel="Clear run filters"
              onAction={clearRunFilters}
            />
          )}
        </div>

        <KpiStrip
          filter={filter}
          setFilter={setFilter}
          hwMtbi={metrics?.hwMtbi ?? 0}
          combinedMtbi={metrics?.combinedMtbi ?? 0}
          gpuHoursRaw={metrics?.gpuHoursRaw ?? 0}
          daily={daily}
          hwFailures={metrics?.hwFailures ?? 0}
          swFailures={metrics?.swFailures ?? 0}
          mixFailures={metrics?.mixFailures ?? 0}
          combinedFailures={metrics?.combinedFailures ?? 0}
          range={range}
          gpuCount={metrics?.gpuCount ?? 0}
          runCount={metrics?.windowedRuns ?? 0}
          loading={windowLoading}
          failed={windowFailed}
        />

        <ParetoCard
          filter={filter}
          setFilter={setFilter}
          symptomLedger={symptomLedger}
          nodeLedger={nodeLedger}
          userLedger={userLedger}
          ruleLedger={ruleLedger}
          outcomeLedger={outcomeLedger}
          categoryCounts={categoryCounts}
          largestTies={largestTies}
          loading={windowLoading}
          ledgersLoading={dashboardLoading}
          failed={windowFailed}
          ledgersFailed={ledgersFailed}
        />

        <BreakdownByDay
          hardware={hwBreakdown}
          software={swBreakdown}
          filter={filter}
          setFilter={setFilter}
          loading={windowLoading}
          failed={windowFailed}
        />

        {/* The server reports no movers today; the section appears only when it does. */}
        {movers.length > 0 && (
          <MoversStrip
            movers={movers}
            windowLabel={windowLabel(preset, range ?? null)}
            filter={filter}
            setFilter={setFilter}
          />
        )}

        <RunsSection
          groups={knownGroups}
          matchingRuns={matchingRuns}
          visibleRuns={table.visibleRuns}
          filter={filter}
          setFilter={setFilter}
          pageByOutcome={pageByOutcome}
          pageSize={pageSize}
          setPageSize={setPageSize}
          setGroupPage={setGroupPage}
          loading={table.isLoading}
          // A failed revalidation keeps showing the runs already fetched.
          failed={table.error !== null && table.groups.length === 0}
          databaseEmpty={databaseEmpty}
        />
      </div>
    </div>
  )
}
