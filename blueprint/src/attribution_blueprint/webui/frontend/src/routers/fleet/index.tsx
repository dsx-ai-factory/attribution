import { useMemo, useCallback, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { CircleAlert } from 'lucide-react'
import { OUTCOMES } from '@/lib/attribution-data'
import { useDashboard } from '@/hooks/use-dashboard'
import { useRunsTable, type PageByOutcome } from '@/hooks/use-runs-table'
import { parseFilter, serializeFilter } from '@/lib/filter-state'
import { boundsOutsideWindow, detectPreset, isoDate, presetRange, windowLabel, type WindowPreset } from '@/lib/date-window'
import type { OutcomeKey } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { DatePicker } from '@/routers/fleet/components/date-picker'
import { KpiStrip } from '@/routers/fleet/components/kpi-strip'
import { ParetoCard } from '@/routers/fleet/components/pareto'
import { BreakdownByDay } from '@/routers/fleet/components/breakdown-by-day'
import { MoversStrip } from '@/routers/fleet/components/movers-strip'
import { RunsSection } from '@/routers/fleet/components/runs-section'
import { scrollToRuns } from '@/routers/fleet/components/mtbi-primitives'

const OUTCOME_ORDER: OutcomeKey[] = ['hardware', 'software', 'multi', 'user', 'undetermined', 'pass']
const DEFAULT_PAGE_SIZE = 10

function SectionLabel({ children, count }: { children: React.ReactNode; count?: number }) {
  return (
    <div className="font-mono text-xs font-semibold tracking-widest uppercase text-foreground-muted my-5 mb-2.5 flex items-center gap-2.5">
      {children}
      {count !== undefined && (
        <Badge variant="outline" className="font-mono text-foreground-secondary">
          {count}
        </Badge>
      )}
      <span className="flex-1 h-px bg-border" />
    </div>
  )
}

function NoResultsNotice({
  title,
  detail,
  actionLabel,
  onAction,
}: {
  title: string
  detail: string
  actionLabel?: string
  onAction?: () => void
}) {
  return (
    <Alert role="presentation" className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <CircleAlert className="text-primary" />
      <div className="flex flex-col gap-1 min-w-0 flex-1">
        <AlertTitle className="font-mono text-xs tracking-widest uppercase">
          {title}
        </AlertTitle>
        <AlertDescription className="font-mono text-xs">{detail}</AlertDescription>
      </div>
      {actionLabel !== undefined && onAction !== undefined && (
        <Button variant="outline" size="sm" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </Alert>
  )
}

export function FleetView() {
  const [searchParams, setSearchParams] = useSearchParams()
  const filter = useMemo(() => parseFilter(searchParams), [searchParams])
  const [pageSize, setPageSizeState] = useState(DEFAULT_PAGE_SIZE)
  const [pageByOutcome, setPageByOutcome] = useState<PageByOutcome>({})

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
  const { dashboard, isLoading: dashboardLoading, error: dashboardError } = useDashboard({
    from: effectiveFilter.from,
    to: effectiveFilter.to,
    outcome: effectiveFilter.outcome,
  })

  const setFilter = useCallback(
    (next: FleetFilter) => {
      const sp = serializeFilter(next)
      // Skip no-op writes so re-clicking a preset that resolves to the same
      // window doesn't push duplicate history entries. Compare entries
      // order-insensitively: bookmarked or hand-written URLs can carry the
      // same params in a different order than serializeFilter emits.
      const canonical = (p: URLSearchParams) => [...p.entries()].sort().toString()
      if (canonical(sp) === canonical(searchParams)) return
      setSearchParams(sp, { replace: false })
    },
    [searchParams, setSearchParams],
  )

  const setPreset = useCallback(
    (p: WindowPreset) => {
      if (p === 'all') {
        const { from: _f, to: _t, window: _w, ...rest } = filter
        setFilter(rest)
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
    [filter, range, setFilter],
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

  const tableFilterKey = JSON.stringify({
    from: effectiveFilter.from,
    to: effectiveFilter.to,
    status: effectiveFilter.status,
    outcome: effectiveFilter.outcome,
    symptom: effectiveFilter.symptom,
    node: effectiveFilter.node,
    rule: effectiveFilter.rule,
    user: effectiveFilter.user,
    q: effectiveFilter.q,
    pageSize,
  })
  useEffect(() => {
    setPageByOutcome({})
  }, [tableFilterKey])

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
  const setPageSize = useCallback((next: number) => {
    setPageSizeState(next)
    setPageByOutcome({})
  }, [])
  const setGroupPage = useCallback((outcome: OutcomeKey, page: number) => {
    setPageByOutcome((prev) => ({ ...prev, [outcome]: page }))
  }, [])

  const metrics = dashboard?.metrics
  const daily = dashboard?.daily ?? []
  const hwBreakdown = dashboard?.hwBreakdown ?? { days: [], series: [], total: 0 }
  const swBreakdown = dashboard?.swBreakdown ?? { days: [], series: [], total: 0 }
  const symptomLedger = dashboard?.ledgers.symptom ?? []
  const nodeLedger = dashboard?.ledgers.node ?? []
  const userLedger = dashboard?.ledgers.user ?? []
  const ruleLedger = dashboard?.ledgers.rule ?? []
  const outcomeLedger = dashboard?.ledgers.outcome ?? []
  const movers = dashboard?.movers ?? []
  const dbTotalRuns = dashboard?.meta.dbTotalRuns ?? 0
  const outsideBounds = boundsOutsideWindow(dashboard?.meta?.dateBounds ?? null, filter)
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

  const focusRuns = useCallback(() => {
    document.getElementById('runs-section')?.focus({ preventScroll: true })
    scrollToRuns()
  }, [])

  const jumpToBounds = useCallback(() => {
    if (outsideBounds === null) return
    setFilter({ ...filter, window: 'custom', from: outsideBounds.min, to: outsideBounds.max })
    focusRuns()
  }, [filter, outsideBounds, setFilter, focusRuns])

  const clearRunFilters = useCallback(() => {
    setFilter({
      ...filter,
      outcome: undefined,
      symptom: undefined,
      node: undefined,
      rule: undefined,
      user: undefined,
      q: undefined,
      status: 'all',
    })
    focusRuns()
  }, [filter, setFilter, focusRuns])

  return (
    <div>
      {/* Filter bar */}
      <div className="sticky top-0 z-20 mb-5 bg-background flex flex-col gap-2">
        <div className="flex items-center gap-2.5 bg-background-active border border-border rounded-md px-3 py-2.5">
          <ToggleGroup
            type="single"
            value={preset}
            // Re-clicking the pressed preset emits '' (Radix deselect); re-apply
            // it instead so clicking 30d always re-anchors the window to now.
            onValueChange={(v) => setPreset((v || preset) as WindowPreset)}
            variant="outline"
            size="sm"
            className="font-mono"
          >
            <ToggleGroupItem value="all">All</ToggleGroupItem>
            <ToggleGroupItem value="24h">24h</ToggleGroupItem>
            <ToggleGroupItem value="7d">7d</ToggleGroupItem>
            <ToggleGroupItem value="30d">30d</ToggleGroupItem>
            <ToggleGroupItem value="custom">Custom</ToggleGroupItem>
          </ToggleGroup>
          <div className="flex items-center gap-1.5">
            <label className="font-mono text-xs uppercase tracking-wider text-foreground-muted">
              From
            </label>
            <DatePicker value={range?.from} onChange={setFrom} maxDate={range?.to} />
          </div>
          <div className="flex items-center gap-1.5">
            <label className="font-mono text-xs uppercase tracking-wider text-foreground-muted">
              To
            </label>
            <DatePicker value={range?.to} onChange={setTo} minDate={range?.from} />
          </div>
          <div className="flex-1" />
          {runCountLabel !== null && (
            <span className="font-mono text-xs text-foreground-muted whitespace-nowrap">
              {runCountLabel}
            </span>
          )}
          <Button variant="default" size="sm" onClick={() => setSearchParams(serializeFilter(filter), { replace: false })}>
            Refresh
          </Button>
        </div>

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
              actionLabel={`View ${outsideBounds.min} → ${outsideBounds.max}`}
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
      </div>

      {/* KPI strip (compressed top-line) */}
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
        loading={dashboardLoading}
      />

      {/* Pareto */}
      <SectionLabel>Where it's concentrated</SectionLabel>
      <div className="mb-5">
        <ParetoCard
          filter={filter}
          setFilter={setFilter}
          symptomLedger={symptomLedger}
          nodeLedger={nodeLedger}
          userLedger={userLedger}
          ruleLedger={ruleLedger}
          outcomeLedger={outcomeLedger}
          runs={[]}
        />
      </div>

      {/* Failure breakdown by day */}
      <SectionLabel>Failure breakdown by day</SectionLabel>
      <div className="font-mono text-[11px] text-foreground-muted -mt-1 mb-2.5">
        Failed attempts per day, stacked by root symptom · click a segment to filter · best read at 7d / 30d
      </div>
      <div className="mb-5">
        <BreakdownByDay
          hardware={hwBreakdown}
          software={swBreakdown}
          filter={filter}
          setFilter={setFilter}
        />
      </div>

      {/* Movers */}
      <SectionLabel>Movers · {windowLabel(preset, range ?? null)}</SectionLabel>
      {movers.length > 0 && (
        <div className="font-mono text-[11px] text-foreground-muted -mt-1 mb-2.5">
          Ranked by movement within the window ·{' '}
          <span style={{ color: OUTCOMES.hardware.color }}>▲</span> rising (worse) ·{' '}
          <span style={{ color: 'var(--color-primary)' }}>▼</span> falling (better) · vs the earlier half
        </div>
      )}
      <MoversStrip movers={movers} filter={filter} setFilter={setFilter} />

      {/* Unified runs table */}
      <div id="runs-section" role="region" aria-label="Runs" tabIndex={-1} className="scroll-mt-16">
        <RunsSection
          groups={table.groups}
          matchingRuns={table.total}
          visibleRuns={table.visibleRuns}
          filter={filter}
          setFilter={setFilter}
          pageByOutcome={pageByOutcome}
          pageSize={pageSize}
          setPageSize={setPageSize}
          setGroupPage={setGroupPage}
          loading={table.isLoading}
          databaseEmpty={databaseEmpty}
        />
      </div>

      <div className="font-mono text-xs text-zinc-700 mt-6 text-center">
        Attribution v0.3.0 · schema {dashboard?.meta.versions.schema ?? 'unknown'} · window {range ? `${isoDate(range.from)} → ${isoDate(range.to)}` : 'all time'}
      </div>
    </div>
  )
}
