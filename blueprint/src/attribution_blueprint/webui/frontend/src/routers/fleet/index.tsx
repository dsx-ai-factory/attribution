import { useMemo, useCallback, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { OUTCOMES } from '@/lib/attribution-data'
import { useDashboard } from '@/hooks/use-dashboard'
import { useRunsTable, type PageByOutcome } from '@/hooks/use-runs-table'
import { parseFilter, serializeFilter } from '@/lib/filter-state'
import { detectPreset, isoDate, presetRange, windowLabel, type WindowPreset } from '@/lib/date-window'
import type { OutcomeKey } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { DatePicker } from '@/routers/fleet/components/date-picker'
import { KpiStrip } from '@/routers/fleet/components/kpi-strip'
import { ParetoCard } from '@/routers/fleet/components/pareto'
import { BreakdownByDay } from '@/routers/fleet/components/breakdown-by-day'
import { MoversStrip } from '@/routers/fleet/components/movers-strip'
import { RunsSection } from '@/routers/fleet/components/runs-section'

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
  const { dashboard, isLoading: dashboardLoading } = useDashboard({
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

  return (
    <div>
      {/* Filter bar */}
      <div className="sticky top-0 z-20 flex items-center gap-2.5 bg-background-active border border-border rounded-md px-3 py-2.5 mb-5">
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
        <Button variant="default" size="sm" onClick={() => setSearchParams(serializeFilter(filter), { replace: false })}>
          Refresh
        </Button>
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
        runCount={metrics?.totalRuns ?? 0}
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
      <div id="runs-section">
        <RunsSection
          groups={table.groups}
          totalRuns={table.total}
          visibleRuns={table.visibleRuns}
          filter={filter}
          setFilter={setFilter}
          pageByOutcome={pageByOutcome}
          pageSize={pageSize}
          setPageSize={setPageSize}
          setGroupPage={setGroupPage}
          loading={table.isLoading}
        />
      </div>

      <div className="font-mono text-xs text-zinc-700 mt-6 text-center">
        Attribution v0.3.0 · schema {dashboard?.meta.versions.schema ?? 'unknown'} · window {range ? `${isoDate(range.from)} → ${isoDate(range.to)}` : 'all time'}
      </div>
    </div>
  )
}
