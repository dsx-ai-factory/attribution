import type { ReactNode } from 'react'
import type { DailyBucket, OutcomeKey } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import { withDimension } from '@/lib/filter-state'
import { DefHover } from '@/components/def-hover'
import { OutcomeHoverCard } from '@/routers/fleet/components/outcome-hover-card'
import {
  FailureLink,
  MtbiSparkline,
  scrollToRuns,
} from '@/routers/fleet/components/mtbi-primitives'
import type { DefTerm } from '@/interfaces/definitions'
import { fmtGpuHours, fmtNum } from '@/lib/format'

type CellProps = {
  span: number
  term: DefTerm
  children: ReactNode
  onActivate?: () => void
  selected?: boolean
}

function Cell({ span, term, children, onActivate, selected }: CellProps) {
  const interactive = !!onActivate
  return (
    <div
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={onActivate}
      onKeyDown={(e) => {
        if (!interactive) return
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onActivate?.()
        }
      }}
      style={{ gridColumn: `span ${span} / span ${span}` }}
      className={[
        'flex flex-col gap-1 px-4 py-3 border-r border-border last:border-r-0',
        interactive ? 'cursor-pointer hover:bg-background-active transition' : '',
        interactive ? 'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary' : '',
        selected ? 'bg-primary/5 ring-inset ring-1 ring-primary/40' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <div className="font-mono text-[10px] uppercase tracking-widest text-foreground-muted">
        <DefHover term={term} />
      </div>
      {children}
    </div>
  )
}

export function KpiStrip({
  filter,
  setFilter,
  hwMtbi,
  combinedMtbi,
  gpuHoursRaw,
  daily,
  hwFailures,
  swFailures,
  mixFailures,
  combinedFailures,
  range,
  gpuCount,
  runCount,
  loading = false,
}: {
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
  hwMtbi: number
  combinedMtbi: number
  gpuHoursRaw: number
  daily: DailyBucket[]
  hwFailures: number
  swFailures: number
  mixFailures: number
  combinedFailures: number
  range: { from: Date; to: Date } | null
  gpuCount: number
  runCount: number
  loading?: boolean
}) {
  if (loading) {
    return (
      <div
        className="grid grid-cols-12 bg-card border border-border rounded-md overflow-hidden mb-5"
        style={{ minHeight: 96 }}
        aria-busy="true"
        aria-live="polite"
      >
        {[3, 3, 2, 2, 2].map((span, i) => (
          <div
            key={i}
            style={{ gridColumn: `span ${span} / span ${span}` }}
            className="px-4 py-3 border-r border-border last:border-r-0"
          >
            <div className="h-2.5 w-16 bg-background-muted rounded-sm animate-pulse mb-3" />
            <div className="h-6 w-24 bg-background-muted rounded-sm animate-pulse" />
          </div>
        ))}
      </div>
    )
  }

  // Raw GPU-hours below 1000 render as-is; only at/above 1000 do we switch to "k".
  const gpuHours = fmtGpuHours(gpuHoursRaw)
  // Honest per-day failure-count trend over the window (no fudge).
  const dailyCombined = daily.map((d) => d.hw + d.sw)
  const dailyHw = daily.map((d) => d.hw)

  const onCombinedClick = () => {
    setFilter(withDimension({ ...filter, status: 'failed' }, 'outcome', undefined))
    scrollToRuns()
  }
  const onHwClick = () => {
    setFilter(withDimension({ ...filter, status: 'failed' }, 'outcome', 'hardware'))
    scrollToRuns()
  }

  const combinedActive =
    filter.status === 'failed' &&
    !filter.outcome &&
    !filter.symptom &&
    !filter.node &&
    !filter.rule
  const hwActive = filter.outcome === 'hardware'

  const fmtDate = (d: Date) =>
    `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`

  const failureSlot = (outcome: OutcomeKey, count: number, label: string) => (
    <OutcomeHoverCard outcome={outcome} count={count}>
      <FailureLink
        outcome={outcome}
        filter={filter}
        setFilter={setFilter}
        className="font-mono text-xs"
      >
        <span className="tabular-nums">{count}</span>{' '}
        <span className="uppercase tracking-wider text-foreground-muted">{label}</span>
      </FailureLink>
    </OutcomeHoverCard>
  )

  return (
    <div
      className="grid grid-cols-12 bg-card border border-border rounded-md overflow-hidden mb-5"
      style={{ minHeight: 96 }}
    >
      <Cell
        span={3}
        term="combined-mtbi"
        onActivate={onCombinedClick}
        selected={combinedActive}
      >
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-2xl font-semibold tabular-nums text-foreground">
            {combinedFailures === 0 ? 'n/a' : combinedMtbi.toFixed(2)}
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-foreground-muted">
            K GPU-hours
          </span>
        </div>
        <div className="flex items-end justify-between gap-2 mt-auto">
          <span className="font-mono text-[10px] tabular-nums text-foreground-muted">
            {combinedFailures === 0
              ? `no infra failures${range ? ' in window' : ''}`
              : `${fmtNum(gpuHoursRaw)} GPU-HOURS / ${combinedFailures} FAILS / 1000`}
          </span>
          <div className="w-14 h-5 shrink-0">
            <MtbiSparkline values={dailyCombined} />
          </div>
        </div>
      </Cell>

      <Cell span={3} term="hw-mtbi" onActivate={onHwClick} selected={hwActive}>
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-2xl font-semibold tabular-nums text-foreground">
            {hwFailures === 0 ? 'n/a' : hwMtbi.toFixed(2)}
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-foreground-muted">
            K GPU-hours
          </span>
        </div>
        <div className="flex items-end justify-between gap-2 mt-auto">
          <span className="font-mono text-[10px] tabular-nums text-foreground-muted">
            {hwFailures === 0
              ? `no HW failures${range ? ' in window' : ''}`
              : `${fmtNum(gpuHoursRaw)} GPU-HOURS / ${hwFailures} FAILS / 1000`}
          </span>
          <div className="w-14 h-5 shrink-0">
            <MtbiSparkline values={dailyHw} color="var(--color-foreground-secondary)" />
          </div>
        </div>
      </Cell>

      <Cell span={2} term="failures">
        <div className="flex flex-col gap-0.5">
          {failureSlot('hardware', hwFailures, 'HW')}
          {failureSlot('software', swFailures, 'SW')}
          {failureSlot('multi', mixFailures, 'MIX')}
        </div>
      </Cell>

      <Cell span={2} term="gpu-hours">
        <div className="font-mono text-2xl font-semibold tabular-nums text-foreground">
          {gpuHours.value}
          {gpuHours.k && <span className="text-foreground-muted text-xs ml-1">k</span>}
        </div>
      </Cell>

      <Cell span={2} term="window">
        <div className="font-mono text-xs text-foreground tabular-nums">
          {gpuCount} GPUs · {runCount} runs
        </div>
        <div className="font-mono text-[10px] text-foreground-muted tabular-nums mt-auto">
          {range ? <>{fmtDate(range.from)} → {fmtDate(range.to)} <span className="tracking-wide">UTC</span></> : <span className="tracking-wide">all time</span>}
        </div>
      </Cell>
    </div>
  )
}
