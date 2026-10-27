// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId } from 'react'
import type { DailyBucket } from '@/interfaces/attribution'
import type { DefTerm } from '@/interfaces/definitions'
import type { FleetFilter } from '@/interfaces/filter'
import { DEFINITIONS } from '@/lib/definitions'
import { withDimension } from '@/lib/filter-state'
import { counted, fmtGpuHours, fmtNum } from '@/lib/format'
import { cn, pressedStat } from '@/lib/utils'
import { Card } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { NoValue } from '@/components/field'
import { InfoHint } from '@/components/info-hint'
import {
  FailureFilterButton,
  MtbiSparkline,
  type FilterChangeOptions,
} from '@/routers/fleet/components/mtbi-primitives'

// Four equal stats on one card, split by hairlines; below lg they stack. Each
// reads down from its 12px label (with a definition hint where the figure is
// derived): the 20px value line, then a 12px caption held to the stat's foot,
// so side by side the captions end level with the last Failures row.
const STATS = 'grid grid-cols-1 divide-y lg:grid-cols-4 lg:divide-x lg:divide-y-0'

// A stat that filters the runs: its label is a button whose ::after stretches
// over the whole stat, so the stat is one target, while the definition hint
// (positioned and later in the tree, so painted above the stretch) stays out
// of it. The window's daily trend, neutral, ends the value's line on the
// right, its foot on the value's baseline. Pressed, the stat takes the shared
// pressed-stat recipe (lib/utils pressedStat) and its label turns green
// (10.83:1); hovered while not pressed (its toggle hovered, not its hint), the
// hover fill. While the view's figures load, the toggle stays (it may be the
// control just pressed) and only its value, caption and trend wait.
function MtbiStat({
  term,
  pressed,
  onToggle,
  loading,
  mtbi,
  caption,
  trend,
}: {
  term: DefTerm
  pressed: boolean
  onToggle: () => void
  loading: boolean
  /**
   * The mean time between failures, or null where there is none to give (no
   * failures, or no figures); the caption says which.
   */
  mtbi: number | null
  caption: string
  trend: number[]
}) {
  const { label } = DEFINITIONS[term]
  const detailId = useId()
  return (
    <div
      data-pressed={pressed || undefined}
      className={cn(
        'relative flex min-w-0 flex-col gap-1 px-card-x py-card-y transition-colors not-data-pressed:has-[[data-slot=stat-toggle]:hover]:bg-background-active',
        pressedStat,
      )}
    >
      <p className="text-xs text-foreground-muted">
        <button
          type="button"
          data-slot="stat-toggle"
          aria-pressed={pressed}
          aria-label={`Filter runs: ${label}`}
          aria-describedby={detailId}
          onClick={onToggle}
          className="cursor-pointer outline-none after:absolute after:inset-0 focus-visible:after:ring-[3px] focus-visible:after:ring-inset focus-visible:after:ring-border-active/80 aria-pressed:text-primary-300"
        >
          {label}
        </button>{' '}
        <InfoHint term={term} />
      </p>
      <div id={detailId} className="flex flex-1 flex-col gap-1">
        {loading ? (
          <StatPlaceholder />
        ) : (
          <>
            <div className="flex items-baseline justify-between gap-3">
              {mtbi === null ? <StatValue value={null} /> : <StatValue value={mtbi.toFixed(2)} unit="k GPU-hours" />}
              <div aria-hidden="true" className="h-5 w-16 shrink-0">
                <MtbiSparkline values={trend} />
              </div>
            </div>
            <p className="mt-auto text-xs text-foreground-muted">{caption}</p>
          </>
        )}
      </div>
    </div>
  )
}

// A stat's value and caption while the view's figures load: blocks on the
// 28px value line and the 16px caption line, the caption's at the foot.
function StatPlaceholder() {
  return (
    <>
      <Skeleton height="7" width="1/2" />
      <Skeleton height="4" width="2/3" className="mt-auto" />
    </>
  )
}

// A stat's value: a 20px number with its unit stepped back beside it. A value
// that cannot be derived (null) is the shared muted dash at the same size, read
// as "Not available"; the caption below it says why.
function StatValue({ value, unit }: { value: string | null; unit?: string }) {
  return (
    <p className="flex items-baseline gap-1.5">
      {value === null ? (
        <span className="text-xl">
          <NoValue label="Not available" />
        </span>
      ) : (
        <span className="num text-xl font-medium text-foreground">{value}</span>
      )}
      {unit && <span className="text-xs text-foreground-muted">{unit}</span>}
    </p>
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
  failed = false,
}: {
  filter: FleetFilter
  setFilter: (next: FleetFilter, options?: FilterChangeOptions) => void
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
  /** The window's figures could not be loaded: the stats keep their labels and toggles and state no figure. */
  failed?: boolean
}) {
  const failuresId = useId()
  // Where a failed request leaves no figure, each stat's caption says so.
  const noFigures = 'Figures could not be loaded'

  // Raw GPU-hours below 1000 render as-is; only at/above 1000 do we switch to "k".
  const gpuHours = fmtGpuHours(gpuHoursRaw)
  const inWindow = range ? ' in this window' : ''
  // Honest per-day failure-count trend over the window (no fudge).
  const dailyCombined = daily.map((d) => d.hw + d.sw)
  const dailyHw = daily.map((d) => d.hw)

  // A stat is pressed exactly when the filter holds what pressing it sets, and
  // pressing it again releases just that, as every filter toggle here does:
  // Combined MTBI sets failed runs of any outcome; HW MTBI, as a Failures row,
  // sets its outcome alone, which already leaves passing runs out, so the
  // status the reader chose stays as it was. A symptom, node or rule filter
  // narrows either list and stays.
  const combinedPressed = filter.status === 'failed' && !filter.outcome
  const hwPressed = filter.outcome === 'hardware'

  const toggleCombined = () => {
    if (combinedPressed) {
      setFilter({ ...filter, status: 'all' })
      return
    }
    setFilter(withDimension({ ...filter, status: 'failed' }, 'outcome', undefined), { revealRuns: true })
  }
  const toggleHw = () => {
    if (hwPressed) {
      setFilter(withDimension(filter, 'outcome', undefined))
      return
    }
    setFilter(withDimension(filter, 'outcome', 'hardware'), { revealRuns: true })
  }

  // Loading, the stats keep their labels and toggles; only the figures wait.
  // Failed, the figures give way to the shared muted dash, read as "Not
  // available", and the captions say why.
  return (
    <Card aria-busy={loading || undefined} className={STATS}>
      <MtbiStat
        term="combined-mtbi"
        pressed={combinedPressed}
        onToggle={toggleCombined}
        loading={loading}
        mtbi={failed || combinedFailures === 0 ? null : combinedMtbi}
        caption={
          failed
            ? noFigures
            : combinedFailures === 0
              ? `No infrastructure failures${inWindow}`
              : `${fmtNum(gpuHoursRaw)} GPU-hours ÷ ${counted(combinedFailures, 'failure', 'failures')} ÷ 1000`
        }
        trend={dailyCombined}
      />
      <MtbiStat
        term="hw-mtbi"
        pressed={hwPressed}
        onToggle={toggleHw}
        loading={loading}
        mtbi={failed || hwFailures === 0 ? null : hwMtbi}
        caption={
          failed
            ? noFigures
            : hwFailures === 0
              ? `No hardware failures${inWindow}`
              : `${fmtNum(gpuHoursRaw)} GPU-hours ÷ ${counted(hwFailures, 'failure', 'failures')} ÷ 1000`
        }
        trend={dailyHw}
      />
      <div className="flex min-w-0 flex-col gap-1 px-card-x py-card-y">
        <p id={failuresId} className="text-xs text-foreground-muted">Failures</p>
        {/* Pulled out by the toggles' padding, so their dots sit on the label's
            edge, and by the 2px each 24px target keeps above and below its
            20px line, so the first row's text starts where the value lines
            beside it do and the last ends level with their captions. */}
        <div role="group" aria-labelledby={failuresId} className="-mx-1.5 -my-0.5 flex flex-col items-start">
          <FailureFilterButton outcome="hardware" count={loading || failed ? undefined : hwFailures} filter={filter} setFilter={setFilter} />
          <FailureFilterButton outcome="software" count={loading || failed ? undefined : swFailures} filter={filter} setFilter={setFilter} />
          <FailureFilterButton outcome="multi" count={loading || failed ? undefined : mixFailures} filter={filter} setFilter={setFilter} />
        </div>
      </div>
      <div className="flex min-w-0 flex-col gap-1 px-card-x py-card-y">
        <p className="text-xs text-foreground-muted">
          GPU-hours <InfoHint term="gpu-hours" />
        </p>
        {loading ? (
          <StatPlaceholder />
        ) : (
          <>
            {failed ? <StatValue value={null} /> : <StatValue value={gpuHours.value} unit={gpuHours.k ? 'k' : undefined} />}
            <p className="mt-auto text-xs text-foreground-muted tabular-nums">
              {failed ? noFigures : `${counted(gpuCount, 'GPU', 'GPUs')} · ${counted(runCount, 'run', 'runs')}`}
            </p>
          </>
        )}
      </div>
    </Card>
  )
}
