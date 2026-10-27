// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useRef, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { useECharts } from '@/hooks/use-echarts'
import { OUTCOMES, breakdownHighlight } from '@/lib/attribution-data'
import { outcomeColor, readColor } from '@/lib/tokens'
import { escapeHtml } from '@/lib/html'
import type { DailyBreakdown, OutcomeKey } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import { withDimension } from '@/lib/filter-state'
import { isWindowed } from '@/lib/date-window'
import { Skeleton } from '@/components/ui/skeleton'
import { Toggle } from '@/components/ui/toggle'
import { OutcomeDot } from '@/components/outcome'
import { SectionCard } from '@/components/section-card'
import { Identifier } from '@/components/field'
import { cn, subtitle } from '@/lib/utils'

// Every segment of a pane takes that pane's outcome color, so a symptom's color
// never depends on its rank; the legend below names and filters the symptoms.
function BreakdownChart({
  outcome,
  breakdown,
  selectedSymptom,
  onSymptomClick,
}: {
  outcome: OutcomeKey
  breakdown: DailyBreakdown
  selectedSymptom?: string
  onSymptomClick: (symptom: string) => void
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [hoveredSymptom, setHoveredSymptom] = useState<string>()

  const option = useMemo<EChartsOption>(() => {
    const { days, series } = breakdown
    const color = outcomeColor(outcome)
    // A card-colored hairline around each segment keeps stacked symptoms apart.
    const gap = readColor('--chart-surface')
    const showTotals = days.length <= 10
    const totals = days.map((_, dayIndex) =>
      series.reduce((sum, s) => sum + (s.data[dayIndex] || 0), 0),
    )
    return {
      animation: true,
      animationDuration: 200,
      // Room above the tallest stack for its total label.
      grid: { left: 8, right: 12, top: 24, bottom: 8, containLabel: true },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params) => {
          const arr = params as unknown as Array<{
            axisValue: string; seriesName: string; value: number
          }>
          const total = arr.reduce((a, p) => a + (p.value || 0), 0)
          if (total === 0) return ''
          // The tooltip is DOM, so it takes the app's classes: neutral text, numbers in mono.
          const head = `<div class="mb-1 font-medium text-foreground">${escapeHtml(arr[0].axisValue)}</div>`
          // Every segment shares the pane color, so rows are told apart by name alone.
          const body = arr
            .filter((p) => p.value > 0)
            .map((p) =>
              `<div class="text-foreground-muted">${escapeHtml(p.seriesName)}: <span class="num text-foreground">${p.value}</span></div>`,
            )
            .join('')
          const foot = `<div class="mt-1 border-t border-border pt-1 text-foreground-muted">Total: <span class="num text-foreground">${total}</span></div>`
          return head + body + foot
        },
      },
      xAxis: {
        type: 'category',
        data: days.map((d) => d.slice(5)),
      },
      yAxis: {
        type: 'value',
        minInterval: 1,
      },
      series: [
        ...series.map((s) => ({
          name: s.symptom,
          type: 'bar' as const,
          stack: 'sym',
          data: s.data,
          itemStyle: {
            color,
            borderColor: gap,
            borderWidth: 1,
            opacity: selectedSymptom && selectedSymptom !== s.symptom ? 0.25 : 1,
          },
          emphasis: { focus: 'series' as const },
        })),
        // Each day's total caps its stack from a layer of its own: an empty,
        // silent bar at the top of the stack that carries only the label. A
        // symptom selected or hovered dims the symptoms, never the totals.
        ...(showTotals
          ? [{
              id: 'day-totals',
              type: 'bar' as const,
              stack: 'sym',
              silent: true,
              data: totals.map((total) => ({ value: 0, label: { show: total > 0, formatter: String(total) } })),
              label: { position: 'top' as const },
              blur: { label: { opacity: 1 } },
            }]
          : []),
      ],
    }
  }, [breakdown, outcome, selectedSymptom])

  const chart = useECharts(ref, option, {
    click: (p) => {
      const name = (p as { seriesName?: string }).seriesName
      if (name) onSymptomClick(name)
    },
    mouseover: (p) => setHoveredSymptom(p.seriesName),
    mouseout: () => setHoveredSymptom(undefined),
  })
  const emphasize = (symptom: string, on: boolean) =>
    chart.current?.dispatchAction({ type: on ? 'highlight' : 'downplay', seriesName: symptom })

  return (
    <div className="flex flex-col gap-3">
      <div ref={ref} className="w-full h-56" />
      <ul aria-label={`${OUTCOMES[outcome].label} root symptoms`} className="-mx-1.5 flex flex-wrap gap-x-3 gap-y-1">
        {breakdown.series.map((s) => (
          <li key={s.symptom}>
            {/* The server folds the tail into Other without naming its members, so it cannot filter. */}
            {s.symptom === 'Other' ? (
              <span className="inline-flex h-6 items-center gap-1 px-1.5 text-sm text-foreground-muted">
                Other <span className="tabular-nums">{s.total}</span>
              </span>
            ) : (
              // A recorded symptom of any length: it wraps after its separators,
              // as a filter chip's value does, never past the card.
              <Toggle
                pressed={selectedSymptom === s.symptom}
                onPressedChange={() => onSymptomClick(s.symptom)}
                onMouseEnter={() => emphasize(s.symptom, true)}
                onMouseLeave={() => emphasize(s.symptom, false)}
                onFocus={() => emphasize(s.symptom, true)}
                onBlur={() => emphasize(s.symptom, false)}
                data-highlighted={hoveredSymptom === s.symptom || undefined}
                className="h-auto min-h-6 py-0.5 text-left whitespace-normal wrap-anywhere not-aria-pressed:data-highlighted:bg-background-active not-aria-pressed:data-highlighted:text-foreground"
              >
                <span><Identifier>{s.symptom}</Identifier> <span className="text-foreground-muted tabular-nums">{s.total}</span></span>
              </Toggle>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}


export function BreakdownByDay({
  hardware,
  software,
  filter,
  setFilter,
  loading = false,
  failed = false,
}: {
  hardware: DailyBreakdown
  software: DailyBreakdown
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
  /** This view's breakdowns are on their way: the panes keep their titles, and their totals and charts wait. */
  loading?: boolean
  /** This view's breakdowns could not be loaded: the section says so instead of charting nothing. */
  failed?: boolean
}) {
  const charts: { outcome: OutcomeKey; breakdown: DailyBreakdown }[] = [
    { outcome: 'hardware', breakdown: hardware },
    { outcome: 'software', breakdown: software },
  ]

  // Symptom-only, matching how Pareto and Movers set the symptom dimension. Highlight
  // (breakdownHighlight) keys on filter.symptom, so any source — this chart, Pareto,
  // Movers, a deep link — stays in sync. Setting outcome here would desync them and
  // leave a stale outcome on the next symptom click elsewhere.
  const onSymptomClick = (symptom: string) => {
    if (symptom === 'Other') return
    const active = filter.symptom === symptom
    setFilter(withDimension(filter, 'symptom', active ? undefined : symptom))
  }

  // When the OUTCOME filter targets the OTHER of these two panes, lightly de-emphasize
  // this one (focus follows the filter). Only hardware/software are represented here, so
  // outcome multi/user/pass leaves both panes untouched (greying both would read as broken).
  const paneMuted = (outcome: OutcomeKey): boolean =>
    (filter.outcome === 'hardware' || filter.outcome === 'software') && filter.outcome !== outcome

  // With neither outcome failing, one empty state stands for both panes, in
  // the KPI's words for hardware and system software failures together; a pane
  // empty beside a charted one says so in a line of muted text. Nothing is
  // called empty before the view's breakdowns arrive, or when they could not
  // be loaded: then the section says that, as the run list does.
  const inWindow = isWindowed(filter) ? ' in this window' : ''
  const noFailures = !loading && charts.every(({ breakdown }) => breakdown.total === 0)

  return (
    <SectionCard
      title="Failures by day"
      description="Failed attempts per day, stacked by root symptom. Select a segment to filter runs."
      flush
      empty={
        failed
          ? { title: 'Failures by day could not be loaded', description: 'Refresh to try again.' }
          : noFailures
            ? { title: 'No failures to chart', description: `No infrastructure failures${inWindow}.` }
            : undefined
      }
    >
      <div aria-busy={loading || undefined} className="grid grid-cols-1 divide-y lg:grid-cols-4 lg:divide-x lg:divide-y-0">
        {charts.map(({ outcome, breakdown }) => (
          <div
            key={outcome}
            className={cn(
              'flex min-w-0 flex-col gap-3 px-card-x py-card-y transition-opacity duration-200 lg:col-span-2',
              paneMuted(outcome) && 'opacity-60',
            )}
          >
            {/* The pane's count sits beside its title, as a section count does, and stays out of its name. */}
            <div className="flex items-center gap-2">
              <OutcomeDot outcome={outcome} />
              <h3 className={subtitle}>{OUTCOMES[outcome].label}</h3>
              {loading ? (
                <Skeleton height="4" className="w-6" />
              ) : (
                <span className="text-foreground-muted tabular-nums">{breakdown.total}</span>
              )}
            </div>
            {loading ? (
              <Skeleton className="h-56" />
            ) : breakdown.total === 0 ? (
              <p className="text-foreground-muted">{`No ${OUTCOMES[outcome].label} results${inWindow}.`}</p>
            ) : (
              <BreakdownChart
                outcome={outcome}
                breakdown={breakdown}
                selectedSymptom={breakdownHighlight(breakdown, filter.symptom)}
                onSymptomClick={onSymptomClick}
              />
            )}
          </div>
        ))}
      </div>
    </SectionCard>
  )
}
