// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import Decimal from 'decimal.js'
import type { EChartsOption, LineSeriesOption, YAXisComponentOption } from 'echarts'
import { ChevronRight } from 'lucide-react'
import { OUTCOMES, dominantOutcome } from '@/lib/attribution-data'
import { useECharts } from '@/hooks/use-echarts'
import { chartFonts, outcomeColor, readColor } from '@/lib/tokens'
import { escapeHtml } from '@/lib/html'
import type {
  NodeLedgerEntry,
  OutcomeKey,
  OutcomeLedgerEntry,
  RuleLedgerEntry,
  SymptomLedgerEntry,
  UserLedgerEntry,
} from '@/interfaces/attribution'
import type { FleetFilter, ParetoAxis, ParetoMetric } from '@/interfaces/filter'
import type { DashboardPayload } from '@/interfaces/meta'
import { withDimension } from '@/lib/filter-state'
import { Button } from '@/components/ui/button'
import { CardFooter } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Toggle } from '@/components/ui/toggle'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Identifier } from '@/components/field'
import { OutcomeDot } from '@/components/outcome'
import { SectionCard } from '@/components/section-card'
import { counted, fmtShare, plural, shareReaches } from '@/lib/format'

interface ParetoRow {
  label: string
  value: number
  outcomes: Partial<Record<OutcomeKey, number>>
}

// One ranked category: its value, the running total up to it where shares add
// up, and the outcome that dominates it (the bar's color).
interface RankedRow extends ParetoRow {
  running?: number
  dominant: OutcomeKey
}

// The categories the card ranks: the ten largest by the measure. The server
// sends the ten largest by each measure, so these are exact either way.
const PARETO_LIMIT = 10

// Whether a grouping's shares add up to a running share: no failed attempt
// counts in two of its categories. An attempt has one root symptom, one user
// and one root rule, each given its whole GPU-hours. It can name several
// responsible nodes, or none: counted, it counts once for each; in GPU-hours
// its cost is split evenly among the nodes it names (the server's node
// ledger), and those shares add up.
function sharesAddUp(axis: ParetoAxis, metric: ParetoMetric): boolean {
  return axis !== 'node' || metric === 'nodeHours'
}

// The failures a card's figures are of: the outcome chosen, or every failed
// attempt; the ledgers are this scope's. A Pass run is no failure, so with
// Pass chosen there is none to rank and the card still speaks of failures.
function scopeOf(outcome: OutcomeKey | undefined): { one: string; many: string } {
  if (outcome === undefined || outcome === 'pass') return { one: 'failed attempt', many: 'failed attempts' }
  return { one: `${OUTCOMES[outcome].label} attempt`, many: `${OUTCOMES[outcome].label} attempts` }
}

// What the measure totals over the scope, e.g. "GPU-hours of failed attempts".
function measured(metric: ParetoMetric, scope: { many: string }): string {
  return metric === 'count' ? scope.many : `GPU-hours of ${scope.many}`
}

// A phrase that starts a sentence.
function capitalized(phrase: string): string {
  return phrase.charAt(0).toUpperCase() + phrase.slice(1)
}

// Tooltips are DOM, so their outcome mark paints from the data-outcome tone like OutcomeDot.
function outcomeSwatch(outcome: OutcomeKey): string {
  return `<span data-outcome="${outcome}" style="display:inline-block;width:8px;height:8px;border:1px solid var(--tone-edge);border-radius:2px;background:var(--tone);margin-right:6px;vertical-align:middle"></span>`
}

const AXIS_DIM_KEY: Record<ParetoAxis, keyof FleetFilter> = {
  symptom: 'symptom',
  node:    'node',
  user:    'user',
  rule:    'rule',
}

const AXIS_NOUN: Record<ParetoAxis, { one: string; many: string; column: string }> = {
  symptom: { one: 'root symptom', many: 'root symptoms', column: 'Root symptom' },
  node:    { one: 'responsible node', many: 'responsible nodes', column: 'Responsible node' },
  user:    { one: 'user', many: 'users', column: 'User' },
  rule:    { one: 'rule', many: 'rules', column: 'Rule' },
}

function buildParetoSeries(
  axis: ParetoAxis,
  metric: ParetoMetric,
  symptomLedger: SymptomLedgerEntry[],
  nodeLedger: NodeLedgerEntry[],
  userLedger: UserLedgerEntry[],
  ruleLedger: RuleLedgerEntry[],
): ParetoRow[] {
  const valueOf = (d: { count: number; nodeHours: number }) =>
    metric === 'count' ? d.count : d.nodeHours

  if (axis === 'node') {
    return nodeLedger.map((d) => ({ label: d.node, value: valueOf(d), outcomes: d.outcomes }))
  }
  if (axis === 'rule') {
    return ruleLedger.map((d) => ({ label: d.rule, value: valueOf(d), outcomes: d.outcomes }))
  }
  if (axis === 'user') {
    return userLedger.map((d) => ({ label: d.user, value: valueOf(d), outcomes: d.outcomes }))
  }
  return symptomLedger.map((d) => ({ label: d.symptom, value: valueOf(d), outcomes: d.outcomes }))
}

// The ten largest by the measure (ties by name, as the server orders them),
// where shares add up with the running total up to each; and the vital few:
// the fewest leading categories whose running share of all the failures in
// scope (the outcome split's total, not the ten's: the ledger holds only the
// leading categories) reaches 80% as the Ranked list shows it (shareReaches,
// fmtShare's judgment), or none when the ten do not reach it (a long tail),
// or when shares do not add up. Five fifths of 1/3 GPU-hour, each kept to a
// millionth, reach 79.99995% at the fourth, shown and so cut as 80%; with a
// total under half a GPU-hour, a running share shown as all of it is cut
// there, and one shown as none is not.
function rankSeries(series: ParetoRow[], total: number, addsUp: boolean, digits: number): { rows: RankedRow[]; vitalCount: number | null } {
  const sorted = [...series]
    .sort((a, b) => b.value - a.value || (a.label < b.label ? -1 : a.label > b.label ? 1 : 0))
    .slice(0, PARETO_LIMIT)
  let running = new Decimal(0)
  let vitalCount: number | null = null
  const rows = sorted.map((d, index) => {
    running = running.plus(d.value)
    if (addsUp && vitalCount === null && total > 0 && shareReaches(running, total, digits, 80)) vitalCount = index + 1
    return { ...d, running: addsUp ? running.toNumber() : undefined, dominant: dominantOutcome(d.outcomes) }
  })
  return { rows, vitalCount }
}

// The decimals the card shows a value to: a count whole, GPU-hours to a tenth.
function decimals(metric: ParetoMetric): number {
  return metric === 'count' ? 0 : 1
}

// A value as this card lists it (the bars' tooltip, the donut's legend and
// tooltip, the Ranked list and the concentration summary): to the card's
// precision, grouped in thousands so its size reads at a glance, as the value
// axis's 300k does.
function fmtValue(metric: ParetoMetric, v: number): string {
  const digits = decimals(metric)
  return v.toLocaleString('en', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

// Thousands and millions, for the readouts with a fixed room: the value
// axis's ticks and the donut's center.
const MAGNITUDES = [
  { size: 1e6, suffix: 'M' },
  { size: 1e3, suffix: 'k' },
]

// A value-axis tick is a round number the axis picks for itself: counts step
// by whole numbers, GPU-hours by at least a tenth, never finer than the values
// above. From a thousand up it reads in thousands or millions, so no tick is
// wider than five characters and every one fits the plot's left inset.
function fmtTick(v: number): string {
  const unit = MAGNITUDES.find(({ size }) => Math.abs(v) >= size)
  return unit ? `${new Decimal(v).div(unit.size).toString()}${unit.suffix}` : new Decimal(v).toString()
}

// The donut's center value has room for six mono characters in the ring's
// hole, so from a thousand up it reads to a tenth of a thousand or a million
// (113.3k, 1.1M); the legend below gives each value exactly. A unit is taken
// once the value reaches a thousand of the unit below as that unit shows it,
// so 999,960 reads 1.0M, never 1000.0k.
function fmtCenter(metric: ParetoMetric, v: number): string {
  const unit = MAGNITUDES.find(({ size }) => new Decimal(v).div(size / 1e3).toDecimalPlaces(1).gte(1e3))
  return unit ? `${new Decimal(v).div(unit.size).toFixed(1)}${unit.suffix}` : fmtValue(metric, v)
}

// The plot's inset in the chart: room for the value axes' labels at the sides
// and for the axis names above. The plot keeps exactly this inset (the axis
// labels are sized to fit it rather than moving it), so a category's band is
// known before the chart lays out.
const GRID = { left: 48, right: 56, top: 32, bottom: 32 }
// Space kept between neighbouring category names under the bars.
const CATEGORY_LABEL_GAP = 8

function ParetoChart({
  axis,
  metric,
  ranked,
  total,
  selectedCategory,
  outcomeHighlight,
  onCategoryClick,
}: {
  axis: ParetoAxis
  metric: ParetoMetric
  /** The ranked categories; each carries a running total where shares add up, drawn as a line over the bars. */
  ranked: RankedRow[]
  /** All the failures in scope, in the card's measure: what every share is of. */
  total: number
  selectedCategory?: string
  outcomeHighlight: OutcomeKey | null
  onCategoryClick: (label: string) => void
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const empty = ranked.length === 0
  // A category's name gets the width of its band under the bars, less a gap,
  // so it is cut only when the band is narrower than the name.
  const [plotWidth, setPlotWidth] = useState(0)
  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return
    const measure = () => setPlotWidth(element.clientWidth - GRID.left - GRID.right)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const categoryLabelWidth = empty ? 0 : Math.max(plotWidth / ranked.length - CATEGORY_LABEL_GAP, 0)

  const option = useMemo<EChartsOption>(() => {
    const metricLabel = metric === 'count' ? 'Failed attempts' : 'GPU-hours'
    // The running share and the 80% mark are chrome, not categories: neutral, never green.
    const cumulativeColor = readColor('--foreground-secondary')
    const thresholdColor = readColor('--foreground-muted')
    // Shares add up to a running share on every grouping but the responsible
    // node counted (sharesAddUp); without one there is no line, no 80% mark
    // and no share axis, and the plot keeps its inset, so the bars never move.
    const running = ranked.length > 0 && ranked[0].running !== undefined
    const share = (part: number) => fmtShare(part, total, decimals(metric))
    const shareAxis: YAXisComponentOption = {
      type: 'value',
      name: 'Running share',
      min: 0,
      max: 100,
      interval: 25,
      splitLine: { show: false },
      axisLabel: { formatter: '{value}%' },
    }
    const runningLine: LineSeriesOption = {
      name: 'cumulative',
      type: 'line',
      yAxisIndex: 1,
      data: ranked.map((d) => (d.running === undefined ? null : total > 0 ? new Decimal(d.running).div(total).times(100).toNumber() : 0)),
      symbol: 'none',
      lineStyle: { color: cumulativeColor, width: 1.5 },
      itemStyle: { color: cumulativeColor },
      smooth: false,
      // The 80% mark is a fixed reference, drawn in place rather than
      // swept in: each option replaces the whole chart (useECharts), which
      // rebuilds the mark, so a sweep would replay from the left on every
      // selection and every change of width.
      markLine: {
        silent: true,
        symbol: 'none',
        animation: false,
        data: [
          {
            yAxis: 80,
            lineStyle: { color: thresholdColor, type: 'dashed', width: 1 },
            label: { color: thresholdColor, formatter: '80%', position: 'insideEndTop' },
          },
        ],
      },
    }
    const bars = ranked.map((d) => {
      // Category selection wins visually. Otherwise dim by OUTCOME MEMBERSHIP —
      // whether this category contains the selected outcome at all — NOT by its
      // dominant outcome. A mixed-outcome category shouldn't vanish just because a
      // different outcome happens to dominate it. Outcome dim (0.3) is lighter than
      // category dim (0.25) so the two never read as the same state.
      let opacity = 0.92
      if (selectedCategory) {
        opacity = d.label === selectedCategory ? 0.92 : 0.25
      } else if (outcomeHighlight) {
        opacity = (d.outcomes[outcomeHighlight] ?? 0) > 0 ? 0.92 : 0.3
      }
      return { row: d, opacity }
    })
    return {
      animation: true,
      animationDuration: 200,
      animationDurationUpdate: 150,
      animationEasing: 'quarticOut',
      animationEasingUpdate: 'quarticOut',
      // The bars carry no value labels: the running share would cross them
      // wherever it passes a bar's top, and the tooltip, the value axis and the
      // Ranked list give each value.
      grid: { ...GRID, outerBoundsMode: 'none' },
      tooltip: {
        show: !empty,
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params) => {
          const arr = params as unknown as Array<{
            seriesIndex: number
            axisValue: string
            data: { value: number; dominant?: OutcomeKey; running?: number }
          }>
          const bar = arr.find((p) => p.seriesIndex === 0)
          if (!bar) return ''
          const valueText = fmtValue(metric, bar.data.value)
          const outcomeKey = bar.data.dominant
          const outcomeLabel = outcomeKey ? OUTCOMES[outcomeKey].label : '—'
          const swatch = outcomeKey ? outcomeSwatch(outcomeKey) : ''
          // The tooltip is DOM, so it takes the app's classes: neutral text, numbers in mono.
          return [
            `<div class="mb-1 font-medium text-foreground">${swatch}${escapeHtml(bar.axisValue)}</div>`,
            `<div class="text-foreground-muted">${escapeHtml(metricLabel)}: <span class="num text-foreground">${escapeHtml(valueText)}</span></div>`,
            `<div class="text-foreground-muted">Share: <span class="num text-foreground">${escapeHtml(share(bar.data.value))}</span></div>`,
            ...(bar.data.running !== undefined
              ? [`<div class="text-foreground-muted">Running share: <span class="num text-foreground">${escapeHtml(share(bar.data.running))}</span></div>`]
              : []),
            `<div class="text-foreground-muted">Dominant outcome: <span class="text-foreground">${escapeHtml(outcomeLabel)}</span></div>`,
          ].join('')
        },
      },
      xAxis: {
        type: 'category',
        show: !empty,
        data: ranked.map((d) => d.label),
        axisLabel: { interval: 0, width: categoryLabelWidth, overflow: 'truncate', ellipsis: '…' },
      },
      yAxis: [
        {
          type: 'value',
          show: !empty,
          name: metricLabel,
          minInterval: metric === 'count' ? 1 : 0.1,
          axisLabel: { formatter: fmtTick },
        },
        ...(running ? [shareAxis] : []),
      ],
      series: [
        {
          name: 'value',
          type: 'bar',
          data: bars.map(({ row, opacity }) => ({
            value: row.value,
            itemStyle: { color: outcomeColor(row.dominant), opacity },
            name: row.label,
            dominant: row.dominant,
            running: row.running,
          })),
          barWidth: '55%',
        },
        ...(running ? [runningLine] : []),
      ],
    }
  }, [ranked, total, metric, empty, selectedCategory, outcomeHighlight, categoryLabelWidth])

  useECharts(ref, option, {
    click: (p) => {
      if (p.seriesIndex !== 0) return
      const label = (p.data as { name?: string })?.name
      if (label) onCategoryClick(label)
    },
  })
  return (
    <div className="relative h-72 w-full">
      <div ref={ref} className="absolute inset-0" />
      {/* Failures can exist with no category on this axis, e.g. none named a
          responsible node. The donut beside still has them, so this pane's
          empty state is a line of muted text, not the section's. */}
      {empty && (
        <p role="status" className="absolute inset-0 flex items-center justify-center p-6 text-center text-foreground-muted">
          {`No ${AXIS_NOUN[axis].many} for this filter.`}
        </p>
      )}
    </div>
  )
}

interface OutcomeSlice {
  key: OutcomeKey
  full: string
  value: number
}

// The outcome split of the window's failures, from the outcome ledger.
function outcomeSlices(outcomeLedger: OutcomeLedgerEntry[], metric: ParetoMetric): OutcomeSlice[] {
  return outcomeLedger
    .map((entry) => ({
      key: entry.outcome,
      full: OUTCOMES[entry.outcome].label,
      value: metric === 'count' ? entry.count : entry.nodeHours,
    }))
    .filter((slice) => slice.value > 0)
}

// The total of some slices, exact: their values are kept to a millionth.
function sumOf(slices: OutcomeSlice[]): number {
  return slices.reduce((sum, slice) => sum.plus(slice.value), new Decimal(0)).toNumber()
}

function OutcomeDonut({
  slices,
  selected,
  onSelect,
  metric,
}: {
  slices: OutcomeSlice[]
  selected: OutcomeKey | null
  onSelect: (k: OutcomeKey | null) => void
  metric: ParetoMetric
}) {
  const ref = useRef<HTMLDivElement | null>(null)

  const total = sumOf(slices)
  const sel = selected ? slices.find((s) => s.key === selected) : undefined
  const centerNum = sel ? sel.value : total
  const centerNumText = fmtCenter(metric, centerNum)
  // A selected outcome is already named by its pressed legend row, so the
  // center gives its value and share; the full label would not fit the ring.
  const centerLbl = sel
    ? `${fmtShare(sel.value, total, decimals(metric))} of total`
    : metric === 'count'
      ? 'failed attempts'
      : 'GPU-hours'

  const option = useMemo<EChartsOption>(() => {
    const foreground = readColor('--foreground')
    const muted = readColor('--foreground-muted')
    return {
      animation: true,
      animationDuration: 200,
      animationDurationUpdate: 150,
      animationEasing: 'quarticOut',
      animationEasingUpdate: 'quarticOut',
      tooltip: {
        trigger: 'item',
        formatter: (params) => {
          const p = params as {
            name: string
            value: number
            data: unknown
          }
          const outcome = (p.data as { __key: OutcomeKey }).__key
          const unit = metric === 'count' ? 'Failed attempts' : 'GPU-hours'
          const name = escapeHtml(p.name)
          const value = escapeHtml(fmtValue(metric, p.value))
          // The share is the legend's, of the exact total, not the chart's own rounded percent.
          return [
            `<div class="mb-1 font-medium text-foreground">${outcomeSwatch(outcome)}${name}</div>`,
            `<div class="text-foreground-muted">${unit}: <span class="num text-foreground">${value}</span></div>`,
            `<div class="text-foreground-muted">Share: <span class="num text-foreground">${fmtShare(p.value, total, decimals(metric))}</span> of total</div>`,
          ].join('')
        },
      },
      // The center readout is a title of the chart, not a label of the slices:
      // a slice's label takes the slice's dimmed opacity when another outcome
      // is selected. The value is the page's KPI size (20 at 500, mono); its
      // caption is muted chart text.
      title: {
        text: `{num|${centerNumText}}\n{lbl|${centerLbl}}`,
        // Centered this way, ECharts also centers each line in the block.
        left: 'center',
        top: 'middle',
        padding: 0,
        // A title is bolder by default; the caption is chart text at 400.
        textStyle: {
          fontWeight: 400,
          rich: {
            num: { color: foreground, fontFamily: chartFonts().mono, fontSize: 20, fontWeight: 500, lineHeight: 28 },
            lbl: { color: muted, fontSize: 12, lineHeight: 16 },
          },
        },
      },
      series: [
        {
          name: 'outcome',
          type: 'pie',
          // Thin enough that a 12px caption fits the 102px hole of the h-40 chart.
          radius: ['64%', '90%'],
          center: ['50%', '50%'],
          label: { show: false },
          labelLine: { show: false },
          itemStyle: {
            borderColor: readColor('--chart-surface'),
            borderWidth: 1.5,
          },
          data: slices.map((s) => ({
            name: s.full,
            value: s.value,
            itemStyle: {
              color: outcomeColor(s.key),
              opacity: selected && selected !== s.key ? 0.18 : 1,
            },
            __key: s.key,
          })),
          emphasis: { scale: false, focus: 'none' as const },
        },
      ],
    }
  }, [slices, selected, centerNumText, centerLbl, metric, total])

  useECharts(ref, option, {
    click: (p) => {
      const k = (p.data as { __key?: OutcomeKey })?.__key
      if (k) onSelect(k === selected ? null : k)
    },
  })

  return (
    <div className="flex flex-col gap-3">
      <div ref={ref} className="h-40 w-full" />
      {/* The donut's keyboard path and its legend: each outcome filters the runs
          like its slice. The numbers keep their width and a label that does
          not fit the rest wraps rather than being cut; each row reads from
          its first line, its dot centred on it. */}
      <div role="group" aria-label="Outcomes" className="-mx-1.5 grid grid-cols-[auto_minmax(0,1fr)_auto_auto] gap-x-2 gap-y-1">
        {slices.map((s) => {
          const isSel = selected === s.key
          return (
            <Toggle
              key={s.key}
              pressed={isSel}
              onPressedChange={() => onSelect(isSel ? null : s.key)}
              className="col-span-full grid h-auto min-h-6 grid-cols-subgrid items-start justify-items-start gap-x-2 py-0.5 text-left whitespace-normal"
            >
              <span className="flex h-5 items-center"><OutcomeDot outcome={s.key} /></span>
              <span className="min-w-0">{s.full}</span>{' '}
              <span className="num justify-self-end">{fmtValue(metric, s.value)}</span>{' '}
              <span className="num justify-self-end text-foreground-muted">{fmtShare(s.value, total, decimals(metric))}</span>
            </Toggle>
          )
        })}
      </div>
    </div>
  )
}

// The bars in a table, in the same order: the keyboard path to what the chart
// only offers to a pointer. Each category toggles the same filter as its bar.
// Its share is of all the failures in scope; the running share column is there
// where shares add up, as the chart's line is.
function RankedList({
  axis,
  metric,
  ranked,
  total,
  selectedCategory,
  onCategoryClick,
}: {
  axis: ParetoAxis
  metric: ParetoMetric
  ranked: RankedRow[]
  /** All the failures in scope, in the card's measure: what every share is of. */
  total: number
  selectedCategory?: string
  onCategoryClick: (label: string) => void
}) {
  const running = ranked.length > 0 && ranked[0].running !== undefined
  const share = (part: number) => fmtShare(part, total, decimals(metric))
  return (
    <Table aria-label={`Failures ranked by ${AXIS_NOUN[axis].one}`}>
      <TableHeader>
        <TableRow>
          <TableHead className="text-right">Rank</TableHead>
          <TableHead className="w-full">{AXIS_NOUN[axis].column}</TableHead>
          <TableHead className="text-right">{metric === 'count' ? 'Failed attempts' : 'GPU-hours'}</TableHead>
          <TableHead className="text-right">Share</TableHead>
          {running && <TableHead className="text-right">Running share</TableHead>}
        </TableRow>
      </TableHeader>
      <TableBody>
        {ranked.map((row, index) => (
          <TableRow key={row.label}>
            <TableCell className="num text-right text-foreground-muted">{index + 1}</TableCell>
            {/* A recorded name of any length (a symptom, node, user or rule): the
                toggle wraps it after its separators, its dot kept on its first
                line, so the list fits its card. */}
            <TableCell wrap>
              <Toggle
                pressed={row.label === selectedCategory}
                onPressedChange={() => onCategoryClick(row.label)}
                className="-my-0.5 -ml-1.5 h-auto min-h-6 items-start gap-2 py-0.5 text-left align-top whitespace-normal"
              >
                <span className="flex h-5 items-center"><OutcomeDot outcome={row.dominant} /></span>
                <span className="min-w-0"><Identifier>{row.label}</Identifier></span>
              </Toggle>
            </TableCell>
            <TableCell className="num text-right">{fmtValue(metric, row.value)}</TableCell>
            <TableCell className="num text-right">{share(row.value)}</TableCell>
            {row.running !== undefined && <TableCell className="num text-right">{share(row.running)}</TableCell>}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

// The summary fills its footer row beside the Ranked list button, and takes
// a row of its own before it would have under 16rem there.
const SUMMARY = 'min-w-0 grow basis-64 text-sm text-foreground-muted'

// How concentrated the failures are, in words, every share of all the
// failures in scope (the outcome chosen, or every failed attempt). Where
// shares add up: the vital few (the fewest leading categories whose running
// share reaches 80%, or the ten ranked when they do not reach it), out of how
// many categories there are ("All 5 …", "Both …" when they are every one),
// and the share they reach, as the Ranked list's running share gives it; then
// the largest category, its value in the card's measure and its share.
// Counted by responsible node, where they do not add up, only the largest node
// and the share of the attempts it was responsible in. A lone category is
// named as the only one. When several categories share the largest value,
// none is called the largest alone: the first is named with how many more
// share it, the server's count, past the ten ranked too. A tie is an exact
// tie: two values shown alike but not equal (16.04 and 15.96 GPU-hours, both
// 16.0) are not tied, since their shares can differ as shown, and the larger
// is the largest.
function ConcentrationSummary({ axis, metric, scope, ranked, vitalCount, total, categories, tied }: {
  axis: ParetoAxis
  metric: ParetoMetric
  /** The failures in scope, e.g. "failed attempts" or "Hardware Fault attempts". */
  scope: { one: string; many: string }
  ranked: RankedRow[]
  vitalCount: number | null
  /** All the failures in scope, in the card's measure. */
  total: number
  /** How many categories the grouping has in all. */
  categories: number
  /** How many categories tie for the largest value in the card's measure. */
  tied: number
}) {
  if (!(total > 0)) {
    return <p className={SUMMARY}>{`These ${scope.many} recorded no GPU-hours.`}</p>
  }
  const noun = AXIS_NOUN[axis]
  const largest = ranked[0]
  const addsUp = largest.running !== undefined
  const share = (part: number) => fmtShare(part, total, decimals(metric))
  const attempts = (count: number) => plural(count, scope.one, scope.many)
  const name = <span className="text-foreground"><Identifier>{largest.label}</Identifier></span>
  const responsibleIn = `responsible in ${fmtValue(metric, largest.value)} of ${fmtValue(metric, total)} ${attempts(total)}`
  const amount = `${fmtValue(metric, largest.value)} ${metric === 'count' ? attempts(largest.value) : 'GPU-hours'}`
  if (categories === 1) {
    return <p className={SUMMARY}>{`The only ${noun.one} is `}{name}{`: ${addsUp ? amount : responsibleIn} (${share(largest.value)})`}</p>
  }
  const named = <>{name}{tied > 1 && ` and ${tied - 1} more`}</>
  if (!addsUp) {
    return (
      <p className={SUMMARY}>
        {'Largest: '}{named}{`, ${tied > 1 ? 'each ' : ''}${responsibleIn} (${share(largest.value)})`}
      </p>
    )
  }
  const shown = vitalCount ?? ranked.length
  const every = shown === categories
  return (
    <p className={SUMMARY}>
      {every ? '' : 'Top '}
      <span className="text-foreground">
        {every ? (categories === 2 ? `Both ${noun.many}` : `All ${counted(categories, noun.one, noun.many)}`) : `${shown} of ${counted(categories, noun.one, noun.many)}`}
      </span>
      {` ${plural(shown, 'accounts', 'account')} for `}
      <span className="text-foreground">{share(ranked[shown - 1].running!)}</span>
      {` of ${measured(metric, scope)} · Largest: `}
      {named}
      {tied > 1 ? `, ${amount} each (${share(largest.value)} each)` : `, ${amount} (${share(largest.value)})`}
    </p>
  )
}

// The chart and the donut in their places while the window's dashboard loads.
function ParetoPlaceholder() {
  return (
    <div aria-busy="true" className="grid grid-cols-1 divide-y lg:grid-cols-4 lg:divide-x lg:divide-y-0">
      <div className="min-w-0 px-card-x py-card-y lg:col-span-3">
        <Skeleton className="h-72" />
      </div>
      <div className="flex min-w-0 flex-col gap-3 px-card-x py-card-y">
        <Skeleton height="40" />
        <Skeleton height="16" />
      </div>
    </div>
  )
}

export function ParetoCard({
  filter,
  setFilter,
  symptomLedger,
  nodeLedger,
  userLedger,
  ruleLedger,
  outcomeLedger,
  categoryCounts,
  largestTies,
  loading = false,
  ledgersLoading = false,
  failed = false,
  ledgersFailed = false,
}: {
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
  symptomLedger: SymptomLedgerEntry[]
  nodeLedger: NodeLedgerEntry[]
  userLedger: UserLedgerEntry[]
  ruleLedger: RuleLedgerEntry[]
  outcomeLedger: OutcomeLedgerEntry[]
  /** How many categories each grouping has in all; absent until the ledgers arrive. */
  categoryCounts?: DashboardPayload['categoryCounts']
  /** How many categories tie for largest in each grouping and measure; absent until the ledgers arrive. */
  largestTies?: DashboardPayload['largestTies']
  /** The window's dashboard is on its way: the header and its choices stay, the body waits. */
  loading?: boolean
  /**
   * Only the category ledgers are on their way (another outcome of the same
   * window): the donut and its legend stay, the bars and the ranked list wait.
   */
  ledgersLoading?: boolean
  /** The window's dashboard could not be loaded: the section says so instead of ranking nothing. */
  failed?: boolean
  /**
   * Only the category ledgers could not be loaded (the window's outcome split
   * is at hand): the bars' pane says so, the donut and its legend stay.
   */
  ledgersFailed?: boolean
}) {
  const axis = filter.axis
  const metric = filter.metric
  const groupById = useId()
  const measureId = useId()
  const [rankedOpen, setRankedOpen] = useState(false)

  const series = useMemo(
    () => buildParetoSeries(axis, metric, symptomLedger, nodeLedger, userLedger, ruleLedger),
    [axis, metric, symptomLedger, nodeLedger, userLedger, ruleLedger],
  )
  const slices = useMemo(() => outcomeSlices(outcomeLedger, metric), [outcomeLedger, metric])
  // The failures every share is of: the donut's total (the window's outcome
  // split) for the outcome chosen, which is the ledgers' own scope, or for
  // every failure outcome.
  const total = useMemo(
    () => sumOf(slices.filter((slice) => filter.outcome === undefined || slice.key === filter.outcome)),
    [slices, filter.outcome],
  )
  const addsUp = sharesAddUp(axis, metric)
  const { rows: ranked, vitalCount } = useMemo(() => rankSeries(series, total, addsUp, decimals(metric)), [series, total, addsUp, metric])
  const scope = scopeOf(filter.outcome)
  // With no failure at all, one empty state stands for the chart, the donut
  // and the summary; nothing is called empty before the ledgers arrive.
  const noFailures = !loading && !ledgersLoading && ranked.length === 0 && slices.length === 0

  const noun = AXIS_NOUN[axis]
  const dimKey = AXIS_DIM_KEY[axis]
  const selectedCategory = (filter[dimKey] as string | undefined) ?? undefined
  const outcomeSelected = filter.outcome ?? null
  const hasSelection = selectedCategory !== undefined || outcomeSelected !== null

  const onCategoryClick = (label: string) => {
    setFilter(withDimension(filter, dimKey, label === selectedCategory ? undefined : label))
  }
  const onOutcomeSelect = (k: OutcomeKey | null) => {
    setFilter(withDimension(filter, 'outcome', k ?? undefined))
  }
  const clearSelection = () => {
    let next = filter
    if (selectedCategory) next = withDimension(next, dimKey, undefined)
    if (outcomeSelected) next = withDimension(next, 'outcome', undefined)
    setFilter(next)
  }

  return (
    <SectionCard
      title="Failure concentration"
      description={addsUp
        ? `${capitalized(measured(metric, scope))} ranked by ${noun.one}${axis === 'node' ? ', each with an even share of the GPU-hours of every attempt that names it' : ''}, the ten largest; the line is their running share of all ${measured(metric, scope)}.`
        : `${capitalized(measured(metric, scope))} ranked by ${noun.one}, the ten largest. An attempt can name several responsible nodes, or none, so their counts do not add up and no running share is drawn.`}
      flush
      empty={
        failed
          ? { title: 'Failure concentration could not be loaded', description: 'Refresh to try again.' }
          : noFailures
            ? { title: 'No failures to rank', description: 'No failed attempts for this filter.', status: true }
            : undefined
      }
      action={
        <div className="flex flex-wrap items-center justify-end gap-x-4 gap-y-1">
          {/* Each label names a group of toggles, not a field, so it is no <label>. */}
          <div className="flex min-w-0 items-start gap-2">
            <Label asChild><span id={groupById} className="flex h-7 shrink-0 items-center">Group by</span></Label>
            <ToggleGroup
              type="single"
              aria-labelledby={groupById}
              value={axis}
              onValueChange={(v) => v && setFilter({ ...filter, axis: v as ParetoAxis })}
            >
              <ToggleGroupItem value="symptom">Root symptom</ToggleGroupItem>
              <ToggleGroupItem value="node">Responsible node</ToggleGroupItem>
              <ToggleGroupItem value="user">User</ToggleGroupItem>
              <ToggleGroupItem value="rule">Rule</ToggleGroupItem>
            </ToggleGroup>
          </div>
          <div className="flex min-w-0 items-start gap-2">
            <Label asChild><span id={measureId} className="flex h-7 shrink-0 items-center">Measure</span></Label>
            <ToggleGroup
              type="single"
              aria-labelledby={measureId}
              value={metric}
              onValueChange={(v) => v && setFilter({ ...filter, metric: v as ParetoMetric })}
            >
              <ToggleGroupItem value="count">Count</ToggleGroupItem>
              <ToggleGroupItem value="nodeHours">GPU-hours</ToggleGroupItem>
            </ToggleGroup>
          </div>
          {/* Always in its place at the end, so the choices before it never
              move as a selection comes and goes. With nothing selected it is
              unavailable but stays focusable (aria-disabled, as the pagination
              ends are), so once it has cleared a selection it keeps focus. */}
          <Button
            variant="ghost"
            aria-disabled={!hasSelection || undefined}
            onClick={() => {
              if (hasSelection) clearSelection()
            }}
          >
            Clear selection
          </Button>
        </div>
      }
    >
      {loading ? <ParetoPlaceholder /> : (
        <>
          <div className="grid grid-cols-1 divide-y lg:grid-cols-4 lg:divide-x lg:divide-y-0">
            <div aria-busy={ledgersLoading || undefined} className="min-w-0 px-card-x py-card-y lg:col-span-3">
              {ledgersLoading ? (
                <Skeleton className="h-72" />
              ) : ledgersFailed ? (
                <p className="flex h-72 items-center justify-center p-6 text-center text-foreground-muted">
                  {`Failures by ${noun.one} could not be loaded. Refresh to try again.`}
                </p>
              ) : (
                <ParetoChart
                  axis={axis}
                  metric={metric}
                  ranked={ranked}
                  total={total}
                  selectedCategory={selectedCategory}
                  outcomeHighlight={outcomeSelected}
                  onCategoryClick={onCategoryClick}
                />
              )}
            </div>
            <div className="min-w-0 px-card-x py-card-y">
              <OutcomeDonut
                slices={slices}
                selected={outcomeSelected}
                onSelect={onOutcomeSelect}
                metric={metric}
              />
            </div>
          </div>
          {/* Only categories that exist have a concentration and a ranked list. */}
          {!ledgersLoading && ranked.length > 0 && categoryCounts && largestTies && (
            <Collapsible open={rankedOpen} onOpenChange={setRankedOpen}>
              <CardFooter className="justify-between border-t">
                <ConcentrationSummary
                  axis={axis}
                  metric={metric}
                  scope={scope}
                  ranked={ranked}
                  vitalCount={vitalCount}
                  total={total}
                  categories={categoryCounts[axis]}
                  tied={largestTies[axis][metric]}
                />
                <CollapsibleTrigger asChild>
                  <Button variant="ghost">
                    <ChevronRight data-icon="inline-start" className="transition-transform group-aria-expanded/button:rotate-90 motion-reduce:transition-none" />
                    Ranked list
                  </Button>
                </CollapsibleTrigger>
              </CardFooter>
              <CollapsibleContent className="border-t">
                <RankedList
                  axis={axis}
                  metric={metric}
                  ranked={ranked}
                  total={total}
                  selectedCategory={selectedCategory}
                  onCategoryClick={onCategoryClick}
                />
              </CollapsibleContent>
            </Collapsible>
          )}
        </>
      )}
    </SectionCard>
  )
}
