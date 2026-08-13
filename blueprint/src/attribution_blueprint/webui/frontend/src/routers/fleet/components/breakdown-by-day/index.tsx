import { useMemo, useRef } from 'react'
import type { EChartsOption } from 'echarts'
import { useECharts, ECHART_COMMON } from '@/hooks/use-echarts'
import { OUTCOMES, SYMPTOM_PALETTE, OTHER_COLOR, breakdownHighlight } from '@/lib/attribution-data'
import type { DailyBreakdown, OutcomeKey } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import { withDimension } from '@/lib/filter-state'
import { Card, CardContent } from '@/components/ui/card'

const MONO = 'JetBrains Mono Variable, ui-monospace, monospace'

function colorFor(symptom: string, i: number): string {
  return symptom === 'Other' ? OTHER_COLOR : SYMPTOM_PALETTE[i % SYMPTOM_PALETTE.length]
}

function BreakdownChart({
  breakdown,
  selectedSymptom,
  onSymptomClick,
}: {
  breakdown: DailyBreakdown
  selectedSymptom?: string
  onSymptomClick: (symptom: string) => void
}) {
  const ref = useRef<HTMLDivElement | null>(null)

  const option = useMemo<EChartsOption>(() => {
    const { days, series } = breakdown
    const showTotals = days.length <= 10
    const totals = days.map((_, dayIndex) =>
      series.reduce((sum, s) => sum + (s.data[dayIndex] || 0), 0),
    )
    const topSeriesByDay = days.map((_, dayIndex) => {
      let top = -1
      series.forEach((s, seriesIndex) => {
        if ((s.data[dayIndex] || 0) > 0) top = seriesIndex
      })
      return top
    })
    return {
      ...ECHART_COMMON,
      animation: true,
      animationDuration: 200,
      grid: { left: 8, right: 12, top: 12, bottom: 40, containLabel: true },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(255,255,255,0.04)' } },
        backgroundColor: '#0a0a0a',
        borderColor: '#27272a',
        borderWidth: 1,
        padding: [8, 10],
        textStyle: { color: '#fff', fontFamily: MONO, fontSize: 11 },
        formatter: (params) => {
          const arr = params as unknown as Array<{
            axisValue: string; seriesName: string; value: number; color: string
          }>
          const total = arr.reduce((a, p) => a + (p.value || 0), 0)
          if (total === 0) return ''
          const head = `<div style="font-weight:600;color:#fff;margin-bottom:4px">${arr[0].axisValue}</div>`
          const body = arr
            .filter((p) => p.value > 0)
            .map((p) =>
              `<div style="color:#a1a1aa"><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${p.color};margin-right:6px;vertical-align:middle"></span>${p.seriesName}: <span style="color:#fff">${p.value}</span></div>`,
            )
            .join('')
          const foot = `<div style="color:#a1a1aa;margin-top:4px;border-top:1px solid #27272a;padding-top:4px">total: <span style="color:#fff">${total}</span></div>`
          return head + body + foot
        },
      },
      legend: {
        type: 'scroll',
        bottom: 0,
        itemWidth: 8,
        itemHeight: 8,
        textStyle: { color: '#a1a1aa', fontSize: 10, fontFamily: MONO },
        pageTextStyle: { color: '#a1a1aa' },
        pageIconColor: '#a1a1aa',
        pageIconInactiveColor: '#3f3f46',
      },
      xAxis: {
        type: 'category',
        data: days.map((d) => d.slice(5)),
        axisLabel: { color: '#a1a1aa', fontSize: 10 },
        axisLine: { lineStyle: { color: '#27272a' } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        minInterval: 1,
        splitLine: { lineStyle: { color: '#27272a', type: 'dashed' } },
        axisLabel: { color: '#a1a1aa', fontSize: 10 },
        axisLine: { show: false },
      },
      series: series.map((s, i) => ({
        name: s.symptom,
        type: 'bar' as const,
        stack: 'sym',
        data: s.data.map((value, dayIndex) => {
          if (!showTotals || value <= 0 || topSeriesByDay[dayIndex] !== i) return value
          return {
            value,
            label: {
              show: true,
              position: 'top',
              color: '#f4f4f5',
              fontFamily: MONO,
              fontSize: 10,
              formatter: String(totals[dayIndex]),
            },
          }
        }),
        label: { show: false },
        itemStyle: {
          color: colorFor(s.symptom, i),
          opacity: selectedSymptom && selectedSymptom !== s.symptom ? 0.25 : 1,
        },
        emphasis: { focus: 'series' as const },
      })),
    }
  }, [breakdown, selectedSymptom])

  useECharts(ref, option, {
    click: (p) => {
      const name = (p as { seriesName?: string }).seriesName
      if (name) onSymptomClick(name)
    },
  })

  return <div ref={ref} className="w-full h-64" />
}

function EmptyChart({ label }: { label: string }) {
  return (
    <div className="w-full h-64 flex items-center justify-center font-mono text-xs text-foreground-muted">
      No {label} results in this window
    </div>
  )
}

export function BreakdownByDay({
  hardware,
  software,
  filter,
  setFilter,
}: {
  hardware: DailyBreakdown
  software: DailyBreakdown
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
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

  return (
    <Card size="sm">
      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {charts.map(({ outcome, breakdown }) => (
            <div
              key={outcome}
              className={`transition-opacity duration-200 ${paneMuted(outcome) ? 'opacity-60' : ''}`}
            >
              <div className="font-mono text-[11px] uppercase tracking-wider text-foreground-muted mb-1">
                {OUTCOMES[outcome].label}
              </div>
              {breakdown.total === 0 ? (
                <EmptyChart label={OUTCOMES[outcome].label} />
              ) : (
                <BreakdownChart
                  breakdown={breakdown}
                  selectedSymptom={breakdownHighlight(breakdown, filter.symptom)}
                  onSymptomClick={onSymptomClick}
                />
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
