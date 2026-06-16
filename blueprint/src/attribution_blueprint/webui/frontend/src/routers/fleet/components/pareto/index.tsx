import { useMemo, useRef } from 'react'
import Decimal from 'decimal.js'
import type { EChartsOption } from 'echarts'
import { OUTCOMES } from '@/lib/attribution-data'
import { useECharts, ECHART_COMMON } from '@/hooks/use-echarts'
import type {
  NodeLedgerEntry,
  OutcomeKey,
  OutcomeLedgerEntry,
  RuleLedgerEntry,
  Run,
  SymptomLedgerEntry,
  UserLedgerEntry,
} from '@/interfaces/attribution'
import type { FleetFilter, ParetoAxis, ParetoMetric } from '@/interfaces/filter'
import { withDimension } from '@/lib/filter-state'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { DefHover } from '@/components/def-hover'
import { fmtPercent } from '@/lib/format'
import { attemptGpuHours } from '@/lib/metric'

interface ParetoRow {
  label: string
  value: number
  outcomes: Partial<Record<OutcomeKey, number>>
  /** Axis-specific extras for tooltip rendering. */
  topSymptom?: string
  nodeHours?: number
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (ch) => {
    switch (ch) {
      case '&':
        return '&amp;'
      case '<':
        return '&lt;'
      case '>':
        return '&gt;'
      case '"':
        return '&quot;'
      case "'":
        return '&#39;'
      default:
        return ch
    }
  })
}

const AXIS_DIM_KEY: Record<ParetoAxis, keyof FleetFilter> = {
  symptom: 'symptom',
  node:    'node',
  user:    'user',
  rule:    'rule',
}

function buildParetoSeries(
  axis: ParetoAxis,
  metric: ParetoMetric,
  symptomLedger: SymptomLedgerEntry[],
  nodeLedger: NodeLedgerEntry[],
  userLedger: UserLedgerEntry[],
  ruleLedger: RuleLedgerEntry[],
  runs: Run[],
): ParetoRow[] {
  const valueOf = (d: { count: number; nodeHours: number }) =>
    metric === 'count' ? d.count : d.nodeHours

  if (axis === 'node') {
    return nodeLedger.map((d) => ({
      label: d.node,
      value: valueOf(d),
      outcomes: d.outcomes,
    }))
  }
  if (axis === 'rule') {
    return ruleLedger.map((d) => ({
      label: d.rule,
      value: valueOf(d),
      outcomes: d.outcomes,
      nodeHours: d.nodeHours,
    }))
  }
  if (axis === 'user') {
    if (userLedger.length > 0) {
      return userLedger.map((d) => ({
        label: d.user,
        value: valueOf(d),
        outcomes: d.outcomes,
        nodeHours: d.nodeHours,
      }))
    }
    // Accumulate the user-axis value with Decimal (exact) like the ledgers;
    // convert to number only at the boundary so no float-accum noise re-enters
    // the downstream Decimal math (sort / cumulative % / donut / vital-few).
    const vals: Record<string, Decimal> = {}
    const outcomes: Record<string, Partial<Record<OutcomeKey, number>>> = {}
    runs.forEach((r) => {
      vals[r.user] = (vals[r.user] ?? new Decimal(0)).plus(metric === 'count' ? 1 : attemptGpuHours(r))
      outcomes[r.user] = {
        ...(outcomes[r.user] ?? {}),
        [r.outcome]: (outcomes[r.user]?.[r.outcome] ?? 0) + 1,
      }
    })
    return Object.keys(vals).map((user) => ({
      label: user,
      value: vals[user].toNumber(),
      outcomes: outcomes[user],
    }))
  }
  return symptomLedger.map((d) => ({
    label: d.symptom,
    value: valueOf(d),
    outcomes: d.outcomes,
    nodeHours: d.nodeHours,
  }))
}

function dominantOutcome(outcomes: Partial<Record<OutcomeKey, number>>): OutcomeKey {
  let best: OutcomeKey = 'hardware'
  let bestN = -1
  ;(Object.entries(outcomes) as [OutcomeKey, number][]).forEach(([k, n]) => {
    if (n > bestN) {
      best = k
      bestN = n
    }
  })
  return best
}

function ParetoChart({
  axis,
  metric,
  series,
  selectedCategory,
  outcomeHighlight,
  onCategoryClick,
}: {
  axis: ParetoAxis
  metric: ParetoMetric
  series: ParetoRow[]
  selectedCategory?: string
  outcomeHighlight: OutcomeKey | null
  onCategoryClick: (label: string) => void
}) {
  const ref = useRef<HTMLDivElement | null>(null)

  const sorted = useMemo(() => {
    const s = [...series].sort((a, b) => b.value - a.value)
    const total = s.reduce((acc, d) => acc.plus(d.value), new Decimal(0))
    const denom = total.isZero() ? new Decimal(1) : total
    const cums = s.reduce<Decimal[]>((acc, d) => {
      const prev = acc.length === 0 ? new Decimal(0) : acc[acc.length - 1]
      acc.push(prev.plus(d.value))
      return acc
    }, [])
    return s.map((d, i) => ({
      ...d,
      cumPct: cums[i].div(denom).times(100).toNumber(),
      dominant: dominantOutcome(d.outcomes),
    }))
  }, [series])
  const showEmptyResponsibleNodes = axis === 'node' && sorted.length === 0

  const option = useMemo<EChartsOption>(() => {
    const maxV = Math.max(1, ...sorted.map((d) => d.value))
    const metricLabel = metric === 'count' ? '# attempts' : 'GPU-hours'
    return {
      ...ECHART_COMMON,
      animation: true,
      animationDuration: 200,
      animationDurationUpdate: 150,
      animationEasing: 'quarticOut',
      animationEasingUpdate: 'quarticOut',
      grid: { left: 48, right: 56, top: 24, bottom: 48, containLabel: false },
      tooltip: {
        show: !showEmptyResponsibleNodes,
        trigger: 'axis',
        backgroundColor: '#0a0a0a',
        borderColor: '#27272a',
        borderWidth: 1,
        padding: [8, 10],
        textStyle: {
          color: '#fff',
          fontFamily: 'JetBrains Mono Variable, ui-monospace, monospace',
          fontSize: 11,
        },
        axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(255,255,255,0.04)' } },
        formatter: (params) => {
          const arr = params as unknown as Array<{
            seriesIndex: number
            axisValue: string
            data: {
              value: number
              dominant?: OutcomeKey
              cumPct?: number
              topSymptom?: string
              nodeHours?: number
            }
          }>
          const bar = arr.find((p) => p.seriesIndex === 0)
          if (!bar) return ''
          const v = bar.data.value
          const valueText =
            metric === 'count' ? Math.round(v).toString() : `${v.toFixed(1)}h`
          const cumPct =
            bar.data.cumPct !== undefined ? `${fmtPercent(bar.data.cumPct)}%` : '—'
          const outcomeKey = bar.data.dominant
          const outcomeLabel = outcomeKey ? OUTCOMES[outcomeKey].label : '—'
          const outcomeColor = outcomeKey ? OUTCOMES[outcomeKey].color : '#a1a1aa'
          const axisValue = escapeHtml(bar.axisValue)
          const metricLabelHtml = escapeHtml(metricLabel)
          const valueTextHtml = escapeHtml(valueText)
          const cumPctHtml = escapeHtml(cumPct)
          const outcomeLabelHtml = escapeHtml(outcomeLabel)
          const swatch = outcomeKey
            ? `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${outcomeColor};margin-right:6px;vertical-align:middle"></span>`
            : ''
          return [
            `<div style="font-weight:600;color:#fff;margin-bottom:4px;word-break:break-all">${swatch}${axisValue}</div>`,
            `<div style="color:#a1a1aa">${metricLabelHtml}: <span style="color:#fff">${valueTextHtml}</span></div>`,
            `<div style="color:#a1a1aa">cumulative: <span style="color:#76b900">${cumPctHtml}</span></div>`,
            `<div style="color:#a1a1aa">dominant outcome: <span style="color:${outcomeColor}">${outcomeLabelHtml}</span></div>`,
          ].join('')
        },
      },
      xAxis: {
        type: 'category',
        show: !showEmptyResponsibleNodes,
        data: sorted.map((d) => d.label),
        axisLabel: {
          color: '#a1a1aa',
          fontSize: 10,
          rotate: 0,
          hideOverlap: false,
          formatter: (v: string) => (v.length > 14 ? v.slice(0, 13) + '…' : v),
        },
        axisLine: { lineStyle: { color: '#27272a' } },
        axisTick: { show: false },
      },
      yAxis: [
        {
          type: 'value',
          show: !showEmptyResponsibleNodes,
          name: metricLabel,
          nameTextStyle: { color: '#a1a1aa', fontSize: 10 },
          max: maxV,
          splitLine: { lineStyle: { color: '#27272a', type: 'dashed' } },
          axisLabel: { color: '#a1a1aa', fontSize: 10 },
          axisLine: { show: false },
        },
        {
          type: 'value',
          show: !showEmptyResponsibleNodes,
          name: 'cumulative %',
          nameTextStyle: { color: '#57920d', fontSize: 10 },
          min: 0,
          max: 100,
          interval: 25,
          splitLine: { show: false },
          axisLabel: { color: '#76b900', fontSize: 10, formatter: '{value}%' },
          axisLine: { show: false },
        },
      ],
      series: [
        {
          name: 'value',
          type: 'bar',
          data: sorted.map((d) => {
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
            return {
              value: d.value,
              itemStyle: {
                color: OUTCOMES[d.dominant].color,
                opacity,
              },
              name: d.label,
              dominant: d.dominant,
              cumPct: d.cumPct,
            }
          }),
          barWidth: '55%',
          label: {
            show: true,
            position: 'top',
            color: '#fff',
            fontSize: 10,
            fontFamily: 'JetBrains Mono Variable, ui-monospace, monospace',
            fontWeight: 600,
            formatter: (p) => {
              const v = Number((p as { value: number }).value)
              return metric === 'count' ? String(v) : v.toFixed(1)
            },
          },
        },
        {
          name: 'cumulative',
          type: 'line',
          yAxisIndex: 1,
          data: sorted.map((d) => d.cumPct),
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: { color: '#76b900', width: 1.5 },
          itemStyle: { color: '#76b900' },
          smooth: false,
          markLine: {
            silent: true,
            symbol: 'none',
            data: showEmptyResponsibleNodes
              ? []
              : [
                  {
                    yAxis: 80,
                    lineStyle: { color: '#76b900', type: 'dashed', width: 1, opacity: 0.6 },
                    label: { color: '#76b900', formatter: '80%', position: 'insideEndTop' },
                  },
                ],
          },
        },
      ],
    }
  }, [sorted, metric, showEmptyResponsibleNodes, selectedCategory, outcomeHighlight])

  useECharts(ref, option, {
    click: (p) => {
      if (p.seriesIndex !== 0) return
      const label = (p.data as { name?: string })?.name
      if (label) onCategoryClick(label)
    },
  })
  return (
    <div className="relative w-full h-72">
      <div ref={ref} className="absolute inset-0" />
      {showEmptyResponsibleNodes && (
        <div
          role="status"
          className="pointer-events-none absolute inset-0 flex items-center justify-center px-6 text-center font-mono text-xs font-medium text-foreground-muted"
        >
          No responsible nodes for this filter.
        </div>
      )}
    </div>
  )
}

function OutcomeDonut({
  series,
  outcomeLedger,
  selected,
  onSelect,
  metric,
}: {
  series: ParetoRow[]
  outcomeLedger: OutcomeLedgerEntry[]
  selected: OutcomeKey | null
  onSelect: (k: OutcomeKey | null) => void
  metric: ParetoMetric
}) {
  const ref = useRef<HTMLDivElement | null>(null)

  const slices = useMemo(() => {
    if (outcomeLedger.length > 0) {
      const totalDec = outcomeLedger.reduce(
        (total, entry) => total.plus(metric === 'count' ? entry.count : entry.nodeHours),
        new Decimal(0),
      )
      const denom = totalDec.isZero() ? new Decimal(1) : totalDec
      return outcomeLedger
        .map((entry) => {
          const value = metric === 'count' ? entry.count : entry.nodeHours
          return {
            key: entry.outcome,
            full: OUTCOMES[entry.outcome].label,
            color: OUTCOMES[entry.outcome].color,
            value,
            pct: new Decimal(value).div(denom).toNumber(),
          }
        })
        .filter((slice) => slice.value > 0)
    }

    const buckets: Partial<Record<OutcomeKey, Decimal>> = {}
    series.forEach((row) => {
      const t = Object.values(row.outcomes).reduce(
        (a, b) => a.plus(b ?? 0),
        new Decimal(0),
      )
      if (t.lte(0)) return
      ;(Object.entries(row.outcomes) as [OutcomeKey, number][]).forEach(([oc, n]) => {
        const share = new Decimal(n).div(t).times(row.value)
        buckets[oc] = (buckets[oc] ?? new Decimal(0)).plus(share)
      })
    })
    const totalDec = Object.values(buckets).reduce(
      (a, b) => a.plus(b ?? 0),
      new Decimal(0),
    )
    const denom = totalDec.isZero() ? new Decimal(1) : totalDec
    const order: OutcomeKey[] = ['hardware', 'software', 'multi', 'user', 'undetermined', 'pass']
    return order
      .filter((k) => buckets[k])
      .map((k) => ({
        key: k,
        full: OUTCOMES[k].label,
        color: OUTCOMES[k].color,
        value: buckets[k]!.toNumber(),
        pct: buckets[k]!.div(denom).toNumber(),
      }))
  }, [series, outcomeLedger, metric])

  const total = slices.reduce((a, b) => a + b.value, 0) || 1
  const sel = selected ? slices.find((s) => s.key === selected) : undefined
  const centerNum = sel ? sel.value : total
  const centerNumText = Number.isInteger(centerNum)
    ? String(centerNum)
    : centerNum.toFixed(1)
  const centerLbl = sel
    ? sel.full
    : metric === 'count'
      ? 'total attempts'
      : 'GPU-hours'
  const centerPct = sel ? `${fmtPercent(new Decimal(sel.pct).times(100))}% of total` : ''

  const option = useMemo<EChartsOption>(
    () => ({
      ...ECHART_COMMON,
      animation: true,
      animationDuration: 200,
      animationDurationUpdate: 150,
      animationEasing: 'quarticOut',
      animationEasingUpdate: 'quarticOut',
      tooltip: {
        trigger: 'item',
        backgroundColor: '#0a0a0a',
        borderColor: '#27272a',
        borderWidth: 1,
        padding: [8, 10],
        textStyle: {
          color: '#fff',
          fontFamily: 'JetBrains Mono Variable, ui-monospace, monospace',
          fontSize: 11,
        },
        formatter: (params) => {
          const p = params as {
            name: string
            value: number
            percent: number
            color?: string
          }
          const swatch = `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${p.color};margin-right:6px;vertical-align:middle"></span>`
          const v = Number.isInteger(p.value) ? p.value : p.value.toFixed(1)
          const unit = metric === 'count' ? 'attempts' : 'GPU-hours'
          const name = escapeHtml(p.name)
          const value = escapeHtml(String(v))
          return [
            `<div style="font-weight:600;color:#fff;margin-bottom:4px">${swatch}${name}</div>`,
            `<div style="color:#a1a1aa">${unit}: <span style="color:#fff">${value}</span></div>`,
            `<div style="color:#a1a1aa">share: <span style="color:${p.color}">${p.percent.toFixed(0)}%</span> of total</div>`,
          ].join('')
        },
      },
      series: [
        {
          name: 'outcome',
          type: 'pie',
          radius: ['56%', '85%'],
          center: ['50%', '50%'],
          avoidLabelOverlap: true,
          label: {
            show: true,
            position: 'center',
            formatter: () => `{num|${centerNumText}}\n{lbl|${centerLbl.toUpperCase()}}${centerPct ? `\n{pct|${centerPct}}` : ''}`,
            rich: {
              num: {
                color: sel ? sel.color : '#fff',
                fontSize: 20,
                fontWeight: 'bold' as const,
                fontFamily: 'JetBrains Mono Variable, ui-monospace, monospace',
                lineHeight: 26,
              },
              lbl: {
                color: '#a1a1aa',
                fontSize: 9,
                fontFamily: 'JetBrains Mono Variable, ui-monospace, monospace',
                lineHeight: 14,
              },
              pct: {
                color: sel ? sel.color : '#a1a1aa',
                fontSize: 10,
                fontFamily: 'JetBrains Mono Variable, ui-monospace, monospace',
                lineHeight: 14,
              },
            },
          },
          labelLine: { show: false },
          itemStyle: {
            borderColor: '#121212',
            borderWidth: 1.5,
          },
          data: slices.map((s) => ({
            name: s.full,
            value: s.value,
            itemStyle: {
              color: s.color,
              opacity: selected && selected !== s.key ? 0.18 : 1,
            },
            __key: s.key,
          })),
          emphasis: { scale: false, focus: 'none' as const },
        },
      ],
    }),
    [slices, selected, centerNumText, centerLbl, centerPct, sel, metric],
  )

  useECharts(ref, option, {
    click: (p) => {
      const k = (p.data as { __key?: OutcomeKey })?.__key
      if (k) onSelect(k === selected ? null : k)
    },
  })

  return (
    <div className="flex flex-col gap-3">
      <div ref={ref} className="w-full h-44" />
      <div className="w-full flex flex-col gap-1 font-mono text-xs">
        {slices.map((s) => {
          const isSel = selected === s.key
          const dim = selected && !isSel ? 'opacity-35' : ''
          return (
            <div
              key={s.key}
              className={`flex items-center gap-2 px-1 py-0.5 rounded-sm cursor-pointer hover:bg-background-active transition duration-150 ${dim}`}
              onClick={() => onSelect(isSel ? null : s.key)}
            >
              <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: s.color }} />
              <span className="text-foreground-secondary whitespace-nowrap overflow-hidden text-ellipsis flex-1">
                {s.full}
              </span>
              <span className="text-foreground tabular-nums w-8 text-right">
                {Number.isInteger(s.value) ? s.value : s.value.toFixed(1)}
              </span>
              <span className="text-foreground-muted tabular-nums w-10 text-right">
                {fmtPercent(new Decimal(s.pct).times(100))}%
              </span>
            </div>
          )
        })}
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
  runs,
}: {
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
  symptomLedger: SymptomLedgerEntry[]
  nodeLedger: NodeLedgerEntry[]
  userLedger: UserLedgerEntry[]
  ruleLedger: RuleLedgerEntry[]
  outcomeLedger: OutcomeLedgerEntry[]
  runs: Run[]
}) {
  const axis = filter.axis
  const metric = filter.metric

  const series = useMemo(
    () => buildParetoSeries(axis, metric, symptomLedger, nodeLedger, userLedger, ruleLedger, runs),
    [axis, metric, symptomLedger, nodeLedger, userLedger, ruleLedger, runs],
  )

  const sorted = useMemo(() => [...series].sort((a, b) => b.value - a.value), [series])
  const vital = useMemo(() => {
    const total = sorted.reduce((s, d) => s.plus(d.value), new Decimal(0))
    const denom = total.isZero() ? new Decimal(1) : total
    let running = new Decimal(0)
    let eighty = -1
    for (let i = 0; i < sorted.length; i += 1) {
      running = running.plus(sorted[i].value)
      if (running.div(denom).times(100).gte(80)) {
        eighty = i
        break
      }
    }
    return sorted.slice(0, Math.max(1, eighty + 1))
  }, [sorted])
  const vitalNoun =
    axis === 'node' ? 'nodes' :
    axis === 'rule' ? 'rules' :
    axis === 'user' ? 'users' :
                      'symptoms'

  const dimKey = AXIS_DIM_KEY[axis]
  const selectedCategory = (filter[dimKey] as string | undefined) ?? undefined
  const outcomeSelected = filter.outcome ?? null

  const onCategoryClick = (label: string) => {
    const next = withDimension(filter, dimKey, label === selectedCategory ? undefined : label)
    setFilter(next)
  }
  const onOutcomeSelect = (k: OutcomeKey | null) => {
    setFilter(withDimension(filter, 'outcome', k ?? undefined))
  }

  return (
    <Card size="sm">
      <CardHeader className="flex flex-row items-baseline gap-3">
        <CardTitle className="text-xs font-semibold font-mono tracking-wide uppercase">
          Failure Concentration · Pareto + Outcome
        </CardTitle>
        <span className="text-foreground-muted text-xs">
          Bars: {metric === 'count' ? 'attempt count' : 'GPU-hours'}. Line: <DefHover term="cumulative-pct">cumulative %</DefHover>. Donut: <DefHover term="outcome-share">outcome share</DefHover>. Click a bar to filter the runs table; click a donut slice to filter by outcome.
        </span>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2 my-2 font-mono text-xs min-h-8">
          <span className="text-xs uppercase tracking-wider text-zinc-700">x-axis</span>
          <ToggleGroup
            type="single"
            value={axis}
            onValueChange={(v) => v && setFilter({ ...filter, axis: v as ParetoAxis })}
            variant="outline"
            size="sm"
          >
            <ToggleGroupItem value="symptom">
              <DefHover term="root-symptom">Root symptom</DefHover>
            </ToggleGroupItem>
            <ToggleGroupItem value="node">
              <DefHover term="responsible-node">Responsible node</DefHover>
            </ToggleGroupItem>
            <ToggleGroupItem value="user">User</ToggleGroupItem>
            <ToggleGroupItem value="rule">
              <DefHover term="rule">Rule</DefHover>
            </ToggleGroupItem>
          </ToggleGroup>
          <span className="ml-4 text-xs uppercase tracking-wider text-zinc-700">metric</span>
          <ToggleGroup
            type="single"
            value={metric}
            onValueChange={(v) => v && setFilter({ ...filter, metric: v as ParetoMetric })}
            variant="outline"
            size="sm"
          >
            <ToggleGroupItem value="count">
              <DefHover term="count">Count</DefHover>
            </ToggleGroupItem>
            <ToggleGroupItem value="nodeHours">
              <DefHover term="gpu-hours">GPU-hours</DefHover>
            </ToggleGroupItem>
          </ToggleGroup>
          {(selectedCategory || outcomeSelected) && (
            <Button
              variant="outline"
              size="sm"
              className="ml-auto"
              onClick={() => {
                let next = filter
                if (selectedCategory) next = withDimension(next, dimKey, undefined)
                if (outcomeSelected) next = withDimension(next, 'outcome', undefined)
                setFilter(next)
              }}
            >
              clear pareto filter
            </Button>
          )}
        </div>
        <div className="flex items-stretch gap-5 my-2 h-72">
          <div className="min-w-0 flex-1">
            <ParetoChart
              axis={axis}
              metric={metric}
              series={series}
              selectedCategory={selectedCategory}
              outcomeHighlight={outcomeSelected}
              onCategoryClick={onCategoryClick}
            />
          </div>
          <div className="border-l border-border pl-5 w-64 shrink-0 flex items-center">
            <OutcomeDonut
              series={series}
              outcomeLedger={outcomeLedger}
              selected={outcomeSelected}
              onSelect={onOutcomeSelect}
              metric={metric}
            />
          </div>
        </div>
        <div className="flex flex-wrap gap-x-5 gap-y-1 mt-2 font-mono text-xs text-foreground-muted min-h-5">
          {vital.length === 0 ? (
            <div>No responsible {vitalNoun} for this filter.</div>
          ) : (
            <>
              <div>
                <DefHover term="vital-few">Vital few</DefHover>: <b className="text-foreground font-medium">{vital.length}</b> {vitalNoun}
              </div>
              <div>
                account for <b className="text-foreground font-medium">~80%</b> of{' '}
                {metric === 'count' ? 'failed attempts' : 'lost GPU-hours'}
              </div>
              {vital[0] && (
                <div>
                  · top contributor:{' '}
                  <b className="text-foreground font-medium">{vital[0].label}</b> (
                  {metric === 'count' ? vital[0].value : `${vital[0].value.toFixed(1)}h`})
                </div>
              )}
            </>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
