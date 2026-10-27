// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TrendingDown, TrendingUp } from 'lucide-react'
import type { SymptomMover } from '@/interfaces/attribution'
import { OUTCOMES } from '@/lib/attribution-data'
import type { FleetFilter } from '@/interfaces/filter'
import { withDimension } from '@/lib/filter-state'
import { plural } from '@/lib/format'
import { cn, focusRingInset, markInText, pressedStat } from '@/lib/utils'
import { Identifier } from '@/components/field'
import { OutcomeDot } from '@/components/outcome'
import { SectionCard } from '@/components/section-card'
import { MtbiSparkline } from '@/routers/fleet/components/mtbi-primitives'

// Symptoms whose failures moved most within the window, each a filter toggle
// laid out like the KPI stats: one card, hairline-split cells, pressed in the
// KPI stat's recipe (lib/utils pressedStat). Only a worsening trend takes a color (the red
// of worse); a falling one stays neutral. The page leaves this section out when
// nothing moved.
export function MoversStrip({
  movers,
  windowLabel,
  filter,
  setFilter,
}: {
  movers: SymptomMover[]
  /** The window the movement is measured in, e.g. "7d" or "all time". */
  windowLabel: string
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
}) {
  return (
    <SectionCard
      title="Movers"
      count={windowLabel}
      description="Symptoms ranked by how much their failures moved against the earlier half of the window."
      flush
    >
      <div className="grid grid-cols-1 divide-y sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-5 lg:divide-x">
        {movers.map((m) => {
          const pressed = filter.symptom === m.symptom
          const rising = m.score > 0
          const Trend = rising ? TrendingUp : TrendingDown
          return (
            <button
              key={m.symptom}
              type="button"
              aria-pressed={pressed}
              data-pressed={pressed || undefined}
              onClick={() => setFilter(withDimension(filter, 'symptom', pressed ? undefined : m.symptom))}
              className={cn(
                'flex min-w-0 cursor-pointer flex-col gap-1 px-card-x py-card-y text-left transition-colors not-data-pressed:hover:bg-background-active',
                pressedStat,
                focusRingInset,
              )}
            >
              <span className="flex items-start justify-between gap-2">
                <span className="min-w-0 break-words text-sm font-medium text-foreground"><Identifier>{m.symptom}</Identifier></span>
                {/* A mark beside the name, on its first line and centered on its lowercase. */}
                <span className="shrink-0 text-sm">
                  <Trend
                    role="img"
                    aria-label={rising ? 'Rising within window' : 'Falling within window'}
                    className={cn('inline size-3.5', rising ? 'text-trend-worse' : 'text-trend-better', markInText)}
                  />
                </span>
              </span>
              <span className="flex flex-wrap items-center gap-x-3 text-xs text-foreground-muted">
                <span>
                  <span className="num text-foreground-secondary">{m.count}</span> {plural(m.count, 'run', 'runs')}
                </span>
                <span className="flex items-center gap-1.5">
                  <OutcomeDot outcome={m.dominantOutcome} />
                  {OUTCOMES[m.dominantOutcome].label}
                </span>
              </span>
              <span aria-hidden="true" className="h-5">
                <MtbiSparkline values={m.spark} outcome={m.dominantOutcome} showDot={false} />
              </span>
            </button>
          )
        })}
      </div>
    </SectionCard>
  )
}
