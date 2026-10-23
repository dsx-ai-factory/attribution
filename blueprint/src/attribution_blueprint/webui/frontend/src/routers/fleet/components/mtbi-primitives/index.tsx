// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { OutcomeKey } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import { OUTCOMES } from '@/lib/attribution-data'
import { withDimension } from '@/lib/filter-state'
import { Toggle } from '@/components/ui/toggle'
import { OutcomeDot } from '@/components/outcome'

/** How a filter change asks the page to show its result. */
export interface FilterChangeOptions {
  /** Bring the run list into view once it lists the new filter's runs. */
  revealRuns?: boolean
}

// The trend's drawing box. The line stretches to whatever box its caller
// gives it (preserveAspectRatio none, at a constant stroke), so this only sets
// its proportions and its 3px inset from top and bottom.
const SPARKLINE_WIDTH = 60
const SPARKLINE_HEIGHT = 20

// A neutral trend line; given an outcome it takes that outcome's tone instead.
// A trend needs two points: one day alone draws nothing rather than a stray dot.
export function MtbiSparkline({
  values,
  outcome,
  showDot = true,
}: {
  values: number[]
  outcome?: OutcomeKey
  showDot?: boolean
}) {
  if (values.length < 2) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const stepX = SPARKLINE_WIDTH / (values.length - 1)
  const pts = values.map<[number, number]>((v, i) => [
    i * stepX,
    SPARKLINE_HEIGHT - ((v - min) / range) * (SPARKLINE_HEIGHT - 6) - 3,
  ])
  const path = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`)
    .join(' ')
  const last = pts[pts.length - 1]
  const color = outcome ? 'var(--tone)' : 'var(--foreground-secondary)'
  return (
    <svg
      data-outcome={outcome}
      viewBox={`0 0 ${SPARKLINE_WIDTH} ${SPARKLINE_HEIGHT}`}
      preserveAspectRatio="none"
      className="block w-full h-full"
    >
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        vectorEffect="non-scaling-stroke"
      />
      {showDot && <circle cx={last[0]} cy={last[1]} r="2" fill={color} />}
    </svg>
  )
}

/**
 * One outcome's failure count as a filter toggle: pressing it sets that
 * outcome, whose runs are all failed ones, and brings them into view; pressing
 * it again drops the outcome and nothing else. Its name says what pressing
 * does and keeps the visible count and full outcome label. While the count
 * loads, or when it could not be loaded (undefined), the toggle stays, named
 * by its outcome alone.
 */
export function FailureFilterButton({
  outcome,
  count,
  filter,
  setFilter,
}: {
  outcome: OutcomeKey
  count?: number
  filter: FleetFilter
  setFilter: (next: FleetFilter, options?: FilterChangeOptions) => void
}) {
  const { label } = OUTCOMES[outcome]
  return (
    <Toggle
      pressed={filter.outcome === outcome}
      aria-label={count === undefined ? `Filter runs: ${label}` : `Filter runs: ${count} ${label}`}
      onPressedChange={(pressed) => {
        if (!pressed) {
          setFilter(withDimension(filter, 'outcome', undefined))
          return
        }
        setFilter(withDimension(filter, 'outcome', outcome), { revealRuns: true })
      }}
      className="justify-start gap-2"
    >
      <OutcomeDot outcome={outcome} />
      {count !== undefined && <span className="num">{count}</span>}
      {label}
    </Toggle>
  )
}
