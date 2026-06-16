import type { OutcomeKey } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import { withDimension } from '@/lib/filter-state'
import { Badge } from '@/components/ui/badge'

/**
 * Scrolls the Runs table into view. Caller is responsible for first
 * applying any filter change so the smooth-scroll lands on filtered rows.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function scrollToRuns() {
  const el = document.getElementById('runs-section')
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

export function MtbiSparkline({
  values,
  color = 'var(--color-primary)',
  width = 60,
  height = 20,
  showDot = true,
}: {
  values: number[]
  color?: string
  width?: number
  height?: number
  showDot?: boolean
}) {
  if (!values || values.length === 0) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const stepX = width / Math.max(1, values.length - 1)
  const pts = values.map<[number, number]>((v, i) => [
    i * stepX,
    height - ((v - min) / range) * (height - 6) - 3,
  ])
  const path = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`)
    .join(' ')
  const last = pts[pts.length - 1]
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
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

export function DeltaPill({
  now,
  prior,
  higherIsBetter = true,
}: {
  now: number
  prior: number
  higherIsBetter?: boolean
}) {
  const delta = now - prior
  const pct = prior === 0 ? 0 : (delta / prior) * 100
  const isUp = delta >= 0
  const isGood = higherIsBetter ? isUp : !isUp
  return (
    <Badge
      variant={isGood ? 'default' : 'outline'}
      className={`font-mono ${
        isGood ? 'bg-primary/15 text-primary border-transparent' : 'text-foreground border-zinc-700'
      }`}
    >
      {isUp ? '▲' : '▼'} {Math.abs(pct).toFixed(0)}%
    </Badge>
  )
}

/**
 * Clickable failure counter: applies an outcome filter and scrolls to Runs.
 * Spreads <a> props so Radix's asChild Slot can forward to it.
 */
export function FailureLink({
  outcome,
  filter,
  setFilter,
  children,
  className,
  onClick,
  ...props
}: {
  outcome: OutcomeKey
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
  children?: React.ReactNode
} & Omit<React.ComponentProps<'a'>, 'children'>) {
  return (
    <a
      {...props}
      onClick={(e) => {
        e.stopPropagation()
        onClick?.(e)
        setFilter(withDimension({ ...filter, status: 'failed' }, 'outcome', outcome))
        scrollToRuns()
      }}
      className={[
        'cursor-pointer underline decoration-foreground-muted underline-offset-2 hover:decoration-primary hover:text-primary',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {children}
    </a>
  )
}
