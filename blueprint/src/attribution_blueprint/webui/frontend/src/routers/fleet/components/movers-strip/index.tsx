import type { SymptomMover } from '@/interfaces/attribution'
import { OUTCOMES } from '@/lib/attribution-data'
import type { FleetFilter } from '@/interfaces/filter'
import { withDimension } from '@/lib/filter-state'
import { Card, CardContent } from '@/components/ui/card'
import { MtbiSparkline } from '@/routers/fleet/components/mtbi-primitives'

// Direction colors: rising failures = worse (red), falling = better (brand green).
const RISING = OUTCOMES.hardware.color // rising failures → hardware-red token (kept in sync via OUTCOMES)
const FALLING = 'var(--color-primary)'

export function MoversStrip({
  movers,
  filter,
  setFilter,
}: {
  movers: SymptomMover[]
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
}) {
  if (movers.length === 0) {
    return (
      <div className="mb-5 rounded-md border border-border bg-background-active px-3 py-3 font-mono text-xs text-foreground-muted">
        No symptoms moved materially in this window.
      </div>
    )
  }
  return (
    <div
      className="grid gap-2.5 mb-5"
      style={{ gridTemplateColumns: `repeat(${Math.min(movers.length, 5)}, minmax(0, 1fr))` }}
    >
      {movers.map((m) => {
        const active = filter.symptom === m.symptom
        const rising = m.score > 0
        const toggle = () =>
          setFilter(withDimension(filter, 'symptom', active ? undefined : m.symptom))
        return (
          <Card
            key={m.symptom}
            size="sm"
            role="button"
            tabIndex={0}
            className={`gap-1 cursor-pointer hover:ring-foreground/20 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary ${active ? 'ring-1 ring-primary/50' : ''}`}
            onClick={toggle}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                toggle()
              }
            }}
          >
            <CardContent>
              <div className="font-mono text-sm text-foreground font-medium truncate">
                {m.symptom}
              </div>
              <div className="flex items-baseline gap-1.5 mt-1.5 font-mono">
                <span className="text-xl font-semibold text-foreground tabular-nums">{m.count}</span>
                <span className="text-xs text-foreground-muted">runs</span>
                <span className="text-[10px] text-foreground-secondary uppercase tracking-wider">
                  · {OUTCOMES[m.dominantOutcome].short}
                </span>
                <span
                  className="ml-auto text-sm leading-none"
                  style={{ color: rising ? RISING : FALLING }}
                  title={rising ? 'Rising within window' : 'Falling within window'}
                >
                  {rising ? '▲' : '▼'}
                </span>
              </div>
              <div className="h-5 mt-2">
                <MtbiSparkline
                  values={m.spark}
                  color={OUTCOMES[m.dominantOutcome].color}
                  showDot={false}
                />
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
