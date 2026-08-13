import { OUTCOMES } from '@/lib/attribution-data'
import type { OutcomeKey } from '@/interfaces/attribution'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'

const OUTCOME_DOT_HEX: Record<OutcomeKey, string> = {
  hardware: '#ec6358',
  software: '#3a96d9',
  multi:    '#a1a1a1',
  user:     '#c58420',
  undetermined: '#a05ab4',
  pass:     '#76b900',
}

export function OutcomeHoverCard({
  outcome,
  count,
  children,
}: {
  outcome: OutcomeKey
  count: number
  children: React.ReactNode
}) {
  const label = OUTCOMES[outcome].label
  const color = OUTCOME_DOT_HEX[outcome]

  return (
    <HoverCard openDelay={200} closeDelay={75}>
      <HoverCardTrigger asChild>{children}</HoverCardTrigger>
      <HoverCardContent
        align="start"
        sideOffset={6}
        className="w-72 p-3 bg-background-secondary border-border ring-0"
      >
        <div className="flex items-center gap-2 mb-2.5">
          <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: color }} />
          <span className="font-mono text-xs text-foreground uppercase tracking-wider">
            {label}
          </span>
        </div>
        <div className="grid grid-cols-[6rem_1fr] gap-y-1.5 gap-x-3 font-mono text-xs">
          <div className="text-foreground-muted">failures</div>
          <div className="text-foreground">{count}</div>
          <div className="text-foreground-muted">type</div>
          <div className="text-foreground">{label}</div>
        </div>
        <div className="text-xs text-foreground-muted mt-3 pt-2 border-t border-border">
          Click to filter runs by {label} →
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
