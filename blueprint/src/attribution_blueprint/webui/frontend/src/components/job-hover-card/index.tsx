import dayjs from 'dayjs'
import utc from 'dayjs/plugin/utc'
import type { RunSummary } from '@/interfaces/attribution'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { VerdictPill } from '@/components/chip'
import { useNav } from '@/hooks/use-nav'

dayjs.extend(utc)

export function JobHoverCard({
  run,
  linkable,
  children,
}: {
  run: RunSummary
  /** When set, the card is interactive and opens the run's verdict on click/Enter. */
  linkable?: boolean
  children: React.ReactNode
}) {
  const nav = useNav()
  const hasWorkload = run.workload && run.workload !== '—'
  const open = () => nav('verdict', { id: run.id })
  return (
    <HoverCard openDelay={200} closeDelay={75}>
      <HoverCardTrigger asChild>
        <span className="cursor-pointer underline decoration-foreground-muted underline-offset-2 hover:decoration-primary">
          {children}
        </span>
      </HoverCardTrigger>
      <HoverCardContent
        align="start"
        sideOffset={6}
        onClick={linkable ? open : undefined}
        role={linkable ? 'button' : undefined}
        tabIndex={linkable ? 0 : undefined}
        onKeyDown={(e) => {
          if (!linkable) return
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            open()
          }
        }}
        className={`w-80 p-3 bg-background-secondary border-border ring-0 ${linkable ? 'cursor-pointer hover:bg-background-active transition-colors' : ''}`}
      >
        <div className="flex items-baseline gap-2 mb-2.5 min-w-0">
          <span className="font-mono text-xs text-foreground truncate">{run.jobId}</span>
          {hasWorkload && (
            <span className="font-mono text-xs text-foreground-muted shrink-0">
              · attempt {run.attempt}
            </span>
          )}
        </div>
        <VerdictPill outcome={run.outcome} label={run.verdictLabel} />
        <div className="grid grid-cols-[5rem_1fr] gap-y-1.5 gap-x-3 mt-3 font-mono text-xs">
          <div className="text-foreground-muted">user</div>
          <div className="text-foreground">{run.user}</div>
          <div className="text-foreground-muted">end</div>
          <div className="text-foreground">
            {dayjs(run.endTime).utc().format('YYYY-MM-DD HH:mm:ss')} UTC
          </div>
          <div className="text-foreground-muted">elapsed</div>
          <div className="text-foreground">{run.elapsedH.toFixed(2)}h</div>
          <div className="text-foreground-muted">symptom</div>
          <div className="text-foreground">{run.rootSymptom.symptom}</div>
          <div className="text-foreground-muted">rule</div>
          <div className="text-primary">{run.rootSymptom.rule}</div>
          <div className="text-foreground-muted">gpus</div>
          <div className="text-foreground">
            {run.gpuCount}
            {run.responsibleNodes.length > 0 && (
              <span className="text-foreground-muted ml-2">
                · {run.responsibleNodes.length} responsible
              </span>
            )}
          </div>
        </div>
        <div className="text-xs text-foreground-muted mt-3 pt-2 border-t border-border">
          Click to open verdict →
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
