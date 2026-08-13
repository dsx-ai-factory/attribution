import { useState, type ReactNode } from 'react'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { DEFINITIONS } from '@/lib/definitions'
import type { DefTerm } from '@/interfaces/definitions'

export function DefHover({
  term,
  children,
  className,
}: {
  term: DefTerm
  children?: ReactNode
  className?: string
}) {
  const entry = DEFINITIONS[term]
  const tooltipId = `def-${term}`
  const [open, setOpen] = useState(false)
  return (
    <HoverCard openDelay={150} closeDelay={75} open={open} onOpenChange={setOpen}>
      <HoverCardTrigger asChild>
        <span
          role="button"
          tabIndex={0}
          aria-describedby={tooltipId}
          aria-expanded={open}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              setOpen((v) => !v)
            } else if (e.key === 'Escape' && open) {
              e.preventDefault()
              setOpen(false)
            }
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          className={[
            'inline underline decoration-dotted decoration-foreground-muted',
            'underline-offset-2 hover:decoration-primary cursor-help',
            'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary',
            'focus-visible:ring-offset-0 rounded-[1px]',
            className,
          ]
            .filter(Boolean)
            .join(' ')}
        >
          {children ?? entry.label}
        </span>
      </HoverCardTrigger>
      <HoverCardContent
        id={tooltipId}
        align="start"
        sideOffset={6}
        className="w-72 p-3 bg-background-secondary border-border ring-0"
      >
        <div className="font-mono text-[11px] leading-snug text-foreground space-y-2">
          <div>
            <div className="text-foreground-muted uppercase tracking-widest text-[10px] mb-1">
              {entry.label}
            </div>
            <div className="text-foreground">{entry.def}</div>
          </div>
          {entry.formula && (
            <div>
              <div className="text-foreground-muted uppercase tracking-widest text-[10px] mb-1">
                Formula
              </div>
              <div className="text-primary tabular-nums">{entry.formula}</div>
            </div>
          )}
          {entry.example && (
            <div>
              <div className="text-foreground-muted uppercase tracking-widest text-[10px] mb-1">
                Example
              </div>
              <div className="text-foreground tabular-nums">{entry.example}</div>
            </div>
          )}
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
