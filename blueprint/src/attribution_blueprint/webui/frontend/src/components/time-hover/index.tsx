import { type ReactNode } from 'react'
import dayjs from 'dayjs'
import utc from 'dayjs/plugin/utc'
import timezone from 'dayjs/plugin/timezone'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'

dayjs.extend(utc)
dayjs.extend(timezone)

const DEFAULT_FORMAT = 'YYYY-MM-DD HH:mm:ss'

/**
 * Hover-driven local↔UTC time popover. Renders UTC inline with a small
 * muted "UTC" suffix; hover panel shows UTC + local with IANA zone.
 * Pass omitSuffix when the surrounding text already carries (UTC) once.
 */
export function TimeHover({
  date,
  format = DEFAULT_FORMAT,
  children,
  omitSuffix = false,
}: {
  date: string | number | Date
  format?: string
  children?: ReactNode
  omitSuffix?: boolean
}) {
  const instant = dayjs(date)
  if (!instant.isValid()) {
    return <span className="text-foreground-muted">Unavailable</span>
  }

  const utcText = instant.utc().format(format)
  const localText = instant.format(format)
  const localTz = dayjs.tz.guess()

  return (
    <HoverCard openDelay={150} closeDelay={75}>
      <HoverCardTrigger asChild>
        <span className="underline decoration-dotted decoration-foreground-muted underline-offset-2 hover:decoration-primary">
          {children ?? (
            <>
              {utcText}
              {!omitSuffix && (
                <span className="text-foreground-muted text-[10px] ml-0.5 tracking-wide">
                  UTC
                </span>
              )}
            </>
          )}
        </span>
      </HoverCardTrigger>
      <HoverCardContent
        align="start"
        sideOffset={6}
        className="w-auto p-2 bg-background-secondary border-border ring-0"
      >
        <div className="grid grid-flow-col auto-cols-auto gap-4 font-mono text-xs text-foreground whitespace-nowrap">
          <div className="flex flex-col gap-1 text-foreground-muted">
            <span>UTC</span>
            <span>{localTz}</span>
          </div>
          <div className="flex flex-col gap-1 text-foreground">
            <span>{utcText}</span>
            <span>{localText}</span>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
