// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { useId, type ReactNode } from 'react'
import dayjs from 'dayjs'
import utc from 'dayjs/plugin/utc'
import timezone from 'dayjs/plugin/timezone'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { cn } from '@/lib/utils'

dayjs.extend(utc)
dayjs.extend(timezone)

const DEFAULT_FORMAT = 'YYYY-MM-DD HH:mm:ss'

/**
 * Hover- and focus-driven local↔UTC time popover. Renders UTC inline with a
 * small muted "UTC" suffix; the panel shows UTC + local with IANA zone.
 * Pass omitSuffix when the surrounding text already carries (UTC) once.
 */
export function TimeHover({
  date,
  format = DEFAULT_FORMAT,
  children,
  omitSuffix = false,
  className,
}: {
  date: string | number | Date
  format?: string
  children?: ReactNode
  omitSuffix?: boolean
  className?: string
}) {
  const descriptionId = useId()
  const instant = dayjs(date)
  if (!instant.isValid()) {
    return <span className={cn('text-foreground-muted', className)}>Unavailable</span>
  }

  const utcText = instant.utc().format(format)
  const localText = instant.format(format)
  const localTz = dayjs.tz.guess()

  return (
    <HoverCard openDelay={150} closeDelay={75}>
      <HoverCardTrigger asChild>
        <span tabIndex={0} aria-describedby={descriptionId} className={cn('underline decoration-dotted decoration-foreground-muted underline-offset-2 hover:decoration-primary focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-foreground-muted', className)}>
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
      <span id={descriptionId} className="sr-only">Local time: {localText} ({localTz})</span>
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
