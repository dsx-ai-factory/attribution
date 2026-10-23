// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId } from 'react'
import dayjs from 'dayjs'
import utc from 'dayjs/plugin/utc'
import timezone from 'dayjs/plugin/timezone'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { NoValue } from '@/components/field'
import { parseUtc, STAMP_FORMAT } from '@/lib/time'
import { cn, focusRing } from '@/lib/utils'

dayjs.extend(utc)
dayjs.extend(timezone)

// The house DateParser: a timestamp whose hover card gives it in UTC and in
// the reader's own zone, its columns 12px and its rows 6px apart at the
// compact density, sized by its content, inset 12px as every hover card is;
// under 640px each zone stands over its reading, so a long zone name
// ("America/Los_Angeles") never outruns a narrow window's card.
// Tuned: every time the app shows is this one stamp, whatever holds it (a
// page fact, a table cell, a log line), as the evidence it copies writes it
// (lib/evidence): the full 24-hour ISO-style date and time in UTC with its
// muted " UTC" mark, where the house DateParser shortens by age and reads in
// the local zone. So no caller picks a format or drops the mark, and no two
// times on a page read differently. The trigger is a <time> carrying the
// recorded string, focusable so the card opens from the keyboard too, except
// in a list of lines (a log), where each line's time would be one more stop
// and the list's own scrolling is its stop, as the house DateParser's trigger
// is never one; a help cursor and a brighter hover color instead of the house
// pointer and dashed hover underline, since the time is not a link; the stamp
// itself is a number (mono, tabular) while its mark is muted text. The local
// reading is the trigger's description through aria-describedby, so it does
// not depend on hovering, save for a log line's time, which is no stop of its
// own: there, as on the house DateParser, the card alone gives it. Its source
// is hidden, not sr-only, as InfoHint's
// definition is: sr-only text is still page text, so it was read once in place
// and again as the description, while aria-describedby reads hidden text all
// the same.
// A recorded time without a zone is UTC, as the server reads it (parseUtc).
// A time that does not parse is said to be unavailable by the empty-value
// rule of its slot: the word in a key-value row, the shared muted dash (read
// as "Unavailable") in a table cell, as every empty cell is.
export function TimeHover({
  date,
  cell = false,
  focusable = true,
}: {
  /** The time as recorded; kept verbatim in dateTime. */
  date: string
  /** The time fills a table cell. */
  cell?: boolean
  /** False for a line's time in a list of lines, which takes no tab stop of its own. */
  focusable?: boolean
}) {
  const descriptionId = useId()
  const instant = parseUtc(date)
  if (!instant.isValid()) {
    return cell ? <NoValue label="Unavailable" /> : <span className="text-foreground-muted">Unavailable</span>
  }

  const utcDetail = instant.format(STAMP_FORMAT)
  const localDetail = instant.local().format(STAMP_FORMAT)
  const localZone = dayjs.tz.guess()

  return (
    <HoverCard openDelay={150} closeDelay={100}>
      <HoverCardTrigger asChild>
        <time
          dateTime={date}
          tabIndex={focusable ? 0 : undefined}
          aria-describedby={descriptionId}
          className={cn('cursor-help rounded-xs text-nowrap transition-colors hover:text-foreground', focusRing)}
        >
          <span className="num">{utcDetail}</span> <span className="text-foreground-muted">UTC</span>
        </time>
      </HoverCardTrigger>
      <span id={descriptionId} hidden>Local time: {localDetail} ({localZone})</span>
      <HoverCardContent size="auto">
        <div className="grid grid-cols-1 gap-x-3 sm:grid-cols-[auto_auto] sm:gap-y-1.5">
          <span className="text-foreground-muted">UTC</span>
          <span className="num text-nowrap text-foreground">{utcDetail}</span>
          <span className="mt-1.5 wrap-anywhere text-foreground-muted sm:mt-0">{localZone}</span>
          <span className="num text-nowrap text-foreground">{localDetail}</span>
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
