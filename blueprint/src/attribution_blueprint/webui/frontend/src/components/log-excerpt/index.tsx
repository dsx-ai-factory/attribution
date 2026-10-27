// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useLayoutEffect, useRef, useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { TimeHover } from '@/components/time-hover'
import { cn, focusRingInset } from '@/lib/utils'

// The ways a reader moves the lines: wheel or trackpad, touch, the scrollbar
// or a text selection, and the keyboard once the well has focus.
const READER_INPUT = ['wheel', 'touchstart', 'pointerdown', 'keydown'] as const

// A recorded message: mono, wrapping instead of being cut at the right edge.
const MESSAGE = 'font-mono whitespace-pre-wrap text-foreground-secondary group-data-[root=true]/row:text-foreground'

export interface LogExcerptLine {
  /** Stable identity for the row. */
  key: string
  /** When the line was recorded, as recorded; kept verbatim in dateTime. */
  time: string
  message: string
  /** The line the reader came for, the root symptom. */
  emphasis?: boolean
}

// The house LogExcerpt: recorded lines as plain mono text at 12px, wrapping
// instead of being cut at the right edge, each after its time, the shared
// timestamp (DateParser there, TimeHover here), and each row tinting under the
// pointer.
// Tuned: rows are 20px (house 24px), the compact density's line; the lines
// sit flush in the card body that holds them, on the page's darker surface,
// their times on the card's inset, and that surface fills the height its
// column gives it (never under 240px, whatever the lines) rather than growing
// with them: size containment keeps the lines out of the card's height, so
// beside a taller card it takes that card's height. Past it the lines scroll
// within themselves and become a named, focusable group (not a region: the
// section that holds them is already the landmark of that name), centered on
// the emphasized line (as near as the ends allow), so that line is always in
// view; there is no collapsed tail, since the emphasized line can come before
// it. That height follows the layout around it, not the window, so it is
// observed on the element itself.
// A time reads as every time in the app does (the full stamp in UTC with its
// muted mark), at the full muted tone instead of the house opacity-80, and,
// as the house stamp is, takes no tab stop: the scrolling lines are the one
// stop. The rows share the well's two columns, so the times line up however
// wide the stamp; under 640px a line's time sits above its message, which
// then has the row's width. The emphasized line is marked the same way
// whatever the verdict, in the NVIDIA accent rather than the house
// destructive tint (red stays for blame and worsening): a primary-500/10 tint
// (time 6.87:1, message 17.65:1 in the full foreground), kept and deepened to
// /15 under the pointer, where the house hover would replace it (time 6.29:1,
// message 16.16:1); a 2px primary-500 edge (7.32:1 on the tint); and a
// neutral "Root symptom" label (zinc-subtle, 11.84:1) at the row's far end,
// level with the message's first line, centered on its lowercase by the mono
// lines' own mark offset (0.7px), the message wrapping before it, so the
// mark never rests on color alone; where the message would have under 16
// characters beside it, the label moves under the message, still at the
// row's end. The label is UI text and cannot be selected, so copying the line
// copies its time and message as shown. One mono face, the app's, for the
// stamps and messages, and no level words or line numbers the record does not
// hold. With no lines, a muted sentence on the card body's inset says so.
export function LogExcerpt({
  lines,
  label,
  emptyLabel,
}: {
  lines: LogExcerptLine[]
  /** Names the scrolling lines, once they scroll, for the tab stop they become. */
  label: string
  emptyLabel: string
}) {
  const scroller = useRef<HTMLDivElement>(null)
  const [overflowing, setOverflowing] = useState(false)
  // Measured on mount, whenever the well changes size and when the lines say
  // something else (another run), not whenever the array is new: a caller
  // that rebuilds equal lines on each render (pointer hover, revalidation,
  // copy feedback) must not snap a reader's scroll back to the root line.
  // While the lines scroll, the root line is kept centered through every size
  // change (the card beside the well reflowing, the columns stacking below
  // 1024px, where the well can shrink by half and its lines rewrap) until the
  // reader takes the well in hand (wheel, touch, pointer or key on it); from
  // then on a size change only re-measures. Scroll events cannot tell the
  // reader apart: a layout change that clamps the scroll fires them too. Once
  // every line shows there is nothing left of their scroll, so lines that
  // scroll again are centered afresh.
  const content = JSON.stringify(lines)

  useLayoutEffect(() => {
    const element = scroller.current
    if (!element) return
    let following = true
    const measure = () => {
      const overflows = element.scrollHeight > element.clientHeight
      setOverflowing(overflows)
      if (!overflows) {
        following = true
        return
      }
      const root = element.querySelector<HTMLElement>('[data-root="true"]')
      if (following && root) {
        element.scrollTop = root.offsetTop - (element.clientHeight - root.offsetHeight) / 2
      }
    }
    const release = () => {
      following = false
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    for (const type of READER_INPUT) element.addEventListener(type, release, { passive: true })
    return () => {
      observer.disconnect()
      for (const type of READER_INPUT) element.removeEventListener(type, release)
    }
  }, [content])

  if (lines.length === 0) {
    return <p className="px-card-x py-card-y text-sm text-foreground-muted">{emptyLabel}</p>
  }

  return (
    <div
      ref={scroller}
      data-slot="log-excerpt"
      role={overflowing ? 'group' : undefined}
      aria-label={overflowing ? label : undefined}
      tabIndex={overflowing ? 0 : undefined}
      className={cn(
        'relative grid min-h-60 flex-1 grid-cols-1 content-start gap-x-3 overflow-x-hidden overflow-y-auto overscroll-contain bg-background-secondary py-2 contain-size [--mark-offset:0.7px] sm:grid-cols-[auto_minmax(0,1fr)]',
        focusRingInset,
      )}
    >
      {lines.map((line) => (
        <div
          key={line.key}
          data-root={line.emphasis || undefined}
          className="group/row col-span-full grid grid-cols-subgrid items-start px-card-x text-xs leading-5 hover:bg-background-active data-[root=true]:bg-primary-500/10 data-[root=true]:shadow-[inset_2px_0_0_var(--color-primary-500)] data-[root=true]:hover:bg-primary-500/15"
        >
          <span className="text-foreground-muted">
            <TimeHover date={line.time} cell focusable={false} />
          </span>
          {line.emphasis ? (
            <div className="flex min-w-0 flex-wrap items-start gap-x-3">
              <code className={cn(MESSAGE, 'min-w-0 grow basis-[16ch] break-words')}>{line.message}</code>
              <Badge variant="zinc-subtle" className="ml-auto select-none">Root symptom</Badge>
            </div>
          ) : (
            <div className="min-w-0 break-words">
              <code className={MESSAGE}>{line.message}</code>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
