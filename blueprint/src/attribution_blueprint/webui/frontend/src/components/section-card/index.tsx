// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId, type ReactNode } from 'react'
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@/components/ui/empty'
import { InfoHint } from '@/components/info-hint'
import { cn, focusRing } from '@/lib/utils'

interface SectionEmpty {
  title: string
  description: string
  /** Announces the state, for a section that empties in place when a filter changes. */
  status?: boolean
}

// A titled region of a page on the house Card: a <section> named by its h2,
// so it is a landmark readers can jump to. The header is one line (the card
// header's height) that ends in a divider: the 13px title at 500; a count, or
// a short summary of counts, right after it as its own paragraph, out of the
// region's name; what the section shows, when it needs saying, in a hint
// after those (never a second header line); an action at the right end. flush
// drops the body padding for a table or list that insets its own rows. fill
// lets the body take the rest of a card stretched beside a taller one in a
// grid row, as a flex column its content can grow into. A section with
// nothing to show takes empty instead of its body: the compact Empty, a title
// and a description, on a flush body so its inset is the only one. An empty
// sub-block inside a body that has other content stays a line of muted text.
// A section the page moves focus to (after an action that changes what it
// lists) takes an id and tabIndex -1, so the landmark itself receives focus,
// and shows it with the one focus ring instead of the browser outline. The
// ring stays outside: row and header fills paint over an inset shadow. A
// caller that labels something else by the title (a table) passes titleId.
export function SectionCard({
  title,
  titleId,
  count,
  description,
  action,
  flush = false,
  fill = false,
  empty,
  className,
  id,
  tabIndex,
  children,
}: {
  title: string
  /** Id for the title, for a caller that labels its own content by it. */
  titleId?: string
  count?: ReactNode
  /** What the section shows, given in a hint beside the title. */
  description?: string
  action?: ReactNode
  flush?: boolean
  /** The body grows to the card's height, a flex column its content can fill. */
  fill?: boolean
  empty?: SectionEmpty
  /** Layout classes for the card in its page grid. */
  className?: string
  id?: string
  tabIndex?: number
  children?: ReactNode
}) {
  const ownTitleId = useId()
  const headingId = titleId ?? ownTitleId
  return (
    <Card asChild className={cn(tabIndex !== undefined && focusRing, className)}>
      <section id={id} tabIndex={tabIndex} aria-labelledby={headingId}>
        <CardHeader className="border-b">
          <CardTitle asChild>
            <h2 id={headingId}>{title}</h2>
          </CardTitle>
          {count !== undefined && <p className="text-foreground-muted tabular-nums">{count}</p>}
          {/* Outside the heading's name. It sits inline on a line of its own text size, as a
              label's hint does, so it takes the hint's inline centering, 4px closer to the text before it. */}
          {description !== undefined && (
            <span className="-ml-1">
              <InfoHint definition={{ label: title, def: description }} />
            </span>
          )}
          {action !== undefined && <CardAction>{action}</CardAction>}
        </CardHeader>
        {empty !== undefined ? (
          <CardContent flush>
            <Empty size="sm" role={empty.status ? 'status' : undefined}>
              <EmptyHeader>
                <EmptyTitle>{empty.title}</EmptyTitle>
                <EmptyDescription>{empty.description}</EmptyDescription>
              </EmptyHeader>
            </Empty>
          </CardContent>
        ) : (
          <CardContent flush={flush} className={cn(fill && 'flex flex-1 flex-col')}>{children}</CardContent>
        )}
      </section>
    </Card>
  )
}
