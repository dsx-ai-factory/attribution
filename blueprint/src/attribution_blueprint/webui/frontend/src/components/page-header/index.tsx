// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Fragment, type ReactNode } from 'react'
import type { To } from 'react-router-dom'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import type { DescriptionListItem } from '@/components/ui/description-list'
import { RouterLink } from '@/components/ui/router-link'
import { Skeleton } from '@/components/ui/skeleton'
import { cn, dividedRow, gapHairline, markInText } from '@/lib/utils'
import { Field } from '@/components/field'

export interface PageCrumb {
  label: string
  /** Where the crumb leads; the crumb without one is the current page. */
  to?: To
}

// The house Breadcrumb for a page's place in the app. Ancestor crumbs are
// RouterLinks placed directly, so a Slot wrapper cannot rename their link slot.
// It stands alone only where a page has no head to show, its record missing or
// failed to load, so the way back is where it always is.
export function PageBreadcrumb({ crumbs }: { crumbs: PageCrumb[] }) {
  return (
    <Breadcrumb>
      <BreadcrumbList>
        {crumbs.map((crumb, index) => (
          <Fragment key={crumb.label}>
            {index > 0 && <BreadcrumbSeparator />}
            <BreadcrumbItem>
              {crumb.to === undefined ? (
                <BreadcrumbPage>{crumb.label}</BreadcrumbPage>
              ) : (
                <RouterLink to={crumb.to}>{crumb.label}</RouterLink>
              )}
            </BreadcrumbItem>
          </Fragment>
        ))}
      </BreadcrumbList>
    </Breadcrumb>
  )
}

// A page's facts as a stat strip: each a 12px muted label over its 13px value
// (a Field), in one row split by hairlines, the first flush with the title.
// A label may carry an InfoHint. The row wraps on a narrow screen, every line
// flush with the title: each fact draws its hairline in the 24px gap before
// it, which the strip clips where a line starts (dividedRow). While the values
// load, each label stays and a placeholder holds its value's 20px line, so
// the strip is as tall as it will be; the strip is marked busy.
function StatStrip({ items, loading }: { items: DescriptionListItem[]; loading: boolean }) {
  return (
    <div className={dividedRow}>
      <dl aria-busy={loading || undefined} className="flex flex-wrap gap-x-6 gap-y-2">
        {items.map((item) => (
          <Field key={item.key} label={item.label} className={gapHairline[24]}>
            {loading ? <Skeleton height="5" className="w-8" /> : item.value}
          </Field>
        ))}
      </dl>
    </div>
  )
}

// A page's head at the compact density: the 12px PageBreadcrumb, the page
// title (16px at 500, the house Heading base recipe on a 24px line) with the
// page-level actions at the right end of its row, kept on that row from 1024px
// up, and the page's facts as a stat strip, 8px apart. It heads the pages
// under Fleet; the root page shows no head (its title is for assistive tech
// alone), so every head has a trail. While the record behind a page loads,
// the head keeps its shape: a title the record gives is a placeholder on the
// title's line until it arrives (no empty heading), a caller passes its
// actions' placeholders, and the facts keep their labels (metaLoading).
export function PageHeader({
  breadcrumb,
  title,
  titleAddon,
  actions,
  meta,
  metaLoading = false,
}: {
  breadcrumb: PageCrumb[]
  /** Left out while the record that names the page loads. */
  title?: ReactNode
  /** Sits right after the title, e.g. a copy button or a status badge. */
  titleAddon?: ReactNode
  /** Actions on the whole page. */
  actions?: ReactNode
  meta?: DescriptionListItem[]
  /** The facts' values are still loading: placeholders stand in for them. */
  metaLoading?: boolean
}) {
  return (
    <header className="flex flex-col gap-2">
      <PageBreadcrumb crumbs={breadcrumb} />
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 lg:flex-nowrap">
        {/* The title and its addon share the title's line: the addon, one
            unbroken group 8px after the title, is a mark in the title's
            running text (markInText), centered on its lowercase; when the
            title wraps it follows the title's last line, and when it does not
            fit there it starts the next line at the title's edge. */}
        <div className="min-w-0 text-base leading-6">
          {title === undefined ? (
            <Skeleton height="6" className="w-64" />
          ) : (
            <h1 className={cn('inline font-medium break-words text-foreground', titleAddon !== undefined && 'mr-2')}>{title}</h1>
          )}
          {titleAddon !== undefined && <span className={cn('inline-flex items-center gap-2', markInText)}>{titleAddon}</span>}
        </div>
        {/* From 1024px the actions keep one line and the title wraps; below it
            they take their own row and wrap there too, never overflowing a
            narrow window. */}
        {actions && <div className="flex flex-wrap gap-2 lg:shrink-0">{actions}</div>}
      </div>
      {meta && meta.length > 0 && <StatStrip items={meta} loading={metaLoading} />}
    </header>
  )
}
