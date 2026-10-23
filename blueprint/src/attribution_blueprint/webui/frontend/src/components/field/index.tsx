// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Fragment, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

// A stacked term and value, for facts laid out in columns rather than on a
// shared label axis (use DescriptionList for that). It keeps the house Field's
// vertical rhythm (4px between label and value) but not its form semantics:
// this is a read-only <dt>/<dd> pair, so it goes inside a <dl>. The label is a
// 12px muted caption and the value reads as a DescriptionList value does. A
// caller places the pair in its row or strip through className.
export function Field({ label, className, children }: { label: ReactNode; className?: string; children: ReactNode }) {
  return (
    <div className={cn('flex min-w-0 flex-col gap-1', className)}>
      <dt className="text-xs text-foreground-muted">{label}</dt>
      <dd className="min-w-0 text-sm break-words text-foreground">{children}</dd>
    </div>
  )
}

// The value of a fact the record left empty: said in words, muted, so it is
// never a blank value and never mistaken for data.
export function NotRecorded() {
  return <span className="text-foreground-muted">Not recorded</span>
}

// A value slot with nothing in it: a muted dash, since a word would crowd a
// table column or a stat, read by screen readers as its label: "None" for a
// value the record left empty, or what the caller says, e.g. "Not available"
// for a value that cannot be derived.
export function NoValue({ label = 'None' }: { label?: string }) {
  return (
    <span className="text-foreground-muted">
      <span aria-hidden="true">—</span>
      <span className="sr-only">{label}</span>
    </span>
  )
}

// The separators inside a recorded identifier (XID-171-UNCORRECTABLE_DRAM_ERROR,
// node01.cluster, team/job) after which Chrome breaks no line, as it does
// after a '-'.
const IDENTIFIER_SEPARATOR = /(?<=[_.:/])/

// A recorded identifier (a symptom, rule, node, host, job or source name, or a
// reason made of them) where its line may wrap: a break may fall after each of
// its separators, so the break-anywhere of a narrow slot is only the last
// resort, for a part wider than the line. Each break is a zero-width space
// that assistive tech and a copy both skip (aria-hidden, select-none), so the
// identifier is named, copied and found as recorded; a <wbr> would do the
// same for layout, but Chrome names it as a space ("team-alpha/ job"). The
// parts and their breaks sit in one inline element, so a flex or grid parent
// (a badge, a filter chip) lays the identifier out as one item that wraps as
// a line, not each part as an item of its own, set apart by the parent's gap.
// With nothing to break after, it is the text alone.
export function Identifier({ children }: { children: string }) {
  const parts = children.split(IDENTIFIER_SEPARATOR)
  if (parts.length === 1) return children
  return (
    <span>
      {parts.map((part, index) => (
        <Fragment key={index}>
          {index > 0 && <span aria-hidden="true" className="select-none">{'\u200b'}</span>}
          {part}
        </Fragment>
      ))}
    </span>
  )
}
