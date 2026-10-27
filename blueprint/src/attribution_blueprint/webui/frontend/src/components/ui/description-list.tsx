// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FC, ReactNode } from "react"

import { cn } from "@/lib/utils"

export interface DescriptionListItem {
  /** Stable identity for the row. */
  key: string
  /** The term. Rendered muted, so the value carries the eye. */
  label: ReactNode
  /** The description. */
  value: ReactNode
}

interface DescriptionListProps {
  items: DescriptionListItem[]
  /**
   * Width of the shared label column. Every value in the list lines up on one
   * vertical edge, which is the whole point of the pattern — pick the width
   * from the longest label the list can hold, not per row.
   */
  labelWidth?: "sm" | "md"
  className?: string
}

// The house widths re-measured for 12px labels: md holds the longest label a
// page or hover card lists, in NVIDIA Sans ("Root symptom time", 110.8px) and
// in its Arial fallback ("GPU-hours on node", 106.7px), sm the short labels of
// the graph dialog's step panel. The house lg width is dropped, as no list
// here uses it.
const labelWidthMap = {
  sm: "sm:grid-cols-[96px_minmax(min-content,1fr)]",
  md: "sm:grid-cols-[112px_minmax(min-content,1fr)]",
} as const

/**
 * DescriptionList renders term / description pairs on a shared label axis.
 *
 * Each pair is a real `<dt>` / `<dd>`, so the relationship survives for anything
 * reading the page rather than looking at it. The rows are grid cells of the
 * one `<dl>` rather than nested grids, so a long value in one row cannot push
 * the column out of line for the others. The term is a 12px caption and the
 * value body text, so each row aligns them on their first baseline.
 *
 * The values' column is never narrower than the widest value that cannot
 * wrap (a timestamp), and a recorded value wraps anywhere, so it never
 * widens that column: a container sized by its content (the verdict's answer
 * column) holds the label and that value, and a long value wraps within it.
 * Under 640px a label stands over its value, which then has the list's whole
 * width, as a line's time stands over its message in the log, so a timestamp
 * never runs out of a phone's card; its pairs stay 8px apart.
 */
export const DescriptionList: FC<DescriptionListProps> = ({
  items,
  labelWidth = "md",
  className,
}) => (
  <dl className={cn("grid grid-cols-1 items-baseline gap-x-3 gap-y-0.5 text-sm sm:gap-y-1.5", labelWidthMap[labelWidth], className)}>
    {items.map((item, index) => (
      <div key={item.key} className="contents">
        <dt className={cn("text-xs text-foreground-muted", index > 0 && "mt-1.5 sm:mt-0")}>{item.label}</dt>
        <dd className="wrap-anywhere text-foreground">{item.value}</dd>
      </div>
    ))}
  </dl>
)
