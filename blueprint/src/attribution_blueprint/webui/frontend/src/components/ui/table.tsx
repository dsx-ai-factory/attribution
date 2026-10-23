// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"

import { cellInset, cn } from "@/lib/utils"

// The house Table at the compact density: 28px header rows (12px at 500,
// muted) and 28px body rows (a 13px, 20px text line in py-1 cells), so a row
// holding a 20px badge is 28px too; a 24px control in a cell takes -2px block
// margins where it is placed, so it keeps its target size without heightening
// the row. Cells take the shared cell inset (lib/utils), whose first and last
// cells sit on the card's inset, so a table flush in a card lines up with the
// card header; a caller changing a cell's inset changes those first:/last:
// sides too, as a plain px- class does not reach them. Headers never
// highlight on hover. Cell text is secondary (the house cell inherits the
// card's foreground), so every table reads in the same tiers: muted headers,
// secondary values, and foreground only where a cell marks its row's own
// subject, such as a disclosure. Recorded values are not truncated; a cell
// that may hold a long value opts into wrapping instead. The house option to
// style the scroll container is dropped, as no table here uses it. That
// container scrolls sideways, so it clips on every side: it keeps 1px under
// its last row, where the 3px focus ring of a 24px control in the row would
// be cut. It is also a size container, so a message row (TableMessage) can
// keep to the part of the table in view.
function Table({
  className,
  layout = "auto",
  ...props
}: React.ComponentProps<"table"> & {
  layout?: "auto" | "fixed"
}) {
  return (
    <div
      data-slot="table-container"
      className="@container relative w-full overflow-x-auto pb-px"
    >
      <table
        data-slot="table"
        className={cn(
          "w-full text-sm text-foreground-secondary",
          layout === "fixed" && "table-fixed",
          className
        )}
        {...props}
      />
    </div>
  )
}

function TableHeader({ className, ...props }: React.ComponentProps<"thead">) {
  return (
    <thead
      data-slot="table-header"
      className={cn("[&_tr]:border-b [&_tr]:hover:bg-transparent", className)}
      {...props}
    />
  )
}

function TableBody({ className, ...props }: React.ComponentProps<"tbody">) {
  return (
    <tbody
      data-slot="table-body"
      className={cn("[&_tr:last-child]:border-0", className)}
      {...props}
    />
  )
}

function TableRow({ className, ...props }: React.ComponentProps<"tr">) {
  return (
    <tr
      data-slot="table-row"
      className={cn(
        "border-b transition-colors hover:bg-background-active/50",
        className
      )}
      {...props}
    />
  )
}

function TableHead({ className, ...props }: React.ComponentProps<"th">) {
  return (
    <th
      data-slot="table-head"
      className={cn(
        "h-7 py-1 text-left align-middle text-xs font-medium whitespace-nowrap text-foreground-muted",
        cellInset,
        className
      )}
      {...props}
    />
  )
}

function TableCell({
  className,
  wrap = false,
  ...props
}: React.ComponentProps<"td"> & {
  /**
   * Lets a recorded value wrap within the column instead of widening the
   * table: it may break anywhere, so its longest unbroken run never sets the
   * column's least width.
   */
  wrap?: boolean
}) {
  return (
    <td
      data-slot="table-cell"
      className={cn(
        "py-1 align-middle",
        cellInset,
        wrap ? "whitespace-normal wrap-anywhere" : "whitespace-nowrap",
        className
      )}
      {...props}
    />
  )
}

// A row saying why a table has no rows to show (none match, or they could not
// be loaded): one cell across every column, its message kept to the part of
// the table in view, the container's width less the card's inset each side,
// and stuck at the inset as the table scrolls sideways. Tuned: the house
// table centers its message across the whole table, off-screen when the
// table is wider than a narrow window. The row never highlights on hover.
function TableMessage({ colSpan, children }: { colSpan: number; children: React.ReactNode }) {
  return (
    <TableRow className="hover:bg-transparent">
      <TableCell colSpan={colSpan} wrap>
        <div data-slot="table-message" className="sticky left-card-x w-[calc(100cqw-2*var(--spacing-card-x))]">
          {children}
        </div>
      </TableCell>
    </TableRow>
  )
}

export {
  Table,
  TableHeader,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
  TableMessage,
}
