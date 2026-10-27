// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

// The house Card with the padding moved from the card onto its slots: a flush
// table or divided panes must reach the card's edges, which the house
// card-level py plus slot gap cannot do (house framed shells override it with
// py-0 gap-0). The spacing is the page's (the card-* values in globals.css):
// the header is one line of the card-header height, its divider included (the
// 13px title at 500, whatever sits beside it, and actions up to the 28px
// control height at its right end, which the height leaves room above and
// below for their focus ring), on the card-x inset, so a header with a control
// is as tall as one without. A header too long for its line wraps, and its 4px
// block padding keeps each wrapped line's text and controls, focus rings
// included, off its edges. Content sits card-x from the sides and card-y from
// the top and bottom; a footer is a line like the header. There is no
// description slot: a card explains itself in a hint beside its title, never
// in a second header line.
function Card({
  className,
  asChild = false,
  ...props
}: React.ComponentProps<"div"> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "div"
  return (
    <Comp
      data-slot="card"
      className={cn(
        "group/card flex flex-col overflow-hidden rounded-md border border-border bg-background text-sm text-foreground",
        className
      )}
      {...props}
    />
  )
}

function CardHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-header"
      className={cn(
        "flex min-h-card-header flex-wrap items-center gap-x-2 gap-y-1 px-card-x py-1",
        className
      )}
      {...props}
    />
  )
}

function CardTitle({
  className,
  asChild = false,
  ...props
}: React.ComponentProps<"div"> & { asChild?: boolean }) {
  // asChild lets a section render its title as the heading element.
  const Comp = asChild ? Slot.Root : "div"
  return (
    <Comp
      data-slot="card-title"
      className={cn("text-sm font-medium", className)}
      {...props}
    />
  )
}

// A header's actions end its line, and never widen the card: on a narrow
// screen they take a line of their own and shrink to it, their contents
// wrapping.
function CardAction({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-action"
      className={cn(
        "ml-auto flex min-w-0 items-center",
        className
      )}
      {...props}
    />
  )
}

function CardContent({
  className,
  flush = false,
  ...props
}: React.ComponentProps<"div"> & {
  /** Drops the padding so a table or list reaches the card edges and insets its own rows. */
  flush?: boolean
}) {
  return (
    <div
      data-slot="card-content"
      className={cn(flush ? "p-0" : "px-card-x py-card-y", className)}
      {...props}
    />
  )
}

function CardFooter({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-footer"
      className={cn(
        "flex min-h-card-header flex-wrap items-center gap-x-2 gap-y-1 px-card-x py-1",
        className
      )}
      {...props}
    />
  )
}

export {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardAction,
  CardContent,
}
