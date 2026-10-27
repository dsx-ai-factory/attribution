// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"

import { cn } from "@/lib/utils"

// The house Empty. The title is the body size at 500 in foreground, without
// the house tight tracking. The description keeps the house secondary tone
// (12.59:1) but loses its link styling (resting underline, green hover).
// size="sm" is Attribution's own, the compact state inside a card or a table
// body, where the house 48px inset would dwarf the content around it: 20px
// above and below and the card's inset at the sides, the title 4px over a
// 12px muted description (7.30:1), the caption tier of the card around it.
function Empty({
  className,
  size = "default",
  ...props
}: React.ComponentProps<"div"> & { size?: "default" | "sm" }) {
  return (
    <div
      data-slot="empty"
      data-size={size}
      className={cn(
        "group/empty flex w-full min-w-0 flex-1 flex-col items-center justify-center gap-4 rounded-md border-dashed p-12 text-center text-balance data-[size=sm]:gap-3 data-[size=sm]:px-card-x data-[size=sm]:py-5",
        className
      )}
      {...props}
    />
  )
}

function EmptyHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty-header"
      className={cn("flex max-w-sm flex-col items-center gap-2 group-data-[size=sm]/empty:gap-1", className)}
      {...props}
    />
  )
}

function EmptyTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty-title"
      className={cn("text-sm font-medium text-foreground", className)}
      {...props}
    />
  )
}

function EmptyDescription({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty-description"
      className={cn("text-sm/relaxed text-foreground-secondary group-data-[size=sm]/empty:text-xs group-data-[size=sm]/empty:text-foreground-muted", className)}
      {...props}
    />
  )
}

export { Empty, EmptyHeader, EmptyTitle, EmptyDescription }
