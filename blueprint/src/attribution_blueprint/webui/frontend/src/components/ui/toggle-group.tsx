// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { ToggleGroup as ToggleGroupPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"
import { toggleVariants } from "@/components/ui/toggle"

// The house ToggleGroup as its one use here: a horizontal segmented control of
// outline segments at the toggle's default 28px size, its items joined edge to
// edge (the house's spacing 0). Tuned: under 640px the segments take the
// house's spaced form (its spacing 1: 4px apart, each with its own border and
// corners) and wrap within the room they are given, as joined segments cannot
// break across lines, so no choice is ever cut off a narrow screen. The
// spacing option gives way to that breakpoint, and the orientation option is
// dropped, as nothing here uses it; so are the variant and size every group
// here sets alike, with the context that handed them to the items.
function ToggleGroup({
  className,
  ...props
}: React.ComponentProps<typeof ToggleGroupPrimitive.Root>) {
  return (
    <ToggleGroupPrimitive.Root
      data-slot="toggle-group"
      className={cn("flex w-fit max-w-full min-w-0 flex-row flex-wrap items-center gap-1 rounded-md sm:flex-nowrap sm:gap-0", className)}
      {...props}
    />
  )
}

// A single-choice item is a radio (aria-checked, data-state=on), so its
// selected state comes from data-state. A segment is a control, not content,
// so the selected one takes the shared primary-selected recipe (globals.css):
// the fill of the green default Button with a 1px inset primary-600 ring,
// where the house fill is neutral. Joined, from 640px, segments are
// square-cornered but for the group's two ends, and share their inner borders;
// these come after the toggle's own classes, so they replace its rounding and
// inline padding.
function ToggleGroupItem({
  className,
  ...props
}: React.ComponentProps<typeof ToggleGroupPrimitive.Item>) {
  return (
    <ToggleGroupPrimitive.Item
      data-slot="toggle-group-item"
      className={cn(
        toggleVariants({ variant: "outline" }),
        "shrink-0 px-2 shadow-none focus:z-10 focus-visible:z-10 data-[state=on]:primary-selected sm:rounded-none sm:border-l-0 sm:first:rounded-l-md sm:first:border-l sm:last:rounded-r-md",
        className
      )}
      {...props}
    />
  )
}

export { ToggleGroup, ToggleGroupItem }
