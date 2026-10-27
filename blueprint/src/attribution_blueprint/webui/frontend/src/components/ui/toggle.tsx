// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { cva } from "class-variance-authority"
import { Toggle as TogglePrimitive } from "radix-ui"

import { cn, focusRing } from "@/lib/utils"

// The house Toggle, which here is a content button that filters by what it
// shows (a failure count, a legend entry, a ranked category). Pressed, it
// takes the NVIDIA accent where the house fill is neutral: a primary-500/10
// tint, its text in primary-300 (9.95:1 on the tint; muted parts 6.30:1) and a
// 1px inset primary-500/60 ring, since a state indicator needs 3:1 and no tint
// that light reaches it (the ring is 3.83:1 against the card, 3.31:1 against
// its own tint). The ring is an inset-ring, a shadow of its own, so the 3px
// focus ring still draws around a pressed toggle. Text is 400 like every
// button, not the house 500: the tint, the ring and the green text say which
// toggle is pressed. Focus uses the shared /80 ring, not the house /50. The
// default size is the compact 28px control with a 14px icon; sm stays 24px.
// The house lg size is dropped, as nothing here uses it. Each size and
// variant has one user: a content toggle is always the borderless sm one, and
// the segments of a ToggleGroup the outline default one, so neither takes
// them as options.
const toggleVariants = cva(
  `group/toggle inline-flex cursor-pointer items-center justify-center gap-1 rounded-md text-sm whitespace-nowrap text-foreground-secondary transition-[color,box-shadow] hover:bg-background-active hover:text-foreground focus-visible:border-border-active disabled:pointer-events-none disabled:opacity-50 aria-pressed:bg-primary-500/10 aria-pressed:text-primary-300 aria-pressed:inset-ring aria-pressed:inset-ring-primary-500/60 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-3.5 ${focusRing}`,
  {
    variants: {
      variant: {
        default: "bg-transparent",
        outline: "border border-border bg-transparent hover:bg-background-active",
      },
      size: {
        default: "h-7 min-w-7 px-2",
        sm: "h-6 min-w-6 px-1.5",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
)

function Toggle({
  className,
  ...props
}: React.ComponentProps<typeof TogglePrimitive.Root>) {
  return (
    <TogglePrimitive.Root
      data-slot="toggle"
      className={cn(toggleVariants({ size: "sm", className }))}
      {...props}
    />
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export { Toggle, toggleVariants }
