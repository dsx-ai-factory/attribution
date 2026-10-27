// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { HoverCard as HoverCardPrimitive } from "radix-ui"

import { cn, floatingEdge } from "@/lib/utils"

function HoverCard({
  ...props
}: React.ComponentProps<typeof HoverCardPrimitive.Root>) {
  return <HoverCardPrimitive.Root data-slot="hover-card" {...props} />
}

function HoverCardTrigger({
  ...props
}: React.ComponentProps<typeof HoverCardPrimitive.Trigger>) {
  return (
    <HoverCardPrimitive.Trigger data-slot="hover-card-trigger" {...props} />
  )
}

// The house width presets this app uses (the node summary's lg, a time's
// auto); the others are dropped, as nothing here uses them. Every card names
// its width, and none is wider than the window leaves it inside the page's
// gutter (the room Radix leaves within the collision padding). Every card is
// inset 12px, as the house DateParser's card is: a hover card is a small
// reading beside what it describes, not a page card.
const hoverCardSizeMap = {
  lg: "w-96",
  auto: "w-auto max-w-[min(40rem,var(--radix-hover-card-content-available-width))]",
} as const

function HoverCardContent({
  className,
  align = "center",
  sideOffset = 4,
  collisionPadding = floatingEdge,
  size,
  ...props
}: React.ComponentProps<typeof HoverCardPrimitive.Content> & {
  size: keyof typeof hoverCardSizeMap
}) {
  return (
    <HoverCardPrimitive.Portal data-slot="hover-card-portal">
      <HoverCardPrimitive.Content
        data-slot="hover-card-content"
        align={align}
        sideOffset={sideOffset}
        collisionPadding={collisionPadding}
        className={cn(
          "pointer-events-auto z-[60] max-w-(--radix-hover-card-content-available-width) min-w-0 origin-(--radix-hover-card-content-transform-origin) overflow-hidden rounded-md bg-background p-3 text-sm text-foreground shadow-md ring-1 ring-foreground/10 outline-hidden duration-100 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95",
          hoverCardSizeMap[size],
          className
        )}
        {...props}
      />
    </HoverCardPrimitive.Portal>
  )
}

export { HoverCard, HoverCardTrigger, HoverCardContent }
