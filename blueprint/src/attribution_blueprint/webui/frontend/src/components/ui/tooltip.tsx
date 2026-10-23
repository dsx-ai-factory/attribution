// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { Tooltip as TooltipPrimitive } from "radix-ui"

import { cn, floatingEdge } from "@/lib/utils"

// The house Tooltip, its 12px text inset 10px by 6px at the compact density
// (the house 12px by 6px), without the house truncate option nothing here
// uses. Every Tooltip carries its own provider, so no call site or
// test wraps one; the provider stays private for that reason. A caller that
// wants a hover delay sets delayDuration on the Tooltip, which Radix prefers
// over the provider's 0.
function TooltipProvider({
  delayDuration = 0,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
  return (
    <TooltipPrimitive.Provider
      data-slot="tooltip-provider"
      delayDuration={delayDuration}
      {...props}
    />
  )
}

function Tooltip({
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Root>) {
  return (
    <TooltipProvider>
      <TooltipPrimitive.Root data-slot="tooltip" {...props} />
    </TooltipProvider>
  )
}

function TooltipTrigger({
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Trigger>) {
  return <TooltipPrimitive.Trigger data-slot="tooltip-trigger" {...props} />
}

function TooltipContent({
  className,
  sideOffset = 0,
  collisionPadding = floatingEdge,
  children,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content>) {
  // Portaled to the body so a tooltip escapes overflow-clipped cells and scroll regions.
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        data-slot="tooltip-content"
        sideOffset={sideOffset}
        collisionPadding={collisionPadding}
        className={cn(
          "z-[60] w-fit max-w-[min(var(--container-xs),var(--radix-tooltip-content-available-width))] origin-(--radix-tooltip-content-transform-origin) rounded-md bg-background px-2.5 py-1.5 text-xs break-words text-foreground shadow-md ring-1 ring-foreground/10 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-[state=delayed-open]:animate-in data-[state=delayed-open]:fade-in-0 data-[state=delayed-open]:zoom-in-95 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95",
          className
        )}
        {...props}
      >
        {children}
      </TooltipPrimitive.Content>
    </TooltipPrimitive.Portal>
  )
}

export { Tooltip, TooltipContent, TooltipTrigger }
