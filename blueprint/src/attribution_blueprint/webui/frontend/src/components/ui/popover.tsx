// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { Popover as PopoverPrimitive } from "radix-ui"

import { cn, floatingEdge } from "@/lib/utils"

function Popover({
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Root>) {
  return <PopoverPrimitive.Root data-slot="popover" {...props} />
}

function PopoverTrigger({
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Trigger>) {
  return <PopoverPrimitive.Trigger data-slot="popover-trigger" {...props} />
}

// The house Popover as its one use here, the date picker's panel: the house
// padding option is dropped for the one inset that use takes, none, since the
// calendar inside pads itself.
function PopoverContent({
  className,
  align = "center",
  sideOffset = 4,
  collisionPadding = floatingEdge,
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Content>) {
  return (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Content
        data-slot="popover-content"
        align={align}
        sideOffset={sideOffset}
        collisionPadding={collisionPadding}
        // React events bubble out of a portal to its owner; a wheel over the
        // panel must not reach an ancestor's wheel handler.
        onWheel={(event) => event.stopPropagation()}
        className={cn(
          "pointer-events-auto z-[60] flex w-72 max-w-(--radix-popover-content-available-width) origin-(--radix-popover-content-transform-origin) flex-col gap-2.5 rounded-md bg-background text-sm text-foreground shadow-md ring-1 ring-foreground/10 outline-hidden duration-100 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95",
          className
        )}
        {...props}
      />
    </PopoverPrimitive.Portal>
  )
}

export { Popover, PopoverContent, PopoverTrigger }
