// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"

import { cn } from "@/lib/utils"
import { Input } from "@/components/ui/input"

// The house InputGroup for an input with an inline icon or text addon: the
// group draws the border and the focus ring (the shared border-active /80
// instead of the house /50) while the input inside stays bare. It is the
// compact 28px control with a 14px icon. Button, kbd and textarea addons,
// block alignment, the sm size and the inline-end place are left out as
// unused: an addon leads the input.
function InputGroup({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="input-group"
      role="group"
      className={cn(
        "group/input-group relative flex h-7 w-full min-w-0 items-center rounded-md border border-border bg-background transition-[color,box-shadow] outline-none has-[[data-slot=input-group-control]:focus-visible]:border-border-active has-[[data-slot=input-group-control]:focus-visible]:ring-[3px] has-[[data-slot=input-group-control]:focus-visible]:ring-border-active/80 has-[>[data-slot=input-group-addon]]:[&>input]:pl-1.5",
        className
      )}
      {...props}
    />
  )
}

function InputGroupAddon({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      role="group"
      data-slot="input-group-addon"
      className={cn(
        "order-first flex h-auto cursor-text items-center justify-center gap-2 py-1.5 pl-2 text-sm font-medium text-foreground-secondary select-none [&>svg:not([class*='size-'])]:size-3.5",
        className
      )}
      // A click on the icon or text focuses the input, as a click on a label would.
      onClick={(event) => {
        event.currentTarget.parentElement?.querySelector("input")?.focus()
      }}
      {...props}
    />
  )
}

function InputGroupInput({
  className,
  ...props
}: React.ComponentProps<"input">) {
  return (
    <Input
      data-slot="input-group-control"
      className={cn(
        "flex-1 rounded-none border-0 bg-transparent shadow-none focus-visible:ring-0",
        className
      )}
      {...props}
    />
  )
}

export { InputGroup, InputGroupAddon, InputGroupInput }
