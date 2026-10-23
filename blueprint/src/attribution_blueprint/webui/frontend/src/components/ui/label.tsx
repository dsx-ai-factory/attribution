// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { Label as LabelPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

// The house Label. A label here only introduces the control after it (From,
// To), so it steps back to muted (7.30:1) instead of inheriting foreground.
function Label({
  className,
  ...props
}: React.ComponentProps<typeof LabelPrimitive.Root>) {
  return (
    <LabelPrimitive.Root
      data-slot="label"
      className={cn(
        "text-sm leading-none text-foreground-muted peer-disabled:cursor-not-allowed",
        className
      )}
      {...props}
    />
  )
}

export { Label }
