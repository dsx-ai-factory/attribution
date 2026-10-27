// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"

import { cn, focusRing } from "@/lib/utils"

// The house Input. Its house focus ring is border/80, zinc-800 at 1.18:1 on
// the card, so it takes the shared border-active /80 ring (9.65:1). It is the
// compact 28px control, its text inset 10px. File-input
// styles, the password variant and the invalid ring are left out: nothing here
// uploads, takes a secret or validates.
function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "flex h-7 w-full rounded-md border border-border bg-background px-2.5 py-1 text-sm placeholder:text-foreground-muted focus-visible:border-border-active disabled:cursor-not-allowed disabled:bg-background-secondary disabled:text-foreground-muted",
        focusRing,
        className
      )}
      {...props}
    />
  )
}

export { Input }
