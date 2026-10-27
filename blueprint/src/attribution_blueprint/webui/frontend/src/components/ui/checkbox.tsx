// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { Checkbox as CheckboxPrimitive } from "radix-ui"
import { CheckIcon } from "lucide-react"

import { cn, focusRing } from "@/lib/utils"

// The house Checkbox. Checked, it takes the NVIDIA accent: a primary-600 box
// (4.93:1 on the card, 5.21:1 on the page) with a white check (3.80:1 on the
// box), where the house fill is neutral. Unchecked, the box is identified by
// its border alone, so the border is neutral-500 (3.86:1 on the card, 4.08:1
// on the page) instead of the house border color at 1.24:1. The after: box
// widens the click target.
function Checkbox({
  className,
  ...props
}: React.ComponentProps<typeof CheckboxPrimitive.Root>) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      className={cn(
        "peer relative flex size-4 shrink-0 cursor-pointer items-center justify-center rounded-[4px] border border-neutral-500 bg-background transition-shadow after:absolute after:-inset-x-3 after:-inset-y-2 focus-visible:border-border-active disabled:cursor-not-allowed disabled:opacity-50 data-checked:border-primary-600 data-checked:bg-primary-600 data-checked:text-white",
        focusRing,
        className
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator
        data-slot="checkbox-indicator"
        className="grid place-content-center text-current transition-none [&>svg]:size-3.5"
      >
        <CheckIcon />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  )
}

export { Checkbox }
