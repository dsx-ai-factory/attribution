// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"

import { cn } from "@/lib/utils"
import { Checkbox } from "@/components/ui/checkbox"

// The house CheckboxLabel: the label wraps the box, so its text names the
// checkbox and a click on the text toggles it. It renders the one Checkbox
// rather than a second copy of its styles. The text is secondary, the tier of
// the other control text (buttons, toggles), not the house foreground.
function CheckboxLabel({
  checked,
  onCheckedChange,
  children,
  className,
}: {
  checked: boolean
  onCheckedChange: (checked: boolean) => void
  children: React.ReactNode
  className?: string
}) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center gap-2 select-none",
        className
      )}
    >
      <Checkbox
        checked={checked}
        onCheckedChange={(value) => onCheckedChange(value === true)}
      />
      <span className="text-sm leading-none text-foreground-secondary">
        {children}
      </span>
    </label>
  )
}

export { CheckboxLabel }
