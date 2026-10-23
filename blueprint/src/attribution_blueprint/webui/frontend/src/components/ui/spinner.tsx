// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"

import { cn } from "@/lib/utils"

// The house Spinner SVG. Its pixel-size presets are left out, and it sets no
// size of its own: the one caller, Button's loading state, sizes it as it
// sizes the icon it stands in for (14px at the default size), so a button
// keeps its width when it turns busy.
function Spinner({ className, ...props }: React.ComponentProps<"svg">) {
  return (
    <svg
      role="status"
      aria-label="Loading"
      className={cn("animate-spin", className)}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      {...props}
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  )
}

export { Spinner }
