// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

// The house Alert tuned for the dark theme. Text stays neutral in every
// variant: the house destructive text (destructive-600, 2.92:1 on the card)
// fails AA, so only the icon carries the variant (destructive-400, 5.79:1;
// muted by default). The action takes a grid column after the text instead of
// the house absolute corner with a 72px reserve, which a long action label
// would overlap. There is no default role: an announcement passes role="alert"
// and an informational notice passes none. Links inside get no decoration. It
// is inset like a card's content and its 14px icon is a mark beside the title:
// 3px down centers it on the title's 20px line, and --mark-offset more on the
// title's lowercase, as every mark beside text is centered (lib/utils).
const alertVariants = cva(
  "group/alert grid w-full gap-x-2.5 gap-y-0.5 rounded-md border border-border px-card-x py-card-y text-left text-sm has-[>svg]:grid-cols-[auto_minmax(0,1fr)] has-data-[slot=alert-action]:grid-cols-[minmax(0,1fr)_auto] has-[>svg]:has-data-[slot=alert-action]:grid-cols-[auto_minmax(0,1fr)_auto] *:[svg]:translate-y-[calc(3px+var(--mark-offset))] *:[svg:not([class*='size-'])]:size-3.5",
  {
    variants: {
      variant: {
        default: "bg-background text-foreground *:[svg]:text-foreground-muted",
        destructive:
          "bg-background text-foreground *:[svg]:text-destructive-400",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Alert({
  className,
  variant,
  ...props
}: React.ComponentProps<"div"> & VariantProps<typeof alertVariants>) {
  return (
    <div
      data-slot="alert"
      className={cn(alertVariants({ variant }), className)}
      {...props}
    />
  )
}

function AlertTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="alert-title"
      className={cn("font-medium group-has-[>svg]/alert:col-start-2", className)}
      {...props}
    />
  )
}

function AlertDescription({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="alert-description"
      className={cn(
        "text-sm text-balance text-foreground-secondary group-has-[>svg]/alert:col-start-2 md:text-pretty",
        className
      )}
      {...props}
    />
  )
}

function AlertAction({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="alert-action"
      className={cn(
        "col-start-2 row-span-2 row-start-1 self-center group-has-[>svg]/alert:col-start-3",
        className
      )}
      {...props}
    />
  )
}

export { Alert, AlertTitle, AlertDescription, AlertAction }
