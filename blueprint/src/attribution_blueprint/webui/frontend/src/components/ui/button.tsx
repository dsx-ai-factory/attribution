// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Loader2 } from "lucide-react"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "rounded-md border border-transparent bg-clip-padding text-sm font-medium [&_svg:not([class*='size-'])]:size-4 inline-flex items-center justify-center whitespace-nowrap transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none shrink-0 [&_svg]:shrink-0 outline-none group/button select-none focus-visible:ring-[3px] focus-visible:ring-border-active/80 focus-visible:outline-hidden aria-invalid:ring-destructive-500/20 dark:aria-invalid:ring-destructive-500/40 aria-invalid:border-destructive-500 aria-invalid:ring-[3px]",
  {
    variants: {
      variant: {
        default:
          "border-primary-600 bg-primary-500/80 text-primary-900 focus-visible:bg-primary-500 not-disabled:not-aria-disabled:hover:bg-primary-500 not-disabled:not-aria-disabled:hover:text-primary-950 dark:border-primary-600 dark:bg-primary-700/60 dark:text-white dark:focus-visible:bg-primary-600/70 dark:not-disabled:not-aria-disabled:hover:border-primary-600 dark:not-disabled:not-aria-disabled:hover:bg-primary-600/70 dark:not-disabled:not-aria-disabled:hover:text-white not-disabled:not-aria-disabled:active:bg-primary-600 dark:not-disabled:not-aria-disabled:active:bg-primary-600/90",
        destructive:
          "border-destructive-600 bg-destructive-500/80 text-white focus-visible:bg-destructive-500 not-disabled:not-aria-disabled:hover:border-destructive-500 not-disabled:not-aria-disabled:hover:bg-destructive-500 dark:border-destructive-600 dark:bg-destructive-700/80 dark:text-white dark:focus-visible:bg-destructive-600 dark:not-disabled:not-aria-disabled:hover:border-destructive-600 dark:not-disabled:not-aria-disabled:hover:bg-destructive-600 not-disabled:not-aria-disabled:active:bg-destructive-600 dark:not-disabled:not-aria-disabled:active:bg-destructive-700",
        outline:
          "border-border bg-background text-foreground-secondary focus-visible:bg-background-active not-disabled:not-aria-disabled:hover:bg-background-active not-disabled:not-aria-disabled:hover:text-foreground aria-expanded:bg-background-active aria-expanded:text-foreground not-disabled:not-aria-disabled:active:bg-background-secondary/70",
        secondary:
          "border-border bg-background-secondary text-foreground-secondary focus-visible:bg-background-active not-disabled:not-aria-disabled:hover:bg-background-active not-disabled:not-aria-disabled:hover:text-foreground aria-expanded:bg-background-active aria-expanded:text-foreground not-disabled:not-aria-disabled:active:bg-background-secondary/90",
        ghost:
          "bg-transparent text-foreground-secondary focus-visible:bg-background-active not-disabled:not-aria-disabled:hover:bg-background-active not-disabled:not-aria-disabled:hover:text-foreground aria-expanded:bg-background-active aria-expanded:text-foreground not-disabled:not-aria-disabled:active:bg-background-secondary/60",
        link: "px-0! text-foreground-secondary underline-offset-4 focus-visible:text-foreground not-disabled:not-aria-disabled:hover:text-foreground not-disabled:not-aria-disabled:hover:underline not-disabled:not-aria-disabled:active:text-foreground-secondary/80",
      },
      size: {
        default:
          "h-8 gap-1.5 px-2.5 in-data-[slot=button-group]:rounded-md has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2",
        xs: "h-5 gap-1 rounded-[min(var(--radius-md),8px)] px-2 text-xs in-data-[slot=button-group]:rounded-md has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-6 gap-1 rounded-[min(var(--radius-md),10px)] px-2.5 in-data-[slot=button-group]:rounded-md has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5",
        lg: "h-10 gap-1.5 px-2.5 has-data-[icon=inline-end]:pr-3 has-data-[icon=inline-start]:pl-3",
        icon: "size-8",
        "icon-xs":
          "size-5 rounded-[min(var(--radius-md),8px)] in-data-[slot=button-group]:rounded-md [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-6 rounded-[min(var(--radius-md),10px)] in-data-[slot=button-group]:rounded-md",
        "icon-lg": "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  loading = false,
  disabled,
  children,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
    loading?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "button"
  const isDisabled = disabled || (loading && !asChild)

  if (asChild) {
    return (
      <Comp
        data-slot="button"
        data-variant={variant}
        data-size={size}
        className={cn(buttonVariants({ variant, size, className }))}
        disabled={disabled}
        {...props}
      >
        {children}
      </Comp>
    )
  }

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      disabled={isDisabled}
      aria-busy={loading || undefined}
      {...props}
    >
      {loading ? <Loader2 className="animate-spin" /> : null}
      {children}
    </Comp>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export { Button, buttonVariants }
