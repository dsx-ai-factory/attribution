// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn, focusRing } from "@/lib/utils"
import { Spinner } from "@/components/ui/spinner"

// The house Button with its sizes and default type, tuned for the dark-only
// theme. default is the house green fill at its dark values, the NVIDIA accent
// (white on it 9.99:1 on the card, 6.29:1 hovered), defined once as
// primary-fill in globals.css, which the selected segment and day share; it is
// kept for a page's one primary action, where it has one; it presses to
// primary-600 at 80%, not the house 90% (4.47:1), so its label stays AA while
// pressed (5.28:1).
// Every other button is outline (what a Button is unless asked) or ghost, at
// 400 as the house ones are; the green fill is 500, as in house. The link
// variant's hover turns the text green (9.59:1) under its underline; at rest
// it is neutral. Only the variants and sizes some control here uses are kept:
// dropped are destructive (the product has no destructive action), secondary,
// lg and icon-lg. Hover and press states are gated on
// not-disabled:not-aria-disabled: instead of the house enabled prefix, which
// an <a> rendered through asChild never matches, so a link keeps its hover
// color and hover underline. aria-disabled dims like disabled and drops the
// pointer cursor. At the compact density a default control is 28px (a page
// action, a toolbar control) and a lucide icon is 14px, the size a 13px label
// pairs with; the 24px and 20px small sizes stay for tables and inline hints.
const buttonVariants = cva(
  `group/button inline-flex shrink-0 cursor-pointer items-center justify-center rounded-md border border-transparent bg-clip-padding text-sm whitespace-nowrap transition-all select-none disabled:pointer-events-none disabled:opacity-50 aria-disabled:cursor-default aria-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-3.5 ${focusRing}`,
  {
    variants: {
      variant: {
        default:
          "border-primary-600 primary-fill font-medium focus-visible:bg-primary-600/70 not-disabled:not-aria-disabled:hover:bg-primary-600/70 not-disabled:not-aria-disabled:active:bg-primary-600/80",
        outline:
          "border-border bg-background text-foreground-secondary focus-visible:bg-background-active not-disabled:not-aria-disabled:hover:bg-background-active not-disabled:not-aria-disabled:hover:text-foreground aria-expanded:bg-background-active aria-expanded:text-foreground not-disabled:not-aria-disabled:active:bg-background-secondary/70",
        ghost:
          "bg-transparent text-foreground-secondary focus-visible:bg-background-active not-disabled:not-aria-disabled:hover:bg-background-active not-disabled:not-aria-disabled:hover:text-foreground aria-expanded:bg-background-active aria-expanded:text-foreground not-disabled:not-aria-disabled:active:bg-background-secondary/60",
        // With linkVariants, the only source of an underline, and only on hover.
        link: "px-0! text-foreground-secondary underline-offset-4 focus-visible:text-foreground not-disabled:not-aria-disabled:hover:text-primary-400 not-disabled:not-aria-disabled:hover:underline not-disabled:not-aria-disabled:active:text-primary-400/80",
      },
      size: {
        default: "h-7 gap-1.5 px-3 has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2",
        xs: "h-5 gap-1.5 rounded-[min(var(--radius-md),8px)] px-2 text-xs has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-6 gap-1.5 rounded-[min(var(--radius-md),10px)] px-2.5 has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5",
        icon: "size-7",
        "icon-xs":
          "size-5 rounded-[min(var(--radius-md),8px)] [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-6 rounded-[min(var(--radius-md),10px)]",
      },
    },
    defaultVariants: {
      variant: "outline",
      size: "default",
    },
  },
)

function Button({
  className,
  variant = "outline",
  size = "default",
  asChild = false,
  loading = false,
  disabled,
  type,
  onClick,
  children,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
    /**
     * Marks the button busy and unavailable and prepends a spinner; it ignores
     * activation until done. Ignored under asChild.
     */
    loading?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "button"

  if (asChild) {
    return (
      <Comp
        data-slot="button"
        data-variant={variant}
        data-size={size}
        className={cn(buttonVariants({ variant, size, className }))}
        disabled={disabled}
        type={type}
        onClick={onClick}
        {...props}
      >
        {children}
      </Comp>
    )
  }

  // Busy, the button stays focusable (disabled would drop keyboard focus to
  // the page) and swallows activation, form submission included.
  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      disabled={disabled}
      aria-disabled={loading || undefined}
      aria-busy={loading || undefined}
      type={type ?? "button"}
      onClick={loading ? (event) => event.preventDefault() : onClick}
      {...props}
    >
      {/* aria-busy carries the state; a spinner announcing "Loading" would rename the button.
          It leads the label, so it takes the inline-start icon padding an icon in its place has. */}
      {loading ? <Spinner aria-hidden="true" data-icon="inline-start" /> : null}
      {children}
    </Comp>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export { Button, buttonVariants }
