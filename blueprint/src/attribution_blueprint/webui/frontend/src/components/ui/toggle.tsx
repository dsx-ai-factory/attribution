import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Toggle as TogglePrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

const toggleVariants = cva(
  "text-foreground-secondary hover:text-foreground aria-pressed:bg-primary-700/60 aria-pressed:text-white aria-pressed:ring-1 aria-pressed:ring-primary-600 aria-pressed:border-transparent aria-pressed:z-10 focus-visible:border-border-active focus-visible:ring-border-active/50 aria-invalid:ring-destructive-500/20 dark:aria-invalid:ring-destructive-500/40 aria-invalid:border-destructive-500 gap-1 rounded-md text-sm font-medium transition-[color,box-shadow] [&_svg:not([class*='size-'])]:size-4 group/toggle hover:bg-background-active inline-flex items-center justify-center whitespace-nowrap outline-none focus-visible:ring-[3px] disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 relative",
  {
    variants: {
      variant: {
        default: "bg-transparent",
        outline: "border-border hover:bg-background-active border bg-transparent",
      },
      size: {
        default: "h-8 min-w-8 px-2",
        sm: "h-6 min-w-6 px-1.5",
        lg: "h-10 min-w-10 px-2.5",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
)

function Toggle({
  className,
  variant = "default",
  size = "default",
  ...props
}: React.ComponentProps<typeof TogglePrimitive.Root> &
  VariantProps<typeof toggleVariants>) {
  return (
    <TogglePrimitive.Root
      data-slot="toggle"
      className={cn(toggleVariants({ variant, size, className }))}
      {...props}
    />
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export { Toggle, toggleVariants }
