// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn, focusRing } from "@/lib/utils"

// The house Badge resolved to its dark values. Kept: outline (12.59:1 text on
// the card), zinc-subtle for neutral evidence (11.84:1) and red-subtle for
// blame (destructive-400 on destructive-900, 5.06:1). Dropped: the green
// default fill, destructive (3.44:1), link and the colored subtle variants,
// since an outcome category only ever shows as a dot. dashed is Attribution's
// own, for discarded decision steps: its zinc-600 dash (2.41:1 on the card) is
// a shape cue beside the muted text and icon that carry the state. Focus uses
// the shared /80 ring instead of the house /50. At the compact density a badge
// is 20px (12px text inset 6px), as tall as a 13px text line, so it sits in
// that line and a badge row of a table stays 28px. Placed inline, it aligns to
// the top of the line box: on the baseline, its 12px text's baseline would
// meet the 13px line's and push the box 0.5px out of the line. A wrapping
// badge is 20px too while its text fits on one line; wrapped, it keeps what it
// holds to the first line. Tuned: every badge here is a mark beside text (a
// row's, a table row's, a line's), so it takes the mark rule's row placement
// itself (markInRow in lib/utils), lowered onto that text's lowercase; set in
// running text it takes markInText, which cancels the offset. A badge that is
// itself a title, beside no text, takes no offset. A mark it holds (an icon,
// an outcome dot, a chip's clear button) is beside its own label wherever the
// badge is set, so the badge declares its label's offset for what it holds
// (0.93px, measured for its 12px at 500) and lowers its icons by it.
const badgeVariants = cva(
  `group/badge relative top-(--mark-offset) inline-flex w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-sm border border-transparent align-top font-medium whitespace-nowrap transition-all *:[--mark-offset:0.93px] has-data-[icon=inline-end]:pr-1 has-data-[icon=inline-start]:pl-1 [&>svg]:pointer-events-none [&>svg]:relative [&>svg]:top-(--mark-offset) [&>svg]:size-3! focus-visible:border-border-active ${focusRing}`,
  {
    variants: {
      variant: {
        outline: "border-border text-foreground-secondary",
        dashed: "border-dashed border-neutral-600 text-foreground-muted",
        "zinc-subtle": "bg-neutral-800 text-neutral-200",
        "red-subtle": "bg-destructive-900 text-destructive-400",
      },
      size: {
        sm: "h-5 px-1.5 text-xs",
        // Recorded values such as source names can be long; they wrap instead of being cut.
        wrap: "h-auto min-h-5 items-start px-1.5 py-px text-xs whitespace-normal wrap-anywhere",
      },
    },
    defaultVariants: {
      variant: "outline",
      size: "sm",
    },
  }
)

function Badge({
  className,
  variant = "outline",
  size = "sm",
  asChild = false,
  ...props
}: React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "span"

  return (
    <Comp
      data-slot="badge"
      data-variant={variant}
      className={cn(badgeVariants({ variant, size }), className)}
      {...props}
    />
  )
}

export { Badge }
