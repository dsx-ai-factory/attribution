// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cva, type VariantProps } from "class-variance-authority"

import { focusRing } from "@/lib/utils"

// The house link styles, neutral at rest: the house green default and accent
// tones are gone and muted is the default (12.59:1 on the card), which turns
// to the NVIDIA green on hover (primary-400, 9.59:1 on the card, 10.13:1 on
// the page), the accent a link shares with the Button link variant. The
// foreground tone, for the brand, only dims on hover. A link takes the weight
// of its text, or 500 where asked (the brand), and no always-on underline:
// with the Button link variant, this is the only source of an underline,
// shown on hover. Focus is the shared /80
// ring without the house ring offset, whose default #fff would draw a white
// band on the dark surfaces. Size follows the surrounding text.
const linkVariants = cva(`cursor-pointer rounded-xs transition-colors ${focusRing}`, {
  variants: {
    variant: {
      muted: "text-foreground-secondary hover:text-primary-400",
      foreground: "text-foreground hover:text-foreground/80",
    },
    weight: {
      medium: "font-medium",
    },
    underline: {
      hover: "underline-offset-4 hover:underline",
      none: "no-underline",
    },
  },
  defaultVariants: {
    variant: "muted",
    underline: "hover",
  },
})

type LinkVariantProps = VariantProps<typeof linkVariants>

export { linkVariants }
export type { LinkVariantProps }
