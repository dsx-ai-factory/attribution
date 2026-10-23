// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { useHref, useLinkClickHandler, type To } from "react-router-dom"

import { cn } from "@/lib/utils"
import { linkVariants, type LinkVariantProps } from "@/components/ui/link-variants"

// The house RouterLink: a real <a href> that navigates in-app on a plain click
// and leaves modified clicks (new tab, new window) to the browser. A caller's
// onClick runs first and can cancel navigation with preventDefault. Replacing
// the history entry, route prefetching and external targets are left out as
// unused. data-slot is set after the spread props so a Slot wrapper (a
// hover-card trigger rendering through asChild) cannot rename it.
function RouterLink({
  to,
  variant,
  weight,
  underline,
  onClick,
  className,
  target,
  ...props
}: Omit<React.ComponentProps<"a">, "href"> &
  LinkVariantProps & {
    to: To
  }) {
  const href = useHref(to)
  const navigate = useLinkClickHandler<HTMLAnchorElement>(to, { target })
  return (
    <a
      {...props}
      href={href}
      target={target}
      onClick={(event) => {
        onClick?.(event)
        if (!event.defaultPrevented) navigate(event)
      }}
      className={cn(linkVariants({ underline, variant, weight }), className)}
      data-slot="link"
    />
  )
}

export { RouterLink }
