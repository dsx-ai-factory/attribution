// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CheckIcon, CopyIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard"
import { cn, markInRow } from "@/lib/utils"

// The house Copyable: a value with an icon button that copies it and shows a
// check for a second, only after the copy succeeded. Tuned: the value wraps,
// anywhere, instead of truncating, since a recorded identifier is never cut
// and never sets the least width of what holds it; the check
// is neutral, not the house green; the button keeps the caller's name (e.g.
// "Copy node ID") in both states, and a polite live region says "Copied", or
// "Copy failed" when the text did not reach the clipboard. The value shown is
// the text copied (the house option to show something else is dropped, as
// nothing here uses it). The button is a mark beside the value, as an info
// hint is beside its label: on the value's first line, centered on its
// lowercase (markInRow); alone, it takes the place of what holds it.
function Copyable({
  text,
  label,
  iconOnly = false,
}: {
  /** The text that is copied. */
  text: string
  /** Accessible name of the copy button. */
  label: string
  /** Shows only the button, when the value is already on screen. */
  iconOnly?: boolean
}) {
  const { copied, announcement, copy } = useCopyToClipboard(1000)
  return (
    <span className="inline-flex max-w-full min-w-0 items-start gap-1">
      {!iconOnly && <span className="min-w-0 wrap-anywhere">{text}</span>}
      <Button
        variant="ghost"
        size="icon-xs"
        aria-label={label}
        className={cn('flex-none text-foreground-secondary/80', markInRow)}
        onClick={() => void copy(text)}
      >
        {copied ? <CheckIcon /> : <CopyIcon />}
      </Button>
      <span aria-live="polite" className="sr-only">
        {announcement}
      </span>
    </span>
  )
}

export { Copyable }
