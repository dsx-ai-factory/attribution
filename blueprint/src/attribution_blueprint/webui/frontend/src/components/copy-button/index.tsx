// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { Check, Copy, type LucideIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useCopyToClipboard } from '@/hooks/use-copy-to-clipboard'

// A page action that copies text built at click time: the house code block
// copy button (the icon turns to a check for 2s, only after a copy that
// succeeded) at page-action size, an outline button beside the other header
// actions, on every page alike, so the same action never changes weight from
// one page to the next.
// Tuned as Copyable: the check is the button's own text color, not the house
// green; the name never changes, so the button keeps its width and the name
// tests and screen readers find it by, and a polite live region says "Copied",
// or "Copy failed" when the text did not reach the clipboard.
// It is always labelled, as the page actions beside it are, so a row of them
// reads alike; its icon is marked inline-start, so the house Button trims the
// padding on its side, and the check carries the same mark, so the swap does
// not shift the button.
export function CopyButton({
  getText,
  label,
  icon: Icon = Copy,
  tooltip,
}: {
  /** Builds the text when clicked, so it reflects the page at that moment. */
  getText: () => string
  /** The button's name and visible text. */
  label: string
  /** Shown until a copy succeeds; a copy mark by default. */
  icon?: LucideIcon
  /** Says more than the name, e.g. what the copied text is. */
  tooltip?: ReactNode
}) {
  const { copied, announcement, copy } = useCopyToClipboard(2000)
  const Mark = copied ? Check : Icon
  const button = (
    <Button onClick={() => void copy(getText())}>
      <Mark data-icon="inline-start" />
      {label}
    </Button>
  )
  return (
    <>
      {tooltip === undefined ? button : (
        <Tooltip>
          <TooltipTrigger asChild>{button}</TooltipTrigger>
          <TooltipContent>{tooltip}</TooltipContent>
        </Tooltip>
      )}
      <span aria-live="polite" className="sr-only">
        {announcement}
      </span>
    </>
  )
}
