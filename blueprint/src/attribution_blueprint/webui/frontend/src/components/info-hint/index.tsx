// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId } from 'react'
import { Info } from 'lucide-react'
import type { DefEntry, DefTerm } from '@/interfaces/definitions'
import { DEFINITIONS } from '@/lib/definitions'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn, markInText } from '@/lib/utils'

// The house icon action (a named icon Button under a Tooltip) carrying the
// muted info mark house chart panels set beside a title. Tuned: the tooltip
// opens after 300ms instead of the house 0, because hints sit in dense header
// and label rows where a pointer sweeping across would pop one after another.
// The definition is always the button's description through
// aria-describedby, so reading it never depends on hovering. Its source is
// hidden, not sr-only: sr-only text is still content, so it would join the
// name of the header, cell or label holding the hint and be read on every
// move into it, while aria-describedby reads hidden text all the same. The
// hint is a button of its own: place it inline after a label's text, never
// inside another control. There it is a mark in running text (markInText):
// its mark centers on the label's lowercase, as every mark beside text does,
// and the 20px button keeps out of the line's height. A metric names its term
// from the shared definitions; anything else, such as a card that explains
// itself beside its title, passes its own label and definition.
export function InfoHint({ term, definition }: { term: DefTerm; definition?: undefined } | { term?: undefined; definition: DefEntry }) {
  const { label, def, formula } = term !== undefined ? DEFINITIONS[term] : definition
  const descriptionId = useId()
  return (
    <>
      <Tooltip delayDuration={300}>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={`About ${label}`}
            aria-describedby={descriptionId}
            className={cn(markInText, 'text-foreground-muted')}
          >
            <Info className="size-3.5" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <div className="flex flex-col gap-1">
            <p>{def}</p>
            {formula && <p className="text-foreground-muted">{formula}</p>}
          </div>
        </TooltipContent>
      </Tooltip>
      <span id={descriptionId} hidden>
        {formula ? `${def} Formula: ${formula}` : def}
      </span>
    </>
  )
}
