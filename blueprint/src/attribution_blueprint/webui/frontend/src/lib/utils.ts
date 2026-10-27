// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { clsx, type ClassValue } from "clsx"
import { extendTailwindMerge } from "tailwind-merge"

// The named spacing values from globals.css, so a class such as py-card-y
// replaces a primitive's py-1 instead of both landing on the element. A test
// holds this list to the names globals.css declares.
const twMerge = extendTailwindMerge({
  extend: {
    theme: {
      spacing: ["page-x", "page-top", "page-bottom", "page-gap", "card-gap", "card-x", "card-y", "card-header"],
    },
  },
})

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// The one keyboard focus treatment, taken from the house Button: a 3px
// border-active ring (9.65:1 on the card), plus a transparent outline that
// forced-colors mode turns visible when it drops the ring. The inset form is
// for scroll regions and flush rows whose outer edge is clipped.
export const focusRing =
  "outline-none focus-visible:ring-[3px] focus-visible:ring-border-active/80 focus-visible:outline-hidden"
export const focusRingInset = `${focusRing} focus-visible:ring-inset`

// A whole stat or card that filters by what it shows (a KPI stat, a mover),
// marked data-pressed while its filter applies: the NVIDIA accent of a pressed
// Toggle at the lighter tint a whole block needs (primary-500/5), with a 1px
// inset primary-500/60 ring (3.64:1 against the card, 3.42:1 against the
// tint), since no tint that light marks a state on its own. Its hover fill,
// shown only while not pressed, is the block's own: what counts as hovering
// it depends on where its target is.
export const pressedStat = "data-pressed:bg-primary-500/5 data-pressed:inset-ring data-pressed:inset-ring-primary-500/60"

// A table cell's inset: 12px on each side (so neighbouring cells' text is 24px
// apart), the first and last cell on the card's inset instead, so a table flush
// in a card lines up with the card header. Anything laid out on a table's
// columns (an expanded row's facts) takes it too.
export const cellInset = "px-3 first:pl-card-x last:pr-card-x"

// How near a floating layer (a menu, a select, a tooltip, a hover card, a
// popover) may come to the window's edge: the page's own gutter (the page-x
// spacing, 1.5rem), in pixels as Radix takes it, so a layer pushed off an edge
// lines up with the page's cards instead of touching the window. Every layer is
// also no wider than the room that leaves (Radix's available width), so a
// narrow window never pushes one past the other edge. A chart's tooltip is not
// a Radix layer: it stays inside its chart, which the gutter already holds
// (chartTheme in lib/tokens).
export const floatingEdge = 24

// A row of items that may wrap, split by hairlines (or middle dots) that never
// start or end a line: each item draws its divider in the middle of the gap
// before it (gapHairline, gapDot, by the row's gap in px), and the row clips
// 5px outside its edge (dividedRow), room for a focus ring, an info hint's
// too, which stands 2px out of its label's line, so the divider of a line's
// first item, out in the gap before the line, is never seen. A dot is
// decoration, so it has no text for assistive tech.
export const dividedRow = "-m-[5px] overflow-hidden p-[5px]"
export const gapHairline = {
  16: "relative before:absolute before:inset-y-0 before:-left-2 before:w-px before:bg-border",
  24: "relative before:absolute before:inset-y-0 before:-left-3 before:w-px before:bg-border",
} as const
export const gapDot = {
  16: "relative before:absolute before:-left-2.5 before:text-foreground-muted before:content-['·'_/_'']",
} as const

// A mark set beside text (a badge, an icon, a status dot, an info hint, a
// copy button) is centered on the text's lowercase, not on its line box, whose
// middle the lowercase sits below by the mark offset (--mark-offset in
// globals.css): centered on the box, a mark reads high beside a lowercase
// name. In a flex row or a table row, where vertical-align does not apply, the
// mark is centered on the row as the text beside it is, then lowered by the
// offset, which moves only its paint, so the row keeps its height
// (markInRow); a Badge takes this placement itself, and gives the marks it
// holds its own label's offset. In running text, vertical-align: middle
// centers the mark on the text's x-height, negative block margins keep a mark
// up to 20px tall from heightening the line, and the offset is 0 for the mark
// and the marks it holds (markInText). A mark beside text that may wrap sits
// in a box of the text's line, as running text, so it keeps to the first
// line. A control's own icon (a button's, a chevron) stays centered in the
// control, as the house draws it, and a mark between digits and capitals
// alone (the arrow from a window's first stamp, ending "UTC", to its second)
// stays centered on the row, where they center: the rule is for marks beside
// lowercase text.
export const markInText = "align-middle -my-1 [--mark-offset:0px]"
export const markInRow = "relative top-(--mark-offset)"

// The title of a sub-block inside a card or panel (an h3 or h4 under the
// section's h2): the body size at 500 in the full foreground, the size of the
// compact section title, whose place in the card header ranks it above.
export const subtitle = "text-sm font-medium text-foreground"
