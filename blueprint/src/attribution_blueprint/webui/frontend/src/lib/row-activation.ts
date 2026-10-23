// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { MouseEvent } from 'react'

// Controls a row can hold; a click on one is meant for the control, not the row.
const ROW_CONTROLS = 'a, button, input, [role="button"], [role="menuitem"]'

/**
 * Whether a click on a clickable table row should activate the row. It should
 * not when the click was for a control inside the row, when it landed in a
 * popover or dialog rendered from the row (React bubbles portal events through
 * the component tree, so they reach the row although they happen outside it),
 * or when it ends a text selection made inside the row.
 */
export function shouldActivateRow(event: MouseEvent<Element>): boolean {
  const row = event.currentTarget
  const target = event.target
  if (!(target instanceof Element) || !row.contains(target) || target.closest(ROW_CONTROLS)) return false
  const selection = window.getSelection()
  return !(
    selection?.toString() &&
    (row.contains(selection.anchorNode) || row.contains(selection.focusNode))
  )
}
