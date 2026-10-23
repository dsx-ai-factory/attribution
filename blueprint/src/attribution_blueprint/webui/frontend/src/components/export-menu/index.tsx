// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { ChevronDown, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

export interface ExportItem {
  /** The file type the item downloads, as "Download as JSON". */
  label: string
  onSelect: () => void
}

// Downloads offered as a real menu (arrow keys, typeahead, Escape) on the
// house DropdownMenu, opened from an outline button like every other page
// action, and opening under its end, as page and toolbar actions end their
// rows. Each item names only the file type it downloads; what the file holds
// is the page's own. The panel is as wide as its longest item and never
// narrower than its trigger, since those names can outrun the house
// trigger-width panel.
export function ExportMenu({
  items,
  triggerLabel,
}: {
  items: ExportItem[]
  triggerLabel: ReactNode
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button>
          <Download data-icon="inline-start" />
          {triggerLabel}
          <ChevronDown data-icon="inline-end" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-auto min-w-(--radix-dropdown-menu-trigger-width)">
        {items.map((item) => (
          <DropdownMenuItem key={item.label} onSelect={item.onSelect} className="whitespace-nowrap">
            {item.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
