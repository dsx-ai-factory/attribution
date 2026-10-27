// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

type PageSizeProps =
  | {
      pageSizeOptions: readonly number[]
      onPageSizeChange: (pageSize: number) => void
    }
  | { pageSizeOptions?: undefined; onPageSizeChange?: undefined }

// The house PagePagination controls (outline First/Previous/Next/Last and an
// "n / page" select) placed as the house DataTable places them, flush right.
// Tuned: the icon buttons and the select are the compact 28px control (house
// 32px); no numbered page buttons, whose current page is the house green
// button; the range the page shows ("1–10 of 25 sites") stands in for them
// and is announced when it changes. Pages are counted from 0.
export function TablePagination({
  page,
  pageSize,
  total,
  onPageChange,
  label,
  itemLabel,
  controlsId,
  'aria-label': groupLabel,
  ...pageSizes
}: {
  page: number
  pageSize: number
  total: number
  onPageChange: (page: number) => void
  /** Names the buttons and the select: "Next {label} page", "{label} records per page". */
  label: string
  /** What the range counts, e.g. "sites". */
  itemLabel: string
  /** Id of the region whose content the buttons replace. */
  controlsId?: string
  /** Names the group of controls. */
  'aria-label': string
} & PageSizeProps) {
  const lastPage = Math.max(0, Math.ceil(total / pageSize) - 1)
  const start = total === 0 ? 0 : page * pageSize + 1
  const end = Math.min((page + 1) * pageSize, total)
  const steps = [
    { verb: 'First', to: 0, disabled: page === 0, Icon: ChevronsLeft },
    { verb: 'Previous', to: page - 1, disabled: page === 0, Icon: ChevronLeft },
    { verb: 'Next', to: page + 1, disabled: page >= lastPage, Icon: ChevronRight },
    { verb: 'Last', to: lastPage, disabled: page >= lastPage, Icon: ChevronsRight },
  ]

  return (
    <div role="group" aria-label={groupLabel} className="flex flex-wrap items-center justify-end gap-2">
      <span aria-live="polite" aria-atomic="true" className="text-sm text-foreground-muted tabular-nums">
        {`${start}–${end} of ${total} ${itemLabel}`}
      </span>
      <div className="flex items-center gap-1">
        {/* A step past either end is unavailable but stays focusable
            (aria-disabled, not disabled), so the button just pressed to
            reach an end keeps keyboard focus instead of dropping it to the
            page. */}
        {steps.map(({ verb, to, disabled, Icon }) => (
          <Button
            key={verb}
            size="icon"
            aria-disabled={disabled || undefined}
            aria-label={`${verb} ${label} page`}
            aria-controls={controlsId}
            onClick={() => {
              if (!disabled) onPageChange(to)
            }}
          >
            <Icon />
          </Button>
        ))}
      </div>
      {pageSizes.pageSizeOptions && (
        <Select value={String(pageSize)} onValueChange={(value) => pageSizes.onPageSizeChange(Number(value))}>
          <SelectTrigger aria-label={`${label} records per page`} className="w-28">
            <SelectValue />
          </SelectTrigger>
          <SelectContent align="end">
            {pageSizes.pageSizeOptions.map((option) => (
              <SelectItem key={option} value={String(option)}>{`${option} / page`}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  )
}
