// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId } from 'react'
import dayjs from 'dayjs'
import { CalendarIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Calendar } from '@/components/ui/calendar'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

// react-day-picker renders and emits calendar days in the host's LOCAL timezone,
// but the whole dashboard is UTC: run timestamps, and the date-range window and
// per-day buckets the server returns, are all UTC. Selecting a day must
// therefore mean a UTC calendar day, independent of the viewer's zone.
//
// We keep this component's public contract in UTC — `value`/`onChange` are
// UTC-midnight Dates, matching the rest of the app — and bridge to react-day-picker
// with a "local proxy": a Date whose LOCAL Y/M/D equal the UTC day's, so the grid
// highlights (and returns) the intended day in every timezone. Without this bridge,
// `toISOString()` on a local-midnight Date rolls back a day east of UTC, and feeding
// a UTC-midnight Date back as `selected` rolls forward a day west of UTC.
function toLocalProxy(utc: Date | undefined): Date | undefined {
  if (!utc) return undefined
  return new Date(utc.getUTCFullYear(), utc.getUTCMonth(), utc.getUTCDate())
}

function fromLocalProxy(local: Date | undefined): Date | undefined {
  if (!local) return undefined
  return new Date(Date.UTC(local.getFullYear(), local.getMonth(), local.getDate()))
}

// A labelled date field: the Label names the outline trigger, which shows the
// chosen UTC day (tabular digits in the UI font) or a muted placeholder. The
// trigger is named by the label and its own text together ("From 2026-05-01"),
// because a label alone would hide the chosen day from a screen reader.
export function DatePicker({
  label,
  value,
  onChange,
  placeholder = 'Pick a date',
  minDate,
  maxDate,
}: {
  label: string
  value: Date | undefined
  onChange: (d: Date | undefined) => void
  placeholder?: string
  /** Earliest selectable UTC day, inclusive. Days before it are disabled. */
  minDate?: Date
  /** Latest selectable UTC day, inclusive. Days after it are disabled. */
  maxDate?: Date
}) {
  const triggerId = useId()
  const labelId = useId()
  const valueId = useId()
  const proxy = toLocalProxy(value)
  const minProxy = toLocalProxy(minDate)
  const maxProxy = toLocalProxy(maxDate)
  const disabled = [
    ...(minProxy ? [{ before: minProxy }] : []),
    ...(maxProxy ? [{ after: maxProxy }] : []),
  ]
  return (
    <div className="flex items-center gap-2">
      <Label id={labelId} htmlFor={triggerId}>{label}</Label>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            id={triggerId}
            aria-labelledby={`${labelId} ${valueId}`}
            className={cn('tabular-nums', !value && 'text-foreground-muted')}
          >
            <CalendarIcon data-icon="inline-start" />
            <span id={valueId}>{proxy ? dayjs(proxy).format('YYYY-MM-DD') : placeholder}</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-auto">
          <Calendar
            mode="single"
            selected={proxy}
            defaultMonth={proxy}
            disabled={disabled.length > 0 ? disabled : undefined}
            onSelect={(d) => onChange(fromLocalProxy(d))}
            autoFocus
          />
        </PopoverContent>
      </Popover>
    </div>
  )
}
