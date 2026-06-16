import dayjs from 'dayjs'
import { CalendarIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Calendar } from '@/components/ui/calendar'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'

// react-day-picker renders and emits calendar days in the host's LOCAL timezone,
// but the whole dashboard is UTC: run timestamps, the date-range window
// (filterRunsByDate) and the per-day buckets (buildDaily) are all UTC. Selecting a
// day must therefore mean a UTC calendar day, independent of the viewer's zone.
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

export function DatePicker({
  value,
  onChange,
  placeholder = 'Pick a date',
  className,
  minDate,
  maxDate,
}: {
  value: Date | undefined
  onChange: (d: Date | undefined) => void
  placeholder?: string
  className?: string
  /** Earliest selectable UTC day, inclusive. Days before it are disabled. */
  minDate?: Date
  /** Latest selectable UTC day, inclusive. Days after it are disabled. */
  maxDate?: Date
}) {
  const proxy = toLocalProxy(value)
  const minProxy = toLocalProxy(minDate)
  const maxProxy = toLocalProxy(maxDate)
  const disabled = [
    ...(minProxy ? [{ before: minProxy }] : []),
    ...(maxProxy ? [{ after: maxProxy }] : []),
  ]
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={`justify-start text-left font-mono ${value ? '' : 'text-foreground-muted'} ${className ?? ''}`}
        >
          <CalendarIcon className="size-3" />
          {proxy ? dayjs(proxy).format('YYYY-MM-DD') : placeholder}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-auto p-0">
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
  )
}
