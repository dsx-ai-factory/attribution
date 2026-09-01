import type { FleetFilter } from '@/interfaces/filter'
import type { DateBounds } from '@/interfaces/meta'

export type WindowPreset = 'all' | '24h' | '7d' | '30d' | 'custom'

const DAY_MS = 86400000

export function presetRange(preset: Exclude<WindowPreset, 'custom'>, anchor: Date): { from: Date; to: Date } {
  const to = new Date(anchor.getTime())
  const fromMs = preset === '24h' ? DAY_MS : preset === '7d' ? 7 * DAY_MS : 30 * DAY_MS
  const from = new Date(to.getTime() - fromMs)
  return { from, to }
}

export function windowLabel(preset: WindowPreset, range: { from: Date; to: Date } | null): string {
  if (preset === 'all') return 'all time'
  if (preset !== 'custom' || !range) return preset
  const days = Math.max(1, Math.round((range.to.getTime() - range.from.getTime()) / DAY_MS))
  return `${days}d`
}

export function isoDate(d: Date): string {
  return d.toISOString().slice(0, 10)
}

function windowRange(filter: FleetFilter): { from: string; to: string } | null {
  return filter.from && filter.to ? { from: filter.from, to: filter.to } : null
}

// Presets mean "a window ending now", so `now` must be the current time:
// a window that stopped covering now more than a day ago is custom no matter
// its span. Staleness is measured from the end of the `to` day, so a preset
// clicked just before UTC midnight stays detected through the next day.
export function detectPreset(filter: FleetFilter, now: Date): WindowPreset {
  const selected = windowRange(filter)
  if (selected === null) return 'all'
  const toEnd = new Date(`${selected.to}T23:59:59.999Z`)
  const diffMs = now.getTime() - toEnd.getTime()
  if (Math.abs(diffMs) > DAY_MS) return 'custom'
  const spanMs = new Date(selected.to).getTime() - new Date(selected.from).getTime()
  if (Math.abs(spanMs - DAY_MS) < DAY_MS * 0.5) return '24h'
  if (Math.abs(spanMs - 7 * DAY_MS) < DAY_MS) return '7d'
  if (Math.abs(spanMs - 30 * DAY_MS) < DAY_MS * 2) return '30d'
  return 'custom'
}

export function isWindowed(filter: FleetFilter): boolean {
  return windowRange(filter) !== null
}

// The bounds are database-wide and unfiltered, so this answers "does anything at
// all sit outside the selected window" and nothing narrower.
// `>=` / `<=` is inclusive because the backend covers both endpoint days in full
export function boundsOutsideWindow(
  bounds: DateBounds | null,
  filter: FleetFilter,
): { min: string; max: string } | null {
  const selected = windowRange(filter)
  if (bounds === null || bounds.min === null || bounds.max === null || selected === null) return null
  if (bounds.min >= selected.from && bounds.max <= selected.to) return null
  return { min: bounds.min, max: bounds.max }
}
