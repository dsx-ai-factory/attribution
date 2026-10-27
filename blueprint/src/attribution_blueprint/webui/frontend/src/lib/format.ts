// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Decimal from 'decimal.js'

/** Whole numbers render without a trailing ".00"; decimals kept (≤2dp) only when real. */
export function fmtNum(n: Decimal.Value): string {
  const d = new Decimal(n)
  return d.isInteger() ? d.toString() : d.toDecimalPlaces(2).toString()
}

/** GPU-hours display: raw hours below 1000; switch to "k" units only at/above 1000.
 *  Returns the formatted number and whether the "k" suffix applies. */
export function fmtGpuHours(hours: Decimal.Value): { value: string; k: boolean } {
  const d = new Decimal(hours)
  if (d.abs().gte(1000)) return { value: fmtNum(d.div(1000).toDecimalPlaces(2)), k: true }
  return { value: fmtNum(d.toDecimalPlaces(2)), k: false }
}

// A part's share of a whole as the page shows it, judged at the precision
// both are shown to (`digits` decimals): a part shown as 0 is none of it, and
// a part that leaves 0 shown over is all of it, unless the whole itself shows
// as 0, where only the exact values tell. Otherwise a part short of the whole
// is never rounded up to all of it, nor a part down to none (99.6% reads as
// nearly all, 0.3% as nearly none), so a stated share never contradicts what
// is shown left over; a part beyond the whole reads as it is.
type JudgedShare =
  | { shown: 'none' | 'all' | 'nearly all' | 'nearly none' }
  | { shown: 'percent'; percent: Decimal }

function judgeShare(part: Decimal.Value, whole: Decimal.Value, digits: number): JudgedShare {
  const shownAsZero = (value: Decimal) => value.toDecimalPlaces(digits, Decimal.ROUND_HALF_UP).isZero()
  const share = new Decimal(part)
  const all = new Decimal(whole)
  const judged = !shownAsZero(all)
  if (share.isZero() || (judged && shownAsZero(share))) return { shown: 'none' }
  if (share.eq(all) || (judged && shownAsZero(all.minus(share)))) return { shown: 'all' }
  const percent = share.div(all).times(100).toDecimalPlaces(0, Decimal.ROUND_HALF_UP)
  if (percent.gte(100) && share.lt(all)) return { shown: 'nearly all' }
  if (percent.lte(0)) return { shown: 'nearly none' }
  return { shown: 'percent', percent }
}

/** A part's share of a whole as a whole percent, as the page shows it (judgeShare): "0%", "100%", ">99%", "<1%" or "n%". */
export function fmtShare(part: Decimal.Value, whole: Decimal.Value, digits = 0): string {
  const judged = judgeShare(part, whole, digits)
  if (judged.shown === 'percent') return `${judged.percent.toString()}%`
  return { none: '0%', all: '100%', 'nearly all': '>99%', 'nearly none': '<1%' }[judged.shown]
}

/** Whether a part's share of a whole, as fmtShare shows it, is at least `percent`, a whole percent from 1 to 99. */
export function shareReaches(part: Decimal.Value, whole: Decimal.Value, digits: number, percent: number): boolean {
  const judged = judgeShare(part, whole, digits)
  if (judged.shown === 'percent') return judged.percent.gte(percent)
  return judged.shown === 'all' || judged.shown === 'nearly all'
}

/** The noun for a count: `one` for exactly 1, `many` otherwise (0 included). */
export function plural(count: number, one: string, many: string): string {
  return count === 1 ? one : many
}

/** A count and its noun, e.g. "1 site", "3 sites". */
export function counted(count: number, one: string, many: string): string {
  return `${count} ${plural(count, one, many)}`
}
