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

/** Percent, rounded for display only. */
export function fmtPercent(n: Decimal.Value, dp = 0): string {
  return new Decimal(n).toDecimalPlaces(dp).toString()
}
