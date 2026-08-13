import { describe, expect, it } from 'vitest'
import { parseFilter, serializeFilter } from '@/lib/filter-state'
import type { FleetFilter } from '@/interfaces/filter'

describe('filter date window mode', () => {
  it('preserves explicit custom mode even when dates match a preset span', () => {
    const filter = parseFilter(new URLSearchParams('status=failed&window=custom&from=2026-05-01&to=2026-05-31'))

    expect(filter.window).toBe('custom')
    expect(serializeFilter(filter).get('window')).toBe('custom')
  })

  it('does not serialize preset-like window values', () => {
    const filter: FleetFilter = {
      status: 'failed',
      axis: 'symptom',
      metric: 'count',
      from: '2026-05-01',
      to: '2026-05-31',
    }

    expect(parseFilter(new URLSearchParams('window=30d')).window).toBeUndefined()
    expect(serializeFilter(filter).has('window')).toBe(false)
  })
})

describe('date range validation', () => {
  it('keeps a well-formed ordered pair', () => {
    const filter = parseFilter(new URLSearchParams('from=2026-05-01&to=2026-05-31'))

    expect(filter.from).toBe('2026-05-01')
    expect(filter.to).toBe('2026-05-31')
  })

  it('keeps a single-day range (from equals to)', () => {
    const filter = parseFilter(new URLSearchParams('from=2026-05-01&to=2026-05-01'))

    expect(filter.from).toBe('2026-05-01')
    expect(filter.to).toBe('2026-05-01')
  })

  it('drops an inverted range (from after to)', () => {
    const filter = parseFilter(new URLSearchParams('from=2026-05-31&to=2025-05-01'))

    expect(filter.from).toBeUndefined()
    expect(filter.to).toBeUndefined()
  })

  it('drops a partial pair (only one bound present)', () => {
    expect(parseFilter(new URLSearchParams('from=2026-05-01')).from).toBeUndefined()
    expect(parseFilter(new URLSearchParams('to=2026-05-01')).to).toBeUndefined()
  })

  it('drops malformed dates', () => {
    const filter = parseFilter(new URLSearchParams('from=garbage&to=2026-05-31'))

    expect(filter.from).toBeUndefined()
    expect(filter.to).toBeUndefined()
  })

  it('drops impossible calendar dates', () => {
    const filter = parseFilter(new URLSearchParams('from=2026-02-30&to=2026-05-31'))

    expect(filter.from).toBeUndefined()
    expect(filter.to).toBeUndefined()
  })
})
