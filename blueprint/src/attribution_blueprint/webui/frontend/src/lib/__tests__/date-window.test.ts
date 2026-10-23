// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { boundsOutsideWindow, detectPreset, isoDate, presetRange } from '@/lib/date-window'
import type { FleetFilter } from '@/interfaces/filter'

const now = new Date('2026-06-11T10:00:00Z')

function filterWith(from: string, to: string): FleetFilter {
  return { status: 'failed', axis: 'symptom', metric: 'count', from, to }
}

describe('presetRange', () => {
  it('returns a window ending at the anchor', () => {
    const { from, to } = presetRange('30d', now)

    expect(to.getTime()).toBe(now.getTime())
    expect(to.getTime() - from.getTime()).toBe(30 * 86400000)
  })

  it('does not alias the anchor instance', () => {
    expect(presetRange('7d', now).to).not.toBe(now)
  })
})

describe('detectPreset', () => {
  it('defaults to all when no dates are set', () => {
    expect(detectPreset({ status: 'failed', axis: 'symptom', metric: 'count' }, now)).toBe('all')
  })

  it('detects a 30-day window ending now as 30d', () => {
    const { from, to } = presetRange('30d', now)

    expect(detectPreset(filterWith(isoDate(from), isoDate(to)), now)).toBe('30d')
  })

  it('reports a stale 30-day window as custom, not 30d', () => {
    expect(detectPreset(filterWith('2025-09-01', '2025-10-01'), now)).toBe('custom')
  })

  it('reports a stale 7-day window as custom, not 7d', () => {
    expect(detectPreset(filterWith('2025-09-24', '2025-10-01'), now)).toBe('custom')
  })

  it('detects a 24h window ending now as 24h', () => {
    const { from, to } = presetRange('24h', now)

    expect(detectPreset(filterWith(isoDate(from), isoDate(to)), now)).toBe('24h')
  })

  it('keeps a preset detected through the day after a click near UTC midnight', () => {
    // Clicked 7d at 23:55Z on June 10; rendered again the next morning.
    const nextMorning = new Date('2026-06-11T09:00:00Z')

    expect(detectPreset(filterWith('2026-06-03', '2026-06-10'), nextMorning)).toBe('7d')
  })

  it('reports a window whose to-day ended more than a day ago as custom', () => {
    expect(detectPreset(filterWith('2026-06-02', '2026-06-09'), now)).toBe('custom')
  })

  it('reports a window ending in the future as custom', () => {
    expect(detectPreset(filterWith('2026-06-06', '2026-06-13'), now)).toBe('custom')
  })
})

describe('boundsOutsideWindow', () => {
  it('returns null for an all-time filter', () => {
    expect(boundsOutsideWindow(
      { min: '2026-05-01', max: '2026-05-03' },
      { status: 'failed', axis: 'symptom', metric: 'count' },
    )).toBeNull()
  })

  it('returns null when the bounds sit inside the window', () => {
    expect(boundsOutsideWindow(
      { min: '2026-05-02', max: '2026-05-02' },
      filterWith('2026-05-01', '2026-05-03'),
    )).toBeNull()
  })

  it('returns null when the bounds land exactly on both window endpoints', () => {
    // The comparison is inclusive on purpose: the backend covers both endpoint days
    // in full, so a run on the first or last day is inside the window, not outside it.
    expect(boundsOutsideWindow(
      { min: '2026-05-01', max: '2026-05-03' },
      filterWith('2026-05-01', '2026-05-03'),
    )).toBeNull()
  })

  it('returns the bounds when they fall entirely outside the window', () => {
    const bounds = { min: '2026-05-01', max: '2026-05-03' }

    expect(boundsOutsideWindow(bounds, filterWith('2020-01-01', '2020-01-02'))).toEqual(bounds)
    expect(boundsOutsideWindow(bounds, filterWith('2030-01-01', '2030-01-02'))).toEqual(bounds)
  })

  it('returns the bounds when they straddle the window edge', () => {
    const bounds = { min: '2026-04-28', max: '2026-05-02' }

    expect(boundsOutsideWindow(bounds, filterWith('2026-05-01', '2026-05-03'))).toEqual(bounds)
  })

  it('returns null when the database has no rows', () => {
    expect(boundsOutsideWindow(null, filterWith('2026-05-01', '2026-05-03'))).toBeNull()
    expect(boundsOutsideWindow(
      { min: null, max: null },
      filterWith('2026-05-01', '2026-05-03'),
    )).toBeNull()
  })

  it('returns the bounds when the window sits strictly inside them', () => {
    const bounds = { min: '2026-05-01', max: '2026-05-10' }

    expect(boundsOutsideWindow(bounds, filterWith('2026-05-03', '2026-05-05'))).toEqual(bounds)
  })

  it('returns null when only one end of the window is set', () => {
    const bounds = { min: '2026-05-01', max: '2026-05-10' }
    const allTime: FleetFilter = { status: 'failed', axis: 'symptom', metric: 'count' }

    expect(boundsOutsideWindow(bounds, { ...allTime, from: '2026-05-03' })).toBeNull()
    expect(boundsOutsideWindow(bounds, { ...allTime, to: '2026-05-05' })).toBeNull()
  })

  it('returns null when only one end of the bounds is known', () => {
    expect(boundsOutsideWindow(
      { min: null, max: '2026-05-10' },
      filterWith('2026-05-01', '2026-05-03'),
    )).toBeNull()
    expect(boundsOutsideWindow(
      { min: '2026-04-01', max: null },
      filterWith('2026-05-01', '2026-05-03'),
    )).toBeNull()
  })
})
