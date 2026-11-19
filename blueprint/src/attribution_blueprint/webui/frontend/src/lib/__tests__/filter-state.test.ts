// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { applyFilter, DEFAULT_FILTER, parseFilter, serializeFilter } from '@/lib/filter-state'
import type { RunSummary } from '@/interfaces/attribution'
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

describe('status default', () => {
  it('shows every run when the URL carries no status', () => {
    expect(parseFilter(new URLSearchParams('')).status).toBe('all')
  })

  it('still restricts to failures for a bookmarked status=failed URL', () => {
    expect(parseFilter(new URLSearchParams('status=failed')).status).toBe('failed')
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

describe('free-text search', () => {
  const run = (id: string, nodeIds: string[], attemptUid: string): RunSummary => ({
    id,
    jobId: id,
    workload: '—',
    attempt: 1,
    attemptUid,
    user: 'alice',
    status: 'FAILED',
    nodes: nodeIds.length,
    nodeIds,
    elapsedH: 1,
    gpusPerNode: 8,
    gpuCount: 8,
    endTime: '2026-05-01T01:00:00Z',
    startTime: '2026-05-01T00:00:00Z',
    outcome: 'hardware',
    verdictLabel: 'Hardware',
    detail: '',
    decisionGraphState: 'absent',
    rootSymptom: { symptom: 'Xid79', rule: 'xid', time: '', line: '' },
    symptoms: [],
    responsibleNodes: ['node-a'],
    pipeline: '1',
    service: '2',
  })
  const runs = [run('100', ['alloc-node-7', 'node-a'], 'attempt-uid-7'), run('101', ['node-a'], 'sluid-b')]

  it.each(['ALLOC-NODE-7', 'attempt-uid-7'])('matches %s against allocated nodes and attempt UID', (q) => {
    expect(applyFilter(runs, { ...DEFAULT_FILTER, q }).map((r) => r.id)).toEqual(['100'])
  })
})
