// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest'
import { counted, fmtGpuHours, fmtNum, fmtShare, plural, shareReaches } from '@/lib/format'
import { gpuHours, elapsedHours } from '@/lib/metric'
import { nodeExportData, runsToCSV, verdictExportData } from '@/lib/export'
import type { RunSummary } from '@/interfaces/attribution'

function run(over: Partial<RunSummary>): RunSummary {
  return {
    id: 'r', jobId: 'team/x', workload: 'team/x', attempt: 1, attemptUid: 'u', user: 'u',
    status: 'FAILED', nodes: 1, nodeIds: ['n1'], elapsedH: 0.1, gpusPerNode: 8, gpuCount: 8,
    endTime: '2026-05-01T01:00:00Z', startTime: '2026-05-01T00:00:00Z',
    outcome: 'hardware', verdictLabel: '', detail: '',
    decisionGraphState: 'absent',
    rootSymptom: { symptom: 'S', rule: 'r', time: '', line: '' },
    symptoms: [], responsibleNodes: ['n1'], pipeline: '', service: '',
    ...over,
  }
}

describe('fmtNum', () => {
  it('drops the trailing .00 on whole numbers', () => {
    expect(fmtNum(30)).toBe('30')
    expect(fmtNum(1000)).toBe('1000')
  })
  it('keeps real decimals at ≤2dp', () => {
    expect(fmtNum(31.666)).toBe('31.67')
    expect(fmtNum(31.6)).toBe('31.6')
  })
})

describe('fmtGpuHours', () => {
  it('renders raw hours below 1000 with no k', () => {
    expect(fmtGpuHours(31.67)).toEqual({ value: '31.67', k: false })
    expect(fmtGpuHours(999.99)).toEqual({ value: '999.99', k: false })
  })
  it('switches to k only at/above 1000', () => {
    expect(fmtGpuHours(1000)).toEqual({ value: '1', k: true })
    expect(fmtGpuHours(1234)).toEqual({ value: '1.23', k: true })
  })
})

describe('fmtShare', () => {
  it('rounds a share to a whole percent, but never a part to all of it or a nonzero part to none', () => {
    expect([79.6, 12.345, 100, 0, 50.5].map((part) => fmtShare(part, 100))).toEqual(['80%', '12%', '100%', '0%', '51%'])
    expect([99.5, 99.96, 0.4, 0.01].map((part) => fmtShare(part, 100, 2))).toEqual(['>99%', '>99%', '<1%', '<1%'])
    expect(fmtShare(1, 3)).toBe('33%')
  })

  it('judges all and none at the precision the values are shown to', () => {
    // A float sum of the same GPU-hours that differs in its last digits is all of it.
    expect(fmtShare(179.944771, 179.94477100000003, 1)).toBe('100%')
    expect(fmtShare(179.94477100000003, 179.944771, 1)).toBe('100%')
    // What is left shows as 0.0 GPU-hours, so the part is all of it; left 0.1, it is not.
    expect(fmtShare(179.91, 179.94, 1)).toBe('100%')
    expect(fmtShare(179.86, 179.94, 1)).toBe('>99%')
    // A part shown as 0.0 is none of it; shown as 0.1 it is some.
    expect(fmtShare(0.04, 179.94, 1)).toBe('0%')
    expect(fmtShare(0.05, 179.94, 1)).toBe('<1%')
    // A whole too small to show tells nothing, so its exact parts do.
    expect(fmtShare(0.03, 0.03, 1)).toBe('100%')
    expect(fmtShare(0.02, 0.03, 1)).toBe('67%')
    // A part beyond the whole reads as it is, never as nearly all of it.
    expect(fmtShare(2, 1)).toBe('200%')
  })

  it('tells whether a share, as it is shown, reaches a percent', () => {
    expect(shareReaches(79.6, 100, 0, 80)).toBe(true)
    expect(shareReaches(79.4, 100, 0, 80)).toBe(false)
    // Shown as all of it, or nearly: it reaches; shown as none, or nearly: it does not.
    expect(shareReaches(179.91, 179.94, 1, 80)).toBe(true)
    expect(shareReaches(99.6, 100, 0, 99)).toBe(true)
    expect(shareReaches(0.04, 179.94, 1, 1)).toBe(false)
    expect(shareReaches(0.4, 100, 0, 1)).toBe(false)
    // As fmtShare judges it: 0.049 of 0.06 is shown as none, 0.151 of 0.2 as all.
    expect([fmtShare(0.049, 0.06, 1), shareReaches(0.049, 0.06, 1, 80)]).toEqual(['0%', false])
    expect([fmtShare(0.151, 0.2, 1), shareReaches(0.151, 0.2, 1, 80)]).toEqual(['100%', true])
  })
})

describe('counts and their nouns', () => {
  it('takes the singular for exactly one and the plural otherwise, zero included', () => {
    expect([0, 1, 2].map((n) => counted(n, 'run', 'runs'))).toEqual(['0 runs', '1 run', '2 runs'])
    expect(plural(1, 'GPU', 'GPUs')).toBe('GPU')
    expect(plural(8, 'GPU', 'GPUs')).toBe('GPUs')
  })
})

describe('exact Decimal sums avoid IEEE float noise', () => {
  it('gpuHours multiplies job GPU count by runtime without float noise', () => {
    const runs = [run({}), run({}), run({})]
    expect(gpuHours(runs).toNumber()).toBe(2.4)
    expect(elapsedHours(runs).toNumber()).toBe(0.3)
  })

  it('nodeExportData.gpu_hours_observed carries no float noise', () => {
    const runs = [run({}), run({}), run({})] // 3 × 0.1
    const data = nodeExportData('n1', runs)
    expect(data.gpu_hours_observed).toBe(2.4)
    // Round-trips through JSON without a 0.30000000000000004 tail.
    expect(JSON.stringify(data)).not.toContain('0000000')
  })

  it('attempt GPU-hours supports partial-node jobs', () => {
    const runs = [run({ nodes: 2, nodeIds: ['n1', 'n2'], elapsedH: 1.25, gpusPerNode: 8, gpuCount: 10 })]
    expect(gpuHours(runs).toNumber()).toBe(12.5)
  })

  it('per-run elapsed_hours is rounded clean at the export boundary', () => {
    const runs = [run({ elapsedH: 1.234567 })]
    const data = nodeExportData('n1', runs)
    expect(data.attempts[0].elapsed_hours).toBe(1.2346)
    expect(JSON.stringify(data)).not.toContain('0000000')
  })

  it('UI export data names the run identifier execution_context', () => {
    const data = verdictExportData(run({}))
    expect(data).toHaveProperty('execution_context', 'team/x')
    expect(data).not.toHaveProperty('job_id')
  })

  it('node evidence export data names attempt identifiers execution_context', () => {
    const data = nodeExportData('n1', [run({})])
    expect(data.attempts[0]).toHaveProperty('execution_context', 'team/x')
    expect(data.attempts[0]).not.toHaveProperty('job_id')
  })

  it('filtered run CSV exports execution_context as the identifier column', () => {
    const [header] = runsToCSV([run({})]).split('\n')
    expect(header.split(',')).toContain('execution_context')
    expect(header.split(',')).not.toContain('job_id')
  })
})
