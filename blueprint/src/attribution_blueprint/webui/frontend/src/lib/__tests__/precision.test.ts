import { describe, it, expect } from 'vitest'
import { fmtGpuHours, fmtNum, fmtPercent } from '@/lib/format'
import { gpuHours, elapsedHours } from '@/lib/metric'
import { nodeExportData, runsToCSV, verdictExportData } from '@/lib/export'
import type { RunSummary } from '@/interfaces/attribution'

function run(over: Partial<RunSummary>): RunSummary {
  return {
    id: 'r', jobId: 'team/x', workload: 'team/x', attempt: 1, attemptUid: 'u', user: 'u',
    status: 'FAILED', nodes: 1, nodeIds: ['n1'], elapsedH: 0.1, gpusPerNode: 8, gpuCount: 8,
    endTime: '2026-05-01T01:00:00Z', startTime: '2026-05-01T00:00:00Z',
    outcome: 'hardware', verdictLabel: '', detail: '',
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

describe('fmtPercent', () => {
  it('rounds to the requested dp', () => {
    expect(fmtPercent(79.6)).toBe('80')
    expect(fmtPercent(12.345, 1)).toBe('12.3')
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
