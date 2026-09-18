import { afterEach, describe, expect, it, vi } from 'vitest'
import type { RunDetail } from '@/interfaces/attribution'
import {
  logToText,
  nodeExportData,
  runsExportData,
  runsToCSV,
  slug,
  toYaml,
  utcDateStamp,
} from '@/lib/export'

function run(overrides: Partial<RunDetail> = {}): RunDetail {
  return {
    id: 'run-1',
    jobId: 'team/job',
    workload: 'team/job',
    attempt: 1,
    attemptUid: 'attempt-1',
    user: 'alice',
    status: 'FAILED',
    nodes: 2,
    nodeIds: ['node-a', 'node-b'],
    elapsedH: 1.25,
    gpusPerNode: 8,
    gpuCount: 12,
    endTime: '2026-05-01T01:15:00Z',
    startTime: '2026-05-01T00:00:00Z',
    outcome: 'hardware',
    verdictLabel: 'GPU failure',
    detail: 'A GPU failed.',
    decisionGraphState: 'absent',
    rootSymptom: { symptom: 'GpuFell', rule: 'gpu_fell', time: '00:30', line: 'Xid 79' },
    symptoms: [{ name: 'GpuFell', sites: 1 }],
    responsibleNodes: ['node-a'],
    pipeline: '1.0',
    service: '2.0',
    log: [],
    attribution: {
      rootSymptom: { state: 'absent', data: null },
      symptomSites: { state: 'absent', data: null },
      decisionGraph: { state: 'absent', data: null },
    },
    ...overrides,
  }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('toYaml', () => {
  it('serializes nested values, empty collections, and quoted scalars', () => {
    const yaml = toYaml({
      plain: 'value',
      'unsafe key': 'line one\n"line two"',
      enabled: true,
      invalid: Number.POSITIVE_INFINITY,
      empty: [],
      nested: { count: 2 },
      rows: [{ name: 'node:1', tags: ['gpu', 'test'] }, {}],
    })

    expect(yaml).toContain('plain: value')
    expect(yaml).toContain('"unsafe key": "line one\\n\\"line two\\""')
    expect(yaml).toContain('enabled: true')
    expect(yaml).toContain('invalid: null')
    expect(yaml).toContain('empty: []')
    expect(yaml).toContain('nested:\n  count: 2')
    expect(yaml).toContain('rows:\n  - name: "node:1"')
    expect(yaml).toContain('    tags:\n      - gpu\n      - test')
    expect(yaml).toContain('  - {}')
  })

  it('serializes top-level arrays and primitive values', () => {
    expect(toYaml([])).toBe('[]')
    expect(toYaml({})).toBe('{}')
    expect(toYaml(['yes', null, 4])).toBe('- "yes"\n- null\n- 4')
    expect(toYaml(undefined)).toBe('null')
  })
})

describe('run exports', () => {
  it('quotes CSV delimiters and removes the workload placeholder', () => {
    const csv = runsToCSV([
      run({ jobId: 'team,"blue"\njob', workload: '—', user: 'last, first' }),
    ])

    expect(csv).toContain('"team,""blue""\njob"')
    expect(csv).toContain(',"last, first",')
    expect(csv).not.toContain(',—,')
  })

  it('exports each run and summarizes only attempts attributed to a node', () => {
    const runs = [
      run({ attemptUid: 'a1', responsibleNodes: ['node-a'], rootSymptom: { symptom: 'Xid79', rule: 'r1', time: '', line: '' } }),
      run({ attemptUid: 'a2', responsibleNodes: ['node-b'], rootSymptom: { symptom: 'Xid48', rule: 'r2', time: '', line: '' } }),
      run({ attemptUid: 'a3', responsibleNodes: ['node-a'], rootSymptom: { symptom: 'Xid79', rule: 'r1', time: '', line: '' } }),
    ]

    expect(runsExportData(runs)).toHaveLength(3)
    expect(nodeExportData('node-a', runs)).toMatchObject({
      attempts_impacted: 2,
      distinct_symptoms: 1,
      symptoms: [{ name: 'Xid79', count: 2 }],
      attempts: [{ attempt_uid: 'a1' }, { attempt_uid: 'a3' }],
    })
  })

  it('formats each log level as fixed-width text', () => {
    const text = logToText(run({
      log: [
        { t: '00:01', level: 'symptom', text: 'symptom' },
        { t: '00:02', level: 'error', text: 'error' },
        { t: '00:03', level: 'debug', text: 'debug' },
        { t: '00:04', level: 'info', text: 'info' },
      ],
    }))

    expect(text).toBe([
      '00:01 SYMPTOM symptom',
      '00:02 ERROR   error',
      '00:03 DEBUG   debug',
      '00:04 INFO    info',
    ].join('\n'))
  })
})

describe('export filenames', () => {
  it('normalizes unsafe names and applies the maximum length', () => {
    expect(slug('  Team / GPU Job  ')).toBe('team-gpu-job')
    expect(slug('***')).toBe('export')
    expect(slug('abcdefgh', 4)).toBe('abcd')
  })

  it('uses UTC for date stamps', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-05-01T23:07:00Z'))

    expect(utcDateStamp()).toBe('20260501-2307')
  })
})
