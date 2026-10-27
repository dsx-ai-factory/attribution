// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest'
import type { PersonaDiagnose, RunDetail } from '@/interfaces/attribution'
import {
  logToText,
  nodeExportData,
  runsExportData,
  runsToCSV,
  slug,
  toYaml,
  utcDateStamp,
  verdictExportData,
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
  const tenant: PersonaDiagnose = {
    hasPersona: true,
    title: 'Application failure',
    summary: 'The application could not complete its operation.',
    impact: 'The attempt stopped.',
    action: '1. Check the application settings.\n2. Retry your workload.\n3. Confirm the failed operation completes.',
    supportDetails: [
      'Include the NVIDIA Attribution diagnosis, job ID and available logs; incomplete evidence is OK.',
      'Include the application version.\nDescribe any changes and their results.',
    ],
  }
  const operator: PersonaDiagnose = {
    ...tenant,
    summary: 'PRIVATE_OPERATOR_SUMMARY',
    impact: 'PRIVATE_OPERATOR_IMPACT',
    action: 'PRIVATE_OPERATOR_ACTION',
    supportDetails: ['PRIVATE_OPERATOR_SUPPORT'],
  }

  it('includes only tenant guidance in JSON and YAML without changing existing fields', () => {
    const input = run({
      diagnose: Object.assign({}, tenant, { internal_notes: 'PRIVATE_CATALOG_NOTE' }),
      operatorDiagnose: operator,
    })
    const output = verdictExportData(input)
    const { tenant_guidance: guidance, ...existingFields } = output
    const { tenant_guidance: unavailable, ...withoutGuidance } = verdictExportData(run())
    expect(unavailable).toBeNull()
    expect(existingFields).toEqual(withoutGuidance)
    expect(guidance).toEqual({
      summary: tenant.summary,
      impact: tenant.impact,
      action: tenant.action,
      support_details: tenant.supportDetails,
    })
    expect(guidance!.support_details).not.toBe(tenant.supportDetails)

    const json = JSON.stringify(output, null, 2)
    const yaml = toYaml(output)
    expect(JSON.parse(json).tenant_guidance).toEqual(guidance)
    expect(yaml).toContain([
      'tenant_guidance:',
      `  summary: ${tenant.summary}`,
      `  impact: ${tenant.impact}`,
      `  action: ${JSON.stringify(tenant.action)}`,
      '  support_details:',
      `    - ${JSON.stringify(tenant.supportDetails[0])}`,
      `    - ${JSON.stringify(tenant.supportDetails[1])}`,
    ].join('\n'))
    for (const serialized of [json, yaml]) {
      expect(serialized).not.toContain('PRIVATE_')
      expect(serialized).not.toContain('operatorDiagnose')
      expect(serialized).not.toContain('internal_notes')
    }
  })

  it.each([undefined, { ...tenant, hasPersona: false }])(
    'exports null without falling back to operator guidance when tenant guidance is unavailable', (diagnose) => {
      const output = verdictExportData(run({ diagnose, operatorDiagnose: operator }))
      expect(JSON.parse(JSON.stringify(output)).tenant_guidance).toBeNull()
      expect(toYaml(output)).toContain('tenant_guidance: null')
      expect(toYaml(output)).not.toContain(tenant.action)
      expect(JSON.stringify(output)).not.toContain('PRIVATE_')
      expect(toYaml(output)).not.toContain('PRIVATE_')
    },
  )

  it('keeps empty tenant fields empty instead of substituting operator guidance', () => {
    const output = verdictExportData(run({
      diagnose: { ...tenant, summary: '', impact: '', action: '', supportDetails: [] },
      operatorDiagnose: operator,
    }))
    expect(JSON.parse(JSON.stringify(output)).tenant_guidance).toEqual({
      summary: '', impact: '', action: '', support_details: [],
    })
    expect(toYaml(output)).toContain('tenant_guidance:\n  summary: ""\n  impact: ""\n  action: ""\n  support_details: []')
  })

  it('includes each run\'s own tenant guidance in bulk JSON and YAML exports', () => {
    const secondTenant = { ...tenant, action: 'No action is needed.', supportDetails: [] }
    const output = runsExportData([
      run({ attemptUid: 'a1', diagnose: tenant, operatorDiagnose: operator }),
      run({ attemptUid: 'a2', diagnose: secondTenant, operatorDiagnose: operator }),
      run({ attemptUid: 'a3', operatorDiagnose: operator }),
    ])
    expect(JSON.parse(JSON.stringify(output)).map((item: { tenant_guidance: unknown }) => item.tenant_guidance)).toEqual([
      { summary: tenant.summary, impact: tenant.impact, action: tenant.action, support_details: tenant.supportDetails },
      { summary: secondTenant.summary, impact: secondTenant.impact, action: secondTenant.action, support_details: [] },
      null,
    ])
    const yaml = toYaml(output)
    expect(yaml.match(/tenant_guidance:/g)).toHaveLength(3)
    expect(yaml).toContain(`    action: ${JSON.stringify(tenant.action)}`)
    expect(yaml).toContain(`    action: ${secondTenant.action}`)
    expect(yaml).toContain('  tenant_guidance: null')
    expect(yaml).not.toContain('PRIVATE_')
    expect(JSON.stringify(output)).not.toContain('PRIVATE_')
  })

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
