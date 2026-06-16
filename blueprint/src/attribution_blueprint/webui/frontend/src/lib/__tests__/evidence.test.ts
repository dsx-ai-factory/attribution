import { describe, it, expect } from 'vitest'
import { buildVerdictMarkdown } from '@/lib/evidence'
import type { Run } from '@/interfaces/attribution'

const base: Run = {
  id: '1', jobId: 'team/x', workload: 'team/x', attempt: 1, attemptUid: 'u', user: 'alice',
  status: 'FAILED', nodes: 1, nodeIds: ['n1'], elapsedH: 1.5, gpusPerNode: 8, gpuCount: 8,
  endTime: '2026-05-01T01:00:00Z',
  startTime: '2026-05-01T00:00:00Z', outcome: 'hardware', verdictLabel: 'Hardware',
  detail: 'GPU off bus', rootSymptom: { symptom: 'GpuFell', rule: 'gpu_fell_off_bus', time: 'T', line: 'Xid 79' },
  symptoms: [{ name: 'GpuFell', sites: 1 }], responsibleNodes: ['n1'], pipeline: '0.1.0', service: 'v2', log: [],
}

describe('buildVerdictMarkdown', () => {
  it('includes tenant and operator guidance when personas are present', () => {
    const run = {
      ...base,
      diagnose: {
        hasPersona: true,
        title: 'NVLink fault',
        summary: 'cascade',
        impact: 'rank exits',
        action: 'Resubmit on healthy allocation',
        supportDetails: ['tenant-only'],
      },
      operatorDiagnose: {
        hasPersona: true,
        title: 'NVLink fault',
        summary: 'fabric issue',
        impact: 'node needs drain',
        action: 'Drain node and inspect fabric',
        supportDetails: ['dmesg', 'fabricmanager.log'],
      },
    }
    const md = buildVerdictMarkdown(run, 'https://x/')
    expect(md).toContain('**Summary:** fabric issue')
    expect(md).not.toContain('**Summary:** GPU off bus')
    expect(md).toContain('## Tenant SRE guidance')
    expect(md).toContain('**Recommended action:** Resubmit on healthy allocation')
    expect(md).toContain('## Cloud operator guidance')
    expect(md).toContain('**Recommended action:** Drain node and inspect fabric')
    expect(md).toContain('**Include on escalation:** dmesg, fabricmanager.log')
    expect(md).not.toContain('tenant-only')
  })

  it('falls back to raw fields without diagnose', () => {
    const md = buildVerdictMarkdown(base, 'https://x/')
    expect(md).toContain('Hardware')
    expect(md).toContain('GpuFell')
  })
})
