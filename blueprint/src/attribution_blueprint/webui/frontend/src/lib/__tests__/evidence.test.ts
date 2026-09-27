// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { describe, it, expect } from 'vitest'
import { buildVerdictMarkdown } from '@/lib/evidence'
import { findSymptomDocumentation, symptomDocumentationApiUrl } from '@/lib/documentation'
import type { RunSummary } from '@/interfaces/attribution'
import type { SymptomDocumentationCatalog } from '@/interfaces/documentation'

const base: RunSummary = {
  decisionGraphState: 'absent',
  id: '1', jobId: 'team/x', workload: 'team/x', attempt: 1, attemptUid: 'u', user: 'alice',
  status: 'FAILED', nodes: 1, nodeIds: ['n1'], elapsedH: 1.5, gpusPerNode: 8, gpuCount: 8,
  endTime: '2026-05-01T01:00:00Z',
  startTime: '2026-05-01T00:00:00Z', outcome: 'hardware', verdictLabel: 'Hardware',
  detail: 'GPU off bus', rootSymptom: { symptom: 'GpuFell', rule: 'gpu_fell_off_bus', time: 'T', line: 'Xid 79' },
  symptoms: [{ name: 'GpuFell', sites: 1 }], responsibleNodes: ['n1'], pipeline: '0.1.0', service: 'v2',
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

describe('symptom documentation API configuration', () => {
  const endpoint = 'https://reference.example/api/docs/symptoms'

  it.each(['webui.example', 'localhost', '127.0.0.1', '[::1]'])('accepts explicitly configured HTTPS from %s', (hostname) => {
    expect(symptomDocumentationApiUrl(endpoint, hostname)).toBe(endpoint)
  })

  it.each(['localhost', '127.0.0.1', '[::1]'])('allows explicitly configured HTTP for loopback preview %s', (hostname) => {
    const httpEndpoint = 'http://reference.example/api/docs/symptoms'
    expect(symptomDocumentationApiUrl(httpEndpoint, hostname)).toBe(httpEndpoint)
  })

  it.each(['webui.example', 'localhost.example', '127.0.0.1.example', '127.0.0.2', '::1', '192.168.1.2'])('rejects HTTP outside the exact loopback hostnames: %s', (hostname) => {
    expect(symptomDocumentationApiUrl('http://reference.example/api/docs/symptoms', hostname)).toBeNull()
  })

  it.each([undefined, '', '/api/docs/symptoms', 'not a URL', 'https://', '//reference.example/api/docs/symptoms', 'ftp://reference.example/catalog', 'javascript:alert(1)', 'data:application/json,{}'])('rejects absent, relative, malformed, or unsafe configuration %s', (url) => {
    expect(symptomDocumentationApiUrl(url, 'localhost')).toBeNull()
  })

  it.each(['username', 'password'] as const)('rejects a configured URL with a %s', (field) => {
    const url = new URL(endpoint)
    url[field] = 'test'
    expect(symptomDocumentationApiUrl(url.href, 'localhost')).toBeNull()
  })
})

describe('symptom documentation references', () => {
  const entry = { id: 'xid-48', title: 'XID 48', doc_url: 'https://docs.example/xid-48/' }
  const catalog: SymptomDocumentationCatalog = { symptoms: {
    'xid-48': entry,
    'XID 48': entry,
    'ECC Error': entry,
  } }

  it('uses the exact API key for the recorded symptom and preserves the canonical ID and URL', () => {
    for (const symptom of ['xid-48', 'XID 48', 'ECC Error']) {
      expect(findSymptomDocumentation(catalog, symptom)).toEqual(entry)
    }
    for (const symptom of [undefined, null, '', 'XID-48', 'xid 48', ' xid-48', 'xid-48 ', 'not-published']) {
      expect(findSymptomDocumentation(catalog, symptom)).toBeUndefined()
    }
  })

  it('does not search titles or aliases or read inherited entries', () => {
    const canonicalOnly = { symptoms: { 'xid-48': { ...entry, aliases: ['ECC Error'] } } }
    expect(findSymptomDocumentation(canonicalOnly, 'XID 48')).toBeUndefined()
    expect(findSymptomDocumentation(canonicalOnly, 'ECC Error')).toBeUndefined()
    expect(findSymptomDocumentation({ symptoms: Object.create({ 'xid-48': entry }) }, 'xid-48')).toBeUndefined()
    expect(findSymptomDocumentation({ symptoms: JSON.parse(JSON.stringify({ ['__proto__']: entry })) }, '__proto__')).toEqual(entry)
  })

  it('hides unavailable or malformed references without affecting attribution', () => {
    for (const value of [undefined, null, [], {}, { symptoms: null }, { symptoms: [] }, { symptoms: { 'xid-48': null } }]) {
      expect(findSymptomDocumentation(value, 'xid-48')).toBeUndefined()
    }
    for (const patch of [{ id: undefined }, { id: null }, { id: '' }, { id: 48 }, { title: null }, { title: '' }, { doc_url: undefined }]) {
      expect(findSymptomDocumentation({ symptoms: { 'xid-48': { ...entry, ...patch } } }, 'xid-48')).toBeUndefined()
    }
  })

  it.each(['http://docs.example/xid-48/', 'javascript:alert(1)', 'data:text/html,unsafe', 'file:///etc/passwd', '/relative', ''])('rejects unsafe or invalid link %s', (doc_url) => {
    expect(findSymptomDocumentation({ symptoms: { 'xid-48': { ...entry, doc_url } } }, 'xid-48')).toBeUndefined()
  })

  it.each(['username', 'password'] as const)('rejects a document URL with a %s', (field) => {
    const url = new URL(entry.doc_url)
    url[field] = 'test'
    expect(findSymptomDocumentation({ symptoms: { 'xid-48': { ...entry, doc_url: url.href } } }, 'xid-48')).toBeUndefined()
  })
})
