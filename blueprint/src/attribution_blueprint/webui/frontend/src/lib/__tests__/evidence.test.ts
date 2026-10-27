// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi } from 'vitest'
import { buildNodeMarkdown, buildVerdictMarkdown } from '@/lib/evidence'
import { utcStamp } from '@/lib/time'
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
  it('names the verdict and every unrecorded value as the page does', () => {
    // The verdict badge's label, whichever outcome bucket holds it; the outcome's name only when it has none.
    expect(buildVerdictMarkdown({ ...base, outcome: 'multi', verdictLabel: 'Cancelled, Workload Owner' }, 'https://x/'))
      .toMatch(/^\*Attribution escalation: Cancelled, Workload Owner · GpuFell\*/)
    expect(buildVerdictMarkdown({ ...base, verdictLabel: '' }, 'https://x/')).toMatch(/^\*Attribution escalation: Hardware Fault · GpuFell\*/)
    // A value the record does not hold reads "Not recorded", never an empty pair of backticks.
    const md = buildVerdictMarkdown({ ...base, rootSymptom: { symptom: '', rule: '', time: '', line: '' } }, 'https://x/')
    expect(md).toMatch(/^\*Attribution escalation: Hardware · Root symptom not recorded\*/)
    expect(md).toContain('**Root symptom:** Not recorded')
    expect(md).toContain('**Rule:** Not recorded')
    expect(md).toContain('**Matched log:**\nNot recorded')
    expect(md).not.toContain('``')
    // The verdict line names it as the header does; empty names say so; an
    // empty list reads "None"; the summary, as the page's service detail, and
    // a node link when no node is responsible are left out.
    const bare = buildVerdictMarkdown({ ...base, verdictLabel: '', detail: '', user: '', pipeline: '', service: '', responsibleNodes: [] }, 'https://x/')
    expect(bare).toContain('**Verdict:** Hardware Fault')
    expect(bare).not.toContain('**Summary:**')
    expect(bare).toContain('**User / tenant:** Not recorded')
    expect(bare).toContain('**Responsible node(s):**\nNone\n')
    expect(bare).not.toContain('Node detail')
    expect(bare).toContain('- Pipeline / service: Not recorded / Not recorded')
    // No label is left with nothing after it, and no placeholder of another wording.
    expect(bare).not.toMatch(/ $|: \/ | \/ $|n\/a|\(none/m)
    const nodeMd = buildNodeMarkdown('n1', [{ ...base, rootSymptom: { ...base.rootSymptom, symptom: '' } }], 'https://x/')
    expect(nodeMd).toContain('- Not recorded × 1')
    expect(nodeMd).toContain('· Root symptom not recorded ·')
    expect(nodeMd).not.toContain('``')
    // A node no attempt blames lists no symptom and no attempt.
    const quiet = buildNodeMarkdown('n2', [base], 'https://x/')
    expect(quiet).toContain('**Symptoms on this node:**\nNone\n')
    expect(quiet).toContain('**Latest attempts it is responsible in:**\nNone\n')
    expect(quiet).not.toMatch(/\(none/)
    // Its counts carry the node page's names, over every recorded run.
    expect(nodeMd).toContain('**Responsible in:** 1 attempt\n**Distinct symptoms:** 1\n')
    expect(quiet).toContain('**Responsible in:** 0 attempts')
    expect(`${nodeMd}${quiet}`).not.toMatch(/window|impacted|recurring/)
  })

  it('includes tenant evidence and ignores legacy operator guidance', () => {
    const run = {
      ...base,
      diagnose: {
        hasPersona: true,
        title: 'NVLink fault',
        summary: 'cascade',
        impact: 'rank exits',
        action: 'Resubmit on healthy allocation',
        supportDetails: ['Include the NVIDIA Attribution diagnosis (verdict, symptom chain); send available logs.'],
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
    expect(md).toContain('**Summary:** cascade')
    expect(md).not.toContain('**Summary:** GPU off bus')
    expect(md).toContain('## Tenant SRE guidance')
    expect(md).toContain('**Recommended action:**\n\nResubmit on healthy allocation')
    expect(md).toContain('**Include on escalation:**\n\nInclude the NVIDIA Attribution diagnosis (verdict, symptom chain); send available logs.')
    for (const privateText of ['Cloud operator guidance', 'fabric issue', 'node needs drain', 'Drain node', 'dmesg', 'fabricmanager.log']) {
      expect(md).not.toContain(privateText)
    }
  })

  it('preserves numbered actions and support instructions as Markdown blocks', () => {
    const action = '1. Check your workload.\n2. Retry only if safe.\n3. Confirm the failed operation completes.'
    const evidence = 'Include the diagnosis (verdict, symptom chain), job ID and logs; ask for help even with incomplete evidence.'
    const md = buildVerdictMarkdown({
      ...base,
      diagnose: { hasPersona: true, title: 'Test', summary: '', impact: '', action, supportDetails: [evidence, 'Record the retry result.'] },
    }, 'https://x/')
    expect(md).toContain(`**Recommended action:**\n\n${action}\n`)
    expect(md).toContain(`**Include on escalation:**\n\n${evidence}\n\nRecord the retry result.`)
    expect(md).toContain('**Summary:** GPU off bus')
  })

  it.each([undefined, { hasPersona: false, title: '', summary: 'Do not display', impact: '', action: 'Do not display', supportDetails: ['Do not display'] }])(
    'does not fall back to operator guidance when tenant guidance is unavailable', (diagnose) => {
      const md = buildVerdictMarkdown({
        ...base,
        diagnose,
        operatorDiagnose: { hasPersona: true, title: 'Private', summary: 'PRIVATE_SUMMARY', impact: 'PRIVATE_IMPACT', action: 'PRIVATE_ACTION', supportDetails: ['PRIVATE_EVIDENCE'] },
      }, 'https://x/')
      expect(md).toContain('**Summary:** GPU off bus')
      expect(md).not.toContain('PRIVATE_')
      expect(md).not.toContain('Do not display')
      expect(md).not.toContain('## Tenant SRE guidance')
      expect(md).not.toContain('## Cloud operator guidance')
    },
  )

  it('omits empty tenant guidance fields without importing operator evidence', () => {
    const md = buildVerdictMarkdown({
      ...base,
      diagnose: { hasPersona: true, title: 'Test', summary: '', impact: '', action: '', supportDetails: [] },
      operatorDiagnose: { hasPersona: true, title: 'Private', summary: 'PRIVATE_SUMMARY', impact: '', action: '', supportDetails: ['PRIVATE_EVIDENCE'] },
    }, 'https://x/')
    expect(md).not.toContain('**Recommended action:**')
    expect(md).not.toContain('**Include on escalation:**')
    expect(md).not.toContain('PRIVATE_')
  })

  it('falls back to raw fields without diagnose', () => {
    const md = buildVerdictMarkdown(base, 'https://x/')
    expect(md).toContain('Hardware')
    expect(md).toContain('GpuFell')
  })

  it('writes every time as the UI shows it: the full UTC stamp with its mark', () => {
    const md = buildVerdictMarkdown({ ...base, rootSymptom: { ...base.rootSymptom, time: '2026-05-01T00:30:01+00:00' } }, 'https://x/')
    expect(md).toContain('**Window:** 2026-05-01 00:00:00 UTC → 2026-05-01 01:00:00 UTC')
    expect(md).toContain('**Matched at:** 2026-05-01 00:30:01 UTC')
    expect(md).not.toMatch(/\dT\d|\+00:00|Z\b|\(UTC\)/)
    // A time recorded without a zone is UTC, as the server reads it, never the
    // reader's zone: read here in UTC+8, where a local reading would differ.
    vi.stubEnv('TZ', 'Asia/Shanghai')
    try {
      expect(new Date(2026, 4, 1).getTimezoneOffset()).toBe(-8 * 60)
      expect(utcStamp('2026-05-01T00:30:01')).toBe('2026-05-01 00:30:01 UTC')
      expect(utcStamp('2026-05-01T08:30:01+08:00')).toBe('2026-05-01 00:30:01 UTC')
    } finally {
      vi.unstubAllEnvs()
    }
    // A time that does not parse is unavailable, as the UI says.
    expect(buildVerdictMarkdown(base, 'https://x/')).toContain('**Matched at:** Unavailable')
    expect(buildNodeMarkdown('n1', [base], 'https://x/')).toContain('- 2026-05-01 01:00:00 UTC · GpuFell · https://x/verdict/1')
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
