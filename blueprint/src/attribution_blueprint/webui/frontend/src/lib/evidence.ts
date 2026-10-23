// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PersonaDiagnose, RunSummary } from '@/interfaces/attribution'
import { OUTCOMES } from '@/lib/attribution-data'
import { counted } from '@/lib/format'
import { summarizeNode } from '@/lib/node-summary'
import { utcStamp } from '@/lib/time'

// A recorded value as the page gives it: in code, or "Not recorded" where the
// record has none, as the page says, never an empty pair of backticks.
function recorded(value: string): string {
  return value !== '' ? `\`${value}\`` : 'Not recorded'
}

// A recorded name the page gives as text, or "Not recorded".
function named(value: string): string {
  return value !== '' ? value : 'Not recorded'
}

function absoluteUrl(path: string, base?: string): string {
  // `path` is router-relative (e.g. "/verdict/123"); resolve it against the deploy origin
  // so links in escalation markdown are absolute and shareable.
  if (base) return new URL(path.replace(/^\//, ''), base).toString()
  if (typeof window !== 'undefined') {
    return new URL(path.replace(/^\//, ''), window.location.origin + '/').toString()
  }
  return path
}

function appendGuidance(
  lines: string[],
  heading: string,
  card: PersonaDiagnose | undefined,
  includeSupport: boolean,
): void {
  if (card?.hasPersona !== true) return

  lines.push(``, `## ${heading}`)
  if (card.summary !== '') {
    lines.push(`**What happened:** ${card.summary}`)
  }
  if (card.impact !== '') {
    lines.push(`**Impact:** ${card.impact}`)
  }
  if (card.action !== '') {
    lines.push(`**Recommended action:** ${card.action}`)
  }
  if (includeSupport && card.supportDetails.length > 0) {
    lines.push(`**Include on escalation:** ${card.supportDetails.join(', ')}`)
  }
}

export function buildVerdictMarkdown(run: RunSummary, base?: string): string {
  const verdictUrl = absoluteUrl(`/verdict/${encodeURIComponent(run.id)}`, base)
  const nodeUrls = run.responsibleNodes.map((n) => `- Node detail: ${absoluteUrl(`/node/${encodeURIComponent(n)}`, base)}`)
  const responsibleLines = run.responsibleNodes.length
    ? run.responsibleNodes.map((n) => `- \`${n}\``).join('\n')
    : 'None'

  const personaLines: string[] = []
  appendGuidance(personaLines, 'Tenant SRE guidance', run.diagnose, false)
  appendGuidance(personaLines, 'Cloud operator guidance', run.operatorDiagnose, true)
  const operatorSummary = run.operatorDiagnose?.hasPersona === true ? run.operatorDiagnose.summary : ''
  const verdictSummary = operatorSummary !== '' ? operatorSummary : run.detail

  // The verdict by the name the page gives it (the verdict badge's), and every
  // time in the page's stamp. An empty value reads "Not recorded", as on the
  // page; an empty list "None", which is how the page's empty value (a muted
  // dash) is read, as the page's own lists show it only in a count or an
  // empty message; the summary, as the page's service detail, is left out, as
  // is a node link when no node is responsible.
  const verdict = run.verdictLabel || OUTCOMES[run.outcome].label
  return [
    `*Attribution escalation: ${verdict} · ${run.rootSymptom.symptom || 'Root symptom not recorded'}*`,
    ``,
    `**Decision requested:** retry workload | operator investigation`,
    `**Verdict:** ${verdict}`,
    ...(verdictSummary !== '' ? [`**Summary:** ${verdictSummary}`] : []),
    ``,
    `**Execution context:** ${run.workload && run.workload !== '—' ? run.workload : run.jobId}`,
    `**Attempt:** #${run.attempt}`,
    `**Attempt UID:** \`${run.attemptUid}\``,
    `**User / tenant:** ${named(run.user)}`,
    `**Window:** ${utcStamp(run.startTime)} → ${utcStamp(run.endTime)}`,
    `**Elapsed:** ${run.elapsedH.toFixed(2)}h`,
    ``,
    `**Root symptom:** ${recorded(run.rootSymptom.symptom)}`,
    `**Rule:** ${recorded(run.rootSymptom.rule)}`,
    `**Matched at:** ${utcStamp(run.rootSymptom.time)}`,
    ``,
    `**Responsible node(s):**`,
    responsibleLines,
    ``,
    `**Matched log:**`,
    ...(run.rootSymptom.line !== '' ? ['```text', run.rootSymptom.line, '```'] : ['Not recorded']),
    ...personaLines,
    ``,
    `**Links:**`,
    `- Verdict: ${verdictUrl}`,
    ...nodeUrls,
    `- Pipeline / service: ${named(run.pipeline)} / ${named(run.service)}`,
  ].join('\n')
}

export function buildNodeMarkdown(
  nodeId: string,
  attempts: RunSummary[],
  base?: string,
): string {
  // The node's page's own summary of its runs (summarizeNode), in its words:
  // the attempts it is responsible in and its distinct symptoms, across every
  // recorded run (the page has no window).
  const { responsible, symptoms } = summarizeNode(attempts, nodeId)
  const recurring = symptoms.map(([symptom, count]) => `- ${recorded(symptom)} × ${count}`).join('\n')

  const nodeUrl = absoluteUrl(`/node/${encodeURIComponent(nodeId)}`, base)
  const recent = responsible
    .slice(0, 5)
    .map((r) => `- ${utcStamp(r.endTime)} · ${r.rootSymptom.symptom || 'Root symptom not recorded'} · ${absoluteUrl(`/verdict/${encodeURIComponent(r.id)}`, base)}`)
    .join('\n')

  return [
    `*Attribution node escalation: \`${nodeId}\`*`,
    ``,
    `**Decision requested:** operator investigation`,
    `**Responsible in:** ${counted(responsible.length, 'attempt', 'attempts')}`,
    `**Distinct symptoms:** ${symptoms.length}`,
    ``,
    `**Symptoms on this node:**`,
    recurring || 'None',
    ``,
    `**Latest attempts it is responsible in:**`,
    recent || 'None',
    ``,
    `**Links:**`,
    `- Node detail: ${nodeUrl}`,
  ].join('\n')
}
