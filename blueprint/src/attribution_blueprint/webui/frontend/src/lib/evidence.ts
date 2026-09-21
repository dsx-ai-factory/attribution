import type { OutcomeKey, PersonaDiagnose, RunSummary } from '@/interfaces/attribution'

const OUTCOME_HEADER: Record<OutcomeKey, string> = {
  hardware: 'HARDWARE',
  software: 'SOFTWARE',
  multi:    'MULTICATEGORY',
  user:     'USER',
  undetermined: 'UNDETERMINED',
  pass:     'PASS',
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
  const nodeUrls = run.responsibleNodes
    .map((n) => `- Node detail: ${absoluteUrl(`/node/${encodeURIComponent(n)}`, base)}`)
    .join('\n')
  const responsibleLines = run.responsibleNodes.length
    ? run.responsibleNodes.map((n) => `- \`${n}\``).join('\n')
    : '- (none listed)'

  const personaLines: string[] = []
  appendGuidance(personaLines, 'Tenant SRE guidance', run.diagnose, false)
  appendGuidance(personaLines, 'Cloud operator guidance', run.operatorDiagnose, true)
  const operatorSummary = run.operatorDiagnose?.hasPersona === true ? run.operatorDiagnose.summary : ''
  const verdictSummary = operatorSummary !== '' ? operatorSummary : run.detail

  return [
    `*Attribution escalation: ${OUTCOME_HEADER[run.outcome]} · ${run.rootSymptom.symptom}*`,
    ``,
    `**Decision requested:** retry workload | operator investigation`,
    `**Verdict:** ${run.verdictLabel}`,
    `**Summary:** ${verdictSummary}`,
    ``,
    `**Execution context:** ${run.workload && run.workload !== '—' ? run.workload : run.jobId}`,
    `**Attempt:** #${run.attempt}`,
    `**Attempt UID:** \`${run.attemptUid}\``,
    `**User / tenant:** ${run.user}`,
    `**Window (UTC):** ${run.startTime} → ${run.endTime}`,
    `**Elapsed:** ${run.elapsedH.toFixed(2)}h`,
    ``,
    `**Root symptom:** \`${run.rootSymptom.symptom}\``,
    `**Rule:** \`${run.rootSymptom.rule}\``,
    `**Matched at:** ${run.rootSymptom.time} UTC`,
    ``,
    `**Responsible node(s):**`,
    responsibleLines,
    ``,
    `**Matched log:**`,
    '```text',
    run.rootSymptom.line,
    '```',
    ...personaLines,
    ``,
    `**Links:**`,
    `- Verdict: ${verdictUrl}`,
    nodeUrls || '- Node detail: n/a',
    `- Pipeline / service: ${run.pipeline} / ${run.service}`,
  ].join('\n')
}

export function buildNodeMarkdown(
  nodeId: string,
  attempts: RunSummary[],
  base?: string,
): string {
  const impactedRuns = attempts.filter((r) => r.responsibleNodes.includes(nodeId))
  const symptomCounts: Record<string, number> = {}
  impactedRuns.forEach((r) => {
    const s = r.rootSymptom.symptom
    symptomCounts[s] = (symptomCounts[s] || 0) + 1
  })
  const recurring = Object.entries(symptomCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([s, n]) => `- \`${s}\` × ${n}`)
    .join('\n')

  const nodeUrl = absoluteUrl(`/node/${encodeURIComponent(nodeId)}`, base)
  const recent = impactedRuns
    .slice(0, 5)
    .map((r) => `- ${r.endTime} · ${r.rootSymptom.symptom} · ${absoluteUrl(`/verdict/${encodeURIComponent(r.id)}`, base)}`)
    .join('\n')

  return [
    `*Attribution node escalation: \`${nodeId}\`*`,
    ``,
    `**Decision requested:** operator investigation`,
    `**Attempts impacted (in window):** ${impactedRuns.length}`,
    `**Distinct recurring symptoms:** ${Object.keys(symptomCounts).length}`,
    ``,
    `**Symptoms attributed:**`,
    recurring || '- (none)',
    ``,
    `**Recent impacted attempts:**`,
    recent || '- (none)',
    ``,
    `**Links:**`,
    `- Node detail: ${nodeUrl}`,
  ].join('\n')
}
