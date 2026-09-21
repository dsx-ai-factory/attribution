import { useState } from 'react'
import { useParams } from 'react-router-dom'
import useSWR from 'swr'
import { ChevronRight } from 'lucide-react'
import type { RunDetail } from '@/interfaces/attribution'
import { useApi } from '@/services/api-context'
import { buildVerdictMarkdown } from '@/lib/evidence'
import {
  downloadBlob,
  logToText,
  slug,
  toYaml,
  utcDateStamp,
  verdictExportData,
} from '@/lib/export'
import { ExportMenu } from '@/components/export-menu'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import {
  Table,
  TableBody,
  TableCell,
  TableRow,
} from '@/components/ui/table'
import { VerdictPill } from '@/components/chip'
import { DefHover } from '@/components/def-hover'
import { NodeBubble } from '@/routers/verdict/components/node-bubble'
import { NodeHoverCard } from '@/components/node-hover-card'
import { TimeHover } from '@/components/time-hover'
import { attemptGpuHours } from '@/lib/metric'
import { useNav } from '@/hooks/use-nav'
import { LogHero } from '@/routers/verdict/components/log-hero'
import { DecisionGraphAction } from '@/routers/verdict/components/decision-graph/dialog'

const SECTION_LABEL =
  'font-mono text-xs font-semibold tracking-widest uppercase text-foreground-muted my-5 mb-2.5 flex items-center gap-2.5'

function Actions({ run }: { run: RunDetail }) {
  const nav = useNav()
  const [copied, setCopied] = useState('')
  const flash = (label: string) => {
    setCopied(label)
    window.setTimeout(() => setCopied(''), 1200)
  }
  const writeClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      // Fallback for non-secure contexts
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      try { document.execCommand('copy') } catch { /* swallow */ }
      document.body.removeChild(ta)
    }
  }
  const copyEvidence = async () => {
    await writeClipboard(buildVerdictMarkdown(run))
    flash('evidence')
  }
  const copyLocalLink = async () => {
    await writeClipboard(window.location.href)
    flash('link')
  }

  const baseName = `verdict-${slug(run.jobId)}-att${run.attempt}-${utcDateStamp()}`
  const exportItems = [
    {
      label: 'Verdict + evidence (JSON)',
      hint: 'machine-parseable, schema-stable',
      onSelect: () =>
        downloadBlob(
          `${baseName}.json`,
          JSON.stringify(verdictExportData(run), null, 2),
          'application/json',
        ),
    },
    {
      label: 'Verdict + evidence (YAML)',
      hint: 'human-readable, drop into a ticket',
      onSelect: () =>
        downloadBlob(`${baseName}.yaml`, toYaml(verdictExportData(run)), 'application/yaml'),
    },
    {
      label: 'Raw log (text)',
      hint: 'attempt log lines, no scaffolding',
      onSelect: () =>
        downloadBlob(`${baseName}.log`, logToText(run), 'text/plain'),
    },
    {
      label: 'Evidence (markdown)',
      hint: 'same as the Copy button',
      onSelect: () =>
        downloadBlob(`${baseName}.md`, buildVerdictMarkdown(run), 'text/markdown'),
    },
  ]

  return (
    <>
      <Button variant="outline" size="sm" onClick={copyEvidence} className="min-w-[12rem] justify-center">
        {copied === 'evidence' ? 'Copied evidence ✓' : 'Copy evidence (markdown)'}
      </Button>
      <ExportMenu items={exportItems} triggerLabel="Download" />
      <HoverCard openDelay={200} closeDelay={75}>
        <HoverCardTrigger asChild>
          <Button variant="outline" size="sm" onClick={copyLocalLink} className="min-w-[8.5rem] justify-center">
            {copied === 'link' ? 'Copied link ✓' : 'Copy local link'}
          </Button>
        </HoverCardTrigger>
        <HoverCardContent
          align="start"
          sideOffset={6}
          className="w-72 p-3 bg-background-secondary border-border ring-0 font-mono text-[11px] leading-snug"
        >
          Only resolves on this Attribution instance. For cross-team sharing, prefer
          Download (JSON / YAML / log).
        </HoverCardContent>
      </HoverCard>
      {run.responsibleNodes.length > 0 && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => nav('node', { id: run.responsibleNodes[0] })}
        >
          Other runs on {run.responsibleNodes[0]}
        </Button>
      )}
    </>
  )
}

function CardHead({ title, sub }: { title: string; sub?: React.ReactNode }) {
  return (
    <CardHeader className="flex flex-row items-baseline gap-3">
      <CardTitle className="text-xs font-semibold font-mono tracking-wide uppercase">
        {title}
      </CardTitle>
      {sub && <span className="text-foreground-muted text-xs">{sub}</span>}
    </CardHeader>
  )
}

function KvCell({
  label,
  children,
  className,
  valueClass = 'font-mono text-xs text-foreground',
}: {
  label: React.ReactNode
  children: React.ReactNode
  className?: string
  valueClass?: string
}) {
  return (
    <div className={`flex flex-col gap-1.5 min-w-0 ${className ?? ''}`}>
      <div className="font-mono text-xs text-foreground-muted tracking-wide uppercase">
        {label}
      </div>
      <div className={`${valueClass} min-w-0`}>{children}</div>
    </div>
  )
}

function GuidanceLine({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[9rem_1fr] gap-3 text-xs">
      <div className="font-mono uppercase tracking-wide text-foreground-muted">{label}</div>
      <div className="text-foreground-secondary leading-snug">{children}</div>
    </div>
  )
}

function CloudOperatorGuidance({ run }: { run: RunDetail }) {
  const card = run.operatorDiagnose
  if (card?.hasPersona !== true) return null

  return (
    <div className="mt-4 border-t border-border pt-4 flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs font-semibold uppercase tracking-wide text-foreground-muted">
          Cloud operator guidance
        </span>
        {card.catalogId && (
          <Badge variant="outline" className="font-mono text-[10px] text-foreground-muted border-border">
            {card.catalogId}
          </Badge>
        )}
      </div>
      {card.summary !== '' && <GuidanceLine label="Summary">{card.summary}</GuidanceLine>}
      {card.impact !== '' && <GuidanceLine label="Impact">{card.impact}</GuidanceLine>}
      {card.action !== '' && <GuidanceLine label="Action">{card.action}</GuidanceLine>}
      {card.supportDetails.length > 0 && (
        <GuidanceLine label="Include">{card.supportDetails.join(', ')}</GuidanceLine>
      )}
    </div>
  )
}

function EvidenceList({ run }: { run: RunDetail }) {
  return (
    <Card size="sm">
      <CardHead title="Verdict" />
      <CardContent>
        <div className="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-2 gap-x-5 gap-y-4">
          <KvCell label={<DefHover term="attribution-outcome">attribution_outcome</DefHover>}>
            <VerdictPill outcome={run.outcome} label={run.verdictLabel} />
          </KvCell>
          <KvCell label={<DefHover term="root-symptom-symptom">root_symptom.symptom</DefHover>}>
            <b className="font-semibold">{run.rootSymptom.symptom}</b>
          </KvCell>
          <KvCell label={<DefHover term="rule">root_symptom.rule</DefHover>}>
            <Badge variant="outline" className="font-mono text-primary border-border">
              {run.rootSymptom.rule}
            </Badge>
          </KvCell>
          <KvCell label="root_symptom.time">
            <TimeHover date={run.rootSymptom.time} format="YYYY-MM-DD HH:mm:ss" />
          </KvCell>
          <KvCell
            label="detail"
            className="col-span-2 lg:col-span-3 xl:col-span-2"
            valueClass="text-sm text-foreground leading-snug"
          >
            {run.detail}
          </KvCell>
          {run.errorPairs !== undefined && run.errorPairs !== '' && (
            <KvCell
              label="error_pairs"
              className="col-span-2 lg:col-span-4 xl:col-span-2"
              valueClass="font-mono text-xs text-foreground-secondary whitespace-pre-wrap break-words"
            >
              <ul aria-label="error pairs" className="m-0 list-none p-0">
                {run.errorPairs.split('; ').map((pair) => (
                  <li key={pair}>{pair}</li>
                ))}
              </ul>
            </KvCell>
          )}
          <KvCell label={<DefHover term="attempt-uid">attempt_uid</DefHover>} valueClass="font-mono text-xs text-foreground-secondary truncate">
            {run.attemptUid}
          </KvCell>
        </div>
        <CloudOperatorGuidance run={run} />
        <div className="mt-4"><DecisionGraphAction run={run} detail={run} /></div>
      </CardContent>
      <CardFooter className="border-t flex-wrap gap-2 bg-background-secondary p-3">
        <Actions run={run} />
      </CardFooter>
    </Card>
  )
}

function SymptomsCard({ run }: { run: RunDetail }) {
  return (
    <Card size="sm">
      <CardHead title="Symptoms" sub={`${run.symptoms.length} distinct`} />
      <CardContent>
        <div className="flex flex-wrap gap-1.5 py-1">
          {run.symptoms.map((s, i) => (
            <Badge
              key={i}
              variant="outline"
              className="font-mono text-foreground border-border gap-2"
            >
              <span className="font-semibold">{s.name}</span>
              <span className="text-foreground-muted">{s.sites}</span>
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function NodesCard({ run }: { run: RunDetail }) {
  const nav = useNav()
  return (
    <Card size="sm">
      <CardHead title="GPUs" sub={`${run.gpuCount} this attempt`} />
      <CardContent>
        <div className="flex flex-wrap gap-1.5 py-1">
          {run.nodeIds.map((n) => (
            <NodeHoverCard key={n} nodeId={n}>
              <NodeBubble
                node={n}
                responsible={run.responsibleNodes.includes(n)}
                onClick={() => nav('node', { id: n })}
              />
            </NodeHoverCard>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function ResponsibleNodesCard({ run }: { run: RunDetail }) {
  const nav = useNav()
  if (run.responsibleNodes.length === 0) return null

  return (
    <Card size="sm">
      <CardHead title="Responsible nodes" sub={`${run.responsibleNodes.length} listed`} />
      <CardContent>
        <ScrollArea className="max-h-72 w-full">
          <Table>
            <TableBody>
              {run.responsibleNodes.map((node) => (
                <TableRow key={node}>
                  <TableCell className="font-mono text-xs">
                    <NodeHoverCard nodeId={node}>
                      <a
                        className="text-foreground hover:text-primary cursor-pointer"
                        onClick={() => nav('node', { id: node })}
                      >
                        {node}
                      </a>
                    </NodeHoverCard>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <ScrollBar orientation="vertical" />
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

function RawPayload({ label, payload }: { label: string; payload: object }) {
  const json = JSON.stringify(payload, null, 2)
  const [copied, setCopied] = useState(false)
  const copy = async (e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await navigator.clipboard.writeText(json)
    } catch {
      const ta = document.createElement('textarea')
      ta.value = json
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      try { document.execCommand('copy') } catch { /* swallow */ }
      document.body.removeChild(ta)
    }
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1200)
  }
  return (
    <Collapsible className="border border-border rounded-sm bg-card group/raw overflow-hidden">
      <div className="flex items-center w-full">
        <CollapsibleTrigger className="flex-1 text-left flex items-center gap-1 px-3 py-2 cursor-pointer font-mono text-xs text-foreground-secondary select-none">
          <ChevronRight className="size-3 transition-transform group-data-[state=open]/raw:rotate-90 text-foreground-muted" />
          {label}
        </CollapsibleTrigger>
        <button
          type="button"
          onClick={copy}
          aria-label={`Copy ${label} as JSON`}
          className="opacity-0 group-hover/raw:opacity-100 focus-visible:opacity-100 transition-opacity mr-2 px-2 py-1 rounded-sm font-mono text-[10px] uppercase tracking-wider text-foreground-muted hover:text-foreground hover:bg-background-active cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
        >
          {copied ? 'Copied ✓' : 'Copy JSON'}
        </button>
      </div>
      <CollapsibleContent className="border-t border-border overflow-hidden">
        <ScrollArea className="h-80 w-full">
          <pre className="m-0 px-3.5 py-2.5 font-mono text-xs text-foreground-secondary leading-normal w-max">
            {json}
          </pre>
          <ScrollBar orientation="horizontal" />
          <ScrollBar orientation="vertical" />
        </ScrollArea>
      </CollapsibleContent>
    </Collapsible>
  )
}

function RawPayloads({ run }: { run: RunDetail }) {
  return (
    <div>
      <div className={SECTION_LABEL}>
        Raw payloads
        <span className="flex-1 h-px bg-border" />
      </div>
      <div className="flex flex-col gap-1.5">
        <RawPayload
          label="raw_result_json"
          payload={{
            attempt_uid: run.attemptUid,
            outcome: run.outcome,
            root_symptom: run.rootSymptom,
            symptoms: run.symptoms,
            nodes: run.nodeIds,
            responsible_nodes: run.responsibleNodes,
            elapsed_hours: run.elapsedH,
            pipeline: run.pipeline,
            service: run.service,
          }}
        />
        <RawPayload
          label="raw_scheduler_json"
          payload={{
            execution_context: run.jobId,
            attempt: run.attempt,
            user: run.user,
            status: run.status,
            nodes: run.nodeIds,
            start_time: run.startTime,
            end_time: run.endTime,
          }}
        />
      </div>
    </div>
  )
}

export function VerdictView() {
  const { id: verdictId = '' } = useParams<{ id: string }>()
  const nav = useNav()
  const api = useApi()
  const { data: fetchedRun, isLoading } = useSWR<RunDetail | undefined, Error>(
    verdictId !== '' ? ['run', verdictId] : null,
    () => api.getRun(verdictId),
  )
  const run = fetchedRun
  if (!run) {
    return (
      <div className="p-6 text-foreground-muted">
        {isLoading ? 'Loading execution context...' : 'Execution context not found'}
      </div>
    )
  }

  const hasWorkload = run.workload && run.workload !== '—'

  return (
    <div>
      <Breadcrumb className="mb-3.5 font-mono">
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink
              onClick={() => nav('fleet')}
              className="text-primary cursor-pointer"
            >
              Attribution
            </BreadcrumbLink>
          </BreadcrumbItem>
          {hasWorkload && (
            <>
              <BreadcrumbSeparator />
              <BreadcrumbItem>{run.workload}</BreadcrumbItem>
            </>
          )}
          <BreadcrumbSeparator />
          <BreadcrumbItem>
            <BreadcrumbPage className="inline-block text-foreground font-medium">
              {run.jobId}
              {hasWorkload ? ` · attempt ${run.attempt}` : ''}
            </BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <div className="flex items-start gap-4 my-1.5 mb-2">
        <h1 className="min-w-0 break-words text-3xl font-bold m-0 tracking-tight">
          {hasWorkload ? (
            <>
              Attempt <span className="font-mono text-foreground">{run.attempt}</span> ·{' '}
              <span className="text-foreground-muted text-lg font-normal">{run.workload}</span>
            </>
          ) : (
            <>
              Execution Context <span className="font-mono text-foreground">{run.jobId}</span>
            </>
          )}
        </h1>
      </div>
      <div className="flex flex-wrap gap-3.5 font-mono text-xs text-foreground-muted mb-1">
        <span>
          User <b className="text-foreground-secondary font-medium">{run.user}</b>
        </span>
        <span>
          Status <b className="text-foreground-secondary font-medium">{run.status}</b>
        </span>
        <span>
          GPUs <b className="text-foreground-secondary font-medium">{run.gpuCount}</b>
        </span>
        <span>
          <DefHover term="gpu-hours">GPU-hours</DefHover>{' '}
          <b className="text-foreground-secondary font-medium">{attemptGpuHours(run).toFixed(2)}</b>
        </span>
        <span>
          Window{' '}
          <b className="text-foreground-secondary font-medium">
            <TimeHover date={run.startTime} format="MM-DD HH:mm" />
          </b>{' '}
          →{' '}
          <b className="text-foreground-secondary font-medium">
            <TimeHover date={run.endTime} format="MM-DD HH:mm" />
          </b>
        </span>
        <span>
          Pipeline <b className="text-foreground-secondary font-medium">{run.pipeline}</b>
        </span>
        <span>
          Service <b className="text-foreground-secondary font-medium">{run.service}</b>
        </span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 mt-3 mb-5 xl:items-start">
        <div className="xl:col-span-8 min-w-0">
          <LogHero run={run} />
        </div>
        <div className="xl:col-span-4 xl:sticky xl:top-16 self-start min-w-0">
          <EvidenceList run={run} />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-5">
        <SymptomsCard run={run} />
        <NodesCard run={run} />
      </div>
      {run.responsibleNodes.length > 0 && (
        <div className="mb-5">
          <ResponsibleNodesCard run={run} />
        </div>
      )}

      <RawPayloads run={run} />
    </div>
  )
}
