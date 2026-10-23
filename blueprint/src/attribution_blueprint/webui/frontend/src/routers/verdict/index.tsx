// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId } from 'react'
import { useParams } from 'react-router-dom'
import useSWR from 'swr'
import { ArrowRight, ChevronRight, Link2, Server } from 'lucide-react'
import type { PersonaDiagnose, RunDetail, RunSummary } from '@/interfaces/attribution'
import { useApi } from '@/services/api-context'
import { useDocumentTitle } from '@/hooks/use-document-title'
import { navigateService } from '@/services/navigate.service'
import { buildVerdictMarkdown } from '@/lib/evidence'
import {
  downloadBlob,
  logToText,
  slug,
  toYaml,
  utcDateStamp,
  verdictExportData,
} from '@/lib/export'
import { attemptGpuHours } from '@/lib/metric'
import { cn, focusRing, markInText, subtitle } from '@/lib/utils'
import { CopyButton } from '@/components/copy-button'
import { ExportMenu } from '@/components/export-menu'
import { Identifier, NotRecorded } from '@/components/field'
import { InfoHint } from '@/components/info-hint'
import { LogExcerpt } from '@/components/log-excerpt'
import { NodeHoverCard } from '@/components/node-hover-card'
import { CompoundOutcomeBadge } from '@/components/outcome'
import { PageBreadcrumb, PageHeader, type PageCrumb } from '@/components/page-header'
import { SectionCard } from '@/components/section-card'
import { SiteStatusBadge } from '@/components/site-status-badge'
import { TimeHover } from '@/components/time-hover'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardAction, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  CodeBlock,
  CodeBlockActions,
  CodeBlockBody,
  CodeBlockCopyButton,
  CodeBlockHeader,
} from '@/components/ui/code-block'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Copyable } from '@/components/ui/copyable'
import { DescriptionList, type DescriptionListItem } from '@/components/ui/description-list'
import { Empty, EmptyHeader, EmptyTitle } from '@/components/ui/empty'
import { RouterLink } from '@/components/ui/router-link'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { DecisionGraphAction } from '@/routers/verdict/components/decision-graph/dialog'
import { SymptomsSection } from '@/routers/verdict/components/symptom-sites'

const TERMINAL_STATE_LABELS: Record<string, string> = {
  Pass: 'Pass',
  Fail: 'Fail',
  ExpectedFail: 'Expected failure',
  Cancelled: 'Cancelled',
}

const FAILURE_DOMAIN_LABELS: Record<string, string> = {
  User: 'Application',
  InfraHW: 'Infrastructure (hardware)',
  InfraSW: 'Infrastructure (software)',
}

const CANCELLATION_ACTOR_LABELS: Record<string, string> = {
  User: 'Workload owner',
  Operator: 'Operator',
}

const UNCERTAINTY_LABELS: Record<string, string> = {
  Aleatoric: 'Aleatoric (data ambiguity)',
  Epistemic: 'Epistemic (rule gap)',
}

const EVIDENCE_STATUS_LABELS: Record<string, { label: string; tooltip: string }> = {
  complete: { label: 'Complete', tooltip: 'All collected evidence was delivered to the attribution service' },
  partial: { label: 'Partial', tooltip: 'A collector failed or some evidence was not delivered' },
  none: { label: 'None', tooltip: 'No evidence reached the attribution service' },
}

// Focusable so the tooltip opens from the keyboard; the dotted underline marks it as explained.
function HelpTooltip({ text, children }: { text: string; children: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} className="cursor-help underline decoration-dotted underline-offset-4">{children}</span>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={6}>{text}</TooltipContent>
    </Tooltip>
  )
}

function evidenceStatusValue(status: string) {
  const entry = EVIDENCE_STATUS_LABELS[status]
  return entry ? <HelpTooltip text={entry.tooltip}>{entry.label}</HelpTooltip> : status
}

// The scheduler's status enum, read as words.
const RUN_STATUS_LABELS: Record<RunSummary['status'], string> = {
  FAILED: 'Failed',
  COMPLETED: 'Completed',
}

// The page's place in the app, shown whether or not its record loads.
const VERDICT_CRUMBS: PageCrumb[] = [{ label: 'Fleet', to: navigateService.getPath('fleet') }, { label: 'Verdict' }]

// The answer and its lines share a row from 1024px, both as tall as the
// taller: the lines take the first two of three equal columns, the answer the
// last. That third is never narrower than what in the answer card cannot
// wrap: its header on one line (the outcome beside the action), so the header
// never wraps, and a fact that holds a timestamp, so the stamp never runs out
// of its cell. Where either needs more than a third, near 1024px, the width
// comes out of the lines. Stacked, the answer comes first, as it does in
// reading order.
const ANSWER_ROW = 'grid grid-cols-1 gap-card-gap lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(min-content,1fr)]'

// Page-wide actions: the evidence as Markdown, a link to this page, and downloads.
function VerdictActions({ run }: { run: RunDetail }) {
  // Named when the download starts, so the stamp is the moment of export.
  const fileName = (extension: string) =>
    `verdict-${slug(run.jobId)}-att${run.attempt}-${utcDateStamp()}.${extension}`
  const exportItems = [
    {
      label: 'Download as JSON',
      onSelect: () =>
        downloadBlob(fileName('json'), JSON.stringify(verdictExportData(run), null, 2), 'application/json'),
    },
    {
      label: 'Download as YAML',
      onSelect: () => downloadBlob(fileName('yaml'), toYaml(verdictExportData(run)), 'application/yaml'),
    },
    {
      label: 'Download as log',
      onSelect: () => downloadBlob(fileName('log'), logToText(run), 'text/plain'),
    },
    {
      label: 'Download as Markdown',
      onSelect: () => downloadBlob(fileName('md'), buildVerdictMarkdown(run), 'text/markdown'),
    },
  ]

  return (
    <>
      <CopyButton
        label="Copy evidence"
        getText={() => buildVerdictMarkdown(run)}
        tooltip="Copies a Markdown summary"
      />
      <CopyButton
        icon={Link2}
        label="Copy link"
        getText={() => window.location.href}
        tooltip="Only opens on this Attribution instance"
      />
      <ExportMenu items={exportItems} triggerLabel="Export" />
    </>
  )
}

// The page's facts, labelled whether or not the record has loaded; while it
// loads (no run yet) the head holds each value's place.
function verdictMeta(run: RunDetail | undefined): DescriptionListItem[] {
  return [
    { key: 'user', label: 'User', value: run && (run.user ? <Identifier>{run.user}</Identifier> : <NotRecorded />) },
    { key: 'status', label: 'Status', value: run && RUN_STATUS_LABELS[run.status] },
    { key: 'gpus', label: 'GPUs', value: run && <span className="num">{run.gpuCount}</span> },
    {
      key: 'gpu-hours',
      label: <>GPU-hours <InfoHint term="gpu-hours" /></>,
      value: run && <span className="num">{attemptGpuHours(run).toFixed(2)}</span>,
    },
    {
      key: 'window',
      label: 'Window',
      value: run && (
        <span className="inline-flex flex-wrap items-center gap-x-1.5">
          <TimeHover date={run.startTime} />
          <ArrowRight aria-hidden="true" className="size-3.5 text-foreground-muted" />
          <span className="sr-only">to</span>
          <TimeHover date={run.endTime} />
        </span>
      ),
    },
  ]
}

// The facts behind the answer, one per row on one label axis, in reading
// order: what the service said, the rule and moment that matched, then where
// the record comes from. A row the record leaves empty says so, except the
// service detail and error pairs, which only appear when recorded.
function verdictFacts(run: RunDetail): DescriptionListItem[] {
  const errorPairs = run.errorPairs ?? ''
  return [
    ...(run.detail !== '' ? [{ key: 'detail', label: 'Service detail', value: run.detail }] : []),
    { key: 'rule', label: 'Rule', value: run.rootSymptom.rule ? <Identifier>{run.rootSymptom.rule}</Identifier> : <NotRecorded /> },
    { key: 'root-time', label: 'Root symptom time', value: <TimeHover date={run.rootSymptom.time} /> },
    {
      key: 'attempt-uid',
      label: 'Attempt UID',
      value: <Copyable text={run.attemptUid} label="Copy attempt UID" />,
    },
    {
      key: 'pipeline',
      label: 'Pipeline',
      value: run.pipeline ? <span className="num">{run.pipeline}</span> : <NotRecorded />,
    },
    { key: 'service', label: 'Service', value: run.service ? <Identifier>{run.service}</Identifier> : <NotRecorded /> },
    ...(errorPairs !== ''
      ? [{
          key: 'error-pairs',
          label: 'Error pairs',
          value: (
            <ul aria-label="error pairs" className="flex flex-col gap-1">
              {errorPairs.split('; ').map((pair) => <li key={pair}><Identifier>{pair}</Identifier></li>)}
            </ul>
          ),
        }]
      : []),
    ...((run.terminalStates ?? []).length > 0
      ? [{ key: 'terminal-state', label: 'Terminal state', value: (run.terminalStates ?? []).map(s => TERMINAL_STATE_LABELS[s] ?? s).join(', ') }]
      : []),
    ...((run.failureDomains ?? []).length > 0
      ? [{ key: 'failure-domain', label: 'Failure domain', value: (run.failureDomains ?? []).map(d => FAILURE_DOMAIN_LABELS[d] ?? d).join(', ') }]
      : []),
    ...((run.cancellationActors ?? []).length > 0
      ? [{ key: 'cancellation-actor', label: 'Cancelled by', value: (run.cancellationActors ?? []).map(a => CANCELLATION_ACTOR_LABELS[a] ?? a).join(', ') }]
      : []),
    ...((run.uncertaintyType ?? null) !== null
      ? [{ key: 'uncertainty-type', label: 'Uncertainty type', value: UNCERTAINTY_LABELS[run.uncertaintyType!] ?? run.uncertaintyType! }]
      : []),
    ...((run.evidenceStatus ?? null) !== null
      ? [{ key: 'evidence-status', label: <HelpTooltip text="Whether all collected evidence reached the attribution service before this verdict was issued">Evidence</HelpTooltip>, value: evidenceStatusValue(run.evidenceStatus!) }]
      : []),
  ]
}

function OperatorGuidance({ card }: { card: PersonaDiagnose }) {
  const lines: DescriptionListItem[] = [
    { key: 'summary', label: 'Summary', value: card.summary },
    { key: 'impact', label: 'Impact', value: card.impact },
    { key: 'action', label: 'Action', value: card.action },
    { key: 'include', label: 'Include', value: card.supportDetails.join(', ') },
  ].filter((line) => line.value !== '')
  return (
    <CardContent className="flex flex-col gap-2 border-t">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className={subtitle}>Cloud operator guidance</h3>
        {card.catalogId && <Badge variant="zinc-subtle" size="wrap"><Identifier>{card.catalogId}</Identifier></Badge>}
      </div>
      {lines.length > 0 && <DescriptionList items={lines} />}
    </CardContent>
  )
}

// The lines the answer rests on, in the main column: the section fills the
// height of its grid row and the lines fill the section, scrolling within it
// once they outgrow it.
function MatchedLines({ run }: { run: RunDetail }) {
  return (
    <SectionCard
      title="Matched lines"
      count={run.log.length > 0 ? run.log.length : undefined}
      flush
      fill
      className="lg:order-first lg:col-span-2"
    >
      <LogExcerpt
        label="Matched lines"
        emptyLabel="No matched lines were recorded."
        lines={run.log.map((line, index) => ({
          key: String(index),
          time: line.t,
          message: line.text,
          // The root line is marked the same way whatever the outcome: it is the line to read, not a blame.
          emphasis: line.level === 'symptom',
        }))}
      />
    </SectionCard>
  )
}

// The answer, compact beside the lines it rests on: the outcome titles the
// card, with the attribution logic, the page's one primary action, at the
// header's end; the root symptom opens the body, and the facts that back it
// follow in one column on one label axis. Documentation links live beside
// symptoms in the table, never here. In the row, the header stays on one line
// and, with the facts' timestamp, sets the column's least width; every other
// recorded value (the root symptom, the facts, the guidance) wraps anywhere,
// so a long one wraps instead of widening the answer.
function VerdictCard({ run }: { run: RunDetail }) {
  const rootSymptom = run.attribution.rootSymptom.state === 'available'
    ? run.attribution.rootSymptom.data.symptom
    : run.rootSymptom.symptom
  const guidance = run.operatorDiagnose?.hasPersona === true ? run.operatorDiagnose : null
  return (
    <Card asChild>
      <section aria-label="Verdict">
        <CardHeader className="border-b lg:flex-nowrap">
          <CardTitle asChild>
            <h2 className="flex items-center">
              <span className="sr-only">Verdict:</span>{' '}
              {/* The badge is the title itself, beside the card's action and
                  no text, so it takes no mark offset: centered in the header
                  as the action's label is. */}
              <CompoundOutcomeBadge run={run} className="top-0" />
            </h2>
          </CardTitle>
          <CardAction>
            <DecisionGraphAction run={run} detail={run} />
          </CardAction>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {rootSymptom !== ''
            ? <p className="font-medium wrap-anywhere"><span className="sr-only">Root symptom:</span> <Identifier>{rootSymptom}</Identifier></p>
            : <p className="text-foreground-muted">Root symptom not recorded</p>}
          <DescriptionList items={verdictFacts(run)} />
        </CardContent>
        {guidance && <OperatorGuidance card={guidance} />}
      </section>
    </Card>
  )
}

function NodesSection({ run }: { run: RunDetail }) {
  const attemptNodes = new Set(run.nodeIds)
  const responsibleNodes = new Set(run.responsibleNodes)
  const nodes = [...new Set([...run.nodeIds, ...run.responsibleNodes])]
  return (
    <SectionCard
      title="Nodes"
      count={`${attemptNodes.size} in attempt · ${responsibleNodes.size} responsible`}
      empty={nodes.length === 0 ? { title: 'No recorded nodes', description: 'No nodes recorded for this attempt.' } : undefined}
    >
      {/* The list scrolls within itself; its 4px inner margin, given back
          outside, keeps the links' 3px focus ring from being cut at its edges.
          The icon, on the name's first line, and the badge, after the name,
          are marks in its running text (markInText): centered on its
          lowercase, and never heightening its 20px line. The whole row is the
          node's link, but its hover underline marks the name alone, not the
          note or badge after it. */}
      <ul
        aria-label="Nodes"
        className="-m-1 grid max-h-72 grid-cols-1 gap-x-4 gap-y-1 overflow-y-auto p-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
      >
        {nodes.map((node) => (
          <li key={node} className="flex min-w-0 items-start gap-2 leading-5">
            <span className="shrink-0">
              <Server aria-hidden="true" className={cn('inline size-3.5 text-foreground-muted', markInText)} />
            </span>
            <NodeHoverCard nodeId={node}>
              <RouterLink to={navigateService.getPath('node', { id: node })} underline="none" className="group/link min-w-0 break-words">
                <span className="underline-offset-4 group-hover/link:underline"><Identifier>{node}</Identifier></span>
                {!attemptNodes.has(node) && (
                  <>
                    {' '}<span aria-hidden="true" className="text-foreground-muted">·</span>{' '}
                    <span className="text-foreground-muted">Not listed in attempt nodes</span>
                  </>
                )}
                {responsibleNodes.has(node) && <>{' '}<SiteStatusBadge status="bad" className={markInText} /></>}
              </RouterLink>
            </NodeHoverCard>
          </li>
        ))}
      </ul>
    </SectionCard>
  )
}

// A payload rebuilt from the fields above, folded until asked for; copying it
// never needs it open. Open, its body is as tall as the JSON, up to 20rem, and
// scrolls within itself past that.
function Payload({ label, payload }: { label: string; payload: object }) {
  const titleId = useId()
  return (
    <Collapsible>
      <CodeBlock code={JSON.stringify(payload, null, 2)}>
        <CodeBlockHeader>
          <CollapsibleTrigger
            className={cn(
              'group/payload flex min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-xs text-left transition-colors hover:text-foreground',
              focusRing,
            )}
          >
            <ChevronRight
              aria-hidden="true"
              className="size-3.5 shrink-0 text-foreground-muted transition-transform group-data-[state=open]/payload:rotate-90"
            />
            <span id={titleId}>{label}</span>{' '}
            <span className="text-foreground-muted">JSON</span>
          </CollapsibleTrigger>
          <CodeBlockActions>
            <CodeBlockCopyButton label={`Copy ${label} as JSON`} />
          </CodeBlockActions>
        </CodeBlockHeader>
        <CollapsibleContent>
          <CodeBlockBody labelledBy={titleId} className="max-h-80" />
        </CollapsibleContent>
      </CodeBlock>
    </Collapsible>
  )
}

// The scheduler summary leaves out the attempt number: the adapter records 1
// for every run, so, as everywhere on the page, it is not shown as data. The
// downloads keep their fields.
function PayloadsSection({ run }: { run: RunDetail }) {
  return (
    <SectionCard title="Payloads">
      <div className="flex flex-col gap-2">
        <Payload
          label="Result summary"
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
        <Payload
          label="Scheduler summary"
          payload={{
            execution_context: run.jobId,
            user: run.user,
            status: run.status,
            nodes: run.nodeIds,
            start_time: run.startTime,
            end_time: run.endTime,
          }}
        />
        {(run.ingestionReports ?? null) !== null && (
          <Payload label="Ingestion reports" payload={run.ingestionReports!} />
        )}
      </div>
    </SectionCard>
  )
}

// While the record loads, the page head holds its shape as the node page's
// does (the way back, a placeholder title and actions, the facts' labels over
// placeholder values), blocks shaped like the answer card and the matched
// lines beside it hold theirs, and a status names what is loading.
function VerdictLoading() {
  return (
    <div className="flex flex-col gap-page-gap">
      <PageHeader
        breadcrumb={VERDICT_CRUMBS}
        actions={
          <>
            <Skeleton height="7" className="w-32" />
            <Skeleton height="7" className="w-24" />
            <Skeleton height="7" className="w-24" />
          </>
        }
        meta={verdictMeta(undefined)}
        metaLoading
      />
      <p role="status" className="sr-only">Loading execution context…</p>
      <div className={ANSWER_ROW}>
        <Card>
          <CardHeader className="border-b">
            <Skeleton height="5" className="w-28" />
            <CardAction>
              <Skeleton height="7" className="w-44" />
            </CardAction>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            <Skeleton height="5" width="2/3" />
            <Skeleton height="4" width="3/4" />
            <Skeleton height="4" width="1/2" />
            <Skeleton height="4" width="2/3" />
          </CardContent>
        </Card>
        <Card className="lg:order-first lg:col-span-2">
          <CardHeader className="border-b">
            <Skeleton height="4" className="w-32" />
          </CardHeader>
          <CardContent flush className="flex flex-1 flex-col">
            <Skeleton className="min-h-60 flex-1 rounded-none" />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export function VerdictView() {
  const { id: verdictId = '' } = useParams<{ id: string }>()
  const api = useApi()
  const { data: run, error, isLoading, mutate } = useSWR<RunDetail | undefined, Error>(
    verdictId !== '' ? ['run', verdictId] : null,
    () => api.getRun(verdictId),
  )
  useDocumentTitle('Verdict', run?.jobId)
  // Without a record there is no title to show, but the breadcrumb stays where it always is.
  if (!run) {
    if (isLoading) return <VerdictLoading />
    return (
      <div className="flex flex-col gap-page-gap">
        <PageBreadcrumb crumbs={VERDICT_CRUMBS} />
        <Empty>
          <EmptyHeader>
            {error ? (
              <>
                <EmptyTitle role="alert">Execution context could not be loaded</EmptyTitle>
                <Button onClick={() => void mutate()}>Try again</Button>
              </>
            ) : (
              <EmptyTitle>Execution context not found</EmptyTitle>
            )}
          </EmptyHeader>
        </Empty>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-page-gap">
      <PageHeader
        breadcrumb={VERDICT_CRUMBS}
        title={<Identifier>{run.jobId}</Identifier>}
        actions={<VerdictActions run={run} />}
        meta={verdictMeta(run)}
      />
      <div className="flex flex-col gap-card-gap">
        <div className={ANSWER_ROW}>
          <VerdictCard run={run} />
          <MatchedLines run={run} />
        </div>
        <NodesSection run={run} />
        <SymptomsSection run={run} />
        <PayloadsSection run={run} />
      </div>
    </div>
  )
}
