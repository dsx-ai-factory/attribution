// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId, useState } from 'react'
import { ChevronRight, FileText, Server } from 'lucide-react'
import type { RunDetail, SymptomSite } from '@/interfaces/attribution'
import { SITE_STATUS_LABELS } from '@/lib/attribution-data'
import { counted } from '@/lib/format'
import { shouldActivateRow } from '@/lib/row-activation'
import { cellInset, cn, focusRing, focusRingInset, markInText } from '@/lib/utils'
import { Field, Identifier, NoValue, NotRecorded } from '@/components/field'
import { SectionCard } from '@/components/section-card'
import { SiteStatusBadge } from '@/components/site-status-badge'
import { SymptomDocumentationLink } from '@/components/symptom-documentation'
import { TablePagination } from '@/components/table-pagination'
import { TimeHover } from '@/components/time-hover'
import { Badge } from '@/components/ui/badge'
import {
  CodeBlock,
  CodeBlockActions,
  CodeBlockBody,
  CodeBlockCopyButton,
  CodeBlockHeader,
  CodeBlockTitle,
} from '@/components/ui/code-block'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

const unavailableMessage = {
  absent: 'Symptom-site evidence was not included in this run.',
  null: 'Symptom-site evidence is unavailable for this run.',
  invalid: 'Stored symptom-site evidence could not be read.',
  empty: 'No symptom sites were recorded for this run.',
}

function SiteContext({ site }: { site: SymptomSite }) {
  const isNode = site.type === 'NodeSource'
  const Icon = isNode ? Server : FileText
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-foreground-secondary">
      {/* The icon is a mark beside a name that may wrap, so it keeps to the first line. */}
      <span className="inline-flex min-w-0 items-start gap-1.5">
        <span><Icon aria-hidden="true" className={cn('inline size-3.5 text-foreground-muted', markInText)} /></span>
        <span className="min-w-0 break-words">{isNode ? (site.node ? <Identifier>{site.node}</Identifier> : 'Node not recorded') : 'Job-level evidence'}</span>
      </span>
      <span className="min-w-0">
        <span className="sr-only">Source: </span>
        <Badge variant="zinc-subtle" size="wrap">{site.source.source ? <Identifier>{site.source.source}</Identifier> : 'Source not recorded'}</Badge>
      </span>
      {site.status === 'bad' && <SiteStatusBadge status="bad" />}
    </div>
  )
}

function ScopeSummary({ sites }: { sites: SymptomSite[] }) {
  if (sites.length === 0) return <NoValue />
  if (sites.length === 1) {
    if (sites[0].type !== 'NodeSource') return 'Job-level evidence'
    return sites[0].node ? <Identifier>{sites[0].node}</Identifier> : 'Node not recorded'
  }
  const nodes = new Set(sites.flatMap((site) => site.type === 'NodeSource' && site.node ? [site.node] : []))
  return (
    <div className="flex flex-col gap-1">
      {nodes.size > 0 && <div className="tabular-nums">{counted(nodes.size, 'node', 'nodes')}</div>}
      {sites.some((site) => site.type === 'ResourcelessSource') && <div>Job-level evidence</div>}
      {sites.some((site) => site.type === 'NodeSource' && !site.node) && <div>Node not recorded</div>}
    </div>
  )
}

function SourceSummary({ sites }: { sites: SymptomSite[] }) {
  if (sites.length === 0) return <NoValue />
  const sources = new Set(sites.map((site) => site.source.source))
  if (sources.size === 1) return sites[0].source.source ? <Identifier>{sites[0].source.source}</Identifier> : 'Source not recorded'
  const hasMissingSource = sources.delete('')
  return (
    <div className="flex flex-col gap-1">
      <div className="tabular-nums">{counted(sources.size, 'source', 'sources')}</div>
      {hasMissingSource && <div>Source not recorded</div>}
    </div>
  )
}

function StatusSummary({ sites }: { sites: SymptomSite[] }) {
  if (sites.length === 0) return <NoValue />
  const responsibleCount = sites.filter((site) => site.status === 'bad').length
  return (
    <div className="flex flex-wrap gap-x-2 gap-y-1">
      {responsibleCount > 0 && <SiteStatusBadge status="bad" count={responsibleCount} />}
      {responsibleCount < sites.length && <SiteStatusBadge status="interesting" count={sites.length - responsibleCount} />}
    </div>
  )
}

// The recorded line itself, as machine output: mono, wrapped so it is never
// cut at the right edge, and copyable without selecting it.
function SourceEvidence({ line }: { line: string | null }) {
  const titleId = useId()
  if (line === null || line === '') {
    return (
      <dl>
        <Field label="Source evidence">
          <span className="text-foreground-muted">{line === null ? 'No source line recorded.' : 'Source line is empty.'}</span>
        </Field>
      </dl>
    )
  }
  return (
    <CodeBlock code={line}>
      <CodeBlockHeader>
        <CodeBlockTitle id={titleId}>Source evidence</CodeBlockTitle>
        <CodeBlockActions>
          <CodeBlockCopyButton label="Copy source evidence" />
        </CodeBlockActions>
      </CodeBlockHeader>
      <CodeBlockBody labelledBy={titleId} wrap className="max-h-48" />
    </CodeBlock>
  )
}

function SiteEvidence({ site, showContext }: { site: SymptomSite; showContext: boolean }) {
  const { source } = site
  const label = site.type === 'NodeSource' ? site.node || 'Node not recorded' : 'Job-level evidence'
  const status = SITE_STATUS_LABELS[site.status]
  return (
    <article aria-label={`${label}: ${status}`} className="flex min-w-0 flex-col gap-2 py-card-y first:pt-0 last:pb-0">
      {showContext && <SiteContext site={site} />}
      {/* The four facts stand on the table's own columns: the grid spans the
          row's full width with the table's column widths (SYMPTOM_COLUMNS;
          the last fact takes the Status and Sites columns), each fact with
          the table's own cell inset, so Rule, Observation, Source position
          and Recorded time start where Symptom, Scope, Source and Status do. */}
      <dl className="-mx-card-x grid grid-cols-[var(--col-symptom)_var(--col-scope)_var(--col-source)_minmax(0,1fr)]">
        <Field label="Rule" className={cellInset}>{source.rule ? <Identifier>{source.rule}</Identifier> : <NotRecorded />}</Field>
        <Field label="Observation" className={cellInset}><span className="num">{source.observation_id}</span></Field>
        <Field label="Source position" className={cellInset}>
          {source.pos === null ? <NotRecorded /> : <span className="num">{source.pos}</span>}
        </Field>
        <Field label="Recorded time" className={cellInset}>
          {source.time ? <TimeHover date={source.time} /> : <NotRecorded />}
        </Field>
      </dl>
      <SourceEvidence line={source.line} />
    </article>
  )
}

const EVIDENCE_PAGE_SIZE = 10

// The Symptoms table's column widths, as shares of the table's width, set once
// on the table: its colgroup takes all five, and an expanded site's facts
// (inside the table) the first three, so the two always agree. The Sites
// share holds its header and the card inset at the table's 880px least width.
const SYMPTOM_COLUMNS = '[--col-symptom:38%] [--col-scope:19%] [--col-source:13%] [--col-status:23%] [--col-sites:7%]'

function EvidenceSites({ symptom, sites }: { symptom: string; sites: SymptomSite[] }) {
  const scrollId = useId()
  const [requestedPage, setPage] = useState(0)
  const lastPage = Math.max(0, Math.ceil(sites.length / EVIDENCE_PAGE_SIZE) - 1)
  const page = Math.min(requestedPage, lastPage)
  if (page !== requestedPage) setPage(page)
  const start = page * EVIDENCE_PAGE_SIZE
  const end = Math.min(start + EVIDENCE_PAGE_SIZE, sites.length)
  const name = symptom || 'Unnamed symptom'

  return (
    <div className="flex flex-col gap-2">
      {/* The region spans the row's full width, its content kept on the
          cell's inset, so a site's facts can reach the table's column edges
          without scrolling sideways. */}
      <div
        key={page}
        id={scrollId}
        role="region"
        aria-label={`${name} evidence`}
        tabIndex={0}
        className={cn('-mx-card-x max-h-[min(60vh,28rem)] min-w-0 overflow-auto rounded-sm px-card-x', focusRingInset)}
      >
        <div className="divide-y divide-border">
          {sites.slice(start, end).map((site, index) => <SiteEvidence key={start + index} site={site} showContext={sites.length > 1} />)}
        </div>
      </div>
      {sites.length > EVIDENCE_PAGE_SIZE && (
        <TablePagination
          page={page}
          pageSize={EVIDENCE_PAGE_SIZE}
          total={sites.length}
          onPageChange={setPage}
          label="evidence"
          itemLabel="sites"
          controlsId={scrollId}
          aria-label={`${name} evidence pages`}
        />
      )}
    </div>
  )
}

function SymptomRows({ symptom, sites }: { symptom: string; sites: SymptomSite[] }) {
  const detailId = useId()
  const [open, setOpen] = useState(false)
  const name = symptom || 'Unnamed symptom'
  return (
    <>
      <TableRow
        onClick={(event) => { if (shouldActivateRow(event)) setOpen((value) => !value) }}
        className="cursor-pointer border-t border-b-0"
      >
        <TableCell wrap>
          {/* A name that wraps keeps its toggle's chevron, centered in the
              control, and its documentation link on its first line. */}
          <div className="flex min-w-0 items-start gap-1.5">
            <button
              type="button"
              aria-label={`${name} ${counted(sites.length, 'site', 'sites')}`}
              aria-expanded={open}
              aria-controls={detailId}
              onClick={() => setOpen((value) => !value)}
              className={cn('inline-flex min-w-0 cursor-pointer items-start gap-1.5 rounded-xs text-left text-foreground', focusRing)}
            >
              <span className="flex h-5 shrink-0 items-center">
                <ChevronRight aria-hidden="true" className={cn('size-3.5 text-foreground-muted transition-transform', open && 'rotate-90')} />
              </span>
              <span className="min-w-0 break-words">{symptom ? <Identifier>{symptom}</Identifier> : name}</span>
            </button>
            <SymptomDocumentationLink symptomId={symptom} />
          </div>
        </TableCell>
        <TableCell wrap><ScopeSummary sites={sites} /></TableCell>
        <TableCell wrap><SourceSummary sites={sites} /></TableCell>
        <TableCell wrap><StatusSummary sites={sites} /></TableCell>
        <TableCell className="num text-right">{sites.length}</TableCell>
      </TableRow>
      <TableRow id={detailId} hidden={!open} className="border-t border-b-0 bg-background-secondary hover:bg-background-secondary">
        <TableCell colSpan={5} wrap className="py-card-y">
          {open && (sites.length > 0 ? (
            <EvidenceSites symptom={symptom} sites={sites} />
          ) : <p className="text-foreground-muted">No sites recorded for this symptom.</p>)}
        </TableCell>
      </TableRow>
    </>
  )
}

export function SymptomsSection({ run }: { run: RunDetail }) {
  const section = run.attribution.symptomSites
  const symptoms = section.state === 'available' ? Object.entries(section.data) : []
  const symptomCount = section.state === 'available' ? symptoms.length : run.symptoms.length
  // The table is named by the section title itself, so the two never drift apart.
  const titleId = useId()
  // Without sites or a recorded summary the section has nothing to show; beside
  // a summary, why its sites are missing is a note below it. The summary's rows
  // are flush 28px rows on the card's inset, as the table's are, so a symptom
  // row is as tall on every verdict whichever form it takes.
  const nothingToShow = section.state !== 'available' && run.symptoms.length === 0

  return (
    <SectionCard
      title="Symptoms"
      titleId={titleId}
      count={`${symptomCount} distinct`}
      flush
      empty={nothingToShow ? { title: 'No symptoms to show', description: unavailableMessage[section.state] } : undefined}
    >
      {section.state !== 'available' ? (
        <>
          <ul aria-label="Recorded symptom summary" className="divide-y divide-border">
            {run.symptoms.map((symptom) => (
              <li key={symptom.name} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-2 px-card-x py-1">
                <span><FileText aria-hidden="true" className={cn('inline size-3.5 text-foreground-muted', markInText)} /></span>
                <span className="break-words text-foreground-secondary">{symptom.name ? <Identifier>{symptom.name}</Identifier> : 'Unnamed symptom'}</span>
                <span className="text-foreground-muted tabular-nums">{counted(symptom.sites, 'site', 'sites')}</span>
              </li>
            ))}
          </ul>
          <p className="border-t px-card-x py-card-y text-foreground-muted">{unavailableMessage[section.state]}</p>
        </>
      ) : (
        <Table aria-labelledby={titleId} layout="fixed" className={cn('min-w-[880px]', SYMPTOM_COLUMNS)}>
          <colgroup>
            <col className="w-(--col-symptom)" />
            <col className="w-(--col-scope)" />
            <col className="w-(--col-source)" />
            <col className="w-(--col-status)" />
            <col className="w-(--col-sites)" />
          </colgroup>
          <TableHeader>
            <TableRow>
              <TableHead scope="col">Symptom</TableHead>
              <TableHead scope="col">Scope</TableHead>
              <TableHead scope="col">Source</TableHead>
              <TableHead scope="col">Status</TableHead>
              <TableHead scope="col" className="text-right">Sites</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody key={run.id} className="[&_tr:last-child]:border-t">
            {symptoms.map(([symptom, sites]) => <SymptomRows key={symptom} symptom={symptom} sites={sites} />)}
          </TableBody>
        </Table>
      )}
    </SectionCard>
  )
}
