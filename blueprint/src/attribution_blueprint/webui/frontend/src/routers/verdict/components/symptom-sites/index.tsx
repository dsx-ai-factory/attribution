// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { useId, useState, type MouseEvent, type ReactNode } from 'react'
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, FileText, Server } from 'lucide-react'
import type { RunDetail, SymptomSite } from '@/interfaces/attribution'
import { TimeHover } from '@/components/time-hover'
import { SymptomDocumentationLink } from '@/components/symptom-documentation'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

const unavailableMessage = {
  absent: 'Symptom-site evidence was not included in this run.',
  null: 'Symptom-site evidence is unavailable for this run.',
  invalid: 'Stored symptom-site evidence could not be read.',
  empty: 'No symptom sites were recorded for this run.',
}

const SITE_STATUS_LABELS: Record<SymptomSite['status'], string> = {
  bad: 'Responsible',
  interesting: 'Supporting evidence',
}

function EvidenceField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="mb-1 text-xs leading-[18px] text-foreground-muted">{label}</dt>
      <dd className="m-0 min-w-0 break-words text-[13px] leading-5 text-foreground-secondary">{children}</dd>
    </div>
  )
}

function SiteStatus({ status, count }: { status: SymptomSite['status']; count?: number }) {
  return (
    <Badge variant={status === 'bad' ? 'accent' : 'neutral'}>
      {count !== undefined && <><span className="font-mono tabular-nums">{count}</span>{' '}</>}
      {SITE_STATUS_LABELS[status]}
    </Badge>
  )
}

function SiteContext({ site }: { site: SymptomSite }) {
  const isNode = site.type === 'NodeSource'
  const Icon = isNode ? Server : FileText
  return (
    <div className="mb-3 flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1.5 text-[13px] text-foreground-secondary">
      <span className="inline-flex min-w-0 items-center gap-1.5">
        <Icon aria-hidden="true" className="size-3.5 shrink-0 text-foreground-muted" />
        <span className="min-w-0 break-words">{isNode ? site.node || 'Node not recorded' : 'Job-level evidence'}</span>
      </span>
      <span className="min-w-0 break-words text-foreground-muted">
        <span className="sr-only">Source: </span>
        <Badge variant="neutral">{site.source.source || 'Source not recorded'}</Badge>
      </span>
      {site.status === 'bad' && <SiteStatus status="bad" />}
    </div>
  )
}

function ScopeSummary({ sites }: { sites: SymptomSite[] }) {
  if (sites.length === 0) return <>—</>
  if (sites.length === 1) return <>{sites[0].type === 'NodeSource' ? sites[0].node || 'Node not recorded' : 'Job-level evidence'}</>
  const nodes = new Set(sites.flatMap((site) => site.type === 'NodeSource' && site.node ? [site.node] : []))
  return (
    <div className="space-y-1">
      {nodes.size > 0 && <div><span className="font-mono tabular-nums">{nodes.size}</span> {nodes.size === 1 ? 'node' : 'nodes'}</div>}
      {sites.some((site) => site.type === 'ResourcelessSource') && <div>Job-level evidence</div>}
      {sites.some((site) => site.type === 'NodeSource' && !site.node) && <div>Node not recorded</div>}
    </div>
  )
}

function SourceSummary({ sites }: { sites: SymptomSite[] }) {
  if (sites.length === 0) return <>—</>
  const sources = new Set(sites.map((site) => site.source.source))
  if (sources.size === 1) return <>{sites[0].source.source || 'Source not recorded'}</>
  const hasMissingSource = sources.delete('')
  return (
    <div className="space-y-1">
      <div><span className="font-mono tabular-nums">{sources.size}</span> {sources.size === 1 ? 'source' : 'sources'}</div>
      {hasMissingSource && <div>Source not recorded</div>}
    </div>
  )
}

function StatusSummary({ sites }: { sites: SymptomSite[] }) {
  if (sites.length === 0) return <>—</>
  const responsibleCount = sites.filter((site) => site.status === 'bad').length
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1">
      {responsibleCount > 0 && <SiteStatus status="bad" count={responsibleCount} />}
      {responsibleCount < sites.length && <SiteStatus status="interesting" count={sites.length - responsibleCount} />}
    </div>
  )
}

function SourceLog({ line }: { line: string | null }) {
  const titleId = useId()
  return (
    <div className="mt-3 min-w-0">
      <div id={titleId} className="mb-1.5 text-xs leading-[18px] text-foreground-muted">Source evidence</div>
      {line === null || line === '' ? (
        <p className="m-0 text-[13px] leading-5 text-foreground-muted">{line === null ? 'No source line recorded.' : 'Source line is empty.'}</p>
      ) : (
        <div role="region" aria-labelledby={titleId} tabIndex={0} className="max-h-48 overflow-auto rounded-sm border border-border bg-background-secondary py-1.5 focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-foreground-muted">
          <pre className="m-0 min-h-[22px] min-w-0 whitespace-pre-wrap break-words px-3 font-sans text-[13px] leading-[22px] text-foreground-secondary">{line}</pre>
        </div>
      )}
    </div>
  )
}

function SiteEvidence({ site, showContext }: { site: SymptomSite; showContext: boolean }) {
  const { source } = site
  const label = site.type === 'NodeSource' ? site.node || 'Node not recorded' : 'Job-level evidence'
  const status = SITE_STATUS_LABELS[site.status]
  return (
    <article aria-label={`${label}: ${status}`} className="min-w-0 py-3.5 first:pt-0 last:pb-0">
      {showContext && <SiteContext site={site} />}
      <dl className="m-0 grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,.65fr)_minmax(0,1fr)] gap-5">
        <EvidenceField label="Rule">{source.rule || 'Not recorded'}</EvidenceField>
        <EvidenceField label="Observation"><span className="font-mono">{source.observation_id}</span></EvidenceField>
        <EvidenceField label="Source position">
          {source.pos === null ? 'Not recorded' : <span className="font-mono">{source.pos}</span>}
        </EvidenceField>
        <EvidenceField label="Recorded time">
          {source.time ? <time dateTime={source.time} className="font-mono"><TimeHover date={source.time} className="no-underline cursor-default" /></time> : 'Not recorded'}
        </EvidenceField>
      </dl>
      <SourceLog line={source.line} />
    </article>
  )
}

const EVIDENCE_PAGE_SIZE = 10

function EvidenceSites({ symptom, sites }: { symptom: string; sites: SymptomSite[] }) {
  const scrollId = useId()
  const [requestedPage, setPage] = useState(0)
  const lastPage = Math.max(0, Math.ceil(sites.length / EVIDENCE_PAGE_SIZE) - 1)
  const page = Math.min(requestedPage, lastPage)
  if (page !== requestedPage) setPage(page)
  const start = page * EVIDENCE_PAGE_SIZE
  const end = Math.min(start + EVIDENCE_PAGE_SIZE, sites.length)

  return (
    <>
      <div
        key={page}
        id={scrollId}
        role="region"
        aria-label={`${symptom || 'Unnamed symptom'} evidence`}
        tabIndex={0}
        className="max-h-[min(60vh,28rem)] min-w-0 overflow-auto rounded-sm focus-visible:outline-1 focus-visible:outline-offset-[-1px] focus-visible:outline-foreground-muted"
      >
        <div className="divide-y divide-border">
          {sites.slice(start, end).map((site, index) => <SiteEvidence key={start + index} site={site} showContext={sites.length > 1} />)}
        </div>
      </div>
      {sites.length > EVIDENCE_PAGE_SIZE && (
        <div role="group" aria-label={`${symptom || 'Unnamed symptom'} evidence pages`} className="mt-3 flex items-center justify-between gap-3 border-t border-border pt-2.5 text-xs text-foreground-muted">
          <span aria-live="polite" aria-atomic="true">
            <span className="font-mono tabular-nums">{start + 1}–{end}</span> of <span className="font-mono tabular-nums">{sites.length}</span> sites
          </span>
          <div className="flex items-center gap-1">
            <Button type="button" variant="ghost" size="icon-sm" disabled={page === 0} aria-label="First evidence page" aria-controls={scrollId} onClick={() => setPage(0)}><ChevronsLeft aria-hidden="true" className="size-3.5" /></Button>
            <Button type="button" variant="ghost" size="icon-sm" disabled={page === 0} aria-label="Previous evidence page" aria-controls={scrollId} onClick={() => setPage(page - 1)}><ChevronLeft aria-hidden="true" className="size-3.5" /></Button>
            <Button type="button" variant="ghost" size="icon-sm" disabled={page === lastPage} aria-label="Next evidence page" aria-controls={scrollId} onClick={() => setPage(page + 1)}><ChevronRight aria-hidden="true" className="size-3.5" /></Button>
            <Button type="button" variant="ghost" size="icon-sm" disabled={page === lastPage} aria-label="Last evidence page" aria-controls={scrollId} onClick={() => setPage(lastPage)}><ChevronsRight aria-hidden="true" className="size-3.5" /></Button>
          </div>
        </div>
      )}
    </>
  )
}

function SymptomRows({ symptom, sites }: { symptom: string; sites: SymptomSite[] }) {
  const detailId = useId()
  const [open, setOpen] = useState(false)
  const onRowClick = (event: MouseEvent<HTMLTableRowElement>) => {
    if (!(event.target instanceof Element) || event.target.closest('button, a')) return
    const selection = window.getSelection()
    if (selection?.toString() && (event.currentTarget.contains(selection.anchorNode) || event.currentTarget.contains(selection.focusNode))) return
    setOpen((value) => !value)
  }
  return (
    <>
      <TableRow onClick={onRowClick} className="cursor-pointer border-t border-b-0 border-border hover:bg-background-active focus-within:bg-background-active has-aria-expanded:bg-transparent has-aria-expanded:hover:bg-background-active has-aria-expanded:focus-within:bg-background-active [&>td]:px-3.5 [&>td]:py-2.5 [&>td]:align-middle [&>td]:whitespace-normal [&>td]:break-words">
        <TableCell>
          <div className="flex min-w-0 items-center gap-1.5">
            <button type="button" aria-label={`${symptom || 'Unnamed symptom'} ${sites.length} ${sites.length === 1 ? 'site' : 'sites'}`} aria-expanded={open} aria-controls={detailId} onClick={() => setOpen((value) => !value)} className="inline-flex min-w-0 cursor-pointer items-center gap-1.5 rounded-sm py-1 text-left font-medium text-foreground-secondary hover:text-foreground focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-foreground-muted">
              <ChevronRight aria-hidden="true" className={`size-3 shrink-0 text-foreground-muted transition-transform ${open ? 'rotate-90' : ''}`} />
              <span className="min-w-0 break-words">{symptom || 'Unnamed symptom'}</span>
            </button>
            <SymptomDocumentationLink symptomId={symptom} />
          </div>
        </TableCell>
        <TableCell><ScopeSummary sites={sites} /></TableCell>
        <TableCell className="text-foreground-muted"><SourceSummary sites={sites} /></TableCell>
        <TableCell><StatusSummary sites={sites} /></TableCell>
        <TableCell className="text-right font-mono tabular-nums text-foreground-muted">{sites.length}</TableCell>
      </TableRow>
      <TableRow id={detailId} hidden={!open} className="border-t border-b-0 border-border bg-background-active/30 transition-none hover:bg-background-active/30">
        <TableCell colSpan={5} className="px-4 py-3.5 whitespace-normal">
          {open && (sites.length > 0 ? (
            <EvidenceSites symptom={symptom} sites={sites} />
          ) : <p className="m-0 text-[13px] leading-5 text-foreground-muted">No sites recorded for this symptom.</p>)}
        </TableCell>
      </TableRow>
    </>
  )
}

export function SymptomsSection({ run }: { run: RunDetail }) {
  const titleId = useId()
  const section = run.attribution.symptomSites
  const symptoms = section.state === 'available' ? Object.entries(section.data) : []
  const symptomCount = section.state === 'available' ? symptoms.length : run.symptoms.length

  return (
    <section aria-labelledby={titleId} className="min-w-0 overflow-hidden rounded-sm border border-border bg-card">
      <div className="flex items-baseline gap-3 border-b border-border px-3.5 py-3">
        <h2 id={titleId} className="m-0 text-xs font-semibold uppercase tracking-wide">Symptoms</h2>
        <span className="text-[11px] text-foreground-muted"><span className="font-mono tabular-nums">{symptomCount}</span> distinct</span>
      </div>
      {section.state !== 'available' ? (
        <div className="px-3.5 py-3">
          {run.symptoms.length > 0 && (
            <ul aria-label="Recorded symptom summary" className="mb-3 mt-0 list-none divide-y divide-border p-0">
              {run.symptoms.map((symptom) => (
                <li key={symptom.name} className="flex min-h-10 min-w-0 items-center gap-2 py-2 text-[13px] text-foreground-secondary">
                  <FileText aria-hidden="true" className="size-3.5 shrink-0 text-foreground-muted" />
                  <span className="min-w-0 break-words">{symptom.name || 'Unnamed symptom'}</span>
                  <span className="ml-auto shrink-0 text-xs text-foreground-muted"><span className="font-mono tabular-nums">{symptom.sites}</span> {symptom.sites === 1 ? 'site' : 'sites'}</span>
                </li>
              ))}
            </ul>
          )}
          <p className="m-0 text-[13px] leading-5 text-foreground-muted">{unavailableMessage[section.state]}</p>
        </div>
      ) : (
        <Table aria-labelledby={titleId} className="min-w-[880px] table-fixed border-collapse text-[13px] text-foreground-secondary">
          <colgroup><col className="w-[39%]" /><col className="w-[19%]" /><col className="w-[13%]" /><col className="w-[23%]" /><col className="w-[6%]" /></colgroup>
          <TableHeader className="bg-background-active/30 text-xs text-foreground-muted [&_tr]:border-b-0">
            <TableRow className="border-0 transition-none hover:bg-transparent [&>th]:px-3.5 [&>th]:py-2">
              <TableHead scope="col" className="h-auto whitespace-normal text-foreground-muted">Symptom</TableHead>
              <TableHead scope="col" className="h-auto whitespace-normal text-foreground-muted">Scope</TableHead>
              <TableHead scope="col" className="h-auto whitespace-normal text-foreground-muted">Source</TableHead>
              <TableHead scope="col" className="h-auto whitespace-normal text-foreground-muted">Status</TableHead>
              <TableHead scope="col" className="h-auto whitespace-normal text-right text-foreground-muted">Sites</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody key={run.id} className="[&_tr:last-child]:border-t">
            {symptoms.map(([symptom, sites]) => <SymptomRows key={symptom} symptom={symptom} sites={sites} />)}
          </TableBody>
        </Table>
      )}
    </section>
  )
}
