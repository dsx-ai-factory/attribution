import { Fragment, useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import type { OutcomeKey, RunSummary } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import type { PageByOutcome, RunsTableGroup } from '@/hooks/use-runs-table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { OutcomeChip } from '@/components/chip'
import { RunsToolbar } from '@/routers/fleet/components/runs-toolbar'
import { DefHover } from '@/components/def-hover'
import { JobHoverCard } from '@/components/job-hover-card'
import { NodeHoverCard } from '@/components/node-hover-card'
import { TimeHover } from '@/components/time-hover'
import { attemptGpuHours } from '@/lib/metric'
import { isWindowed } from '@/lib/date-window'
import { useNav } from '@/hooks/use-nav'
import { DecisionGraphAction } from '@/routers/verdict/components/decision-graph/dialog'

const OUTCOME_ORDER: { key: OutcomeKey; label: string; dot: string }[] = [
  { key: 'hardware', label: 'Hardware Fault',        dot: 'bg-red-400' },
  { key: 'software', label: 'System Software Fault', dot: 'bg-blue-400' },
  { key: 'multi',    label: 'Ambiguous Fault',       dot: 'bg-zinc-400' },
  { key: 'user',     label: 'Application Fault',     dot: 'bg-yellow-400' },
  { key: 'undetermined', label: 'Unattributed',    dot: 'bg-purple-500' },
  { key: 'pass',     label: 'Pass',                  dot: 'bg-primary' },
]

const COL_COUNT = 10
const PAGE_SIZE_OPTIONS = [10, 25, 50]
const RESPONSIBLE_NODE_INLINE_LIMIT = 3

function RunColumnHeaders({ className = '' }: { className?: string }) {
  return (
    <TableRow className={className}>
      <TableHead className="w-36">
        <DefHover term="end-time-utc">End time (UTC)</DefHover>
      </TableHead>
      <TableHead>Execution Context</TableHead>
      <TableHead className="w-14">Att</TableHead>
      <TableHead>User</TableHead>
      <TableHead className="w-28">
        <DefHover term="outcome">Outcome</DefHover>
      </TableHead>
      <TableHead className="w-14 text-right">GPUs</TableHead>
      <TableHead className="w-24 text-right">
        <DefHover term="gpu-hours">GPU-hours</DefHover>
      </TableHead>
      <TableHead>
        <DefHover term="root-symptom">Root symptom</DefHover>
      </TableHead>
      <TableHead>
        <DefHover term="responsible-nodes">Responsible nodes</DefHover>
      </TableHead>
      <TableHead className="w-16 text-center font-sans">Actions</TableHead>
    </TableRow>
  )
}

function RunRow({ r }: { r: RunSummary }) {
  const nav = useNav()
  const visibleResponsibleNodes = r.responsibleNodes.slice(0, RESPONSIBLE_NODE_INLINE_LIMIT)
  const hasMoreResponsibleNodes = r.responsibleNodes.length > visibleResponsibleNodes.length
  return (
    <TableRow
      onClick={(event) => {
        // Portaled dialogs and tooltips are not part of the clickable row.
        if (event.currentTarget.contains(event.target as Node)) nav('verdict', { id: r.id })
      }}
      className="cursor-pointer group"
    >
      <TableCell className="font-mono text-xs text-foreground">
        <TimeHover date={r.endTime} />
      </TableCell>
      <TableCell className="font-mono text-xs text-foreground group-hover:text-primary">
        <JobHoverCard run={r} linkable>
          <span>{r.jobId}</span>
        </JobHoverCard>
      </TableCell>
      <TableCell className="font-mono text-xs">
        <span className="text-foreground-secondary">{r.attempt}</span>
      </TableCell>
      <TableCell className="text-foreground-secondary">{r.user}</TableCell>
      <TableCell>
        <div className="flex items-center">
          <OutcomeChip outcome={r.outcome} />
        </div>
      </TableCell>
      <TableCell className="font-mono text-foreground text-right">{r.gpuCount}</TableCell>
      <TableCell className="font-mono text-foreground text-right">
        {attemptGpuHours(r).toFixed(2)}
      </TableCell>
      <TableCell className="font-mono text-xs text-foreground">
        {r.rootSymptom.symptom}
      </TableCell>
      <TableCell className="font-mono text-xs">
        {r.responsibleNodes.length > 0 ? (
          <span className="inline-flex flex-wrap gap-x-1.5" title={r.responsibleNodes.join(', ')}>
            {visibleResponsibleNodes.map((n, i) => (
              <NodeHoverCard key={n} nodeId={n}>
                <a
                  onClick={(e) => {
                    e.stopPropagation()
                    nav('node', { id: n })
                  }}
                  className="text-foreground hover:text-primary"
                >
                  {n}
                  {i < visibleResponsibleNodes.length - 1 || hasMoreResponsibleNodes ? ',' : ''}
                </a>
              </NodeHoverCard>
            ))}
            {hasMoreResponsibleNodes && <span className="text-foreground-muted">...</span>}
          </span>
        ) : (
          <span aria-label="no responsible nodes" className="text-zinc-700 select-none">·</span>
        )}
      </TableCell>
      <TableCell className="py-1">
        <div className="flex justify-center">
          <DecisionGraphAction run={r} compact />
        </div>
      </TableCell>
    </TableRow>
  )
}

function PaginationRow({
  label,
  page,
  pageSize,
  total,
  onPageChange,
  onPageSizeChange,
}: {
  label: string
  page: number
  pageSize: number
  total: number
  onPageChange: (page: number) => void
  onPageSizeChange: (pageSize: number) => void
}) {
  const pageCount = Math.ceil(total / pageSize)
  const start = page * pageSize
  const end = Math.min(start + pageSize, total)

  return (
    <TableRow className="bg-background-secondary hover:bg-background-secondary">
      <TableCell colSpan={COL_COUNT} className="py-2">
        <div className="flex items-center justify-end gap-3 font-mono text-xs text-foreground-secondary">
          <label className="flex items-center gap-1.5">
            <span>Records per page:</span>
            <select
              value={pageSize}
              onChange={(event) => onPageSizeChange(Number(event.target.value))}
              className="bg-card border border-border rounded-sm px-1.5 py-0.5 text-foreground focus:outline-none focus:border-zinc-700"
              aria-label={`${label} records per page`}
            >
              {PAGE_SIZE_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <span>
            {total === 0 ? '0-0' : `${start + 1}-${end}`} of {total}
          </span>
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              disabled={page === 0}
              aria-label={`Previous ${label} page`}
              onClick={() => onPageChange(page - 1)}
            >
              <ChevronLeft className="size-3" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              disabled={page >= pageCount - 1}
              aria-label={`Next ${label} page`}
              onClick={() => onPageChange(page + 1)}
            >
              <ChevronRight className="size-3" />
            </Button>
          </div>
        </div>
      </TableCell>
    </TableRow>
  )
}

export function RunsSection({
  groups,
  matchingRuns,
  visibleRuns,
  filter,
  setFilter,
  pageByOutcome,
  pageSize,
  setPageSize,
  setGroupPage,
  loading = false,
  databaseEmpty = false,
}: {
  groups: RunsTableGroup[]
  matchingRuns: number
  visibleRuns: RunSummary[]
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
  pageByOutcome: PageByOutcome
  pageSize: number
  setPageSize: (next: number) => void
  setGroupPage: (outcome: OutcomeKey, page: number) => void
  loading?: boolean
  databaseEmpty?: boolean
}) {
  const [collapsed, setCollapsed] = useState<Set<OutcomeKey>>(new Set())

  const displayGroups = useMemo(
    () =>
      OUTCOME_ORDER.flatMap((g) => {
        const group = groups.find((candidate) => candidate.outcome === g.key)
        if (!group) return []
        return [{ ...g, ...group, rows: group.runs }]
      }),
    [groups],
  )

  const windowed = isWindowed(filter)
  const hints = databaseEmpty ? [] : [
    filter.outcome || filter.symptom || filter.node || filter.rule || filter.user || filter.q
      ? 'clear the active filter chips above'
      : null,
    filter.status === 'failed' ? 'uncheck "Hide passing runs"' : null,
    windowed ? 'widen the date range' : null,
  ].filter((hint) => hint !== null)

  const toggleCollapse = (k: OutcomeKey) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(k)) next.delete(k)
      else next.add(k)
      return next
    })
  }

  return (
    <div>
      <RunsToolbar runs={matchingRuns} filteredRuns={visibleRuns} filter={filter} setFilter={setFilter} />

      <ScrollArea className="border border-border rounded-md bg-card">
        <Table>
          <TableHeader>
            <RunColumnHeaders />
          </TableHeader>
          <TableBody>
            {loading ? (
              Array.from({ length: 6 }).map((_, i) => (
                <TableRow key={`sk-${i}`}>
                  <TableCell colSpan={COL_COUNT} className="py-2.5">
                    <div className="h-3 w-full bg-background-muted rounded-sm animate-pulse" />
                  </TableCell>
                </TableRow>
              ))
            ) : matchingRuns === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={COL_COUNT}
                  className="text-foreground-muted text-center py-8"
                >
                  <div className="font-mono text-sm text-foreground mb-1">
                    {databaseEmpty ? 'No runs in this database' : 'No runs in this view'}
                  </div>
                  {hints.length > 0 && <div className="text-xs">Try to {hints.join(', or ')}.</div>}
                </TableCell>
              </TableRow>
            ) : (
              displayGroups.map((g) => {
                const isCollapsed = collapsed.has(g.key)
                const shouldPaginate = g.total > pageSize
                const pageCount = Math.ceil(g.total / pageSize)
                const page = shouldPaginate
                  ? Math.min(
                      Math.max(pageByOutcome[g.key] ?? 0, 0),
                      pageCount - 1,
                    )
                  : 0
                return (
                  <Fragment key={g.key}>
                    <TableRow
                      onClick={() => toggleCollapse(g.key)}
                      className="cursor-pointer bg-background-secondary hover:bg-background-muted"
                    >
                      <TableCell
                        colSpan={COL_COUNT}
                        className="font-mono text-xs uppercase tracking-widest text-foreground py-2"
                      >
                        <span className="inline-flex items-center gap-2">
                          <ChevronRight
                            className={`size-3 transition-transform ${
                              isCollapsed ? '' : 'rotate-90'
                            } text-foreground-muted`}
                          />
                          <span className={`inline-block size-2 rounded-sm ${g.dot}`} />
                            {g.label}
                          <Badge
                            variant="outline"
                            className="font-mono text-foreground-secondary ml-1"
                          >
                            {g.total}
                          </Badge>
                        </span>
                      </TableCell>
                    </TableRow>
                    {!isCollapsed && (
                      <RunColumnHeaders className="bg-card hover:bg-card [&>th]:h-8 [&>th]:text-[11px] [&>th]:text-foreground-muted" />
                    )}
                    {!isCollapsed &&
                      g.rows.map((r) => <RunRow key={r.id} r={r} />)}
                    {!isCollapsed && shouldPaginate && (
                      <PaginationRow
                        label={g.label}
                        page={page}
                        pageSize={pageSize}
                        total={g.total}
                        onPageChange={(nextPage) => setGroupPage(g.key, nextPage)}
                        onPageSizeChange={setPageSize}
                      />
                    )}
                  </Fragment>
                )
              })
            )}
          </TableBody>
        </Table>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  )
}
