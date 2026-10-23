// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useId, useMemo, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import type { OutcomeKey, RunSummary } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import type { PageByOutcome, RunsTableGroup } from '@/hooks/use-runs-table'
import { OUTCOMES, OUTCOME_ORDER } from '@/lib/attribution-data'
import { counted } from '@/lib/format'
import { Badge } from '@/components/ui/badge'
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@/components/ui/empty'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableMessage,
  TableRow,
} from '@/components/ui/table'
import { RouterLink } from '@/components/ui/router-link'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { OutcomeDot } from '@/components/outcome'
import { Identifier, NoValue } from '@/components/field'
import { InfoHint } from '@/components/info-hint'
import { NodeHoverCard } from '@/components/node-hover-card'
import { SectionCard } from '@/components/section-card'
import { TablePagination } from '@/components/table-pagination'
import { TimeHover } from '@/components/time-hover'
import { ActiveFilters, RunsToolbar } from '@/routers/fleet/components/runs-toolbar'
import { attemptGpuHours } from '@/lib/metric'
import { isWindowed } from '@/lib/date-window'
import { focusRing } from '@/lib/utils'
import { useNav } from '@/hooks/use-nav'
import { shouldActivateRow } from '@/lib/row-activation'
import { navigateService } from '@/services/navigate.service'
import { DecisionGraphAction } from '@/routers/verdict/components/decision-graph/dialog'

const COL_COUNT = 8
const PAGE_SIZE_OPTIONS = [10, 25, 50] as const
const RESPONSIBLE_NODE_INLINE_LIMIT = 3

// Up to three responsible nodes as links, the rest behind a +N button. A node
// is a recorded identifier of any length (a cloud node name runs to 40
// characters): its link wraps after its separators, the list onto further
// lines, so neither widens the table.
function ResponsibleNodes({ nodes }: { nodes: string[] }) {
  if (nodes.length === 0) return <NoValue />
  const inline = nodes.slice(0, RESPONSIBLE_NODE_INLINE_LIMIT)
  const rest = nodes.slice(RESPONSIBLE_NODE_INLINE_LIMIT)
  return (
    <ul className="flex flex-wrap items-start gap-x-1.5 gap-y-1">
      {inline.map((node, index) => (
        <li key={node} className="min-w-0">
          <NodeHoverCard nodeId={node}>
            <RouterLink to={navigateService.getPath('node', { id: node })}><Identifier>{node}</Identifier></RouterLink>
          </NodeHoverCard>
          {(index < inline.length - 1 || rest.length > 0) && <span aria-hidden="true">,</span>}
        </li>
      ))}
      {rest.length > 0 && (
        <li>
          {/* A button, so it takes focus and carries its name as a control; it only opens the list. */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge asChild variant="zinc-subtle">
                <button
                  type="button"
                  aria-label={`${counted(rest.length, 'more responsible node', 'more responsible nodes')}: ${rest.join(', ')}`}
                >
                  {`+${rest.length}`}
                </button>
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <ul className="flex flex-col gap-1">
                {rest.map((node) => <li key={node}>{node}</li>)}
              </ul>
            </TooltipContent>
          </Tooltip>
        </li>
      )}
    </ul>
  )
}

function RunRow({ r }: { r: RunSummary }) {
  const nav = useNav()
  return (
    <TableRow
      onClick={(event) => {
        if (shouldActivateRow(event)) nav('verdict', { id: r.id })
      }}
      className="cursor-pointer"
    >
      <TableCell>
        <TimeHover date={r.endTime} cell />
      </TableCell>
      {/* The execution context, user, root symptom and responsible nodes are
          recorded identifiers of any length: they wrap in their columns (the
          context and the nodes as inline links) so the table fits its card
          instead of running past it. */}
      <TableCell wrap>
        <RouterLink to={navigateService.getPath('verdict', { id: r.id })}><Identifier>{r.jobId}</Identifier></RouterLink>
      </TableCell>
      <TableCell wrap>{r.user ? <Identifier>{r.user}</Identifier> : <NoValue />}</TableCell>
      <TableCell className="num text-right">{r.gpuCount}</TableCell>
      <TableCell className="num text-right">{attemptGpuHours(r).toFixed(2)}</TableCell>
      <TableCell wrap>{r.rootSymptom.symptom ? <Identifier>{r.rootSymptom.symptom}</Identifier> : <NoValue />}</TableCell>
      <TableCell wrap>
        <ResponsibleNodes nodes={r.responsibleNodes} />
      </TableCell>
      <TableCell>
        <DecisionGraphAction run={r} compact />
      </TableCell>
    </TableRow>
  )
}

// While rows load, the groups already known for this filter (their headers,
// totals and pagination, which do not depend on the page shown) stay mounted,
// so a page or page-size control keeps focus, and only the rows wait. With no
// known group (a new filter, or the first load), placeholder rows stand in for
// the whole list. The section's count, like the page's, waits for the rows. A
// fetch that failed with no runs to show states no count and says it failed,
// so it never reads as a view with no runs.
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
  failed = false,
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
  /** The runs could not be fetched and none are shown. */
  failed?: boolean
  databaseEmpty?: boolean
}) {
  const [collapsed, setCollapsed] = useState<Set<OutcomeKey>>(new Set())
  const groupIdPrefix = useId()

  const displayGroups = useMemo(
    () =>
      OUTCOME_ORDER.flatMap((key) => {
        const group = groups.find((candidate) => candidate.outcome === key)
        if (!group) return []
        return [{ key, label: OUTCOMES[key].label, ...group }]
      }),
    [groups],
  )

  const windowed = isWindowed(filter)
  const hints = databaseEmpty ? [] : [
    filter.outcome || filter.symptom || filter.node || filter.rule || filter.user || filter.q
      ? 'clear the active filters above'
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
    <SectionCard
      id="runs-section"
      tabIndex={-1}
      title="Runs"
      count={loading || failed ? undefined : matchingRuns.toLocaleString()}
      action={<RunsToolbar filteredRuns={visibleRuns} filter={filter} setFilter={setFilter} />}
      flush
    >
      <ActiveFilters filter={filter} setFilter={setFilter} />
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>End time</TableHead>
            <TableHead>Execution context</TableHead>
            <TableHead>User</TableHead>
            <TableHead className="text-right">GPUs</TableHead>
            <TableHead className="text-right">
              GPU-hours <InfoHint term="gpu-hours" />
            </TableHead>
            <TableHead>Root symptom</TableHead>
            <TableHead>Responsible nodes</TableHead>
            <TableHead>
              <span className="sr-only">Attribution logic</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        {loading && displayGroups.length === 0 ? (
          <TableBody aria-busy="true">
            {Array.from({ length: 6 }, (_, i) => (
              <TableRow key={i}>
                <TableCell colSpan={COL_COUNT}>
                  <Skeleton height="5" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        ) : failed || matchingRuns === 0 ? (
          <TableBody>
            <TableMessage colSpan={COL_COUNT}>
              <Empty size="sm">
                <EmptyHeader>
                  {failed ? (
                    <>
                      <EmptyTitle>Runs could not be loaded</EmptyTitle>
                      <EmptyDescription>Refresh to try again.</EmptyDescription>
                    </>
                  ) : (
                    <>
                      <EmptyTitle>{databaseEmpty ? 'No runs in this database' : 'No runs in this view'}</EmptyTitle>
                      {databaseEmpty ? (
                        <EmptyDescription>Nothing has been attributed yet.</EmptyDescription>
                      ) : hints.length > 0 && (
                        <EmptyDescription>{`Try to ${hints.join(', or ')}.`}</EmptyDescription>
                      )}
                    </>
                  )}
                </EmptyHeader>
              </Empty>
            </TableMessage>
          </TableBody>
        ) : (
          displayGroups.map((g) => {
            const isCollapsed = collapsed.has(g.key)
            const rowsId = `${groupIdPrefix}-${g.key}`
            const pageCount = Math.ceil(g.total / pageSize)
            const page = Math.min(Math.max(pageByOutcome[g.key] ?? 0, 0), pageCount - 1)
            const rowsOnPage = Math.min(pageSize, g.total - page * pageSize)
            return (
              <Fragment key={g.key}>
                {/* The group header: one full-width disclosure button per outcome, on the
                    top of the cell's line, since its leading chevron gives it no text baseline. */}
                <TableBody>
                  <TableRow
                    onClick={(event) => {
                      if (shouldActivateRow(event)) toggleCollapse(g.key)
                    }}
                    className="cursor-pointer bg-background-secondary"
                  >
                    <TableCell colSpan={COL_COUNT}>
                      <button
                        type="button"
                        aria-expanded={!isCollapsed}
                        aria-controls={rowsId}
                        onClick={() => toggleCollapse(g.key)}
                        className={`inline-flex cursor-pointer items-center gap-2 rounded-sm align-top font-medium text-foreground ${focusRing}`}
                      >
                        <ChevronDown
                          aria-hidden="true"
                          className={`size-3.5 text-foreground-muted transition-transform motion-reduce:transition-none ${isCollapsed ? '-rotate-90' : ''}`}
                        />
                        <OutcomeDot outcome={g.key} />
                        {g.label}{' '}
                        <span className="font-normal text-foreground-muted tabular-nums">{g.total.toLocaleString()}</span>
                      </button>
                    </TableCell>
                  </TableRow>
                </TableBody>
                <TableBody id={rowsId} hidden={isCollapsed} aria-busy={loading || undefined}>
                  {loading
                    ? Array.from({ length: rowsOnPage }, (_, i) => (
                        <TableRow key={i}>
                          <TableCell colSpan={COL_COUNT}>
                            <Skeleton height="5" />
                          </TableCell>
                        </TableRow>
                      ))
                    : g.runs.map((r) => <RunRow key={r.id} r={r} />)}
                  {/* The pager stays for any group the smallest page size would
                      page, so choosing a size that fits the whole group keeps
                      the select, and the keyboard focus on it, in place. */}
                  {g.total > PAGE_SIZE_OPTIONS[0] && (
                    <TableRow className="hover:bg-transparent">
                      <TableCell colSpan={COL_COUNT}>
                        <TablePagination
                          aria-label={`${g.label} pages`}
                          label={g.label}
                          itemLabel="runs"
                          controlsId={rowsId}
                          page={page}
                          pageSize={pageSize}
                          total={g.total}
                          onPageChange={(nextPage) => setGroupPage(g.key, nextPage)}
                          pageSizeOptions={PAGE_SIZE_OPTIONS}
                          onPageSizeChange={setPageSize}
                        />
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Fragment>
            )
          })
        )}
      </Table>
    </SectionCard>
  )
}
