// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useParams } from 'react-router-dom'
import { TriangleAlert } from 'lucide-react'
import type { RunSummary } from '@/interfaces/attribution'
import { counted } from '@/lib/format'
import { nodeGpuHours } from '@/lib/metric'
import { summarizeNode } from '@/lib/node-summary'
import { buildNodeMarkdown } from '@/lib/evidence'
import { downloadBlob, nodeExportData, slug, toYaml, utcDateStamp } from '@/lib/export'
import { shouldActivateRow } from '@/lib/row-activation'
import { useNodeRuns } from '@/hooks/use-node-runs'
import { useNav } from '@/hooks/use-nav'
import { useDocumentTitle } from '@/hooks/use-document-title'
import { navigateService } from '@/services/navigate.service'
import { CopyButton } from '@/components/copy-button'
import { ExportMenu } from '@/components/export-menu'
import { Identifier, NoValue } from '@/components/field'
import { InfoHint } from '@/components/info-hint'
import { OutcomeBadge } from '@/components/outcome'
import { PageHeader } from '@/components/page-header'
import { SectionCard } from '@/components/section-card'
import { SiteStatusBadge } from '@/components/site-status-badge'
import { TimeHover } from '@/components/time-hover'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Copyable } from '@/components/ui/copyable'
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@/components/ui/empty'
import { RouterLink } from '@/components/ui/router-link'
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

// Every fact on the page is derived from the node's runs, so none is stated
// before they arrive or when they could not be fetched: a pending or failed
// fetch must not read as a node with no attempts.
type RunsState = 'loading' | 'failed' | 'ready'

const PLACEHOLDER_ROWS = 3

// What a table body shows in place of its rows: pulsing rows while the runs
// load, otherwise one row saying why there are none.
function PlaceholderRows({
  state,
  columns,
  emptyTitle,
  emptyDescription,
}: {
  state: RunsState
  columns: number
  emptyTitle: string
  emptyDescription: string
}) {
  if (state === 'loading') {
    return Array.from({ length: PLACEHOLDER_ROWS }, (_, row) => (
      <TableRow key={row}>
        {Array.from({ length: columns }, (_, column) => (
          <TableCell key={column}>
            <Skeleton height="5" />
          </TableCell>
        ))}
      </TableRow>
    ))
  }
  return (
    <TableMessage colSpan={columns}>
      <Empty size="sm">
        <EmptyHeader>
          {state === 'failed' ? (
            <EmptyTitle>Runs for this node could not be loaded</EmptyTitle>
          ) : (
            <>
              <EmptyTitle>{emptyTitle}</EmptyTitle>
              <EmptyDescription>{emptyDescription}</EmptyDescription>
            </>
          )}
        </EmptyHeader>
      </Empty>
    </TableMessage>
  )
}

export function NodeView() {
  const { id: nodeId = '' } = useParams<{ id: string }>()
  const nav = useNav()
  const { runs, isLoading, error } = useNodeRuns(nodeId)
  useDocumentTitle('Node', nodeId)
  // A failed revalidation keeps showing the runs already fetched.
  const state: RunsState = isLoading ? 'loading' : error !== null && runs.length === 0 ? 'failed' : 'ready'
  const ready = state === 'ready'

  const isResponsible = (run: RunSummary) => run.responsibleNodes.includes(nodeId)
  // The notice names the node's symptom only when one stands out
  // (summarizeNode); at a tie it counts them instead.
  const { attempts, responsible: responsibleAttempts, symptoms, leading, gpuHours: gpuHoursOnNode } = summarizeNode(runs, nodeId)

  const download = (extension: string, text: string, type: string) =>
    downloadBlob(`node-${slug(nodeId)}-${utcDateStamp()}.${extension}`, text, type)

  // Loading, the head is already as tall as it will be, so it keeps its
  // height when the runs arrive: the actions' places hold 28px blocks and the
  // facts keep their labels over placeholders. A failed fetch drops both.
  return (
    <div className="flex flex-col gap-page-gap">
      <PageHeader
        breadcrumb={[{ label: 'Fleet', to: navigateService.getPath('fleet') }, { label: 'Node' }]}
        title={<Identifier>{nodeId}</Identifier>}
        titleAddon={
          <>
            <Copyable text={nodeId} label="Copy node ID" iconOnly />
            {responsibleAttempts.length > 0 && <SiteStatusBadge status="bad" />}
          </>
        }
        actions={
          state === 'loading' ? (
            <>
              <Skeleton height="7" className="w-32" />
              <Skeleton height="7" className="w-24" />
            </>
          ) : ready ? (
            <>
              <CopyButton
                label="Copy evidence"
                tooltip="Copies a Markdown summary"
                getText={() => buildNodeMarkdown(nodeId, attempts)}
              />
              <ExportMenu
                triggerLabel="Export"
                items={[
                  {
                    label: 'Download as JSON',
                    onSelect: () =>
                      download('json', JSON.stringify(nodeExportData(nodeId, attempts), null, 2), 'application/json'),
                  },
                  {
                    label: 'Download as YAML',
                    onSelect: () => download('yaml', toYaml(nodeExportData(nodeId, attempts)), 'application/yaml'),
                  },
                  {
                    label: 'Download as Markdown',
                    onSelect: () => download('md', buildNodeMarkdown(nodeId, attempts), 'text/markdown'),
                  },
                ]}
              />
            </>
          ) : undefined
        }
        meta={
          state === 'failed'
            ? undefined
            : [
                { key: 'attempts', label: 'Attempts', value: <span className="num">{attempts.length}</span> },
                {
                  key: 'responsible-in',
                  label: <>Responsible in <InfoHint term="responsible-in" /></>,
                  value: <span className="num">{responsibleAttempts.length}</span>,
                },
                {
                  key: 'distinct-symptoms',
                  label: <>Distinct symptoms <InfoHint term="distinct-symptoms" /></>,
                  value: <span className="num">{symptoms.length}</span>,
                },
                {
                  key: 'gpu-hours-on-node',
                  label: <>GPU-hours on node <InfoHint term="gpu-hours-on-node" /></>,
                  value: <span className="num">{gpuHoursOnNode.toFixed(2)}</span>,
                },
              ]
        }
        metaLoading={state === 'loading'}
      />

      <div className="flex flex-col gap-card-gap">
        {responsibleAttempts.length > 0 && (
          <Alert variant="destructive">
            <TriangleAlert />
            <AlertTitle>Responsible node</AlertTitle>
            <AlertDescription>
              {leading ? (
                <>
                  {leading[1] > 1 && 'Recurring '}
                  <span className="text-foreground"><Identifier>{leading[0]}</Identifier></span> attributed to this node.{' '}
                </>
              ) : symptoms.length > 1 && `${counted(symptoms.length, 'symptom', 'symptoms')} attributed to this node. `}
              Consider escalating to the cloud operator with the evidence.
            </AlertDescription>
          </Alert>
        )}

        <SectionCard title="Symptoms on this node" count={ready ? symptoms.length : undefined} flush>
          <Table aria-busy={state === 'loading' || undefined}>
            <TableHeader>
              <TableRow>
                <TableHead>Symptom</TableHead>
                <TableHead className="text-right">Attempts</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ready && symptoms.length > 0 ? (
                symptoms.map(([name, count]) => (
                  <TableRow key={name}>
                    <TableCell wrap>{name === '' ? <NoValue /> : <Identifier>{name}</Identifier>}</TableCell>
                    <TableCell className="num text-right">{count}</TableCell>
                  </TableRow>
                ))
              ) : (
                <PlaceholderRows
                  state={state}
                  columns={2}
                  emptyTitle="No attributed symptoms"
                  emptyDescription="No recorded attempt lists this node as responsible."
                />
              )}
            </TableBody>
          </Table>
        </SectionCard>

        <SectionCard title="Attempts" count={ready ? attempts.length : undefined} flush>
          <Table aria-busy={state === 'loading' || undefined}>
            <TableHeader>
              <TableRow>
                <TableHead>End time</TableHead>
                <TableHead>Execution context</TableHead>
                <TableHead>User</TableHead>
                <TableHead>Outcome</TableHead>
                <TableHead>Responsible</TableHead>
                <TableHead className="text-right">
                  GPU-hours on node <InfoHint term="gpu-hours-on-node" />
                </TableHead>
                <TableHead>Root symptom</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ready && attempts.length > 0 ? (
                attempts.map((r) => (
                  <TableRow
                    key={r.id}
                    className="cursor-pointer"
                    onClick={(event) => {
                      if (shouldActivateRow(event)) nav('verdict', { id: r.id })
                    }}
                  >
                    <TableCell>
                      <TimeHover date={r.endTime} cell />
                    </TableCell>
                    {/* A recorded identifier of any length: it wraps in its column as an inline link, as on Fleet. */}
                    <TableCell wrap>
                      <RouterLink to={navigateService.getPath('verdict', { id: r.id })}><Identifier>{r.jobId}</Identifier></RouterLink>
                    </TableCell>
                    <TableCell wrap>{r.user ? <Identifier>{r.user}</Identifier> : <NoValue />}</TableCell>
                    <TableCell>
                      <OutcomeBadge outcome={r.outcome} />
                    </TableCell>
                    <TableCell>{isResponsible(r) ? <SiteStatusBadge status="bad" /> : <NoValue />}</TableCell>
                    <TableCell className="num text-right">{nodeGpuHours(r, nodeId).toFixed(2)}</TableCell>
                    <TableCell wrap>{r.rootSymptom.symptom === '' ? <NoValue /> : <Identifier>{r.rootSymptom.symptom}</Identifier>}</TableCell>
                  </TableRow>
                ))
              ) : (
                <PlaceholderRows
                  state={state}
                  columns={7}
                  emptyTitle="No recorded attempts"
                  emptyDescription="No recorded run lists this node."
                />
              )}
            </TableBody>
          </Table>
        </SectionCard>
      </div>
    </div>
  )
}
