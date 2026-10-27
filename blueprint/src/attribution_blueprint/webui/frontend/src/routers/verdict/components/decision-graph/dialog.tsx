// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Component, lazy, Suspense, useId, useState, type ReactNode } from 'react'
import { GitBranch, LoaderCircle } from 'lucide-react'
import useSWR from 'swr'
import type { RunDetail, RunSummary } from '@/interfaces/attribution'
import { useApi } from '@/services/api-context'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { Identifier } from '@/components/field'
import { OutcomeBadge } from '@/components/outcome'
import { cn, dividedRow, gapDot, markInRow } from '@/lib/utils'

const DecisionGraphView = lazy(() => import('./index').then((module) => ({ default: module.DecisionGraphView })))

const ACTION_LABEL = 'View attribution logic'

const UNAVAILABLE: Record<Exclude<RunSummary['decisionGraphState'], 'available'>, string> = {
  absent: 'No decision graph was recorded for this attribution.',
  null: 'No decision graph was saved with this attribution.',
  empty: 'This attribution has no recorded decision steps.',
  invalid: 'The recorded decision graph cannot be displayed.',
}

function LoadingGraph() {
  return <div role="status" className="flex min-h-60 flex-1 items-center justify-center gap-2 text-sm text-foreground-muted"><LoaderCircle className={cn('size-4 animate-spin', markInRow)} />Loading attribution logic…</div>
}

class GraphErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <div role="alert" className="flex min-h-60 flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-sm text-foreground-secondary">The graph viewer could not be displayed. You can close this window to return to the attribution.</p>
        <Button onClick={() => window.location.reload()}>Reload page</Button>
      </div>
    )
  }
}

function GraphBody({ run, error, isLoading, retry }: { run?: RunDetail; error?: Error; isLoading: boolean; retry: () => void }) {
  if (!run) {
    if (isLoading) return <LoadingGraph />
    return (
      <div role="status" className="flex min-h-60 flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-sm text-foreground-secondary">{error ? 'Attribution logic could not be loaded.' : 'This attribution is no longer available.'}</p>
        <Button onClick={retry}>Try again</Button>
      </div>
    )
  }
  const section = run.attribution.decisionGraph
  if (section.state !== 'available') {
    return <div role="status" className="flex min-h-60 flex-1 items-center justify-center px-6 text-sm text-foreground-muted">{UNAVAILABLE[section.state]}</div>
  }
  return (
    <Suspense fallback={<LoadingGraph />}>
      <DecisionGraphView graph={section.data} />
    </Suspense>
  )
}

// The header names the attribution the logic belongs to as the verdict's
// answer card does (execution context, outcome and root symptom), on the
// title's line: one 44px row, the compact dialog header, inset 16px like the
// toolbar under it. Only the graph viewer sits inside the error boundary, so
// the header and the close button survive a viewer failure.
function GraphDialogContent({ run, detail }: { run: RunSummary; detail?: RunDetail }) {
  const api = useApi()
  const { data, error, isLoading, mutate } = useSWR<RunDetail | undefined, Error>(
    detail ? null : ['run', run.id],
    () => api.getRun(run.id),
    { revalidateIfStale: false, shouldRetryOnError: false },
  )
  const loaded = detail ?? data
  const root = loaded?.attribution.rootSymptom
  const rootName = root?.state === 'available' ? root.data.symptom : run.rootSymptom.symptom
  return (
    <>
      <DialogHeader className="min-h-11 shrink-0 flex-row flex-wrap items-center gap-x-3 gap-y-1 border-b border-border py-1.5 pr-12 pl-4">
        <DialogTitle>Attribution logic</DialogTitle>
        {/* The dot between the job and its verdict is drawn in the gap before
            the verdict, so it never ends a line the verdict wraps from. */}
        <DialogDescription className={cn('flex flex-wrap items-center gap-x-4 gap-y-1', dividedRow)}>
          <span className="min-w-0 break-words"><Identifier>{run.jobId}</Identifier></span>{' '}
          <span className={gapDot[16]}><OutcomeBadge outcome={run.outcome} label={run.verdictLabel} /></span>{' '}
          {rootName
            ? <span className="min-w-0 break-words"><Identifier>{rootName}</Identifier></span>
            : <span className="text-foreground-muted">Root symptom not recorded</span>}
        </DialogDescription>
      </DialogHeader>
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto lg:overflow-hidden">
        <GraphErrorBoundary>
          <GraphBody run={loaded} error={error} isLoading={isLoading} retry={() => void mutate()} />
        </GraphErrorBoundary>
      </div>
    </>
  )
}

// Unavailable, the action says why: in its tooltip, and as its description,
// kept in a hidden node so it is read without hover. The compact form in a
// table row also leaves the tab order, so a list of runs without graphs is not
// a list of dead stops; pointer and screen reader users still reach its reason.
// That 24px icon button takes -2px block margins on the top of the cell's
// line, so its row stays 28px; an icon has no baseline to sit on. The full
// form, on the verdict, is that page's one primary action: the green default.
export function DecisionGraphAction({ run, detail, compact = false }: { run: RunSummary; detail?: RunDetail; compact?: boolean }) {
  const [showTooltip, setShowTooltip] = useState(false)
  const reasonId = useId()
  const available = run.decisionGraphState === 'available'
  const reason = available ? undefined : UNAVAILABLE[run.decisionGraphState as keyof typeof UNAVAILABLE]
  // Set only when unavailable: an available trigger keeps the description its
  // tooltip gives it, which an explicit undefined here would override.
  const unavailableProps = available ? {} : { 'aria-describedby': reasonId, ...(compact && { tabIndex: -1 }) }
  const action = (
    <Button
      variant={compact ? 'ghost' : 'default'}
      size={compact ? 'icon-sm' : 'default'}
      aria-label={compact ? ACTION_LABEL : undefined}
      aria-disabled={!available}
      {...unavailableProps}
      onClick={(event) => {
        if (!available) {
          // Keep the reason accessible on touch, where there is no hover.
          event.preventDefault()
          setShowTooltip(true)
        }
      }}
      className={cn(compact && '-my-0.5 align-top')}
    >
      <GitBranch data-icon={compact ? undefined : 'inline-start'} />
      {!compact && ACTION_LABEL}
    </Button>
  )
  const trigger = available ? <DialogTrigger asChild>{action}</DialogTrigger> : action

  return (
    // React bubbles events from the portaled tooltip and dialog through this
    // component, so every click stops here and a clickable row around the
    // action never activates. The wrapper draws no box of its own.
    <span className="contents" onClick={(event) => event.stopPropagation()}>
      <Dialog>
        {available && !compact ? trigger : (
          <Tooltip open={showTooltip} onOpenChange={setShowTooltip}>
            <TooltipTrigger asChild>{trigger}</TooltipTrigger>
            <TooltipContent side="top" sideOffset={6}>{reason ?? ACTION_LABEL}</TooltipContent>
          </Tooltip>
        )}
        {reason !== undefined && <span id={reasonId} hidden>{reason}</span>}
        {/* The panel itself takes focus as it opens, not its first control: the
            graph loads after the panel the first time, so that control would
            be the close button then and the step list every time after. */}
        <DialogContent
          closeLabel="Close attribution logic"
          onOpenAutoFocus={(event) => {
            event.preventDefault()
            ;(event.currentTarget as HTMLElement | null)?.focus()
          }}
        >
          <GraphDialogContent run={run} detail={detail} />
        </DialogContent>
      </Dialog>
    </span>
  )
}
