// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { Component, lazy, Suspense, useState, type ReactNode } from 'react'
import { Dialog } from 'radix-ui'
import { GitBranch, LoaderCircle, X } from 'lucide-react'
import useSWR from 'swr'
import type { RunDetail, RunSummary } from '@/interfaces/attribution'
import { useApi } from '@/services/api-context'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

const DecisionGraphView = lazy(() => import('./index').then((module) => ({ default: module.DecisionGraphView })))

const UNAVAILABLE: Record<Exclude<RunSummary['decisionGraphState'], 'available'>, string> = {
  absent: 'No decision graph was recorded for this attribution.',
  null: 'No decision graph was saved with this attribution.',
  empty: 'This attribution has no recorded decision steps.',
  invalid: 'The recorded decision graph cannot be displayed.',
}

function LoadingGraph() {
  return <div role="status" className="flex min-h-60 flex-1 items-center justify-center gap-2 text-sm text-foreground-muted"><LoaderCircle className="size-4 animate-spin" />Loading attribution logic…</div>
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
        <p className="m-0 text-sm text-foreground-secondary">The graph viewer could not be displayed. You can close this window to return to the attribution.</p>
        <Button variant="outline" onClick={() => window.location.reload()}>Reload page</Button>
      </div>
    )
  }
}

function GraphContent({ runId, detail }: { runId: string; detail?: RunDetail }) {
  const api = useApi()
  const { data, error, isLoading, mutate } = useSWR<RunDetail | undefined, Error>(
    detail ? null : ['run', runId],
    () => api.getRun(runId),
    { revalidateIfStale: false, shouldRetryOnError: false },
  )
  const run = detail ?? data
  if (!run) {
    if (isLoading) return <LoadingGraph />
    return (
      <div role="status" className="flex min-h-60 flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="m-0 text-sm text-foreground-secondary">{error ? 'Attribution logic could not be loaded.' : 'This attribution is no longer available.'}</p>
        <Button variant="outline" onClick={() => void mutate()}>Try again</Button>
      </div>
    )
  }
  const section = run.attribution.decisionGraph
  if (section.state !== 'available') {
    return <div role="status" className="flex min-h-60 flex-1 items-center justify-center px-6 text-sm text-foreground-muted">{UNAVAILABLE[section.state]}</div>
  }

  return (
    <>
      <div className="shrink-0 px-5 py-4 sm:px-6">
        <p className="m-0 text-sm leading-6 text-foreground">
          FACT attributed this run to <strong className="font-medium">{run.verdictLabel}</strong>.
          {run.rootSymptom.symptom && <> The recorded root symptom is <span className="font-medium">{run.rootSymptom.symptom}</span>.</>}
        </p>
        <div className="mt-2 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2 text-xs text-foreground-muted">
          {run.detail && <span className="min-w-0 break-words">Service detail <span className="ml-1.5 text-foreground-secondary">{run.detail}</span></span>}
          <span className="shrink-0"><span className="font-mono">{section.data.nodes.length}</span> steps · <span className="font-mono">{section.data.edges.length}</span> relationships</span>
        </div>
      </div>
      <Suspense fallback={<LoadingGraph />}>
        <DecisionGraphView graph={section.data} />
      </Suspense>
    </>
  )
}

export function DecisionGraphAction({ run, detail, compact = false }: { run: RunSummary; detail?: RunDetail; compact?: boolean }) {
  const [open, setOpen] = useState(false)
  const [showTooltip, setShowTooltip] = useState(false)
  const available = run.decisionGraphState === 'available'
  const action = (
    <Button
      type="button"
      variant={compact ? 'ghost' : 'outline'}
      size={compact ? 'icon-sm' : 'default'}
      aria-label="View Attribution Logic"
      aria-disabled={!available}
      onClick={(event) => {
        event.stopPropagation()
        if (!available) {
          // Keep the reason accessible on touch, where there is no hover.
          event.preventDefault()
          setShowTooltip(true)
        }
      }}
      className={cn(
        !compact && 'w-full gap-2 px-3 font-sans text-xs font-medium',
        !available && 'cursor-default text-foreground-muted opacity-60 aria-disabled:hover:bg-transparent aria-disabled:hover:text-foreground-muted',
      )}
    >
      <GitBranch strokeWidth={1.5} />
      {!compact && 'View Attribution Logic'}
    </Button>
  )

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <span className={cn('flex', compact ? 'w-fit' : 'w-full')} onClick={(event) => event.stopPropagation()}>
        {available && !compact ? <Dialog.Trigger asChild>{action}</Dialog.Trigger> : (
          <TooltipProvider>
            <Tooltip open={showTooltip} onOpenChange={setShowTooltip}>
              <TooltipTrigger asChild>{available ? <Dialog.Trigger asChild>{action}</Dialog.Trigger> : action}</TooltipTrigger>
              <TooltipContent side="top" sideOffset={6}>{available ? 'View Attribution Logic' : UNAVAILABLE[run.decisionGraphState as keyof typeof UNAVAILABLE]}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </span>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60 data-[state=open]:animate-in data-[state=open]:fade-in-0 motion-reduce:animate-none" />
        <Dialog.Content
          className="fixed top-1/2 left-1/2 z-50 flex h-[min(880px,92dvh)] w-[calc(100vw-24px)] max-w-[1440px] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-2xl outline-none sm:w-[calc(100vw-64px)]"
          onClick={(event) => event.stopPropagation()}
        >
          <div className="flex shrink-0 items-start justify-between gap-4 border-b border-border px-5 py-4 sm:px-6">
            <div className="min-w-0">
              <Dialog.Title className="m-0 flex items-center gap-2 text-base font-semibold tracking-tight"><GitBranch className="size-4 text-primary-300" />Attribution logic</Dialog.Title>
              <Dialog.Description className="mt-1 mb-0 truncate text-xs text-foreground-muted">{run.jobId} · attempt <span className="font-mono">{run.attempt}</span></Dialog.Description>
            </div>
            <Dialog.Close asChild><Button variant="ghost" size="icon" aria-label="Close attribution logic" className="text-foreground-muted"><X className="size-4" strokeWidth={1.5} /></Button></Dialog.Close>
          </div>
          <div className="flex min-h-0 flex-1 flex-col overflow-y-auto lg:overflow-hidden">
            {open && <GraphErrorBoundary><GraphContent runId={run.id} detail={detail} /></GraphErrorBoundary>}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
