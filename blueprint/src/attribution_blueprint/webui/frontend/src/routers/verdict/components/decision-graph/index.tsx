// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { useEffect, useMemo, useState, type ReactNode, type TouchEvent } from 'react'
import {
  Background,
  Controls,
  getViewportForBounds,
  ViewportPortal,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  useReactFlow,
  useStore,
  type NodeProps,
} from '@xyflow/react'
import { ChevronRight, FileText, GitBranch, LocateFixed, RotateCcw, ScanLine } from 'lucide-react'
import type { DecisionGraph, DecisionNode, DecisionNodeType } from '@/interfaces/attribution'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { SymptomDocumentationLink } from '@/components/symptom-documentation'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { layoutGraph, NODE_HEIGHT, NODE_WIDTH, type TraceNode, type GraphStage } from './layout'
import { TraceEdge } from './edge'
import { traceRelationships } from './trace'
import '@xyflow/react/dist/style.css'
import './style.css'

const STAGES: Record<DecisionNodeType, string> = {
  UNKNOWN: 'Unknown stage',
  EXTRACTION: 'Extraction',
  FILTER: 'Filter',
  SCRUB: 'Scrub',
  REFINER: 'Refinement',
  CASCADE_PRUNE: 'Cascade prune',
  OUTCOME: 'Outcome',
}

function stepLabel(step: DecisionNode) {
  return step.rule_name || step.symptom || STAGES[step.node_type]
}

function stepTone(step: DecisionNode) {
  if (step.action === 'REMOVED' || step.action === 'PRUNED') return 'text-red-300 border-red-400/40'
  if (step.action === 'MATCHED') return 'text-emerald-300 border-emerald-400/40'
  return 'text-foreground-secondary border-border'
}

function StepNode({ data: { step }, selected }: NodeProps<TraceNode>) {
  return (
    <div
      className={cn(
        'decision-step flex flex-col gap-1.5 rounded-md border bg-background px-3 py-2.5 shadow-sm transition-colors',
        selected ? 'border-primary-300 ring-1 ring-primary-300/40' : 'border-border hover:border-foreground-muted',
      )}
      style={{ width: NODE_WIDTH, height: NODE_HEIGHT }}
    >
      <Handle type="target" position={Position.Left} className="opacity-0" />
      <div className="flex items-center justify-between gap-2 text-[10px] font-medium">
        <span className={cn('flex items-center gap-1.5', stepTone(step))}>
          <GitBranch className="size-3" />{STAGES[step.node_type]}
        </span>
        <span className={cn('text-[9px] border rounded-sm px-1.5 py-0.5', stepTone(step))}>
          {step.action}
        </span>
      </div>
      <div className={cn('line-clamp-2 break-words text-sm font-medium leading-4', selected ? 'text-primary-300' : 'text-foreground')} title={step.symptom}>
        {step.symptom || 'No symptom recorded'}
      </div>
      <div className="mt-auto space-y-0.5 text-[10px] leading-3 text-foreground-muted">
        <div className="truncate" title={step.rule_name ?? undefined}>{step.rule_name ?? 'No rule recorded'}</div>
        <div className="truncate" title={step.hostname ?? undefined}>{step.hostname ?? 'No host recorded'}</div>
      </div>
      <Handle type="source" position={Position.Right} className="opacity-0" />
    </div>
  )
}

const NODE_TYPES = { decision: StepNode }
const EDGE_TYPES = { decision: TraceEdge }

function keepSingleTouchForScrolling(event: TouchEvent) {
  // Let the dialog own one-finger scrolling; only multi-touch reaches graph pan/zoom.
  if (event.touches.length === 1) event.stopPropagation()
}

function DetailField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="mb-1 text-[11px] text-foreground-muted">{label}</dt>
      <dd className="m-0 break-words text-xs leading-5 text-foreground-secondary">{children}</dd>
    </div>
  )
}

function StepDetails({ step, graph, onSelect }: {
  step: DecisionNode
  graph: DecisionGraph
  onSelect: (id: string) => void
}) {
  const related = graph.edges.flatMap((edge) => {
    const incoming = edge.child_id === step.node_id
    if (!incoming && edge.parent_id !== step.node_id) return []
    const other = graph.nodes.find((node) => node.node_id === (incoming ? edge.parent_id : edge.child_id))!
    return [{ edge, incoming, other }]
  })
  return (
    <aside aria-label="Selected decision step" className="min-w-0 border-t border-border bg-background p-5 lg:min-h-0 lg:overflow-y-auto lg:border-t-0 lg:border-l">
      <div className="mb-4 flex items-center justify-between gap-3">
        <span className="text-[11px] font-medium uppercase tracking-wider text-foreground-muted">Step details</span>
        <span className={cn('rounded-sm border px-2 py-1 text-[10px]', stepTone(step))}>{step.action}</span>
      </div>
      <div className="flex min-w-0 items-center gap-1.5">
        <h3 className="m-0 min-w-0 break-words text-sm font-semibold leading-7">{step.symptom || 'No symptom recorded'}</h3>
        <SymptomDocumentationLink symptomId={step.symptom} />
      </div>
      <dl className="my-5 grid grid-cols-2 gap-x-4 gap-y-4">
        <DetailField label="Stage">{STAGES[step.node_type]}</DetailField>
        <DetailField label="Host">{step.hostname ?? 'Not recorded'}</DetailField>
        <div className="col-span-2"><DetailField label="Rule">{step.rule_name ?? 'Not recorded'}</DetailField></div>
      </dl>
      <div className="mb-5 rounded-md border border-border bg-background-active p-3">
        <div className="mb-1.5 text-[11px] font-medium text-foreground-muted">Recorded reason</div>
        <p className="m-0 whitespace-pre-wrap break-words text-xs leading-5 text-foreground-secondary">{step.reason ?? 'No reason recorded for this step.'}</p>
      </div>
      <div className="border-t border-border pt-4">
        <h4 className="mb-3 flex items-center gap-2 text-xs font-medium"><FileText className="size-3.5 text-foreground-muted" />Source evidence</h4>
        {step.datum != null || step.obs_source != null || step.line_num != null || step.timestamp != null ? (
          <>
            <dl className="mb-3 grid grid-cols-2 gap-3">
              {step.obs_source != null && <DetailField label="Source">{step.obs_source}</DetailField>}
              {step.line_num != null && <DetailField label="Source position"><span className="font-mono">{step.line_num}</span></DetailField>}
              {step.timestamp != null && <div className="col-span-2"><DetailField label="Timestamp"><span className="font-mono">{step.timestamp}</span></DetailField></div>}
            </dl>
            {step.datum != null && <pre className="m-0 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-background-secondary p-3 font-sans text-[11px] leading-5 text-foreground-secondary">{step.datum || 'Empty source datum'}</pre>}
          </>
        ) : <p className="m-0 text-xs leading-5 text-foreground-muted">No source evidence recorded for this step.</p>}
      </div>
      {related.length > 0 && (
        <div className="mt-5 border-t border-border pt-4">
          <h4 className="mb-3 flex items-center justify-between text-xs font-medium">Connected steps <span className="font-mono text-foreground-muted">{related.length}</span></h4>
          <ul className="m-0 list-none divide-y divide-border overflow-hidden rounded-md border border-border p-0">
            {related.map(({ edge, incoming, other }, index) => (
              <li key={index}>
                <Button type="button" variant="ghost" aria-label={`${incoming ? 'From' : 'To'} ${edge.edge_type}: ${stepLabel(other)}, ${STAGES[other.node_type]}, ${other.hostname ?? 'No host recorded'}`} onClick={() => onSelect(other.node_id)} className="h-auto w-full min-w-0 justify-between gap-3 rounded-none border-0 px-3 py-3 text-left whitespace-normal focus-visible:ring-2 focus-visible:ring-inset">
                  <span className="min-w-0 space-y-1">
                    <span className="block break-words text-xs font-medium leading-5">{stepLabel(other)}</span>
                    <span className="block break-words text-[11px] font-normal leading-4 text-foreground-muted">{STAGES[other.node_type]} · {other.hostname ?? 'No host recorded'}</span>
                    <span className="block text-[10px] font-normal leading-4 text-foreground-muted">{incoming ? 'From' : 'To'} · {edge.edge_type}</span>
                  </span>
                  <ChevronRight aria-hidden="true" className="size-3.5 shrink-0 text-foreground-muted" />
                </Button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </aside>
  )
}

function GraphViewport({ stages, nodes, request }: { stages: GraphStage[]; nodes: TraceNode[]; request: { revision: number; nodeId: string | null } }) {
  const { setViewport, viewportInitialized } = useReactFlow()
  const width = useStore((state) => state.width)
  const height = useStore((state) => state.height)
  useEffect(() => {
    if (!viewportInitialized || width === 0 || height === 0) return
    const x = Math.min(...stages.map((stage) => stage.x))
    const y = Math.min(...stages.map((stage) => stage.y))
    const focused = nodes.find((node) => node.id === request.nodeId)
    const bounds = focused ? { ...focused.position, width: NODE_WIDTH, height: NODE_HEIGHT } : {
      x, y,
      width: Math.max(...stages.map((stage) => stage.x + stage.width)) - x,
      height: Math.max(...stages.map((stage) => stage.y + stage.height)) - y,
    }
    void setViewport(getViewportForBounds(bounds, width, height, 0.02, 1, 0.08))
  }, [setViewport, stages, nodes, request, viewportInitialized, width, height])
  return null
}

export function DecisionGraphView({ graph }: { graph: DecisionGraph }) {
  const layout = useMemo(() => layoutGraph(graph), [graph])
  const initialId = graph.nodes.find((node) => node.node_type === 'OUTCOME')?.node_id ?? graph.nodes[0].node_id
  const [selectedId, setSelectedId] = useState(initialId)
  const [viewRequest, setViewRequest] = useState<{ revision: number; nodeId: string | null }>({ revision: 0, nodeId: null })
  const selected = graph.nodes.find((node) => node.node_id === selectedId) ?? graph.nodes[0]
  const trace = useMemo(() => traceRelationships(graph.edges, selected.node_id), [graph.edges, selected.node_id])
  const nodes = useMemo(() => layout.nodes.map((node) => ({
    ...node,
    selected: node.id === selected.node_id,
    className: node.id === selected.node_id ? undefined : trace.nodeIds.has(node.id) ? 'decision-node-related' : 'decision-node-muted',
  })), [layout.nodes, selected.node_id, trace])
  const edges = useMemo(() => layout.edges.map((edge, index) => {
    const highlighted = trace.edgeIndexes.has(index)
    const color = highlighted ? 'var(--color-primary-300)' : 'var(--foreground-muted)'
    return {
      ...edge,
      animated: highlighted,
      markerEnd: { type: MarkerType.ArrowClosed, color },
      style: { ...edge.style, stroke: color, strokeWidth: highlighted ? 2 : 1.25, opacity: highlighted ? 1 : 0.3 },
    }
  }), [layout.edges, trace])

  return (
    <div className="flex shrink-0 flex-col lg:min-h-0 lg:flex-1">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-y border-border bg-background px-5 py-3">
        <div className="flex flex-wrap gap-x-4 gap-y-2 text-[11px] text-foreground-muted" aria-label="Graph legend">
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-3 bg-primary-300" />Selected path</span>
          <span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full bg-foreground-muted" />Recorded evidence</span>
          <span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full bg-emerald-300" />Matched</span>
          <span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full bg-red-300" />Removed / pruned</span>
        </div>
        <div className="flex w-full min-w-0 max-w-full flex-wrap items-center gap-2 sm:w-auto sm:gap-3">
          <div role="group" aria-label="Inspect a step" className="contents sm:flex sm:min-w-0 sm:max-w-full sm:flex-wrap sm:items-center sm:gap-1">
            <Select value={String(graph.nodes.indexOf(selected))} onValueChange={(value) => setSelectedId(graph.nodes[Number(value)].node_id)}>
              <SelectTrigger aria-label="Inspect decision step" className="w-full min-w-0 max-w-full bg-background text-xs text-foreground-secondary sm:w-60 dark:bg-background dark:hover:bg-background-active">
                <SelectValue><span className="truncate">{STAGES[selected.node_type]} · {stepLabel(selected)}</span></SelectValue>
              </SelectTrigger>
              <SelectContent position="popper" align="end" collisionPadding={12} className="max-h-[min(24rem,var(--radix-select-content-available-height))] w-96 max-w-[calc(100vw-24px)]">
                <SelectGroup>
                  {graph.nodes.map((node, index) => (
                    <SelectItem key={node.node_id} value={String(index)} textValue={`${STAGES[node.node_type]} ${stepLabel(node)} ${node.hostname ?? ''}`} className="py-2 [&>span:last-child]:min-w-0">
                      <span className="min-w-0">
                        <span className="mb-1 flex items-center gap-2 text-[10px] text-foreground-muted">
                          {STAGES[node.node_type]} <span className={stepTone(node)}>{node.action}</span>
                        </span>
                        <span className="line-clamp-2 break-words text-[11px] leading-4">{stepLabel(node)}</span>
                        <span className="mt-1 block truncate text-[10px] text-foreground-muted">{node.hostname ?? 'No host recorded'} · {node.symptom || 'No symptom recorded'}</span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
            <Button variant="ghost" className="gap-1.5 text-xs" onClick={() => setViewRequest(({ revision }) => ({ revision: revision + 1, nodeId: selected.node_id }))}>
              <LocateFixed className="size-3.5 text-foreground-muted" strokeWidth={1.5} />Focus step
            </Button>
          </div>
          <div role="group" aria-label="Graph view" className="flex items-center gap-0.5 border-l border-border pl-3">
            <Button variant="ghost" className="gap-1.5 text-xs" onClick={() => setViewRequest(({ revision }) => ({ revision: revision + 1, nodeId: null }))}>
              <ScanLine className="size-3.5 text-foreground-muted" strokeWidth={1.5} />Fit view
            </Button>
            <Button variant="ghost" className="gap-1.5 text-xs" onClick={() => { setSelectedId(initialId); setViewRequest(({ revision }) => ({ revision: revision + 1, nodeId: null })) }}>
              <RotateCcw className="size-3.5 text-foreground-muted" strokeWidth={1.5} />Reset
            </Button>
          </div>
        </div>
      </div>
      <div className="grid items-stretch lg:min-h-0 lg:flex-1 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div
          className="decision-canvas relative min-h-[380px] min-w-0 bg-background-secondary lg:min-h-0"
          aria-label="Decision graph canvas"
          onTouchStartCapture={keepSingleTouchForScrolling}
          onTouchMoveCapture={keepSingleTouchForScrolling}
        >
          <div className="absolute inset-0">
            <ReactFlow<TraceNode>
              nodes={nodes}
              edges={edges}
              nodeTypes={NODE_TYPES}
              edgeTypes={EDGE_TYPES}
              onNodesChange={(changes) => {
                const selection = changes.find((change) => change.type === 'select' && change.selected)
                if (selection?.type === 'select') setSelectedId(selection.id)
              }}
              nodesDraggable={false}
              nodesConnectable={false}
              edgesFocusable={false}
              deleteKeyCode={null}
              multiSelectionKeyCode={null}
              selectionKeyCode={null}
              zoomOnScroll={false}
              preventScrolling={false}
              minZoom={0.02}
              maxZoom={1.5}
              colorMode="dark"
              ariaLabelConfig={{
                'node.a11yDescription.default': 'Press Enter or Space to inspect this step.',
                'node.a11yDescription.keyboardDisabled': 'Press Enter or Space to inspect this step.',
              }}
            >
              <GraphViewport stages={layout.stages} nodes={layout.nodes} request={viewRequest} />
              <ViewportPortal>
                {layout.stages.map((stage) => (
                  <div key={stage.key} className="decision-stage pointer-events-none absolute rounded-xl border border-border/60 bg-background/25" style={{ left: stage.x, top: stage.y, width: stage.width, height: stage.height }}>
                    <div className="flex items-center justify-between px-4 py-3 text-xs font-medium text-foreground-secondary">
                      <span>{stage.title}</span><span className="font-mono text-foreground-muted">{stage.count}</span>
                    </div>
                  </div>
                ))}
              </ViewportPortal>
              <Background color="var(--border)" gap={20} size={1} />
              <Controls showInteractive={false} showFitView={false} position="bottom-left" />
            </ReactFlow>
          </div>
        </div>
        <StepDetails step={selected} graph={graph} onSelect={setSelectedId} />
      </div>
      <div className="flex shrink-0 flex-wrap justify-between gap-2 border-t border-border bg-background px-5 py-2.5 text-[11px] leading-4 text-foreground-muted">
        <span>Arrows show recorded relationships. Extraction does not imply survival.</span>
        <span className="decision-pointer-hint">Drag to pan · Use + / − to zoom</span>
        <span className="decision-touch-hint">Swipe to scroll · Two fingers to pan · + / − to zoom</span>
      </div>
    </div>
  )
}
