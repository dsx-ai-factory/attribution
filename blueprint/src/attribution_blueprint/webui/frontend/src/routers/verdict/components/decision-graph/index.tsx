// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useEffect, useId, useMemo, useRef, useState, type TouchEvent } from 'react'
import { flushSync } from 'react-dom'
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
  useStoreApi,
  type Edge,
  type NodeProps,
  type Rect,
} from '@xyflow/react'
import { Check, ChevronRight, LocateFixed, Minus, RotateCcw, ScanLine } from 'lucide-react'
import type { DecisionAction, DecisionGraph, DecisionNode } from '@/interfaces/attribution'
import { counted } from '@/lib/format'
import { cn, dividedRow, gapHairline, markInRow, subtitle } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  CodeBlock,
  CodeBlockActions,
  CodeBlockBody,
  CodeBlockCopyButton,
  CodeBlockHeader,
  CodeBlockTitle,
} from '@/components/ui/code-block'
import { DescriptionList, type DescriptionListItem } from '@/components/ui/description-list'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Identifier, NotRecorded } from '@/components/field'
import { SymptomDocumentationLink } from '@/components/symptom-documentation'
import { TimeHover } from '@/components/time-hover'
import { edgeBoxes, enclose, layoutGraph, NODE_HEIGHT, NODE_WIDTH, type GraphStage, type TraceNode } from './layout'
import { ACTION_LABELS, edgeLabel, STAGE_LABELS, stepLabel } from './labels'
import { TraceEdge } from './edge'
import { onTrace, traceRelationships } from './trace'
import '@xyflow/react/dist/style.css'
import './style.css'

const LEGEND_ACTIONS: DecisionAction[] = ['CREATED', 'MATCHED', 'REMOVED', 'PRUNED']

// Fit view may shrink a large graph this far to show all of it.
const MIN_ZOOM = 0.02
// The opening view fits the selected chain but stays at or above this zoom,
// where card text is still readable; a longer chain is cut at its far end.
const TRACE_MIN_ZOOM = 0.75
const FIT_PADDING = 0.08
// How far inside the canvas edge the view brings a step it had left partly off.
const ANCHOR_MARGIN = 24
// The focus ring drawn around a focused step (style.css), in graph units: the
// step is in view only with its ring.
export const FOCUS_RING = 3
// The zoom controls' corner strip at the canvas bottom (style.css: two 24px
// buttons in a 1px frame, 16px in from the edges). Every view is fitted to the
// canvas above it, so what a view is fitted to (the chain it opens on, the
// step it focuses, or the whole graph) never opens under the controls; a step
// outside it, past the chain, may.
export const CONTROLS_INSET = 16 + 2 * 24 + 2

// Actions are neutral: red means blame and green means the selected path, so a
// discarded step is told apart by a dashed outline and a minus, a match by a check.
function ActionBadge({ action }: { action: DecisionAction }) {
  const label = ACTION_LABELS[action]
  if (action === 'MATCHED') return <Badge variant="outline" className="text-foreground"><Check aria-hidden="true" />{label}</Badge>
  if (action === 'REMOVED' || action === 'PRUNED') return <Badge variant="dashed"><Minus aria-hidden="true" />{label}</Badge>
  return <Badge variant="zinc-subtle">{label}</Badge>
}

// A step's rule and host on one line, each shortened on its own; the inspector
// shows them in full. Values that were not recorded are left out.
function StepMeta({ step }: { step: DecisionNode }) {
  const values = [step.rule_name, step.hostname].filter((value): value is string => Boolean(value))
  return (
    <div className="flex min-w-0 items-center gap-1.5 text-xs text-foreground-muted">
      {values.length === 0 ? <span className="truncate">No rule or host recorded</span> : values.map((value, index) => (
        <Fragment key={index}>
          {index > 0 && <span aria-hidden="true">·</span>}
          <span className="min-w-0 truncate" title={value}>{value}</span>
        </Fragment>
      ))}
    </div>
  )
}

// 248 × 112: 8px padding, a 24px stage row, two 20px symptom lines and a 16px
// meta line, 6px apart (108px). The symptom's region takes the free height, so
// the meta line sits on the card's bottom edge whether the symptom has one line
// or two; the clamped text inside keeps its own height.
function StepNode({ data: { step }, selected }: NodeProps<TraceNode>) {
  return (
    <div
      className={cn(
        'decision-step flex flex-col gap-1.5 rounded-md border bg-background px-3 py-2 transition-colors',
        selected ? 'border-graph-trace ring-1 ring-graph-trace/40' : 'border-border hover:border-foreground-muted',
      )}
      style={{ width: NODE_WIDTH, height: NODE_HEIGHT }}
    >
      <Handle type="target" position={Position.Left} className="opacity-0" />
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs text-foreground-muted">{STAGE_LABELS[step.node_type]}</span>
        <ActionBadge action={step.action} />
      </div>
      <div className="min-h-0 flex-1">
        <div className="line-clamp-2 text-sm leading-5 font-medium break-words text-foreground" title={step.symptom}>
          {step.symptom ? <Identifier>{step.symptom}</Identifier> : 'No symptom recorded'}
        </div>
      </div>
      <StepMeta step={step} />
      <Handle type="source" position={Position.Right} className="opacity-0" />
    </div>
  )
}

// A relationship as assistive technology reads it (layout edges): named by
// its kind and steps, drawing nothing, as its route's pieces draw it.
function RelationshipEdge() {
  return null
}

const NODE_TYPES = { decision: StepNode }
const EDGE_TYPES = { decision: TraceEdge, relationship: RelationshipEdge }

function keepSingleTouchForScrolling(event: TouchEvent) {
  // Let the dialog own one-finger scrolling; only multi-touch reaches graph pan/zoom.
  if (event.touches.length === 1) event.stopPropagation()
}

// Where the step was read from, one fact per row on the panel's one label
// axis, as the facts above it are: the 96px labels leave values 188px, room
// for a full timestamp, which never wraps. The recorded time is the time the
// evidence line was recorded, the same value a symptom site's source carries,
// so it has the same label as in the Symptoms table. A fact the step does not
// carry is left out.
function SourceEvidence({ step }: { step: DecisionNode }) {
  const datumTitleId = useId()
  const facts: DescriptionListItem[] = [
    ...(step.obs_source != null ? [{ key: 'source', label: 'Source', value: step.obs_source ? <Identifier>{step.obs_source}</Identifier> : <NotRecorded /> }] : []),
    ...(step.line_num != null ? [{ key: 'position', label: 'Source position', value: <span className="num">{step.line_num}</span> }] : []),
    ...(step.timestamp != null ? [{ key: 'time', label: 'Recorded time', value: <TimeHover date={step.timestamp} /> }] : []),
  ]
  if (facts.length === 0 && step.datum == null) {
    return <p className="text-sm text-foreground-muted">No source evidence recorded for this step.</p>
  }
  return (
    <>
      {facts.length > 0 && <DescriptionList labelWidth="sm" items={facts} />}
      {step.datum === '' && <p className="text-sm text-foreground-muted">Empty source datum</p>}
      {step.datum && (
        <CodeBlock code={step.datum}>
          <CodeBlockHeader>
            <CodeBlockTitle id={datumTitleId}>Source datum</CodeBlockTitle>
            <CodeBlockActions><CodeBlockCopyButton label="Copy source datum" /></CodeBlockActions>
          </CodeBlockHeader>
          <CodeBlockBody labelledBy={datumTitleId} wrap className="max-h-60" />
        </CodeBlock>
      )}
    </>
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
  const titleRef = useRef<HTMLHeadingElement>(null)
  // Following a connected step replaces the panel, and the step followed to may
  // have no row where the pressed one was. Focus moves to the new step's title,
  // so a keyboard or screen reader user lands on what they asked to inspect.
  const follow = (id: string) => {
    flushSync(() => onSelect(id))
    titleRef.current!.focus()
  }
  return (
    <aside aria-label="Selected decision step" className="flex min-w-0 flex-col gap-3 border-t border-border bg-background p-3 lg:min-h-0 lg:overflow-y-auto lg:border-t-0 lg:border-l">
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-3">
          <span className="text-xs text-foreground-muted">{STAGE_LABELS[step.node_type]}</span>
          <ActionBadge action={step.action} />
        </div>
        <div className="flex min-w-0 items-start gap-2">
          <h3 ref={titleRef} tabIndex={-1} className={cn(subtitle, 'min-w-0 break-words outline-none')}>{step.symptom ? <Identifier>{step.symptom}</Identifier> : 'No symptom recorded'}</h3>
          <SymptomDocumentationLink symptomId={step.symptom} />
        </div>
      </div>
      <DescriptionList labelWidth="sm" items={[
        { key: 'host', label: 'Host', value: step.hostname ? <Identifier>{step.hostname}</Identifier> : <NotRecorded /> },
        { key: 'rule', label: 'Rule', value: step.rule_name ? <Identifier>{step.rule_name}</Identifier> : <NotRecorded /> },
        { key: 'reason', label: 'Reason', value: step.reason ? <span className="whitespace-pre-wrap"><Identifier>{step.reason}</Identifier></span> : <NotRecorded /> },
      ]} />
      <div className="flex flex-col gap-2 border-t border-border pt-3">
        <h4 className={subtitle}>Source evidence</h4>
        <SourceEvidence step={step} />
      </div>
      {related.length > 0 && (
        <div className="flex flex-col gap-2 border-t border-border pt-3">
          <div className="flex items-baseline gap-2">
            <h4 className={subtitle}>Connected steps</h4>
            <span className="text-sm text-foreground-muted tabular-nums">{related.length}</span>
          </div>
          <ul className="flex flex-col divide-y divide-border overflow-hidden rounded-md border border-border">
            {related.map(({ edge, incoming, other }, index) => {
              const direction = incoming ? 'From' : 'To'
              const relationship = edgeLabel(edge.edge_type)
              return (
                <li key={index}>
                  <Button
                    variant="ghost"
                    aria-label={`${direction} ${relationship}: ${stepLabel(other)}, ${STAGE_LABELS[other.node_type]}, ${other.hostname ?? 'No host recorded'}`}
                    onClick={() => follow(other.node_id)}
                    className="h-auto w-full justify-between gap-3 rounded-none px-3 py-2 text-left whitespace-normal focus-visible:ring-inset"
                  >
                    {/* Recorded names break after their separators; their lines
                        balance their ends (text-pretty, as the house alert's),
                        so no host ends a line on one letter ("demo-node-" / "b"). */}
                    <span className="flex min-w-0 flex-col">
                      <span className="break-words text-pretty text-foreground"><Identifier>{stepLabel(other)}</Identifier></span>
                      <span className="text-xs break-words text-pretty text-foreground-muted">
                        <Identifier>{[direction, relationship, STAGE_LABELS[other.node_type], other.hostname].filter(Boolean).join(' · ')}</Identifier>
                      </span>
                    </span>
                    <ChevronRight aria-hidden="true" />
                  </Button>
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </aside>
  )
}

interface ViewRequest {
  revision: number
  /** The part of the graph to fit. */
  bounds: Rect
  minZoom: number
  /** A step kept on screen when the bounds overflow the canvas at minZoom. */
  anchor?: Rect
}

// The offset along one axis that keeps a box (its start and size on the
// canvas, at the view's zoom) wholly within the extent: the given one if it
// does, else the one that moves it just far enough to bring it in by the margin.
function keptInView(offset: number, start: number, size: number, extent: number): number {
  if (start + offset >= 0 && start + size + offset <= extent) return offset
  return Math.min(Math.max(offset, ANCHOR_MARGIN - start), extent - ANCHOR_MARGIN - (start + size))
}

function GraphViewport({ request }: { request: ViewRequest }) {
  const { setViewport, viewportInitialized } = useReactFlow()
  const width = useStore((state) => state.width)
  const height = useStore((state) => state.height)
  useEffect(() => {
    if (!viewportInitialized || width === 0 || height === 0) return
    const { bounds, minZoom, anchor } = request
    const fitHeight = height - CONTROLS_INSET
    const fitted = getViewportForBounds(bounds, width, fitHeight, minZoom, 1, FIT_PADDING)
    if (anchor === undefined) {
      void setViewport(fitted)
      return
    }
    // The centered view stands unless it leaves the anchor partly off the
    // canvas; then it moves just far enough to bring the anchor in by the
    // margin. Where what it fits is too tall for the canvas, its top (the
    // bands' titles) stays on the canvas if the anchor stays wholly on it too.
    const { zoom } = fitted
    const anchoredY = keptInView(fitted.y, anchor.y * zoom, anchor.height * zoom, fitHeight)
    const topY = -bounds.y * zoom
    const anchorFits = (offset: number) => anchor.y * zoom + offset >= 0 && (anchor.y + anchor.height) * zoom + offset <= fitHeight
    void setViewport({
      zoom,
      x: keptInView(fitted.x, anchor.x * zoom, anchor.width * zoom, width),
      y: bounds.y * zoom + anchoredY < 0 && anchorFits(topY) ? topY : anchoredY,
    })
  }, [setViewport, request, viewportInitialized, width, height])
  return null
}

// A step taking keyboard focus comes wholly onto the canvas above the zoom
// controls, its focus ring too: the view moves just far enough to bring it in
// by the margin, at the zoom it has. React Flow's own panning (off here) moves
// only to a step wholly off the canvas, and centers it, so a step half past
// the canvas's edge would take focus out of sight.
function StepInViewOnFocus() {
  const store = useStoreApi()
  const { setViewport } = useReactFlow()
  useEffect(() => {
    const canvas = store.getState().domNode
    if (canvas === null) return
    const bringIn = (event: FocusEvent) => {
      const step = (event.target as Element).closest('.react-flow__node')
      if (step === null || !step.matches(':focus-visible')) return
      const { transform: [x, y, zoom], width, height, nodeLookup } = store.getState()
      const node = nodeLookup.get(step.getAttribute('data-id') ?? '')
      if (node === undefined) return
      const { x: left, y: top } = node.internals.positionAbsolute
      const nextX = keptInView(x, (left - FOCUS_RING) * zoom, ((node.width ?? NODE_WIDTH) + 2 * FOCUS_RING) * zoom, width)
      const nextY = keptInView(y, (top - FOCUS_RING) * zoom, ((node.height ?? NODE_HEIGHT) + 2 * FOCUS_RING) * zoom, height - CONTROLS_INSET)
      if (nextX !== x || nextY !== y) void setViewport({ x: nextX, y: nextY, zoom })
    }
    canvas.addEventListener('focusin', bringIn)
    return () => canvas.removeEventListener('focusin', bringIn)
  }, [store, setViewport])
  return null
}

// A band's title and count keep to the band's top row, moving along it to stay
// over the part of the band that is on the canvas, so a view that shows only
// some of a stage's columns, as the opening view on a chain often does, still
// names the stage. They do not follow the view down the band: below its top
// row a title would sit among the lanes and steps, drawn under them, and every
// step names its own stage. Offsets are in graph units.
function StageBands({ stages }: { stages: GraphStage[] }) {
  const [x, , zoom] = useStore((state) => state.transform)
  const width = useStore((state) => state.width)
  const visibleLeft = -x / zoom
  const visibleRight = (width - x) / zoom
  return (
    <ViewportPortal>
      {stages.map((stage) => (
        <div key={stage.key} className="decision-stage pointer-events-none absolute overflow-hidden rounded-md border border-border/60 bg-background/25" style={{ left: stage.x, top: stage.y, width: stage.width, height: stage.height }}>
          <div
            className="flex items-center justify-between gap-2 px-4 py-3 text-xs font-medium text-foreground-secondary"
            style={{
              marginLeft: Math.min(Math.max(visibleLeft - stage.x, 0), stage.width),
              marginRight: Math.min(Math.max(stage.x + stage.width - visibleRight, 0), stage.width),
            }}
          >
            <span className="min-w-0 truncate">{stage.title}</span><span className="text-foreground-muted tabular-nums">{stage.count}</span>
          </div>
        </div>
      ))}
    </ViewportPortal>
  )
}

export function DecisionGraphView({ graph }: { graph: DecisionGraph }) {
  const layout = useMemo(() => layoutGraph(graph), [graph])
  const initialId = graph.nodes.find((node) => node.node_type === 'OUTCOME')?.node_id ?? graph.nodes[0].node_id
  const [selectedId, setSelectedId] = useState(initialId)
  const stepBox = (id: string): Rect => ({ ...layout.nodes.find((node) => node.id === id)!.position, width: NODE_WIDTH, height: NODE_HEIGHT })
  // The graph opens, and resets, on the chain through the step it starts at:
  // its steps and its relationships with their labels, taken from the top of
  // the bands (every route runs inside or between them) so their titles open
  // in view too, wherever the step it starts at stays in view with them.
  const openingView = (revision: number): ViewRequest => {
    const chain = traceRelationships(graph.edges, initialId)
    const chainBounds = enclose([
      ...[...chain.nodeIds].map(stepBox),
      ...[...chain.edgeIndexes].flatMap((index) => edgeBoxes(layout.edges[index].data!)),
    ])
    const top = layout.stages[0].y
    return {
      revision,
      bounds: { ...chainBounds, y: top, height: chainBounds.y + chainBounds.height - top },
      minZoom: TRACE_MIN_ZOOM,
      anchor: stepBox(initialId),
    }
  }
  const [viewRequest, setViewRequest] = useState(() => openingView(0))
  const selected = graph.nodes.find((node) => node.node_id === selectedId) ?? graph.nodes[0]
  const trace = useMemo(() => traceRelationships(graph.edges, selected.node_id), [graph.edges, selected.node_id])
  const nodes = useMemo(() => layout.nodes.map((node) => ({
    ...node,
    selected: node.id === selected.node_id,
    className: node.id === selected.node_id ? undefined : trace.nodeIds.has(node.id) ? 'decision-node-related' : 'decision-node-muted',
  })), [layout.nodes, selected.node_id, trace])
  // Each piece of a route is lit where a traced relationship runs along it, so
  // a stretch shared by several is lit by any of them. Lit pieces are drawn
  // after the muted ones, so they lie over them and still under every label
  // (a raised layer would cover the labels). Only traced relationships show
  // their label, so the chain reads without the rest competing; an arrow ends
  // only the pieces that reach their target step. The relationships follow,
  // drawing nothing, for assistive technology to read.
  const edges = useMemo((): Edge[] => [...layout.paths.map((path) => {
    const highlighted = onTrace(path.data!.members, trace.edgeIndexes)
    const color = highlighted ? 'var(--graph-trace)' : 'var(--foreground-muted)'
    return {
      ...path,
      data: { ...path.data!, labels: path.data!.labels.filter((label) => onTrace(label.members, trace.edgeIndexes)) },
      animated: highlighted,
      markerEnd: path.data!.toHandle ? { type: MarkerType.ArrowClosed, color } : undefined,
      style: { stroke: color, strokeWidth: highlighted ? 2 : 1.25, opacity: highlighted ? 1 : 0.3 },
    }
  }).sort((a, b) => Number(a.animated) - Number(b.animated)), ...layout.edges], [layout.paths, layout.edges, trace])
  const counts = `${counted(graph.nodes.length, 'step', 'steps')} · ${counted(graph.edges.length, 'relationship', 'relationships')}`

  return (
    <div className="flex shrink-0 flex-col lg:min-h-0 lg:flex-1">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-x-6 gap-y-1.5 border-b border-border px-4 py-1.5">
        <ul aria-label="Graph legend" className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <li className="flex items-center gap-1.5 text-xs text-foreground-muted"><span aria-hidden="true" className={cn('h-0.5 w-4 rounded-full bg-graph-trace', markInRow)} />Selected path</li>
          {LEGEND_ACTIONS.map((action) => <li key={action} className="flex"><ActionBadge action={action} /></li>)}
        </ul>
        {/* The view's two groups are split by a hairline in the 16px between
            them, which never starts a line when they wrap (dividedRow); on a
            narrow screen the row spans the bar, its clip's 5px a side included. */}
        <div className={cn('flex w-[calc(100%+10px)] min-w-0 flex-wrap items-center gap-x-4 gap-y-2 sm:w-auto', dividedRow)}>
          <span className="text-sm text-foreground-muted tabular-nums">{counts}</span>
          <div role="group" aria-label="Inspect a step" className="contents sm:flex sm:min-w-0 sm:items-center sm:gap-2">
            <Select value={String(graph.nodes.indexOf(selected))} onValueChange={(value) => setSelectedId(graph.nodes[Number(value)].node_id)}>
              <SelectTrigger aria-label="Inspect decision step" className="w-full min-w-0 sm:w-72">
                <SelectValue><span className="truncate">{STAGE_LABELS[selected.node_type]} · {stepLabel(selected)}</span></SelectValue>
              </SelectTrigger>
              <SelectContent align="end" className="max-h-[min(24rem,var(--radix-select-content-available-height))] w-96">
                <SelectGroup>
                  {graph.nodes.map((node, index) => (
                    <SelectItem key={node.node_id} value={String(index)} textValue={`${STAGE_LABELS[node.node_type]} ${stepLabel(node)} ${node.hostname ?? ''}`}>
                      <span className="flex min-w-0 flex-col">
                        <span className="text-xs text-foreground-muted">{STAGE_LABELS[node.node_type]} · {ACTION_LABELS[node.action]}</span>
                        <span className="line-clamp-2 break-words"><Identifier>{stepLabel(node)}</Identifier></span>
                        <span className="truncate text-xs text-foreground-muted">{node.hostname ?? 'No host recorded'} · {node.symptom || 'No symptom recorded'}</span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
            <Button variant="ghost" onClick={() => setViewRequest(({ revision }) => ({ revision: revision + 1, bounds: stepBox(selected.node_id), minZoom: MIN_ZOOM }))}>
              <LocateFixed data-icon="inline-start" />Focus step
            </Button>
          </div>
          <div role="group" aria-label="Graph view" className={cn('flex items-center gap-2', gapHairline[16])}>
            <Button variant="ghost" onClick={() => setViewRequest(({ revision }) => ({ revision: revision + 1, bounds: layout.bounds, minZoom: MIN_ZOOM }))}>
              <ScanLine data-icon="inline-start" />Fit view
            </Button>
            <Button variant="ghost" onClick={() => { setSelectedId(initialId); setViewRequest(({ revision }) => openingView(revision + 1)) }}>
              <RotateCcw data-icon="inline-start" />Reset
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
              autoPanOnNodeFocus={false}
              nodesConnectable={false}
              edgesFocusable={false}
              deleteKeyCode={null}
              multiSelectionKeyCode={null}
              selectionKeyCode={null}
              zoomOnScroll={false}
              preventScrolling={false}
              minZoom={MIN_ZOOM}
              maxZoom={1.5}
              colorMode="dark"
              ariaLabelConfig={{
                'node.a11yDescription.default': 'Press Enter or Space to inspect this step.',
                'node.a11yDescription.keyboardDisabled': 'Press Enter or Space to inspect this step.',
              }}
            >
              <GraphViewport request={viewRequest} />
              <StepInViewOnFocus />
              <StageBands stages={layout.stages} />
              <Background color="var(--border)" gap={20} size={1} />
              <Controls showInteractive={false} showFitView={false} position="bottom-left" />
            </ReactFlow>
          </div>
        </div>
        <StepDetails step={selected} graph={graph} onSelect={setSelectedId} />
      </div>
      <div className="flex shrink-0 flex-wrap justify-between gap-x-6 gap-y-1 border-t border-border bg-background px-4 py-1.5 text-xs text-foreground-muted">
        <span>Arrows show recorded relationships. Extraction does not imply survival.</span>
        <span className="decision-pointer-hint">Drag to pan · Use + / − to zoom</span>
        <span className="decision-touch-hint">Swipe to scroll · Two fingers to pan · + / − to zoom</span>
      </div>
    </div>
  )
}
