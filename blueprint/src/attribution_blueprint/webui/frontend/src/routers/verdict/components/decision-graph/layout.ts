// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import dagre from '@dagrejs/dagre'
import { MarkerType, type Edge, type Node } from '@xyflow/react'
import type { DecisionGraph, DecisionNode } from '@/interfaces/attribution'

export type TraceNode = Node<{ step: DecisionNode }, 'decision'>

export const NODE_WIDTH = 248
export const NODE_HEIGHT = 112
export const EDGE_LABEL_WIDTH = 72
export const EDGE_LABEL_HEIGHT = 20

export interface GraphStage {
  key: 'extraction' | 'rules' | 'outcome' | 'unknown'
  title: string
  x: number
  y: number
  width: number
  height: number
  count: number
}

export type TraceEdgeData = {
  edgeType: string
  points: { x: number; y: number }[]
  labelX: number
  labelY: number
  labelWidth: number
}

const ROW_GAP = 28
const COLUMN_GAP = 32
const STAGE_GAP = 80
const CONTENT_TOP = 150
const LABEL_GAP = 8
const CHANNEL_PADDING = 10
const OVERVIEW_ASPECT_RATIO = 2
const EDGE_LABELS: Record<string, string> = { CREATES: 'Creates', MATCHES: 'Matches', DEMOTES: 'Demotes', PRUNES: 'Prunes' }

const stageWidth = (columns: number) => columns * NODE_WIDTH + (columns - 1) * COLUMN_GAP + 48
const stageContentHeight = (count: number, columns: number) => Math.max(1, Math.ceil(count / columns)) * (NODE_HEIGHT + ROW_GAP) - ROW_GAP

function balancedColumns(counts: number[]) {
  let bestColumns = counts.map(() => 1)
  let bestScale = 0
  let bestArea = Infinity
  // For a maximum row count, ceil(count / rows) is each stage's minimum
  // column count. Extra columns only widen the graph unless they reduce its
  // maximum rows, a case covered by another iteration. This finds the best
  // landscape fit in O(stages * largest stage), without column combinations.
  // Descending rows also preserves the fewest-columns tie break.
  for (let rows = Math.max(1, ...counts); rows >= 1; rows--) {
    const columns = counts.map((count) => Math.max(1, Math.ceil(count / rows)))
    const totalWidth = columns.reduce((width, count) => width + stageWidth(count), 0) + STAGE_GAP * (counts.length - 1)
    const totalHeight = CONTENT_TOP + Math.max(...counts.map((count, index) => stageContentHeight(count, columns[index]))) + 24
    const scale = Math.min(OVERVIEW_ASPECT_RATIO / totalWidth, 1 / totalHeight)
    const area = totalWidth * totalHeight
    if (scale > bestScale || (scale === bestScale && area < bestArea)) {
      bestColumns = columns
      bestScale = scale
      bestArea = area
    }
  }
  return bestColumns
}

function stageKey(node: DecisionNode): GraphStage['key'] {
  if (node.node_type === 'EXTRACTION') return 'extraction'
  if (node.node_type === 'OUTCOME') return 'outcome'
  if (node.node_type === 'UNKNOWN') return 'unknown'
  return 'rules'
}

// Stage bands describe the Service's node types, not additional causal steps.
// In particular, an extraction-to-outcome edge never becomes a rule-to-outcome edge.
export function layoutGraph(graph: DecisionGraph): { nodes: TraceNode[]; edges: Edge<TraceEdgeData>[]; stages: GraphStage[] } {
  const layout = new dagre.graphlib.Graph({ multigraph: true })
    .setGraph({ rankdir: 'LR', ranksep: 100, nodesep: 32, marginx: 24, marginy: 24 })
    .setDefaultEdgeLabel(() => ({}))
  // Graphlib uses object keys internally. Keep opaque Service IDs out of that
  // namespace (e.g. "constructor"), while preserving them in the rendered graph.
  const layoutIds = new Map(graph.nodes.map((node, index) => [node.node_id, `node-${index}`]))
  graph.nodes.forEach((node) => layout.setNode(layoutIds.get(node.node_id)!, { width: NODE_WIDTH, height: NODE_HEIGHT }))
  graph.edges.forEach((edge, index) => layout.setEdge(layoutIds.get(edge.parent_id)!, layoutIds.get(edge.child_id)!, {}, String(index)))
  dagre.layout(layout)

  const groups: { key: GraphStage['key']; title: string }[] = [
    { key: 'extraction', title: 'Extracted symptoms' },
    { key: 'rules', title: 'Applied rules' },
    { key: 'outcome', title: 'Outcome' },
  ]
  if (graph.nodes.some((node) => stageKey(node) === 'unknown')) groups.push({ key: 'unknown', title: 'Unknown stage' })
  const groupedNodes = groups.map(({ key }) => graph.nodes
    .filter((node) => stageKey(node) === key)
    .sort((a, b) => layout.node(layoutIds.get(a.node_id)!).y - layout.node(layoutIds.get(b.node_id)!).y))
  const columns = balancedColumns(groupedNodes.map((nodes) => nodes.length))
  const contentHeights = groupedNodes.map((nodes, index) => stageContentHeight(nodes.length, columns[index]))
  const maxContentHeight = Math.max(...contentHeights)
  let nextX = 24
  const stages: GraphStage[] = groups.map((group, index) => {
    const width = stageWidth(columns[index])
    const stage = { ...group, x: nextX, y: 0, width, height: CONTENT_TOP + maxContentHeight + 24, count: groupedNodes[index].length }
    nextX += width + STAGE_GAP
    return stage
  })
  const positions = new Map<string, { x: number; y: number }>()
  const nodeStages = new Map<string, number>()
  const nodeColumns = new Map<string, number>()
  groupedNodes.forEach((nodes, stageIndex) => nodes.forEach((node, index) => {
    const column = index % columns[stageIndex]
    positions.set(node.node_id, {
      x: stages[stageIndex].x + 24 + column * (NODE_WIDTH + COLUMN_GAP),
      y: CONTENT_TOP + (maxContentHeight - contentHeights[stageIndex]) / 2
        + Math.floor(index / columns[stageIndex]) * (NODE_HEIGHT + ROW_GAP),
    })
    nodeStages.set(node.node_id, stageIndex)
    nodeColumns.set(node.node_id, column)
  }))
  const edgeGeometry = graph.edges.map((edge) => {
    const sourceStage = nodeStages.get(edge.parent_id)!
    const targetStage = nodeStages.get(edge.child_id)!
    const sourceNeedsGutter = nodeColumns.get(edge.parent_id)! < columns[sourceStage] - 1
    const targetNeedsGutter = nodeColumns.get(edge.child_id)! > 0
    const source = positions.get(edge.parent_id)!
    const target = positions.get(edge.child_id)!
    const sourceY = source.y + (sourceNeedsGutter ? NODE_HEIGHT + ROW_GAP / 2 : NODE_HEIGHT / 2)
    const targetY = target.y + (targetNeedsGutter ? -ROW_GAP / 2 : NODE_HEIGHT / 2)
    const label = Object.hasOwn(EDGE_LABELS, edge.edge_type) ? EDGE_LABELS[edge.edge_type] : edge.edge_type
    // A fixed label box keeps geometry independent of font metrics and loading.
    // The renderer truncates long names; connected-step details retain the text.
    const labelWidth = EDGE_LABEL_WIDTH
    return { sourceStage, targetStage, sourceNeedsGutter, targetNeedsGutter, sourceY, targetY, label, labelWidth }
  })

  // Adjacent stages share a channel only when their label boxes do not overlap.
  // Route each edge through its assigned label, widening the gap only as needed.
  const channels = groups.map(() => [] as { bottom: number; width: number; x: number }[])
  const edgeChannels = new Map<number, number>()
  edgeGeometry.map((edge, index) => ({ ...edge, index, labelY: (edge.sourceY + edge.targetY) / 2 }))
    .filter((edge) => edge.targetStage - edge.sourceStage === 1)
    .sort((a, b) => a.labelY - b.labelY)
    .forEach((edge) => {
      const lanes = channels[edge.sourceStage]
      let laneIndex = lanes.findIndex((lane) => lane.bottom + LABEL_GAP <= edge.labelY - EDGE_LABEL_HEIGHT / 2)
      if (laneIndex === -1) {
        laneIndex = lanes.length
        lanes.push({ bottom: 0, width: 0, x: 0 })
      }
      const lane = lanes[laneIndex]
      lane.bottom = edge.labelY + EDGE_LABEL_HEIGHT / 2
      lane.width = Math.max(lane.width, edge.labelWidth)
      edgeChannels.set(edge.index, laneIndex)
    })
  nextX = stages[0].x
  stages.forEach((stage, index) => {
    const shift = nextX - stage.x
    stage.x = nextX
    groupedNodes[index].forEach((node) => { positions.get(node.node_id)!.x += shift })
    const lanes = channels[index]
    const channelWidth = lanes.reduce((width, lane) => width + lane.width, 0) + Math.max(0, lanes.length - 1) * LABEL_GAP
    const gap = Math.max(STAGE_GAP, channelWidth + CHANNEL_PADDING * 2)
    let laneX = stage.x + stage.width + (gap - channelWidth) / 2
    lanes.forEach((lane) => {
      lane.x = laneX + lane.width / 2
      laneX += lane.width + LABEL_GAP
    })
    nextX += stage.width + gap
  })

  // Skipped, same-stage and reverse edges use tracks above the cards. Pack label
  // boxes into free horizontal spans; additional tracks reserve their own space.
  const tracks: { left: number; right: number }[][] = []
  const topRoutes = new Map<number, { sourceExitX: number; targetEntryX: number; labelX: number; labelY: number }>()
  edgeGeometry.forEach((edge, index) => {
    if (edgeChannels.has(index)) return
    const sourceStage = stages[edge.sourceStage]
    const targetStage = stages[edge.targetStage]
    const sourceExitX = sourceStage.x + sourceStage.width - 12
    const targetEntryX = targetStage.x + 12
    const left = Math.min(sourceExitX, targetEntryX)
    const right = Math.max(sourceExitX, targetEntryX)
    const preferredX = (left + right) / 2
    let trackIndex = 0
    let labelX = preferredX
    for (; trackIndex < tracks.length; trackIndex++) {
      const occupied = tracks[trackIndex]
      let freeLeft = left
      const candidates: number[] = []
      for (const interval of [...occupied, { left: right + LABEL_GAP, right }]) {
        const freeRight = Math.min(right, interval.left - LABEL_GAP)
        if (freeRight - freeLeft >= edge.labelWidth) {
          candidates.push(Math.max(freeLeft + edge.labelWidth / 2, Math.min(preferredX, freeRight - edge.labelWidth / 2)))
        }
        freeLeft = Math.max(freeLeft, interval.right + LABEL_GAP)
      }
      if (candidates.length > 0) {
        labelX = candidates.reduce((closest, candidate) => Math.abs(candidate - preferredX) < Math.abs(closest - preferredX) ? candidate : closest)
        break
      }
    }
    const track = tracks[trackIndex] ?? (tracks[trackIndex] = [])
    track.push({ left: labelX - edge.labelWidth / 2, right: labelX + edge.labelWidth / 2 })
    track.sort((a, b) => a.left - b.left)
    topRoutes.set(index, { sourceExitX, targetEntryX, labelX, labelY: 54 + trackIndex * (EDGE_LABEL_HEIGHT + LABEL_GAP) })
  })
  const extraTop = Math.max(0, 54 + tracks.length * (EDGE_LABEL_HEIGHT + LABEL_GAP) + ROW_GAP - CONTENT_TOP)
  stages.forEach((stage) => { stage.height += extraTop })
  positions.forEach((position) => { position.y += extraTop })

  return {
    stages,
    nodes: graph.nodes.map((step) => ({
      id: step.node_id,
      type: 'decision',
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      position: positions.get(step.node_id)!,
      data: { step },
      ariaLabel: `${step.node_type}, ${step.action}: ${step.symptom}. Rule: ${step.rule_name ?? 'not recorded'}. Host: ${step.hostname ?? 'not recorded'}`,
    })),
    edges: graph.edges.map((edge, index) => {
      const source = positions.get(edge.parent_id)!
      const target = positions.get(edge.child_id)!
      const geometry = edgeGeometry[index]
      const { sourceNeedsGutter, targetNeedsGutter, labelWidth } = geometry
      const sourceY = geometry.sourceY + extraTop
      const targetY = geometry.targetY + extraTop
      const route = topRoutes.get(index)
      const labelX = route?.labelX ?? channels[geometry.sourceStage][edgeChannels.get(index)!].x
      const labelY = route?.labelY ?? (sourceY + targetY) / 2
      const data: TraceEdgeData = { edgeType: edge.edge_type, points: [], labelX, labelY, labelWidth }
      // Inner columns leave and enter through their row gutters. This keeps
      // every real edge visible without drawing through neighboring cards.
      if (sourceNeedsGutter) {
        const x = source.x + NODE_WIDTH + COLUMN_GAP / 2
        data.points.push({ x, y: source.y + NODE_HEIGHT / 2 }, { x, y: sourceY })
      }
      if (route) {
        data.points.push({ x: route.sourceExitX, y: sourceY }, { x: route.sourceExitX, y: labelY }, { x: route.targetEntryX, y: labelY }, { x: route.targetEntryX, y: targetY })
      } else {
        data.points.push({ x: labelX, y: sourceY }, { x: labelX, y: targetY })
      }
      if (targetNeedsGutter) {
        const x = target.x - COLUMN_GAP / 2
        data.points.push({ x, y: targetY }, { x, y: target.y + NODE_HEIGHT / 2 })
      }
      return {
        id: `edge-${index}`,
        source: edge.parent_id,
        target: edge.child_id,
        label: geometry.label,
        data,
        type: 'decision',
        markerEnd: { type: MarkerType.ArrowClosed, color: 'var(--foreground-muted)' },
        style: { stroke: 'var(--foreground-muted)', strokeWidth: 1.25 },
        selectable: false,
        focusable: false,
      }
    }),
  }
}
