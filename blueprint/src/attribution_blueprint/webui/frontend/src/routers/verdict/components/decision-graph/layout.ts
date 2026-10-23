// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import dagre from '@dagrejs/dagre'
import type { Edge, Node, Rect } from '@xyflow/react'
import type { DecisionGraph, DecisionNode } from '@/interfaces/attribution'
import { ACTION_LABELS, edgeLabel, relationshipName, STAGE_LABELS } from './labels'

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

// A piece of what the view draws. Routes share stretches: the ways out of a
// step until they part, the ways into one from where they meet, a lane's
// riser and trunk. Every stretch is drawn once, by one piece carrying every
// relationship whose route runs along it, so it is lit wherever a traced one
// runs; a piece runs on through its corners while those relationships do not
// change, and ends where routes join or part.
export type TracePathData = {
  /**
   * Its corners, its ends included, in flow coordinates: an end at a step
   * lies on the step's border at its middle, as the route was laid out, so
   * the piece is drawn from these points alone, never from where React Flow
   * measures the step's handles on screen.
   */
  points: { x: number; y: number }[]
  /** It starts at its source step; otherwise at its first point, where routes join or part. */
  fromHandle: boolean
  /** It ends at its target step, where its arrow is; otherwise at its last point, where routes join or part. */
  toHandle: boolean
  /** The relationships (graph edge indexes) whose routes run along it. */
  members: number[]
  /**
   * The labels on it, each naming a group of relationships of one kind into
   * one step. A stretch holds more than one only where relationships of
   * different kinds between the same steps run as one line.
   */
  labels: { text: string; x: number; y: number; width: number; members: number[] }[]
}

const ROW_GAP = 28
const COLUMN_GAP = 32
const STAGE_GAP = 80
// Risers and drops beside a band run this far apart, so each stays its own line.
const LINE_GAP = 8
// A gutter between steps carries two lines a line gap apart, so a way out of
// a step and a way into one never share a line or meet at a point (where two
// unrelated routes would read as one line through it). A row gutter's ways
// out of the row above run along its middle, where the middles of a
// neighbouring band's rows fall when that band is offset by half a row (bands
// are centered), so the most common route, from an inner column to a step's
// middle in the next band, runs level; its ways into the row below run a
// line gap lower. A column gutter's ways out of the column before it run
// nearer that column, and its ways into the column after it nearer that one.
const EXIT_INSET = ROW_GAP / 2
const ENTRY_INSET = ROW_GAP / 2 - LINE_GAP
const COLUMN_INSET = (COLUMN_GAP - LINE_GAP) / 2
// Steps start below a band's title row (StageBands: a 16px line 12px from the
// band's top edge), as far below the title as the band's 24px inset, and
// below the lanes of the relationships passing the bands, if any: the way
// into a first-row step of an inner column runs in the row's gutter above
// it, a label gap below the lowest lane's label, so it never runs through it.
const CONTENT_TOP = 52
const LABEL_GAP = 8
// The lanes under the band titles: the first label box 8px under the 40px
// title row, each next one a label and a gap lower.
const LANE_TOP = 40 + LABEL_GAP + EDGE_LABEL_HEIGHT / 2
const LANE_PITCH = EDGE_LABEL_HEIGHT + LABEL_GAP
const CHANNEL_PADDING = 10
// Lines keep this far from a label's box, so none seems to touch it; a line
// level with a label (crossing its height, or its width for a label on a
// level line) keeps farther along its way, as one ending beside a label reads
// as the label's own.
const LABEL_CLEARANCE = 6
const LEVEL_CLEARANCE = 24
const OVERVIEW_ASPECT_RATIO = 2

const stageWidth = (columns: number) => columns * NODE_WIDTH + (columns - 1) * COLUMN_GAP + 48
const stageContentHeight = (count: number, columns: number) => Math.max(1, Math.ceil(count / columns)) * (NODE_HEIGHT + ROW_GAP) - ROW_GAP

function balancedColumns(counts: number[], contentTop: number) {
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
    const totalHeight = contentTop + Math.max(...counts.map((count, index) => stageContentHeight(count, columns[index]))) + 24
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

export function enclose(boxes: Rect[]): Rect {
  const x = Math.min(...boxes.map((box) => box.x))
  const y = Math.min(...boxes.map((box) => box.y))
  return {
    x, y,
    width: Math.max(...boxes.map((box) => box.x + box.width)) - x,
    height: Math.max(...boxes.map((box) => box.y + box.height)) - y,
  }
}

// What a relationship draws: its label's box and every corner of its route.
export function edgeBoxes({ labelX, labelY, labelWidth, points }: TraceEdgeData): Rect[] {
  return [
    { x: labelX - labelWidth / 2, y: labelY - EDGE_LABEL_HEIGHT / 2, width: labelWidth, height: EDGE_LABEL_HEIGHT },
    ...points.map(({ x, y }) => ({ x, y, width: 0, height: 0 })),
  ]
}

function stageKey(node: DecisionNode): GraphStage['key'] {
  if (node.node_type === 'EXTRACTION') return 'extraction'
  if (node.node_type === 'OUTCOME') return 'outcome'
  if (node.node_type === 'UNKNOWN') return 'unknown'
  return 'rules'
}

// Stage bands describe the Service's node types, not additional causal steps.
// In particular, an extraction-to-outcome edge never becomes a rule-to-outcome edge.
export function layoutGraph(graph: DecisionGraph): { nodes: TraceNode[]; edges: Edge<TraceEdgeData>[]; paths: Edge<TracePathData>[]; stages: GraphStage[]; bounds: Rect } {
  // Dagre only orders the steps within their bands, which asks only how
  // strongly two steps are related: parallel relationships are one edge
  // weighted by their number, and a step's relationship to itself orders
  // nothing. Given parallel edges as a multigraph, its layout can fail
  // outright (three between one pair beside an unrelated step).
  const layout = new dagre.graphlib.Graph()
    .setGraph({ rankdir: 'LR', ranksep: 100, nodesep: 32, marginx: 24, marginy: 24 })
  // Graphlib uses object keys internally. Keep opaque Service IDs out of that
  // namespace (e.g. "constructor"), while preserving them in the rendered graph.
  const layoutIds = new Map(graph.nodes.map((node, index) => [node.node_id, `node-${index}`]))
  graph.nodes.forEach((node) => layout.setNode(layoutIds.get(node.node_id)!, { width: NODE_WIDTH, height: NODE_HEIGHT }))
  graph.edges.forEach((edge) => {
    const parent = layoutIds.get(edge.parent_id)!
    const child = layoutIds.get(edge.child_id)!
    if (parent !== child) layout.setEdge(parent, child, { weight: (layout.edge(parent, child)?.weight ?? 0) + 1 })
  })
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
  // Relationships between neighboring stages cross the gap between their
  // bands. Every other one (skipping a stage, staying within one or going back
  // to an earlier one) runs through the bands between in a lane under their
  // titles, above every step: from its step to a riser in the gap after its
  // source's band, up to its lane, across, and down a drop in the gap before
  // its target's band, so it passes a band only above its steps, and its label
  // sits inside a band it passes, never on a border. The relationships of one
  // kind into one step from one band (the matches of every extraction of the
  // winning symptom into the outcome) share a riser, a lane, a label and a
  // drop, so a symptom seen on many hosts adds branches, not lanes.
  const stageOf = new Map<string, number>()
  groupedNodes.forEach((nodes, stageIndex) => nodes.forEach((node) => stageOf.set(node.node_id, stageIndex)))
  const laneKeys = new Map<number, string>()
  graph.edges.forEach((edge, index) => {
    const sourceStage = stageOf.get(edge.parent_id)!
    if (stageOf.get(edge.child_id)! - sourceStage !== 1) laneKeys.set(index, JSON.stringify([sourceStage, edge.child_id, edge.edge_type]))
  })
  const laneCount = new Set(laneKeys.values()).size
  const contentTop = laneCount === 0
    ? CONTENT_TOP
    : LANE_TOP + (laneCount - 1) * LANE_PITCH + EDGE_LABEL_HEIGHT / 2 + LABEL_GAP + ENTRY_INSET
  const columns = balancedColumns(groupedNodes.map((nodes) => nodes.length), contentTop)
  const contentHeights = groupedNodes.map((nodes, index) => stageContentHeight(nodes.length, columns[index]))
  const maxContentHeight = Math.max(...contentHeights)
  let nextX = 24
  const stages: GraphStage[] = groups.map((group, index) => {
    const width = stageWidth(columns[index])
    const stage = { ...group, x: nextX, y: 0, width, height: contentTop + maxContentHeight + 24, count: groupedNodes[index].length }
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
      y: contentTop + (maxContentHeight - contentHeights[stageIndex]) / 2
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
    const sourceY = source.y + (sourceNeedsGutter ? NODE_HEIGHT + EXIT_INSET : NODE_HEIGHT / 2)
    const targetY = target.y + (targetNeedsGutter ? -ENTRY_INSET : NODE_HEIGHT / 2)
    const label = edgeLabel(edge.edge_type)
    // A fixed label box keeps geometry independent of font metrics and loading.
    // The renderer truncates long names; connected-step details retain the text.
    const labelWidth = EDGE_LABEL_WIDTH
    return { sourceStage, targetStage, sourceNeedsGutter, targetNeedsGutter, sourceY, targetY, label, labelWidth }
  })

  // Relationships into the next band cross the gap between the two down or up
  // a line of their own, in the gap's channel. Those of one kind into one step
  // joining its way in from one side (above it, below it or level with it)
  // share a line and a label, as a lane's do, so the demotions of several
  // extractions of one symptom add branches, not lines or labels; the label
  // rides the stretch nearest the step, which all of them run along. Two
  // groups' lines share a place in the channel only when all of one, its
  // label included, lies a label gap past all of the other, so no two share a
  // stretch, and the gap widens only as far as its lines need.
  const channels = groups.map(() => [] as { bottom: number; width: number; x: number }[])
  const channelKeys = new Map<number, string>()
  edgeGeometry.forEach(({ sourceStage, targetStage, sourceY, targetY }, index) => {
    if (targetStage - sourceStage !== 1) return
    const { child_id: target, edge_type: kind } = graph.edges[index]
    channelKeys.set(index, JSON.stringify([sourceStage, target, kind, Math.sign(sourceY - targetY)]))
  })
  const channelGroups = [...new Set(channelKeys.values())].map((key) => {
    const members = [...channelKeys].filter(([, group]) => group === key).map(([index]) => index)
    const { sourceStage, targetY, labelWidth } = edgeGeometry[members[0]]
    const joins = members.map((index) => edgeGeometry[index].sourceY)
    const nearest = joins.reduce((closest, y) => (Math.abs(y - targetY) < Math.abs(closest - targetY) ? y : closest))
    return {
      members,
      sourceStage,
      labelWidth,
      joins,
      nearest,
      targetY,
      labelY: (nearest + targetY) / 2,
      // Its stretch nearest the step is too short to hold its label clear of
      // the level lines at its ends, where others may run on past its line.
      short: Math.abs(nearest - targetY) < EDGE_LABEL_HEIGHT + LABEL_CLEARANCE * 2,
      top: Math.min(targetY, ...joins) - EDGE_LABEL_HEIGHT / 2,
      bottom: Math.max(targetY, ...joins) + EDGE_LABEL_HEIGHT / 2,
      lane: 0,
    }
  })
  // A group's label starts where no other group's line crosses the channel
  // level with it (their ways out, and their ways into their steps), on the
  // stretch nearest the step, as near its middle as that allows: the lines
  // are spaced by it, and it is placed along the drawn lines (below).
  channelGroups.forEach((group) => {
    const [top, bottom] = [Math.min(group.nearest, group.targetY), Math.max(group.nearest, group.targetY)]
    const half = EDGE_LABEL_HEIGHT / 2
    if (group.short) return
    const heights = channelGroups
      .filter((other) => other !== group && other.sourceStage === group.sourceStage)
      .flatMap((other) => [...other.joins, other.targetY])
    const clearOf = (y: number) => heights.every((height) => Math.abs(height - y) >= half + LABEL_CLEARANCE)
    const best = [(top + bottom) / 2, ...heights.flatMap((height) => [height - half - LABEL_CLEARANCE, height + half + LABEL_CLEARANCE])]
      .filter((y) => y >= top + half && y <= bottom - half && clearOf(y))
      .sort((a, b) => Math.abs(a - (top + bottom) / 2) - Math.abs(b - (top + bottom) / 2))[0]
    if (best !== undefined) group.labelY = best
  })
  // How far right of a gap's first line each of its lines runs: as near the
  // lines before it as keeps every line clear of the labels level with it and
  // the labels clear of each other, a line gap at least.
  type ChannelGroup = (typeof channelGroups)[number]
  const laneOffsets = (inGap: ChannelGroup[], count: number) => {
    const line = (group: ChannelGroup) => [group.top + EDGE_LABEL_HEIGHT / 2, group.bottom - EDGE_LABEL_HEIGHT / 2] as const
    const band = (group: ChannelGroup) => [group.labelY - EDGE_LABEL_HEIGHT / 2 - LABEL_CLEARANCE, group.labelY + EDGE_LABEL_HEIGHT / 2 + LABEL_CLEARANCE] as const
    // A line within a label's width that ends near it would read as its own,
    // as would a level line ending beside it: a way out ending at the line
    // before the label's, or a way in starting at the line after it.
    const reach = (group: ChannelGroup) => [group.labelY - EDGE_LABEL_HEIGHT / 2 - LEVEL_CLEARANCE, group.labelY + EDGE_LABEL_HEIGHT / 2 + LEVEL_CLEARANCE] as const
    const levelWith = (group: ChannelGroup) => (y: number) => Math.abs(y - group.labelY) < EDGE_LABEL_HEIGHT / 2
    const meet = ([a, b]: readonly [number, number], [c, d]: readonly [number, number]) => a <= d && c <= b
    const spacing = (before: ChannelGroup, after: ChannelGroup) => Math.max(
      LINE_GAP,
      meet(line(before), reach(after)) ? after.labelWidth / 2 + LABEL_CLEARANCE : 0,
      meet(line(after), reach(before)) ? before.labelWidth / 2 + LABEL_CLEARANCE : 0,
      before.joins.some(levelWith(after)) ? after.labelWidth / 2 + LEVEL_CLEARANCE : 0,
      levelWith(before)(after.targetY) ? before.labelWidth / 2 + LEVEL_CLEARANCE : 0,
      meet(band(before), band(after)) ? (before.labelWidth + after.labelWidth) / 2 + LABEL_GAP : 0,
    )
    const offsets: number[] = []
    for (let lane = 0; lane < count; lane++) {
      const own = inGap.filter((group) => group.lane === lane)
      offsets.push(Math.max(0, ...inGap.filter((group) => group.lane < lane).flatMap((before) => own.map((group) => offsets[before.lane] + spacing(before, group)))))
    }
    return offsets
  }
  // The groups whose stretch is short take the lines nearest the next band,
  // so every way on from their steps turns off before it reaches their label.
  for (const short of [false, true]) {
    const first = channels.map((lanes) => lanes.length)
    channelGroups.filter((group) => group.short === short).sort((a, b) => a.top - b.top || a.members[0] - b.members[0]).forEach((group) => {
      const lanes = channels[group.sourceStage]
      group.lane = lanes.findIndex((lane, index) => index >= first[group.sourceStage] && lane.bottom + LABEL_GAP <= group.top)
      if (group.lane === -1) {
        group.lane = lanes.length
        lanes.push({ bottom: 0, width: 0, x: 0 })
      }
      lanes[group.lane].bottom = group.bottom
      lanes[group.lane].width = Math.max(lanes[group.lane].width, group.labelWidth)
    })
  }
  // A short group's label keeps to its stretch, so no other line level with
  // the label may pass its line: a way out along the label's height, running
  // on to its own line, takes a line nearer the band before, and a way in,
  // running from its own line into its step, one nearer the band after. The
  // channel's lines keep that order wherever these do not contradict.
  channels.forEach((lanes, stage) => {
    const inGap = channelGroups.filter((group) => group.sourceStage === stage)
    const after = lanes.map(() => new Set<number>())
    inGap.filter((group) => group.short).forEach((group) => {
      const level = (y: number) => Math.abs(y - group.labelY) < EDGE_LABEL_HEIGHT / 2 + LABEL_CLEARANCE
      inGap.filter((other) => other.lane !== group.lane).forEach((other) => {
        if (other.joins.some(level)) after[other.lane].add(group.lane)
        if (level(other.targetY)) after[group.lane].add(other.lane)
      })
    })
    const order: number[] = []
    const waiting = lanes.map((_, lane) => lane)
    while (waiting.length > 0) {
      const ready = waiting.find((lane) => waiting.every((other) => !after[other].has(lane) || other === lane)) ?? waiting[0]
      order.push(ready)
      waiting.splice(waiting.indexOf(ready), 1)
    }
    const moved = order.map((lane) => lanes[lane])
    inGap.forEach((group) => { group.lane = order.indexOf(group.lane) })
    lanes.splice(0, lanes.length, ...moved)
  })
  const channelOf = new Map(channelGroups.flatMap((group) => group.members.map((index) => [index, group] as const)))

  // Lane 0 is nearest the titles and the last nearest the steps. Lower groups
  // (by their lowest source and their target) take the lanes nearer the
  // steps, and the nearer the steps its lane, the farther from the bands a
  // group rises and drops, which keeps two groups rising from or dropping into
  // one gap from crossing there whenever the lower one's source and target
  // both lie lower.
  const laneGroups = [...new Set(laneKeys.values())]
    .map((key) => {
      const members = [...laneKeys].filter(([, group]) => group === key).map(([index]) => index)
      const { sourceStage, targetStage, targetY } = edgeGeometry[members[0]]
      const lowness = Math.max(...members.map((index) => edgeGeometry[index].sourceY)) + targetY
      return { key, members, sourceStage, targetStage, lowness }
    })
    .sort((a, b) => a.lowness - b.lowness || a.members[0] - b.members[0])
  // Left to right in a gap: the risers from the band before it (the lane
  // nearest the titles nearest that band), the channel, and the drops into the
  // band after it (the lane nearest the titles nearest that band).
  const risers = groups.map((_, stage) => laneGroups.filter((group) => group.sourceStage === stage))
  const drops = groups.map((_, stage) => laneGroups.filter((group) => group.targetStage === stage).reverse())
  const riserX = new Map<string, number>()
  const dropX = new Map<string, number>()
  const lineSpan = (count: number) => Math.max(0, count - 1) * LINE_GAP
  nextX = stages[0].x
  stages.forEach((stage, index) => {
    const shift = nextX - stage.x
    stage.x = nextX
    groupedNodes[index].forEach((node) => { positions.get(node.node_id)!.x += shift })
    const lanes = channels[index]
    const rising = risers[index]
    const dropping = drops[index + 1] ?? []
    const offsets = laneOffsets(channelGroups.filter((group) => group.sourceStage === index), lanes.length)
    const channelWidth = lanes.length === 0 ? 0 : offsets.at(-1)! + (lanes[0].width + lanes.at(-1)!.width) / 2
    const parts = [
      ...(rising.length > 0 ? [lineSpan(rising.length)] : []),
      ...(lanes.length > 0 ? [channelWidth] : []),
      ...(dropping.length > 0 ? [lineSpan(dropping.length)] : []),
    ]
    const content = parts.reduce((width, part) => width + part, 0) + Math.max(0, parts.length - 1) * LABEL_GAP
    const gap = Math.max(STAGE_GAP, content + CHANNEL_PADDING * 2)
    const left = stage.x + stage.width + CHANNEL_PADDING
    const right = stage.x + stage.width + gap - CHANNEL_PADDING
    rising.forEach((group, order) => riserX.set(group.key, left + order * LINE_GAP))
    dropping.forEach((group, order) => dropX.set(group.key, right - lineSpan(dropping.length) + order * LINE_GAP))
    // The channel's labels are centered in the room the risers and drops leave.
    const channelLeft = rising.length > 0 ? left + lineSpan(rising.length) + LABEL_GAP : left
    const channelRight = dropping.length > 0 ? right - lineSpan(dropping.length) - LABEL_GAP : right
    const firstX = (channelLeft + channelRight - channelWidth) / 2 + (lanes[0]?.width ?? 0) / 2
    lanes.forEach((lane, order) => { lane.x = firstX + offsets[order] })
    nextX += stage.width + gap
  })
  // Drops into the first band come down before it.
  drops[0].forEach((group, order) => dropX.set(group.key, stages[0].x - CHANNEL_PADDING - lineSpan(drops[0].length) + order * LINE_GAP))
  // A group's label rides its lane above the middle of a column of a band the
  // lane passes wholly (the one nearest the lane's middle), never above a gap,
  // where the risers and drops run, nor above a column gutter, where the ways
  // into a band's first row turn down, and one starting under the label would
  // read as its own.
  const laneOf = new Map(laneGroups.flatMap((group, lane) => group.members.map((index) => [index, { group, lane }] as const)))
  const laneLabelX = (group: (typeof laneGroups)[number]) => {
    const middle = (riserX.get(group.key)! + dropX.get(group.key)!) / 2
    const [first, last] = [Math.min(group.sourceStage + 1, group.targetStage), Math.max(group.targetStage - 1, group.sourceStage)]
    return stages
      .flatMap((stage, index) => (index < first || index > last ? [] : Array.from({ length: columns[index] }, (_, column) =>
        stage.x + 24 + column * (NODE_WIDTH + COLUMN_GAP) + NODE_WIDTH / 2)))
      .reduce((closest, center) => Math.abs(center - middle) < Math.abs(closest - middle) ? center : closest)
  }

  const steps = new Map(graph.nodes.map((node) => [node.node_id, node]))
  const edges = graph.edges.map((edge, index): Edge<TraceEdgeData> => {
    const source = positions.get(edge.parent_id)!
    const target = positions.get(edge.child_id)!
    const { sourceNeedsGutter, targetNeedsGutter, sourceY, targetY, labelWidth, label } = edgeGeometry[index]
    const lane = laneOf.get(index)
    const channel = channelOf.get(index)
    const labelX = channel !== undefined ? channels[channel.sourceStage][channel.lane].x : laneLabelX(lane!.group)
    const labelY = channel !== undefined ? channel.labelY : LANE_TOP + lane!.lane * LANE_PITCH
    const data: TraceEdgeData = { edgeType: edge.edge_type, points: [], labelX, labelY, labelWidth }
    // Inner columns leave and enter through their row gutters. This keeps
    // every real edge visible without drawing through neighboring cards.
    if (sourceNeedsGutter) {
      const x = source.x + NODE_WIDTH + COLUMN_INSET
      data.points.push({ x, y: source.y + NODE_HEIGHT / 2 }, { x, y: sourceY })
    }
    if (lane === undefined) {
      data.points.push({ x: labelX, y: sourceY }, { x: labelX, y: targetY })
    } else {
      const rise = riserX.get(lane.group.key)!
      const drop = dropX.get(lane.group.key)!
      data.points.push({ x: rise, y: sourceY }, { x: rise, y: labelY }, { x: drop, y: labelY }, { x: drop, y: targetY })
    }
    if (targetNeedsGutter) {
      const x = target.x - COLUMN_INSET
      data.points.push({ x, y: targetY }, { x, y: target.y + NODE_HEIGHT / 2 })
    }
    // The relationship's whole route and label. The view draws its pieces
    // (paths); this is what assistive technology reads of it, once, by name.
    return {
      id: `edge-${index}`,
      source: edge.parent_id,
      target: edge.child_id,
      label,
      data,
      type: 'relationship',
      ariaLabel: relationshipName(edge.edge_type, steps.get(edge.parent_id)!, steps.get(edge.child_id)!),
      selectable: false,
      focusable: false,
    }
  })

  // What the view draws: every stretch of every route once (drawnPieces), and
  // one label per group. A lane's label keeps its place above a band. A
  // channel group's label rides a stretch that all of its relationships run
  // along, so it is lit whenever one of them is traced, and that only
  // relationships of their kind into their step run along, so it names no
  // other line (a group level with its step shares every stretch: with the
  // step's other ways out, then with the others of its kind into the step):
  // at the middle of the stretch nearest its step, or as near it as keeps
  // every other line and label out of it, within the gap.
  const pieces = drawnPieces(edges.map(({ source, target, data }) => [
    { x: positions.get(source)!.x + NODE_WIDTH, y: positions.get(source)!.y + NODE_HEIGHT / 2 },
    ...data!.points,
    { x: positions.get(target)!.x, y: positions.get(target)!.y + NODE_HEIGHT / 2 },
  ]))
  const placed: Rect[] = []
  const labelsOf = new Map<(typeof pieces)[number], TracePathData['labels']>()
  const label = (members: number[], x: number, y: number, piece: (typeof pieces)[number] | undefined) => {
    const { labelWidth: width } = edges[members[0]].data!
    members.forEach((index) => Object.assign(edges[index].data!, { labelX: x, labelY: y }))
    placed.push(labelBox(x, y, width))
    if (piece !== undefined) labelsOf.set(piece, [...(labelsOf.get(piece) ?? []), { text: edges[members[0]].label as string, x, y, width, members }])
  }
  // The pieces carrying all of a group's relationships and, besides them, only
  // relationships of their kind into their step.
  const carrying = (members: number[]) => {
    const { edge_type: kind, child_id: step } = graph.edges[members[0]]
    return pieces.filter((piece) => members.every((member) => piece.members.includes(member))
      && piece.members.every((member) => graph.edges[member].edge_type === kind && graph.edges[member].child_id === step))
  }
  laneGroups.forEach((group) => {
    const { labelX, labelY } = edges[group.members[0]].data!
    label(group.members, labelX, labelY, carrying(group.members).find((piece) => runs(piece.corners).some((run) => onRun(run, { x: labelX, y: labelY }))))
  })
  channelGroups.forEach((group) => {
    const { labelX, labelY, labelWidth: width } = edges[group.members[0]].data!
    const room = { left: stages[group.sourceStage].x + stages[group.sourceStage].width + CHANNEL_PADDING, right: stages[group.sourceStage + 1].x - CHANNEL_PADDING }
    const { edge_type: kind, child_id: step } = graph.edges[group.members[0]]
    // Every line in the gap that none of the group's relationships runs along
    // (a line one of them shares is the group's own), the only ones a label
    // within it can meet; a line of another kind or into another step would
    // be named by it, one of its kind into its step only covered.
    const inGap = pieces.filter((piece) => !piece.members.some((member) => group.members.includes(member)))
      .map((piece) => ({
        piece,
        foreign: piece.members.some((member) => graph.edges[member].edge_type !== kind || graph.edges[member].child_id !== step),
        runs: runs(piece.corners).filter(([from, to]) => Math.min(from.x, to.x) < room.right + LEVEL_CLEARANCE && Math.max(from.x, to.x) > room.left - LEVEL_CLEARANCE),
      }))
    const places = carrying(group.members).flatMap((piece) => {
      const others = inGap.filter((other) => other.piece !== piece)
      return runs(piece.corners)
        .flatMap((run) => placesOn(run, width, others.flatMap((other) => other.runs)))
        .filter(({ x }) => x - width / 2 >= room.left - 1e-6 && x + width / 2 <= room.right + 1e-6)
        .map((place) => {
          const box = labelBox(place.x, place.y, width)
          // What a place costs, if no place is clear: a line through the label
          // costs most, one only near it least, and one of another kind or
          // step (which the label would seem to name) ten times one of its
          // own kind into its step; a place overlapping a label is never taken.
          const cost = others.reduce((sum, other) => {
            const through = other.runs.some((run) => crosses(box, run))
            const near = through || other.runs.some((run) => meets(box, run))
            return sum + (through ? 100 : near ? 1 : 0) * (other.foreign ? 10 : 1)
          }, 0)
          return { piece, ...place, cost, blocked: placed.some((other) => overlaps(box, other)) }
        })
        .filter((place) => !place.blocked)
    })
    const best = places.sort((a, b) => a.cost - b.cost || Math.hypot(a.x - labelX, a.y - labelY) - Math.hypot(b.x - labelX, b.y - labelY))[0]
    if (best !== undefined) label(group.members, best.x, best.y, best.piece)
    else label(group.members, labelX, labelY, pieces.find((piece) => piece.members.some((member) => group.members.includes(member))
      && runs(piece.corners).some((run) => onRun(run, { x: labelX, y: labelY }))))
  })
  const paths = pieces.map((piece, index): Edge<TracePathData> => ({
    id: `piece-${index}`,
    source: edges[piece.members[0]].source,
    target: edges[piece.members[0]].target,
    data: {
      points: piece.corners,
      fromHandle: piece.fromHandle,
      toHandle: piece.toHandle,
      members: piece.members,
      labels: labelsOf.get(piece) ?? [],
    },
    type: 'decision',
    selectable: false,
    focusable: false,
    // A picture of the routes, not a relationship: a stretch several share is one piece.
    domAttributes: { 'aria-hidden': true },
  }))

  return {
    stages,
    paths,
    nodes: graph.nodes.map((step) => ({
      id: step.node_id,
      type: 'decision',
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      position: positions.get(step.node_id)!,
      data: { step },
      ariaLabel: `${STAGE_LABELS[step.node_type]}, ${ACTION_LABELS[step.action]}: ${step.symptom || 'No symptom recorded'}. Rule: ${step.rule_name || 'not recorded'}. Host: ${step.hostname ?? 'not recorded'}`,
    })),
    edges,
    // Everything drawn: the bands, and every route and label (the lanes run
    // inside the bands, the risers and drops in the gaps between them).
    bounds: enclose([...stages, ...edges.flatMap((edge) => edgeBoxes(edge.data!))]),
  }
}

type Point = { x: number; y: number }
type Run = readonly [Point, Point]

// A label's box: its fixed size, centered on its place.
const labelBox = (x: number, y: number, width: number): Rect => ({ x: x - width / 2, y: y - EDGE_LABEL_HEIGHT / 2, width, height: EDGE_LABEL_HEIGHT })
const runs = (corners: Point[]): Run[] => corners.slice(1).map((point, index) => [corners[index], point] as const)
const onRun = ([from, to]: Run, { x, y }: Point) =>
  x >= Math.min(from.x, to.x) && x <= Math.max(from.x, to.x) && y >= Math.min(from.y, to.y) && y <= Math.max(from.y, to.y)
// Whether a straight run passes through a label's box.
const crosses = (box: Rect, [from, to]: Run) =>
  Math.min(from.x, to.x) < box.x + box.width && Math.max(from.x, to.x) > box.x
  && Math.min(from.y, to.y) < box.y + box.height && Math.max(from.y, to.y) > box.y
// Whether a straight run comes too near a label's box: within the clearance,
// or, level with the label (a level run crossing its height, an upright one
// its width), within the level clearance along its way.
function meets(box: Rect, [from, to]: Run): boolean {
  const upright = from.x === to.x
  const level = upright ? from.x > box.x && from.x < box.x + box.width : from.y > box.y && from.y < box.y + box.height
  const alongX = !upright && level ? LEVEL_CLEARANCE : LABEL_CLEARANCE
  const alongY = upright && level ? LEVEL_CLEARANCE : LABEL_CLEARANCE
  return Math.min(from.x, to.x) < box.x + box.width + alongX && Math.max(from.x, to.x) > box.x - alongX
    && Math.min(from.y, to.y) < box.y + box.height + alongY && Math.max(from.y, to.y) > box.y - alongY
}
const overlaps = (a: Rect, b: Rect) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height

// The places a label can take on a straight run: centered on it, and wholly
// on it when the run is long enough (a run shorter than the label only at its
// middle); at the run's middle, and just clear of each line near it, on
// either side of the stretch of places where the label would meet that line.
function placesOn([from, to]: Run, width: number, others: Run[]): Point[] {
  const upright = from.x === to.x
  const [start, end] = upright ? [Math.min(from.y, to.y), Math.max(from.y, to.y)] : [Math.min(from.x, to.x), Math.max(from.x, to.x)]
  const half = upright ? EDGE_LABEL_HEIGHT / 2 : width / 2
  const middle = (start + end) / 2
  const along = [middle, ...others.flatMap(([a, b]) => {
    // Where the line lies along the run, and how far from it the label's
    // middle must keep: farther for a line level with the label.
    const [near, far] = upright ? [Math.min(a.y, b.y), Math.max(a.y, b.y)] : [Math.min(a.x, b.x), Math.max(a.x, b.x)]
    const across = upright ? (a.x + b.x) / 2 : (a.y + b.y) / 2
    const reach = upright ? width / 2 : EDGE_LABEL_HEIGHT / 2
    const acrossLine = upright ? a.x === b.x : a.y === b.y
    const level = acrossLine && Math.abs(across - (upright ? from.x : from.y)) < reach
    const keep = half + (level ? LEVEL_CLEARANCE : LABEL_CLEARANCE)
    return [near - keep, far + keep]
  })].filter((value) => end - start < half * 2 ? value === middle : value >= start + half && value <= end - half)
  return along.map((value) => (upright ? { x: from.x, y: value } : { x: value, y: from.y }))
}

// Every stretch of every route once, from each route's corners, handle to
// handle: each route's straight runs are cut wherever a run on the same line
// starts or ends, every stretch between two cuts is one, carrying the
// relationships running along it (in their direction), and a piece runs on
// from one stretch to the next wherever its relationships all run on
// together and no other joins them: it ends only where its own routes join
// others or part, never where an unrelated route only touches it, so its
// corners there stay rounded and two routes touching never read as one line.
function drawnPieces(routes: Point[][]) {
  // Coordinates of one line are computed along different routes' ways: rounded
  // to a millionth of a pixel, they meet exactly.
  const round = (value: number) => Math.round(value * 1e6) / 1e6
  const at = ({ x, y }: Point) => `${x} ${y}`
  const ways = routes.map((points) => points.map(({ x, y }) => ({ x: round(x), y: round(y) })))
  const lines = new Map<string, { member: number; from: Point; to: Point }[]>()
  ways.forEach((points, member) => points.slice(1).forEach((to, index) => {
    const from = points[index]
    if (from.x === to.x && from.y === to.y) return
    const line = from.y === to.y ? `y ${from.y}` : `x ${from.x}`
    lines.set(line, [...(lines.get(line) ?? []), { member, from, to }])
  }))
  type Stretch = { from: Point; to: Point; members: number[] }
  const stretches: Stretch[] = []
  lines.forEach((onLine, line) => {
    const along = (point: Point) => (line.startsWith('y') ? point.x : point.y)
    const point = (value: number) => (line.startsWith('y') ? { x: value, y: onLine[0].from.y } : { x: onLine[0].from.x, y: value })
    const cuts = [...new Set(onLine.flatMap((run) => [along(run.from), along(run.to)]))].sort((a, b) => a - b)
    cuts.slice(1).forEach((end, index) => {
      const start = cuts[index]
      for (const forward of [true, false]) {
        const members = onLine
          .filter((run) => (along(run.to) > along(run.from)) === forward
            && Math.min(along(run.from), along(run.to)) <= start && Math.max(along(run.from), along(run.to)) >= end)
          .map((run) => run.member)
        if (members.length > 0) {
          stretches.push({ from: point(forward ? start : end), to: point(forward ? end : start), members: [...new Set(members)].sort((a, b) => a - b) })
        }
      }
    })
  })
  const leaving = new Map<string, Stretch[]>()
  const arriving = new Map<string, Stretch[]>()
  stretches.forEach((stretch) => {
    leaving.set(at(stretch.from), [...(leaving.get(at(stretch.from)) ?? []), stretch])
    arriving.set(at(stretch.to), [...(arriving.get(at(stretch.to)) ?? []), stretch])
  })
  const sharing = (members: number[]) => (other: Stretch) => other.members.some((member) => members.includes(member))
  const next = (stretch: Stretch) => {
    const out = (leaving.get(at(stretch.to)) ?? []).filter(sharing(stretch.members))
    if (out.length !== 1 || out[0].members.join() !== stretch.members.join()) return undefined
    return arriving.get(at(stretch.to))!.filter(sharing(out[0].members)).length === 1 ? out[0] : undefined
  }
  const continued = new Set(stretches.map(next))
  return stretches.filter((stretch) => !continued.has(stretch)).map((first) => {
    const chain = [first]
    for (let stretch = next(first); stretch !== undefined && chain.length <= stretches.length; stretch = next(stretch)) chain.push(stretch)
    // Its corners: a point where it runs straight on is none.
    const corners = [first.from, ...chain.map((stretch) => stretch.to)].filter((point, i, all) => i === 0 || i === all.length - 1
      || !((all[i - 1].x === point.x && point.x === all[i + 1].x) || (all[i - 1].y === point.y && point.y === all[i + 1].y)))
    const { members } = first
    return {
      members,
      corners,
      fromHandle: members.every((member) => at(ways[member][0]) === at(corners[0])),
      toHandle: members.every((member) => at(ways[member].at(-1)!) === at(corners.at(-1)!)),
    }
  })
}
