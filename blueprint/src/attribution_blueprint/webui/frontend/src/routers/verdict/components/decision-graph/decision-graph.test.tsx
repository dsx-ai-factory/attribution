// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { Position } from '@xyflow/react'
import { SWRConfig } from 'swr'
import type { DecisionGraph, DecisionNode, RunDetail, RunSummary } from '@/interfaces/attribution'
import { ApiContext } from '@/services/api-context'
import { ApiServerService } from '@/services/api.server.service'
import { DecisionGraphAction } from './dialog'
import { TraceEdge } from './edge'
import { DecisionGraphView } from './index'
import { EDGE_LABEL_HEIGHT, layoutGraph, NODE_HEIGHT, NODE_WIDTH } from './layout'
import * as graphLayout from './layout'
import { traceRelationships } from './trace'

const referenceMocks = vi.hoisted(() => ({ docs: vi.fn() }))
vi.mock('@/hooks/use-symptom-documentation', () => ({ useSymptomDocumentation: referenceMocks.docs }))

const evidence: DecisionNode = {
  node_id: '__proto__', node_type: 'EXTRACTION', action: 'CREATED',
  symptom: 'Same symptom', rule_name: 'extract-rule', hostname: 'node-a',
  line_num: 0, obs_source: 'syslog', timestamp: '2026-08-01T00:00:00Z',
  datum: '<script>alert("evidence")</script>',
}
const graph: DecisionGraph = {
  nodes: [
    evidence,
    { ...evidence, node_id: 'constructor', hostname: null },
    { node_id: '', node_type: 'CASCADE_PRUNE', action: 'PRUNED', symptom: 'Same symptom', rule_name: 'prune-rule', hostname: null, reason: 'A recorded reason' },
    { node_id: 'toString', node_type: 'OUTCOME', action: 'MATCHED', symptom: 'Final symptom', rule_name: 'symptom_to_outcome', hostname: null },
    { node_id: 'unknown', node_type: 'UNKNOWN', action: 'UNKNOWN', symptom: '', rule_name: null, hostname: null },
  ],
  edges: [
    { parent_id: '__proto__', child_id: '', edge_type: 'PRUNES' },
    { parent_id: '__proto__', child_id: 'toString', edge_type: 'MATCHES' },
    { parent_id: 'constructor', child_id: 'toString', edge_type: 'MATCHES' },
  ],
}

const runDetail: RunDetail = {
  id: 'graph-run', jobId: 'graph-job', workload: 'Training', attempt: 1, attemptUid: 'attempt-1',
  user: 'alice', status: 'FAILED', nodes: 1, nodeIds: ['node-a'], elapsedH: 1, gpusPerNode: 8, gpuCount: 8,
  endTime: '2026-08-01T01:00:00Z', startTime: '2026-08-01T00:00:00Z', outcome: 'hardware',
  verdictLabel: 'Hardware', detail: 'A recorded hardware symptom determined this result.',
  rootSymptom: { symptom: 'Final symptom', rule: 'symptom_to_outcome', time: '', line: '' },
  symptoms: [], responsibleNodes: ['node-a'], pipeline: '1', service: '2', log: [], decisionGraphState: 'available',
  attribution: {
    rootSymptom: { state: 'absent', data: null }, symptomSites: { state: 'absent', data: null },
    decisionGraph: { state: 'available', data: graph },
  },
}

function renderAction(run: RunSummary = runDetail, detail?: RunDetail) {
  const api = new ApiServerService()
  const getRun = vi.spyOn(api, 'getRun').mockResolvedValue(runDetail)
  const rowNavigation = vi.fn()
  render(
    <SWRConfig value={{ provider: () => new Map(), revalidateOnFocus: false, revalidateOnReconnect: false }}>
      <ApiContext.Provider value={api}>
        <div onClick={rowNavigation}><DecisionGraphAction run={run} detail={detail} /></div>
      </ApiContext.Provider>
    </SWRConfig>,
  )
  return { getRun, rowNavigation }
}

beforeEach(() => {
  referenceMocks.docs.mockReturnValue({ data: undefined })
  // Geometry and pan/zoom are checked in the real browser; jsdom has no layout.
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe('selected decision relationships', () => {
  const relationships = (pairs: [string, string][]): DecisionGraph['edges'] => pairs.map(([parent_id, child_id]) => ({
    parent_id, child_id, edge_type: 'MATCHES',
  }))

  it('traces upstream and downstream independently without including siblings or a bypass between traced nodes', () => {
    const edges = relationships([
      ['A', 'S'], ['S', 'B'], ['A', 'X'], ['Y', 'B'], ['A', 'B'],
    ])
    const original = structuredClone(edges)
    edges.forEach(Object.freeze)
    Object.freeze(edges)

    const trace = traceRelationships(edges, 'S')

    expect(trace.nodeIds).toEqual(new Set(['A', 'S', 'B']))
    expect(trace.edgeIndexes).toEqual(new Set([0, 1]))
    expect(edges).toEqual(original)
  })

  it('includes every hop through upstream and downstream branches and convergence', () => {
    const edges = relationships([
      ['U', 'A'], ['U', 'C'], ['A', 'S'], ['C', 'S'],
      ['S', 'B'], ['S', 'D'], ['B', 'V'], ['D', 'V'], ['V', 'W'],
      ['A', 'side'], ['other', 'B'], ['A', 'B'], ['disconnected', 'island'],
    ])

    const trace = traceRelationships(edges, 'S')

    expect(trace.nodeIds).toEqual(new Set(['U', 'A', 'C', 'S', 'B', 'D', 'V', 'W']))
    expect(trace.edgeIndexes).toEqual(new Set([0, 1, 2, 3, 4, 5, 6, 7, 8]))
  })

  it('terminates cycles while preserving self and parallel edges, opaque IDs and isolated selections', () => {
    const edges = relationships([
      ['', '__proto__'], ['__proto__', 'constructor'], ['constructor', ''],
      ['__proto__', '__proto__'], ['', '__proto__'], ['other', 'toString'],
    ])
    const original = structuredClone(edges)

    const trace = traceRelationships(edges, '')

    expect(trace.nodeIds).toEqual(new Set(['', '__proto__', 'constructor']))
    expect(trace.edgeIndexes).toEqual(new Set([0, 1, 2, 3, 4]))
    expect(traceRelationships(edges, 'isolated')).toEqual({ nodeIds: new Set(['isolated']), edgeIndexes: new Set() })
    expect(traceRelationships([], '')).toEqual({ nodeIds: new Set(['']), edgeIndexes: new Set() })
    expect(edges).toEqual(original)
  })
})

describe('decision graph', () => {
  it.each([
    { ruleName: null, symptom: '', label: 'Extraction' },
    { ruleName: '', symptom: '', label: 'Extraction' },
    { ruleName: '', symptom: 'Recorded symptom', label: 'Recorded symptom' },
  ])('keeps connected steps identifiable when rule=$ruleName and symptom=$symptom', ({ ruleName, symptom, label }) => {
    render(<DecisionGraphView graph={{
      nodes: [{ ...evidence, rule_name: ruleName, symptom }, graph.nodes[3]],
      edges: [{ parent_id: evidence.node_id, child_id: graph.nodes[3].node_id, edge_type: 'MATCHES' }],
    }} />)
    const details = screen.getByRole('complementary', { name: 'Selected decision step' })
    const connected = within(details).getByRole('button', { name: `From MATCHES: ${label}, Extraction, node-a` })
    expect(within(connected).getByText(label)).toBeTruthy()
    fireEvent.click(connected)
    expect(screen.getByRole('combobox', { name: 'Inspect decision step' }).textContent).toBe(`Extraction · ${label}`)
  })

  it('follows the selected step exact symptom key in the inspector without a summary link', async () => {
    const reference = { id: 'canonical-id', title: 'Extraction documentation', doc_url: 'https://docs.example/canonical-id/' }
    const finalReference = { id: 'final-id', title: 'Outcome documentation', doc_url: 'https://docs.example/final-id/' }
    referenceMocks.docs.mockReturnValue({ data: { symptoms: {
      'canonical-id': reference, 'Same symptom': reference, 'Final symptom': finalReference,
    } } })
    const detail: RunDetail = {
      ...runDetail,
      attribution: {
        ...runDetail.attribution,
        rootSymptom: { state: 'available', data: {
          symptom: evidence.symptom, catalog_id: 'canonical-id',
          source: { observation_id: '0', source: 'syslog', pos: 0, line: null, rule: null, time: null },
        } },
        decisionGraph: { state: 'available', data: {
          ...graph,
          nodes: graph.nodes.map((node) => node.node_id === 'unknown'
            ? { ...node, symptom: 'Unmapped symptom', rule_name: 'canonical-id' } : node),
        } },
      },
    }
    renderAction(detail, detail)
    fireEvent.click(screen.getByRole('button', { name: 'View Attribution Logic' }))
    const doc = await screen.findByRole('link', { name: /Symptom documentation/ })
    expect(doc.getAttribute('href')).toBe(finalReference.doc_url)
    expect(doc.getAttribute('target')).toBe('_blank')
    const details = await screen.findByRole('complementary', { name: 'Selected decision step' })
    expect(within(details).getByRole('link', { name: /Symptom documentation/ })).toBe(doc)
    expect(doc.parentElement).toBe(within(details).getByRole('heading', { name: 'Final symptom' }).parentElement)
    expect(screen.getAllByRole('link', { name: /Symptom documentation/ })).toHaveLength(1)
    fireEvent.click(within(details).getByRole('button', { name: 'From MATCHES: extract-rule, Extraction, node-a' }))
    expect(within(details).getByRole('link', { name: /Symptom documentation/ }).getAttribute('href')).toBe(reference.doc_url)
    expect(within(details).getByText('node-a')).toBeTruthy()
    expect(screen.getAllByRole('link', { name: /Symptom documentation/ })).toHaveLength(1)

    fireEvent.keyDown(screen.getByLabelText('Inspect decision step'), { key: 'ArrowDown' })
    fireEvent.click(screen.getByRole('option', { name: /Unknown stage/ }))
    expect(within(details).getByRole('heading', { name: 'Unmapped symptom' })).toBeTruthy()
    expect(screen.queryByRole('link', { name: /Symptom documentation/ })).toBeNull()
  })

  it('links a selected symptom exact API key when the root has no catalog ID', async () => {
    referenceMocks.docs.mockReturnValue({ data: { symptoms: {
      'Final symptom': { id: 'canonical-id', title: 'Final symptom', doc_url: 'https://docs.example/canonical-id/' },
    } } })
    renderAction(runDetail, runDetail)
    fireEvent.click(screen.getByRole('button', { name: 'View Attribution Logic' }))
    const details = await screen.findByRole('complementary', { name: 'Selected decision step' })
    expect(within(details).getByRole('link', { name: /Symptom documentation/ }).getAttribute('href')).toBe('https://docs.example/canonical-id/')
    expect(screen.getAllByRole('link', { name: /Symptom documentation/ })).toHaveLength(1)
  })

  it.each([0, -0.000022, 0.000022, 5])('keeps rounded edge tangents on their segments with a %s handle offset', (offset) => {
    const source = { x: -0.00014, y: offset }
    const target = { x: 100.00008, y: 80 + offset }
    const points = [source, { x: 50, y: 0 }, { x: 50, y: 80 }, target]
    const { container } = render(<svg><TraceEdge
      id="rounded" source="source" target="target"
      sourceX={source.x} sourceY={source.y} targetX={target.x} targetY={target.y}
      sourcePosition={Position.Right} targetPosition={Position.Left}
      data={{ points: points.slice(1, -1), edgeType: 'MATCHES', labelX: 50, labelY: 40, labelWidth: 72 }}
    /></svg>)
    const path = container.querySelector('.react-flow__edge-path')!.getAttribute('d')!
    const commands = [...path.matchAll(/([MLQ])\s+([^MLQ]+)/g)].map((match) => ({
      type: match[1], values: match[2].trim().split(/\s+/).map(Number),
    }))
    expect(commands.map(({ type }) => type)).toEqual(['M', 'L', 'Q', 'L', 'Q', 'L'])
    for (let index = 1; index < points.length - 1; index++) {
      const corner = points[index]
      const entry = commands[index * 2 - 1].values
      const curve = commands[index * 2].values
      expect(curve.slice(0, 2)).toEqual([corner.x, corner.y])
      for (const [tangent, neighbor] of [[entry, points[index - 1]], [curve.slice(2), points[index + 1]]] as const) {
        const dx = neighbor.x - corner.x, dy = neighbor.y - corner.y
        const tx = tangent[0] - corner.x, ty = tangent[1] - corner.y
        // A tangent must lie on the segment toward its neighbor, 12px from
        // the corner. This also makes the quadratic and straight sections join smoothly.
        expect(Math.abs(tx * dy - ty * dx) / Math.hypot(dx, dy)).toBeLessThan(1e-9)
        expect(tx * dx + ty * dy).toBeGreaterThan(0)
        expect(Math.hypot(tx, ty)).toBeCloseTo(12, 9)
      }
    }
  })

  it('keeps aligned handles straight when the route contains a repeated point', () => {
    const { container } = render(<svg><TraceEdge
      id="straight" source="source" target="target"
      sourceX={0} sourceY={0} targetX={100} targetY={0}
      sourcePosition={Position.Right} targetPosition={Position.Left}
      data={{ points: [{ x: 50, y: 0 }, { x: 50, y: 0 }], edgeType: 'MATCHES', labelX: 50, labelY: 0, labelWidth: 72 }}
    /></svg>)
    const coordinates = container.querySelector('.react-flow__edge-path')!.getAttribute('d')!
      .replace(/[MLQ]/g, '').trim().split(/\s+/).map(Number)
    expect(coordinates.slice(0, 2)).toEqual([0, 0])
    expect(coordinates.slice(-2)).toEqual([100, 0])
    expect(coordinates.every((value, index) => Number.isFinite(value) && (index % 2 === 0 ? value >= 0 && value <= 100 : value === 0))).toBe(true)
  })

  it.each([
    { counts: [0, 0, 0], columns: [1, 1, 1] },
    { counts: [5, 3, 1], columns: [2, 1, 1] },
    { counts: [40, 0, 1], columns: [6, 1, 1] },
    { counts: [0, 1, 40], columns: [1, 1, 6] },
    { counts: [11, 7, 2, 3], columns: [2, 2, 1, 1] },
    { counts: [250, 250, 250, 250], columns: [8, 8, 8, 8] },
    { counts: [1000, 1000, 1000, 1000], columns: [16, 16, 16, 16] },
  ])('balances stage counts $counts without losing or overlapping steps', ({ counts, columns }) => {
    const types = ['EXTRACTION', 'SCRUB', 'OUTCOME', 'UNKNOWN'] as const
    const input: DecisionGraph = {
      nodes: counts.flatMap((count, stage) => Array.from({ length: count }, (_, index) => ({
        ...evidence, node_id: `${stage}-${index}`, node_type: types[stage],
      }))),
      edges: [],
    }
    const { nodes, stages } = layoutGraph(input)
    expect(nodes.map((node) => node.id)).toEqual(input.nodes.map((node) => node.node_id))
    expect(stages.map((stage) => stage.count)).toEqual(counts)
    expect(stages.map((stage) => stage.width)).toEqual(columns.map((count) => count * NODE_WIDTH + (count - 1) * 32 + 48))
    expect(new Set(stages.map((stage) => stage.height)).size).toBe(1)
    for (const [index, stage] of stages.entries()) {
      const members = nodes.filter((node) => node.id.startsWith(`${index}-`))
      const occupied = new Set(members.map((node) => `${node.position.x},${node.position.y}`))
      expect(occupied.size).toBe(counts[index])
      expect(members.every(({ position }) => position.x >= stage.x && position.x + NODE_WIDTH <= stage.x + stage.width
        && position.y >= stage.y && position.y + NODE_HEIGHT <= stage.y + stage.height)).toBe(true)
    }
  })

  it('lays out opaque IDs and disconnected steps without collapsing evidence or changing relationships', () => {
    const original = structuredClone(graph)
    const { nodes, edges, stages } = layoutGraph(graph)
    expect(nodes.map((node) => node.id)).toEqual(graph.nodes.map((node) => node.node_id))
    expect(nodes.every((node) => Number.isFinite(node.position.x) && Number.isFinite(node.position.y))).toBe(true)
    expect(new Set(nodes.map((node) => `${node.position.x},${node.position.y}`)).size).toBe(graph.nodes.length)
    expect(edges.map((edge) => [edge.source, edge.target, edge.data?.edgeType])).toEqual(graph.edges.map((edge) => [edge.parent_id, edge.child_id, edge.edge_type]))
    expect(edges.map((edge) => edge.label)).toEqual(['Prunes', 'Matches', 'Matches'])
    expect(stages.map((stage) => [stage.key, stage.count])).toEqual([['extraction', 2], ['rules', 1], ['outcome', 1], ['unknown', 1]])
    expect(nodes.find((node) => node.id === '__proto__')!.position.x).toBeLessThan(nodes.find((node) => node.id === '')!.position.x)
    expect(nodes.find((node) => node.id === '')!.position.x).toBeLessThan(nodes.find((node) => node.id === 'toString')!.position.x)
    expect(graph).toEqual(original)
  })

  it('keeps crossing and skipped relationship labels separate and attached to their real edges', () => {
    const extractions = Array.from({ length: 8 }, (_, index) => ({ ...evidence, node_id: `ext-${index}` }))
    const rules: DecisionNode[] = Array.from({ length: 4 }, (_, index) => ({ ...evidence, node_id: `rule-${index}`, node_type: 'SCRUB', action: 'REMOVED' }))
    const outcome: DecisionNode = { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED' }
    const denseGraph: DecisionGraph = {
      nodes: [...extractions, ...rules, outcome],
      edges: [
        ...extractions.flatMap((source) => rules.map((target, index) => ({ parent_id: source.node_id, child_id: target.node_id, edge_type: index % 2 === 0 ? 'DEMOTES' : 'PRUNES' }))),
        ...extractions.map((source) => ({ parent_id: source.node_id, child_id: outcome.node_id, edge_type: 'MATCHES' })),
      ],
    }
    const { nodes, edges, stages } = layoutGraph(denseGraph)
    const byId = new Map(nodes.map((node) => [node.id, node]))
    expect(edges.map((edge) => [edge.source, edge.target, edge.data!.edgeType])).toEqual(denseGraph.edges.map((edge) => [edge.parent_id, edge.child_id, edge.edge_type]))
    for (const [index, edge] of edges.entries()) {
      const label = edge.data!
      for (const other of edges.slice(index + 1)) {
        const dx = Math.abs(label.labelX - other.data!.labelX)
        const dy = Math.abs(label.labelY - other.data!.labelY)
        expect(dx >= (label.labelWidth + other.data!.labelWidth) / 2 || dy >= EDGE_LABEL_HEIGHT).toBe(true)
      }
      const source = byId.get(edge.source)!
      const target = byId.get(edge.target)!
      const points = [
        { x: source.position.x + NODE_WIDTH, y: source.position.y + NODE_HEIGHT / 2 },
        ...label.points,
        { x: target.position.x, y: target.position.y + NODE_HEIGHT / 2 },
      ]
      expect(points.slice(1).some((point, segment) => {
        const previous = points[segment]
        return label.labelX >= Math.min(point.x, previous.x) && label.labelX <= Math.max(point.x, previous.x)
          && label.labelY >= Math.min(point.y, previous.y) && label.labelY <= Math.max(point.y, previous.y)
      })).toBe(true)
      for (const node of nodes) {
        expect(label.labelX + label.labelWidth / 2 <= node.position.x
          || label.labelX - label.labelWidth / 2 >= node.position.x + NODE_WIDTH
          || label.labelY + EDGE_LABEL_HEIGHT / 2 <= node.position.y
          || label.labelY - EDGE_LABEL_HEIGHT / 2 >= node.position.y + NODE_HEIGHT).toBe(true)
      }
      expect(label.labelX - label.labelWidth / 2).toBeGreaterThanOrEqual(stages[0].x)
      expect(label.labelX + label.labelWidth / 2).toBeLessThanOrEqual(stages.at(-1)!.x + stages.at(-1)!.width)
    }
  })

  it('starts at the outcome and follows actual incoming and outgoing relationships', () => {
    const { container } = render(<DecisionGraphView graph={graph} />)
    const expectTrace = (selected: string, related: string[], muted: string[]) => {
      const nodeIds = (className: string) => new Set([...container.querySelectorAll(`.react-flow__node.${className}`)]
        .map((node) => node.getAttribute('data-id')))
      expect(nodeIds('selected')).toEqual(new Set([selected]))
      expect(nodeIds('decision-node-related')).toEqual(new Set(related))
      expect(nodeIds('decision-node-muted')).toEqual(new Set(muted))
    }
    const details = screen.getByRole('complementary', { name: 'Selected decision step' })
    expect(within(details).getByRole('heading', { name: 'Final symptom' })).toBeTruthy()
    expectTrace('toString', ['__proto__', 'constructor'], ['', 'unknown'])
    fireEvent.click(within(details).getByRole('button', { name: 'From MATCHES: extract-rule, Extraction, node-a' }))
    expectTrace('__proto__', ['', 'toString'], ['constructor', 'unknown'])
    expect(within(details).getByText('0')).toBeTruthy()
    expect(within(details).getByText(evidence.datum!)).toBeTruthy()
    expect(details.querySelector('script')).toBeNull()
    expect(within(details).getByText('2026-08-01T00:00:00Z')).toBeTruthy()
    fireEvent.click(within(details).getByRole('button', { name: /To PRUNES: prune-rule/ }))
    expectTrace('', ['__proto__'], ['constructor', 'toString', 'unknown'])
    expect(within(details).getByText('PRUNED')).toBeTruthy()
    expect(within(details).getByText('A recorded reason')).toBeTruthy()
    expect(within(details).getByText('No source evidence recorded for this step.')).toBeTruthy()
    fireEvent.keyDown(screen.getByLabelText('Inspect decision step'), { key: 'ArrowDown' })
    fireEvent.click(screen.getByRole('option', { name: /Unknown stage/ }))
    expectTrace('unknown', [], ['__proto__', 'constructor', '', 'toString'])
    expect(within(details).getByText('Unknown stage')).toBeTruthy()
    expect(within(details).getByText('No reason recorded for this step.')).toBeTruthy()
    expect(within(details).queryByRole('button')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }))
    expect(within(details).getByRole('heading', { name: 'Final symptom' })).toBeTruthy()
    expectTrace('toString', ['__proto__', 'constructor'], ['', 'unknown'])
  })

  it('inspects the first recorded step when a graph has no outcome', () => {
    render(<DecisionGraphView graph={{ nodes: [evidence], edges: [] }} />)
    expect(screen.getByRole('complementary').textContent).toContain('Same symptom')
    expect(screen.getByRole('complementary').textContent).toContain('Source position0')
  })
})

describe('decision graph dialog', () => {
  it('contains viewer failures inside the dialog and leaves the attribution usable', async () => {
    vi.spyOn(graphLayout, 'layoutGraph').mockImplementation(() => { throw new Error('Viewer failed') })
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { rowNavigation } = renderAction(runDetail, runDetail)

    fireEvent.click(screen.getByRole('button', { name: 'View Attribution Logic' }))
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Reload page' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Close attribution logic' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getByRole('button', { name: 'View Attribution Logic' })).toBeTruthy()
    expect(rowNavigation).not.toHaveBeenCalled()
  })

  it('loads only on demand and reuses the detail when the viewer is reopened', async () => {
    const { getRun, rowNavigation } = renderAction()
    expect(getRun).not.toHaveBeenCalled()
    expect(screen.queryByRole('dialog')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'View Attribution Logic' }))
    expect(await screen.findByText(runDetail.detail)).toBeTruthy()
    expect(await screen.findByRole('complementary', { name: 'Selected decision step' })).toBeTruthy()
    expect(getRun).toHaveBeenCalledExactlyOnceWith(runDetail.id)
    expect(rowNavigation).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Close attribution logic' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'View Attribution Logic' }))
    expect(await screen.findByText(runDetail.detail)).toBeTruthy()
    expect(getRun).toHaveBeenCalledTimes(1)
    expect(rowNavigation).not.toHaveBeenCalled()
  })

  it.each(['absent', 'null', 'empty', 'invalid'] as const)('keeps a %s graph unavailable without navigating the row or fetching', (state) => {
    const { getRun, rowNavigation } = renderAction({ ...runDetail, decisionGraphState: state })
    const action = screen.getByRole('button', { name: 'View Attribution Logic' })
    expect(action.getAttribute('aria-disabled')).toBe('true')
    fireEvent.click(action)
    expect(screen.getByRole('tooltip').textContent).toMatch(/No decision graph|no recorded decision steps|cannot be displayed/)
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(getRun).not.toHaveBeenCalled()
    expect(rowNavigation).not.toHaveBeenCalled()
  })

  it.each([
    ['error', 'Attribution logic could not be loaded.'],
    ['missing', 'This attribution is no longer available.'],
  ] as const)('shows loading and the %s state, then allows an explicit retry', async (result, message) => {
    const { getRun } = renderAction()
    if (result === 'error') getRun.mockRejectedValueOnce(new Error('Network unavailable'))
    else getRun.mockResolvedValueOnce(undefined)

    fireEvent.click(screen.getByRole('button', { name: 'View Attribution Logic' }))
    expect(screen.getByRole('status').textContent).toContain('Loading attribution logic')
    expect(await screen.findByText(message)).toBeTruthy()
    expect(screen.queryByRole('complementary')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByText(runDetail.detail)).toBeTruthy()
    expect(await screen.findByRole('complementary', { name: 'Selected decision step' })).toBeTruthy()
    expect(getRun).toHaveBeenCalledTimes(2)
  })

  it('uses the already loaded detail without another request', async () => {
    const { getRun } = renderAction(runDetail, runDetail)
    fireEvent.click(screen.getByRole('button', { name: 'View Attribution Logic' }))
    expect(await screen.findByText(runDetail.detail)).toBeTruthy()
    expect(await screen.findByRole('complementary', { name: 'Selected decision step' })).toBeTruthy()
    expect(getRun).not.toHaveBeenCalled()
  })
})
