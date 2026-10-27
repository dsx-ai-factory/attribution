// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { Position } from '@xyflow/react'
import { SWRConfig } from 'swr'
import type { DecisionGraph, DecisionNode, RunDetail, RunSummary } from '@/interfaces/attribution'
import { ApiContext } from '@/services/api-context'
import { ApiServerService } from '@/services/api.server.service'
import { DecisionGraphAction } from './dialog'
import { TraceEdge } from './edge'
import { CONTROLS_INSET, DecisionGraphView, FOCUS_RING } from './index'
import { EDGE_LABEL_HEIGHT, layoutGraph, NODE_HEIGHT, NODE_WIDTH } from './layout'
import * as graphLayout from './layout'
import { relationshipName } from './labels'
import { onTrace, traceRelationships } from './trace'

const referenceMocks = vi.hoisted(() => ({ docs: vi.fn() }))
vi.mock('@/hooks/use-symptom-documentation', () => ({ useSymptomDocumentation: referenceMocks.docs }))
// React Flow portals edge labels into its canvas, which jsdom never lays out
// (it lays out no edges either); a piece's labels render in place instead.
vi.mock('@xyflow/react', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@xyflow/react')>()),
  EdgeLabelRenderer: ({ children }: { children: ReactNode }) => children,
}))

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

// A cascade of demotions and matches, as the service records a symptom seen
// on several hosts: eleven extractions, five rules and the outcome.
const cascade: DecisionGraph = {
  nodes: [
    ...Array.from({ length: 11 }, (_, index) => ({ ...evidence, node_id: `ext-${index}` })),
    ...[11, 12, 13, 14, 15].map((index) => ({ ...evidence, node_id: `rule-${index}`, node_type: index === 13 || index === 14 ? 'SCRUB' as const : 'REFINER' as const, action: 'REMOVED' as const })),
    { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED' },
  ],
  edges: [
    ['ext-8', 'rule-11', 'CREATES'], ['ext-0', 'rule-12', 'DEMOTES'], ['ext-5', 'rule-12', 'DEMOTES'], ['ext-6', 'rule-12', 'DEMOTES'],
    ['ext-7', 'rule-12', 'DEMOTES'], ['ext-0', 'rule-13', 'DEMOTES'], ['ext-6', 'rule-13', 'DEMOTES'], ['ext-7', 'rule-13', 'DEMOTES'],
    ['ext-5', 'rule-14', 'DEMOTES'], ['ext-6', 'rule-14', 'DEMOTES'], ['ext-7', 'rule-14', 'DEMOTES'], ['ext-8', 'rule-15', 'DEMOTES'],
    ['ext-2', 'outcome', 'MATCHES'], ['ext-10', 'outcome', 'MATCHES'],
  ].map(([parent, child, kind]) => ({ parent_id: parent, child_id: child, edge_type: kind })),
}

// Seeded numbers from Mulberry32, a 32-bit generator in integer arithmetic,
// so no sample repeats another (an LCG in floating point loses its low bits
// and cycles).
function seededRandom(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// A graph in the Service's shape (its decision_trace.py) with a rule on a
// host, which the Service feeds from its symptom's extractions there and the
// hostless ones: a scrub on one host, two refiners fed by every extraction of
// their symptoms, and the outcome. Two groups' labels in its channel meet
// unless each starts clear of the other's lines.
const hostedRule: DecisionGraph = {
  nodes: [
    ...Array.from({ length: 10 }, (_, index) => ({ ...evidence, node_id: `ext_${index}` })),
    ...(['SCRUB', 'REFINER', 'REFINER'] as const).map((type, index): DecisionNode => ({ ...evidence, node_id: `dec_${index}`, node_type: type, action: 'REMOVED' })),
    { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED' },
  ],
  edges: [
    ['ext_2', 'dec_0', 'DEMOTES'], ['ext_4', 'dec_0', 'DEMOTES'], ['ext_1', 'dec_1', 'DEMOTES'], ['ext_9', 'dec_1', 'DEMOTES'],
    ['ext_0', 'dec_2', 'DEMOTES'], ['ext_2', 'dec_2', 'DEMOTES'], ['ext_3', 'dec_2', 'DEMOTES'], ['ext_4', 'dec_2', 'DEMOTES'],
    ['ext_5', 'dec_2', 'DEMOTES'], ['ext_8', 'dec_2', 'DEMOTES'], ['ext_1', 'outcome', 'MATCHES'], ['ext_9', 'outcome', 'MATCHES'],
  ].map(([parent, child, kind]) => ({ parent_id: parent, child_id: child, edge_type: kind })),
}

// Named graphs of every shape the drawing meets: the fixtures, the cascade, a
// rule on a host, eight extractions each demoting or pruning four rules and
// matching the outcome, every band shape up to 16 extractions matching the
// outcome and 10 rules fed from them, seeded graphs of every kind of
// relationship, and seeded graphs in the shape the Service records (its
// decision_trace.py): extractions each of one of a few symptoms, on a few hosts
// or none; rules of a symptom, each fed with one kind by its extractions (on
// its host and hostless ones, for a rule on a host): refiners removing it,
// scrubs and prunes on a host, and refiners created on a host; and the
// outcome matching every extraction of the symptom it names. Those graphs
// have at most four rules: with more, a label can still meet a line of
// another kind (a tracked follow-up).
function shapes(): [string, DecisionGraph][] {
  const extractions = Array.from({ length: 8 }, (_, index) => ({ ...evidence, node_id: `ext-${index}` }))
  const rules: DecisionNode[] = Array.from({ length: 4 }, (_, index) => ({ ...evidence, node_id: `rule-${index}`, node_type: 'SCRUB', action: 'REMOVED' }))
  const dense: DecisionGraph = {
    nodes: [...extractions, ...rules, { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED' }],
    edges: [
      ...extractions.flatMap((source) => rules.map((target, index) => ({ parent_id: source.node_id, child_id: target.node_id, edge_type: index % 2 === 0 ? 'DEMOTES' : 'PRUNES' }))),
      ...extractions.map((source) => ({ parent_id: source.node_id, child_id: 'outcome', edge_type: 'MATCHES' })),
    ],
  }
  const random = seededRandom(11)
  const types = ['EXTRACTION', 'SCRUB', 'REFINER', 'OUTCOME'] as const
  const kinds = ['CREATES', 'DEMOTES', 'PRUNES', 'MATCHES']
  const seeded = Array.from({ length: 100 }, (_, sample): [string, DecisionGraph] => {
    const nodes = Array.from({ length: 2 + Math.floor(random() * 18) }, (_, index) => ({
      ...evidence, node_id: `step-${index}`, node_type: types[Math.floor(random() * types.length)],
    }))
    const edges = Array.from({ length: Math.floor(random() * nodes.length * 2) }, () => ({
      parent_id: nodes[Math.floor(random() * nodes.length)].node_id,
      child_id: nodes[Math.floor(random() * nodes.length)].node_id,
      edge_type: kinds[Math.floor(random() * kinds.length)],
    }))
    return [`seeded ${sample}`, { nodes, edges }]
  })
  const bands = Array.from({ length: 16 * 10 }, (_, index): [string, DecisionGraph] => {
    const [extractionCount, ruleCount] = [Math.floor(index / 10) + 1, (index % 10) + 1]
    const sources = Array.from({ length: extractionCount }, (_, source) => ({ ...evidence, node_id: `ext-${source}` }))
    const fed: DecisionNode[] = Array.from({ length: ruleCount }, (_, rule) => ({ ...evidence, node_id: `rule-${rule}`, node_type: 'SCRUB', action: 'REMOVED' }))
    return [`${extractionCount} extractions, ${ruleCount} rules`, {
      nodes: [...sources, ...fed, { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED' }],
      edges: [
        ...fed.map((rule, at) => ({ parent_id: sources[at % extractionCount].node_id, child_id: rule.node_id, edge_type: 'DEMOTES' })),
        ...sources.map((source) => ({ parent_id: source.node_id, child_id: 'outcome', edge_type: 'MATCHES' })),
      ],
    }]
  })
  const ruleKinds = [
    { node_type: 'REFINER', action: 'REMOVED', edge_type: 'DEMOTES', hosted: false },
    { node_type: 'SCRUB', action: 'REMOVED', edge_type: 'DEMOTES', hosted: true },
    { node_type: 'CASCADE_PRUNE', action: 'PRUNED', edge_type: 'PRUNES', hosted: true },
    { node_type: 'REFINER', action: 'CREATED', edge_type: 'CREATES', hosted: true },
  ] as const
  const service = Array.from({ length: 200 }, (_, sample): [string, DecisionGraph] => {
    const [symptoms, hosts] = [1 + Math.floor(random() * 5), 1 + Math.floor(random() * 4)]
    const extractions = Array.from({ length: 1 + Math.floor(random() * 24) }, (_, index) => ({
      ...evidence, node_id: `ext-${index}`, symptom: `S${Math.floor(random() * symptoms)}`, hostname: random() < 0.25 ? null : `h${Math.floor(random() * hosts)}`,
    }))
    const nodes: DecisionNode[] = [...extractions]
    const edges: DecisionGraph['edges'] = []
    const count = Math.floor(random() * 5)
    for (let index = 0; index < count; index++) {
      const { hosted, edge_type, ...rule } = ruleKinds[Math.floor(random() * ruleKinds.length)]
      const [symptom, host] = [extractions[Math.floor(random() * extractions.length)].symptom, hosted ? `h${Math.floor(random() * hosts)}` : null]
      const sources = extractions.filter((source) => source.symptom === symptom && (host === null || source.hostname === host || source.hostname === null))
      if (sources.length === 0) continue
      nodes.push({ ...evidence, ...rule, node_id: `rule-${index}`, symptom, hostname: host })
      sources.forEach((source) => edges.push({ parent_id: source.node_id, child_id: `rule-${index}`, edge_type }))
    }
    const named = extractions[Math.floor(random() * extractions.length)].symptom
    extractions.filter((source) => source.symptom === named).forEach((source) => edges.push({ parent_id: source.node_id, child_id: 'outcome', edge_type: 'MATCHES' }))
    return [`service ${sample}`, { nodes: [...nodes, { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED', symptom: named }], edges }]
  })
  return [['fixture', graph], ['cascade', cascade], ['rule on a host', hostedRule], ['dense', dense], ...bands, ...seeded, ...service]
}

type Point = { x: number; y: number }
type Laid = ReturnType<typeof graphLayout.layoutGraph>
const handles = (layout: Laid, source: string, target: string) => {
  const from = layout.nodes.find((node) => node.id === source)!.position
  const to = layout.nodes.find((node) => node.id === target)!.position
  return { out: { x: from.x + NODE_WIDTH, y: from.y + NODE_HEIGHT / 2 }, in: { x: to.x, y: to.y + NODE_HEIGHT / 2 } }
}
// A relationship's whole route, handle to handle.
const routeCorners = (layout: Laid, { source, target, data }: Laid['edges'][number]) => {
  const ends = handles(layout, source, target)
  return [ends.out, ...data!.points, ends.in]
}
// A drawn piece's corners, its ends at steps included: its points, which
// meet the handles the steps' positions give.
const pieceCorners = (layout: Laid, { source, target, data }: Laid['paths'][number]) => {
  const ends = handles(layout, source, target)
  const [first, last] = [data!.points[0], data!.points.at(-1)!]
  if (data!.fromHandle) expect([first.x, first.y]).toEqual([ends.out.x, ends.out.y])
  if (data!.toHandle) expect([last.x, last.y]).toEqual([ends.in.x, ends.in.y])
  return data!.points
}
const runsOf = (points: Point[]) => points.slice(1).map((point, index) => [points[index], point] as const)
// How long a stretch two straight runs share along one line.
const shared = ([a, b]: readonly [Point, Point], [c, d]: readonly [Point, Point]) => {
  const overlap = (p: number, q: number, r: number, s: number) => Math.max(0, Math.min(Math.max(p, q), Math.max(r, s)) - Math.max(Math.min(p, q), Math.min(r, s)))
  if (a.x === b.x && c.x === d.x && Math.abs(a.x - c.x) < 1e-6) return overlap(a.y, b.y, c.y, d.y)
  if (a.y === b.y && c.y === d.y && Math.abs(a.y - c.y) < 1e-6) return overlap(a.x, b.x, c.x, d.x)
  return 0
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

function renderAction(run: RunSummary = runDetail, detail?: RunDetail, compact = false) {
  const api = new ApiServerService()
  const getRun = vi.spyOn(api, 'getRun').mockResolvedValue(runDetail)
  const rowNavigation = vi.fn()
  render(
    <SWRConfig value={{ provider: () => new Map(), revalidateOnFocus: false, revalidateOnReconnect: false }}>
      <ApiContext.Provider value={api}>
        <div onClick={rowNavigation}><DecisionGraphAction run={run} detail={detail} compact={compact} /></div>
      </ApiContext.Provider>
    </SWRConfig>,
  )
  return { getRun, rowNavigation }
}

// The value a description list pairs with a label.
// What a reader copies: a recorded identifier's break points (hidden zero-width spaces) are no text.
const copied = (node: Node) => node.textContent!.replaceAll('\u200b', '')

function valueOf(container: HTMLElement, label: string) {
  const term = within(container).getByText(label, { selector: 'dt' })
  expect(term.nextElementSibling?.tagName).toBe('DD')
  return term.nextElementSibling as HTMLElement
}

// The rendered React Flow viewport, parsed from its transform.
function viewportOf(container: HTMLElement) {
  const [, x, y, zoom] = container.querySelector<HTMLElement>('.react-flow__viewport')!.style.transform
    .match(/translate\((-?[\d.]+)px, ?(-?[\d.]+)px\) scale\(([\d.]+)\)/)!.map(Number)
  return { x, y, zoom }
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
    const connected = within(details).getByRole('button', { name: `From Matches: ${label}, Extraction, node-a` })
    expect(within(connected).getByText(label)).toBeTruthy()
    expect(within(connected).getByText('From · Matches · Extraction · node-a')).toBeTruthy()
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
    fireEvent.click(screen.getByRole('button', { name: 'View attribution logic' }))
    // The header names the root symptom as the verdict does: the recorded attribution's, over the summary's.
    expect(screen.getByRole('dialog', { name: 'Attribution logic' }).getAttribute('aria-describedby')).toBeTruthy()
    expect(screen.getByRole('dialog', { description: 'graph-job Hardware Same symptom' })).toBeTruthy()
    const doc = await screen.findByRole('link', { name: /Symptom documentation/ })
    expect(doc.getAttribute('href')).toBe(finalReference.doc_url)
    expect(doc.getAttribute('target')).toBe('_blank')
    const details = await screen.findByRole('complementary', { name: 'Selected decision step' })
    expect(within(details).getByRole('link', { name: /Symptom documentation/ })).toBe(doc)
    expect(doc.parentElement).toBe(within(details).getByRole('heading', { name: 'Final symptom' }).parentElement)
    // A mark beside the step's name: on its first line, centered on its lowercase.
    expect(doc.parentElement!.classList.contains('items-start')).toBe(true)
    expect([...doc.classList]).toEqual(expect.arrayContaining(['relative', 'top-(--mark-offset)']))
    expect(screen.getAllByRole('link', { name: /Symptom documentation/ })).toHaveLength(1)
    // A connected step's lines balance their ends, so a wrapped host never leaves one letter.
    const connected = within(details).getByRole('button', { name: 'From Matches: extract-rule, Extraction, node-a' })
    expect([...connected.querySelectorAll('span.break-words')].every((line) => line.classList.contains('text-pretty'))).toBe(true)
    fireEvent.click(connected)
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
    fireEvent.click(screen.getByRole('button', { name: 'View attribution logic' }))
    const details = await screen.findByRole('complementary', { name: 'Selected decision step' })
    expect(within(details).getByRole('link', { name: /Symptom documentation/ }).getAttribute('href')).toBe('https://docs.example/canonical-id/')
    expect(screen.getAllByRole('link', { name: /Symptom documentation/ })).toHaveLength(1)
  })

  it('draws a piece from its own points, wherever React Flow measures the steps\' handles', () => {
    // Handles measured 3px outside the steps' borders and at 95% of the way
    // (a transform around the canvas as it mounted) move no end.
    const points = [{ x: 296, y: 418 }, { x: 330, y: 418 }, { x: 330, y: 500 }, { x: 374, y: 500 }]
    const { container } = render(<svg><TraceEdge
      id="own" source="source" target="target"
      sourceX={286.75} sourceY={415.35} targetX={371} targetY={497.2}
      sourcePosition={Position.Right} targetPosition={Position.Left}
      data={{ points, fromHandle: true, toHandle: true, members: [0], labels: [] }}
    /></svg>)
    const coordinates = container.querySelector('.react-flow__edge-path')!.getAttribute('d')!
      .replace(/[MLQ]/g, '').trim().split(/\s+/).map(Number)
    expect(coordinates.slice(0, 2)).toEqual([296, 418])
    expect(coordinates.slice(-2)).toEqual([374, 500])
  })

  it.each([0, -0.000022, 0.000022, 5])('keeps rounded edge tangents on their segments with its ends %s off level', (offset) => {
    const source = { x: -0.00014, y: offset }
    const target = { x: 100.00008, y: 80 + offset }
    const points = [source, { x: 50, y: 0 }, { x: 50, y: 80 }, target]
    const { container } = render(<svg><TraceEdge
      id="rounded" source="source" target="target"
      sourceX={0} sourceY={0} targetX={0} targetY={0}
      sourcePosition={Position.Right} targetPosition={Position.Left}
      data={{ points, fromHandle: true, toHandle: true, members: [0], labels: [] }}
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

  it('draws its labels as part of the picture, hidden from assistive technology', () => {
    render(<svg><TraceEdge
      id="labelled" source="source" target="target"
      sourceX={0} sourceY={0} targetX={100} targetY={0}
      sourcePosition={Position.Right} targetPosition={Position.Left}
      data={{ points: [{ x: 0, y: 0 }, { x: 100, y: 0 }], fromHandle: true, toHandle: true, members: [0], labels: [{ text: 'Demotes', x: 50, y: 0, width: 72, members: [0] }] }}
    /></svg>)
    const label = document.querySelector('[data-slot="decision-edge-label"]')
    expect(label?.textContent).toBe('Demotes')
    expect(label?.getAttribute('aria-hidden')).toBe('true')
  })

  it('keeps a level piece straight when its route contains a repeated point', () => {
    const { container } = render(<svg><TraceEdge
      id="straight" source="source" target="target"
      sourceX={0} sourceY={0} targetX={100} targetY={0}
      sourcePosition={Position.Right} targetPosition={Position.Left}
      data={{ points: [{ x: 0, y: 0 }, { x: 50, y: 0 }, { x: 50, y: 0 }, { x: 100, y: 0 }], fromHandle: true, toHandle: true, members: [0], labels: [] }}
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
    { counts: [40, 0, 1], columns: [5, 1, 1] },
    { counts: [0, 1, 40], columns: [1, 1, 5] },
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

  it.each([
    {
      name: 'three parallel relationships beside an unrelated step',
      steps: [['a', 'EXTRACTION'], ['b', 'EXTRACTION'], ['c', 'EXTRACTION']],
      relationships: [['c', 'a'], ['c', 'a'], ['c', 'a']],
    },
    {
      name: 'a cycle through parallel relationships',
      steps: [['e0', 'EXTRACTION'], ['r0', 'SCRUB'], ['r1', 'SCRUB'], ['r2', 'SCRUB'], ['r3', 'SCRUB'], ['r4', 'SCRUB']],
      relationships: [['r1', 'r4'], ['e0', 'r3'], ['r3', 'e0'], ['e0', 'r1'], ['e0', 'r3'], ['r0', 'r2']],
    },
  ] as const)('lays out $name, and a step related to itself, without failing', ({ steps, relationships }) => {
    const input: DecisionGraph = {
      nodes: steps.map(([id, type]) => ({ ...evidence, node_id: id, node_type: type })),
      edges: [...relationships, [steps[0][0], steps[0][0]] as const].map(([parent, child]) => ({ parent_id: parent, child_id: child, edge_type: 'DEMOTES' })),
    }
    const { nodes, edges } = layoutGraph(input)
    expect(new Set(nodes.map((node) => `${node.position.x},${node.position.y}`)).size).toBe(steps.length)
    expect(edges.map((edge) => [edge.source, edge.target])).toEqual(input.edges.map((edge) => [edge.parent_id, edge.child_id]))
  })

  it('keeps relationship labels apart and on their own edges, and passes a band only in a lane above its steps', () => {
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
    const stageOf = (id: string) => stages.findIndex(({ x, width }) => byId.get(id)!.position.x >= x && byId.get(id)!.position.x < x + width)
    const stepsTop = Math.min(...nodes.map((node) => node.position.y))
    // Whether a box (a label, or a straight run of a route) reaches inside a band.
    const inBand = (box: { left: number; right: number; top: number; bottom: number }, stage: (typeof stages)[number]) =>
      box.left < stage.x + stage.width && box.right > stage.x && box.top < stage.y + stage.height && box.bottom > stage.y
    const labelBox = ({ labelX, labelY, labelWidth }: NonNullable<(typeof edges)[number]['data']>) => ({
      left: labelX - labelWidth / 2, right: labelX + labelWidth / 2, top: labelY - EDGE_LABEL_HEIGHT / 2, bottom: labelY + EDGE_LABEL_HEIGHT / 2,
    })
    // One label per place: relationships sharing a line (a lane's, or a
    // channel's of one kind into one step from one side) share its label.
    const places = [...new Map(edges.map(({ data }) => [`${data!.labelX},${data!.labelY}`, data!])).values()]
    for (const [index, label] of places.entries()) {
      for (const other of places.slice(index + 1)) {
        const dx = Math.abs(label.labelX - other.labelX)
        const dy = Math.abs(label.labelY - other.labelY)
        expect(dx >= (label.labelWidth + other.labelWidth) / 2 || dy >= EDGE_LABEL_HEIGHT).toBe(true)
      }
    }
    for (const edge of edges) {
      const label = edge.data!
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
      // A label sits wholly inside a band or wholly in a gap: never on a band's edge.
      const box = labelBox(label)
      for (const stage of stages) {
        const inside = box.left >= stage.x && box.right <= stage.x + stage.width && box.top >= stage.y && box.bottom <= stage.y + stage.height
        expect(inside || !inBand(box, stage)).toBe(true)
      }
      // A line runs through a band it neither starts nor ends in only in the lanes above that band's steps.
      const own = [stageOf(edge.source), stageOf(edge.target)]
      const crossed = points.slice(1).flatMap((point, segment) => {
        const previous = points[segment]
        const run = {
          left: Math.min(point.x, previous.x), right: Math.max(point.x, previous.x),
          top: Math.min(point.y, previous.y), bottom: Math.max(point.y, previous.y),
        }
        return stages.filter((stage, stageIndex) => !own.includes(stageIndex) && inBand(run, stage) && run.bottom >= stepsTop).map((stage) => stage.key)
      })
      expect(crossed).toEqual([])
    }
    // The eight matches into the outcome are one kind into one step: they share
    // one lane and one label, under the bands' titles and above every step, in
    // the band they pass.
    const skipped = edges.filter((edge) => edge.data!.edgeType === 'MATCHES')
    expect(new Set(skipped.map(({ data }) => `${data!.labelX},${data!.labelY}`)).size).toBe(1)
    const lane = labelBox(skipped[0].data!)
    expect(lane.top).toBeGreaterThanOrEqual(stages[0].y + 40)
    expect(lane.bottom).toBeLessThan(stepsTop)
    expect(stages.filter((stage) => inBand(lane, stage)).map((stage) => stage.key)).toEqual(['rules'])
  })

  it('keeps every route clear of the lanes\' labels, a first-row step\'s gutter included', () => {
    // Extractions that each match the outcome through a lane, and rules fed
    // from them, in every band shape up to 16 extractions and 10 rules.
    for (let extractionCount = 1; extractionCount <= 16; extractionCount++) {
      for (let ruleCount = 1; ruleCount <= 10; ruleCount++) {
        const extractions = Array.from({ length: extractionCount }, (_, index) => ({ ...evidence, node_id: `ext-${index}` }))
        const rules: DecisionNode[] = Array.from({ length: ruleCount }, (_, index) => ({ ...evidence, node_id: `rule-${index}`, node_type: 'SCRUB', action: 'REMOVED' }))
        const outcome: DecisionNode = { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED' }
        const { nodes, edges } = layoutGraph({
          nodes: [...extractions, ...rules, outcome],
          edges: [
            ...rules.map((rule, index) => ({ parent_id: extractions[index % extractionCount].node_id, child_id: rule.node_id, edge_type: 'DEMOTES' })),
            ...extractions.map((source) => ({ parent_id: source.node_id, child_id: outcome.node_id, edge_type: 'MATCHES' })),
          ],
        })
        const byId = new Map(nodes.map((node) => [node.id, node]))
        const stepsTop = Math.min(...nodes.map((node) => node.position.y))
        const laneLabels = edges.map(({ data }) => data!).filter((data) => data.labelY < stepsTop)
        expect(laneLabels.length).toBeGreaterThan(0)
        for (const { source, target, data } of edges) {
          const points = [
            { x: byId.get(source)!.position.x + NODE_WIDTH, y: byId.get(source)!.position.y + NODE_HEIGHT / 2 },
            ...data!.points,
            { x: byId.get(target)!.position.x, y: byId.get(target)!.position.y + NODE_HEIGHT / 2 },
          ]
          for (const label of laneLabels) {
            if (label.labelX === data!.labelX && label.labelY === data!.labelY) continue
            const crossing = points.slice(1).some((point, index) => {
              const previous = points[index]
              return Math.min(point.x, previous.x) < label.labelX + label.labelWidth / 2 && Math.max(point.x, previous.x) > label.labelX - label.labelWidth / 2
                && Math.min(point.y, previous.y) < label.labelY + EDGE_LABEL_HEIGHT / 2 && Math.max(point.y, previous.y) > label.labelY - EDGE_LABEL_HEIGHT / 2
            })
            expect(crossing, `${extractionCount} extractions, ${ruleCount} rules: ${source} to ${target}`).toBe(false)
          }
        }
      }
    }
  })

  it('never runs a relationship back along its own way out, within a band or across bands', () => {
    // Seeded graphs of every kind of relationship: into the next band, past a
    // band, within one and back to an earlier one, parallel and to itself.
    const random = seededRandom(7)
    const types = ['EXTRACTION', 'SCRUB', 'REFINER', 'OUTCOME'] as const
    for (let sample = 0; sample < 200; sample++) {
      const nodes = Array.from({ length: 2 + Math.floor(random() * 18) }, (_, index) => ({
        ...evidence, node_id: `step-${index}`, node_type: types[Math.floor(random() * types.length)],
      }))
      const edges = Array.from({ length: Math.floor(random() * nodes.length * 2) }, () => ({
        parent_id: nodes[Math.floor(random() * nodes.length)].node_id,
        child_id: nodes[Math.floor(random() * nodes.length)].node_id,
        edge_type: 'DEMOTES',
      }))
      const layout = layoutGraph({ nodes, edges })
      const byId = new Map(layout.nodes.map((node) => [node.id, node]))
      for (const { source, target, data } of layout.edges) {
        const points = [
          { x: byId.get(source)!.position.x + NODE_WIDTH, y: byId.get(source)!.position.y + NODE_HEIGHT / 2 },
          ...data!.points,
          { x: byId.get(target)!.position.x, y: byId.get(target)!.position.y + NODE_HEIGHT / 2 },
        ]
        const runs = points.slice(1).map((point, index) => [points[index], point] as const)
        // How long two straight runs of the route lie along one line together.
        const shared = runs.flatMap(([a, b], index) => runs.slice(index + 1).map(([c, d]) => {
          const overlap = (p: number, q: number, r: number, t: number) => Math.max(0, Math.min(Math.max(p, q), Math.max(r, t)) - Math.max(Math.min(p, q), Math.min(r, t)))
          if (a.x === b.x && c.x === d.x && a.x === c.x) return overlap(a.y, b.y, c.y, d.y)
          if (a.y === b.y && c.y === d.y && a.y === c.y) return overlap(a.x, b.x, c.x, d.x)
          return 0
        }))
        expect(Math.max(0, ...shared), `sample ${sample}: ${source} to ${target}`).toBe(0)
      }
    }
  })

  // The sweeps over every shape share one layout of each, which they only
  // read, and check a shape in plain code, then assert on all they found in
  // it: an expect for each pair of runs cost more than laying every shape out.
  describe('on every shape', () => {
    let laidOut: { name: string; input: DecisionGraph; layout: Laid }[] = []
    beforeAll(() => {
      laidOut = shapes().map(([name, input]) => ({ name, input, layout: layoutGraph(input) }))
    })

    it('draws every stretch once, carrying the relationships that run along it, and each route whole', () => {
      for (const { name, layout } of laidOut) {
        const pieces = layout.paths.map((path) => ({ path, runs: runsOf(pieceCorners(layout, path)) }))
        // No stretch is drawn twice.
        const drawnTwice = pieces.flatMap((piece, index) => pieces.slice(index + 1)
          .filter((other) => piece.runs.some((run) => other.runs.some((otherRun) => shared(run, otherRun) !== 0)))
          .map((other) => `${piece.path.id}, ${other.path.id}`))
        expect(drawnTwice, name).toEqual([])
        const broken = layout.edges.flatMap((edge, member) => {
          const own = pieces.filter(({ path }) => path.data!.members.includes(member)).flatMap(({ runs }) => runs)
          // Its route is drawn whole by the pieces carrying it (to six places),
          // each run the way the route runs.
          const whole = runsOf(routeCorners(layout, edge)).every((run) => {
            const length = Math.abs(run[1].x - run[0].x) + Math.abs(run[1].y - run[0].y)
            return Math.abs(own.reduce((sum, piece) => sum + shared(run, piece), 0) - length) < 5e-7
              && own.every((piece) => shared(run, piece) === 0 || Math.sign(piece[1].x - piece[0].x) === Math.sign(run[1].x - run[0].x)
                && Math.sign(piece[1].y - piece[0].y) === Math.sign(run[1].y - run[0].y))
          })
          // One arrow ends it, at its target step.
          const arrows = pieces.filter(({ path }) => path.data!.members.includes(member) && path.data!.toHandle).map(({ path }) => path.target)
          return [
            ...(whole ? [] : [`relationship ${member} is not drawn whole`]),
            ...(arrows.length === 1 && arrows[0] === edge.target ? [] : [`relationship ${member} ends in arrows at [${arrows.join(', ')}]`]),
          ]
        })
        expect(broken, name).toEqual([])
      }
    })

    it('labels the relationships of one kind into one step from one side once, on a stretch they alone run along, clear of every line it would misname', () => {
      for (const { name, input, layout } of laidOut.filter(({ name }) => !name.startsWith('seeded'))) {
        const runsAlong = new Map(layout.paths.map((path) => [path, runsOf(pieceCorners(layout, path))]))
        // What a label names of a relationship: its kind and the step it runs into.
        const groupOf = (member: number) => `${input.edges[member].edge_type} ${input.edges[member].child_id}`
        const labelled = layout.paths.flatMap((path) => path.data!.labels.map((label) => ({ path, label })))
        // Every relationship is named by exactly one label, of its own kind and step.
        expect(labelled.flatMap(({ label }) => label.members).sort((a, b) => a - b), name).toEqual(layout.edges.map((_, index) => index))
        const misplaced = labelled.flatMap(({ path, label }) => {
          const { x, y, width, members } = label
          const at = `${label.text} at ${x},${y}`
          const problems: string[] = []
          if (new Set(members.map(groupOf)).size !== 1) problems.push(`${at} names relationships of several kinds or steps`)
          // It rides a piece all of its relationships run along, so it is lit
          // whenever one is traced, and only others of their kind into their step.
          if (!members.every((member) => path.data!.members.includes(member))) problems.push(`${at} rides a piece one of its relationships leaves`)
          if (new Set(path.data!.members.map(groupOf)).size !== 1) problems.push(`${at} rides a piece of several kinds or steps`)
          const ridden = runsAlong.get(path)!.find(([a, b]) => x >= Math.min(a.x, b.x) && x <= Math.max(a.x, b.x) && y >= Math.min(a.y, b.y) && y <= Math.max(a.y, b.y))
          if (ridden === undefined) return [...problems, `${at} is off its piece`]
          const [from, to] = ridden
          const overhangs = from.x === to.x ? Math.abs(to.y - from.y) < EDGE_LABEL_HEIGHT : Math.abs(to.x - from.x) < width
          // No other line runs through it but one of its kind into its step,
          // where it overhangs a run too short to hold it (in the Service's
          // graphs, where two groups of one kind into one step join), which it
          // names rightly. One of another kind or into another step, which it
          // would seem to name, keeps 6px from it, and 24px along its way when
          // level with it (a line ending beside a label reads as its own). A
          // line one of its relationships runs along is its own.
          const box = { left: x - width / 2, right: x + width / 2, top: y - EDGE_LABEL_HEIGHT / 2, bottom: y + EDGE_LABEL_HEIGHT / 2 }
          for (const other of layout.paths.filter((piece) => piece !== path && !piece.data!.members.some((member) => members.includes(member)))) {
            const foreign = other.data!.members.some((member) => groupOf(member) !== groupOf(members[0]))
            for (const [a, b] of runsAlong.get(other)!) {
              const within = (gapX: number, gapY: number) => Math.min(a.x, b.x) < box.right + gapX && Math.max(a.x, b.x) > box.left - gapX
                && Math.min(a.y, b.y) < box.bottom + gapY && Math.max(a.y, b.y) > box.top - gapY
              if (within(0, 0) && (foreign || !overhangs)) problems.push(`${at} under ${a.x},${a.y}->${b.x},${b.y}`)
              if (!foreign) continue
              const level = a.x === b.x ? a.x > box.left && a.x < box.right : a.y > box.top && a.y < box.bottom
              if (within(a.x !== b.x && level ? 24 : 6, a.x === b.x && level ? 24 : 6)) problems.push(`${at} beside ${a.x},${a.y}->${b.x},${b.y}`)
            }
          }
          for (const other of labelled.filter((entry) => entry.label !== label)) {
            const { x: otherX, y: otherY, width: otherWidth } = other.label
            if (!(Math.abs(otherX - x) >= (width + otherWidth) / 2 || Math.abs(otherY - y) >= EDGE_LABEL_HEIGHT)) problems.push(`${at} over ${other.label.text}`)
          }
          for (const node of layout.nodes) {
            if (!(box.right <= node.position.x || box.left >= node.position.x + NODE_WIDTH || box.bottom <= node.position.y || box.top >= node.position.y + NODE_HEIGHT)) problems.push(`${at} over step ${node.id}`)
          }
          return problems
        })
        expect(misplaced, name).toEqual([])
      }
      // The four demotions of one symptom into one rule: three join its way in
      // from above and share a label, one from below has its own.
      const layout = layoutGraph(cascade)
      const intoRule = layout.paths.flatMap((path) => path.data!.labels).filter((label) => cascade.edges[label.members[0]].child_id === 'rule-12')
      expect(intoRule.map((label) => label.members.map((member) => cascade.edges[member].parent_id))).toEqual([['ext-0', 'ext-6', 'ext-7'], ['ext-5']])
    })

    it('never lets two routes meet at a point, nor jog less than a line gap', () => {
      for (const { name, layout } of laidOut) {
        const pieces = layout.paths.map((path) => ({ path, corners: pieceCorners(layout, path) }))
        // Where a piece ends short of a step, only routes that join or part
        // there meet it: every piece starting or ending at that point is
        // linked to it through the relationships they carry, so two unrelated
        // routes never read as one line through it.
        const unrelated = pieces.flatMap(({ path, corners }) => [...(path.data!.fromHandle ? [] : [corners[0]]), ...(path.data!.toHandle ? [] : [corners.at(-1)!])]
          .filter((end) => {
            const there = pieces.filter((other) => [other.corners[0], other.corners.at(-1)!].some((point) => point.x === end.x && point.y === end.y)).map((other) => other.path)
            // This piece, then each piece there sharing a relationship with one linked already.
            const linked = [path]
            for (const piece of linked) linked.push(...there.filter((other) => !linked.includes(other) && other.data!.members.some((member) => piece.data!.members.includes(member))))
            return linked.length !== there.length
          })
          .map((end) => `${path.id} at ${end.x},${end.y}`))
        expect(unrelated, name).toEqual([])
        // A route's way between two level runs is level or a line gap at least.
        const jogs = layout.edges.flatMap((edge) => runsOf(routeCorners(layout, edge)).slice(1, -1)
          .filter(([a, b]) => a.x === b.x && !(Math.abs(b.y - a.y) === 0 || Math.abs(b.y - a.y) >= 8))
          .map(([a, b]) => `${edge.source} to ${edge.target} jogs ${Math.abs(b.y - a.y)}`))
        expect(jogs, name).toEqual([])
      }
    })
  })

  it('lights every stretch a traced relationship runs along, and shows only the traced relationships\' labels', () => {
    // jsdom lays out no edges (React Flow measures their handles), so the
    // pieces are checked as the view lights them: on the trace or not.
    const { paths } = layoutGraph(graph)
    const lit = (selected: string) => {
      const { edgeIndexes } = traceRelationships(graph.edges, selected)
      return {
        pieces: paths.filter((path) => onTrace(path.data!.members, edgeIndexes)),
        labels: paths.flatMap((path) => path.data!.labels).filter((label) => onTrace(label.members, edgeIndexes)).map((label) => label.text).sort(),
      }
    }
    // The outcome is traced back along both matches: every stretch of theirs is lit, under one label.
    expect(lit('toString').pieces.flatMap((path) => path.data!.members).filter((member, index, all) => all.indexOf(member) === index).sort()).toEqual([0, 1, 2])
    expect(lit('toString').pieces.every((path) => path.data!.members.some((member) => member === 1 || member === 2))).toBe(true)
    expect(lit('toString').labels).toEqual(['Matches'])
    // One match traced: its own stretches and the shared ones are lit, the other match's own are not.
    const one = lit('__proto__')
    expect(one.pieces.every((path) => path.data!.members.some((member) => member === 0 || member === 1))).toBe(true)
    expect(paths.filter((path) => path.data!.members.join() === '2').every((path) => !one.pieces.includes(path))).toBe(true)
    expect(one.labels).toEqual(['Matches', 'Prunes'])
  })

  it('names each relationship once for assistive technology, and hides the pictures of its route', () => {
    // jsdom lays out no edges, so what React Flow is given is checked: each
    // relationship as one edge drawing nothing, named by its kind and steps,
    // and every drawn piece hidden, a stretch several share being one piece.
    const { edges, paths } = layoutGraph(graph)
    expect(edges.map((edge) => [edge.type, edge.ariaLabel])).toEqual([
      ['relationship', 'Prunes from Extraction Same symptom on node-a to Cascade prune Same symptom'],
      ['relationship', 'Matches from Extraction Same symptom on node-a to Outcome Final symptom'],
      ['relationship', 'Matches from Extraction Same symptom to Outcome Final symptom'],
    ])
    expect(paths.every((path) => path.domAttributes?.['aria-hidden'] === true)).toBe(true)
    expect(relationshipName('DEMOTES', { ...evidence, symptom: '', hostname: null }, { ...evidence, node_type: 'SCRUB' })).toBe('Demotes from Extraction with no symptom recorded to Scrub Same symptom on node-a')
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
    // Actions are neutral badges: red is blame and emerald is no house token, so neither appears.
    const actionBadge = (id: string) => container.querySelector(`.react-flow__node[data-id="${id}"] [data-slot="badge"]`)
    expect(['__proto__', '', 'toString', 'unknown'].map((id) => actionBadge(id)?.getAttribute('data-variant'))).toEqual(['zinc-subtle', 'dashed', 'outline', 'zinc-subtle'])
    expect(container.querySelector('[class*="red-"], [class*="emerald-"], [class*="primary"]')).toBeNull()
    // Service enums read as sentence-case names wherever they are shown or announced.
    expect(['__proto__', '', 'toString', 'unknown'].map((id) => actionBadge(id)?.textContent)).toEqual(['Created', 'Pruned', 'Matched', 'Unknown'])
    expect(within(screen.getByRole('list', { name: 'Graph legend' })).getAllByRole('listitem').map((item) => item.textContent))
      .toEqual(['Selected path', 'Created', 'Matched', 'Removed', 'Pruned'])
    expect(container.querySelector('.react-flow__node[data-id="toString"]')?.getAttribute('aria-label'))
      .toBe('Outcome, Matched: Final symptom. Rule: symptom_to_outcome. Host: not recorded')
    expect(container.textContent).not.toMatch(/CREATED|MATCHED|REMOVED|PRUNED|MATCHES|PRUNES|EXTRACTION|CASCADE_PRUNE|OUTCOME/)
    expect(screen.getByText('5 steps · 3 relationships')).toBeTruthy()
    expect(within(details).getByText('Matched').getAttribute('data-variant')).toBe('outline')
    expect(valueOf(details, 'Host').textContent).toBe('Not recorded')
    expect(copied(valueOf(details, 'Rule'))).toBe('symptom_to_outcome')
    // A card shows the rule and host it has, and leaves a missing one out.
    expect(container.querySelector('.react-flow__node[data-id="toString"] .decision-step')?.textContent).toBe('OutcomeMatchedFinal symptomsymptom_to_outcome')
    expect(container.querySelector('.react-flow__node[data-id="__proto__"] .decision-step')?.textContent).toBe('ExtractionCreatedSame symptomextract-rule·node-a')
    expect(container.querySelector('.react-flow__node[data-id="unknown"] .decision-step')?.textContent).toBe('Unknown stageUnknownNo symptom recordedNo rule or host recorded')
    // The selected step is the one green outside Pass, drawn with the graph-trace role color.
    expect(container.querySelector('.react-flow__node.selected .decision-step')?.classList.contains('border-graph-trace')).toBe(true)
    fireEvent.click(within(details).getByRole('button', { name: 'From Matches: extract-rule, Extraction, node-a' }))
    expectTrace('__proto__', ['', 'toString'], ['constructor', 'unknown'])
    expect(valueOf(details, 'Source').textContent).toBe('syslog')
    expect(valueOf(details, 'Source position').textContent).toBe('0')
    // The source facts are rows on the panel's one label axis, as the facts
    // above them are: one column, no row spanning another.
    const sourceFacts = valueOf(details, 'Source').closest('dl')!
    expect(sourceFacts.className.split(' ')).toEqual(expect.arrayContaining(['grid', 'grid-cols-1', 'sm:grid-cols-[96px_minmax(min-content,1fr)]']))
    expect([...sourceFacts.querySelectorAll('dt')].map((term) => term.textContent)).toEqual(['Source', 'Source position', 'Recorded time'])
    expect(valueOf(details, 'Host').closest('dl')!.className).toBe(sourceFacts.className)
    expect(details.innerHTML).not.toMatch(/col-span/)
    // The recorded datum is machine output: a wrapped, copyable code block named by its title.
    const datum = within(details).getByRole('region', { name: 'Source datum' })
    expect(within(datum).getByText(evidence.datum!).tagName).toBe('PRE')
    expect(details.querySelector('script')).toBeNull()
    expect(within(details).getByRole('button', { name: 'Copy source datum' })).toBeTruthy()
    // The time the evidence line was recorded, labelled as the Symptoms table labels the same value.
    expect(within(details).queryByText('Timestamp')).toBeNull()
    const time = valueOf(details, 'Recorded time').querySelector('time')!
    expect(time.getAttribute('datetime')).toBe('2026-08-01T00:00:00Z')
    expect(time.textContent).toBe('2026-08-01 00:00:00 UTC')
    fireEvent.click(within(details).getByRole('button', { name: /To Prunes: prune-rule/ }))
    expectTrace('', ['__proto__'], ['constructor', 'toString', 'unknown'])
    expect(within(details).getByText('Pruned').getAttribute('data-variant')).toBe('dashed')
    expect(valueOf(details, 'Reason').textContent).toBe('A recorded reason')
    expect(within(details).getByText('No source evidence recorded for this step.')).toBeTruthy()
    expect(within(details).queryByRole('region')).toBeNull()
    fireEvent.keyDown(screen.getByLabelText('Inspect decision step'), { key: 'ArrowDown' })
    expect(screen.getByRole('option', { name: /Cascade prune · Pruned/ })).toBeTruthy()
    fireEvent.click(screen.getByRole('option', { name: /Unknown stage/ }))
    expectTrace('unknown', [], ['__proto__', 'constructor', '', 'toString'])
    expect(within(details).getByText('Unknown stage')).toBeTruthy()
    expect(valueOf(details, 'Reason').textContent).toBe('Not recorded')
    expect(within(details).queryByRole('button')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }))
    expect(within(details).getByRole('heading', { name: 'Final symptom' })).toBeTruthy()
    // A recorded identifier may break after its separators, where no line breaks on its own.
    const BREAK = '<span aria-hidden="true" class="select-none">\u200b</span>'
    expect(valueOf(details, 'Rule').innerHTML).toBe(`<span>symptom_${BREAK}to_${BREAK}outcome</span>`)
    expectTrace('toString', ['__proto__', 'constructor'], ['', 'unknown'])
  })

  it('moves focus to the step a connected step leads to, even when the pressed row goes away', () => {
    render(<DecisionGraphView graph={graph} />)
    const details = screen.getByRole('complementary', { name: 'Selected decision step' })
    // The outcome's second connected step has only one of its own, so its list has no second row.
    const followed = within(details).getByRole('button', { name: 'From Matches: extract-rule, Extraction, No host recorded' })
    followed.focus()
    fireEvent.click(followed)
    expect(followed.isConnected).toBe(false)
    expect(valueOf(details, 'Host').textContent).toBe('Not recorded')
    const title = within(details).getByRole('heading', { name: 'Same symptom' })
    expect(document.activeElement).toBe(title)
    // The title takes focus from script only; it stays out of the tab order.
    expect(title.tabIndex).toBe(-1)
  })

  it('inspects the first recorded step when a graph has no outcome', () => {
    render(<DecisionGraphView graph={{ nodes: [evidence], edges: [] }} />)
    expect(screen.getByRole('complementary').textContent).toContain('Same symptom')
    expect(valueOf(screen.getByRole('complementary'), 'Source position').textContent).toBe('0')
    expect(screen.getByText('1 step · 0 relationships')).toBeTruthy()
  })

  it('says so when a recorded source datum is empty instead of drawing an empty block', () => {
    render(<DecisionGraphView graph={{ nodes: [{ ...evidence, datum: '' }], edges: [] }} />)
    const details = screen.getByRole('complementary')
    expect(within(details).getByText('Empty source datum')).toBeTruthy()
    expect(within(details).queryByRole('region')).toBeNull()
  })

  it('opens on the outcome chain at a readable zoom, and fits the whole graph only on request', () => {
    const { container } = render(<DecisionGraphView graph={graph} />)
    // jsdom has no layout, so React Flow measures its canvas at its 500 × 500 fallback.
    const canvas = 500
    const { nodes, edges, stages } = layoutGraph(graph)
    const bandHeader = (index: number) => container.querySelectorAll('.decision-stage')[index].firstElementChild as HTMLElement
    const onScreen = (id: string, view: { x: number; y: number; zoom: number }, margin: number) => {
      const { position } = nodes.find((node) => node.id === id)!
      return [position.x, position.y].every((start, axis) => {
        const size = axis === 0 ? NODE_WIDTH : NODE_HEIGHT
        const offset = axis === 0 ? view.x : view.y
        return start * view.zoom + offset >= margin - 1e-6 && (start + size) * view.zoom + offset <= canvas - margin + 1e-6
      })
    }
    const opening = viewportOf(container)
    // The chain from both extractions to the outcome is wider than the canvas at
    // 0.75, so the view stops there and keeps the outcome it opened on in sight.
    expect(opening.zoom).toBe(0.75)
    expect(onScreen('toString', opening, 24)).toBe(true)
    // Views are fitted to the canvas above the zoom controls' corner strip, so the outcome opens clear of them.
    const outcome = nodes.find((node) => node.id === 'toString')!.position
    expect((outcome.y + NODE_HEIGHT) * opening.zoom + opening.y).toBeLessThanOrEqual(canvas - CONTROLS_INSET)
    // The rules band starts left of the canvas, so its title row starts at the
    // canvas edge instead; the extraction band is wholly off it, and its row with it.
    const visibleLeft = -opening.x / opening.zoom
    expect(visibleLeft).toBeGreaterThan(stages[1].x)
    expect(visibleLeft).toBeLessThan(stages[1].x + stages[1].width)
    expect(Number.parseFloat(bandHeader(1).style.marginLeft)).toBeCloseTo(visibleLeft - stages[1].x, 6)
    expect(bandHeader(1).textContent).toBe('Applied rules1')
    expect(visibleLeft).toBeGreaterThan(stages[0].x + stages[0].width)
    expect(bandHeader(0).style.marginLeft).toBe(`${stages[0].width}px`)

    // The view's two groups are split by a hairline in the 16px gap between
    // them, drawn in the gap and clipped where a line starts, so it never
    // dangles before a wrapped group.
    const view = screen.getByRole('group', { name: 'Graph view' })
    expect([...view.classList]).toEqual(expect.arrayContaining(['relative', 'before:-left-2', 'before:w-px', 'before:bg-border']))
    expect(view.className).not.toMatch(/(^|\s)(border-l|pl-)/)
    expect([...view.parentElement!.classList]).toEqual(expect.arrayContaining(['gap-x-4', 'overflow-hidden', '-m-[5px]', 'p-[5px]', 'w-[calc(100%+10px)]', 'sm:w-auto']))

    fireEvent.click(screen.getByRole('button', { name: 'Fit view' }))
    const all = viewportOf(container)
    expect(all.zoom).toBeLessThan(0.75)
    expect(graph.nodes.every(({ node_id }) => onScreen(node_id, all, 0))).toBe(true)
    // So is every relationship's label.
    expect(edges.every(({ data }) => {
      const left = (data!.labelX - data!.labelWidth / 2) * all.zoom + all.x
      const top = (data!.labelY - EDGE_LABEL_HEIGHT / 2) * all.zoom + all.y
      return left >= 0 && left + data!.labelWidth * all.zoom <= canvas
        && top >= 0 && top + EDGE_LABEL_HEIGHT * all.zoom <= canvas - CONTROLS_INSET
    })).toBe(true)
    // With every band on the canvas, each title row sits at its band's own edges.
    expect(stages.map((_, index) => [bandHeader(index).style.marginLeft, bandHeader(index).style.marginRight]))
      .toEqual(stages.map(() => ['0px', '0px']))

    fireEvent.click(within(screen.getByRole('complementary')).getByRole('button', { name: 'From Matches: extract-rule, Extraction, node-a' }))
    fireEvent.click(screen.getByRole('button', { name: 'Focus step' }))
    const step = viewportOf(container)
    expect(step.zoom).toBe(1)
    expect(onScreen('__proto__', step, 0)).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Reset' }))
    expect(viewportOf(container)).toEqual(opening)
    expect(within(screen.getByRole('complementary')).getByRole('heading', { name: 'Final symptom' })).toBeTruthy()
  })

  it.each([6, 8, 10])('opens with the bands\' titles in view when the outcome stays in view with them (%i matches)', (count) => {
    // A chain taller than the canvas at the readable zoom: its top, the
    // bands' titles, stays on the canvas whenever the outcome stays wholly on it too.
    const extractions = Array.from({ length: count }, (_, index) => ({ ...evidence, node_id: `ext-${index}` }))
    const tall: DecisionGraph = {
      nodes: [...extractions, { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED' }],
      edges: extractions.map((source) => ({ parent_id: source.node_id, child_id: 'outcome', edge_type: 'MATCHES' })),
    }
    const { container } = render(<DecisionGraphView graph={tall} />)
    const canvas = 500
    const { nodes, stages } = layoutGraph(tall)
    const view = viewportOf(container)
    const outcome = nodes.find((node) => node.id === 'outcome')!.position
    const inView = outcome.y * view.zoom + view.y >= -1e-6 && (outcome.y + NODE_HEIGHT) * view.zoom + view.y <= canvas - CONTROLS_INSET + 1e-6
    expect(inView).toBe(true)
    if ((outcome.y + NODE_HEIGHT - stages[0].y) * view.zoom <= canvas - CONTROLS_INSET) {
      expect(stages[0].y * view.zoom + view.y).toBeGreaterThanOrEqual(-1e-6)
    }
  })

  it('opens with the bands\' titles in view while the outcome is wholly on the canvas with them, within the margin too', () => {
    const extractions = Array.from({ length: 8 }, (_, index) => ({ ...evidence, node_id: `ext-${index}` }))
    const tall: DecisionGraph = {
      nodes: [...extractions, { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED' }],
      edges: extractions.map((source) => ({ parent_id: source.node_id, child_id: 'outcome', edge_type: 'MATCHES' })),
    }
    const { nodes, stages } = layoutGraph(tall)
    const outcome = nodes.find((node) => node.id === 'outcome')!.position
    // A canvas on which, with the titles at its top at the readable zoom, the
    // outcome ends 12px above the zoom controls' strip: wholly on the canvas,
    // though within the 24px margin the view keeps when it moves a step in.
    const fitHeight = (outcome.y + NODE_HEIGHT - stages[0].y) * 0.75 + 12
    vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(2000)
    vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(fitHeight + CONTROLS_INSET)
    const { container } = render(<DecisionGraphView graph={tall} />)
    const view = viewportOf(container)
    expect(view.zoom).toBe(0.75)
    expect(stages[0].y * view.zoom + view.y).toBeCloseTo(0, 6)
    expect((outcome.y + NODE_HEIGHT) * view.zoom + view.y).toBeCloseTo(fitHeight - 12, 6)
  })

  it('brings a step taking keyboard focus wholly onto the canvas, its focus ring too, just far enough', () => {
    const { container } = render(<DecisionGraphView graph={graph} />)
    const canvas = 500
    const { nodes } = layoutGraph(graph)
    // The view opens on the outcome; the first extraction is off the canvas.
    const extraction = nodes.find((node) => node.id === '__proto__')!.position
    const before = viewportOf(container)
    expect((extraction.x + NODE_WIDTH) * before.zoom + before.x).toBeLessThan(0)
    const step = container.querySelector<HTMLElement>('.react-flow__node[data-id="__proto__"]')!
    fireEvent.keyDown(document.body, { key: 'Tab' })
    act(() => step.focus())
    const after = viewportOf(container)
    // In by the margin, its ring included, at the zoom it had, and no farther.
    expect(after.zoom).toBe(before.zoom)
    expect((extraction.x - FOCUS_RING) * after.zoom + after.x).toBeCloseTo(24, 6)
    expect((extraction.y + NODE_HEIGHT + FOCUS_RING) * after.zoom + after.y).toBeLessThanOrEqual(canvas - CONTROLS_INSET)
  })

  it('lifts a step taking keyboard focus from under the canvas with its focus ring, by the margin above the zoom controls', () => {
    const extractions = Array.from({ length: 8 }, (_, index) => ({ ...evidence, node_id: `ext-${index}` }))
    const tall: DecisionGraph = {
      nodes: [...extractions, { ...evidence, node_id: 'outcome', node_type: 'OUTCOME', action: 'MATCHED' }],
      edges: extractions.map((source) => ({ parent_id: source.node_id, child_id: 'outcome', edge_type: 'MATCHES' })),
    }
    const { container } = render(<DecisionGraphView graph={tall} />)
    const canvas = 500
    const lowest = layoutGraph(tall).nodes.filter((node) => node.id.startsWith('ext-')).reduce((low, node) => (node.position.y > low.position.y ? node : low))
    // The chain is too tall for the canvas: its lowest step runs past the part above the zoom controls.
    const before = viewportOf(container)
    expect((lowest.position.y + NODE_HEIGHT) * before.zoom + before.y).toBeGreaterThan(canvas - CONTROLS_INSET)
    const step = container.querySelector<HTMLElement>(`.react-flow__node[data-id="${lowest.id}"]`)!
    fireEvent.keyDown(document.body, { key: 'Tab' })
    act(() => step.focus())
    const after = viewportOf(container)
    // Up by the margin, its ring included, at the zoom it had.
    expect(after.zoom).toBe(before.zoom)
    expect((lowest.position.y + NODE_HEIGHT + FOCUS_RING) * after.zoom + after.y).toBeCloseTo(canvas - CONTROLS_INSET - 24, 6)
  })

  it('shows the whole chain when it fits at the readable zoom, without moving it for the outcome', () => {
    const chain: DecisionGraph = { nodes: [evidence, graph.nodes[3]], edges: [graph.edges[1]] }
    const { nodes, edges, stages } = layoutGraph(chain)
    const boxes = [
      ...nodes.map(({ position }) => ({ left: position.x, right: position.x + NODE_WIDTH, top: position.y, bottom: position.y + NODE_HEIGHT })),
      ...edges.map(({ data }) => ({
        left: data!.labelX - data!.labelWidth / 2, right: data!.labelX + data!.labelWidth / 2,
        top: data!.labelY - EDGE_LABEL_HEIGHT / 2, bottom: data!.labelY + EDGE_LABEL_HEIGHT / 2,
      })),
    ]
    const left = Math.min(...boxes.map((box) => box.left))
    const right = Math.max(...boxes.map((box) => box.right))
    const bottom = Math.max(...boxes.map((box) => box.bottom))
    // The relationship skips the empty rules stage, so its label rides a lane
    // under the bands' titles, above the steps; the view takes the chain from
    // the top of the bands, so their titles are in it.
    expect(Math.min(...boxes.map((box) => box.top))).toBeLessThan(Math.min(...nodes.map(({ position }) => position.y)))
    const top = Math.min(...stages.map((stage) => stage.y))
    expect(top).toBeLessThan(Math.min(...boxes.map((box) => box.top)))
    // A canvas the chain overfills once padded, so the view holds at 0.75, but
    // which it fits centered, band titles and label included, with 10px to
    // spare on each side of the part above the zoom controls' corner strip.
    const width = (right - left) * 0.75 + 20
    const height = (bottom - top) * 0.75 + 20 + CONTROLS_INSET
    vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(width)
    vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(height)
    const { container } = render(<DecisionGraphView graph={chain} />)
    const view = viewportOf(container)
    expect(view.zoom).toBe(0.75)
    expect([left * view.zoom + view.x, right * view.zoom + view.x]).toEqual([expect.closeTo(10, 6), expect.closeTo(width - 10, 6)])
    expect([top * view.zoom + view.y, bottom * view.zoom + view.y]).toEqual([expect.closeTo(10, 6), expect.closeTo(height - CONTROLS_INSET - 10, 6)])
  })
})

describe('decision graph dialog', () => {
  it('contains viewer failures inside the dialog and leaves the attribution usable', async () => {
    vi.spyOn(graphLayout, 'layoutGraph').mockImplementation(() => { throw new Error('Viewer failed') })
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { rowNavigation } = renderAction(runDetail, runDetail)

    fireEvent.click(screen.getByRole('button', { name: 'View attribution logic' }))
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Reload page' })).toBeTruthy()
    // Only the viewer fails: the dialog keeps its title, description and one close button.
    const dialog = screen.getByRole('dialog', { name: 'Attribution logic', description: 'graph-job Hardware Final symptom' })
    expect(within(dialog).getAllByRole('button', { name: 'Close attribution logic' })).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: 'Close attribution logic' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getByRole('button', { name: 'View attribution logic' })).toBeTruthy()
    expect(rowNavigation).not.toHaveBeenCalled()
  })

  it('loads only on demand and reuses the detail when the viewer is reopened', async () => {
    const { getRun, rowNavigation } = renderAction()
    expect(getRun).not.toHaveBeenCalled()
    expect(screen.queryByRole('dialog')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'View attribution logic' }))
    // The panel itself has focus as it opens, before the graph has loaded.
    expect(document.activeElement).toBe(screen.getByRole('dialog'))
    expect(await screen.findByRole('complementary', { name: 'Selected decision step' })).toBeTruthy()
    expect(getRun).toHaveBeenCalledExactlyOnceWith(runDetail.id)
    expect(rowNavigation).not.toHaveBeenCalled()
    // The header states execution context, outcome and root symptom once; no
    // summary band and no adapter-constant attempt number.
    const dialog = screen.getByRole('dialog', { name: 'Attribution logic', description: 'graph-job Hardware Final symptom' })
    expect(document.activeElement).toBe(dialog)
    const description = document.getElementById(dialog.getAttribute('aria-describedby')!)!
    expect(description.querySelector('[data-slot="badge"]')?.getAttribute('data-outcome')).toBe('hardware')
    expect(dialog.textContent).not.toMatch(/FACT attributed|attempt/i)
    expect(within(dialog).queryByText(runDetail.detail)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Close attribution logic' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'View attribution logic' }))
    expect(await screen.findByRole('complementary', { name: 'Selected decision step' })).toBeTruthy()
    // Reopened with the graph at hand, it starts at the panel again.
    expect(document.activeElement).toBe(screen.getByRole('dialog'))
    expect(getRun).toHaveBeenCalledTimes(1)
    expect(rowNavigation).not.toHaveBeenCalled()
  })

  it.each(['absent', 'null', 'empty', 'invalid'] as const)('keeps a %s graph unavailable without navigating the row or fetching', (state) => {
    const { getRun, rowNavigation } = renderAction({ ...runDetail, decisionGraphState: state })
    const action = screen.getByRole('button', { name: 'View attribution logic' })
    expect(action.getAttribute('aria-disabled')).toBe('true')
    // The reason is its description, read without hover; the labelled page action stays in the tab order.
    expect(screen.getByRole('button', { description: /No decision graph|no recorded decision steps|cannot be displayed/ })).toBe(action)
    expect(action.hasAttribute('tabindex')).toBe(false)
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

    fireEvent.click(screen.getByRole('button', { name: 'View attribution logic' }))
    expect(screen.getByRole('status').textContent).toContain('Loading attribution logic')
    expect(await screen.findByText(message)).toBeTruthy()
    expect(screen.queryByRole('complementary')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByRole('complementary', { name: 'Selected decision step' })).toBeTruthy()
    expect(getRun).toHaveBeenCalledTimes(2)
  })

  it('uses the already loaded detail without another request', async () => {
    const { getRun } = renderAction(runDetail, runDetail)
    fireEvent.click(screen.getByRole('button', { name: 'View attribution logic' }))
    expect(await screen.findByRole('complementary', { name: 'Selected decision step' })).toBeTruthy()
    expect(getRun).not.toHaveBeenCalled()
  })

  it('says so when the root symptom was not recorded', async () => {
    renderAction({ ...runDetail, rootSymptom: { ...runDetail.rootSymptom, symptom: '' } }, { ...runDetail, rootSymptom: { ...runDetail.rootSymptom, symptom: '' } })
    fireEvent.click(screen.getByRole('button', { name: 'View attribution logic' }))
    expect(screen.getByRole('dialog', { description: 'graph-job Hardware Root symptom not recorded' })).toBeTruthy()
    expect(await screen.findByRole('complementary', { name: 'Selected decision step' })).toBeTruthy()
  })

  it('offers the logic as a labelled page action, and as an icon in a table row', async () => {
    renderAction(runDetail, runDetail)
    const action = screen.getByRole('button', { name: 'View attribution logic' })
    // Labelled, it is the verdict's one primary action: the green default; in a row it stays a ghost icon.
    expect([action.getAttribute('data-variant'), action.getAttribute('data-size'), action.textContent]).toEqual(['default', 'default', 'View attribution logic'])
    expect(action.querySelector('svg.lucide-git-branch')?.getAttribute('data-icon')).toBe('inline-start')
    // The labelled action needs no tooltip to say what it does.
    fireEvent.pointerMove(action)
    fireEvent.focus(action)
    expect(screen.queryByRole('tooltip')).toBeNull()
    cleanup()

    const { rowNavigation } = renderAction(runDetail, undefined, true)
    const icon = screen.getByRole('button', { name: 'View attribution logic' })
    expect([icon.getAttribute('data-variant'), icon.getAttribute('data-size'), icon.textContent]).toEqual(['ghost', 'icon-sm', ''])
    expect(icon.hasAttribute('tabindex')).toBe(false)
    fireEvent.focus(icon)
    expect(screen.getByRole('tooltip').textContent).toBe('View attribution logic')
    expect(icon.getAttribute('aria-describedby')).toBe(screen.getByRole('tooltip').id)
    fireEvent.click(icon)
    expect(await screen.findByRole('complementary', { name: 'Selected decision step' })).toBeTruthy()
    expect(rowNavigation).not.toHaveBeenCalled()
    cleanup()

    // Unavailable in a row, the icon leaves the tab order, so runs without a
    // graph add no dead stops; its reason is still its description.
    renderAction({ ...runDetail, decisionGraphState: 'absent' }, undefined, true)
    const unavailable = screen.getByRole('button', { name: 'View attribution logic', description: 'No decision graph was recorded for this attribution.' })
    expect(unavailable.tabIndex).toBe(-1)
    expect(unavailable.getAttribute('aria-disabled')).toBe('true')
  })
})
