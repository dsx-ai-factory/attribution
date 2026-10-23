// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DecisionGraph } from '@/interfaces/attribution'

export function traceRelationships(edges: DecisionGraph['edges'], selectedId: string) {
  const incoming = new Map<string, { nodeId: string; edgeIndex: number }[]>()
  const outgoing = new Map<string, { nodeId: string; edgeIndex: number }[]>()
  edges.forEach((edge, edgeIndex) => {
    const parents = incoming.get(edge.child_id) ?? []
    parents.push({ nodeId: edge.parent_id, edgeIndex })
    incoming.set(edge.child_id, parents)
    const children = outgoing.get(edge.parent_id) ?? []
    children.push({ nodeId: edge.child_id, edgeIndex })
    outgoing.set(edge.parent_id, children)
  })

  const nodeIds = new Set([selectedId])
  const edgeIndexes = new Set<number>()
  // Keep the traversals directed and independent: crossing from an ancestor
  // to its other children would incorrectly include sibling branches.
  for (const adjacency of [incoming, outgoing]) {
    const visited = new Set([selectedId])
    const pending = [selectedId]
    while (pending.length > 0) {
      for (const { nodeId, edgeIndex } of adjacency.get(pending.pop()!) ?? []) {
        edgeIndexes.add(edgeIndex)
        nodeIds.add(nodeId)
        if (!visited.has(nodeId)) {
          visited.add(nodeId)
          pending.push(nodeId)
        }
      }
    }
  }
  return { nodeIds, edgeIndexes }
}

// A drawn piece of a route (TracePathData) is on the trace wherever a traced
// relationship runs along it: a stretch several share for any of them.
export function onTrace(members: readonly number[], traced: ReadonlySet<number>): boolean {
  return members.some((index) => traced.has(index))
}
