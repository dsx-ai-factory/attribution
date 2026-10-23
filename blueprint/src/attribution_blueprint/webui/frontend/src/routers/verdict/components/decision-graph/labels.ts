// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DecisionAction, DecisionNode, DecisionNodeType } from '@/interfaces/attribution'

// Display names for the Service's graph enums, in sentence case. Every place a
// stage, action or relationship is shown or announced reads from here.
export const STAGE_LABELS: Record<DecisionNodeType, string> = {
  UNKNOWN: 'Unknown stage',
  EXTRACTION: 'Extraction',
  FILTER: 'Filter',
  SCRUB: 'Scrub',
  REFINER: 'Refinement',
  CASCADE_PRUNE: 'Cascade prune',
  OUTCOME: 'Outcome',
}

export const ACTION_LABELS: Record<DecisionAction, string> = {
  UNKNOWN: 'Unknown',
  CREATED: 'Created',
  MATCHED: 'Matched',
  REMOVED: 'Removed',
  PRUNED: 'Pruned',
}

const EDGE_LABELS: Record<string, string> = { CREATES: 'Creates', MATCHES: 'Matches', DEMOTES: 'Demotes', PRUNES: 'Prunes' }

// Edge types are open-ended strings; one without a display name is shown as recorded.
export function edgeLabel(edgeType: string) {
  return Object.hasOwn(EDGE_LABELS, edgeType) ? EDGE_LABELS[edgeType] : edgeType
}

export function stepLabel(step: DecisionNode) {
  return step.rule_name || step.symptom || STAGE_LABELS[step.node_type]
}

// A step as a relationship's name refers to it, as the step names itself: by
// its stage and symptom, and its host where one is recorded.
function stepReference(step: DecisionNode) {
  return `${STAGE_LABELS[step.node_type]} ${step.symptom || 'with no symptom recorded'}${step.hostname ? ` on ${step.hostname}` : ''}`
}

// A relationship as assistive technology reads it: its kind and its steps.
export function relationshipName(edgeType: string, source: DecisionNode, target: DecisionNode) {
  return `${edgeLabel(edgeType)} from ${stepReference(source)} to ${stepReference(target)}`
}
