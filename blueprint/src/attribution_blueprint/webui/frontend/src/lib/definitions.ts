// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DefTerm, DefEntry } from '@/interfaces/definitions'

// DEFINITIONS is the single source of truth surfaced via <InfoHint>; a DefTerm
// without an entry here fails typecheck (Record<DefTerm, DefEntry>). Only
// derived metrics whose meaning is not in their name carry a definition.

export const DEFINITIONS: Record<DefTerm, DefEntry> = {
  'combined-mtbi': {
    label: 'Combined MTBI',
    def: 'Mean time between infrastructure incidents, over hardware and system software failures combined. Higher is healthier.',
    formula: 'kGPU-hours ÷ (HW + SW failures)',
  },
  'hw-mtbi': {
    label: 'HW MTBI',
    def: 'Mean time between infrastructure incidents, over hardware failures only. Higher is healthier.',
    formula: 'kGPU-hours ÷ HW failures',
  },
  'gpu-hours': {
    label: 'GPU-hours',
    def: 'Elapsed wall-clock hours multiplied by the number of GPUs in the attempt.',
    formula: 'elapsed_hours x gpu_count',
  },
  'responsible-in': {
    label: 'Responsible in',
    def: 'Number of attempts that listed this node among responsible_nodes across recorded runs.',
  },
  'distinct-symptoms': {
    label: 'Distinct symptoms',
    def: 'Count of unique root symptoms attributed to this node across recorded runs.',
  },
  'gpu-hours-on-node': {
    label: 'GPU-hours on node',
    def: 'Sum of GPU-hours distributed across listed nodes for attempts that ran on this node.',
  },
}

// Compile-time completeness check: any missing DefTerm here will fail TS.
const _exhaustive: Record<DefTerm, DefEntry> = DEFINITIONS
void _exhaustive
