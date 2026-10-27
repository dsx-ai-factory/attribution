// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Keys and shape for the metric definitions surfaced via <InfoHint>.
 * Keep keys kebab-case, labels human-cased.
 */
export type DefTerm =
  | 'combined-mtbi'
  | 'hw-mtbi'
  | 'gpu-hours'
  | 'responsible-in'
  | 'distinct-symptoms'
  | 'gpu-hours-on-node'

export interface DefEntry {
  /** Human-readable label; names the hint's button ("About {label}"). */
  label: string
  /** One-sentence definition. */
  def: string
  /** Optional single-line formula. */
  formula?: string
}
