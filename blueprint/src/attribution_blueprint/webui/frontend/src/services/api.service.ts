// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import type { FleetFilter } from '@/interfaces/filter'
import type { RunDetail, RunSummary } from '@/interfaces/attribution'
import type { Meta } from '@/interfaces/meta'

/** Decode opaque IDs from their original JSON tokens; quantities remain numbers. */
export function parseAttributionJson<T>(text: string): T {
  // TypeScript's JSON declaration does not yet include the native source context.
  return JSON.parse(text, (key: string, value: unknown, context?: { source: string }) => (
    key === 'observation_id' && typeof value === 'number' ? context!.source : value
  )) as T
}

/**
 * The data contract for the UI. Abstract so the provider can swap the concrete
 * implementation (bootstrap blob by default, HTTP later).
 */
export abstract class ApiService {
  /** The FULL injected/served run set, unfiltered. Views own all filtering. */
  abstract getAllRuns(): Promise<RunSummary[]>
  abstract getRuns(filter: FleetFilter): Promise<RunSummary[]>
  abstract getRun(id: string): Promise<RunDetail | undefined>
  abstract getMeta(): Promise<Meta>
}
