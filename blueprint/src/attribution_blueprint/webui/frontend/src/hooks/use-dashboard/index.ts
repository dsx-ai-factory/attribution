// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import useSWR, { type KeyedMutator } from 'swr'
import type { OutcomeKey } from '@/interfaces/attribution'
import type { DashboardPayload } from '@/interfaces/meta'
import { httpGet } from '@/services/http-client.service'

export function useDashboard({
  from,
  to,
  outcome,
}: {
  from?: string
  to?: string
  outcome?: OutcomeKey
}): {
  dashboard: DashboardPayload | null
  /** Whether this view's dashboard is still on its way. */
  isLoading: boolean
  error: Error | null
  /** Re-fetches this view's dashboard. */
  mutate: KeyedMutator<DashboardPayload>
} {
  const key = ['dashboard', from ?? '', to ?? '', outcome ?? '']
  const { data, isLoading, error, mutate } = useSWR<DashboardPayload, Error>(key, () =>
    httpGet<DashboardPayload>('/dashboard', { from, to, outcome }),
  )

  return { dashboard: data ?? null, isLoading, error: error ?? null, mutate }
}
