// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import useSWR, { type KeyedMutator } from 'swr'
import type { Meta } from '@/interfaces/meta'
import { httpGet } from '@/services/http-client.service'

/**
 * Facts about the backend the dashboard is bound to, which no filter narrows.
 * One static SWR key, so mounting this in the shell costs a single request;
 * mutate re-reads it for every caller.
 */
export function useMeta(): { meta: Meta | null; mutate: KeyedMutator<Meta> } {
  const { data, mutate } = useSWR<Meta, Error>('meta', () => httpGet<Meta>('/meta'))
  return { meta: data ?? null, mutate }
}
