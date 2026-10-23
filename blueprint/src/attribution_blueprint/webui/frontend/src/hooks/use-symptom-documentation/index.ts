// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import useSWR from 'swr'
import type { SymptomDocumentationCatalog } from '@/interfaces/documentation'
import { symptomDocumentationApiUrl } from '@/lib/documentation'

async function fetchDocumentation(url: string): Promise<SymptomDocumentationCatalog> {
  const response = await fetch(url, {
    headers: { Accept: 'application/json' },
    redirect: 'error',
    signal: AbortSignal.timeout(10_000),
  })
  if (!response.ok) throw new Error(`Symptom documentation request failed: ${response.status}`)
  return response.json()
}

export function useSymptomDocumentation(enabled = true) {
  const url = symptomDocumentationApiUrl(import.meta.env.VITE_SYMPTOM_DOCS_URL, window.location.hostname)
  return useSWR<SymptomDocumentationCatalog, Error>(
    enabled ? url : null,
    fetchDocumentation,
    { revalidateOnFocus: false, shouldRetryOnError: false },
  )
}
