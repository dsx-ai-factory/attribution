// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import type { SymptomDocumentation } from '@/interfaces/documentation'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function symptomDocumentationApiUrl(
  configuredUrl: string | undefined,
  webuiHostname: string,
): string | null {
  if (!configuredUrl) return null
  try {
    const url = new URL(configuredUrl)
    if (url.username || url.password) return null
    const localPreview = ['localhost', '127.0.0.1', '[::1]'].includes(webuiHostname)
    if (url.protocol !== 'https:' && !(url.protocol === 'http:' && localPreview)) return null
    return configuredUrl
  } catch {
    return null
  }
}

export function findSymptomDocumentation(
  catalog: unknown,
  symptomId?: string | null,
): SymptomDocumentation | undefined {
  if (!symptomId || !isRecord(catalog) || !isRecord(catalog.symptoms)
    || !Object.hasOwn(catalog.symptoms, symptomId)) return undefined
  // Consume only the API's exact runtime-symptom mapping. The returned catalog
  // ID is a different identifier; the browser never derives one from a symptom.
  const entry = catalog.symptoms[symptomId]
  if (!isRecord(entry) || typeof entry.id !== 'string' || !entry.id
    || typeof entry.title !== 'string' || !entry.title
    || typeof entry.doc_url !== 'string') return undefined
  try {
    const url = new URL(entry.doc_url)
    if (url.protocol !== 'https:' || url.username || url.password) return undefined
  } catch {
    return undefined
  }
  // Preserve the API destination verbatim, including paths that may return 404.
  return { id: entry.id, title: entry.title, doc_url: entry.doc_url }
}
