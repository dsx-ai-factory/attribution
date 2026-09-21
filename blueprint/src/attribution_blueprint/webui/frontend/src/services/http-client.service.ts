// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

export class HttpError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'HttpError'
    this.status = status
  }
}

const BASE = (import.meta.env.VITE_API_BASE_PATH as string | undefined) || '/api/v1'

export async function httpGet<T>(
  path: string,
  params?: Record<string, unknown>,
  options: { signal?: AbortSignal; decode?: (text: string) => T } = {},
): Promise<T> {
  const { signal, decode } = options
  const url = new URL(`${BASE}${path}`, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) url.searchParams.set(k, String(v))
    }
  }
  const res = await fetch(url.toString(), { signal, headers: { Accept: 'application/json' } })
  if (!res.ok) throw new HttpError(res.status, `GET ${path} -> ${res.status}`)
  return decode ? decode(await res.text()) : (await res.json()) as T
}
