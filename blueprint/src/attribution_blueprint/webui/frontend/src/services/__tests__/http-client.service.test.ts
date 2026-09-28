// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { afterEach, describe, expect, it, vi } from 'vitest'
import { HttpError, httpGet } from '@/services/http-client.service'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('httpGet', () => {
  it('sends query parameters and omits empty optional values', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ runs: 2 }),
    })
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()

    await expect(httpGet('/runs', {
      page: 2,
      failed: false,
      missing: null,
      absent: undefined,
    }, { signal: controller.signal })).resolves.toEqual({ runs: 2 })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:3000/api/v1/runs?page=2&failed=false',
      {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      },
    )
  })

  it('throws an HttpError with response details', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503 }))

    const request = httpGet('/meta')

    await expect(request).rejects.toMatchObject({
      name: 'HttpError',
      status: 503,
      message: 'GET /meta -> 503',
    })
    await expect(request).rejects.toBeInstanceOf(HttpError)
  })

  it.each(['0', '-1', '9007199254740993', '-6139371663235533333'])(
    'preserves string observation IDs and numeric quantities in the response: %s',
    async (observationId) => {
      const text = `{
        "root": { "source": { "observation_id": "${observationId}", "pos": 0 } },
        "sites": [{ "source": { "observation_id": "${observationId}", "pos": null } }],
        "count": 8,
        "elapsedH": 0.125,
        "log": "observation_id: 9007199254740993"
      }`
      const response = { ok: true, text: vi.fn(), json: vi.fn().mockImplementation(async () => JSON.parse(text)) }
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response))

      await expect(httpGet('/runs/run')).resolves.toEqual({
        root: { source: { observation_id: observationId, pos: 0 } },
        sites: [{ source: { observation_id: observationId, pos: null } }],
        count: 8,
        elapsedH: 0.125,
        log: 'observation_id: 9007199254740993',
      })
      expect(response.json).toHaveBeenCalledOnce()
      expect(response.text).not.toHaveBeenCalled()
    },
  )
})
