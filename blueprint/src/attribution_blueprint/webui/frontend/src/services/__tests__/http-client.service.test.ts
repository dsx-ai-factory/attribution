// @vitest-environment jsdom
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
    }, controller.signal)).resolves.toEqual({ runs: 2 })

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
})
