// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, renderHook } from '@testing-library/react'
import { ApiProvider } from '@/services/api-provider'
import { useApi } from '@/services/api-context'

const mocks = vi.hoisted(() => ({
  bootstrapApi: { source: 'bootstrap' },
  serverApi: { source: 'server' },
  readBootstrap: vi.fn(),
  BootstrapService: vi.fn(),
  ServerService: vi.fn(),
}))

vi.mock('@/services/api.bootstrap.service', () => ({
  readBootstrap: mocks.readBootstrap,
  ApiBootstrapService: class {
    constructor() {
      mocks.BootstrapService()
      return mocks.bootstrapApi
    }
  },
}))
vi.mock('@/services/api.server.service', () => ({
  ApiServerService: class {
    constructor() {
      mocks.ServerService()
      return mocks.serverApi
    }
  },
}))

beforeEach(() => {
  vi.clearAllMocks()
  mocks.readBootstrap.mockReturnValue(null)
})

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
})

describe('ApiProvider and useApi', () => {
  it('rejects consumers outside the provider', () => {
    expect(() => renderHook(() => useApi())).toThrow('useApi must be used within ApiProvider')
  })

  it('uses the bootstrap source when injected data exists', () => {
    mocks.readBootstrap.mockReturnValue({ runs: [] })
    const wrapper = ({ children }: { children: ReactNode }) => <ApiProvider>{children}</ApiProvider>

    const { result } = renderHook(() => useApi(), { wrapper })

    expect(result.current).toBe(mocks.bootstrapApi)
    expect(mocks.BootstrapService).toHaveBeenCalledTimes(1)
  })

  it('uses the server source for explicit HTTP mode or absent bootstrap data', () => {
    vi.stubEnv('VITE_DATA_SOURCE', 'http')
    const wrapper = ({ children }: { children: ReactNode }) => <ApiProvider>{children}</ApiProvider>
    const { result, unmount } = renderHook(() => useApi(), { wrapper })
    expect(result.current).toBe(mocks.serverApi)
    unmount()

    vi.stubEnv('VITE_DATA_SOURCE', '')
    const second = renderHook(() => useApi(), { wrapper })
    expect(second.result.current).toBe(mocks.serverApi)
  })
})
