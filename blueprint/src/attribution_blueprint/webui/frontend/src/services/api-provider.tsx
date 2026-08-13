// src/services/api-provider.tsx
import { useMemo, type ReactNode } from 'react'
import type { ApiService } from '@/services/api.service'
import { ApiBootstrapService, readBootstrap } from '@/services/api.bootstrap.service'
import { ApiServerService } from '@/services/api.server.service'
import { ApiContext } from '@/services/api-context'

export function ApiProvider({ children }: { children: ReactNode }) {
  const api = useMemo<ApiService>(() => {
    if ((import.meta.env.VITE_DATA_SOURCE as string | undefined) === 'http') return new ApiServerService()
    // Bootstrap reads the injected blob when present. Otherwise the UI uses
    // /api/v1/* and views request only the slices they need.
    if (readBootstrap()) return new ApiBootstrapService()
    return new ApiServerService()
  }, [])
  return <ApiContext.Provider value={api}>{children}</ApiContext.Provider>
}
