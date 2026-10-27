// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// src/services/api-context.ts
// Context + accessor hook live here (no component export) so the provider file
// can stay component-only for react-refresh.
import { createContext, useContext } from 'react'
import type { ApiService } from '@/services/api.service'

export const ApiContext = createContext<ApiService | null>(null)

export function useApi(): ApiService {
  const api = useContext(ApiContext)
  if (!api) throw new Error('useApi must be used within ApiProvider')
  return api
}
