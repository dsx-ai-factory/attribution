// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { SWRConfig } from 'swr'
import '@/styles/globals.css'
import App from './app.tsx'
import { ApiProvider } from '@/services/api-provider'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* The MVP data is a static bootstrap blob, so focus-revalidation is noise. */}
    <SWRConfig value={{ revalidateOnFocus: false }}>
      <ApiProvider>
        <App />
      </ApiProvider>
    </SWRConfig>
  </StrictMode>,
)
