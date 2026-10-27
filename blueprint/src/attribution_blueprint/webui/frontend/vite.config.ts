// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference types="vitest/config" />
import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { thirdPartyNotices } from './third-party-notices'

// https://vite.dev/config/
export default defineConfig({
  base: '/',
  build: {
    // Repo-local build target: the package's served static dir.
    // Docker overrides this with `vite build --outDir dist` in its Node stage.
    outDir: '../static',
    emptyOutDir: true,
  },
  plugins: [react(), tailwindcss(), thirdPartyNotices({
    configUrl: import.meta.url,
    sourceNotices: new URL('../../../../../THIRD_PARTY_NOTICES.md', import.meta.url),
    assetPackages: ['@fontsource-variable/jetbrains-mono', 'tailwindcss', 'tw-animate-css'],
    licenseDirectory: new URL('./third-party-licenses/', import.meta.url),
    licenseSupplements: {
      '@radix-ui/number@1.1.1': 'radix-ui-primitives.txt',
      '@radix-ui/react-compose-refs@1.1.2': 'radix-ui-primitives.txt',
      '@radix-ui/react-context@1.1.2': 'radix-ui-primitives.txt',
      '@radix-ui/react-direction@1.1.1': 'radix-ui-primitives.txt',
      '@radix-ui/react-id@1.1.1': 'radix-ui-primitives.txt',
      '@radix-ui/react-use-callback-ref@1.1.1': 'radix-ui-primitives.txt',
      '@radix-ui/react-use-escape-keydown@1.1.1': 'radix-ui-primitives.txt',
      '@radix-ui/react-use-layout-effect@1.1.1': 'radix-ui-primitives.txt',
      '@radix-ui/react-use-previous@1.1.1': 'radix-ui-primitives.txt',
      '@radix-ui/react-use-size@1.1.1': 'radix-ui-primitives.txt',
      'react-remove-scroll-bar@2.3.8': 'react-remove-scroll-bar.txt',
    },
  })],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    // Vitest blanks every stylesheet import, `?raw` included. The design-token
    // guard in application-components.test.tsx reads globals.css as text.
    css: { include: [/\/src\/styles\/globals\.css\?raw$/] },
  },
})
