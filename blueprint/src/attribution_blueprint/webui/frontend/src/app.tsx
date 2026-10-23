// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { TopBar } from '@/components/top-bar'
import { useMeta } from '@/hooks/use-meta'
import { FleetView } from '@/routers/fleet'
import { VerdictView } from '@/routers/verdict'
import { NodeView } from '@/routers/node'
import { navigateService } from '@/services/navigate.service'
import { useNav } from '@/hooks/use-nav'

function Shell() {
  const { pathname } = useLocation()
  const { name } = navigateService.match(pathname)
  const nav = useNav()
  const { meta } = useMeta()

  // Esc returns to fleet from any drill-down — matches the prototype's demo nav.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!e.defaultPrevented && e.key === 'Escape' && name !== 'fleet') nav('fleet')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [name, nav])

  // Reset scroll AFTER the new view mounts (prevents visible scroll-then-route jitter).
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' })
  }, [pathname])

  // Full width on one rail: the bar and every page share the page gutter, and a
  // page starts the page-top spacing under the bar (the page-* values in
  // globals.css).
  return (
    <>
      <TopBar meta={meta} />
      <main className="px-page-x pt-page-top pb-page-bottom">
        <Routes>
          <Route path="/" element={<FleetView />} />
          <Route path="/verdict/:id" element={<VerdictView />} />
          <Route path="/node/:id" element={<NodeView />} />
        </Routes>
      </main>
    </>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  )
}

export default App
