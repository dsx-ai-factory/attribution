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

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-screen-2xl px-7 pt-5 pb-20">
        <TopBar name={name} database={meta?.database ?? null} />
        <Routes>
          <Route path="/" element={<FleetView />} />
          <Route path="/verdict/:id" element={<VerdictView />} />
          <Route path="/node/:id" element={<NodeView />} />
        </Routes>
      </div>
    </div>
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
