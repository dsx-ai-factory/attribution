import { useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { navigateService, type RouterName, type RouteParams } from '@/services/navigate.service'

/**
 * Programmatic navigation by route name. Replaces the prop-drilled `navigate`
 * callback — any component under the router can call this directly. The
 * returned function is stable (safe for dependency arrays).
 */
export function useNav() {
  const navigate = useNavigate()
  return useCallback(
    (name: RouterName, params?: RouteParams) => navigate(navigateService.getPath(name, params)),
    [navigate],
  )
}
