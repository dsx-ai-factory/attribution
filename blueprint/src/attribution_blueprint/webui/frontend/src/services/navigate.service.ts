// src/services/navigate.service.ts
// Single source of truth for route aliases. Navigate by name, never by a
// hand-built path string.
import { matchPath } from 'react-router-dom'

const RoutersMap = {
  fleet: '/',
  verdict: '/verdict/:id',
  node: '/node/:id',
} as const

export type Routers = typeof RoutersMap
export type RouterName = keyof Routers
export type RouteParams = Record<string, string>

export class NavigateService {
  protected routersMap = RoutersMap

  /** Resolve a named route + params to a concrete path. */
  getPath(name: RouterName, params?: RouteParams | null): string {
    const template: string = this.routersMap[name]
    if (!params) return template
    return template.replace(/:([A-Za-z0-9_]+)\??/g, (_match, key: string) => {
      const value = params[key]
      return value === undefined ? '' : encodeURIComponent(value)
    })
  }

  /** Reverse-match a pathname to a route name + params (for the breadcrumb/title). */
  match(pathname: string): { name: RouterName; params: RouteParams } {
    for (const name of Object.keys(this.routersMap) as RouterName[]) {
      if (name === 'fleet') continue // root; falls through as the default
      const matched = matchPath({ path: this.routersMap[name], end: true }, pathname)
      if (matched) return { name, params: (matched.params as RouteParams) ?? {} }
    }
    return { name: 'fleet', params: {} }
  }
}

export const navigateService = new NavigateService()
