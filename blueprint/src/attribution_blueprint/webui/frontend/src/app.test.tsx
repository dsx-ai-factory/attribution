// @vitest-environment jsdom
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import App from '@/app'

vi.mock('@/routers/fleet', () => ({ FleetView: () => <div>Fleet screen</div> }))
vi.mock('@/routers/verdict', () => ({ VerdictView: () => <div>Verdict screen</div> }))
vi.mock('@/routers/node', () => ({ NodeView: () => <div>Node screen</div> }))
vi.mock('@/hooks/use-meta', () => ({
  useMeta: () => ({
    meta: { database: 'demo.db', versions: { schema: '1', pipeline: null, service: null } },
  }),
}))

beforeEach(() => {
  window.history.replaceState({}, '', '/')
  window.scrollTo = vi.fn()
})

afterEach(cleanup)

describe('App routing shell', () => {
  it.each([
    ['/', 'Fleet screen', 'fleet'],
    ['/verdict/run-1', 'Verdict screen', 'verdict'],
    ['/node/node-a', 'Node screen', 'node'],
  ])('renders %s under the app bar and resets scroll', (path, expected, routeName) => {
    window.history.replaceState({}, '', path)
    render(<App />)

    expect(within(screen.getByRole('main')).getByText(expected)).toBeTruthy()
    // One bar on every route: the brand leads home, and the backend is named from /meta alone.
    const bar = screen.getByRole('banner')
    expect(within(bar).getByRole('link', { name: 'NVIDIA Attribution' }).getAttribute('href')).toBe('/')
    expect(bar.textContent).toBe('NVIDIA AttributionDatabase demo.db · schema 1')
    // The bar no longer names the route or claims a state the server does not report.
    expect(bar.textContent).not.toMatch(new RegExp(`${routeName}|v0\\.3\\.0|healthy|ingesting`))
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'instant' })
  })

  it('returns from a drill-down when Escape is pressed', async () => {
    window.history.replaceState({}, '', '/node/node-a')
    render(<App />)

    fireEvent.keyDown(window, { key: 'Escape' })

    await waitFor(() => expect(screen.getByText('Fleet screen')).toBeTruthy())
  })

  it('keeps the current route when a popup has handled Escape', () => {
    window.history.replaceState({}, '', '/verdict/run-1')
    render(<App />)

    const event = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true })
    event.preventDefault()
    fireEvent(window, event)

    expect(screen.getByText('Verdict screen')).toBeTruthy()
    expect(window.location.pathname).toBe('/verdict/run-1')
  })
})
