// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from '@/app'

vi.mock('@/routers/fleet', () => ({ FleetView: () => <div>Fleet screen</div> }))
vi.mock('@/routers/verdict', () => ({ VerdictView: () => <div>Verdict screen</div> }))
vi.mock('@/routers/node', () => ({ NodeView: () => <div>Node screen</div> }))

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
  ])('renders %s and resets scroll', (path, expected, routeName) => {
    window.history.replaceState({}, '', path)
    render(<App />)

    expect(screen.getByText(expected)).toBeTruthy()
    expect(screen.getByText(`NVIDIA Attribution · v0.3.0 · ${routeName}`)).toBeTruthy()
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'instant' })
  })

  it('returns from a drill-down when Escape is pressed', async () => {
    window.history.replaceState({}, '', '/node/node-a')
    render(<App />)

    fireEvent.keyDown(window, { key: 'Escape' })

    await waitFor(() => expect(screen.getByText('Fleet screen')).toBeTruthy())
  })
})
