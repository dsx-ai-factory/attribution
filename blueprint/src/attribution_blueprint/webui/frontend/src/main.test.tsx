// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  createRoot: vi.fn(),
  render: vi.fn(),
}))

vi.mock('react-dom/client', () => ({
  createRoot: mocks.createRoot,
}))
vi.mock('@/app', () => ({ default: () => <div>App</div> }))
vi.mock('@/services/api-provider', () => ({
  ApiProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

beforeEach(() => {
  vi.resetModules()
  vi.clearAllMocks()
  document.body.innerHTML = '<div id="root"></div>'
  mocks.createRoot.mockReturnValue({ render: mocks.render })
})

describe('frontend entry point', () => {
  it('mounts the application into the root element', async () => {
    await import('@/main')

    expect(mocks.createRoot).toHaveBeenCalledWith(document.getElementById('root'))
    expect(mocks.render).toHaveBeenCalledTimes(1)
  })
})
