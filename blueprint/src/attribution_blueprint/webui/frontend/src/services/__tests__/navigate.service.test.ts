import { describe, expect, it } from 'vitest'
import { NavigateService } from '@/services/navigate.service'

describe('NavigateService', () => {
  const navigate = new NavigateService()

  it('builds named paths and encodes route parameters', () => {
    expect(navigate.getPath('fleet')).toBe('/')
    expect(navigate.getPath('verdict', { id: 'team/job 1' })).toBe('/verdict/team%2Fjob%201')
    expect(navigate.getPath('node', { id: 'node-a' })).toBe('/node/node-a')
  })

  it('reverse-matches routed and unknown paths', () => {
    expect(navigate.match('/verdict/team%2Fjob')).toEqual({
      name: 'verdict',
      params: { id: 'team/job' },
    })
    expect(navigate.match('/node/node-a')).toEqual({
      name: 'node',
      params: { id: 'node-a' },
    })
    expect(navigate.match('/unknown')).toEqual({ name: 'fleet', params: {} })
  })
})
