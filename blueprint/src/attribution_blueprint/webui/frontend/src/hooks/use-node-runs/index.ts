import useSWR from 'swr'
import type { RunSummary } from '@/interfaces/attribution'
import type { RunsPagePayload } from '@/interfaces/meta'
import { httpGet } from '@/services/http-client.service'

const PAGE_SIZE = 100

async function fetchNodeRuns(nodeId: string): Promise<RunSummary[]> {
  const runs: RunSummary[] = []
  let page = 0
  let total: number | null = null

  while (total === null || runs.length < total) {
    const payload = await httpGet<RunsPagePayload>('/runs', {
      node: nodeId,
      status: 'all',
      page,
      pageSize: PAGE_SIZE,
    })
    runs.push(...payload.runs)
    total = payload.total
    if (payload.runs.length === 0) break
    page += 1
  }

  return runs
}

export function useNodeRuns(
  nodeId: string,
  enabled = true,
): {
  runs: RunSummary[]
  isLoading: boolean
  error: Error | null
} {
  const key = enabled && nodeId !== '' ? ['node-runs', nodeId] : null
  const { data, isLoading, error } = useSWR<RunSummary[], Error>(key, () => fetchNodeRuns(nodeId))
  return { runs: data ?? [], isLoading, error: error ?? null }
}
