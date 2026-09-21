import useSWR, { type KeyedMutator } from 'swr'
import type { RunSummary } from '@/interfaces/attribution'
import type { Meta } from '@/interfaces/meta'
import { useApi } from '@/services/api-context'

type RunsData = { runs: RunSummary[]; meta: Meta }

const EMPTY: RunSummary[] = [] // stable reference — avoids re-render churn

/**
 * Owns the fleet's server state. SWR caches under a single static key, so the
 * five components that call this share ONE fetch + cache entry instead of each
 * mounting its own effect.
 */
export function useRuns(): {
  runs: RunSummary[]
  meta: Meta | null
  isLoading: boolean
  error: Error | null
  mutate: KeyedMutator<RunsData>
} {
  const api = useApi()
  const { data, isLoading, error, mutate } = useSWR<RunsData, Error>('runs', () =>
    Promise.all([api.getAllRuns(), api.getMeta()]).then(([runs, meta]) => ({ runs, meta })),
  )
  return { runs: data?.runs ?? EMPTY, meta: data?.meta ?? null, isLoading, error: error ?? null, mutate }
}
