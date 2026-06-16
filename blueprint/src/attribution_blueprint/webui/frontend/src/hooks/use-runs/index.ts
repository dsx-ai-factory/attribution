import useSWR, { type KeyedMutator } from 'swr'
import type { Run } from '@/interfaces/attribution'
import type { Meta } from '@/interfaces/meta'
import { useApi } from '@/services/api-context'

type RunsData = { runs: Run[]; meta: Meta }

const EMPTY: Run[] = [] // stable reference — avoids re-render churn

/**
 * Owns the fleet's server state. SWR caches under a single static key, so the
 * five components that call this share ONE fetch + cache entry instead of each
 * mounting its own effect.
 */
export function useRuns(): {
  runs: Run[]
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
