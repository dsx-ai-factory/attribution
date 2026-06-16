import useSWR from 'swr'
import type { OutcomeKey, Run } from '@/interfaces/attribution'
import type { FleetFilter } from '@/interfaces/filter'
import type { RunsPagePayload } from '@/interfaces/meta'
import { httpGet } from '@/services/http-client.service'

export interface RunsTableGroup {
  outcome: OutcomeKey
  runs: Run[]
  total: number
  page: number
  pageSize: number
}

export type PageByOutcome = Partial<Record<OutcomeKey, number>>

export function useRunsTable({
  filter,
  outcomes,
  pageByOutcome,
  pageSize,
}: {
  filter: FleetFilter
  outcomes: OutcomeKey[]
  pageByOutcome: PageByOutcome
  pageSize: number
}): {
  groups: RunsTableGroup[]
  total: number
  visibleRuns: Run[]
  isLoading: boolean
  error: Error | null
} {
  const key = [
    'runs-table',
    filter.from ?? '',
    filter.to ?? '',
    filter.status,
    filter.outcome ?? '',
    filter.symptom ?? '',
    filter.node ?? '',
    filter.rule ?? '',
    filter.user ?? '',
    filter.q ?? '',
    pageSize,
    outcomes.join(','),
    JSON.stringify(pageByOutcome),
  ]
  const { data, isLoading, error } = useSWR<RunsTableGroup[], Error>(key, async () => {
    const pages = await Promise.all(
      outcomes.map(async (outcome) => {
        const page = pageByOutcome[outcome] ?? 0
        const payload = await httpGet<RunsPagePayload>('/runs', {
          from: filter.from,
          to: filter.to,
          outcome,
          q: filter.q,
          status: filter.status,
          symptom: filter.symptom,
          node: filter.node,
          rule: filter.rule,
          user: filter.user,
          page,
          pageSize,
        })
        return {
          outcome,
          runs: payload.runs,
          total: payload.total,
          page: payload.page,
          pageSize: payload.pageSize,
        }
      }),
    )
    return pages.filter((group) => group.total > 0)
  })

  const groups = data ?? []
  const visibleRuns = groups.flatMap((group) => group.runs)
  const total = groups.reduce((sum, group) => sum + group.total, 0)
  return { groups, total, visibleRuns, isLoading, error: error ?? null }
}
