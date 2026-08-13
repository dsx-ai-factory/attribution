import useSWR from 'swr'
import type { OutcomeKey } from '@/interfaces/attribution'
import type { DashboardPayload } from '@/interfaces/meta'
import { httpGet } from '@/services/http-client.service'

export function useDashboard({
  from,
  to,
  outcome,
}: {
  from?: string
  to?: string
  outcome?: OutcomeKey
}): {
  dashboard: DashboardPayload | null
  isLoading: boolean
  error: Error | null
} {
  const key = ['dashboard', from ?? '', to ?? '', outcome ?? '']
  const { data, isLoading, error } = useSWR<DashboardPayload, Error>(key, () =>
    httpGet<DashboardPayload>('/dashboard', { from, to, outcome }),
  )

  return { dashboard: data ?? null, isLoading, error: error ?? null }
}
