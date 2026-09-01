import useSWR from 'swr'
import type { Meta } from '@/interfaces/meta'
import { httpGet } from '@/services/http-client.service'

/**
 * Facts about the backend the dashboard is bound to, which no filter narrows.
 * One static SWR key, so mounting this in the shell costs a single request.
 */
export function useMeta(): { meta: Meta | null } {
  const { data } = useSWR<Meta, Error>('meta', () => httpGet<Meta>('/meta'))
  return { meta: data ?? null }
}
