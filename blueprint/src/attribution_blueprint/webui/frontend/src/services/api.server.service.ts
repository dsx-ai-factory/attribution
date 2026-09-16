import type { FleetFilter } from '@/interfaces/filter'
import type { RunDetail, RunSummary } from '@/interfaces/attribution'
import type { Meta, RunsPagePayload } from '@/interfaces/meta'
import { ApiService } from '@/services/api.service'
import { httpGet } from '@/services/http-client.service'

/** HTTP source: /api/v1/*. */
export class ApiServerService implements ApiService {
  async getAllRuns(): Promise<RunSummary[]> {
    const runs: RunSummary[] = []
    let page = 0
    let total: number | null = null
    while (total === null || runs.length < total) {
      const payload = await httpGet<RunsPagePayload>('/runs', { page, pageSize: 100, status: 'all' })
      runs.push(...payload.runs)
      total = payload.total
      if (payload.runs.length === 0) break
      page += 1
    }
    return runs
  }
  async getRuns(filter: FleetFilter): Promise<RunSummary[]> {
    const res = await httpGet<RunsPagePayload>('/runs', {
      from: filter.from,
      to: filter.to,
      outcome: filter.outcome,
      q: filter.q,
      status: filter.status,
      symptom: filter.symptom,
      node: filter.node,
      rule: filter.rule,
      user: filter.user,
      pageSize: 10,
    })
    return res.runs
  }
  async getRun(id: string): Promise<RunDetail | undefined> {
    return httpGet<RunDetail>(`/runs/${encodeURIComponent(id)}`)
  }
  async getMeta(): Promise<Meta> {
    return httpGet<Meta>('/meta')
  }
}
