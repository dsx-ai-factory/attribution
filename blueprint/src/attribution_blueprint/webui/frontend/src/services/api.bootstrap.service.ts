import type { FleetFilter } from '@/interfaces/filter'
import { applyFilter } from '@/lib/filter-state'
import type { Run } from '@/interfaces/attribution'
import type { BootstrapPayload, Meta } from '@/interfaces/meta'
import { ApiService } from '@/services/api.service'

const EMPTY_META: Meta = {
  versions: { schema: null, pipeline: null, service: null },
  dateBounds: { min: null, max: null }, windowedRuns: 0, dbTotalRuns: 0, database: null, returnedRuns: 0,
}

/** Reads the server-injected bootstrap blob from the DOM, if present. */
export function readBootstrap(): BootstrapPayload | null {
  const el = document.getElementById('attr-bootstrap')
  if (!el || !el.textContent) return null
  try {
    return JSON.parse(el.textContent) as BootstrapPayload
  } catch {
    return null
  }
}

/** MVP source: reads the injected blob; filters client-side via applyFilter. */
export class ApiBootstrapService implements ApiService {
  private payload(): BootstrapPayload {
    return readBootstrap() ?? { runs: [], meta: EMPTY_META }
  }
  async getAllRuns(): Promise<Run[]> {
    return this.payload().runs
  }
  async getRuns(filter: FleetFilter): Promise<Run[]> {
    return applyFilter(this.payload().runs, filter)
  }
  async getRun(id: string): Promise<Run | undefined> {
    return this.payload().runs.find((r) => r.id === id)
  }
  async getMeta(): Promise<Meta> {
    return this.payload().meta
  }
}
