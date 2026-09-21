import type { FleetFilter } from '@/interfaces/filter'
import type { RunDetail, RunSummary } from '@/interfaces/attribution'
import type { Meta } from '@/interfaces/meta'

/**
 * The data contract for the UI. Abstract so the provider can swap the concrete
 * implementation (bootstrap blob by default, HTTP later).
 */
export abstract class ApiService {
  /** The FULL injected/served run set, unfiltered. Views own all filtering. */
  abstract getAllRuns(): Promise<RunSummary[]>
  abstract getRuns(filter: FleetFilter): Promise<RunSummary[]>
  abstract getRun(id: string): Promise<RunDetail | undefined>
  abstract getMeta(): Promise<Meta>
}
