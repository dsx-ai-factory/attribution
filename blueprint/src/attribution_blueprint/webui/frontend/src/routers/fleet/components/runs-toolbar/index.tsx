import { Search, X } from 'lucide-react'
import type { OutcomeKey, RunSummary } from '@/interfaces/attribution'
import { OUTCOMES } from '@/lib/attribution-data'
import type { FleetFilter } from '@/interfaces/filter'
import { withDimension } from '@/lib/filter-state'
import {
  downloadBlob,
  runsToCSV,
  runsExportData,
  toYaml,
  utcDateStamp,
} from '@/lib/export'
import { Badge } from '@/components/ui/badge'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { ExportMenu } from '@/components/export-menu'

type OutcomeChoice = 'all' | OutcomeKey
const OUTCOME_ORDER: { key: OutcomeKey; label: string; dot: string }[] = [
  { key: 'hardware', label: 'Hardware Fault',        dot: 'bg-red-400' },
  { key: 'software', label: 'System Software Fault', dot: 'bg-blue-400' },
  { key: 'multi',    label: 'Ambiguous Fault',       dot: 'bg-zinc-400' },
  { key: 'user',     label: 'Application Fault',     dot: 'bg-yellow-400' },
  { key: 'undetermined', label: 'Unattributed',    dot: 'bg-purple-500' },
  { key: 'pass',     label: 'Pass',                  dot: 'bg-primary' },
]

function Chip({
  label,
  value,
  onClear,
}: {
  label: string
  value: string
  onClear: () => void
}) {
  return (
    <span className="inline-flex items-center gap-1 font-mono text-[11px] uppercase tracking-wide bg-background-active border border-border rounded-sm pl-1.5 pr-0.5 py-0.5">
      <span className="text-foreground-muted">{label}</span>
      <span className="text-foreground normal-case tracking-normal truncate max-w-[12rem]">
        {value}
      </span>
      <button
        type="button"
        onClick={onClear}
        aria-label={`Clear ${label} filter`}
        className="ml-0.5 inline-flex items-center justify-center w-4 h-4 rounded-sm text-foreground-muted hover:text-foreground hover:bg-background-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
      >
        <X className="size-3" />
      </button>
    </span>
  )
}

export function RunsToolbar({
  runs,
  filteredRuns,
  filter,
  setFilter,
}: {
  runs: number
  filteredRuns: RunSummary[]
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
}) {
  const outcomeFilter: OutcomeChoice = filter.outcome ?? 'all'
  const query = filter.q ?? ''
  const hidePassingRuns = filter.status === 'failed'

  // Outcome `pass` and status `failed` select disjoint sets, so holding both yields an empty
  // table. The one most recently touched wins and releases the other.
  const setOutcome = (v: OutcomeChoice) =>
    setFilter({
      ...filter,
      outcome: v === 'all' ? undefined : v,
      status: v === 'pass' ? 'all' : filter.status,
    })
  const setQuery = (q: string) => setFilter({ ...filter, q: q || undefined })
  const setHidePassingRuns = (on: boolean) =>
    setFilter({
      ...filter,
      status: on ? 'failed' : 'all',
      outcome: on && filter.outcome === 'pass' ? undefined : filter.outcome,
    })

  const chips: Array<{ key: string; label: string; value: string; clear: () => void }> = []
  if (filter.outcome) {
    chips.push({
      key: 'outcome',
      label: 'Outcome',
      value: OUTCOMES[filter.outcome].label,
      clear: () => setFilter(withDimension(filter, 'outcome', undefined)),
    })
  }
  if (filter.symptom) {
    chips.push({
      key: 'symptom',
      label: 'Symptom',
      value: filter.symptom,
      clear: () => setFilter(withDimension(filter, 'symptom', undefined)),
    })
  }
  if (filter.node) {
    chips.push({
      key: 'node',
      label: 'Responsible node',
      value: filter.node,
      clear: () => setFilter(withDimension(filter, 'node', undefined)),
    })
  }
  if (filter.rule) {
    chips.push({
      key: 'rule',
      label: 'Rule',
      value: filter.rule,
      clear: () => setFilter(withDimension(filter, 'rule', undefined)),
    })
  }
  if (filter.user) {
    chips.push({
      key: 'user',
      label: 'User',
      value: filter.user,
      clear: () => setFilter(withDimension(filter, 'user', undefined)),
    })
  }
  if (filter.q) {
    chips.push({
      key: 'q',
      label: 'Search',
      value: filter.q,
      clear: () => setFilter({ ...filter, q: undefined }),
    })
  }
  if (hidePassingRuns) {
    chips.push({
      key: 'status',
      label: 'Status',
      value: 'Passing runs hidden',
      clear: () => setHidePassingRuns(false),
    })
  }

  return (
    <div className="flex flex-col gap-2 mb-3.5">
      <div className="font-mono text-xs font-semibold tracking-widest uppercase text-foreground-muted flex items-center gap-2.5">
        Runs
        <Badge variant="outline" className="font-mono text-foreground-secondary">
          {runs}
        </Badge>
        <span className="flex-1 h-px bg-border" />
      </div>

      <div className="flex flex-wrap items-center gap-2.5">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-foreground-muted pointer-events-none" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by execution context, workload, user, symptom, rule, node, UID, detail"
            className="bg-card border border-border rounded-sm pl-8 pr-2 py-1.5 font-mono text-xs text-foreground w-96 focus:outline-none focus:border-zinc-700"
          />
        </div>

        <label className="flex items-center gap-1.5 font-mono text-xs text-foreground-muted cursor-pointer select-none">
          <input
            type="checkbox"
            checked={hidePassingRuns}
            onChange={(e) => setHidePassingRuns(e.target.checked)}
            className="accent-primary"
          />
          Hide passing runs
        </label>

        <div className="flex-1" />

        <ToggleGroup
          type="single"
          value={outcomeFilter}
          onValueChange={(v) => v && setOutcome(v as OutcomeChoice)}
          variant="outline"
          size="sm"
          className="font-mono"
        >
          <ToggleGroupItem value="all">All outcomes</ToggleGroupItem>
          {OUTCOME_ORDER.map((g) => (
            <ToggleGroupItem key={g.key} value={g.key} className="gap-2">
              <span className={`inline-block size-2 rounded-sm ${g.dot}`} />
              {g.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>

        <ExportMenu
          triggerLabel={`Export visible (${filteredRuns.length})`}
          items={[
            {
              label: 'Visible rows (JSON)',
              hint: 'one verdict object per run',
              onSelect: () =>
                downloadBlob(
                  `runs-${utcDateStamp()}.json`,
                  JSON.stringify(runsExportData(filteredRuns), null, 2),
                  'application/json',
                ),
            },
            {
              label: 'Visible rows (YAML)',
              hint: 'human-readable list',
              onSelect: () =>
                downloadBlob(
                  `runs-${utcDateStamp()}.yaml`,
                  toYaml(runsExportData(filteredRuns)),
                  'application/yaml',
                ),
            },
            {
              label: 'Visible rows (CSV)',
              hint: 'flat columns for spreadsheets',
              onSelect: () =>
                downloadBlob(
                  `runs-${utcDateStamp()}.csv`,
                  runsToCSV(filteredRuns),
                  'text/csv',
                ),
            },
          ]}
        />
      </div>

      {chips.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="font-mono text-[10px] uppercase tracking-widest text-foreground-muted">
            Active
          </span>
          {chips.map((c) => (
            <Chip key={c.key} label={c.label} value={c.value} onClear={c.clear} />
          ))}
        </div>
      )}
    </div>
  )
}
