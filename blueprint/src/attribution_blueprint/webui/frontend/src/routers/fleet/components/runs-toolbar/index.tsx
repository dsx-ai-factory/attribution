// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useEffectEvent, useId, useState } from 'react'
import { Search, X } from 'lucide-react'
import type { OutcomeKey, RunSummary } from '@/interfaces/attribution'
import { OUTCOMES, OUTCOME_ORDER } from '@/lib/attribution-data'
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
import { Button } from '@/components/ui/button'
import { CheckboxLabel } from '@/components/ui/checkbox-label'
import { InputGroup, InputGroupAddon, InputGroupInput } from '@/components/ui/input-group'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ExportMenu } from '@/components/export-menu'
import { Identifier } from '@/components/field'
import { OutcomeDot } from '@/components/outcome'
import { cn, markInRow } from '@/lib/utils'

type OutcomeChoice = 'all' | OutcomeKey

// Outcome `pass` and status `failed` select disjoint sets, so holding both yields an empty
// table. The one most recently touched wins and releases the other.
function withOutcome(filter: FleetFilter, choice: OutcomeChoice): FleetFilter {
  return {
    ...filter,
    outcome: choice === 'all' ? undefined : choice,
    status: choice === 'pass' ? 'all' : filter.status,
  }
}

function withPassingRunsHidden(filter: FleetFilter, hidden: boolean): FleetFilter {
  return {
    ...filter,
    status: hidden ? 'failed' : 'all',
    outcome: hidden && filter.outcome === 'pass' ? undefined : filter.outcome,
  }
}

// How long typing pauses before the run list follows the search field.
const SEARCH_SETTLE_MS = 300

// The search field holds what is typed in its own state, so every keystroke
// shows at once: the URL it writes commits later, in the router's transition,
// and a field fed straight from the URL drops keys typed in between. The URL,
// and the runs it lists, follow once typing pauses. A search set from
// elsewhere (a cleared filter chip, history) replaces the text; the field's own
// write coming back does not.
function RunSearch({ value, onSearch }: { value: string; onSearch: (q: string) => void }) {
  const [text, setText] = useState(value)
  const [seen, setSeen] = useState(value)
  const [written, setWritten] = useState<string | null>(null)
  if (value !== seen) {
    setSeen(value)
    if (value === written) setWritten(null)
    else setText(value)
  }
  const search = useEffectEvent((q: string) => {
    setWritten(q)
    onSearch(q)
  })
  useEffect(() => {
    if (text === value) return
    const timer = window.setTimeout(() => search(text), SEARCH_SETTLE_MS)
    return () => window.clearTimeout(timer)
  }, [text, value])

  return (
    <InputGroup className="w-72 max-w-full">
      <InputGroupAddon>
        <Search />
      </InputGroupAddon>
      <InputGroupInput
        name="q"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Filter by context, user, symptom…"
        aria-label="Filter runs by execution context, workload, user, symptom, rule, node, attempt UID or detail"
      />
    </InputGroup>
  )
}

// The run list's own controls, in its card header: text search, the passing
// runs switch, the outcome choice (each option marked by its outcome dot) and
// the export of the rows on screen.
export function RunsToolbar({
  filteredRuns,
  filter,
  setFilter,
}: {
  filteredRuns: RunSummary[]
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
}) {
  return (
    <div className="flex min-w-0 flex-wrap items-center justify-end gap-x-4 gap-y-1">
      <RunSearch value={filter.q ?? ''} onSearch={(q) => setFilter({ ...filter, q: q || undefined })} />
      <CheckboxLabel
        checked={filter.status === 'failed'}
        onCheckedChange={(hidden) => setFilter(withPassingRunsHidden(filter, hidden))}
      >
        Hide passing runs
      </CheckboxLabel>
      <Select
        value={filter.outcome ?? 'all'}
        onValueChange={(v) => setFilter(withOutcome(filter, v as OutcomeChoice))}
      >
        <SelectTrigger aria-label="Outcome">
          <span className="text-foreground-muted">Outcome</span>
          <SelectValue />
        </SelectTrigger>
        <SelectContent align="end">
          {/* All has no dot, so its label starts past where the dot and its
              8px gap would be, on the same column as every outcome's label. */}
          <SelectItem value="all" className="pl-6">All</SelectItem>
          {OUTCOME_ORDER.map((key) => (
            <SelectItem key={key} value={key}>
              <OutcomeDot outcome={key} />
              {OUTCOMES[key].label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <ExportMenu
        triggerLabel={`Export visible (${filteredRuns.length})`}
        items={[
          {
            label: 'Download as JSON',
            onSelect: () =>
              downloadBlob(
                `runs-${utcDateStamp()}.json`,
                JSON.stringify(runsExportData(filteredRuns), null, 2),
                'application/json',
              ),
          },
          {
            label: 'Download as YAML',
            onSelect: () =>
              downloadBlob(
                `runs-${utcDateStamp()}.yaml`,
                toYaml(runsExportData(filteredRuns)),
                'application/yaml',
              ),
          },
          {
            label: 'Download as CSV',
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
  )
}

// Every filter narrowing the run list, each as a "Label: value" badge with its
// own clear button. Recorded values (a symptom, a node, a search) wrap rather
// than being cut, after their separators first; the label before one never
// shrinks with it, since a wrapping badge breaks any of its text anywhere, a
// label's too. Nothing renders while no filter is set. The bar is a 28px row,
// as the table rows under it: a chip is a 20px badge, so its clear button is
// 16px (a 12px icon), with no other target within the 24px circle WCAG 2.5.8
// spaces an undersized target by. The button is a mark beside the value, as a
// copy button is, so it takes the mark rule (markInRow) and keeps to a wrapped
// value's first line. The chip does not clip, so the button's focus ring
// shows whole.
export function ActiveFilters({
  filter,
  setFilter,
}: {
  filter: FleetFilter
  setFilter: (next: FleetFilter) => void
}) {
  const labelId = useId()
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
  if (filter.status === 'failed') {
    chips.push({
      key: 'status',
      label: 'Status',
      value: 'Passing runs hidden',
      clear: () => setFilter(withPassingRunsHidden(filter, false)),
    })
  }
  if (chips.length === 0) return null

  return (
    <div className="flex flex-wrap items-center gap-2 border-b px-card-x py-1">
      <span id={labelId} className="text-sm text-foreground-muted">Active filters</span>
      <ul aria-labelledby={labelId} className="flex flex-wrap items-center gap-2">
        {chips.map((c) => (
          <li key={c.key}>
            <Badge size="wrap" className="gap-1 overflow-visible pr-0.5">
              <span className="shrink-0 text-foreground-muted">{c.label}:</span>
              <Identifier>{c.value}</Identifier>
              <Button
                variant="ghost"
                size="icon-xs"
                className={cn('size-4', markInRow)}
                onClick={c.clear}
                aria-label={`Clear ${c.label} filter`}
              >
                <X />
              </Button>
            </Badge>
          </li>
        ))}
      </ul>
    </div>
  )
}
