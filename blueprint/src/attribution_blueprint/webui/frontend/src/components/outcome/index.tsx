// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { X } from 'lucide-react'
import type { OutcomeKey, RunSummary } from '@/interfaces/attribution'
import { OUTCOMES } from '@/lib/attribution-data'
import { Badge } from '@/components/ui/badge'
import { cn, markInRow } from '@/lib/utils'

const TERMINAL_STATE_LABELS: Record<string, string> = {
  Pass: 'Pass',
  Fail: 'Fail',
  ExpectedFail: 'Expected failure',
  Cancelled: 'Cancelled',
}

// Short forms for the badge right half — verbose forms live in verdictFacts rows.
const FAILURE_DOMAIN_BADGE_LABELS: Record<string, string> = {
  User: 'Application',
  InfraHW: 'Hardware',
  InfraSW: 'Software',
}

const CANCELLATION_ACTOR_LABELS: Record<string, string> = {
  User: 'Workload owner',
  Operator: 'Operator',
}

// The house StatusDots sm mark (a -500 fill inside a -400 edge), painted from
// the data-outcome tone in globals.css rather than per-status color classes.
export function OutcomeDot({ outcome }: { outcome: OutcomeKey }) {
  return (
    <span
      data-slot="outcome-dot"
      data-outcome={outcome}
      aria-hidden="true"
      className={cn('inline-block size-2 shrink-0 rounded-[2px] border border-(--tone-edge) bg-(--tone)', markInRow)}
    />
  )
}

// Color stays on the dot: the label is neutral text on an outline badge, so the
// badge never reads as a status fill (and never matches the solid zinc-subtle
// evidence badge in shape).
export function OutcomeBadge({ outcome, label, className }: { outcome: OutcomeKey; label?: string; className?: string }) {
  return (
    <Badge variant="outline" data-outcome={outcome} className={cn('gap-1.5', className)}>
      <OutcomeDot outcome={outcome} />
      {label || OUTCOMES[outcome].label}
    </Badge>
  )
}

// Terminal-state icons carry their own fixed color, independent of the
// attribution outcome tone: green circle (Pass), red X (Fail), yellow
// upward triangle (Cancelled), neutral square (ExpectedFail / unknown).
function TerminalStateIcon({ state }: { state: string }) {
  if (state === 'Pass') {
    return (
      <span
        aria-hidden="true"
        className={cn('inline-block size-2 shrink-0 rounded-full bg-green-500', markInRow)}
      />
    )
  }
  if (state === 'Fail') {
    return (
      <X
        aria-hidden="true"
        className={cn('size-3 shrink-0 text-red-500', markInRow)}
        strokeWidth={2.5}
      />
    )
  }
  if (state === 'Cancelled') {
    return (
      <svg
        aria-hidden="true"
        viewBox="0 0 10 9"
        className={cn('size-2.5 shrink-0 fill-yellow-400', markInRow)}
      >
        {/* Isosceles triangle, point up */}
        <polygon points="5,0.5 9.5,8.5 0.5,8.5" />
      </svg>
    )
  }
  // ExpectedFail or unknown: neutral square matching the standard dot shape.
  return (
    <span
      aria-hidden="true"
      className={cn('inline-block size-2 shrink-0 rounded-[2px] bg-zinc-400', markInRow)}
    />
  )
}

type CompoundProps = Pick<RunSummary, 'outcome' | 'verdictLabel' | 'terminalStates' | 'failureDomains' | 'cancellationActors' | 'uncertaintyType'>

// Two-part pill: left side shows the terminal-state icon + label; right side
// (when present) shows the attribution detail in the outcome tone color,
// separated by a tone-colored divider. Falls back to OutcomeBadge when
// compound fields are absent (pre-V2 data).
export function CompoundOutcomeBadge({ run, className }: { run: CompoundProps; className?: string }) {
  const terminalStates = run.terminalStates ?? []
  const failureDomains = run.failureDomains ?? []
  const cancellationActors = run.cancellationActors ?? []
  const uncertaintyType = run.uncertaintyType ?? null

  if (terminalStates.length === 0 && failureDomains.length === 0 && cancellationActors.length === 0 && uncertaintyType === null) {
    return <OutcomeBadge outcome={run.outcome} label={run.verdictLabel} className={className} />
  }

  const singleState = terminalStates.length === 1 ? terminalStates[0] : null
  const leftLabel = singleState !== null
    ? (TERMINAL_STATE_LABELS[singleState] ?? singleState)
    : run.verdictLabel

  let rightLabel: string | null = null
  if (failureDomains.length === 1) {
    rightLabel = FAILURE_DOMAIN_BADGE_LABELS[failureDomains[0]] ?? failureDomains[0]
  } else if (failureDomains.length > 1) {
    rightLabel = uncertaintyType === 'Epistemic' ? 'Undetermined' : 'Ambiguous'
  } else if (cancellationActors.length === 1) {
    rightLabel = CANCELLATION_ACTOR_LABELS[cancellationActors[0]] ?? cancellationActors[0]
  } else if (cancellationActors.length > 1) {
    rightLabel = 'Multiple'
  }

  return (
    <Badge variant="outline" data-outcome={run.outcome} className={cn('gap-0 px-0', className)}>
      <span className="flex items-center gap-1.5 px-1.5">
        {singleState !== null && <TerminalStateIcon state={singleState} />}
        {leftLabel}
      </span>
      {rightLabel !== null && (
        <>
          <span aria-hidden="true" className="self-stretch border-l border-(--tone-edge)" />
          <span className="px-1.5 text-(--tone)">{rightLabel}</span>
        </>
      )}
    </Badge>
  )
}
