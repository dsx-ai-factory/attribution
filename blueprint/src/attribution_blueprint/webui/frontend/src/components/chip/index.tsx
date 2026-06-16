import type { OutcomeKey } from '@/interfaces/attribution'
import { OUTCOMES } from '@/lib/attribution-data'
import { Badge } from '@/components/ui/badge'

/**
 * Outcome chips consume the same --outcome-* token family as Pareto bars.
 * Each entry: text/border driven by --outcome-<k>, background by --outcome-<k>-bg.
 * Keep this map in sync with src/styles/globals.css and OUTCOMES[k].color.
 */
const OUTCOME_CHIP_VARS: Record<OutcomeKey, string> = {
  hardware: '[--c:var(--outcome-hw)] [--bg:var(--outcome-hw-bg)]',
  software: '[--c:var(--outcome-sw)] [--bg:var(--outcome-sw-bg)]',
  multi:    '[--c:var(--outcome-mix)] [--bg:var(--outcome-mix-bg)]',
  user:     '[--c:var(--outcome-usr)] [--bg:var(--outcome-usr-bg)]',
  undetermined: '[--c:var(--outcome-und)] [--bg:var(--outcome-und-bg)]',
  pass:     '[--c:var(--outcome-pass)] [--bg:var(--outcome-pass-bg)]',
}

const SHARED =
  'font-mono uppercase tracking-wider font-semibold border text-[color:var(--c)] border-[color:var(--c)]/40 bg-[color:var(--bg)]'

export function OutcomeChip({ outcome }: { outcome: OutcomeKey }) {
  return (
    <Badge
      variant="outline"
      className={`${SHARED} ${OUTCOME_CHIP_VARS[outcome]}`}
    >
      {OUTCOMES[outcome].short}
    </Badge>
  )
}

export function VerdictPill({ outcome, label }: { outcome: OutcomeKey; label?: string }) {
  return (
    <Badge
      variant="outline"
      className={`${SHARED} ${OUTCOME_CHIP_VARS[outcome]} h-6 px-2.5`}
    >
      {label || OUTCOMES[outcome].label}
    </Badge>
  )
}
