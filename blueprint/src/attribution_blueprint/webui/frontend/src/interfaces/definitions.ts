/**
 * Keys and shape for metric / column / chip definitions surfaced via <DefHover>.
 * Keep keys kebab-case, labels human-cased.
 */
export type DefTerm =
  | 'mtbi'
  | 'combined-mtbi'
  | 'hw-mtbi'
  | 'failures'
  | 'gpu-hours'
  | 'window'
  | 'root-symptom'
  | 'responsible-node'
  | 'rule'
  | 'count'
  | 'vital-few'
  | 'cumulative-pct'
  | 'outcome-share'
  | 'end-time-utc'
  | 'outcome'
  | 'responsible-nodes'
  | 'attribution-outcome'
  | 'root-symptom-symptom'
  | 'attempt-uid'
  | 'attempts-impacted'
  | 'distinct-symptoms'
  | 'gpu-hours-observed'

export interface DefEntry {
  /** Human-readable label shown when the trigger has no children. */
  label: string
  /** One-sentence definition. */
  def: string
  /** Optional single-line formula. */
  formula?: string
  /** Optional one-line realistic example. */
  example?: string
}
