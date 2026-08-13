import type { DefTerm, DefEntry } from '@/interfaces/definitions'

// DEFINITIONS is the single source of truth surfaced via <DefHover>; a DefTerm
// without an entry here fails typecheck (Record<DefTerm, DefEntry>).

export const DEFINITIONS: Record<DefTerm, DefEntry> = {
  'mtbi': {
    label: 'MTBI',
    def: 'Mean time between infrastructure incidents. Higher is healthier.',
    formula: 'kGPU-hours ÷ infra incidents',
    example: '25.58k ÷ 6 ≈ 4.26 kGPU-h / incident',
  },
  'combined-mtbi': {
    label: 'Combined MTBI',
    def: 'MTBI computed over HW + SW infrastructure failures combined.',
    formula: 'kGPU-hours ÷ (HW + SW failures)',
    example: '25.58k ÷ (3+2) = 5.12',
  },
  'hw-mtbi': {
    label: 'HW MTBI',
    def: 'MTBI computed over hardware failures only.',
    formula: 'kGPU-hours ÷ HW failures',
    example: '25.58k ÷ 3 = 8.53',
  },
  'failures': {
    label: 'Failures',
    def: 'Failed attempts in the window, split by outcome category. Click a bucket to filter the Runs table.',
  },
  'gpu-hours': {
    label: 'GPU-hours',
    def: 'Elapsed wall-clock hours multiplied by the number of GPUs in the attempt.',
    formula: 'elapsed_hours x gpu_count',
    example: '1.42 h x 12 GPU = 17.04 GPU-h',
  },
  'window': {
    label: 'Window',
    def: 'The date range and fleet scope for everything on this page. Times are UTC.',
  },
  'root-symptom': {
    label: 'Root symptom',
    def: 'The single symptom the attribution pipeline judged primary for this attempt.',
  },
  'responsible-node': {
    label: 'Responsible node',
    def: 'A node the attribution cascade points to as contributing to the root symptom.',
  },
  'rule': {
    label: 'Rule',
    def: 'The named detector that matched a log line and produced the symptom.',
    example: 'gpu_fell_off_bus',
  },
  'count': {
    label: 'Count',
    def: 'Number of failed attempts in the slice. Use Pareto to see concentration.',
  },
  'vital-few': {
    label: 'Vital few',
    def: 'The smallest set of items whose cumulative share reaches ~80%.',
  },
  'cumulative-pct': {
    label: 'Cumulative %',
    def: 'Running share of the metric, top-down, used to read the 80/20 break.',
  },
  'outcome-share': {
    label: 'Outcome share',
    def: 'Proportion of attempts in each outcome category for the current window.',
  },
  'end-time-utc': {
    label: 'End time (UTC)',
    def: 'Attempt end timestamp in UTC. Hover any cell for the user-local equivalent.',
  },
  'outcome': {
    label: 'Outcome',
    def: 'Attribution verdict category for the attempt.',
    example: 'Hardware Fault · System Software Fault · Application Fault · Ambiguous Fault · Unattributed · Pass',
  },
  'responsible-nodes': {
    label: 'Responsible nodes',
    def: 'Nodes the attribution cascade points to as contributing to this attempt\'s failure.',
  },
  'attribution-outcome': {
    label: 'attribution_outcome',
    def: 'Final classification recorded by the attribution pipeline.',
  },
  'root-symptom-symptom': {
    label: 'root_symptom.symptom',
    def: 'Canonical name of the primary symptom selected for this attempt.',
  },
  'attempt-uid': {
    label: 'attempt_uid',
    def: 'Globally unique identifier for this attempt, suitable for log correlation.',
  },
  'attempts-impacted': {
    label: 'Attempts impacted',
    def: 'Number of attempts that listed this node among responsible_nodes in the current window.',
  },
  'distinct-symptoms': {
    label: 'Distinct symptoms',
    def: 'Count of unique root symptoms attributed to this node in the current window.',
  },
  'gpu-hours-observed': {
    label: 'GPU-hours observed',
    def: 'Sum of GPU-hours distributed across listed nodes for attempts that ran on this node.',
  },
}

// Compile-time completeness check: any missing DefTerm here will fail TS.
const _exhaustive: Record<DefTerm, DefEntry> = DEFINITIONS
void _exhaustive
