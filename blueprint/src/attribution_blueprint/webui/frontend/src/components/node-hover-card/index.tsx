import { useMemo, useState } from 'react'
import Decimal from 'decimal.js'
import type { RunSummary } from '@/interfaces/attribution'
import { nodeGpuHours } from '@/lib/metric'
import { useNodeRuns } from '@/hooks/use-node-runs'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'

function summarizeNode(runs: RunSummary[], nodeId: string) {
  const attempts = runs.filter((r) => r.nodeIds.includes(nodeId) || r.responsibleNodes.includes(nodeId))
  const symptomCounts: Record<string, number> = {}
  let impacted = 0
  let nodeHours = new Decimal(0)
  attempts.forEach((r) => {
    nodeHours = nodeHours.plus(nodeGpuHours(r))
    if (r.responsibleNodes.includes(nodeId)) {
      impacted += 1
      const s = r.rootSymptom.symptom
      symptomCounts[s] = (symptomCounts[s] || 0) + 1
    }
  })
  const symptoms = Object.entries(symptomCounts)
    .sort(([, a], [, b]) => b - a)
  const topSymptom = symptoms[0]?.[0]
  return {
    attempts: attempts.length,
    impacted,
    distinctSymptoms: symptoms.length,
    nodeHours: nodeHours.toNumber(),
    topSymptom,
  }
}

export function NodeHoverCard({
  nodeId,
  children,
}: {
  nodeId: string
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(false)
  const { runs, isLoading } = useNodeRuns(nodeId, open)
  const s = useMemo(() => summarizeNode(runs, nodeId), [runs, nodeId])

  return (
    <HoverCard open={open} onOpenChange={setOpen} openDelay={200} closeDelay={75}>
      <HoverCardTrigger asChild>
        <span className="cursor-pointer underline decoration-foreground-muted underline-offset-2 hover:decoration-primary">
          {children}
        </span>
      </HoverCardTrigger>
      <HoverCardContent
        align="start"
        sideOffset={6}
        className="w-72 p-3 bg-background-secondary border-border ring-0"
      >
        <div className="font-mono text-xs text-foreground mb-2.5 truncate">{nodeId}</div>
        {isLoading && runs.length === 0 && (
          <div className="font-mono text-xs text-foreground-muted mb-2">Loading...</div>
        )}
        <div className="grid grid-cols-[7rem_1fr] gap-y-1.5 gap-x-3 font-mono text-xs">
          <div className="text-foreground-muted">attempts</div>
          <div className="text-foreground">{s.attempts}</div>
          <div className="text-foreground-muted">impacted</div>
          <div className="text-foreground">{s.impacted}</div>
          <div className="text-foreground-muted">distinct symptoms</div>
          <div className="text-foreground">{s.distinctSymptoms}</div>
          <div className="text-foreground-muted">GPU-hours</div>
          <div className="text-foreground">{s.nodeHours.toFixed(2)}h</div>
          {s.topSymptom && (
            <>
              <div className="text-foreground-muted">top symptom</div>
              <div className="text-foreground">{s.topSymptom}</div>
            </>
          )}
        </div>
        <div className="text-xs text-foreground-muted mt-3 pt-2 border-t border-border">
          Click to open node detail →
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
