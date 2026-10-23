// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState, type ReactElement } from 'react'
import { summarizeNode } from '@/lib/node-summary'
import { useNodeRuns } from '@/hooks/use-node-runs'
import { DescriptionList } from '@/components/ui/description-list'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { Skeleton } from '@/components/ui/skeleton'
import { Identifier } from '@/components/field'

// A node's recorded history on the house hover card, as a description list on
// the 112px md label axis: "GPU-hours on node", the longest label, measures
// 109.6px in 12px NVIDIA Sans (106.7px in its Arial fallback) and must not
// break onto two lines. The trigger is always the caller's own link, so the
// card adds no control and no link styling of its own; runs are fetched only
// once the card opens. Nothing is summarized before they arrive, so a loading
// or failed fetch never reads as a node with no attempts. The summary is the
// node page's (summarizeNode): a top symptom is named only when one stands
// out, never one of a tie.
export function NodeHoverCard({ nodeId, children }: { nodeId: string; children: ReactElement }) {
  const [open, setOpen] = useState(false)
  const { runs, isLoading, error } = useNodeRuns(nodeId, open)
  const s = useMemo(() => summarizeNode(runs, nodeId), [runs, nodeId])

  return (
    <HoverCard open={open} onOpenChange={setOpen} openDelay={200} closeDelay={75}>
      <HoverCardTrigger asChild>{children}</HoverCardTrigger>
      <HoverCardContent align="start" size="lg">
        <div className="flex flex-col gap-2">
          <div className="font-medium break-words text-foreground"><Identifier>{nodeId}</Identifier></div>
          {/* As on the node page: a failed revalidation keeps showing the runs already fetched. */}
          {isLoading ? (
            <div aria-busy="true" className="flex flex-col gap-2">
              <Skeleton height="4" width="2/3" />
              <Skeleton height="4" width="1/2" />
              <Skeleton height="4" width="2/3" />
            </div>
          ) : error !== null && runs.length === 0 ? (
            <p className="text-foreground-muted">Runs for this node could not be loaded.</p>
          ) : (
            <DescriptionList
              labelWidth="md"
              items={[
                { key: 'attempts', label: 'Attempts', value: <span className="num">{s.attempts.length}</span> },
                { key: 'responsible', label: 'Responsible in', value: <span className="num">{s.responsible.length}</span> },
                { key: 'symptoms', label: 'Distinct symptoms', value: <span className="num">{s.symptoms.length}</span> },
                { key: 'hours', label: 'GPU-hours on node', value: <span className="num">{s.gpuHours.toFixed(2)}</span> },
                ...(s.leading ? [{ key: 'top', label: 'Top symptom', value: <Identifier>{s.leading[0]}</Identifier> }] : []),
              ]}
            />
          )}
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
