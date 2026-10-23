// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SymptomSite } from '@/interfaces/attribution'
import { SITE_STATUS_LABELS } from '@/lib/attribution-data'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

// The house StatusBadge error/default levels. Red is the blame color, so only
// Responsible uses it; supporting evidence stays neutral. The count shares the
// label's text node, so the badge reads and matches as one phrase. Like
// every badge it is a mark beside text; set in running text, it takes
// markInText.
export function SiteStatusBadge({ status, count, className }: {
  status: SymptomSite['status']
  count?: number
  className?: string
}) {
  const label = SITE_STATUS_LABELS[status]
  return (
    <Badge variant={status === 'bad' ? 'red-subtle' : 'zinc-subtle'} className={cn('tabular-nums', className)}>
      {count === undefined ? label : `${count} ${label}`}
    </Badge>
  )
}
