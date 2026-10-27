// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ArrowUpRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useSymptomDocumentation } from '@/hooks/use-symptom-documentation'
import { findSymptomDocumentation } from '@/lib/documentation'
import { markInRow } from '@/lib/utils'

// A 20px button beside a symptom's name, a mark beside it as a copy button is
// beside its value: on the name's first line, where what holds the two keeps
// them, and centered on its lowercase (markInRow).
export function SymptomDocumentationLink({ symptomId }: { symptomId?: string | null }) {
  const { data: catalog } = useSymptomDocumentation(Boolean(symptomId))
  const reference = findSymptomDocumentation(catalog, symptomId)
  if (!reference) return null
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          asChild
          variant="outline"
          size="xs"
          className={markInRow}
        >
          <a href={reference.doc_url} target="_blank" rel="noopener noreferrer" aria-label={`Symptom documentation for ${reference.title} (opens in a new tab)`}>
            Doc
            <ArrowUpRight aria-hidden="true" data-icon="inline-end" />
          </a>
        </Button>
      </TooltipTrigger>
      <TooltipContent side="top" align="end" sideOffset={6}>Symptom documentation</TooltipContent>
    </Tooltip>
  )
}
