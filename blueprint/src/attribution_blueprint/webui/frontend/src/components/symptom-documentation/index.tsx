// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { ArrowUpRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useSymptomDocumentation } from '@/hooks/use-symptom-documentation'
import { findSymptomDocumentation } from '@/lib/documentation'

export function SymptomDocumentationLink({ symptomId }: { symptomId?: string | null }) {
  const { data: catalog } = useSymptomDocumentation(Boolean(symptomId))
  const reference = findSymptomDocumentation(catalog, symptomId)
  if (!reference) return null
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            asChild
            variant="outline"
            size="xs"
            className="rounded-[4px] border-foreground-muted/25 bg-background-active px-1.5 font-sans text-[11px] font-normal focus-visible:border-primary-300 focus-visible:bg-primary-900 focus-visible:text-primary-300 not-disabled:not-aria-disabled:hover:border-primary-300 not-disabled:not-aria-disabled:hover:bg-primary-900 not-disabled:not-aria-disabled:hover:text-primary-300"
          >
            <a href={reference.doc_url} target="_blank" rel="noopener noreferrer" aria-label={`Symptom documentation for ${reference.title} (opens in a new tab)`}>
              Doc
              <ArrowUpRight aria-hidden="true" className="size-2.5" strokeWidth={1.5} />
            </a>
          </Button>
        </TooltipTrigger>
        <TooltipContent side="top" align="end" sideOffset={6} collisionPadding={8} showArrow={false}>Symptom documentation</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
