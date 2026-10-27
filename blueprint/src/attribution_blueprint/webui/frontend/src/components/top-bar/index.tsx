// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Fragment, type ReactNode } from 'react'
import { Database } from 'lucide-react'
import logoUrl from '@/assets/logo.svg'
import type { Meta } from '@/interfaces/meta'
import { RouterLink } from '@/components/ui/router-link'
import { navigateService } from '@/services/navigate.service'
import { cn, markInRow } from '@/lib/utils'

// The app bar: the brand, once, as the way home, and the backend the page reads
// from as /api/v1/meta reports it. Only what the server states is shown: a
// version it leaves null is left out, and there is no health or ingest state
// because the server reports none. The backend facts take the house InfoItem
// shape (a muted icon before the text), stepped down to 12px muted so they sit
// behind the page content. They keep one line, as the bar's height is fixed:
// on a narrow screen they are cut short at the end, whole on hover (title) and
// to assistive tech.
export function TopBar({ meta }: { meta: Meta | null }) {
  const facts: { key: string; text: string; content: ReactNode }[] = []
  if (meta?.database) {
    facts.push({ key: 'database', text: meta.database, content: <><span className="sr-only">Database </span>{meta.database}</> })
  }
  if (meta?.versions.schema) {
    facts.push({ key: 'schema', text: `schema ${meta.versions.schema}`, content: <>schema <span className="num">{meta.versions.schema}</span></> })
  }
  if (meta?.versions.pipeline) {
    facts.push({ key: 'pipeline', text: `pipeline ${meta.versions.pipeline}`, content: <>pipeline <span className="num">{meta.versions.pipeline}</span></> })
  }
  if (meta?.versions.service) {
    facts.push({ key: 'service', text: `service ${meta.versions.service}`, content: <>service {meta.versions.service}</> })
  }

  return (
    <header data-stuck-chrome className="sticky top-0 z-30 flex h-(--app-bar-height) items-center justify-between gap-6 border-b border-border bg-background px-page-x">
      <RouterLink
        to={navigateService.getPath('fleet')}
        variant="foreground"
        weight="medium"
        underline="none"
        className="flex shrink-0 items-center gap-2"
      >
        <img src={logoUrl} alt="" className="size-6 shrink-0" />
        NVIDIA Attribution
      </RouterLink>
      {facts.length > 0 && (
        <div title={facts.map((fact) => fact.text).join(' · ')} className="flex min-w-0 items-center gap-2 text-xs text-foreground-muted">
          <Database aria-hidden="true" className={cn('size-3.5 shrink-0', markInRow)} />
          <p className="min-w-0 truncate">
            {facts.map((fact, index) => (
              <Fragment key={fact.key}>
                {index > 0 && <span aria-hidden="true"> · </span>}
                <span>{fact.content}</span>
              </Fragment>
            ))}
          </p>
        </div>
      )}
    </header>
  )
}
