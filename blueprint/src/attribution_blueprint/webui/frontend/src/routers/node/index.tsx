import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { CircleAlert } from 'lucide-react'
import Decimal from 'decimal.js'
import { OUTCOMES } from '@/lib/attribution-data'
import { nodeGpuHours } from '@/lib/metric'
import { useNodeRuns } from '@/hooks/use-node-runs'
import { buildNodeMarkdown } from '@/lib/evidence'
import { downloadBlob, nodeExportData, slug, toYaml, utcDateStamp } from '@/lib/export'
import { ExportMenu } from '@/components/export-menu'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { VerdictPill } from '@/components/chip'
import { DefHover } from '@/components/def-hover'
import { JobHoverCard } from '@/components/job-hover-card'
import { TimeHover } from '@/components/time-hover'
import { useNav } from '@/hooks/use-nav'

const SECTION_LABEL =
  'font-mono text-xs font-semibold tracking-widest uppercase text-foreground-muted my-5 mb-2.5 flex items-center gap-2.5'

function CardHead({ title, sub }: { title: string; sub?: React.ReactNode }) {
  return (
    <CardHeader className="flex flex-row items-baseline gap-3">
      <CardTitle className="text-xs font-semibold font-mono tracking-wide uppercase">
        {title}
      </CardTitle>
      {sub && <span className="text-foreground-muted text-xs">{sub}</span>}
    </CardHeader>
  )
}

async function writeClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    try { document.execCommand('copy') } catch { /* swallow */ }
    document.body.removeChild(ta)
  }
}

export function NodeView() {
  const { id: nodeId = '' } = useParams<{ id: string }>()
  const nav = useNav()
  const { runs, isLoading } = useNodeRuns(nodeId)
  const [copied, setCopied] = useState('')
  const flash = (label: string) => {
    setCopied(label)
    window.setTimeout(() => setCopied(''), 1200)
  }

  const attempts = runs.filter((r) => r.nodeIds.includes(nodeId) || r.responsibleNodes.includes(nodeId))
  const symptomMap: Record<string, number> = {}
  attempts.forEach((r) => {
    if (r.responsibleNodes.includes(nodeId)) {
      const s = r.rootSymptom.symptom
      symptomMap[s] = (symptomMap[s] || 0) + 1
    }
  })
  const symptoms = Object.entries(symptomMap).map(([name, n]) => ({ name, n }))
  const distinct = symptoms.length
  const impacted = attempts.filter((r) => r.responsibleNodes.includes(nodeId)).length
  const observedGpuHours = attempts.reduce(
    (acc, r) => acc.plus(nodeGpuHours(r)),
    new Decimal(0),
  )

  const copyNodeId = async () => {
    await writeClipboard(nodeId)
    flash('node')
  }
  const exportEvidence = async () => {
    await writeClipboard(buildNodeMarkdown(nodeId, attempts))
    flash('evidence')
  }

  return (
    <div>
      <Breadcrumb className="mb-3.5 font-mono">
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink
              onClick={() => nav('fleet')}
              className="text-primary cursor-pointer"
            >
              Attribution
            </BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>node</BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>
            <BreadcrumbPage className="inline-block text-foreground font-medium border-b-2 border-foreground-secondary pb-0.5">{nodeId}</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <div className="flex items-start gap-4 my-1.5 mb-2">
        <h1 className="text-3xl font-bold m-0 tracking-tight">
          <span className="font-mono text-foreground">{nodeId}</span>
        </h1>
      </div>
      <div className="flex flex-wrap gap-3.5 font-mono text-xs text-foreground-muted mb-1">
        <span>
          <DefHover term="attempts-impacted">Attempts impacted</DefHover>{' '}
          <b className="text-foreground-secondary font-medium">{impacted}</b>
        </span>
        <span>
          <DefHover term="distinct-symptoms">Distinct symptoms</DefHover>{' '}
          <b className="text-foreground-secondary font-medium">{distinct}</b>
        </span>
        <span>
          <DefHover term="gpu-hours-observed">GPU-hours observed</DefHover>{' '}
          <b className="text-foreground-secondary font-medium">
            {observedGpuHours.toFixed(2)}h
          </b>
        </span>
      </div>

      {impacted >= 1 && (
        <Alert className="mt-3.5 mb-3.5 flex flex-wrap items-center gap-x-4 gap-y-2">
          <CircleAlert className="text-primary" />
          <div className="flex flex-col gap-1 min-w-0 flex-1">
            <AlertTitle className="font-mono text-xs tracking-widest uppercase">
              Responsible node
            </AlertTitle>
            <AlertDescription className="font-mono text-xs">
              Recurring{' '}
              <b className="text-foreground font-medium">{symptoms[0]?.name || 'failure'}</b>{' '}
              attributed to this node within the window. Consider escalating to cloud operator with
              evidence.
            </AlertDescription>
          </div>
          <Button variant="outline" size="sm" onClick={copyNodeId} className="min-w-[8rem] justify-center">
            {copied === 'node' ? 'Copied ID ✓' : 'Copy node ID'}
          </Button>
          <Button variant="outline" size="sm" onClick={exportEvidence} className="min-w-[12rem] justify-center">
            {copied === 'evidence' ? 'Copied evidence ✓' : 'Copy evidence (markdown)'}
          </Button>
          <ExportMenu
            triggerLabel="Download"
            items={[
              {
                label: 'Node evidence (JSON)',
                hint: 'machine-parseable',
                onSelect: () => {
                  const stamp = `node-${slug(nodeId)}-${utcDateStamp()}`
                  downloadBlob(
                    `${stamp}.json`,
                    JSON.stringify(nodeExportData(nodeId, attempts), null, 2),
                    'application/json',
                  )
                },
              },
              {
                label: 'Node evidence (YAML)',
                hint: 'drop into a ticket',
                onSelect: () => {
                  const stamp = `node-${slug(nodeId)}-${utcDateStamp()}`
                  downloadBlob(
                    `${stamp}.yaml`,
                    toYaml(nodeExportData(nodeId, attempts)),
                    'application/yaml',
                  )
                },
              },
              {
                label: 'Node evidence (markdown)',
                hint: 'same as Copy evidence',
                onSelect: () => {
                  const stamp = `node-${slug(nodeId)}-${utcDateStamp()}`
                  downloadBlob(
                    `${stamp}.md`,
                    buildNodeMarkdown(nodeId, attempts),
                    'text/markdown',
                  )
                },
              },
            ]}
          />
        </Alert>
      )}

      <div className="mb-5">
        <Card size="sm">
          <CardHead title="Symptoms on this node" />
          <CardContent className="px-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symptom</TableHead>
                  <TableHead className="text-right w-24">Attempts</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading && attempts.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={2} className="text-foreground-muted">
                      Loading attributed symptoms...
                    </TableCell>
                  </TableRow>
                ) : symptoms.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={2} className="text-foreground-muted">
                      No attributed symptoms in window.
                    </TableCell>
                  </TableRow>
                ) : (
                  symptoms.map((s, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-mono text-xs text-foreground">
                        <b className="font-semibold">{s.name}</b>
                      </TableCell>
                      <TableCell className="font-mono text-foreground text-right">{s.n}</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      <div className={SECTION_LABEL}>
        Attempts impacted
        <span className="bg-background-muted border border-border rounded-sm px-1.5 py-0.5 text-foreground-secondary font-medium">
          {impacted}
        </span>
        <span className="flex-1 h-px bg-border" />
      </div>
      <ScrollArea className="border border-border rounded-md bg-card">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-40">
                <DefHover term="end-time-utc">End time (UTC)</DefHover>
              </TableHead>
              <TableHead>Execution Context</TableHead>
              <TableHead className="w-16">Att</TableHead>
              <TableHead>User</TableHead>
              <TableHead className="w-28">Outcome</TableHead>
              <TableHead className="w-24 text-right">
                <DefHover term="gpu-hours">GPU-hours</DefHover>
              </TableHead>
              <TableHead>Symptoms on this node</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {attempts.map((r) => (
              <TableRow
                key={r.id}
                onClick={() => nav('verdict', { id: r.id })}
                className="cursor-pointer group"
              >
                <TableCell className="font-mono text-xs text-foreground">
                  <TimeHover date={r.endTime} />
                </TableCell>
                <TableCell className="font-mono text-xs text-foreground group-hover:text-primary">
                  <JobHoverCard run={r} linkable>
                    <span>{r.jobId}</span>
                  </JobHoverCard>
                </TableCell>
                <TableCell className="font-mono text-xs">
                  <span className="text-foreground-secondary">{r.attempt}</span>
                </TableCell>
                <TableCell className="text-foreground-secondary">{r.user}</TableCell>
                <TableCell>
                  <VerdictPill outcome={r.outcome} label={OUTCOMES[r.outcome].label} />
                </TableCell>
                <TableCell className="font-mono text-foreground text-right">
                  {nodeGpuHours(r).toFixed(2)}
                </TableCell>
                <TableCell className="font-mono text-xs">
                  {r.responsibleNodes.includes(nodeId) ? (
                    <Badge variant="outline" className="font-mono">
                      {r.rootSymptom.symptom}{' '}
                      <span className="text-foreground-muted ml-1">1</span>
                    </Badge>
                  ) : (
                    <span aria-label="no symptom for this attempt" className="text-zinc-700 select-none">·</span>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  )
}
