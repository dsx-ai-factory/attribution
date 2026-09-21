import logoUrl from '@/assets/logo.svg'
import type { RouterName } from '@/services/navigate.service'

export function TopBar({ name, database }: { name: RouterName; database?: string | null }) {
  const view = name
  return (
    <div className="flex flex-wrap items-center gap-4 pt-3 pb-3.5 mb-5 border-b border-border">
      <div className="flex min-w-0 items-center gap-2.5 font-semibold">
        <img src={logoUrl} alt="Attribution" className="w-7 h-7 shrink-0" />
        <div className="min-w-0 break-words">
          <div className="font-mono text-xs uppercase tracking-wider">NVIDIA Attribution</div>
          <div className="font-mono text-xs text-foreground-muted">NVIDIA Attribution · v0.3.0 · {view}</div>
        </div>
      </div>
      <div className="ml-auto flex min-w-0 flex-wrap items-center gap-x-3.5 gap-y-2 break-words font-mono text-xs text-foreground-muted">
        <span className="inline-flex min-w-0 items-center gap-3.5">
          <span aria-hidden="true" className="inline-block w-1.5 h-1.5 shrink-0 rounded-full bg-primary shadow-[0_0_8px_var(--color-primary-glow)]" />
          <span className="min-w-0">service · v2 healthy</span>
          <span aria-hidden="true" className="shrink-0 text-zinc-700">·</span>
        </span>
        <span className="inline-flex min-w-0 items-center gap-3.5">
          <span className="min-w-0">pipeline · 0.1.0</span>
          <span aria-hidden="true" className="shrink-0 text-zinc-700">·</span>
        </span>
        {database !== null && database !== undefined && database !== '' && (
          <span className="inline-flex min-w-0 items-center gap-3.5">
            <span className="min-w-0">db · {database}</span>
            <span aria-hidden="true" className="shrink-0 text-zinc-700">·</span>
          </span>
        )}
        <span>ingesting</span>
      </div>
    </div>
  )
}
