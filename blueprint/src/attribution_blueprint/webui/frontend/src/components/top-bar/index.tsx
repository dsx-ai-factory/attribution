import logoUrl from '@/assets/logo.svg'
import type { RouterName } from '@/services/navigate.service'

export function TopBar({ name, database }: { name: RouterName; database?: string | null }) {
  const view = name
  return (
    <div className="flex items-center gap-4 pt-3 pb-3.5 mb-5 border-b border-border">
      <div className="flex items-center gap-2.5 font-semibold">
        <img src={logoUrl} alt="Attribution" className="w-7 h-7 shrink-0" />
        <div>
          <div className="font-mono text-xs uppercase tracking-wider">NVIDIA Attribution</div>
          <div className="font-mono text-xs text-foreground-muted">NVIDIA Attribution · v0.3.0 · {view}</div>
        </div>
      </div>
      <div className="flex-1" />
      <div className="flex items-center gap-3.5 font-mono text-xs text-foreground-muted">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary shadow-[0_0_8px_var(--color-primary-glow)]" />
        <span>service · v2 healthy</span>
        <span className="text-zinc-700">·</span>
        <span>pipeline · 0.1.0</span>
        <span className="text-zinc-700">·</span>
        {database !== null && database !== undefined && database !== '' && (
          <>
            <span>db · {database}</span>
            <span className="text-zinc-700">·</span>
          </>
        )}
        <span>ingesting</span>
      </div>
    </div>
  )
}
