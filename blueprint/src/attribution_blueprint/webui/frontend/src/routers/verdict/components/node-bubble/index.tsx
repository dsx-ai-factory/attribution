import * as React from 'react'
import { cn } from '@/lib/utils'

type Props = Omit<React.ComponentProps<'a'>, 'children'> & {
  node: string
  responsible?: boolean
}

export function NodeBubble({ node, responsible, className, ...props }: Props) {
  const tone = responsible
    ? 'text-foreground border-foreground-secondary bg-white/5'
    : 'text-foreground-secondary border-border bg-background-secondary hover:text-primary hover:border-primary-700 hover:bg-primary/15'
  return (
    <a
      {...props}
      className={cn(
        'inline-flex items-center gap-1.5 font-mono text-xs px-2.5 py-1 border rounded-sm cursor-pointer',
        tone,
        className,
      )}
    >
      {node}
    </a>
  )
}
