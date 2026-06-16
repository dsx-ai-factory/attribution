import { useState, type ReactNode } from 'react'
import { Download, ChevronDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'

export interface ExportItem {
  label: string
  hint?: string
  onSelect: () => void
}

export function ExportMenu({
  items,
  triggerLabel = 'Download',
  align = 'end',
}: {
  items: ExportItem[]
  triggerLabel?: ReactNode
  align?: 'start' | 'end' | 'center'
}) {
  const [open, setOpen] = useState(false)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="gap-1.5">
          <Download className="size-3.5" />
          {triggerLabel}
          <ChevronDown className="size-3 opacity-70" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align={align}
        sideOffset={4}
        className="w-56 p-1 bg-background-secondary border-border ring-0"
      >
        <div className="flex flex-col">
          {items.map((item, i) => (
            <button
              key={i}
              type="button"
              onClick={() => {
                item.onSelect()
                setOpen(false)
              }}
              className="text-left px-2.5 py-1.5 rounded-sm font-mono text-xs text-foreground hover:bg-background-active focus-visible:outline-none focus-visible:bg-background-active focus-visible:ring-1 focus-visible:ring-primary"
            >
              <div>{item.label}</div>
              {item.hint && (
                <div className="text-[10px] text-foreground-muted mt-0.5">{item.hint}</div>
              )}
            </button>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  )
}
